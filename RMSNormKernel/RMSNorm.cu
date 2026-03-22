#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include "../public.h"

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b){
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float square_reduceFloat4(const float4& f4){
    return f4.x * f4.x + f4.y * f4.y + f4.z * f4.z + f4.w * f4.w;
}

__device__ __forceinline__ float warpReduceSum(float val){
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

template<int N, int C>
__global__ void RMSNorm_block_collapse(const float* input, const float* gamma, float* output, const float eps){
    // block量不变，每个线程处理4个float
    // N为 batch_size * seq_len
    // C为 hidden_size
    // 所以gridDim.x = N
    // blockDim.x = C/4 或更小
    // gamma: (C,)

    const int thread_id = threadIdx.x;
    const float4* input_f4 = reinterpret_cast<const float4*>(input) + blockIdx.x * blockDim.x;
    float4* output_f4 = reinterpret_cast<float4*>(output) + blockIdx.x * blockDim.x;
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    const int num_float4 = C >> 2;

    constexpr int num_warp = C >> 7;
    __shared__ float shared_wrap_data[num_warp];    // C / 4 ：总线程数。 C / 4 / 32: 总 warp 数

    float local_sum = 0.f;
    for(int i = thread_id; i < num_float4; i += blockDim.x){
        local_sum += square_reduceFloat4(input_f4[i]);
    }

    local_sum = warpReduceSum(local_sum);
    if(thread_id % WARP_SIZE == 0){
        shared_wrap_data[thread_id / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    if(thread_id < WARP_SIZE){
        local_sum = 0;
        for(int i = thread_id; i < num_warp; i += WARP_SIZE){
            local_sum += shared_wrap_data[i];
        }
        local_sum = warpReduceSum(local_sum);
    }

    if(thread_id == 0){
        shared_wrap_data[0] = rsqrt(local_sum / C + eps);
    }
    __syncthreads();

    local_sum = shared_wrap_data[0];
    
    for(int i = thread_id; i < num_float4; i += blockDim.x){
        const float4 local_input_f4 = input_f4[i], local_gamma_f4 = gamma_f4[i];
        output_f4[i] = make_float4(
            local_sum * local_input_f4.x * local_gamma_f4.x,
            local_sum * local_input_f4.y * local_gamma_f4.y,
            local_sum * local_input_f4.z * local_gamma_f4.z,
            local_sum * local_input_f4.w * local_gamma_f4.w
        );
    }
}

template<int N, int C>
__global__ void RMSNorm_grid_collapse(const float* input, const float* gamma, float* output, const float eps){
    // 一个block内的线程量不变，但是block数量只要原来的1/4。
    // N为 batch_size * seq_len
    // C为 hidden_size
    // 所以gridDim.x = N / 4
    // blockDim.x = C
    // gamma: (C,)

    const int thread_id = threadIdx.x;
    const int threads_per_row = blockDim.x >> 2;
    const int row_id = thread_id / threads_per_row;
    const int row_thread_id = thread_id % threads_per_row;
    constexpr int wraps_per_row = C >> 7;
    __shared__ float shared_wrap_data[4][wraps_per_row]; 
    
    const float4* input_f4;
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    float4* output_f4;
    {
        const int global_offset = (blockIdx.x * 4 + row_id) * C;
        input_f4 = reinterpret_cast<const float4*>(input + global_offset);
        output_f4 = reinterpret_cast<float4*>(output + global_offset);
    }

    float local_sum = 0.f;
    const int num_float4 = C >> 2;
    for (int i = row_thread_id; i < num_float4; i += threads_per_row) {
        local_sum += square_reduceFloat4(input_f4[i]);
    }

    local_sum = warpReduceSum(local_sum);
    
    const int row_warp_id = row_thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;
    
    if(lane_id == 0){
        shared_wrap_data[row_id][row_warp_id] = local_sum;
    }
    __syncthreads();
    
    if (row_warp_id == 0) {
        local_sum = 0;
        for(int i = lane_id; i < wraps_per_row; i += WARP_SIZE){
            local_sum += shared_wrap_data[row_id][i];
        }
        local_sum = warpReduceSum(local_sum);
        if (lane_id == 0){
            shared_wrap_data[row_id][0] = rsqrt(local_sum / C + eps);
        }
    }
    __syncthreads();

    local_sum = shared_wrap_data[row_id][0];
    for (int i = row_thread_id; i < num_float4; i += threads_per_row) {
        const float4 local_input_f4 = input_f4[i], local_gamma_f4 = gamma_f4[i];
        output_f4[i] = make_float4(
            local_sum * local_input_f4.x * local_gamma_f4.x,
            local_sum * local_input_f4.y * local_gamma_f4.y,
            local_sum * local_input_f4.z * local_gamma_f4.z,
            local_sum * local_input_f4.w * local_gamma_f4.w
        );
    }
}

// Gemini构造的测试用例
void cpu_rmsnorm(float* out, const float* in, const float* gamma, int N, int C, float eps) {
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum += in[i * C + j] * in[i * C + j];
        }
        float inv_rms = 1.0f / sqrtf(sum / C + eps);
        for (int j = 0; j < C; ++j) {
            out[i * C + j] = in[i * C + j] * inv_rms * gamma[j];
        }
    }
}

bool check_result(const float* gpu, const float* cpu, int len) {
    float max_err = 0.0f;
    for (int i = 0; i < len; ++i) {
        max_err = fmax(max_err, fabsf(gpu[i] - cpu[i]));
    }
    std::cout << "  - Max Absolute Error: " << max_err << (max_err < 1e-5 ? " (PASS)" : " (FAIL)") << std::endl;
    return max_err < 1e-5;
}

template<int N, int C>
void run_test() {
    float eps = 1e-5f;
    size_t size = N * C * sizeof(float);
    size_t g_size = C * sizeof(float);

    // Host 分配
    std::vector<float> h_in(N * C), h_gamma(C), h_out_gpu(N * C), h_out_cpu(N * C);
    for (int i = 0; i < N * C; ++i) h_in[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < C; ++i) h_gamma[i] = (float)rand() / RAND_MAX;

    // Device 分配
    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_gamma, g_size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), g_size, cudaMemcpyHostToDevice);

    std::cout << "\n>>> Testing N=" << N << ", C=" << C << " (" << (size / 1024.0 / 1024.0) << " MB)" << std::endl;

    // CPU 计算作为基准
    cpu_rmsnorm(h_out_cpu.data(), h_in.data(), h_gamma.data(), N, C, eps);

    // 定义计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // --- 测试 Kernel 1: Block Collapse ---
    cudaEventRecord(start);
    // 配置：1 个 Block 处理 1 行，每个 Block 256 线程执行跨步循环
    RMSNorm_block_collapse<N, C><<<N, C/4>>>(d_in, d_gamma, d_out, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[Block Collapse] Time: " << std::fixed << std::setprecision(4) << ms << " ms";
    check_result(h_out_gpu.data(), h_out_cpu.data(), N * C);

    // --- 测试 Kernel 2: Grid Collapse ---
    cudaMemset(d_out, 0, size);
    cudaEventRecord(start);
    // 配置：1 个 Block 处理 4 行，线程数固定为 1024
    RMSNorm_grid_collapse<N, C><<<N/4, C>>>(d_in, d_gamma, d_out, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_out_gpu.data(), d_out, size, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[Grid Collapse ] Time: " << ms << " ms";
    check_result(h_out_gpu.data(), h_out_cpu.data(), N * C);

    // 清理
    cudaFree(d_in); cudaFree(d_gamma); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    // 设置随机种子
    srand(2026);
    
    // 测试 Case 1: 小规模（受限于启动开销）
    run_test<8, 4096>();

    // 测试 Case 2: 大规模（测试带宽极限）
    run_test<1024, 4096>();

    return 0;
}