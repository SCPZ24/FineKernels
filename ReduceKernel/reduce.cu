#include <cuda_runtime.h>
#include <stdio.h>
#include "../public.h"

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b){
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float reduceFloat4(const float4& f4){
    return f4.x + f4.y + f4.z + f4.w;
}

template<int num_wraps>
__global__ void reduce_naive(const float* __restrict__ input, float* __restrict__ output){
    // 外部启动：C = blockDim.x
    // 同时，C <= 1024
    // 获取线程信息
    const int thread_id = threadIdx.x;
    const int wrap_id = thread_id / WARP_SIZE;
    __shared__ float shared_data[num_wraps];

    // 读入该线程处理的值。
    float local_val;
    {
        const int begin_idx = blockIdx.x * blockDim.x * 2;
        local_val = input[begin_idx + thread_id] + input[begin_idx + blockDim.x + thread_id];
    }

    // Wrap内线程规约进lane_id为0的线程
    local_val += __shfl_down_sync(FULL_MASK, local_val, 16);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 8);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 4);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 2);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 1);

    if(thread_id % WARP_SIZE == 0){
        shared_data[wrap_id] = local_val;
    }
    __syncthreads();

    if(wrap_id == 0){
        local_val = (thread_id < num_wraps) ? shared_data[thread_id] : 0.0f;
        local_val = warpReduceSum(local_val);
    }

    if(thread_id == 0){
        output[blockIdx.x] = local_val;
    }
}

template<int num_wraps = 32>
__global__ void reduce_optimize(const float* __restrict__ input, float* __restrict__ output){
    // 现在，output不是数组而是单个float的指针。
    // 默认，一个block还是有1024thread，每个thread处理2个float4。
    const int thread_id = threadIdx.x;
    const int wrap_id = thread_id / WARP_SIZE;
    const float4 *input_f4 = reinterpret_cast<const float4*>(input);
    __shared__ float shared_data[num_wraps];

    float local_val = 0.0f;
    {
        const int begin_idx = blockIdx.x * blockDim.x * 2;
        const float4 local_val_f4 = input_f4[begin_idx + thread_id] + input_f4[begin_idx + blockDim.x + thread_id];
        local_val = reduceFloat4(local_val_f4);
    }

    local_val += __shfl_down_sync(FULL_MASK, local_val, 16);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 8);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 4);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 2);
    local_val += __shfl_down_sync(FULL_MASK, local_val, 1);

    if(thread_id % WARP_SIZE == 0){
        shared_data[wrap_id] = local_val;
    }
    __syncthreads();

    if(wrap_id == 0){
        local_val = (thread_id < num_wraps) ? shared_data[thread_id] : 0.0f;
        local_val = warpReduceSum(local_val);
    }

    if(thread_id == 0){
        atomicAdd(output, local_val);
    }
}

// 由Gemini生成的测试函数
void run_benchmark(int N, bool use_optimize) {
    size_t size = N * sizeof(float);
    std::vector<float> h_input(N, 1.0f); // 初始化为 1.0，方便验证结果
    float host_ref = (float)N;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    // 计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 1024;
    int blocks;

    if (use_optimize) {
        // Optimized 每个 Block 处理 1024 * 8 = 8192 个 float
        blocks = N / (threads * 8);
        cudaEventRecord(start);
        reduce_optimize<32><<<blocks, threads>>>(d_input, d_output);
        cudaEventRecord(stop);
    } else {
        // Naive 每个 Block 处理 1024 * 2 = 2048 个 float
        blocks = N / (threads * 2);
        cudaEventRecord(start);
        reduce_naive<32><<<blocks, threads>>>(d_input, d_output);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float h_output = 0;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 计算有效带宽 (GB/s)
    double bandwidth = (size / (ms / 1000.0)) / 1e9;

    std::cout << std::left << std::setw(12) << (use_optimize ? "Optimized" : "Naive")
              << " | N: " << std::setw(8) << N 
              << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | Bandwidth: " << std::setw(8) << bandwidth << " GB/s"
              << " | Correct: " << (std::abs(h_output - host_ref) < 1e-1 ? "PASS" : "FAIL") 
              << " (Res: " << h_output << ")" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Starting Reduction Benchmark..." << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    // 测试 1024 * 1024
    run_benchmark(1024 * 1024, false);
    run_benchmark(1024 * 1024, true);

    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    // 测试 512 * 512
    run_benchmark(512 * 512, false);
    run_benchmark(512 * 512, true);

    return 0;
}