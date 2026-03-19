#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
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
void run_test(int N) {
    size_t size = N * sizeof(float);
    float* h_in = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_inter, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Naive 测试: 启动两次相同的 Kernel
    int threads_1 = 1024;
    int blocks_1 = N / (threads_1 * 2);
    cudaMalloc(&d_inter, blocks_1 * sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    cudaEventRecord(start);
    // 第一次：规约全量数据到 d_inter
    reduce_naive<32><<<blocks_1, threads_1>>>(d_in, d_inter);
    // 第二次：规约 d_inter 到最终 d_out
    // 因为 blocks_1 = 512，所以只需要 256 个线程启动一次即可
    reduce_naive<8><<<1, blocks_1 / 2>>>(d_inter, d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    float res_naive;
    cudaMemcpy(&res_naive, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 2. Optimized 测试: 启动一次
    cudaMemset(d_out, 0, sizeof(float));
    int threads_opt = 1024;
    int blocks_opt = N / (threads_opt * 8);

    cudaEventRecord(start);
    reduce_optimize<32><<<blocks_opt, threads_opt>>>(d_in, d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms_opt;
    cudaEventElapsedTime(&ms_opt, start, stop);
    float res_opt;
    cudaMemcpy(&res_opt, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("N = %d\n", N);
    printf("  Naive (2-Launch): %8.4f ms | BW: %7.2f GB/s | Res: %.1f\n", 
           ms_naive, (size / (ms_naive / 1000.0)) / 1e9, res_naive);
    printf("  Optimized (1-L):  %8.4f ms | BW: %7.2f GB/s | Res: %.1f\n", 
           ms_opt, (size / (ms_opt / 1000.0)) / 1e9, res_opt);
    printf("----------------------------------------------------------\n");

    free(h_in); cudaFree(d_in); cudaFree(d_inter); cudaFree(d_out);
}

int main() {
    run_test(1024 * 1024);
    run_test(512 * 512);
    return 0;
}