#include <cuda_runtime.h>
#include <stdio.h>
#include "../public.h"

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
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

// 由Gemini生成的测试函数
void run_test(int N, int threads_per_block) {
    size_t size = N * sizeof(float);
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f; // 测试用例：全 1，结果应等于 N

    float *d_input, *d_inter, *d_output;
    int blocks = N / (threads_per_block * 2);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_inter, blocks * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // 第一轮归约
    if (threads_per_block == 1024) 
        reduce_naive<32><<<blocks, 1024>>>(d_input, d_inter);
    else if (threads_per_block == 512)
        reduce_naive<16><<<blocks, 512>>>(d_input, d_inter);

    // 第二轮归约：将中间结果 blocks 个元素规约为 1 个
    // 注意：这里为了简化，假设 blocks <= 1024，且满足你的 kernel 逻辑（输入量是线程数2倍）
    // 如果 blocks 很大，这里需要根据 blocks 数量调整启动配置
    if (blocks > 0) {
        int final_threads = 512; // 选一个固定值
        // 调整输入以符合你的“一读二”逻辑：input 长度应为 final_threads * 2
        // 在实际工业代码中，这里通常会写一个更通用的边界检查 kernel
        reduce_naive<16><<<1, final_threads>>>(d_inter, d_output);
    }

    float h_output = 0;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Config: N=" << N << ", BlockSize=" << threads_per_block 
              << " | Result: " << h_output << " (Expected: " << (float)N << ")" << std::endl;

    cudaFree(d_input); cudaFree(d_inter); cudaFree(d_output);
}

void test_naive() {
    std::cout << "--- Starting Reduction Tests ---" << std::endl;
    
    // 测试 1024*1024 (1,048,576 元素)
    // 你的配置：每个线程读2个，BlockSize=1024，则一个Block处理2048个
    // 需要 1024*1024 / 2048 = 512 Blocks
    run_test(1024 * 1024, 1024);

    // 测试 2048*512 (1,048,576 元素)
    // 你的配置：每个线程读2个，BlockSize=512，则一个Block处理1024个
    // 需要 1024*1024 / 1024 = 1024 Blocks
    run_test(2048 * 512, 512);
}