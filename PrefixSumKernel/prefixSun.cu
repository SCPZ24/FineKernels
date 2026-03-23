#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

template<int N>
__global__ void prefix_sum_naive(const float* __restrict__ input, float* __restrict__ output){
    const int thread_id = threadIdx.x;
    __shared__ float shared_input[N];

    if(thread_id < N){
        shared_input[thread_id] = input[thread_id];
    }
    __syncthreads();

    // RAW (Read-After-Write) ：涉及两次sync。
    // 第一次sync以前，最好都是做读取。
    // 第二次sync以前，最好都是做写入。
    for(int s = 1; s < N; s <<= 1){
        const bool in_operation = thread_id >= s;
        if(in_operation){
            const float temp = shared_input[thread_id - s] + shared_input[thread_id];
        }
        __syncthreads();
        if(in_operation){
            shared_input[thread_id] = temp;
        }
        __syncthreads();
    }

    if(thread_id < N){
        output[thread_id] = shared_input[thread_id];
    }
}
