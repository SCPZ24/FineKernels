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

    if(thread_id < N){
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
    }

    if(thread_id < N){
        output[thread_id] = shared_input[thread_id];
    }
}

__device__ void ptr_swap(float*& a, float*& b){
    float* temp = a;
    a = b;
    b = temp;
}

template<int N>
__global__ void prefix_sum_double_buffer(const float* __restrict__ input, float* __restrict__ output){
    const int thread_id = threadIdx.x;
    __shared__ float shared_input_T1[N], shared_input_T2[N];
    float* ptr_src = shared_input_T1;
    float* ptr_dst = shared_input_T2;

    if(thread_id < N){
        const float temp = input[thread_id];
        shared_input_T1[thread_id] = temp;
    }
    __syncthreads();

    if(thread_id < N){
        for(int s = 1; s < N; s <<= 1){
            if(thread_id >= s){
                ptr_dst[thread_id] = ptr_src[thread_id - s] + ptr_src[thread_id];
            }else{
                ptr_dst[thread_id] = ptr_src[thread_id];
            }
            ptr_swap(ptr_src, ptr_dst);
            __syncthreads();
        }
    }

    if(thread_id < N){
        output[thread_id] = ptr_dst[thread_id];
    }
}

template<int N>
__global__ void prefix_sum_tree(const float* __restrict__ input, float* __restrict__ output){
    const int thread_id = threadIdx.x;
    __shared__ float shared_input[N];
   
    if(thread_id < N){
        shared_input[thread_id] = input[thread_id];
    }
    __syncthreads();

    // Sweep Up
    for(int s = 1; s < N; s <<= 1){
        __syncthreads();
        const int root = ((thread_id + 1) * s << 1) - 1;
        if(root < N){
            shared_input[root] += shared_input[root - s];
        }
    }

    // Sweep Down
    for(int s = N >> 1; s; s >>= 1){
        __syncthreads();
        const int child = (s * 3 - 1) + (thread_id * s << 1);
        if(child < N){
            shared_input[child] += shared_input[child - s];
        }
    }

    if(thread_id < N){
        output[thread_id] = shared_input[thread_id];
    }
}

#define offset(N) (N + (N >> 5))

template<int N>
__global__ void prefix_sum_tree_bank_conflict_free(const float* __restrict__ input, float* __restrict__ output){
    const int thread_id = threadIdx.x;
    __shared__ float shared_input[offset(N)];
   
    if(thread_id < N){
        shared_input[offset(thread_id)] = input[thread_id];
    }
    __syncthreads();

    // Sweep Up
    for(int s = 1; s < N; s <<= 1){
        __syncthreads();
        const int root = ((thread_id + 1) * s << 1) - 1;
        if(root < N){
            shared_input[offset(root)] += shared_input[offset(root - s)];
        }
    }

    // Sweep Down
    for(int s = N >> 1; s; s >>= 1){
        __syncthreads();
        const int child = (s * 3 - 1) + (thread_id * s << 1);
        if(child < N){
            shared_input[offset(child)] += shared_input[offset(child - s)];
        }
    }

    if(thread_id < N){
        output[thread_id] = shared_input[offset(thread_id)];
    }
}