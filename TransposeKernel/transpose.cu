#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void transpose_naive(const float* input, float* output, int M, int N){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N){
        output[i * N + j] = input[j * M + i];
    }
}

template<int SM, int SN>
__global__ void transpose_shared_direct_map(const float* input, float* output, int M, int N){
    // 会导致Bank Conflict的直接映射方式
    // 一次访存连续，一次不连续。
    // 转置前 M * N  axis: y -> M, x -> N
    // 转置后 N * M
    // 块内，我们认为blockDim.x = SN, blockDim.y = SM.
    
    __shared__ float cache[SM][SN];
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_offset = global_y * N + global_x;

    int shared_offset_row, shared_offset_col;
    {
        const int block_linear_index = blockDim.x * threadIdx.y + threadIdx.x;
        shared_offset_row = block_linear_index / blockDim.y;
        shared_offset_col = block_linear_index % blockDim.y;
    }

    if(global_x < M && global_y < N){
        cache[threadIdx.y][threadIdx.x] = input[input_offset];
        __syncthreads();
        output[(blockDim.x * blockIdx.x + shared_offset_row) * M + blockDim.y * blockIdx.y + shared_offset_col] = cache[shared_offset_col][shared_offset_row];
    }
}

template<int SM, int SN>
__global__ void transpose_shared_bank_conflict_free(const float* input, float* output, int M, int N){
    // 会导致Bank Conflict的直接映射方式
    // 一次访存连续，一次不连续。
    // 转置前 M * N  axis: y -> M, x -> N
    // 转置后 N * M
    // 块内，我们认为blockDim.x = SN, blockDim.y = SM.
    
    __shared__ float cache[SM][SN+1];
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_offset = global_y * N + global_x;

    int shared_offset_row, shared_offset_col;
    {
        const int block_linear_index = blockDim.x * threadIdx.y + threadIdx.x;
        shared_offset_row = block_linear_index / blockDim.y;
        shared_offset_col = block_linear_index % blockDim.y;
    }

    if(global_x < M && global_y < N){
        cache[threadIdx.y][threadIdx.x] = input[input_offset];
        __syncthreads();
        output[(blockDim.x * blockIdx.x + shared_offset_row) * M + blockDim.y * blockIdx.y + shared_offset_col] = cache[shared_offset_col][shared_offset_row];
    }
}

template<int SM, int SN>
__global__ void transpose_shared_bank_conflict_free_double_fetch(const float* input, float* output, int M, int N){
    // 这个kernel启动时，blockDim.x只要原来的一半。
    // 因为每个thread会读取两个元素，所以需要读取的元素数量减半。

    __shared__ float cache[SM * (SN+1)];
    constexpr int cache_row_size = SN + 1;
    int global_x = (blockIdx.x * blockDim.x << 1) + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_offset = global_y * N + global_x;

    int shared_offset_row, shared_offset_col;
    {
        const int block_linear_index = (blockDim.x * threadIdx.y) + threadIdx.x;
        shared_offset_row = block_linear_index / blockDim.y;
        shared_offset_col = block_linear_index % blockDim.y;
    }

    global_x = blockDim.y * blockIdx.y + shared_offset_col;
    global_y = (blockDim.x * blockIdx.x << 1) + shared_offset_row;

    if(global_x < M && global_y < N){
        {
            const int cache_offset = threadIdx.y * cache_row_size + threadIdx.x;
            cache[cache_offset] = input[input_offset];
            cache[cache_offset + blockDim.x] = input[input_offset + 1];
        }
        __syncthreads();
        output[global_y * M + global_x] = cache[shared_offset_col * cache_row_size + shared_offset_row];
        output[(global_y + blockDim.x) * M + global_x] = cache[shared_offset_col * cache_row_size + shared_offset_row + blockDim.x];
    }
}
