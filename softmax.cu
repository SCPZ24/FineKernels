#include<cuda_runtime.h>
#include "public.h"

__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int N, int C>
__global__ void softmax_forward(const float* input, float* output){
    // 计算线程信息，申明空间。
    const int row = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int wrap_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;
    const int num_wraps = blockDim.x / WARP_SIZE;
    input += row * C;
    extern __shared__ float shared_data[];

    // Wrap级规约，找到每个wrap内的最大值。
    float max_val = -INFINITY;
    for(int colIndex = thread_id; colIndex < C; colIndex += blockDim.x){
        const float val = input[colIndex];
        max_val = fmaxf(max_val, val);
    }
    max_val = warpReduceMax(max_val);

    // 把wrap内的最大值写入共享内存。
    if(lane_id == 0){
        shared_data[wrap_id] = max_val;
    }
    __syncthreads();

    // Block级规约，找到所有wrap内的最大值，并让每个线程都获知。
    if(thread_id == 0){
        for(int i = 0; i < num_wraps; ++i){
            max_val = fmaxf(max_val, shared_data[i]);
        }
        shared_data[0] = max_val;
    }
    __syncthreads();
    max_val = shared_data[0];

    // 所有线程都需要减去最大值，防止指数爆炸。每个线程算出各自负责的和，然后获取wrap级的和。
    float sum = 0.0f;
    for(int colIndex = thread_id; colIndex < C; colIndex += blockDim.x){
        const float val = input[colIndex];
        sum += expf(val - max_val);
    }
    sum = warpReduceSum(sum);

    // 把wrap级和写入共享内存。
    if(lane_id == 0){
        shared_data[wrap_id] = sum;
    }
    __syncthreads();

    // Block级规约，找到所有wrap内的和，并让每个线程都获知。
    if(thread_id == 0){
        for(int i = 0; i < num_wraps; ++i){
            sum += shared_data[i];
        }
        shared_data[0] = sum;
    }
    __syncthreads();
    sum = shared_data[0];

    // 所有线程都需要除以和，得到softmax的输出。
    for(int colIndex = thread_id; colIndex < C; colIndex += blockDim.x){
        const float val = input[colIndex];
        output[row * C + colIndex] = expf(val - max_val) / sum;
    }
}

