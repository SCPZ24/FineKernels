#include<cuda_runtime.h>
#include "public.h"

__device__ __forceinline__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int C>
__global__ void softmax_forward_naive(const float* input, float* output){
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

// 优化：
//  1. 一个block负责计算4个rows。
//  2. 引入float4向量化读取。
//  3. 优化全局访存读取。
template<int C, int rows_per_block = 4>
__global__ void softmax_forward_optimize(const float* input, float* output){
    // 计算线程信息，申明空间。
    // 一个block有4个wrap，每个wrap负责计算一个row。
    const int thread_id = threadIdx.x;
    const int wrap_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;
    const float4 *input_f4 = reinterpret_cast<const float4*>(input + (blockIdx.x * rows_per_block + wrap_id) * C);
    float4 *output_f4 = reinterpret_cast<float4*>(output + (blockIdx.x * rows_per_block + wrap_id) * C);
    constexpr int float4_per_thread = C / WARP_SIZE >> 2;

    // -------|th 1|th 2|th 3|th 4|......--------------------------
    // wrap 0:|----|----|----|----|......|----|
    // wrap 1:|----|----|----|----|......|----|
    // wrap 2:|----|----|----|----|......|----|
    // wrap 3:|----|----|----|----|......|----|
    // ------------------------------------------------------------

    float4 reg_val_f4[float4_per_thread];
    float max_val = -INFINITY;
    for(int i = 0; i < float4_per_thread; ++i){
        const float4 val = input_f4[i * WARP_SIZE + lane_id];
        reg_val_f4[i] = val;
        max_val = fmaxf(max_val, val.x);
        max_val = fmaxf(max_val, val.y);
        max_val = fmaxf(max_val, val.z);
        max_val = fmaxf(max_val, val.w);
    }
    max_val = warpReduceMax(max_val);

    float sum = 0.0f;
    for(int i = 0; i < float4_per_thread; ++i){
        reg_val_f4[i].x = expf(reg_val_f4[i].x - max_val);
        reg_val_f4[i].y = expf(reg_val_f4[i].y - max_val);
        reg_val_f4[i].z = expf(reg_val_f4[i].z - max_val);
        reg_val_f4[i].w = expf(reg_val_f4[i].w - max_val);
        sum += reg_val_f4[i].x + reg_val_f4[i].y + reg_val_f4[i].z + reg_val_f4[i].w;
    }
    sum = warpReduceSum(sum);

    sum = 1.f / sum;
    for(int i = 0; i < float4_per_thread; ++i){
        reg_val_f4[i].x *= sum;
        reg_val_f4[i].y *= sum;
        reg_val_f4[i].z *= sum;
        reg_val_f4[i].w *= sum;
        output_f4[i * WARP_SIZE + lane_id] = reg_val_f4[i];
    }
}
