#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../public.h"

__device__ __forceinline__ float reduceFloat4(const float4& f4){
    return f4.x + f4.y + f4.z + f4.w;
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
    // N为 batch_size * seq_len
    // C为 hidden_size
    // 所以gridDim.x = N
    // blockDim.x = C/4
    // gamma: (C,)

    const int thread_id = threadIdx.x;
    const float4 local_input_f4 = *(reinterpret_cast<const float4*>(input) + blockIdx.x * blockDim.x + thread_id);
    float4* output_f4 = reinterpret_cast<float4*>(output) + blockIdx.x * blockDim.x + thread_id;
    const float4 local_gamma_f4 = *(reinterpret_cast<const float4*>(gamma) + thread_id);

    constexpr int num_warp = C >> 7;
    __shared__ float shared_wrap_data[num_warp];    // C / 4 ：总线程数。 C / 4 / 32: 总 warp 数

    float local_sum = square_reduceFloat4(local_input_f4);
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
    *output_f4 = make_float4(
        local_sum * local_input_f4.x * local_gamma_f4.x,
        local_sum * local_input_f4.y * local_gamma_f4.y,
        local_sum * local_input_f4.z * local_gamma_f4.z,
        local_sum * local_input_f4.w * local_gamma_f4.w
    );
}
