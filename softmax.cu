#include<cuda_runtime.h>
#include "public.h"

template<int N, int C>
__global__ void softmax_forward(const float* input, float* output){
    const int row = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int wrap_id = blockDim.x / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;
    input += row * C;

    for(int colIndex = thread_id; colIndex < C; colIndex += blockDim.x){
        
    }
}

int main(){

}