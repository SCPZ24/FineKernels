#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

template<int BM, int BN, int BK, int TM, int TN>
__global__ void GEMM_naive(const float* A, const float* B, float* C, int M, int N, int K, const float alpha, const float beta){
}
