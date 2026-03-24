#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include "../public.h"

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

__device__ __forceinline__ float warp_scan(float local_value){
    float tmp;
    tmp = __shfl_up_sync(FULL_MASK, local_value, 1);
    if((thread_id & 31) >= 1){
        local_value += tmp;
    }
    tmp = __shfl_up_sync(FULL_MASK, local_value, 2);
    if((thread_id & 31) >= 2){
        local_value += tmp;
    }
    tmp = __shfl_up_sync(FULL_MASK, local_value, 4);
    if((thread_id & 31) >= 4){
        local_value += tmp;
    }
    tmp = __shfl_up_sync(FULL_MASK, local_value, 8);
    if((thread_id & 31) >= 8){
        local_value += tmp;
    }
    tmp = __shfl_up_sync(FULL_MASK, local_value, 16);
    if((thread_id & 31) >= 16){
        local_value += tmp;
    }
    return local_value;
}

template<int N = 1024>
__global__ void prefix_sum_wrap(const float* __restrict__ input, float* __restrict__ output){
    const int thread_id = threadIdx.x;
    const int wrap_id = thread_id >> 5;
    __shared__ float shared_value[N >> 5];

    float local_value;
    if(thread_id < N){
        local_value = input[thread_id];
    }
    __syncthreads();

    local_value = warp_scan(local_value);
    if((thread_id & 31) == 31){
        shared_value[thread_id >> 5] = local_value;
    }
    __syncthreads();

    if(wrap_id == 0){
        const float temp = local_value;
        local_value = shared_value[thread_id];
        local_value = warp_scan(local_value);
        shared_value[thread_id] = local_value;
        local_value = temp;
    }
    __syncthreads();

    if(wrap_id){
        int local_bias = shared_value[wrap_id - 1];
        local_value += local_bias;
    }

    if(thread_id < N){
        output[thread_id] = local_value;
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

//------------------由Gemini创建的测试用例------------------
// 验证函数
bool verify(const float* cpu_res, const float* gpu_res, int n) {
    for (int i = 0; i < n; i++) {
        if (std::abs(cpu_res[i] - gpu_res[i]) > 1e-3) {
            printf("Error at index %d: CPU=%f, GPU=%f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

// 计时辅助函数
template<typename Func>
float benchmark(Func kernel_func, int n, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    kernel_func();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel_func();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds / iterations;
}

int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // 1. 数据准备
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_gpu(N);
    
    for (int i = 0; i < N; i++) h_input[i] = 1.0f; // 简单测试：全1数组前缀和应为 1, 2, 3...

    // CPU 计算参考结果
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_input[i];
        h_output_cpu[i] = sum;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    // 2. 定义测试套件
    auto run_naive = [&]() { prefix_sum_naive<N><<<1, N>>>(d_input, d_output); };
    auto run_double = [&]() { prefix_sum_double_buffer<N><<<1, N>>>(d_input, d_output); };
    auto run_tree = [&]() { prefix_sum_tree<N><<<1, N>>>(d_input, d_output); };
    auto run_tree_bc_free = [&]() { prefix_sum_tree_bank_conflict_free<N><<<1, N>>>(d_input, d_output); };
    auto run_warp = [&]() { prefix_sum_wrap<N><<<1, N>>>(d_input, d_output); };

    // 3. 正确性验证与性能测试
    struct TestNode { std::string name; decltype(run_naive) func; };
    std::vector<TestNode> tests = {
        {"Naive Scan (Double Sync)", run_naive},
        {"Double Buffer Scan", run_double},
        {"Brent-Kung Tree", run_tree},
        {"Brent-Kung (BC-Free)", run_tree_bc_free},
        {"Warp Shuffle Hierarchical", run_warp}
    };

    printf("%-30s | %-12s | %-10s\n", "Kernel Name", "Avg Time(ms)", "Status");
    printf("----------------------------------------------------------------------\n");

    for (auto& test : tests) {
        // 先跑一次验证正确性
        test.func();
        cudaMemcpy(h_output_gpu.data(), d_output, size, cudaMemcpyDeviceToHost);
        bool ok = verify(h_output_cpu.data(), h_output_gpu.data(), N);

        // 测速
        float avg_time = benchmark(test.func, N);
        
        printf("%-30s | %-12.5f | %-10s\n", 
               test.name.c_str(), avg_time, ok ? "PASS" : "FAIL");
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}