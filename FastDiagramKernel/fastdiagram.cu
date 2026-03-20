#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BINS 256
#define BLOCK_SIZE 256

// ==========================================================
// 由Gemini提供的脚手架代码
// 功能为统计频数分布直方图
// ==========================================================
__global__ void histogram_kernel(const uint8_t* __restrict__ input, int* __restrict__ hist, int n) {
    __shared__ int shared_hist[BINS];
    
    // 初始化共享内存
    for(int i = threadIdx.x; i < BINS; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // 每个线程处理多个元素
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for(int i = tid; i < n; i += stride) {
        uint8_t value = input[i];
        atomicAdd(&shared_hist[value], 1);
    }
    
    __syncthreads();
    
    // 将局部直方图合并到全局直方图
    for(int i = threadIdx.x; i < BINS; i += blockDim.x) {
        atomicAdd(&hist[i], shared_hist[i]);
    }
}

// CPU 参考实现用于校验
void cpu_histogram(const uint8_t* input, int* hist, int n) {
    for (int i = 0; i < n; i++) {
        hist[input[i]]++;
    }
}

int main() {
    int n = 1024 * 1024 * 64; // 64MB 数据量
    size_t size = n * sizeof(uint8_t);
    size_t hist_size = BINS * sizeof(int);

    // 分配主机内存
    std::vector<uint8_t> h_input(n);
    std::vector<int> h_hist(BINS, 0);
    std::vector<int> cpu_res(BINS, 0);

    // 随机生成 0-255 的像素数据
    for (int i = 0; i < n; i++) h_input[i] = rand() % BINS;

    // 分配设备内存
    uint8_t* d_input;
    int* d_hist;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_hist, hist_size);

    // 拷贝数据并清空结果表
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_size);

    // 定义执行配置
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 计时开始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 执行 Kernel
    histogram_kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_hist, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 拷贝结果回主机
    cudaMemcpy(h_hist.data(), d_hist, hist_size, cudaMemcpyDeviceToHost);

    // 校验结果
    cpu_histogram(h_input.data(), cpu_res.data(), n);
    bool success = true;
    for (int i = 0; i < BINS; i++) {
        if (h_hist[i] != cpu_res[i]) {
            std::cerr << "Error at bin " << i << ": GPU=" << h_hist[i] << ", CPU=" << cpu_res[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
        std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
        std::cout << "Throughput: " << (n / milliseconds / 1e6) << " GB/s" << std::endl;
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_hist);
    return 0;
}