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
    const int global_x = (blockIdx.x * blockDim.x << 1) + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_offset = global_y * N + global_x;

    int shared_offset_row, shared_offset_col;
    {
        const int block_linear_index = blockDim.x * threadIdx.y + threadIdx.x;
        shared_offset_row = block_linear_index / blockDim.y;
        shared_offset_col = block_linear_index % blockDim.y;
    }

    const int global_x_target = blockDim.y * blockIdx.y + shared_offset_col;
    const int global_y_target = (blockDim.x * blockIdx.x << 1) + shared_offset_row;

    if(global_x < M && global_y < N){
        {
            const int cache_offset = threadIdx.y * cache_row_size + threadIdx.x;
            cache[cache_offset] = input[input_offset];
            cache[cache_offset + blockDim.x] = input[input_offset + blockDim.x];
        }

        __syncthreads();
        {
            const int cache_offset = shared_offset_col * cache_row_size + shared_offset_row;
            output[global_y_target * M + global_x_target] = cache[cache_offset];
            output[(global_y_target + blockDim.x) * M + global_x_target] = cache[cache_offset + blockDim.x];
        }
    }
}


// Gemini构造的测试用例
void checkResult(float *out, float *ref, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(out[i] - ref[i]) > 1e-5) {
            std::cerr << "Result Check Failed at index " << i << ": GPU=" << out[i] << ", REF=" << ref[i] << std::endl;
            return;
        }
    }
    std::cout << "Result Passed!" << std::endl;
}

int main() {
    // 矩阵尺寸 (确保是 Tile 的倍数以便测试，工业级代码需处理边界)
    const int M = 4096;
    const int N = 4096;
    const int size = M * N;
    size_t bytes = size * sizeof(float);

    // Host 内存
    std::vector<float> h_in(size), h_out(size), h_ref(size);
    for (int i = 0; i < size; i++) h_in[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU 参考结果
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            h_ref[x * M + y] = h_in[y * N + x];
        }
    }

    // Device 内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // 计时准备
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto test_kernel = [&](auto kernel, dim3 grid, dim3 block, std::string name) {
        cudaMemset(d_out, 0, bytes);
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_in, d_out, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        
        cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        std::cout << std::left << std::setw(35) << name << " | Time: " << std::fixed << std::setprecision(4) << ms << " ms | ";
        checkResult(h_out.data(), h_ref.data(), size);
    };

    // 1. Naive
    dim3 block1(SN_VAL, SM_VAL);
    dim3 grid1((N + SN_VAL - 1) / SN_VAL, (M + SM_VAL - 1) / SM_VAL);
    test_kernel(transpose_naive, grid1, block1, "Naive Transpose");

    // 2. Shared Direct Map
    test_kernel(transpose_shared_direct_map<SM_VAL, SN_VAL>, grid1, block1, "Smem Direct Map");

    // 3. Shared Bank Conflict Free
    test_kernel(transpose_shared_bank_conflict_free<SM_VAL, SN_VAL>, grid1, block1, "Smem Conflict Free (+1)");

    // 4. Double Fetch (特别注意：blockDim.x 减半)
    dim3 block_half(SN_VAL / 2, SM_VAL);
    dim3 grid_double((N + SN_VAL - 1) / SN_VAL, (M + SM_VAL - 1) / SM_VAL);
    test_kernel(transpose_shared_bank_conflict_free_double_fetch<SM_VAL, SN_VAL>, grid_double, block_half, "Smem Double Fetch (Vectorized)");

    // 资源释放
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}