# CUDA Warp级原语

本文整理了CUDA中常用的warp级原语，特别是`__shfl_sync`、`__shfl_up_sync`和`__shfl_down_sync`三个函数的使用方法和效果。

## 1. __shfl_sync

### 调用方法
```cuda
T __shfl_sync(unsigned mask, T value, int lane, int width=warpSize);
```

### 参数说明
- `mask`: 活动线程掩码，通常使用`__activemask()`获取
- `value`: 要交换的值
- `lane`: 源线程的lane ID
- `width`: warp的宽度，默认为32

### 效果
从指定lane的线程中获取value值，并返回给当前线程。这是一个直接的点对点交换操作。

### 示例
```cuda
__global__ void shfl_example(float* data) {
    int lane_id = threadIdx.x % 32;
    float value = data[threadIdx.x];
    
    // 从lane 0获取值
    float lane0_value = __shfl_sync(__activemask(), value, 0);
    
    // 从lane 15获取值
    float lane15_value = __shfl_sync(__activemask(), value, 15);
    
    printf("Thread %d: lane0_value = %f, lane15_value = %f\n", threadIdx.x, lane0_value, lane15_value);
}
```

## 2. __shfl_up_sync

### 调用方法
```cuda
T __shfl_up_sync(unsigned mask, T value, unsigned int delta, int width=warpSize);
```

### 参数说明
- `mask`: 活动线程掩码
- `value`: 要交换的值
- `delta`: 向上移动的lane数量
- `width`: warp的宽度，默认为32

### 效果
从当前线程上方（lane ID较小）的线程中获取值。具体来说，当前线程会获取lane ID为`lane_id - delta`的线程的值。如果`lane_id < delta`，则返回当前线程的原始值。

### 示例
```cuda
__global__ void shfl_up_example(float* data) {
    int lane_id = threadIdx.x % 32;
    float value = data[threadIdx.x];
    
    // 从上方1个lane获取值
    float up_value = __shfl_up_sync(__activemask(), value, 1);
    
    // 从上方2个lane获取值
    float up2_value = __shfl_up_sync(__activemask(), value, 2);
    
    printf("Thread %d: original = %f, up1 = %f, up2 = %f\n", threadIdx.x, value, up_value, up2_value);
}
```

## 3. __shfl_down_sync

### 调用方法
```cuda
T __shfl_down_sync(unsigned mask, T value, unsigned int delta, int width=warpSize);
```

### 参数说明
- `mask`: 活动线程掩码
- `value`: 要交换的值
- `delta`: 向下移动的lane数量
- `width`: warp的宽度，默认为32

### 效果
从当前线程下方（lane ID较大）的线程中获取值。具体来说，当前线程会获取lane ID为`lane_id + delta`的线程的值。如果`lane_id + delta >= width`，则返回当前线程的原始值。

### 示例
```cuda
__global__ void shfl_down_example(float* data) {
    int lane_id = threadIdx.x % 32;
    float value = data[threadIdx.x];
    
    // 从下方1个lane获取值
    float down_value = __shfl_down_sync(__activemask(), value, 1);
    
    // 从下方2个lane获取值
    float down2_value = __shfl_down_sync(__activemask(), value, 2);
    
    printf("Thread %d: original = %f, down1 = %f, down2 = %f\n", threadIdx.x, value, down_value, down2_value);
}
```

## 4. 应用场景

### 4.1 归约操作
```cuda
__global__ void reduce_example(float* input, float* output) {
    int lane_id = threadIdx.x % 32;
    float value = input[threadIdx.x];
    
    // 使用__shfl_down_sync进行归约
    for (int delta = 16; delta > 0; delta /= 2) {
        value += __shfl_down_sync(__activemask(), value, delta);
    }
    
    if (lane_id == 0) {
        output[blockIdx.x] = value;
    }
}
```

### 4.2 广播操作
```cuda
__global__ void broadcast_example(float* input, float* output) {
    int lane_id = threadIdx.x % 32;
    float value = input[threadIdx.x];
    
    // 将lane 0的值广播到所有线程
    float broadcast_value = __shfl_sync(__activemask(), value, 0);
    
    output[threadIdx.x] = broadcast_value;
}
```

## 5. 性能注意事项

1. **无内存访问**：这些原语在寄存器级别操作，不涉及全局内存或共享内存访问，因此非常快
2. **warp内操作**：这些原语只能在同一个warp内的线程之间进行数据交换
3. **掩码使用**：正确使用`mask`参数确保只与活动线程进行交换
4. **宽度参数**：通过设置`width`参数，可以在warp内创建更小的逻辑分组

## 6. 参考资料

- [CUDA C++ Programming Guide - Warp-Level Primitives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-primitives)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
