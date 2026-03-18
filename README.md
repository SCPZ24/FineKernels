# FineKernels
练习实现最优经典算子。

## Softmax

### 实现概述

实现了两种Softmax CUDA kernel：基础实现和优化实现。

### 基础实现 (softmax_forward_naive)

#### 线程与Block分配
- **Block分配**：每个Block负责计算一个样本（row）
- **Thread分配**：每个Block使用128个线程
- **Warp分配**：128线程 = 4个Warp（每个Warp 32线程）

#### 内存使用
- **全局内存**：输入和输出数据
- **共享内存**：存储Warp级中间结果，大小为 `num_wraps * sizeof(float)`

### 优化实现 (softmax_forward_optimize)

#### 线程与Block分配
- **Block分配**：每个Block负责计算4个样本（rows_per_block = 4）
- **Thread分配**：每个Block使用128个线程
- **Warp分配**：4个Warp，每个Warp负责一个样本（row）

#### 关键优化
1. **向量化读取**：使用`float4`类型一次读取4个元素，提高内存带宽利用率
2. **寄存器优化**：
   - 使用`float4 reg_val_f4[float4_per_thread]`存储中间结果
   - 减少内存访问次数
3. **内存访问模式**：
   - 优化全局内存读取模式，提高缓存命中率
   - 避免非对齐访问

### 性能对比(N = 65536)

| 特征维度 | 基础实现 | 优化实现 | 加速比 |
|---------|---------|---------|--------|
| 128     | 0.0463 ms   | 0.0185 ms   | 2.50x    |
| 1024    | 0.5891 ms   | 0.5914 ms   | 1.00x    |