# FineKernels

练习实现最优经典算子。

## Softmax

实现思路: [块/线程规划](SoftmaxKernel/实现思路.md)

### 测试性能

| 特征维度 | 基础实现      | 优化实现      | 加速比   |
| ---- | --------- | --------- | ----- |
| 128  | 0.0463 ms | 0.0185 ms | 2.50x |
| 1024 | 0.5891 ms | 0.5914 ms | 1.00x |

## Reduce

实现思路: [块/线程规划与线程束原子操作](ReduceKernel/实现思路.md)

### 测试性能

| 元素数量      | 基础实现      | 优化实现      | 加速比   | 基础带宽        | 优化带宽        |
| --------- | --------- | --------- | ----- | ----------- | ----------- |
| 1,048,576 | 0.1126 ms | 0.0154 ms | 7.31x | 37.24 GB/s  | 273.07 GB/s |
| 262,144   | 0.0051 ms | 0.0041 ms | 1.24x | 204.80 GB/s | 256.00 GB/s |

## FastDiagramKernel

实现思路: [原子操作统计](FastDiagramKernel/实现思路.md)

## Transpose Kernel

实现思路: [矩阵转置](TransposeKernel/实现思路.md)

### 测试性能(矩阵尺寸: 8192×8192)

| 实现方案                           | 执行时间      | 相对Naive加速比 | 优化技术     |
| ------------------------------ | --------- | ---------- | -------- |
| Naive Transpose                | 2.0154 ms | 1.00x      | 基础实现     |
| Smem Direct Map                | 0.6605 ms | 3.05x      | 共享内存缓存   |
| Smem Conflict Free (+1)        | 0.6574 ms | 3.07x      | Bank冲突消除 |
| Smem Double Fetch (Vectorized) | 0.6553 ms | 3.08x      | 向量化预取    |

## RMSNorm Kernel

实现思路: [RMSNorm](RMSNormKernel/实现思路.md)

### 测试性能

| 测试配置          | 实现方案      | 执行时间      |
| ------------- | --------- | --------- |
| N=8, C=4096  | Block Collapse | 0.5509 ms |
| N=8, C=4096  | Grid Collapse  | 0.0227 ms |
| N=1024, C=4096 | Block Collapse | 0.1055 ms |
| N=1024, C=4096 | Grid Collapse  | 0.0655 ms |
