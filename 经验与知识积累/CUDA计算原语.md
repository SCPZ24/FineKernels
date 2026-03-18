# CUDA计算原语

本文整理了CUDA中常用的计算原语，特别是类似于`fmaxf`的函数。

## 1. 最大值和最小值函数

### 单精度浮点数
- `fmaxf(float x, float y)`: 返回两个单精度浮点数中的较大值
- `fminf(float x, float y)`: 返回两个单精度浮点数中的较小值

### 双精度浮点数
- `fmax(double x, double y)`: 返回两个双精度浮点数中的较大值
- `fmin(double x, double y)`: 返回两个双精度浮点数中的较小值

### 整数
- `max(int x, int y)`: 返回两个整数中的较大值
- `min(int x, int y)`: 返回两个整数中的较小值
- `maxll(long long x, long long y)`: 返回两个长整数中的较大值
- `minll(long long x, long long y)`: 返回两个长整数中的较小值

## 2. 绝对值函数

- `fabsf(float x)`: 返回单精度浮点数的绝对值
- `fabs(double x)`: 返回双精度浮点数的绝对值
- `abs(int x)`: 返回整数的绝对值
- `absll(long long x)`: 返回长整数的绝对值

## 3. 取整函数

- `ceilf(float x)`: 向上取整
- `ceild(double x)`: 向上取整（双精度）
- `floorf(float x)`: 向下取整
- `floord(double x)`: 向下取整（双精度）
- `roundf(float x)`: 四舍五入
- `roundd(double x)`: 四舍五入（双精度）
- `truncf(float x)`: 截断小数部分
- `trunc(double x)`: 截断小数部分（双精度）

## 4. 数学函数

### 基本运算
- `sqrtf(float x)`: 平方根
- `sqrt(double x)`: 平方根（双精度）
- `rsqrtf(float x)`: 平方根的倒数
- `rsqrt(double x)`: 平方根的倒数（双精度）
- `powf(float x, float y)`: 幂运算
- `pow(double x, double y)`: 幂运算（双精度）

### 三角函数
- `sinf(float x)`: 正弦
- `sin(double x)`: 正弦（双精度）
- `cosf(float x)`: 余弦
- `cos(double x)`: 余弦（双精度）
- `tanf(float x)`: 正切
- `tan(double x)`: 正切（双精度）

### 指数和对数
- `expf(float x)`: 指数函数
- `exp(double x)`: 指数函数（双精度）
- `logf(float x)`: 自然对数
- `log(double x)`: 自然对数（双精度）
- `log10f(float x)`: 以10为底的对数
- `log10(double x)`: 以10为底的对数（双精度）

## 5.  warp级原语

- `__shfl_sync(unsigned mask, T value, int lane, int width=warpSize)`: warp内数据交换
- `__reduce_add_sync(unsigned mask, T value)`: warp内求和
- `__reduce_min_sync(unsigned mask, T value)`: warp内求最小值
- `__reduce_max_sync(unsigned mask, T value)`: warp内求最大值

## 6. 位操作函数

- `__brev(unsigned int x)`: 位反转
- `__clz(unsigned int x)`: 前导零计数
- `__popc(unsigned int x)`: 置位计数
- `__ffs(unsigned int x)`: 最低有效置位位置

## 7. 其他常用原语

- `__float2int_rn(float x)`: 浮点数转整数（四舍五入）
- `__int2float_rn(int x)`: 整数转浮点数
- `__saturatef(float x)`: 饱和到[0,1]范围
- `__fmaf(float a, float b, float c)`: 融合乘加运算 (a*b + c)

## 8. 使用示例

### 示例1: 使用fmaxf
```cuda
float a = 3.14f;
float b = 2.71f;
float max_val = fmaxf(a, b); // 返回3.14f
```

### 示例2: 使用warp级原语
```cuda
__shared__ float sdata[256];
unsigned int mask = __activemask();
float val = sdata[threadIdx.x];
float max_val = __reduce_max_sync(mask, val);
```

## 9. 性能注意事项

1. **精度选择**: 根据需求选择合适的精度（单精度/双精度），单精度通常性能更高
2. **内联函数**: 这些原语通常是内联的，不会产生函数调用开销
3. **硬件支持**: 大部分原语都有硬件指令支持，性能优异
4. **warp级原语**: 在处理warp内数据时，使用warp级原语可以获得更高的性能

## 10. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
- [CUDA Warp-Level Primitives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-level-primitives)