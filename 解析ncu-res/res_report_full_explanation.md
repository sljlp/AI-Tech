# Nsight Compute res.txt 报告完整解读

## 报告概览

- **来源**: `ncu --set basic` 生成的性能分析报告
- **文件**: `/path/to/workspace/gpu_kernels/cusrc/ops/res.txt` (4908 行)
- **GPU**: NVIDIA RTX 4050 Laptop (Ada Lovelace, sm_89, CC 8.9)
- **SM 数量**: 20
- **TPC 数量**: 10

---

## 一、通用结构说明

每个 kernel 分析包含 4 个 Section：

### 1. GPU Speed Of Light Throughput（GPU 极限吞吐量）
衡量 GPU 各核心组件的实际利用率与峰值性能的比值。

### 2. Launch Statistics（启动统计）
描述 kernel 启动配置和调度信息。

### 3. Occupancy（占用率）
分析活跃 warp 数量与理论最大值的比例。

### 4. GPU and Memory Workload Distribution（GPU 和内存工作负载分布）
显示各硬件单元的活跃周期分布。

---

## 二、Ada 架构硬件规格

| 限制 | 值 |
|------|-----|
| 每 SM 最大 block | 24 |
| 每 SM 并发 warp | 48 |
| 每 SM 寄存器 | 65,536 (64K) |
| 每 SM 共享内存 | 100 KB (可配置) |
| 每线程最大寄存器 | 255 |
| 每 block 预留共享内存 | 1 KB |
| L2 缓存 | 2 MB |

---

## 三、Kernel 1: sigmoid 激活函数

### Kernel 信息
```
void vectorized_elementwise_kernel<4, sigmoid_kernel_cuda...>
Grid: (256, 1, 1) × Block: (128, 1, 1)
总线程: 32,768
```

**用途**: PyTorch 的 sigmoid 激活函数。

### GPU Speed Of Light Throughput

| 指标 | 值 | 含义 |
|------|-----|------|
| DRAM Frequency | 7.95 GHz | 显存运行频率 |
| SM Frequency | 1.74 GHz | SM 流处理器频率 |
| Elapsed Cycles | 9,516 | 总时钟周期 |
| Memory Throughput | 51.42% | 内存带宽利用率 |
| Duration | 5.47 us | 执行时间 |
| L1/TEX Cache Throughput | 19.77% | L1 缓存利用率 |
| L2 Cache Throughput | 17.45% | L2 缓存利用率 |
| Compute (SM) Throughput | 26.90% | 计算吞吐量偏低 |

### Launch Statistics

| 指标 | 值 | 含义 |
|------|-----|------|
| Block Size | 128 | 每 block 线程数 |
| Grid Size | 256 | 总 block 数 |
| Registers Per Thread | 32 | 每线程寄存器 |
| Shared Memory Config | 32.77 KB | 每 SM 共享内存配置 |
| Driver Shared Mem | 1.02 KB | 驱动预留 |
| Static/Dynamic Shared | 0 | 未使用共享内存 |
| Stack Size | 1,024 | 每线程栈大小 |
| Threads | 32,768 | 256 × 128 |
| Waves Per SM | 1.07 | 1 完整 + 部分 wave |

### Occupancy

| 指标 | 值 | 计算 |
|------|-----|------|
| Block Limit SM | 24 | 硬件限制 |
| Block Limit Registers | 16 | 65536/(32×128) |
| Block Limit Shared Mem | 32 | 32768/1024 |
| Block Limit Warps | 12 | 48/(128/32) |
| Theoretical Warps | 48 | min(24,16,32,12)×4 |
| Theoretical Occupancy | 100% | 48/48 |
| Achieved Occupancy | 71.75% | 硬件计数器实测 |
| Achieved Active Warps | 34.44 | 硬件计数器实测 |

**OPT**: Est. Speedup 50% - partial wave 浪费。

---

## 四、Kernel 2: 加法运算

### Kernel 信息
```
void elementwise_kernel<128, 4, CUDAFunctor_add<float>>
Grid: (512, 1, 1) × Block: (128, 1, 1)
总线程: 65,536
```

### 关键指标

| 指标 | 值 |
|------|-----|
| Duration | 11.62 us |
| Memory Throughput | 26.40% |
| Compute Throughput | 42.13% |
| Achieved Occupancy | 81.90% |
| Waves Per SM | 2.13 |

**OPT**: Est. Speedup 33.33% - 2 完整 + 部分 wave。

---

## 五、Kernel 3: gatherTopK (大)

### Kernel 信息
```
void sbtopk::gatherTopK<float, unsigned int, 2, 0>
Grid: (8192, 1, 1) × Block: (32, 1, 1)
总线程: 262,144
```

**用途**: MoE 路由 Top-K 选择。

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| Duration | 94.43 us | 最慢 |
| Compute Throughput | 73.71% | >60%，计算密集 |
| DRAM Throughput | 5.90% | 极低 |
| L1 Throughput | 44.21% | 较高 |
| Block Size | 32 | 每 block 仅 1 warp |
| Registers Per Thread | 41 | 高 |
| Static Shared Mem | 128 bytes | 使用共享内存 |
| Theoretical Occupancy | 50% | 受限 |
| Achieved Occupancy | 47.69% | 接近理论 |

**分析**: Block Size 仅 32 (1 warp)，理论 occupancy 上限 50%。

---

## 六、Kernel 4: bitonicSortKVInPlace

### Kernel 信息
```
void bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp>
Grid: (512, 1, 1) × Block: (16, 16, 1) = (256 threads)
总线程: 131,072
```

**用途**: 双调排序，键值对 in-place 排序。

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| Memory Throughput | **81.80%** | >80%，极好 |
| Compute Throughput | **81.80%** | >80%，极好 |
| L1 Throughput | **85.53%** | >80%，极好 |
| L2 Throughput | 1.45% | 极低（L1 命中高） |
| Duration | 46.82 us | 中等 |
| Achieved Occupancy | 78.06% | 良好 |
| Block Limit Registers | 5 | 寄存器瓶颈 |
| Static Shared Mem | 6.66 KB | 使用较多 |

**INF 提示**: 利用率超 80%，是优化最好的 kernel。

---

## 七、Kernel 5: reduce (归约)

### Kernel 信息
```
void reduce_kernel<512, 1, ReduceOp<float, sum_functor>>
Grid: (32, 1, 1) × Block: (512, 1, 1)
总线程: 16,384
```

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| Duration | 4.38 us | 最快 |
| Memory Throughput | **13.96%** | 极低 |
| Compute Throughput | **13.76%** | 极低 |
| Waves Per SM | **0.53** | <1，严重不足 |
| Grid Size | 32 | 太小 |
| Block Limit Registers | 4 | 极低 |

**OPT**: Grid 太小，仅 0.5 wave，SM 大部分空闲。

---

## 八、关键概念详解

### 1. TPCs (Texture/Processor Clusters)

```
RTX 4050: 10 TPCs × 2 SMs/TPC = 20 SMs

TPC 是物理分组，共享纹理单元和 L1。
日常优化无需关心 TPC。
```

### 2. Waves Per SM

```
1 Wave = 填满所有 SM 的 block 集合

Waves = Grid / (SM × blocks_per_sm)

示例:
  Grid = 512, 每 SM = 12 blocks
  Waves = 512 / (20 × 12) = 2.13
  = 2 完整 + 0.13 部分 wave
```

### 3. Partial Wave

```
Grid = 12 blocks, 满 wave 需 100 blocks

12/100 = 12% 利用率
8 个 SM 完全空闲!
```

### 4. Occupancy 计算

```
Block Limit = min(
    SM 限制: 24,
    寄存器: 65536/(regs×threads),
    共享内存: shared_mem_total/shared_mem_per_block,
    Warps: 48/(block_size/32)
)

Theoretical Warps = Block Limit × (block_size/32)
Theoretical Occupancy = Theoretical Warps / 48
Achieved Occupancy = 硬件计数器实测
```

### 5. 为什么 gatherTopK 只有 50% 而 sigmoid 是 100%？

| Kernel | Block Size | 每 block warp | Theoretical Warps | Occupancy |
|--------|-----------|---------------|-------------------|-----------|
| sigmoid | 128 | 4 | 12×4=48 | 48/48=100% |
| gatherTopK | 32 | 1 | 24×1=24 | 24/48=50% |

**原因**: Block Size 32 太小，每 block 仅 1 warp，无法填满 48 warp slot。

---

## 九、Registers Per Thread

### 来源
编译期 PTX/SASS 确定，不是运行时测量。

### 查看方法
```bash
nvcc -Xptxas -v kernel.cu
# Used 32 registers...
```

### 影响
```
每 SM 寄存器 = 65,536
每线程 32 寄存器，每 block 128 线程
每 block 需要 = 32×128 = 4,096
Block Limit = 65,536/4,096 = 16 blocks/SM
```

---

## 十、共享内存详解

### Shared Memory Configuration Size: 32.77 KB
每 SM 分配给共享内存的大小（默认配置）。

### Static/Dynamic Shared Memory
- **Static**: `__shared__ float data[256];` (编译期)
- **Dynamic**: `kernel<<<grid,block,shared_bytes>>>();` (运行时)

### L1 vs 共享内存
```
每 SM 总内存 = 128 KB
可配置比例:
  默认: 32 KB 共享 + 96 KB L1
  最大共享: 128 KB 共享 + 0 KB L1
```

---

## 十一、L1 Cache vs L2 Cache

| | L1 | L2 |
|---|---|---|
| **位置** | 每 SM 内部 | 全局共享 |
| **大小** | 96 KB/SM | 2 MB |
| **延迟** | ~20 周期 | ~150 周期 |
| **可见性** | SM 私有 | 所有 SM |

### L2 访问周期分解

| 阶段 | 周期 |
|------|------|
| 地址翻译 | 5-10 |
| L2 标签查找 | 15-25 |
| 数据传输 (内部) | 20-30 |
| 跨芯片 NoC 传输 | 40-80 |
| 缓存一致性 | 10-20 |
| 队列等待 | 10-35 |
| **总计** | **~150** |

### 报告中的表现

```
bitonicSort:
  L1: 85.53%, L2: 1.45%
  → 数据在 L1 命中，避免 L2 高延迟
  → 性能好的原因
```

---

## 十二、Stack Size 与 Threads

### Stack Size: 1,024
每线程调用栈大小（编译器设置），用于局部变量和函数调用。

### Threads
```
= Grid × Block
运行时有代码决定
```

---

## 十三、性能总结

| Kernel | Duration | 利用率 | 主要瓶颈 |
|--------|----------|--------|----------|
| bitonicSort | 46.82 us | **81%** | 小 |
| gatherTopK (大) | 94.43 us | 73% | Block Size 小 |
| sigmoid | 5.47 us | 51% | Partial Wave |
| add | 11.62 us | 42% | Partial Wave |
| reduce | 4.38 us | **14%** | Grid 太小 |

---

## 十四、优化建议

1. **增大 Grid Size**: reduce (32 blocks) 最紧急
2. **增大 Block Size**: gatherTopK (32→128) 可提升 occupancy
3. **减少寄存器**: bitonicSort (45 regs) 和 gatherTopK (41 regs)
4. **消除 Partial Wave**: sigmoid (50% 浪费)
5. **Kernel Fusion**: sigmoid + add 可融合

---

*报告生成时间: 2026-05-23*
*原始报告: /path/to/workspace/gpu_kernels/cusrc/ops/res.txt*
