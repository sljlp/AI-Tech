# Nsight Compute 报告详细解读

## 报告概览

本报告使用 `ncu --set basic` 生成，包含 5 个 CUDA kernel 的性能分析数据。

---

## 通用结构说明

每个 kernel 分析包含 4 个 Section：

### 1. GPU Speed Of Light Throughput（GPU 极限吞吐量）
衡量 GPU 各核心组件的实际利用率与峰值性能的比值。

### 2. Launch Statistics（启动统计）
描述 kernel 启动配置和调度信息。

### 3. Occupancy（占用率）
分析活跃 warp 数量与理论最大值的比例。

### 4. GPU and Memory Workload Distribution（GPU 和内存工作负载分布）
显示各硬件单元（DRAM、L1、L2、SM、SMSP）的活跃周期分布。

---

## Kernel 1: sigmoid 激活函数

### Kernel 信息
```
void vectorized_elementwise_kernel<4, sigmoid_kernel_cuda...>
Grid: (256, 1, 1) × Block: (128, 1, 1)
总线程: 32,768
```

**用途**: PyTorch 的 sigmoid 激活函数，将输入值映射到 (0, 1) 区间。

### Section 1: GPU Speed Of Light Throughput

| 指标 | 值 | 含义 |
|------|-----|------|
| DRAM Frequency | 7.95 GHz | GPU 显存运行频率，接近 RTX 4050 的 8 GHz 峰值 |
| SM Frequency | 1.74 GHz | SM 流处理器频率，boost 频率约 1.7-2.0 GHz |
| Elapsed Cycles | 9,516 | kernel 执行的总时钟周期数 |
| **Memory Throughput** | **51.42%** | 内存带宽利用率，超过 50% 算较好 |
| DRAM Throughput | 51.42% | 显存控制器利用率，同上 |
| **Duration** | **5.47 us** | kernel 执行时间，非常快 |
| L1/TEX Cache Throughput | 19.77% | L1 缓存利用率偏低 |
| L2 Cache Throughput | 17.45% | L2 缓存利用率偏低 |
| SM Active Cycles | 7,152 | SM 实际工作的周期数 |
| **Compute (SM) Throughput** | **26.90%** | **计算吞吐量偏低 (<60%)** |

**OPT 提示**: 计算和内存吞吐都低于 60%，存在延迟瓶颈。建议查看 Scheduler 和 Warp State 统计。

### Section 2: Launch Statistics

| 指标 | 值 | 含义 |
|------|-----|------|
| Block Size | 128 | 每个 block 的线程数 |
| Grid Size | 256 | 总 block 数 |
| Registers Per Thread | 32 | 每个线程使用的寄存器数 |
| # SMs | 20 | GPU 总 SM 数量 |
| Threads | 32,768 | 256 × 128 = 32,768 线程 |
| **Waves Per SM** | **1.07** | **每个 SM 约 1 个 wave** |

**OPT 提示 (Est. Speedup: 50%)**: 
- 1 个完整 wave + 1 个 partial wave (16 blocks)
- partial wave 可能导致 **50% 时间浪费**
- 建议 grid size 设为 480 的倍数（20 SM × 24 blocks/SM）

### Section 3: Occupancy

| 指标 | 值 | 含义 |
|------|-----|------|
| Block Limit SM | 24 | SM 最多驻留 24 个 block |
| Block Limit Registers | 16 | 寄存器限制为 16 blocks |
| Block Limit Shared Mem | 32 | 共享内存允许 32 blocks |
| Block Limit Warps | 12 | warp 限制为 12 blocks |
| Theoretical Occupancy | 100% | 理论占用率（满） |
| **Achieved Occupancy** | **71.75%** | **实际占用率** |
| Achieved Active Warps | 34.44 | 每个 SM 平均 34 个活跃 warp |

**OPT 提示 (Est. Local Speedup: 28.25%)**:
- 理论与实际差距 28%，可能因 warp 调度开销或负载不均

### Section 4: Workload Distribution

| 指标 | 值 | 含义 |
|------|-----|------|
| Average DRAM Active Cycles | 22,379 | DRAM 平均活跃周期 |
| Total DRAM Elapsed Cycles | 130,560 | DRAM 总周期 |
| Average SM Active Cycles | 7,152 | SM 平均活跃周期 |
| Total SMSP Elapsed Cycles | 747,616 | SMSP 总周期 |

**OPT 提示 (Est. Speedup: 5.774%)**: SMSP 间负载不均，最大偏差 12.72%。

---

## Kernel 2: 加法运算

### Kernel 信息
```
void elementwise_kernel<128, 4, CUDAFunctor_add<float>>
Grid: (512, 1, 1) × Block: (128, 1, 1)
总线程: 65,536
```

**用途**: 浮点数逐元素加法操作。

### 关键指标对比 Kernel 1

| 指标 | Kernel 1 (sigmoid) | Kernel 2 (add) |
|------|-------------------|----------------|
| Duration | 5.47 us | 11.62 us |
| Memory Throughput | 51.42% | **26.40%** ↓ |
| Compute Throughput | 26.90% | **42.13%** ↑ |
| Achieved Occupancy | 71.75% | **81.90%** ↑ |
| Waves Per SM | 1.07 | **2.13** ↑ |

**分析**: 
- 加法 kernel 更慢（11.62 vs 5.47 us）因为 grid 更大
- 内存吞吐更低，但计算吞吐更高
- partial wave 浪费 33.3%（比 sigmoid 的 50% 好）

---

## Kernel 3: gatherTopK (Top-K 选择)

### Kernel 信息
```
void sbtopk::gatherTopK<float, unsigned int, 2, 0>
Grid: (8192, 1, 1) × Block: (32, 1, 1)
总线程: 262,144
```

**用途**: 从张量中选择 Top-K 元素，常用于 MoE (Mixture of Experts) 路由。

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| **Duration** | **94.43 us** | **最慢的 kernel** |
| Compute Throughput | **73.71%** | **>60%，计算密集** |
| Memory Throughput | 43.34% | 中等 |
| DRAM Throughput | 5.90% | **极低** |
| L1 Throughput | 44.21% | 较高 |
| Achieved Occupancy | 47.69% | 偏低 |
| Theoretical Occupancy | 50% | 受限 |
| Registers Per Thread | 41 | 高寄存器压力 |

**OPT 提示**: 计算比内存更重。建议检查计算是否有冗余，或使用查找表。

**分析**:
- Block size 仅 32 线程，导致理论 occupancy 上限 50%
- 高寄存器使用 (41) 和静态共享内存 (128 bytes/block)
- L1 缓存利用率高 (44%)，但 DRAM 极低 (5.9%)，说明数据主要在 L1 中

---

## Kernel 4: bitonicSortKVInPlace (双调排序)

### Kernel 信息
```
void bitonicSortKVInPlace<2, -1, 16, 16, float, long, GTOp>
Grid: (512, 1, 1) × Block: (16, 16, 1)
总线程: 131,072
```

**用途**: 双调排序算法，对键值对进行 in-place 排序。

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| **Memory Throughput** | **81.80%** | **>80%，极好！** |
| **Compute Throughput** | **81.80%** | **>80%，极好！** |
| **L1 Throughput** | **85.53%** | **>80%，极好！** |
| Duration | 46.82 us | 中等 |
| Achieved Occupancy | 78.06% | 良好 |
| Theoretical Occupancy | 83.33% | 受寄存器/共享内存限制 |
| Block Limit Registers | **5** | 寄存器是主要瓶颈 |

**INF 提示 (>80%)**: 
> 此 workload 利用率超过设备峰值性能的 80%。要进一步优化，需将工作从最繁忙的单元转移到其他单元。

**分析**: 这是**优化最好的 kernel**，各指标都超过 80%。瓶颈在寄存器（每线程 45 个，限制每个 SM 只能驻留 5 个 block）。

---

## Kernel 5: reduce (归约运算)

### Kernel 信息
```
void reduce_kernel<512, 1, ReduceOp<float, sum_functor>>
Grid: (32, 1, 1) × Block: (512, 1, 1)
总线程: 16,384
```

**用途**: 归约求和操作，将数组元素累加为单个值。

### 关键指标

| 指标 | 值 | 状态 |
|------|-----|------|
| **Duration** | **4.38 us** | **最快** |
| Memory Throughput | **13.96%** | **极低** |
| Compute Throughput | **13.76%** | **极低** |
| **Waves Per SM** | **0.53** | **<1，严重不足** |
| Achieved Occupancy | 55.45% | 差 |
| Block Limit Registers | **4** | 极低 |
| Block Limit Warps | **3** | 极低 |

**OPT 提示**:
1. **Grid 太小**: 32 个 block 仅占 0.5 个 wave，20 个 SM 大部分空闲
2. **建议**: 每个 SM 至少 2 个 block 以掩盖 `__syncthreads()` 等待时间
3. **Est. Local Speedup: 44.55%**: occupancy 差距大

**分析**: 这是**最需要优化的 kernel**。Grid 仅 32 个 block，远小于 20 个 SM 的容量。

---

## Kernel 6: gatherTopK (第二轮)

与 Kernel 3 相同 kernel，但 Grid 更小 (1024 vs 8192)。

| 指标 | Kernel 3 | Kernel 6 |
|------|----------|----------|
| Grid Size | 8,192 | 1,024 |
| Duration | 94.43 us | 17.12 us |
| Compute Throughput | 73.71% | 55.16% |
| Waves Per SM | 17.07 | 2.13 |

Grid 缩小 8 倍，性能相应下降。

---

## 总结与建议

### 性能排名

| Kernel | Duration | 利用率 | 优化空间 |
|--------|----------|--------|----------|
| bitonicSort | 46.82 us | **81%** | 小 |
| gatherTopK (大) | 94.43 us | 73% | 中 |
| gatherTopK (小) | 17.12 us | 55% | 中 |
| sigmoid | 5.47 us | 51% | 中 |
| add | 11.62 us | 42% | 大 |
| reduce | 4.38 us | **14%** | **极大** |

### 通用优化策略

1. **增大 Grid Size**: reduce kernel (32 blocks) 最紧急
2. **减少寄存器使用**: bitonicSort (45 regs) 和 gatherTopK (41 regs)
3. **消除 Partial Wave**: sigmoid (50% 浪费) 和 add (33% 浪费)
4. **Kernel Fusion**: sigmoid + add 可融合减少 launch 开销

### Partial Wave 快速参考

| Kernel | Grid | Wave 利用 | 浪费 |
|--------|------|-----------|------|
| reduce | 32 | 0.5 wave | **~50%** |
| sigmoid | 256 | 1.07 wave | **~50%** |
| add | 512 | 2.13 wave | **~33%** |
| gatherTopK 小 | 1,024 | 2.13 wave | **~33%** |
| gatherTopK 大 | 8,192 | 17.07 wave | **~7%** |
| bitonicSort | 512 | 5.12 wave | **~12%** |

---

*报告生成时间: 2026-05-23*
