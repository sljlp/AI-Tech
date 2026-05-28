# Concat and Cache MLA 性能分析报告

## 分析环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| CUDA Compute Capability | 8.9 |
| SM 数量 | 20 |
| ncu 版本 | 2025.1.0 |
| PyTorch 版本 | 2.7.1+cu126 |

## 分析方法

### 工具
- **NVIDIA Nsight Compute (ncu)**: 用于分析 CUDA kernel 性能
- **分析模式**: `--set basic` (基础指标集)

### 被测 Kernel
```
void concat_and_cache_mla_kernel<T1, T2>(
    const T1 *, const T1 *, T2 *, const long *, 
    int, int, int, int, int, int, int, const float *)
```

### 配置参数
| 参数 | 值 |
|------|-----|
| num_tokens | 2048 |
| kv_c_dim | 512 |
| k_pe_dim | 64 |
| block_size | 64 |
| num_blocks | 32 |
| Grid Size | (2048, 1, 1) |
| Block Size | (512, 1, 1) |
| 总线程数 | 1,048,576 |

### 测试数据类型
- `torch.bfloat16`
- `torch.float16`
- `torch.float32`

## 性能指标

### bfloat16 性能数据

#### GPU Speed Of Light Throughput
| 指标 | 值 |
|------|-----|
| DRAM Frequency | 7.98 GHz |
| SM Frequency | 1.77 GHz |
| Elapsed Cycles | 88,960 |
| Duration | 50.08 us |
| Memory Throughput | 24.81% |
| DRAM Throughput | 24.81% |
| L1/TEX Cache Throughput | 16.13% |
| L2 Cache Throughput | 10.42% |
| Compute (SM) Throughput | 36.79% |
| SM Active Cycles | 86,342 |

#### Launch Statistics
| 指标 | 值 |
|------|-----|
| Block Size | 512 |
| Grid Size | 2048 |
| Registers Per Thread | 22 |
| Waves Per SM | 34.13 |
| Threads | 1,048,576 |

#### Occupancy
| 指标 | 值 |
|------|-----|
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 68.74% |
| Achieved Active Warps Per SM | 33.00 |
| Theoretical Active Warps per SM | 48 |
| Block Limit Registers | 5 |

### float16 性能数据

| 指标 | 值 |
|------|-----|
| Duration | 50.62 us |
| Memory Throughput | 24.53% |
| Compute Throughput | 36.88% |
| Achieved Occupancy | 68.37% |
| Elapsed Cycles | 88,284 |

### float32 性能数据

| 指标 | 值 |
|------|-----|
| Duration | 50.30 us |
| Memory Throughput | 24.70% |
| Compute Throughput | 36.85% |
| Achieved Occupancy | 68.74% |
| Elapsed Cycles | 88,692 |

## 性能瓶颈分析

### 1. 低内存带宽利用率 (24.81%)

**现象**: Memory Throughput 仅 24.81%，远低于 60% 的理想阈值。

**原因**:
- 数据量较小 (2048 tokens × 576 dim)，无法充分利用 GPU 内存带宽
- 随机内存访问模式 (slot_mapping) 导致 DRAM 效率低下

### 2. 低计算吞吐量 (36.79%)

**现象**: Compute (SM) Throughput 仅 36.79%，未达到计算饱和。

**原因**:
- 该 kernel 是内存密集型操作 (concat + cache write)
- 计算密度低，每个线程执行的算术操作很少

### 3. Occupancy 不足 (68.74% vs 100%)

**瓶颈来源**: Block Limit Registers = 5

**解释**: 每个线程使用 22 个寄存器，导致每个 SM 最多只能驻留 5 个 block，限制了 occupancy。

### 4. 三种数据类型性能接近

| 数据类型 | Duration (us) | Memory % | Compute % | Occupancy % |
|----------|---------------|----------|-----------|-------------|
| bfloat16 | 50.08 | 24.81 | 36.79 | 68.74 |
| float16 | 50.62 | 24.53 | 36.88 | 68.37 |
| float32 | 50.30 | 24.70 | 36.85 | 68.74 |

**结论**: 数据类型对性能影响很小，因为该操作受内存带宽限制而非计算限制。

## 优化建议

### 优先级 1: 减少寄存器使用
- 当前: 22 regs/thread → Block Limit Registers = 5
- 目标: 降低到 16 regs/thread 以下
- 预计提升: Occupancy 从 68% 提升到 80%+

### 优先级 2: 增加计算密度
- 当前操作: 简单的 concat + scatter write
- 可考虑: 融合更多计算 (如 scale/activation) 到同一 kernel
- 预计提升: Compute Throughput 从 36% 提升到 50%+

### 优先级 3: 优化内存访问模式
- 当前: 随机 slot_mapping 导致非连续写入
- 可考虑: 重排数据以增强局部性
- 预计提升: DRAM Throughput 从 24% 提升到 40%+

## 结论

`concat_and_cache_mla_kernel` 是一个典型的**内存带宽受限** kernel：

1. **主要瓶颈**: 内存访问模式不规则 (slot_mapping 随机)
2. **次要瓶颈**: 寄存器压力限制了 occupancy
3. **数据类型影响**: 几乎无影响 (内存密集型操作)
4. **优化空间**: 约 30% 性能提升潜力 (主要来自 occupancy 优化)

对于当前工作负载 (2048 tokens)，kernel 执行时间约 50us，在实际推理中占比很小，优化优先级较低。当 token 数量大幅增加时，内存带宽优化才会产生显著收益。

---

**分析报告生成时间**: 2026-05-23
**原始数据**: `concat_analysis.ncu-rep` (容器内 `/tmp/`)
