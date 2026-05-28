# DeepSpeed ZeRO 通信算子深度分析

> 本文详细分析 ZeRO Stage 1/2/3 三个阶段使用的通信算子、各自的优缺点以及优化手段。

---

## 目录

1. [ZeRO Stage 1 通信分析](#1-zero-stage-1-通信分析)
2. [ZeRO Stage 2 通信分析](#2-zero-stage-2-通信分析)
3. [ZeRO Stage 3 通信分析](#3-zero-stage-3-通信分析)
4. [三个阶段通信对比总结](#4-三个阶段通信对比总结)
5. [通用优化技术详解](#5-通用优化技术详解)

---

## 1. ZeRO Stage 1 通信分析

### 1.1 概述

Stage 1 将 **optimizer states**（优化器状态）分片到各数据并行 rank 上。梯度仍是完整 AllReduce，参数也是完整存储。通信集中在梯度归约这一步。

**核心文件**: `deepspeed/runtime/zero/stage_1_and_2.py` — `DeepSpeedZeroOptimizer` 类

### 1.2 使用的通信算子

#### 1.2.1 `dist.all_reduce` —— 梯度归约

**代码位置**: `stage_1_and_2.py` L1141-1168, `gradient_reduction_w_predivide()`

```
dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
```

**作用**: 对所有数据并行 rank 的梯度求平均，结果每个 rank 都持有完整梯度。

**流程**:
1. 若有 `postscale_gradients`，先乘 `1/gradient_predivide_factor`（预除）
2. 执行 `all_reduce`（SUM 模式）
3. 再乘 `gradient_predivide_factor / dp_world_size`（后乘），等价于平均

**优点**:
- 实现简单、代码路径清晰
- 通信量对称，各 rank 负载均衡
- PyTorch/NCCL 对 `all_reduce` 有深度优化（Ring AllReduce，NVLink/NVSwitch 感知）
- 适合小规模集群（节点数少时效率高）

**缺点**:
- 每个 rank 接收完整的梯度张量，内存占用 O(N)（N 为参数量）
- 通信量 = 2 × (N-1)/N × 参数量（Ring AllReduce 理论值），无法随数据并行度扩展
- 当 `dp_world_size` 增大时，通信成为瓶颈

#### 1.2.2 `dist.all_gather` / `dist.all_gather_into_tensor` —— 参数同步

**代码位置**: `deepspeed/runtime/utils.py` L1005-1051, `all_gather_dp_groups()`

```python
# 新 PyTorch 版本走 all_gather_into_tensor
dist.all_gather_into_tensor(group_flat, partitioned_params[partition_id], dp_process_group)

# 旧版本走分片 all_gather（分桶避免 OOM）
dist.all_gather(shard_list, shard_list[partition_id], dp_process_group)
```

**作用**: optimizer step 后，将更新后的参数分片从各 rank 收集，使每个 rank 得到完整参数。

**优点**:
- `all_gather_into_tensor` 单次调用完成，通信效率高
- 分桶策略避免了大张量 OOM

**缺点**:
- Stage 1 中 optimizer states 已分片节省了内存，但参数仍需完整 AllGather（通信量 O(N)）

### 1.3 关键优化点

| 优化 | 代码位置 | 说明 |
|------|----------|------|
| `postscale_gradients` + `gradient_predivide_factor` | L1152-1160 | 通过预除/后乘组合避免 FP16 溢出，保持精度 |
| `communication_data_type` | L1149 | 支持以 FP16 通信、FP32 计算，节省带宽 |
| `allgather_bucket_size` | 配置项 | 控制 AllGather 分片大小，平衡内存和通信效率 |

---

## 2. ZeRO Stage 2 通信分析

### 2.1 概述

Stage 2 将 **optimizer states + gradients** 都分片。每个 rank 只保存自己分到的梯度分片，梯度归约需要将完整的梯度张量"分散"到各 rank，而非完整的 AllReduce。

**核心文件**: `deepspeed/runtime/zero/stage_1_and_2.py` — `DeepSpeedZeroOptimizer` 类

### 2.2 使用的通信算子

Stage 2 的梯度归约有两个路径，由 `reduce_scatter` 配置控制：

#### 2.2.1 路径一：标准 `dist.all_reduce`（`reduce_scatter=False`）

与 Stage 1 完全相同的 `gradient_reduction_w_predivide()`，完整 AllReduce 梯度后每个 rank 都持有完整梯度。

#### 2.2.2 路径二：基于 Bucket 的 Reduce-Scatter（`reduce_scatter=True`）

**入口**: `average_tensor()` L1227-1327

根据 `use_multi_rank_bucket_allreduce` 又分成两个子路径：

##### 子路径 2a：`allreduce_and_scatter()` —— 多 Rank 桶 AllReduce

**代码位置**: L1191-1225

```python
# 按 rank 分桶，每组 bucket 做 allreduce 后复制给目标 rank
self.allreduce_and_copy_with_multiple_ranks(small_bucket, ..., bucket_ranks=small_bucket_ranks)
```

**流程**:
1. 将梯度张量按 `partition_id`（dst rank）分片
2. 属于同一 dst rank 的片合并为一个 bucket
3. 对每个 bucket 执行 `allreduce`，但只让目标 rank 拷贝结果

**本质**: 手写的 `reduce_scatter` 实现（在 PyTorch 2.0 之前 `reduce_scatter` 支持不完善时的兼容方案）

##### 子路径 2b：`allreduce_no_retain()` —— 单 Rank 归约

**代码位置**: L1947-1957 (stage3.py 中也有)

```python
def allreduce_no_retain(self, bucket, numel_per_bucket, rank=None):
    # 分小桶依次执行 allreduce_and_copy，只保留目标 rank 的结果
    for tensor in bucket:
        self.allreduce_and_copy(small_bucket, rank=rank)
```

**流程**: 每个 bucket 执行 allreduce，但只将最终结果复制到目标 rank 的梯度缓冲区。

##### 两种子路径的对比

| 特性 | 2a: multi-rank bucket | 2b: single-rank reduce |
|------|----------------------|------------------------|
| 通信次数 | 每个 bucket 一次 allreduce | 每个 bucket 一次 allreduce |
| 逻辑复杂度 | 较复杂（需 bucket_ranks 映射） | 较简单 |
| 适用场景 | 多个 partition_id 分布在不同 rank | 所有梯度归约到单一 rank |
| 内存 | 各 bucket 独立释放，内存更高效 | 依赖外部分桶 |

#### 2.2.3 `dist.all_gather` / `dist.all_gather_into_tensor` —— 参数同步

**代码位置**: `deepspeed/runtime/utils.py` L1005-1051

与 Stage 1 完全相同。Optimizer step 后用 AllGather 将更新后的完整参数分发给所有 rank。

### 2.3 核心通信模式对比

```
路径一 (reduce_scatter=False):
  all_reduce → 每 rank 持有完整梯度 → 取 own partition → 更新 optimizer

路径二 (reduce_scatter=True):
  (allreduce + per-rank copy) × n_buckets → 每 rank 只持 own partition → 更新 optimizer
```

### 2.4 关键优化点

| 优化 | 说明 |
|------|------|
| `reduce_scatter` | 避免全量 AllReduce，将通信量从 O(N) 降至 O(N/P) |
| `use_multi_rank_bucket_allreduce` | 合并不同 dst rank 的 bucket，增加单次通信的 tensor 大小，提升带宽利用率 |
| `reduce_bucket_size` | 控制 bucket 阈值，平衡通信延迟（大桶）和内存碎片（小桶） |
| `overlap_comm` | 在 reduction stream 上异步执行通信，与 backward 计算 overlap |

---

## 3. ZeRO Stage 3 通信分析

### 3.1 概述

Stage 3 将 **optimizer states + gradients + parameters** 全部分片。每个 rank 只保存 1/P 的完整训练状态。在前向/反向传播中，需要实时 AllGather 参数；在反向传播中，梯度通过 Reduce-Scatter 归约。

**核心文件**: `deepspeed/runtime/zero/stage3.py` — `DeepSpeedZeroOptimizer_Stage3` 类

### 3.2 使用的通信算子

#### 3.2.1 梯度归约（Reduce-Scatter 路径）

##### 3.2.1a `reduce_scatter_coalesced()` —— 默认路径

**代码位置**: `deepspeed/runtime/comm/coalesced_collectives.py` L156-218

```python
def reduce_scatter_coalesced(tensors, group):
    # 1. 每个 tensor 等分为 world_sz 个 chunk
    # 2. 交错排列各 tensor 的 chunk（tensor0-rank0, tensor1-rank0, ...）
    # 3. 拼成一个大 buffer → pre-divide (÷ world_sz)
    # 4. 单次 reduce_scatter 调用
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                             tensor_partition_buffer_for_each_rank[this_rank], group)
    # 5. 从输出的 buffer 中解交错，恢复各 tensor 的 partition
```

**设计要点**:
- **交错排列（Interleaving）**: 将多个 tensor 的分片在 buffer 中交错，用单次 `reduce_scatter` 代替多次
- **Padding 处理**: 当 tensor 不能整除 world_sz 时自动 padding，保证所有 rank 的输入大小一致
- **Pre-divide**: 在通信前除以 `world_sz`，避免 FP16 溢出（与 Stage 1/2 的 pre-divide 思路一致）

**触发条件**: `__avg_scatter_grads()` 中，当 `all2all_process_group is None` 或 `num_nodes == 1`（单节点）时使用。

##### 3.2.1b `all_to_all_quant_reduce()` —— 量化 All-to-All 路径

**代码位置**: `deepspeed/runtime/comm/coalesced_collectives.py` L29-76

```python
def all_to_all_quant_reduce(tensors, groups):
    # 两阶段 All-to-All + 量化通信
    # 阶段1 (intra-node): INT4 量化 → all_to_all_single (local group)
    # 阶段2 (inter-node): 量化归约 → all_to_all_single (global group)
    # 最终: dequantize + chunk + mean
```

**流程**:
1. **INT4 量化**: 使用 DeepSpeed 的 `QuantizerBuilder` 将梯度量化为 INT4
2. **Intra-node All-to-All**: 在节点内 group 上交换量化后的梯度
3. **Quantized Reduction**: 合并节点内的梯度
4. **Inter-node All-to-All**: 在跨节点 group 上交换归约后的量化梯度
5. **反量化**: 恢复为 FP16/FP32，取平均

**触发条件**: 多节点训练且配置了 `all2all_process_group`。

##### 3.2.1c `all_to_all_loco_quant_reduce()` —— LoCo 量化 All-to-All

**代码位置**: `deepspeed/runtime/comm/coalesced_collectives.py` L79-153

与 `all_to_all_quant_reduce` 类似，但加入了 **局部误差补偿（Local Error Compensation）**:

- 维护 `intra_ef_buf` / `inter_ef_buf` 误差缓冲区
- 每次量化时将上次的量化误差反馈到当前梯度（`err_beta` 控制反馈强度）
- 周期性重置误差缓冲区（`reset_T`）

**触发条件**: 多节点训练 + `zeropp_loco_param is not None`。

##### 量化路径 vs 默认路径对比

| 特性 | `reduce_scatter_coalesced` | `all_to_all_quant_reduce` |
|------|---------------------------|---------------------------|
| 通信量 | 全精度（FP16） | INT4（压缩 4 倍） |
| 计算开销 | 几乎无 | 量化/反量化 + 多次 All-to-All |
| 精度 | 无损 | 有损（INT4 量化误差） |
| 跨节点效率 | 受限于跨节点带宽 | 大幅降低跨节点通信量 |
| 适用场景 | 单节点 / 跨节点带宽充足 | 跨节点带宽瓶颈 |
| Fallback 情况 | — | tensor dim=1 或不能整除 2*world_sz 时退化为 `reduce_scatter_coalesced` |

##### 3.2.1d `dist.all_reduce` —— 连续梯度路径

**代码位置**: L1567-1607, `__avg_scatter_contiguous_grads()`

```python
dist.all_reduce(buffer_to_reduce, group=self.dp_process_group)
# all_reduce 后，各 rank 从 buffer 中取自己的 partition
```

**作用**: 当启用了 `contiguous_gradients` 时使用。梯度在连续 buffer 中排列好，执行一次完整 AllReduce，然后在本地做 scatter（取对应 partition）。

**优点**: 单次 AllReduce 通信高效，连续内存访问友好。
**缺点**: 通信量仍是 O(N)，不随数据并行度减小。

> 这是在 `__avg_scatter_contiguous_grads` 中全部 `ipg_buckets` 完成后执行的方式，与 `__avg_scatter_grads` 是互补的代码路径。

#### 3.2.2 参数 AllGather（前向/反向时获取参数）

##### 3.2.2a `dist.all_gather_into_tensor()` —— 连续缓冲区 AllGather

**代码位置**: L1764-1812, `_partitioned_buffers_all_gather()`

```python
coalesced_buffer = torch.cat(buffers_to_allgather)
partition.data.copy_(coalesced_buffer.data)
dist.all_gather_into_tensor(reduce_buffer, partition, group=self.dp_process_group)
# 后续还需 rearrange 以恢复原始参数布局
```

**作用**: 将多个参数的 partition 拼接后，单次 AllGather 获取完整参数。

**额外步骤**: 由于各 rank 收集到的参数是按 rank 顺序存储的（rank0-param0, rank1-param0, ...），需要 `rearrange` 还原为连续参数顺序。

##### 3.2.2b `dist.all_gather_into_tensor()` —— 逐参数 AllGather

**代码位置**: L1875-1879, `_all_gather()` 中的单参数路径

```python
if self.use_all_gather_into_tensor:
    handle = dist.all_gather_into_tensor(flat_tensor,
                                         param.ds_tensor.to(device),
                                         group=self.get_partition_dp_group(param),
                                         async_op=async_op)
```

**作用**: 逐个参数 AllGather，结合 `async_op=True` 实现异步通信。

##### 3.2.2c `dist.all_gather()` —— 兼容路径

当 PyTorch 版本不支持 `all_gather_into_tensor` 时（旧版），使用传统的 `all_gather` API：

```python
# _all_gather_sequential: 逐个参数
handles = []
for param in params:
    handle = _dist_allgather_fn(param_ds_tensor, param_buffer, ds_process_group)
    handles.append(AllGatherHandle(handle, param))

# 或 coalesced 路径
_all_gather_coalesced: torch.cat 后逐 dtype 分组 allgather
```

##### 3.2.2d `dist.all_reduce()` —— 参数获取的 AllReduce 替代方案

**代码位置**: L1348-1367 (`use_all_reduce_for_fetch_params = True`)

```python
# 用 all_reduce 替代 all_gather 获取参数
flat_tensor = torch.zeros(flat_buffer_size, ...)
# 将本地 partition 放入 flat_tensor 的对应位置
flat_tensor.narrow(0, start, param.ds_tensor.ds_numel).copy_(param.ds_tensor)
handle = dist.all_reduce(flat_tensor, group=ds_process_group, async_op=True)
```

**原理**: 每个 rank 持有参数的一个 partition（非零），其他位置为 0。AllReduce SUM 操作等价于把各 rank 的 partition 拼成完整参数。

**优点**: 对于某些通信拓扑（如层次化 AllReduce），可能比 AllGather 更高效。
**缺点**: 需要额外的 zero 初始化和 copy 操作。

#### 3.2.3 FP32 State AllGather

**代码位置**: L2631-2639, `_fp32_state_allgather()`

```python
dist.all_gather_into_tensor(reduce_buffer, partition, group=self.dp_process_group)
```

**作用**: 在获取 FP32 梯度用于 checkpoint 保存时，将分片的 FP32 梯度收集为完整梯度。

#### 3.2.4 前向/反向 Hook 中的异步 AllGather

**代码位置**: `deepspeed/runtime/zero/parameter_offload.py`

```python
# 前向 Hook: 注册 register_pre_forward_hook(_pre_forward_module_hook)
# 钩子函数调用 param.all_gather() → _all_gather() → all_gather_into_tensor()
# 反向 Hook: register_post_accumulate_grad_hook(_post_accumulate_grad_hook)
# 钩子函数触发 reduce_independent_p_g_buckets_and_remove_grads()
```

Stage 3 的核心效率保障：参数按需 AllGather，用完即释放。

#### 3.2.5 异步 AllGather 流水线

在 `_all_gather_coalesced()` (L1441+) 中，完整的流程是：
1. `_ensure_availability_of_partitioned_params()` — 若 partition 在 NVMe 中则换入
2. 根据 `num_partitions` 选择 no-op（单 rank）或实际 AllGather
3. 根据 `use_all_gather_into_tensor` 选择 `all_gather_into_tensor` 或 `all_gather`
4. 可选启用 `use_all_reduce_for_fetch_params` 替代方案
5. 可选启用 `quantize` 压缩通信

返回 `AllGatherCoalescedHandle`，允许调用方在需要时 `.wait()`。

### 3.3 关键优化点

| 优化 | 说明 |
|------|------|
| `overlap_comm` | 梯度 reduce-scatter 在 reduction_stream 上与 backward 计算 overlap |
| `contiguous_gradients` | 梯度放入连续 buffer，单次 AllReduce 后本地 scatter |
| `all2all_process_group` | 多节点时启用 2 阶段 All-to-All 量化归约 |
| INT4 量化通信 | 通信量压缩 4 倍，适合跨节点带宽瓶颈场景 |
| LoCO 误差补偿 | 通过量化误差反馈提升 INT4 精度 |
| `use_all_reduce_for_fetch_params` | 用 AllReduce 替代 AllGather 获取参数 |
| `allgather_bucket_size` | 控制参数 AllGather 的批大小 |
| 参数按需 AllGather | 前向/反向 Hook 自动管理，避免不必要通信 |
| NVMe Offload 感知 | AllGather 前检查 partition 是否在 NVMe，先换入再通信 |
| ZeRO-Quantized Weights | 量化后的权重 AllGather 通信量更小 |

---

## 4. 三个阶段通信对比总结

### 4.1 通信量对比

| 操作 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| 梯度归约 | O(Ψ) AllReduce | O(Ψ) 或 O(Ψ/P) RS | O(Ψ/P) RS 或 O(Ψ/P) 量化 A2A |
| 参数广播 | O(Ψ) AllGather | O(Ψ) AllGather | O(Ψ) AllGather ⚡ |
| Optimizer State | 无通信 | 无通信 | 无通信 |

> Ψ = 总参数量, P = 数据并行度
> ⚡ Stage 3 参数 AllGather 是按需触发的（前向/反向时），而非全部一次性同步

### 4.2 通信算子汇总

| 算子 | Stage 1 | Stage 2 | Stage 3 | 用途 |
|------|---------|---------|---------|------|
| `all_reduce` | ✅ 梯度归约 | ✅ 梯度归约 | ✅ 连续梯度 + 参数 AllReduce 替代 | 求和/平均 |
| `reduce_scatter` | ❌ | ✅ (模拟) | ✅ (`coalesced`) | 归约 + 分散 |
| `all_gather_into_tensor` | ✅ 参数分发 | ✅ 参数分发 | ✅ 参数 + FP32 状态 | 收集完整数据 |
| `all_gather` | ✅ 旧版兼容 | ✅ 旧版兼容 | ✅ 逐参数 + 旧版兼容 | 收集完整数据 |
| `all_to_all_single` | ❌ | ❌ | ✅ 2-stage 量化 | 跨节点数据交换 |
| `barrier` | ⚠️ debug | ⚠️ debug | ⚠️ debug | 同步 |

### 4.3 内存效率对比

| 维度 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| 参数量（每 rank） | Ψ | Ψ | Ψ/P |
| 梯度量（每 rank） | Ψ | Ψ/P | Ψ/P |
| Optimizer State（每 rank） | Ψ/P | Ψ/P | Ψ/P |
| 梯度通信方式 | AllReduce | AllReduce / Reduce-Scatter | Reduce-Scatter / Quantized A2A |

---

## 5. 通用优化技术详解

### 5.1 Post-Scale 梯度矫正（`postscale_gradients`）

**作用**: 防止 FP16 梯度在 AllReduce 时溢出。

**原理**:
```
normalize_factor = gradient_predivide_factor
tensor = tensor / normalize_factor
all_reduce(tensor, sum)
tensor = tensor * normalize_factor / world_size
```

当 `normalize_factor = world_size` 时，取消乘除操作（因为 `world_size/world_size = 1`）。

**适用**: Stage 1/2/3 的 AllReduce 路径。

### 5.2 Bucket 化通信

**作用**: 将大量小张量合并为大张量通信，提升带宽利用率。

**实现**: `reduce_bucket_size` / `allgather_bucket_size` 配置项。默认 500M elements。

**适用**: 所有阶段的 AllReduce/AllGather。

### 5.3 Overlap 通信与计算

**代码位置**: `stage_1_and_2.py` L1228-1234, `stage3.py` 的 `reduce_independent_p_g_buckets_and_remove_grads()`

```python
if self.overlap_comm:
    stream = self.reduction_stream
    stream.wait_stream(get_accelerator().current_stream())
    # 在 reduction_stream 上执行通信
    with get_accelerator().stream(stream):
        self.gradient_reduction_w_predivide(tensor, ...)
```

**Stage 3 的 Overlap**: 在反向传播的梯度 hook 中，每个参数的梯度就绪后立即触发 reduce-scatter，与后续参数的 backward 计算 overlap。

### 5.4 量化通信（Stage 3 专用）

**INT4 量化归约**: 通信量减少 4 倍，适合跨节点。
**LoCo 误差补偿**: 用历史误差反馈补偿量化精度损失。

**权衡**: 通信节省 vs 量化计算开销 + 精度损失。

### 5.5 2-Stage All-to-All（Stage 3 专用）

**分组策略**:
- `local_{intra_idx}`: 节点内 group（同一节点内各 GPU）
- `global_{inter_idx}`: 跨节点 group（各节点的同一 GPU 索引）

**优点**: 节点内通信利用 NVLink（高带宽），跨节点通信减少传输量（INT4 压缩 × 节点内归约）。

### 5.6 Contiguous Gradients（Stage 3 专用）

**代码位置**: L1567-1607

在 backward 过程中，所有参数的梯度按 `ipg_buckets` 排布到连续 buffer 中。`all_reduce` 在连续 buffer 上执行，比分散的梯度张量通信更高效（更好的缓存局部性和更大的消息粒度）。

### 5.7 连续 Bucket 机制（Stage 1/2 的 Continuous Bucket）

通过 `self.allreduce_and_scatter()` 的 `numel_per_bucket` 参数，控制每个 bucket 包含的 tensor 数量不超过阈值。这是为了：
1. **减少峰值内存**: 分桶通信，不必等待所有梯度就绪
2. **提高 overlap 效率**: 小桶可以更早开始通信

---

## 总结

| 阶段 | 核心通信模式 | 通信量 | 主要优化手段 |
|------|------------|--------|-------------|
| **Stage 1** | AllReduce Gradient → AllGather Params | O(Ψ) | Post-scale, Bucket, Communication Data Type |
| **Stage 2** | AllReduce/RS Gradient → AllGather Params | O(Ψ) ~ O(Ψ/P) | Reduce-Scatter, Multi-Rank Bucket, Overlap |
| **Stage 3** | RS/Quant-A2A Gradient → On-Demand AG Params | O(Ψ/P) | INT4 Quant, LoCo, 2-Stage A2A, Contiguous Buffer, Async Pipeline |

> **核心趋势**: 从 Stage 1 → Stage 3，通信量逐渐降低（内存节省也越大），但通信模式越来越复杂、对通信拓扑的依赖越来越强。Stage 3 需要精确的 AllGather/Release 时机管理（通过 Hook），而 Stage 1 只需简单的 AllReduce。量化通信（A2A）仅在跨节点时启用，属于"锦上添花"而非必需品。
