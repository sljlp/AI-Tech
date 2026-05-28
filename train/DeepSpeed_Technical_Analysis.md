# DeepSpeed 源码深度分析：大规模分布式训练技术全景

> 基于 DeepSpeed master 分支 (Commit: 最新) 源码分析
> 分析日期：2026-05-27

---

## 目录

1. [总体架构概述](#1-总体架构概述)
2. [ZeRO 优化器（核心创新）](#2-zero-优化器核心创新)
   - 2.1 ZeRO Stage 1：优化器状态分区
   - 2.2 ZeRO Stage 2：梯度分区
   - 2.3 ZeRO Stage 3：参数分区
   - 2.4 ZeRO-Offload：CPU/NVMe 卸载
   - 2.5 ZeRO-Infinity：无限显存抽象
3. [混合精度训练](#3-混合精度训练)
   - 3.1 FP16 训练
   - 3.2 BF16 训练
   - 3.3 Loss Scaling 机制
4. [流水线并行（Pipeline Parallelism）](#4-流水线并行)
   - 4.1 1F1B 调度策略
   - 4.2 自动层划分
   - 4.3 P2P 通信
5. [MoE（混合专家模型）](#5-moe-混合专家模型)
   - 5.1 Top-K 门控机制
   - 5.2 Token 路由与负载均衡
   - 5.3 专家并行
6. [通信模块](#6-通信模块)
   - 6.1 通信后端封装
   - 6.2 自定义集合通信操作
   - 6.3 梯度压缩通信
7. [激活检查点（Activation Checkpointing）](#7-激活检查点)
8. [优化器与学习率调度器](#8-优化器与学习率调度器)
9. [DeepSpeed 推理引擎](#9-deepspeed-推理引擎)
10. [训练引擎核心流程](#10-训练引擎核心流程)

---

## 1. 总体架构概述

DeepSpeed 是微软开源的大规模分布式深度学习训练优化库，核心入口在 `deepspeed/__init__.py`。

### 1.1 核心架构图

```
                 用户代码
                    |
            deepspeed.initialize()
                    |
         +----------+-----------+
         |                      |
    DeepSpeedEngine      PipelineEngine
         |                      |
    +----+----+           +----+----+
    |         |           |         |
 ZeROOpt   BF16Opt    Pipeline    MoE
    |         |        Scheduler   |
 Stage1/2/3 混合精度   1F1B调度  TopKGate
    |         |                    |
 Offload   LossScale            Experts
 (CPU/NVMe) 管理
```

### 1.2 核心初始化流程

`deepspeed.initialize()`（`deepspeed/__init__.py:80`）是整个库的入口，核心逻辑：

```python
def initialize(args=None, model=None, optimizer=None, ...):
    # 1. 初始化分布式环境
    dist.init_distributed(...)

    # 2. 加载配置
    config_class = DeepSpeedConfig(config, mpu)

    # 3. 根据模型类型创建引擎
    if not isinstance(model, PipelineModule):
        engine = DeepSpeedEngine(...)    # 标准引擎
    else:
        engine = PipelineEngine(...)     # 流水线引擎

    return (engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler)
```

### 1.3 关键模块组织

| 模块路径 | 功能 | 核心文件 |
|---------|------|---------|
| `deepspeed/runtime/zero/` | ZeRO 优化器（核心） | `stage_1_and_2.py`, `stage3.py`, `parameter_offload.py` |
| `deepspeed/runtime/engine.py` | 训练引擎 | 前向/反向/权重更新全流程 |
| `deepspeed/runtime/fp16/` | FP16 混合精度 | `fused_optimizer.py`, `loss_scaler.py` |
| `deepspeed/runtime/bf16_optimizer.py` | BF16 优化器 | |
| `deepspeed/runtime/pipe/` | 流水线并行 | `engine.py`, `schedule.py`, `module.py` |
| `deepspeed/moe/` | 混合专家模型 | `sharded_moe.py`, `layer.py`, `experts.py` |
| `deepspeed/comm/` | 通信封装 | `comm.py`, `torch.py` |
| `deepspeed/runtime/activation_checkpointing/` | 激活检查点 | `checkpointing.py` |
| `deepspeed/runtime/lr_schedules.py` | 学习率调度 | |
| `deepspeed/ops/` | 自定义 CUDA 算子 | `adam.py`, `transformer.py` |

---

## 2. ZeRO 优化器（核心创新）

ZeRO（Zero Redundancy Optimizer）是 DeepSpeed 的核心创新，论文发表于 SC 2020。其核心思想是**消除数据并行中的内存冗余**。

### 2.1 ZeRO 内存模型

传统数据并行中，每个 GPU 都持有完整的模型状态：
- **Optimizer States**：优化器状态（如 Adam 的 momentum + variance）
- **Gradients**: 梯度
- **Parameters**: 模型参数

对于 Adam 优化器，每个参数需要 16 字节（FP32 参数）+ 8 字节（FP16 参数）+ 16 字节（momentum + variance）= 40 字节/参数。

ZeRO 通过分阶段消除冗余：

| 阶段 | 消除内容 | 内存节省 | 通信量 |
|------|---------|---------|--------|
| Stage 1 | Optimizer States (OS) | 4x | 1x |
| Stage 2 | OS + Gradients (G) | 8x | 1x |
| Stage 3 | OS + G + Parameters (P) | Nx (线性) | 1.5x |

### 2.2 ZeRO Stage 1：优化器状态分区

**文件**: `deepspeed/runtime/zero/stage_1_and_2.py`

Stage 1 将优化器状态分片到各个数据并行进程中，每个进程只负责更新自己分片的部分。

核心数据结构（`DeepSpeedZeroOptimizer.__init__`，行 138-200）：

```python
class DeepSpeedZeroOptimizer(ZeROOptimizer):
    def __init__(self, init_optimizer, ...):
        # FP16 参数组（每个参数组一个列表）
        self.bit16_groups = []           # 原始 FP16 参数
        # 分片后的 FP16 参数（每个DP rank持有部分）
        self.parallel_partitioned_bit16_groups = []
        # FP32 参数分片（每个DP rank持有部分）
        self.single_partition_of_fp32_groups = []
        # 分片大小
        self.partition_size = []
        # 梯度累积相关
        self.averaged_gradients = {}     # 累积梯度
```

核心步骤（`step()` 方法，行 2114）：

```python
def step(self, closure=None):
    # 1. 检测溢出，更新 loss scale
    self.check_overflow(...)
    self._update_scale(self.overflow)

    # 2. 计算全局梯度范数（用于梯度裁剪）
    scaled_global_grad_norm = self.scaled_global_norm()

    # 3. 每个参数组独立更新
    for i, group in enumerate(self.bit16_groups):
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])

        # 创建当前 rank 负责的梯度分片
        single_grad_partition = self.flatten(self.averaged_gradients[i])

        # 梯度裁剪
        self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

        # 执行优化器步骤（只更新当前分片）
        self._optimizer_step(i)

        # 将更新后的 FP32 参数拷贝回 FP16
        bit16_partitions[partition_id].data.copy_(fp32_partition.data)
```

**关键机制**：
- 每个 rank 只更新 `partition_size[i]` 个参数
- 优化后通过 `broadcast` 或 `allgather` 同步更新后的参数
- 配置 `allgather_partitions=True` 使用 allgather，否则使用广播

### 2.3 ZeRO Stage 2：梯度分区

Stage 2 在 Stage 1 的基础上进一步将梯度分片。

核心实现与 Stage 1 在同一文件 `stage_1_and_2.py`，通过 `partition_grads=True` 参数控制。

```python
# 初始化时设置的梯度相关配置
self.partition_grads = partition_grads  # True for stage 2
self.contiguous_gradients = contiguous_gradients  # 连续梯度缓冲

# 梯度累积使用 OrderedDict 按参数组存储
self.averaged_gradients = OrderedDict()
self.all_grad_tensors = OrderedDict()
```

**梯度规约优化**：

```python
# 在 backward hook 中（行 ~1700）
def _reduce_gradients(self, ...):
    if self.partition_grads:
        # 使用 reduce_scatter 替代 allreduce
        # 每个 rank 只获得梯度的一部分
        dist.reduce_scatter(...)
    else:
        # Stage 1: 使用 allreduce
        dist.allreduce(...)
```

**关键优化 - `reduce_scatter`**：
- 传统 allreduce：所有 rank 获得完整梯度（N 倍冗余）
- Stage 2 reduce_scatter：每个 rank 只获得其负责参数的那部分梯度（1/N 内存）
- 通信量相同，但内存减少 N 倍

### 2.4 ZeRO Stage 3：参数分区

**文件**: `deepspeed/runtime/zero/stage3.py`

Stage 3 是最激进的优化，将模型参数也进行分区，每个 rank 只持有 1/N 的参数。

```python
class DeepSpeedZeroOptimizer_Stage3(ZeROOptimizer):
    def __init__(self, module, init_optimizer, ...):
        # 参数卸载管理器（核心）
        self.parameter_offload = DeepSpeedZeRoOffload(...)

        # FP16 参数（完整但只有当前 rank 的副本有效）
        self.fp16_groups = []

        # FP32 参数分片（用于优化器更新）
        self.fp32_partitioned_groups_flat = []

        # 子组大小（用于处理超大模型）
        self.sub_group_size = sub_group_size
```

**关键机制 - 参数获取与释放**：

`parameter_offload.py` 中的 `DeepSpeedZeRoOffload` 类（`deepspeed/runtime/zero/parameter_offload.py:112`）：

```python
class DeepSpeedZeRoOffload(object):
    def __init__(self, module, ...):
        self.module = module
        self.forward_hooks = []   # 前向钩子：前向时获取参数
        self.backward_hooks = []  # 反向钩子：反向时获取参数

        # 注册钩子到每个子模块
        self._register_deepspeed_module(module)
```

**核心流程（`stage3.py` step 方法，行 2415）**：

```python
def step(self, closure=None):
    self._pre_step()
    # 将所有参数再次分区（释放不需要的）
    self._partition_all_parameters()

    # 溢出检测与 loss scale 更新
    if self._overflow_check_and_loss_scale_update():
        return

    # 计算梯度范数
    norm_groups = self._get_norm_groups()
    scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))

    # 按子组逐块更新
    for sub_group_id, group in enumerate(self.fp16_groups):
        # 准备子组：获取参数、优化器状态、梯度
        self._prepare_sub_group(sub_group_id, timer_names)
        # 梯度裁剪
        self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)
        # 优化器步骤
        self._optimizer_step(sub_group_id)
        # 重新分区或换出参数
        self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
        # 释放子组内存
        self._release_sub_group(sub_group_id, timer_names)
```

**Stage 3 的预取机制**：

```python
# partition_parameters.py 中的预取
class PartitionedParameterCoordinator:
    def prefetch_parameters(self, modules):
        """基于模块执行顺序预测并预取参数"""
        for module in self._get_modules_to_prefetch():
            for param in module.parameters():
                if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    param.all_gather()  # 提前 allgather
```

参数通过 `prefetch_bucket_size` 和 `max_reuse_distance` 控制预取行为：
- `prefetch_bucket_size` (5e7)：每次预取的最大参数元素数
- `max_reuse_distance` (1e9)：参数在多久内会被重用，决定是否释放
- `max_live_parameters` (1e9)：GPU 上最多保留的参数数量

### 2.5 ZeRO-Offload：CPU/NVMe 卸载

**文件**: `deepspeed/runtime/zero/offload_config.py`, `offload_states.py`

支持将优化器状态和参数卸载到 CPU 或 NVMe：

```python
# 配置示例
zero_config = {
    "offload_param": {
        "device": "cpu",        # 或 "nvme"
        "pin_memory": True,
        "nvme_path": "/nvme"
    },
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True,
        "ratio": 0.5            # 部分卸载比例（仅 Stage 3）
    }
}
```

**NVMe 卸载** (`deepspeed/runtime/swap_tensor/`)：
- `PartitionedParamSwapper`：参数交换
- `PartitionedOptimizerSwapper` / `PipelinedOptimizerSwapper`：优化器状态交换
- 使用 AIO（异步 I/O）进行高效的 NVMe 读写

---

## 3. 混合精度训练

### 3.1 FP16 训练

**文件**: `deepspeed/runtime/fp16/fused_optimizer.py`, `unfused_optimizer.py`

```python
class FP16_Optimizer:
    def __init__(self, init_optimizer, ...):
        # 半精度参数（用于前向/反向）
        self.bit16_groups = []
        # FP32 主权重（用于优化器更新）
        self.fp32_groups_flat_partition = []
```

**流程**：
1. 前向/反向：使用 FP16 参数（省显存、快）
2. 优化器更新：将梯度拷贝到 FP32 主权重，执行 FP32 更新
3. 将更新后的 FP32 权重转回 FP16

### 3.2 BF16 优化器

**文件**: `deepspeed/runtime/bf16_optimizer.py`

```python
class BF16_Optimizer(ZeROOptimizer):
    def __init__(self, init_optimizer, ...):
        # BF16 不需要 loss scaling（动态范围大）
        self.custom_loss_scaler = False
        # 梯度累积精度
        self.grad_acc_dtype = grad_acc_dtype  # fp32 或 bf16
```

BF16 相比于 FP16 的优势：
- 更大的指数范围（无需 loss scaling）
- 但精度更低（7位 vs 11位尾数）

### 3.3 Loss Scaling 机制

**文件**: `deepspeed/runtime/fp16/loss_scaler.py`

```python
@dataclass
class LossScaleConfig:
    use_grad_scaling: bool
    dynamic_loss_scale: bool    # 动态或静态
    cur_iter: int
    cur_scale: float            # 当前 scale 值
    last_overflow_iter: Optional[int]
```

**动态 Loss Scaling**：
1. 初始 scale = 2^32（Fused profile）或 2^16（Unfused）
2. 每 N 步（默认 1000）没有溢出，scale *= 2
3. 检测到溢出时，scale /= 2，跳过当前步骤
4. 梯度除以 scale 后再做优化器更新

```python
# stage_1_and_2.py 中的溢出检测
def check_overflow(self, partition_gradients=True):
    # 检查分片梯度中是否有 inf/nan
    self.overflow = self.has_overflow(...)
    if self.overflow:
        self.loss_scale /= 2    # scale 减半
```

---

## 4. 流水线并行

**目录**: `deepspeed/runtime/pipe/`

流水线并行将模型的不同层放置在不同 GPU 上，适合层数非常深的模型。

### 4.1 核心文件

| 文件 | 功能 |
|------|------|
| `engine.py` | 流水线引擎（PipelineEngine），继承 DeepSpeedEngine |
| `module.py` | PipelineModule：自动将模型划分到各 stage |
| `schedule.py` | 训练调度策略（1F1B、交错式等） |
| `p2p.py` | 点对点通信（send/recv） |
| `topology.py` | 流水线拓扑定义 |

### 4.2 1F1B 调度

**文件**: `deepspeed/runtime/pipe/schedule.py`

1F1B（One-Forward-One-Backward）是最常用的流水线调度策略：

```
时间 →
GPU3:  F1 F2 F3 B3 B2 B1
GPU2:  F1 F2 F3 B3 B2 B1
GPU1:  F1 F2 F3 B3 B2 B1
GPU0:  F1 F2 F3 B3 B2 B1
       [  预热  ][  稳定运行  ][   排空   ]
```

**优点**：比简单的前向全部完成后才反向的方式，减少了 GPU 空闲时间（bubble）。

PipelineEngine 初始化（`engine.py`）：

```python
class PipelineEngine(DeepSpeedEngine):
    def __init__(self, ...):
        super().__init__(...)
        # 流水线特定配置
        self.pipeline_parallelism = True
        self.micro_batches = self.gradient_accumulation_steps()
        # 创建通信组
        self._create_p2p_groups()
```

### 4.3 通信

**文件**: `deepspeed/runtime/pipe/p2p.py`

```python
# 点对点通信原语
def send_forward(output, peer):
    dist.send(output, peer)

def recv_forward(peer):
    output = torch.empty(...)
    dist.recv(output, peer)
    return output
```

`PipelineModule` (`module.py`) 自动划分：

```python
class PipelineModule(torch.nn.Sequential):
    def __init__(self, layers, topology, ...):
        # 定义层的划分策略
        self.layers = layers          # 所有层
        self._topology = topology      # 拓扑
        # 自动分配层到各 stage
        self._layer_distribution = self._count_layer_ids()
```

---

## 5. MoE（混合专家模型）

**目录**: `deepspeed/moe/`

### 5.1 整体架构

```
         输入
           |
       TopKGate (门控网络)
        /        \
    Expert 1     Expert 2     ...    Expert N
        \        /
         输出（加权合并）
```

### 5.2 Top-K 门控机制

**文件**: `deepspeed/moe/sharded_moe.py`

```python
class TopKGate(Module):
    def __init__(self, hidden_size, num_experts, k, ...):
        self.wg = nn.Linear(hidden_size, num_experts, bias=False)
        self.k = k  # Top-K 选择（通常 k=1 或 k=2）

    def forward(self, input):
        # 计算每个 token 到每个 expert 的得分
        logits = self.wg(input)  # [seq_len, num_experts]

        if self.noisy_gate_policy == 'Jitter':
            logits = multiplicative_jitter(logits, ...)

        # Top-K 选择
        top_k_logits, top_k_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_k_logits[:, :self.k]
        top_k_indices = top_k_indices[:, :self.k]

        # Softmax 归一化门控值
        gate_scores = F.softmax(top_k_logits, dim=-1)

        return gate_scores, top_k_indices
```

### 5.3 Token 路由

`MOELayer` (`sharded_moe.py:152`) 实现完整的路由逻辑：

```python
class MOELayer(Module):
    def forward(self, hidden_states, used_token=None):
        # 1. 门控计算
        gate_scores, top_k_indices = self.gate(hidden_states)

        # 2. All-to-All 分发：将 token 发送到对应 expert 所在的 rank
        dispatched_tokens = _AllToAll.apply(self.ep_group, hidden_states)

        # 3. Expert 前向计算
        expert_outputs = self.experts(dispatched_tokens)

        # 4. All-to-All 收集：将结果收集回来
        gathered_outputs = _AllToAll.apply(self.ep_group, expert_outputs)

        # 5. 加权合并各 expert 的输出
        output = torch.bmm(combined_outputs.unsqueeze(1), gate_scores.unsqueeze(-1))

        return output
```

**关键设计 - `_AllToAll`** (`sharded_moe.py:97`)：

```python
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        # 反向时再次 all-to-all（因为 all-to-all 是自逆的）
        return (None, _AllToAll.apply(ctx.group, *grad_output))
```

### 5.4 负载均衡损失

为了鼓励 token 均匀分配到各个 expert：

```python
# 辅助损失（load balancing loss）
# gate_scores: [seq_len, k, num_experts] 或 [seq_len, num_experts]
# 通过增加门控分布的熵来鼓励均匀分配
l_aux = ...  # 添加到总损失中
```

### 5.5 用户接口

**文件**: `deepspeed/moe/layer.py`

```python
class MoE(nn.Module):
    def __init__(self, hidden_size, expert, num_experts=1, ep_size=1, k=1,
                 capacity_factor=1.0, ...):
        self.num_local_experts = num_experts // self.ep_size
        experts = Experts(expert, self.num_local_experts, ...)
        self.deepspeed_moe = MOELayer(
            TopKGate(hidden_size, num_experts, k, ...),
            experts, ...
        )
```

---

## 6. 通信模块

**目录**: `deepspeed/comm/`

### 6.1 设计目标

`deepspeed/comm/comm.py` 明确规定：

> deepspeed.comm API 必须与 torch.dist API 保持完全兼容（相同签名），以确保向后/跨框架兼容性。

### 6.2 后端架构

```python
# comm.py
class ProcessGroup:
    def __init__(self, comm_id, ranks=[]):
        self.ranks = ranks
        self.comm_id = comm_id
        self.size = len(ranks)

# 后端选择
cdb = None  # Current deepspeed.comm backend
nccl_backend = None
mpi_backend = None
ccl_backend = None
hccl_backend = None
```

**初始化** (`comm.py`):

```python
def init_distributed(dist_backend=None, distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, ...):
    # 选择后端
    if dist_backend == "nccl":
        cdb = TorchBackend()  # 封装 torch.distributed
    elif dist_backend == "ccl":
        cdb = CCLBackend()    # Intel oneCCL 后端
    ...
```

### 6.3 日志与计时包装

```python
# 对所有通信操作进行计时和日志
def timed_op(func):
    def log_wrapper(*args, **kwargs):
        if comms_logger.enabled:
            # 记录消息大小、时长等
            ...
        return func(*args, **kwargs)
    return log_wrapper
```

### 6.4 梯度压缩通信

DeepSpeed 支持量化的梯度通信（`stage_1_and_2.py` 和相关配置）：

```python
# 配置
zero_quantized_gradients: bool = False  # 量化梯度通信
zero_quantized_weights: bool = False    # 量化权重通信

# 实现使用 All-to-All 量化归约
# deepspeed/runtime/comm/coalesced_collectives.py
all_to_all_quant_reduce(...)
all_to_all_loco_quant_reduce(...)  # LoCo 量化
```

---

## 7. 激活检查点

**文件**: `deepspeed/runtime/activation_checkpointing/checkpointing.py`

### 7.1 核心原理

在前向传播时，不保存中间激活值，而是在反向传播时重新计算。以时间换空间，显著减少显存占用。

### 7.2 API

```python
from deepspeed.runtime.activation_checkpointing import checkpointing

# 启用激活检查点
checkpointing.enable()

# 配置
checkpointing.PARTITION_ACTIVATIONS = True   # 分区激活（模型并行）
checkpointing.CPU_CHECKPOINT = True          # CPU 卸载激活
checkpointing.CONTIGUOUS_CHECKPOINTING = True  # 连续内存检查点（减少碎片）
```

### 7.3 核心函数

```python
# checkpoint 函数包装（checkpointing.py）
def checkpoint(function, *args):
    """在 forward 时不保存中间激活，backward 时重新计算"""
    if deepspeed_checkpointing_enabled:
        return CheckPointFunction.apply(function, *args)
    else:
        return function(*args)
```

`CheckPointFunction` 是 `torch.autograd.Function`：

```python
class CheckPointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        # 不保存中间激活，只保存输入
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 重新运行前向计算得到激活值
        with torch.enable_grad():
            outputs = ctx.run_function(*ctx.saved_tensors)
        # 计算梯度
        torch.autograd.backward(outputs, grad_outputs)
        ...
```

### 7.4 RNG 状态追踪

```python
class CudaRNGStatesTracker:
    """追踪模型并行中的 RNG 状态"""
    def fork(self, name):
        """在检查点重计算时 fork RNG 状态，确保结果一致"""
        orig_state = get_accelerator().get_rng_state()
        _set_cuda_rng_state(self.states_[name])
        yield
        self.states_[name] = get_accelerator().get_rng_state()
        _set_cuda_rng_state(orig_state)
```

---

## 8. 优化器与学习率调度器

### 8.1 自定义优化器

**文件**: `deepspeed/ops/adam/`

```python
class FusedAdam(torch.optim.Adam):
    """融合 Adam 优化器，使用 CUDA 内核一次完成所有更新"""
    ...

class DeepSpeedCPUAdam(Optimizer):
    """CPU Adam 优化器，用于 ZeRO-Offload 的 CPU 优化器卸载"""
    ...
```

### 8.2 学习率调度器

**文件**: `deepspeed/runtime/lr_schedules.py`

支持丰富的学习率调度策略：
- Warmup + 余弦退火
- Warmup + 指数衰减
- Warmup + 线性衰减
- OneCycle
- Warmup + 固定 LR

```python
def add_tuning_arguments(parser):
    parser.add_argument('--lr_scheduler_type', ...)
    parser.add_argument('--warmup_min_lr', default=0, ...)
    parser.add_argument('--warmup_max_lr', ...)
    parser.add_argument('--warmup_num_steps', default=1000, ...)
    ...
```

### 8.3 梯度裁剪

```python
# engine.py
def clip_fp32_gradients(self):
    total_norm = 0.0
    for p in self.module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    # 如果总范数超过阈值，按比例缩放梯度
    clip_coef = self.clip_grad() / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in self.module.parameters():
            p.grad.data.mul_(clip_coef)
```

在 ZeRO 优化器中，梯度裁剪在分片上进行：

```python
# stage_1_and_2.py 行 2199
self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
```

---

## 9. DeepSpeed 推理引擎

**文件**: `deepspeed/inference/engine.py`

### 9.1 初始化

```python
def init_inference(model, config=None, **kwargs):
    engine = InferenceEngine(model, config=ds_inference_config)
    return engine
```

### 9.2 推理优化技术

1. **内核注入（Kernel Injection）**：用高性能 CUDA 内核替换 Transformer 层
2. **张量并行推理**：自动将大模型切分到多 GPU
3. **量化推理**：INT8 量化
4. **权重卸载**：ZeRO-Inference，运行时按需加载参数

---

## 10. 训练引擎核心流程

**文件**: `deepspeed/runtime/engine.py`

### 10.1 DeepSpeedEngine 初始化

```python
class DeepSpeedEngine(Module):
    def __init__(self, args, model, optimizer=None, ...):
        # 1. 配置检查
        self._do_args_sanity_check(args)

        # 2. 配置分布式模型（ZeRO, DDP 等）
        self._configure_distributed_model(model)

        # 3. 配置优化器
        self._configure_optimizer(optimizer, model_parameters)

        # 4. 配置 LR 调度器
        self._configure_lr_scheduler()

        # 5. 配置检查点
        self._configure_checkpointing()

        # 6. 配置编译（optional）
        if is_deepcompile_supported():
            self.register_compile_pass(...)
```

### 10.2 前向传播

用户调用 `engine(micro_batch)`：

```python
def forward(self, *inputs, **kwargs):
    # 记录前向计时
    self._start_timers(self.engine_timers.forward_timers)
    # 前向传播（模型输出）
    outputs = super().forward(*inputs, **kwargs)
    self._stop_timers(self.engine_timers.forward_timers)
    return outputs
```

### 10.3 反向传播

```python
def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
    # 梯度累积缩放
    gas_scaled_loss = loss / self.gradient_accumulation_steps() if scale_wrt_gas else loss

    # ZeRO 优化器处理 loss scaling
    if isinstance(self.optimizer, ZeROOptimizer):
        loss = self.optimizer.scale_if_loss(loss)

    # 执行反向传播
    loss.backward(**backward_kwargs)

    # 后处理（梯度规约等）
    self._backward_epilogue()
```

### 10.4 权重更新

```python
def step(self, lr_kwargs=None):
    if self.is_gradient_accumulation_boundary():
        # 执行实际权重更新
        self._take_model_step(lr_kwargs)
```

`_take_model_step` 最终调用优化器的 `step()`：

```
对于 ZeRO:   DeepSpeedZeroOptimizer.step()
对于 BF16:   BF16_Optimizer.step()
对于 FP16:   FP16_Optimizer.step()
```

### 10.5 ZeRO 优化器配置选择

`engine.py` 中的 `_configure_zero_optimizer` 根据配置选择适当的 ZeRO 实现：

```python
# 根据 ZeRO stage 选择实现
if zero_stage == ZeroStageEnum.weights:
    # Stage 3: 使用 DeepSpeedZeroOptimizer_Stage3
    optimizer = DeepSpeedZeroOptimizer_Stage3(...)
elif zero_stage == ZeroStageEnum.gradients or zero_stage == ZeroStageEnum.optimizer_states:
    # Stage 1/2: 使用 DeepSpeedZeroOptimizer
    optimizer = DeepSpeedZeroOptimizer(...)
```

---

## 总结：技术全景图

| 技术 | 解决的问题 | 关键创新点 | 核心文件 |
|------|-----------|-----------|---------|
| **ZeRO Stage 1** | 优化器状态冗余 | 状态分片，每 rank 只持 1/N | `stage_1_and_2.py` |
| **ZeRO Stage 2** | 梯度冗余 | reduce_scatter 替代 allreduce | `stage_1_and_2.py` |
| **ZeRO Stage 3** | 参数冗余 | 动态参数获取/释放，预取 | `stage3.py`, `partition_parameters.py` |
| **ZeRO-Offload** | GPU 显存不足 | CPU/NVMe 卸载，AIO | `offload_config.py`, `swap_tensor/` |
| **混合精度** | 训练加速/省显存 | FP16/BF16 + 动态 loss scaling | `fp16/`, `bf16_optimizer.py` |
| **流水线并行** | 超深模型 | 1F1B 调度，减少 bubble | `pipe/` |
| **MoE** | 超大规模模型 | Top-K 门控，All-to-All 路由 | `moe/` |
| **激活检查点** | 激活值占用显存 | 重计算代替存储 | `activation_checkpointing/` |
| **通信优化** | 通信瓶颈 | 梯度压缩，重叠通信 | `comm/`, `coalesced_collectives.py` |

### 关键设计哲学

1. **渐进式优化**：ZeRO Stage 0/1/2/3 允许用户按需选择优化级别
2. **显存-通信权衡**：每个 Stage 节省更多显存但通信开销略增
3. **用户透明**：通过 `deepspeed