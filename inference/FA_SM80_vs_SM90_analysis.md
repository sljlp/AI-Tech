# FlashAttention SM80 (V2) vs SM90 (V3) 实现对比分析

仓库 `hopper/` 目录下针对 NVIDIA SM80（Ampere，A100/A30/A40）与 SM90（Hopper，H100）分别实现了一套前向和反向内核，分别对应 FA2 和 FA3 的设计思路。下面从架构设计、关键文件、以及优缺点几个维度展开。

## 1. 关键文件对照

| 模块 | SM80 (V2) | SM90 (V3) |
|---|---|---|
| Forward kernel | `flash_fwd_kernel_sm80.h` | `flash_fwd_kernel_sm90.h` |
| Forward mainloop | `mainloop_fwd_sm80.hpp` (~50KB) | `mainloop_fwd_sm90_tma_gmma_ws.hpp` (~104KB) |
| Backward kernel | `flash_bwd_kernel_sm80.h` | `flash_bwd_kernel_sm90.h` |
| Backward mainloop | `mainloop_bwd_sm80.hpp` | `mainloop_bwd_sm90_tma_gmma_ws.hpp` |
| Tile size 选择 | `tile_size_fwd_sm8x()` in `tile_size.h` | `tile_size_fwd_sm90()` in `tile_size.h` |

## 2. 架构层面差异

### SM80 (V2 风格) 设计要点
- **同构线程模型**：所有线程都既是 producer 又是 consumer。`NumThreads == size(TiledMma{})`，没有 warp specialization。kernel 主体只有一个 `for` 循环：调度 → mainloop.mma → epilogue.store。
- **MMA**：`SM80_16x8x16_F32F16F16F32_TN` / BF16 变体，HMMA 指令，操作数从寄存器读取。
- **Memory copy**：`SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>`（cp.async）异步加载 K/V 到 smem，靠多 stage 缓冲流水。
- **共享存储简单**：`SharedStorage` 只有一个 union（mainloop / epilogue 重叠）+ scheduler，没有 pipeline barrier。
- **寄存器分配**：固定 `MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1`，编译期决定。
- **Tile 配置**（`tile_size_fwd_sm8x`，FP16）：典型 `kBlockM=128`，`kBlockN` 在 48~128 之间，`kNWarps=4 或 8`，`kStages=1 或 2`。

### SM90 (V3 风格) 设计要点
- **Warp-Specialized Persistent (WS)**：
  - `NumLoadWarpGroups=1`（producer WG 负责加载），`NumMmaWarpGroups=1/2/3`（consumer WG 负责 GEMM + softmax）。
  - `MaxThreadsPerBlock = NumMmaWG*128 + 128`。
  - 通过 `cutlass::arch::warpgroup_reg_dealloc/alloc` 显式重新分配寄存器：`LoadRegisterRequirement` 24~56、`MmaRegisterRequirement` 160~256，让 consumer 拿到几乎全部寄存器。
- **TMA + GMMA**：
  - `Use_TMA_Q`, `Use_TMA_KV`, `Use_TMA_O` 通过 TMA 异步搬运（绕过寄存器），并支持 cluster mode（`ClusterShape`）做 multicast。
  - 矩阵乘改为 `GMMA::ss_op_selector` / `rs_op_selector`（warpgroup-async MMA，wgmma），可直接读取 smem。
- **多套 Pipeline**：`pipeline_k`、`pipeline_v`、`pipeline_vt`（FP8 转置）、`pipeline_k_new`、`pipeline_v_new`（AppendKV），分别用 ClusterTransactionBarrier / ClusterBarrier 协调 producer-consumer。
- **GEMM 之间的 IntraWGOverlap (Ping-Pong)**：`IntraWGOverlap` 标志使 QK GEMM-I 与 PV GEMM-II 在两个 WG 之间错峰执行，掩盖 softmax 开销 —— 这是 FA3 的核心 trick。
- **更精细的特性开关**：`HasQv`（Q×V 路径，用于某些 MLA decode）、`LargeHeadDimV`（headdim_v>256，例如 MLA 512）、`V_colmajor`、`MmaPV_is_RS` 等，许多代码分支只在 SM90 出现。
- **Tile 配置**（`tile_size_fwd_sm90`，FP16）：典型 `kBlockM=128~192`，`kBlockN=80~192`，比 SM80 大得多；FP8 头维度 128 时甚至 `192×160` 或 `128×224`。

## 3. 优缺点对比

### SM80 实现 (V2)
**优点**
1. **可移植性强**：`Has_cp_async = ArchTag::kMinComputeCapability >= 80`，覆盖 Ampere（A100/A30/A40）、Ada（4090/L40/L4，sm86/89），并通过 `tile_size_fwd_sm8x(sm86_or_89, ...)` 给消费级卡单独调参。
2. **代码简单清晰**：单 kernel 函数体短（仅 ~210 行），调度→加载→计算→写回是直线流程，易于阅读、调试和扩展新功能。
3. **共享内存占用少**：没有多套 pipeline barrier、没有 `pipeline_vt` / `pipeline_k_new`，可以塞下更小的 SRAM 卡。
4. **编译产物少**：模板参数维度比 SM90 少（没有 `ClusterShape`、`HasQv`、`MmaPV_is_RS`、`V_colmajor`、`IntraWGOverlap` 等），实例化数量较小。
5. **支持 PagedKV、Split-KV、AppendKV、PackGQA、FP8 转置等大部分功能**，功能矩阵相对完整。

**缺点**
1. **性能上限较低**：用 HMMA + cp.async 的传统流水，没有利用 H100 的 wgmma/TMA/cluster；在 H100 上跑只能达到 V3 一半左右的吞吐。
2. **Producer/Consumer 不分离**：MMA 线程同时承担数据加载，寄存器压力大，尤其在大 head dim 下需要 `kStages=1` 牺牲流水深度（见 `tile_size_fwd_sm8x` headdim≥192 分支）。
3. **Softmax 不能与 GEMM 并行**：缺少 IntraWGOverlap，softmax/exp/scale 完全串行在 MMA 之后。
4. **Tile 偏小**：受寄存器限制，`kBlockM` 一般 128，`kBlockN` 最低降到 32（headdim>192 + sm86_or_89 + append_kv），导致 K-loop 迭代多。
5. **调度器只能用 SM count × MinBlocksPerSM 启动 grid**，不支持 cluster-aware scheduling。

### SM90 实现 (V3)
**优点**
1. **峰值性能极高**：充分利用 Hopper 三大特性：
   - **TMA**：硬件异步 bulk copy，descriptor prefetch，零寄存器占用搬运；
   - **wgmma**：warpgroup 异步 MMA，可直接 smem×smem，输出 register；
   - **TMA multicast (cluster)**：`ClusterShape > 1` 时跨 SM 共享 K/V。
2. **Warp Specialization + 寄存器再分配**：consumer WG 拿到 232~256 寄存器，能容纳更大 tile（FP16 192×192、FP8 192×160）和更深 accumulator，避免寄存器溢出。
3. **Ping-Pong / IntraWG 重叠**：让 GEMM-I（QK）、softmax、GEMM-II（PV）三阶段在不同 WG 间交错执行，把 softmax 的非张量核开销基本隐藏。
4. **细粒度 pipeline**：K、V、Vt（FP8 行主→列主转置）、K_new/V_new（AppendKV）独立 pipeline，互不阻塞；`MainloopPipelineV` 在需要转置时由 producer-consumer 链式生产。
5. **特殊 workload 优化**：
   - `LargeHeadDimV`（MLA decode 的 hdim_v=512）专门走 1×2×1 atom layout；
   - `HasQv` 支持 Q×V 路径；
   - `V_colmajor` 直接消费列主 V，省一次转置；
   - 支持 `PagedKVNonTMA` 在 cluster=1 时 fallback 到 cp.async。
6. **持久化（persistent）调度**：grid = SM count，scheduler 内部循环吃 work-tile，减少 launch 开销，配合 `wait_on_dependent_grids` 可与上游 kernel 重叠。

**缺点**
1. **架构强绑定**：硬性要求 `ArchTag::kMinComputeCapability >= 90`，只能跑在 H100/H200/H20 等 Hopper 卡上；A100/4090 用不了。
2. **代码复杂度高**：
   - `mainloop_fwd_sm90_tma_gmma_ws.hpp` 1700+ 行，有 19 个模板参数；
   - kernel 入口 459 行，需要手工管理 6+ 种 barrier、producer/consumer 分支、寄存器 dealloc/alloc、cluster 同步。
   - 调试和定位 deadlock/race 难度远大于 SM80 版本。
3. **共享内存与 barrier 占用高**：`SharedStorage` 多了 `barrier_Q`, `barrier_Qv`, `barrier_O`, `pipeline_k`, `pipeline_v`, `pipeline_vt`, `pipeline_k_new`, `pipeline_v_new`，对 H100 228KB SMEM 也算紧张，限制了某些 tile 组合（`tile_size.h` 注释里多次写 "hits the limit of smem"）。
4. **编译实例爆炸**：`instantiations/` 下有 451 个 `.cu` 文件——causal/local/softcap/varlen/paged/split/append/packgqa/fp8/v_colmajor/hdim×hdim_v/cluster 多维笛卡尔积，构建慢、wheel 体积大。
5. **某些 corner case 反而更慢**：tile_size 注释中提到 cutlass 3.8 的 `cute::tuple` 改动让 192×128 比 192×192 慢；PagedKV 因为寄存器占用大而被迫禁用 `IntraWGOverlap`（headdim>192 分支）；MmaPV_is_RS 在 FP8/Transpose_V 时是强制约束。
6. **小问题规模收益小**：persistent + cluster + ping-pong 的固定开销（barrier init、TMA descriptor prefetch、warpgroup reg realloc）对短 seqlen 反而是负担，所以 `flash_api.cpp` 里有 heuristic 在小尺寸时 fallback 到 SM80 路径。

## 4. 选择建议总结

| 场景 | 推荐 |
|---|---|
| A100 / A30 / 4090 / L40 | 必选 SM80（V2），SM90 编不过 |
| H100/H200，长序列 (>=2K)，标准 attention | SM90（V3）显著领先 |
| H100，极短序列或 batch×head 很大但 seqlen 很小 | 视 heuristic 而定，可能 SM80 更省启动开销 |
| MLA decode（hdim_v=512） | 必须 SM90 的 `LargeHeadDimV` 分支 |
| FP8 推理 | SM90 原生支持 wgmma FP8 + V 转置 pipeline，性能优势最大 |
| 想读懂 FlashAttention 算法本身 | 先看 SM80 版本，再读 SM90 |

简而言之：**SM80 路径以"通用 + 可读"取胜，SM90 路径以"针对 Hopper 硬件特性的极限工程化"取胜**。两者共享上层接口（同一份 `flash_api.cpp`、`tile_scheduler.hpp`、`softmax.h`、`mask.h`、`epilogue_*.hpp`），SM90 多出的复杂度几乎全部集中在 mainloop 与 kernel 入口的 producer/consumer/pipeline 调度上。