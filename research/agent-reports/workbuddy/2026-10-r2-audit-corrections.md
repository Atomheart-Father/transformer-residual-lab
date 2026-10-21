# R2 交付 ⑤：旧报告审计纠偏表

**对象**：`docs/INFERENCE_SYSTEM_AUDIT.md`（R1，任务 B 交付）与 `docs/INFERENCE_SYSTEM_AUDIT_SUMMARY.md`
**依据**：Issue #2 的纠偏要求 + 本轮重新核对源码/工件所得
**性质**：只做收窄与勘误，不撤回 R1 中已有一手证据支撑的结论
**日期**：2026-10-21

---

## 0. 使用说明

每条给五栏：

| 栏 | 含义 |
|---|---|
| **R1 原表述** | 报告里的原话或等价转述 |
| **问题类型** | 派生估算 / 口径混淆 / 事实错误 / 过度外推 |
| **收窄后可支持** | 在现有证据下唯一站得住的写法 |
| **不可再支持** | 必须删除或明确标注为未知的说法 |
| **恢复强表述所需测量** | 需要什么数据才能把话讲回去 |

证据分级沿用 R1：E0 源码/规范、E1 本机实测、E2 派生计算、E3 文献转述、E4 推测。

---

## 1. 纠偏总表

### C-01 「有效带宽 124.6 GB/s，是 273 GB/s 峰值的 46%」

- **R1 原表述**：`BW_eff = 880.8 MB / 7.07 ms = 124.6 GB/s`，并称这是"273 GB/s 的 46%"，进而推断"还有 54% 可收回"。
- **问题类型**：派生估算被当成实测；且分母用规格峰值。
- **具体问题**：
  1. 分子 `880.8 MB` 是**由 config 算出**的 28 个 block 权重字节（28 × 15,728,640 × 2 B），不是计数器读数。E2。
  2. 分母 `7.07 ms` 是 `results/stage2_1/bench_a67/bench.json` 的 `serial-lazy` 中位数，计时区间是 `dag_forward`（28 层）+ `mx.eval` + `synchronize`。E1。
  3. 两者相除得到的是"每步搬运的权重量 ÷ 每步墙钟"，它**没有**排除：L2/SLC 命中、lazy 图里被常量折叠掉的部分、以及 head 未计入（见 C-10）。因此它是"搬运速率的量级估计"，不是 DRAM 利用率。
  4. `273 GB/s` 是 Apple 公布的 M4 Pro 内存带宽规格值。GPU 侧可达值通常显著低于规格值（共享内存架构、与 CPU 争用、页表/压缩开销）。用规格值做分母会系统性低估利用率。
- **收窄后可支持**：
  > 在 Qwen3-0.6B-Base（28 block，bf16，**不含 output head**）的 M=1 decode 上，每 token 必须触碰的 block 权重为 880.8 MB，实测 7.07 ms/步。若该步确实读满 880.8 MB，则对应 124.6 GB/s 的搬运速率。这个数字是量级估算，未用硬件计数器验证。
- **不可再支持**：
  - "实测 DRAM 利用率 46%"
  - "剩余 54% 带宽可收回"
  - 任何把 46% 当作"浪费比例"的推论
- **恢复强表述所需测量**：
  - `MTLCounterSampleBuffer` 或 `xctrace` 的 GPU 内存计数器（DRAM read bytes / 步）
  - 一个 streaming copy / triad 微基准，定出**本机可达**带宽 X，再用 124.6/X 报利用率
  - 两者都做完，才能谈"剩余多少"

---

### C-02 「46% 说明有一半带宽没用上，所以优化空间在带宽」

- **问题类型**：过度外推（与 R1 §3.2 自身的解释冲突）。
- **具体问题**：R1 §3.2 已逐条判定"M=1 的 GEMV kernel 太小、per-eval 固定开销 ~0.25 ms、merge 内核吃掉重叠"，这些都不属于"带宽浪费"。同一份报告里一边说瓶颈是固定开销，一边说瓶颈是带宽利用率，二者未并置说明。
- **收窄后可支持**：
  > decode 的 7.07 ms 里，一部分是权重流式（下界 880.8 MB / 可达带宽），一部分是与搬运无关的固定成本（kernel 启动、dispatch、Python/图遍历）。R1 的 A1 线性拟合给出**每步固定 0.5575 ms + 每 token 0.00577 ms**（n=1..256 外推，E2），即固定成本约占 7.07 ms 的 8%。剩余部分才可能归因于搬运效率。
- **不可再支持**："一半带宽浪费"作为主因排序。
- **恢复所需**：C-01 的计数器测量 + 一次"把 batch 从 1 提到 8"的实验（若时间近似不变，说明是固定成本主导；若线性增长，说明是带宽主导）。

---

### C-03 「KV 总计约 14 GiB，接近/超过 `max_buffer_length` 13.32 GiB」

- **问题类型**：口径混淆。
- **具体问题**：`max_buffer_length = 14,302,248,960 B = 13.32 GiB` 是**单个 `MTLBuffer`** 的上限。8k 上下文的 KV 是 **28 层 × 2 个（K/V）× 各自的 buffer**，`KVCache` 每层独立 `concatenate` 增长（`mlx_lm/models/cache.py:333-355`），因此是 56 个独立 buffer，每个远小于 13.32 GiB。E0。
  真正的约束是 `max_recommended_working_set_size = 19,069,665,280 B = 17.76 GiB`（统一内存中建议给 GPU 的工作集上限，`mlx_lm/generate.py:230-266` 默认 `set_wired_limit` 到该值）。E1（本机 `mx.device_info()` 复测，architecture = `applegpu_g16s`）。
- **收窄后可支持**：
  > 8k 上下文时该模型 KV 总量约 14 GiB（按 112 KiB/token 估算，E2），加上 881 MB 权重后逼近 17.76 GiB 工作集上限，因此 8k 上下文在 24 GB M4 Pro 上不可行，除非量化 KV 或缩短上下文。
- **不可再支持**："单 buffer 超过 13.32 GiB 上限"。
- **恢复所需**：逐层 `nbytes` 实测（`sum(c.nbytes for c in cache)`），以及 `mx.get_active_memory()` / `get_peak_memory()` 的峰值记录。

---

### C-04 「M4 没有 NAX，所以用不到 tensor core / MMA」

- **问题类型**：**事实错误**（本轮最需要纠正的一条）。
- **事实**：
  - NAX（Neural Accelerators）指 M5/A19 起的 Metal 4 `MTLTensor` + MPP `matmul2d` 路径，需要 `-std=metal4.0`。M4 没有这条路径。
  - 但 `simdgroup_matrix`（`simdgroup_float8x8` / `simdgroup_half8x8`）是 **Apple9 家族（含 M4）就有的 MMA 指令**。它与 NAX 是两代不同的东西。
  - 直接证据（E0）：
    - `ggml/src/ggml-metal/kernels/mul_mv_mma.metal`：出现 `simdgroup_float8x8` 7 处、`simdgroup_half8x8` 2 处，用于量化 GEMV 的 MMA tile。
    - `ggml/src/ggml-metal/kernels/ssm.metal:303-306`：注释原文 "Chunked SSD SSM scan via Metal simdgroup MMatrix Multiply-Accumulate (simdgroup_float8x8) fast path."
    - MLX 侧：`mlx/backend/metal/kernels/` 有 `quantized_nax.h`（63.8 KB）与 `fp_quantized_nax.h`（28.8 KB）——这两个才是 NAX 专属路径；非 NAX 的 `quantized.h`（91.0 KB）走的是通用 simdgroup 路径。
- **收窄后可支持**：
  > M4 Pro 没有 NAX/`MTLTensor` 路径，因此拿不到 M5 那类大 tile GEMM 的吞吐；但它有 `simdgroup_matrix`，M=1 的量化 GEMV 已经能用 8×8 MMA 摊薄 dequant 的 issue 成本（llama.cpp 就是这么做的）。
- **不可再支持**：
  - "M4 用不到 tensor core"
  - "要等 M5 才能用 MMA"
  - 任何把"M4 缺 NAX"当作"M4 无法做高效低比特 GEMV"的理由
- **影响**：这一条会改变机会排序。"等 M5"不再是低比特直算的前置条件。

---

### C-05 「61.3 ms vs 116.9 ms 的 1.9× 差异原因未定位」

- **问题类型**：未解释的度量冲突（Issue 明确要求解释或标注）。
- **本轮新证据**：两轮 manifest 在**除 `git_commit` 外的所有字段上完全一致**：

  | 字段 | stage2_0 | stage2_1 |
  |---|---|---|
  | model | Qwen3-0.6B-Base-bf16 | 同 |
  | model_revision | `e8e73ff3…` | 同 |
  | mlx_version | 0.31.2 | 同 |
  | platform | macOS-26.6.2-arm64 | 同 |
  | n_tokens | 257 | 257 |
  | batch | 1 | 1 |
  | **git_commit** | **`73d3379`** | **`6a5e6ea`** |

  且**两个变体同比位移**：eager 70.15 → 126.67（×1.81），lazy 61.32 → 116.92（×1.91）。同比例的整体位移说明这是**代码路径或环境的整体位移**，不是某一个变体的算法差异。
- **收窄后可支持**：
  > 同一轮**内部**的相对比较有效（例如 eager vs lazy 的约 2× 差距在两轮都成立，方向一致）。跨轮的绝对毫秒数不可比——两轮跑的是不同 commit 的代码。
- **不可再支持**：任何把 61.32 ms 或 116.92 ms 当作"该模型在该硬件上的 decode 上限/下限"的说法。
- **恢复所需**：见本文 §2 的 **XD-3**（同一进程内复现两轮闭包并二分到行）。

---

### C-06 「block 级并行重叠可达 1.46×」

- **问题类型**：基线污染。
- **具体问题**：`scripts/stage21_bench.py:119-149` 的 `part_a2` 计时区间**包含 `build(k)`**，即 k 次 `embed` + k 次 `mx.eval`。k=1 时该区间为 1.053 ms；而 `part_a1` 的线性拟合给出固定开销 0.5575 ms + 0.00577 ms/token，n=1 时为 0.563 ms。因此 A2 的 k=1 基线多算了约 0.49 ms 的重复 eval。E0+E1。
- **收窄后可支持**：
  > 用 A1 口径（扣除重复 eval）校正后，k=8 的重叠收益约 **1.06×**，不是 1.46×。且该数字来自 A1 的 n=1..256 线性外推，不是同条件实测。
- **不可再支持**：
  - "1.46× 是 block 并行的实测上限"
  - 用 1.46× 论证"并行有希望，只是没做好"
- **更根本的限定**：A2 测的是"k 个独立单 block 图并发"，**不是**"28 个 block 的真实 DAG"。它连一个完整的深度并行收益上界都不构成。

---

### C-07 跨轮绝对时间被当作已验证上限

- **问题类型**：方法论。
- **规则化（建议写进项目纪律）**：
  > 任何绝对毫秒数必须同时绑定五项：`git_commit`、`mlx_version`、`model_revision`、`n_tokens`、**计时区间是否含 output head**。缺任一项即降级为 E2，只能用于同轮内相对比较。
- **本轮具体后果**：R1 摘要里的"7.07 ms"、"162.9 tok/s"、"2.06 TFLOP/s" 三项都需要补注（见 C-10、C-11）。

---

### C-08 「只有投机解码能减少 bytes/token」

- **问题类型**：过度外推（Issue 指出需收窄）。
- **问题所在**：该结论的前提是"固定位宽 + 固定稠密结构 + batch=1"。在这个前提下 bytes/token 是常数，只有"一次权重读取服务多个 token"能降低它——这句话本身成立。但 R1 把它写成了普遍命题，而 Issue 列出的三条路径恰恰都在改这个常数：
  - **位宽**：4.25 bpw → 2.3125 bpw 是 **1.84×** 的字节削减（见 `2026-10-r2-kernel-matrix.md`）
  - **动态稀疏**：Prox 在 70% FFN 稀疏下把 FFN 字节降到 28%（含 INT4 代理读取）
  - **MoE**：只读激活专家，bytes/token 与总参数量解耦
- **收窄后可支持**：
  > 在**固定位宽、固定稠密结构、batch=1** 的前提下，bytes/token 是常数，此时只有"一次读取服务多 token"能降低它。位宽、稀疏、MoE 三条路改变的是这个常数本身，不属于"减少 bytes/token 的手段"，而是"降低基数"。
- **不可再支持**：把该句当作对低比特/稀疏/MoE 的否定。

---

### C-09 「DAG 无价值」不可普遍外推

- **问题类型**：过度外推。
- **可外推的边界**（Stage 2.0/2.1 的实测范围）：Qwen3-0.6B、M4 Pro、M=1 decode、28 block、bf16 权重、固定窗口分组搜索空间。
- **收窄后可支持**：
  > 在 M=1 且权重未量化时，把关键路径从 28 压到 7 不产生墙钟收益，因为 DAG 不改变 bytes/token，而瓶颈是每步固定开销 + 权重流式。prefill / 大 M 场景下 DAG 反而更慢（R1 已测 eager 变慢 9–21%）。
- **不可再支持**：
  - "任何硬件上深度并行都没有收益"
  - "DAG 这条路线应该关闭"（作为普遍结论）
- **方向性说明（不是新结论，是逻辑）**：量化后权重字节变小、固定开销占比**更高**，所以"M=1 下深度并行无收益"这个结论在量化后只会更强，不会更弱。但这条推理不能替代测量。

---

### C-10（新增）「7.07 ms / 162.9 tok/s」不含 output head

- **问题类型**：口径混淆，且影响所有端到端加速比。
- **事实**：
  - `scripts/stage21_bench.py:346-357` 的 decode 计时区间是 `dag_forward(model, h_tok, dag, None, mx, caches=cs)` 然后 `mx.eval(o)`。`dag_forward` 只跑 28 个 block。**没有** `model.model.norm`，**没有** `lm_head` / `embed_tokens.as_linear`，**没有**采样。
  - 该模型 `tie_word_embeddings=True`，head 权重 = 151,936 × 1024 × 2 B = **311.2 MB**，是 block 权重 880.8 MB 的 **35%**。
- **收窄后可支持**：
  > 7.07 ms 是"28 block、不含 head"的一步；162.9 tok/s 是 `1/7.07ms`，不是模型 tok/s。若同带宽成立，完整一步 ≈ (880.8 + 311.2) MB / 124.6 GB/s ≈ **9.57 ms**（E2）。
- **不可再支持**：把 7.07 ms 或 162.9 tok/s 与 mlx-lm 官方 tok/s、llama.cpp tok/s 直接比较。
- **影响**：所有基于 7.07 ms 的"端到端加速比"都系统性高估，因为 head 是**加性固定成本**，稀疏/低位的收益不作用于它（除非也量化 head）。对 27B 级模型 head 占比小得多（vocab 不变但 hidden 更大、层数更多），所以这个偏差在 0.6B 上最大，在大模型上可忽略——但这必须写清楚，不能默认。

---

### C-11（新增）「prefill 2.06 TFLOP/s」是 matmul-only 上界

- **事实**：`scripts/stage21_bench.py:80-88` 的 `flops_per_block` 只统计 projections、MLP、attention scores。a7 自己的 `note` 写了 "softmax / norms / rope excluded, so these are upper bounds on achieved throughput"。E0。
- **收窄后可支持**：
  > prefill 的 **matmul 部分**达到 2.06 TFLOP/s（单 Block 孤立 3.4 TFLOP/s）。这不是硬件利用率，因为分母缺了 softmax/norm/rope 的时间，分子也缺了它们的 FLOPs。
- **不可再支持**："prefill 只跑到峰值的 X%"。

---

### C-12（新增）「46% of 273 GB/s」里的 273 是规格值

并入 C-01，单列是为了让它可被引用：**273 GB/s 是 Apple 公布规格，不是本机可达值。任何以它为分母的"利用率"都必须标注 "vs spec"。**

---

## 2. 三条判定性实验设计（只设计，不执行）

见 `2026-10-r2-report.md` §4 交付 C。这里只登记编号与对应纠偏条目：

| 编号 | 要回答的问题 | 对应纠偏 |
|---|---|---|
| XD-1 | 2.31 bpw 直算相对 4.25 bpw 的实测加速，能兑现字节比 1.84× 的多少 | C-01, C-08 |
| XD-2 | 4-bit 基座上的动态稀疏，临界稀疏率在哪 | C-08, C-09 |
| XD-3 | 61.3 vs 116.9 ms 的 1.85× 归因到代码还是环境 | C-05, C-07 |

---

## 3. 本轮**未**纠正、仍维持 R1 原表述的条目

以下条目本轮复核后认为成立，列出以免读者以为全表都被推翻：

| R1 结论 | 复核依据 |
|---|---|
| mlx-lm 五处特性互斥（KV 量化✗批处理、KV 量化✗滑窗、批处理✗投机、批处理✗keep>0、量化 SDPA✗sinks） | 行号已复核，见 `2026-10-r2-report.md` §2.3 |
| `QuantizedKVCache` 每 token 重新 `mx.quantize` | `cache.py:277-281` |
| `KVCache` 每 256 token `concatenate` 增长 | `cache.py:333-355` |
| `RotatingKVCache.to_quantized` 抛 `NotImplementedError` | `cache.py:551-552` |
| `max_recommended_working_set_size` = 17.76 GiB / `max_buffer_length` = 13.32 GiB | 本机 `mx.device_info()` 复测 |
| 串行状态无收益（B3−B2 ≈ 0.01 nats，CI 跨 0） | `results/stage1_7` 配对 bootstrap |
| 补偿增益 100% 来自输入（打乱配对比不补偿差 0.66–0.68 nats） | `results/stage1_7` |

---

## 4. 一句话总结

R1 的**机制判断**（固定开销 + 权重流式主导 M=1 decode、DAG 不改 bytes/token、mlx-lm 特性互斥）经复核成立；
R1 的**数字包装**（46% 利用率、54% 可收回、1.46× 重叠、14 GiB 撞单 buffer 上限、M4 无 MMA）有五处必须收窄，
其中 **C-04（M4 有 simdgroup_matrix）是事实错误**，**C-10（7.07 ms 不含 head）是新发现的系统性偏差**。
