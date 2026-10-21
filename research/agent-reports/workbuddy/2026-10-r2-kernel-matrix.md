# R2 交付 B：格式 × kernel × shape 兼容矩阵

**范围**：2024–2026 年提出的低位宽/稀疏编码格式，落到 **M4 Pro（`applegpu_g16s`, 16 core, 24 GB 统一内存, Metal 3）** 上的可执行性
**方法**：读源码与论文，不跑 GPU 基准（Issue #2 明确禁止）
**日期**：2026-10-21

---

## 0. 读法

每一行的 bpw 都按**实际结构体大小**算，不用论文宣称值。`QK_K = 256`（GGUF 的 super-block）。

**"能否从 Python 落地"** 是本项目最关心的一列，因为：
- MLX 的量化 GEMV 在 C++ 模板里（`mlx/backend/metal/kernels/quantized.h`，91 KB），**用户不可扩展**；
- 但 `mx.fast.metal_kernel`（本机 `mlx 0.31.2` 已确认存在）允许**在 Python 里直接注入 Metal 源码**并指定 grid/threadgroup/template。这意味着"自定义 kernel"不必 fork MLX C++。

这一条把"移植成本"从"改 MLX 源码并维护 fork"降到"写一个 Python 模块"。全表的成本判断都以此为基准。

---

## 1. 编码格式逐项拆解

### 1.1 已经在 Metal 上跑着的格式

以下格式在 **llama.cpp `master` 的 `ggml/src/ggml-metal/kernels/mul_mv.metal`** 里有对应 GEMV 内核（E0，本轮逐个从内核名列表核对）：

| 格式 | 结构体（`ggml-common.h`） | 每 256 权重字节 | **实际 bpw** | 有 Metal GEMV | 内核名 |
|---|---|---|---|---|---|
| IQ2_XXS | `half d; uint16 qs[32]` | 66 | **2.0625** | ✅ | `kernel_mul_mv_iq2_xxs_f32` |
| IQ2_XS | `half d; uint16 qs[32]; uint8 scales[8]` | 74 | **2.3125** | ✅ | `kernel_mul_mv_iq2_xs_f32` |
| IQ2_S | `half d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]` | 82 | **2.5625** | ✅ | `kernel_mul_mv_iq2_s_f32` |
| IQ1_S | `half d; uint8 qs[32]; uint16 qh[16]` | 50 | **1.5625** | ✅ | `kernel_mul_mv_iq1_s_f32` |
| IQ1_M | `uint8 qs[32]; uint8 qh[16]; uint8 scales[8]` | 56 | **1.75** | ✅ | `kernel_mul_mv_iq1_m_f32` |
| IQ3_XXS | `half d; uint8 qs[96]` | 98 | **3.0625** | ✅ | `kernel_mul_mv_iq3_xxs_f32` |
| IQ3_S | 见 §1.2 | 110 | **3.4375** | ✅ | `kernel_mul_mv_iq3_s_f32` |
| IQ4_NL | `half d; uint8 qs[16]`（`QK4_NL=32`） | 18 / 32 权重 | **4.5** | ✅ | `kernel_mul_mv_iq4_nl_f32` |
| IQ4_XS | `half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]` | 136 | **4.25** | ✅ | `kernel_mul_mv_iq4_xs_f32` |
| Q2_K | `uint8 scales[16]; uint8 qs[64]; half2 dm` | 84 | **2.625** | ✅ | `kernel_mul_mv_q2_K_f32` |
| Q4_K | `half2 dm; uint8 scales[12]; uint8 qs[128]` | 144 | **4.5** | ✅ | `kernel_mul_mv_q4_K_f32` |
| Q6_K | `uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d` | 210 | **6.5625** | ✅ | `kernel_mul_mv_q6_K_f32` |
| TQ2_0 | `uint8 qs[64]; half d` | 66 | **2.0625** | ✅ | `kernel_mul_mv_tq2_0_f32` |
| TQ1_0 | `uint8 qs[48]; uint8 qh[4]; half d`（5 元素/字节，3⁵=243<256） | 54 | **1.6875** | ✅ | —（走通用路径） |
| MXFP4 | `uint8 e; uint8 qs[16]`（`QK_MXFP4=32`） | 17 / 32 权重 | **4.25** | ✅ | `kernel_mul_mv_mxfp4_f32` |

**这张表推翻了 Issue 的一个隐含前提。** Issue 问"是否已有 Metal 实现（含 BaseRT M4 路径）"。答案是：**2.06 bpw 到 4.5 bpw 的 codebook 型直算，llama.cpp 在 Metal 上已经全都有了**，覆盖 IQ1/IQ2/IQ3/IQ4/TQ/MXFP4 六个家族。缺的不是"低位直算"，缺的是 **trellis 型（QTIP/Proteus/CurveTQ）** 和 **动态稀疏**。

另外 `mul_mv.metal` 里还有 `kernel_mul_mv_id_*` 一整套（`id` = 按 expert id gather），即 **MoE 的 per-expert 量化 GEMV 也已就绪**。

### 1.2 IQ3_S 结构

`half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[IQ3S_N_SCALE]`，合计 110 B / 256 权重 = 3.4375 bpw。留在此处只为完整性，本项目不用它。

### 1.3 MLX 自有的 affine / fp 格式

本机实测（`mlx 0.31.2`，E1）：

| mode | 支持的 bits | group 约束 | 每 64 权重实际 bpw（bits=4 为例） |
|---|---|---|---|
| `affine` | 2, 3, 4, 5, 6, 8 | 任意（64 为默认） | `(4×64 + 16 + 16) / 64` = **4.5** |
| `mxfp4` | — | **必须 32** | `(4×32 + 8) / 32` = **4.25** |
| `mxfp8` | — | **必须 32** | `(8×32 + 8) / 32` = **8.25** |
| `nvfp4` | — | **必须 16** | `(4×16 + 8) / 16` = **4.5** |

`mx.quantize` 传错 group 会直接抛 `ValueError: [quantize] mxfp4 quantization requires group size 32 but got 64.`（本轮实测原文）。

**关键数字**：
- MLX `affine` 4-bit g64 = **4.5 bpw**，比 GGUF **IQ4_XS 的 4.25 bpw 更差**（因为 scale 与 bias 都用 bf16 存，每组 32 bit 开销）。
- MLX `affine` 2-bit g64 = **2.5 bpw**，比 **IQ2_XS 的 2.3125 bpw 更差**。
- MLX 的 scale/bias 与输入同 dtype：权重是 bf16 时 scale 就是 bf16。若权重降到 fp16，开销同比减半。

**这不是细节**：如果目标是"在 24 GB 里塞下 27B"，MLX affine 4-bit 比 IQ4_XS 多吃 **0.25 bpw × 27e9 / 8 = 0.84 GB**。0.84 GB 在 24 GB 机器上是能决定"装得下/装不下"的量级。

### 1.4 尚无 Metal 实现、且部分无任何实现的格式

| 方法 | 出处 | 编码 | bpw | 代码状态（本轮核实） | Metal |
|---|---|---|---|---|---|
| **QTIP** | Cornell-RelaxML/qtip，NeurIPS 2024 | 随机正交旋转 + trellis 码本（256 项 / 8 权重）+ 欧氏搜索 | ~2.0 | 公开，仅 **CUDA**（`qtip-kernels/src/inference.cu`, `qtip_torch.cu`），最后 push 2025-06-22 | ❌ |
| **Proteus** | ICML 2026，`YZW-explorer/Proteus` | **lookup-free**：`MUL-BAL` 生成器用整数算术链替代码本 + per-layer affine Gaussianizer，**零运行时表加载**，解码走 INT32 管线 | ~2.0 | **仓库是占位符**（"code release in progress"，3 star，push 2026-05-29）；README 承诺的只有 CUDA kernel | ❌ |
| **CurveTQ** | arXiv 2610.09212（2026-10-06） | **无旋转** trellis；把 Hessian 的 LDL 对角权重放进 Viterbi 分支度量；per-block start state；factored scale field + 闭式 quantile table | ~2.0 | GitHub 搜索 `curvetq` 返回 **0 个仓库**；arXiv 无 HTML | ❌ |
| **XOR-Trellis** | arXiv 2610.00432（2026-09-30） | Hadamard-free + XOR 型低复杂度反量化 | — | 未核实 | 未核实 |
| **GSQ** | arXiv 2604.18556，`IST-DASLab/GSQ`（107 star，push 2026-07-11） | **对称标量网格 + group-wise scale**，用 Gumbel-Softmax 学每坐标的网格分配。repo 描述原文："1–3 bit scalar, **INT/GGUF-compatible**" | 2–3 | 公开，Python | ✅ **不需要新 kernel** |
| **RCO** | arXiv 2605.00649，`IST-DASLab/RCO`（54 star） | 不是量化格式，是**比特预算分配器**：把"给 N 个张量各选一种量化类型、总大小恰好等于预算"建模为黎曼流形上的约束优化 | — | 公开，Python | ✅ 无 kernel 需求 |
| **GSQ-RCO** | GSQ + RCO 组合流水线 | 产出 **non-uniform GGUF**：每张量独立量化类型，主体 IQ2_XS + 敏感张量上调 | 2.3–2.5 | 有公开产物：`Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`（2.50 bpw, 8.4 GB） | ✅ 复用 IQ2_XS kernel |
| **YAQA** | arXiv 2505.22988（"Model-Preserving Adaptive Rounding"） | **不是编码格式**，是自适应舍入算法（把网络输出误差纳入舍入决策），产出标准整数权重 | 跟随基座 | 学术代码 | ✅ **不改变格式，零 kernel 工作** |
| **Prox** | arXiv 2607.27591 | 不是格式：两阶段无训练 **FFN 激活稀疏**。Stage1 用 INT4 代理权重 + 输入稀疏构造共享通道掩码，Stage2 精确算选中通道 | — | 无公开代码 | ❌ 仅 CUDA/Triton |
| **SpQt** | arXiv 2511.04477 | 不是格式：**zigzag 量化布局**（按列分组 + superblock 行主序存储）+ 专用 GEMV + 前缀和索引收集 | Q4_K 基座 | **无公开代码**；论文称基于 llama.cpp **b5711** fork，约 **8K 行 C/C++** | ✅ **是**（论文就在 M2/M2 Pro/M2 Max/A19 Pro 上测的） |

**Issue 里把 YAQA 与 QTIP/Proteus/CurveTQ 并列是不成立的**：YAQA 是舍入算法，不产生新编码，不需要任何新 kernel。把它放进"低位宽直算 kernel"这一栏会高估这条路线的工作量。

---

## 2. 兼容矩阵：格式 × 引擎 × shape

### 2.1 格式 × 引擎

| 格式 | llama.cpp Metal | MLX | 备注 |
|---|---|---|---|
| IQ1_S / IQ1_M | ✅ GEMV | ❌ 无 IQ 家族 | MLX 不支持任何 IQ 格式 |
| IQ2_XXS / XS / S | ✅ GEMV + `id` gather | ❌ | |
| IQ3 / IQ4_XS / IQ4_NL | ✅ | ❌ | |
| Q2_K / Q4_K / Q6_K | ✅ | ❌ | MLX 只认自己的 affine/fp |
| TQ2_0 / TQ1_0 | ✅ | ❌ | 三元特化 |
| MXFP4 | ✅ | ✅ `mode="mxfp4"`（g32） | **唯一两边都有的格式** |
| MLX affine 2/3/4/5/6/8 | ❌ | ✅ | |
| MLX nvfp4 / mxfp8 | ❌ | ✅ | |
| QTIP / Proteus / CurveTQ | ❌ | ❌ | 无 Metal，Proteus/CurveTQ 连代码都没有 |
| GSQ 产出（INT/GGUF） | ✅ | 仅当转成 MLX affine 时 | 转换会丢 GSQ 的非均匀网格 |
| GSQ-RCO 产出（IQ2_XS GGUF） | ✅ 直接跑 | ❌ 需先转 MLX | **`gguf2mlx` 类工具存在（本机 `~/code/gguf2mlx-*`），但 IQ2_XS→MLX affine 是有损重编码** |

**矩阵里最重要的一格**：`MXFP4` 是唯一在 llama.cpp Metal 与 MLX 两边都有原生 kernel 的低位格式（g32，4.25 bpw）。它同时也是 `gpt-oss` 系列用的格式。如果要做跨引擎对照实验，MXFP4 是唯一不需要重编码的公共基准。

### 2.2 shape × 路径（M = batch × 序列长度）

decode 时 M=1；投机解码验证 M=1+k；prefill M=大。各格式在这三档的实际路径：

| shape | llama.cpp Metal 走什么 | MLX 走什么 | 风险 |
|---|---|---|---|
| **M=1**（decode） | `kernel_mul_mv_<type>_f32`，每 threadgroup 4 行，`simdgroup` 归约 | `qmv`（`quantized.h` 模板特化） | 这是所有格式的主战场。**dequant 指令数 / 权重** 决定能否吃满带宽 |
| **M=2..8**（投机验证 / 小 batch） | 同上，`_4` 变体（每 threadgroup 4 行×4 token） | `qmv` 或退化到 `qmm` | IQ 系列的码本查表在多 token 时可复用，单位成本下降；MLX affine 的 unpack 也可复用 |
| **M≥16**（prefill） | `mul_mm.metal`（tensor tile GEMM） | `steel/` GEMM | 低位宽在这里是 **compute-bound**，dequant 成本被 GEMM 摊薄，收益主要来自 bytes；`mul_mm` 对 IQ 系列支持不如 `mul_mv` 完整 |
| **MoE gather** | `kernel_mul_mv_id_<type>_f32` | `mx.gather_qmm`（本机确认存在） | MoE 的 bytes/token 由激活专家数决定，与总参数解耦 |
| **output head GEMV** | 同 M=1 | `as_linear` → `quantized_matmul` | head 是**加性固定成本**，稀疏/深度优化不作用于它（见 audit C-10） |

### 2.3 每 8 权重的解码算术（决定 M=1 的 ALU 预算）

| 格式 | 每 8 权重需要的操作 | 估计 ops/权重 | 依据 |
|---|---|---|---|
| Q4_K / IQ4_XS | 4 B 取指 → 8 次 shift/mask → 8 次 scale-FMA | **~2.0–2.5** | 结构简单 |
| MLX affine 4-bit | 1 次 uint32 取指 → 8 次 shift/mask → 8 次 `(q - bias) × scale` FMA | **~2.0–2.5** | 有 bias 减法，略多于 GGUF |
| IQ2_XS | 1 次 uint16 码本索引 → 查 512×8 表 → 8 次符号应用 → 8 次 scale-FMA | **~2.5–4** | 表查在 GPU 上是 gather，不是 broadcast |
| IQ2_XXS | 同上，256×8 表，无 per-32 子 scale | **~2.5–4** | |
| TQ2_0 | 2 bit/权重直接取值 + 1 次 scale | **~1.5–2** | 三元最省 |
| QTIP | 256 项码本查表（每 8 权重一次）+ 状态转移 | **~3–5** | Proteus 论文称 QTIP 在 CUDA 上受 MIO throttle / scoreboard stall |
| Proteus | `MUL-BAL` 整数算术链，**无表** | **~3–6**（作者声称低于 QTIP） | 未开源，无法核 |
| CurveTQ | Viterbi 分支度量加权 + trellis 状态推进 | **~4–8** | 无旋转省掉了逆旋转，但状态机仍在 |

---

## 3. 2.3 bpw 复杂解码 vs 3.5–4 bpw 简洁直算：crossover 在哪

### 3.1 字节侧的账（确定）

以 27B 级 dense 模型为例：

| 方案 | bpw | 权重字节 | 相对 BF16 | 相对 IQ4_XS |
|---|---|---|---|---|
| BF16 | 16 | 54.0 GB | 1.00× | — |
| MLX affine 4-bit g64 | 4.5 | 15.2 GB | 3.56× | 0.94× |
| IQ4_XS | 4.25 | 14.3 GB | 3.76× | 1.00× |
| MLX affine 2-bit g64 | 2.5 | 8.44 GB | 6.40× | 1.70× |
| IQ2_S | 2.5625 | 8.65 GB | 6.24× | 1.66× |
| **IQ2_XS** | **2.3125** | **7.80 GB** | **6.92×** | **1.84×** |
| IQ2_XXS | 2.0625 | 6.96 GB | 7.76× | 2.06× |
| QTIP / Proteus / CurveTQ | ~2.0 | 6.75 GB | 8.00× | 2.13× |

**结论一（硬上界）**：2.3125 bpw 相对 4.25 bpw 的**字节比是 1.84×**。任何实测 decode 加速 **> 1.84×** 都不可能来自字节，只可能来自别的东西（比如 4-bit kernel 本身没调好）。任何 **< 1.84×** 的部分就是 kernel 开销。

**结论二**：trellis 相对 IQ4_XS 的字节上界是 2.13×，但相对 IQ2_XS 只有 1.16×。**如果 IQ2_XS 已经能用（llama.cpp Metal 已有 kernel），那么 QTIP/Proteus/CurveTQ 相对它的额外收益上限只有 16% 字节**，而代价是"从零写一个 trellis 解码器"。这是本报告最重要的一个性价比判断。

### 3.2 ALU 侧的账（估算，需实测确认）

M4 Pro 16 core GPU：约 2048 个 ALU lane（16 × 128），时钟量级 1.4 GHz → **约 2.9×10¹² ALU ops/s**（E2，量级）。

若 decode 完全被带宽限制，则每权重的 ALU 预算 = ALU 吞吐 ÷ 权重吞吐：

| 有效带宽假设 | 27B @ 2.3125 bpw 的权重吞吐 | **每权重 ALU 预算** |
|---|---|---|
| 124.6 GB/s（R1 估算值） | 4.3×10¹¹ w/s | **≈ 6.7 ops/weight** |
| 190 GB/s（约 70% 规格） | 6.6×10¹¹ w/s | **≈ 4.4 ops/weight** |
| 273 GB/s（规格峰值） | 9.4×10¹¹ w/s | **≈ 3.1 ops/weight** |

对照 §2.3：IQ2_XS 需要 2.5–4 ops/weight，IQ4_XS 需要 2.0–2.5，trellis 需要 3–8。

**结论三（crossover 的机制）**：
- 在 **124.6 GB/s** 的实际带宽下，IQ2_XS 的 ALU 需求（2.5–4）**装得进** 6.7 的预算 → 2.3 bpw 应该能兑现接近 1.84× 的字节收益。
- 在 **接近规格峰值** 的带宽下，预算压到 3.1，IQ2_XS 刚好卡在边界，trellis（3–8）**必然超支**。
- 也就是说：**"2.3 bpw 是否值得"不是一个格式问题，而是"这台机器实际能跑到多少带宽"的问题。** 带宽越低，低位宽越值；带宽越高，低位宽越容易被 ALU 卡住。这与 SpQt 论文观察到的"更大模型 + 更弱硬件 → 加速更明显"一致。

**结论四**：M4 Pro 的实测有效带宽（R1 估 124.6 GB/s，见 audit C-01 已标注为派生值）恰好落在"低位宽划算"的一侧。这是支持做低位宽直算的唯一量化理由，且它依赖一个**尚未用计数器验证**的数字。这是 XD-1 存在的理由。

### 3.3 谁已经证明过这件事

| 证据 | 内容 | 强度 |
|---|---|---|
| SpQt（arXiv 2511.04477） | 在 **M2 / M2 Pro / M2 Max / A19 Pro** 上，Q4_K + 50% 稀疏端到端 decode **1.33–1.55×**；稀疏 GEMV 微基准 50% 稀疏 **1.78×** | **E1（作者实测，Apple 硬件，同代架构）** |
| SpQt 负结果 | **zigzag 布局但无 load balancing**：50% 稀疏只有 **1.02–1.06×** | E1，极其重要 |
| SpQt 负结果 | 原布局逐权重跳过（naive sparse）：**< 0.2×**（warp 发散） | E1 |
| Proteus（ICML 2026） | QTIP 的码本查表在 CUDA 上受 MIO throttle / short-long scoreboard stall；lookup-free 算术解码更快 | E3（无代码可核） |
| R1 本机 | 28 block decode 7.07 ms（不含 head），`mx.compile` 只带来 2.4% | E1 |

**SpQt 的两个负结果是本报告最有价值的反面证据**：它证明"稀疏化"本身不产生收益，**收益全部来自把不规则跳过变成结构化块跳过 + 负载均衡**。任何"加个 mask 就快"的设想都会被这两条打掉。

---

## 4. 动态稀疏 × 量化：数据流与临界稀疏率

### 4.1 两条路线的数据流

**路线 A：Prox（激活稀疏，训练-free，两阶段）**

```
x [1, d_model]
  │
  ├─(a) 输入稀疏:  m_x = 1[|x| >= τ1]              ← 逐层离线校准的静态阈值
  │        ↓
  ├─(b) INT4 代理 up/gate GEMV（只算 m_x 保留的输入行）
  │        s̃ = (x·W̃_up) ⊙ SiLU(x·W̃_gate)          ← 代理值不向下游传播
  │        ↓
  ├─(c) 中间掩码:  m_s = 1[|s̃| >= τ2]              ← 逐 token 动态
  │        ↓
  └─(d) Stage2 精确计算：用 FP16 原权重只算 m_s 选中的通道
           up/gate 做输出稀疏，down 做输入稀疏（同一个 m_s）
```

关键事实（论文原文/表格）：
- **不用 top-k，用阈值**。论文实测阈值在 8 个设置上全部优于 top-k，平均 +1.79 分。副作用是稀疏度逐 token 变化（P10–P90 跨度 17 个百分点），**负载不均衡**。
- 掩码**逐层不同**，但**层内 up/gate/down 共享同一个 `m_s`**。
- 成本模型 `C = 2α(1-s1) + 3(1-s2)`，稠密 = 3。实测 **α = 0.234–0.351，均值 0.285**（A6000 / 3090 / A100）。
- 稀疏度分配（α=1/3 时）：target 40% → (s1=0, s2=62.2%)；50% → (10%, 70%)；60% → (55%, 70%)；70% → (70%, 76.7%)。
- **代理权重常驻显存，额外约 12% 权重存储**。论文自己承认，并列为未来工作。
- 实测：70% 稀疏 **1.99×** 端到端 decode（A6000，single batch）。
- **明确不支持大 batch**；量化兼容只验证了精度，**联合加速 kernel 列为未来工作**。

**路线 B：SpQt（hidden-state 稀疏 × Q4_K，zigzag 布局）**

```
W 按列分组（column-aligned grouping）→ 一个 hidden 元素稀疏 ⇒ 整列 superblock 可整体跳过
  但 superblock 内部仍按行主序存（row-major storage）⇒ 输出向量访存仍连续
        ↓
逐层：hidden state x → 阈值（一次性校准，无预测器）→ 二值 mask
        ↓
索引收集 kernel：Blelloch 并行前缀和 + 全局原子计数器 → 紧凑 idx[0:n_ns]
        ↓
稀疏 GEMV：把 idx 均匀切给各 simdgroup（而非静态切 superblock）→ 负载均衡
        ↓
prefill 走 dense，decode 走 sparse（token 数 >1 自动切）
```

关键事实：
- 作用范围是**所有 GEMV 权重**（不只 FFN），**排除 token/output embedding**。
- 阈值校准基于 **TEAL**，一次性离线。
- **无预测器**（这是相对 neuron-sparsity 的核心优势）。
- 基于 llama.cpp **b5711**，约 8K 行 C/C++。
- 实测：**M2 Max 1.47×、M2 1.39×、M2 Pro 1.33×、A19 Pro 1.55×**（50% 稀疏 vs Q4_K dense）。
- 精度：Llama-2-7B Q4_K 5.28 → 25% 稀疏 5.27 → **50% 5.65** → **65% 7.39**。

### 4.2 临界稀疏率（本报告的定量结论）

**问题**：在 4-bit 基座上开稀疏，要多少稀疏率才打得过"直接降到 2.31 bpw"？

设 FFN 占权重字节比例 `f`，稀疏率 `s`，则相对 BF16 的流量：

```
traffic(s) / B_bf16 = f·(1-s) + (1-f)
```

`f` 的取值：Qwen3-0.6B（16Q/8KV，inter=3h）**f = 9.44/15.73 = 0.60**；
现代 GQA 大模型（如 40Q/8KV，inter≈2.7h）**f ≈ 0.75**。

令 4-bit 基座 + 稀疏 的流量 = IQ2_XS 的流量（0.1445）：

| f | 解出的 s |
|---|---|
| 0.60 | **s > 0.548** |
| 0.75 | **s > 0.563** |

**临界稀疏率 ≈ 0.55–0.56。**

而 SpQt 的精度数据显示 **65% 已经把 PPL 从 5.28 推到 7.39（+40%）**，50% 只到 5.65（+7%）。

**结论五**：在 4-bit 基座上靠稀疏去赢 2.31 bpw，可行窗口只有 **56%–60%**，宽度约 4 个百分点，且落在精度已经开始明显退化的区间。**这条路单独走不通。**

**结论六（真正的机会）**：把两者叠加——**2.31 bpw 基座 + 50% 稀疏**：

```
0.1445 × [0.75 × 0.5 + 0.25] = 0.1445 × 0.625 = 0.0903
→ 相对 IQ4_XS dense 是 2.77×，相对 BF16 是 11.1×
```

这是唯一能在 M4 Pro 上把 27B 的 decode 推到"每 token 6 GB 以下"的组合，但它需要**同时支持低位宽解码和稀疏跳过的 kernel**——而 Prox 论文明确说"联合 kernel 列为未来工作"，SpQt 只在 Q4_K 上做了稀疏、没有和 IQ2 结合。**这个组合目前世界上没有人实现过。**

### 4.3 关键风险：代理/索引的固定成本

| 成本项 | Prox | SpQt |
|---|---|---|
| 额外存储 | **+12% 权重**（INT4 代理） | 0（只有索引数组） |
| 每层额外 kernel | 2 个（代理 GEMV 融合 + 索引） | 1 个（前缀和索引收集） |
| 负载不均衡 | **有**（阈值选择 → 稀疏度逐 token 变） | **有，且是决定性的**（无 load balancing 时只有 1.02–1.06×） |
| 精度代价（70% / 50%） | Qwen3-8B 下游 76.1 → 68.6（**−7.5 分**） | PPL 5.28 → 5.65 @50%（+7%） |

Prox 的 12% 额外存储对 24 GB 机器是硬约束：27B @ 4-bit = 15.2 GB，+12% = **17.0 GB**，加上 KV 就撞 17.76 GiB 工作集上限。**Prox 在 24 GB 上装不进 27B**，除非降到 3-bit 基座或接受更短上下文。

---

## 5. 最强反面证据汇总

按"能杀死哪个想法"排列：

| # | 反面证据 | 杀死什么 |
|---|---|---|
| 1 | SpQt：zigzag 布局但**无 load balancing**，50% 稀疏只有 1.02–1.06× | "只要按列分组就能省字节"——布局是必要条件不是充分条件 |
| 2 | SpQt：naive 逐权重跳过 **< 0.2×**（warp 发散） | "加 mask 就快" |
| 3 | SpQt：65% 稀疏 PPL 5.28→7.39 | "稀疏率越高越好" |
| 4 | Prox：跳过 Stage2 精确计算，直接用代理值喂 down，70% 时 Qwen3-8B 差 **24.3 分** | "代理可以替代真值"（哪怕只是精度上） |
| 5 | Prox：FP16 代理替代 INT4 代理，70% 时 Qwen3-8B 低 9.0 分 | "代理权重可以用高精度换准确"——反了，量化代理反而更好，因为它的误差被隔离 |
| 6 | Proteus 仓库是占位符（3 star，2026-05-29 push，"code release in progress"） | "Proteus 可以复现" |
| 7 | CurveTQ GitHub 搜索 0 结果 | "CurveTQ 可以复现" |
| 8 | QTIP 只有 CUDA（`inference.cu`），无 Metal | "QTIP 能在 Apple 上跑" |
| 9 | llama.cpp Metal 已有 IQ2_XS kernel，trellis 相对它的字节上界只有 **1.16×** | "trellis 值得从零写一个 Metal 解码器" |
| 10 | MLX affine 4-bit g64 = **4.5 bpw**，比 IQ4_XS 的 4.25 更差 | "MLX 的 4-bit 就是 4 bpw" |

---

## 6. 一页结论

1. **2.06–4.5 bpw 的 codebook 型直算在 Metal 上已经有了**（llama.cpp 15 个格式，含 MoE gather）。缺的是 trellis 和动态稀疏。
2. **trellis 相对 IQ2_XS 的字节上界只有 1.16×**，而 Proteus/CurveTQ 无代码、QTIP 只有 CUDA。移植不划算。
3. **GSQ / GSQ-RCO 是唯一"零 kernel 工作 + 有公开 27B 产物"的路线**，产出 IQ2_XS GGUF，llama.cpp Metal 直接跑。
4. **2.3 bpw 值不值得取决于实际带宽**：124.6 GB/s 时 ALU 预算 6.7 ops/weight，够用；接近规格峰值时预算 3.1，不够用。这个数字目前是派生值（audit C-01）。
5. **4-bit + 稀疏要 >56% 才赢 2.31 bpw，但 65% 精度已经崩**。单独走不通；**2.31 bpw + 50% 稀疏 = 2.77×** 是唯一真正的机会，但世界上没人实现过。
6. **Prox 的 12% 代理存储让它在 24 GB 上装不进 27B**。
