# 高能力 LLM 的推理算法与计算数学全景研究
### 面向消费级设备（Apple Silicon / MLX / llama.cpp 级）的证据化综述

**项目**：transformer-residual-lab
**研究日期**：2026-10-21
**文献覆盖**：2024 → 2026-10（2025–2026 的工作逐条打开原始来源核验，不依赖记忆）
**边界**：本轮仅做研究与静态源码/配置核对；未运行 GPU 实验、未训练、未下载模型、未改动旧实验、未启动 Stage 3、未安装插件。
**产出**：本报告 + `research/notes/X0_crosscutting.md`（交叉核验笔记）+ `research/notes/X1_evidence_index.md`（证据索引）+ `research/PROTOCOL.md`（研究协议）

**证据分级标记**（全文统一使用）：
`[P]` 同行评审/预印本论文（已打开摘要页或官方页面）· `[O]` 官方仓库/模型卡/配置文件（已打开）· `[M]` 本仓库自测或我在目标硬件上核到的公开测量 · `[S]` 二手综述/从业者文章（明确标注为低证据等级）

---

## 0. 摘要与我的独立判断

### 0.1 一句话结论

**消费级设备上真正稀缺的不是 FLOPs，而是"每输出 token 需要搬运的权重字节数"和"必须串行完成的依赖轮次"；因此本轮调研中最有价值的两个家族是「低比特权重/格式选择」（压字节）和「推测解码」（压串行轮次），而占文献数量最多的「KV cache 压缩」家族在消费级场景中基本打错了目标。**

这不是文献里现成的结论，而是把 2024–2026 的证据套进本仓库在 M4 Pro 上测到的硬件约束后得到的判断。

### 0.2 本报告的"约束轴"（贯穿全文的判据）

消费级设备（单芯片、16–64 GB 统一内存、batch=1、带宽 100–800 GB/s）上，推理成本可以被拆成四个可分离的量：

| 轴 | 含义 | 谁在管 | batch=1 解码时的重要性 |
|---|---|---|---|
| **容量** | 装不装得下（权重 + KV） | 量化 / 剪枝 / 稀疏 / 驱逐 | 决定"能不能跑"，不决定"跑多快" |
| **带宽** | 每 token 搬运多少字节 | 权重量化、KV 量化、MoE 激活参数、**输出头** | **决定性** |
| **串行轮次** | 生成一个 token 需要几次不可并行的权重流过 | 推测解码、MTP、扩散并行解码、**缩短思考长度** | **决定性** |
| **FLOPs** | 算力 | 稀疏注意力、剪枝、低精度激活 | 几乎不重要（batch=1 时算力是免费的） |

**⚠ 重要修正（2026-10-21，采纳自同仓库并行任务的独立系统审计 `docs/INFERENCE_SYSTEM_AUDIT.md`，任务 B）**
本报告初稿曾把 Stage 2.1 的"每 block 0.558 ms 固定开销 / 解码 90 % 是固定开销"当作机制证据。该独立审计指出**这三个数字的口径有问题**：

- **"0.558 ms 固定开销"是外推而非测量**：n=1..16 的点全部落在 0.47–0.68 ms 且非单调（n=2 比 n=1 慢），边际斜率在这段测不出来；0.0058 ms/token 是由 n=64/256 两点主导的线性拟合。**"解码 90 % 是固定开销"在 n=1 时是恒真的（只有一个 token），不构成机制证据。**
- **真正的机制证据是**：链内 **0.253 ms/Block** vs 孤立单 Block **0.47–0.68 ms/Block** ⇒ **每次 eval 的提交/同步单位成本约 0.2–0.4 ms**（与"eager 14.40 ms vs lazy 7.07 ms、差 2×"互相印证）。
- **A2 的"1.46× 重叠上限"基线被污染**：`part_a2` 的计时区间混入了 k 次 `embed` + k 次 `mx.eval`（`scripts/stage21_bench.py:125-135`），k=1 得 1.053 ms，而 A1 独立测得同规模单 Block（n=64）为 0.762 ms。**按 A1 口径重算为 8×0.762 / 5.751 = 1.06×，不是 1.46×**；且 A2 用的是 n=64，不是 decode 的 n=1。
- **同一段 28-block 代码两轮相差 1.9×**（Stage 2.0 记 61.32 ms，Stage 2.1 记 116.92 ms），来源未定位。**在定位之前，绝对 ms 不可跨报告引用。**

**修正后的、更稳健的锚点**（这三个数字口径可靠、且方向一致）：

| 锚点 | 值 | 口径 |
|---|---|---|
| **decode 有效带宽** | **124.6 GB/s = 273 GB/s 峰值的 46 %** | 880.8 MB ÷ 7.07 ms，28 blocks，batch 1，257-token KV，**不含 LM head** |
| **prefill 有效算力** | **2.06 TFLOP/s**（单 Block 孤立测 3.37–3.49） | 241 GFLOPs ÷ 116.9 ms |
| **LM head 占单步字节** | **311.2 MB = 0.6B 模型单步流量的 26 %** | vocab 151936 × 1024 × 2 B，tied embedding，每 token 必读 |

**46 % 这个数字是本报告真正的地基**，而且它同时否证了两种常见说法：**不是"GPU 已饱和"**（单 Block 约 40 % 算力，端到端只有 2.06 TFLOP/s，都未饱和），**也不是"再多并行几个 Block 就行"**（剩余空间拿不到）。它把瓶颈定位在**访存效率**上——这恰好就是"减少每 token 字节数"这条轴。

同时保留的、本仓库自身的稳健负结果：把逻辑依赖深度从 28 砍到 7，prefill 只赚 5–6 %（**无 CI**）、**decode 赚 0 %**（最好 3.7 %，落在噪声内）；把 4 个 block 权重堆成一个 `(4,out,in)` 矩阵乘（**逐位等价**）在 n=1 时反而**慢 29 %**。

> **推论（本报告的第一条硬约束）**：任何只减少 FLOPs、不减少搬运字节或串行轮次的方法，在本项目的目标设备上期望收益≈0。本仓库之前"跨层 DAG 深度压缩零收益"的负结果，是这条约束的一次干净验证——**深度被压短了，但每步的权重字节数没变**。

### 0.3 五个最重要的发现（全部有原始来源）

**发现 1 —— 量化会让推理模型"想得更久"，从而把加速完全吃掉。**
`[P]` arXiv:2606.25519（*Quantization Inflates Reasoning*, Lian et al., v2 2026-06-29）报告：**Qwen3-4B + AWQ-INT3 的 CoT Token Inflation Ratio = 276.1 %**；在 BBH 上，**BF16 用 285 s，INT4 用 292 s**——尽管 W4A16 Marlin 内核提供 **1.2–1.4× 的单 token 加速**。端到端上量化版比原版**慢 2.5 %**。唯一 Pareto 占优的缓解手段是 **QAT（需要重训）**。
`[P]` arXiv:2606.00206（*Quantized Reasoning Models Think They Need to Think Longer*, Lotfi et al.）补上机制与**可落地的解法**：在 **up to 52 % 的失败里，模型在中间推理步骤已经想对了，但最终没输出答案**；高 KL 位置与高 next-token 熵强相关，量化模型在这些位置过量采样 "wait"/"but"/"alternatively"；一个**免训练的 overthinking-marker logit 惩罚**就能把 CoT 长度压 12–23 % 且准确率不降（5 个模型 1.5B–32B、3 种量化、5 个 benchmark）。
⇒ **对本项目：任何"量化 = 免费加速"的说法，对推理模型不成立。** 而且这项副作用的方向恰好打在消费级设备最在意的"串行轮次"上。

**发现 2 —— 推测解码是全语料里唯一同时满足「压串行轮次」和「严格保持采样分布（L2）」的方法，而且它已经在目标硬件上被实测。**
`[M]` M4 Pro 48 GB（~120 GB/s）上跑 27B Q4（~17 GB/次前向 ⇒ 理论上限 ~7 tok/s，实测 ~7）：llama.cpp 的 `--spec-type draft-mtp --spec-draft-n-max 3` + `--cache-type-k q8_0 --cache-type-v q8_0` 做到 **10.5 tok/s**；换 MLX 原生 Metal 运行时 **MTPLX** 做到 **18.3 tok/s = 2.6×**，MTP depth-3 逐位置接受率 ~73 %/48 %/32 %，25 次 verify 平均 1.84 token/cycle，内存 active/peak 16.2/18.6 GB。该实现用 Leviathan–Chen 概率比接受 + 残差修正，**因此是真正的 L2**（很多"fast decode"工具用 greedy argmax，在 T>0 时是偷偷破坏分布的）。
`[O]` 模型配置核验：**Qwen3.6-27B**（`qwen3_5`，64 层 = 48 linear_attention + 16 full_attention，dense，`mtp_num_hidden_layers: 1`）**原生带 MTP 头**。

**发现 3 —— 2026 年最好的 KV 方法在 Apple Silicon 上根本跑不起来。**
`[O]+[P]` TurboQuant（arXiv:2504.19874, ICLR 2026，Google Research）用 PolarQuant + 1-bit QJL 做到 **~3 bit/通道、≥6× KV 压缩、LongBench 与 FP16 完全同分（50.06 vs 50.06）**、**免校准免训练**、内积估计无偏（≈2.7× Shannon 界）。但：
- `[S]` 截至 2026-04-19，**未合入 vLLM / llama.cpp / Ollama**；llama.cpp 侧是 Issue #20977（OPEN，277+ 👍）与 PR #21089（OPEN，提出 `tbq3_0`/`tbq4_0`）；社区实现 `tonbistudio/turboquant-pytorch`、`scrya-com/rotorquant` 全是 **CUDA/Triton**。
- `[M]` 同一篇 MTPLX 实测文章独立给出结论：TurboQuant 的 llama.cpp PR 是 "**CPU-only, not yet merged**"、"forks 是 **CUDA-focused**"、"**没有 Metal/Apple Silicon 后端**"，而且"**它只压缩 KV cache，不解决限制解码速度的权重带宽瓶颈**"。
`[P]` 同族还有 KVarN（arXiv:2606.03458，Huawei，Hadamard + Sinkhorn 式方差归一 + **K4/V2 非对称精度**，Qwen3-32B 上 3–5× 容量、**吞吐高于 FP16**、MATH500/AIME24/HumanEval 达到 FP16 水平，vLLM 一行 flag、**免校准免改模型**）与 TriAttention（arXiv:2604.04921，MIT/NVIDIA/浙大，**AIME25 长推理 2.5× 吞吐、10.7× KV 压缩、Full Attention 精度 40.8 vs 40.8**，靠 **pre-RoPE 三角函数打分**）。
⇒ **KV 家族在 2026 年同时被两个独立来源判定为"不解决消费级设备的约束"和"没有 Apple 后端"。**

**发现 4 —— "只减 FLOPs"的方法，其端到端收益被系统性高估，而且这个高估是跨模型、跨硬件复现的。**
- `[P]` TurboQuant 作者之一在《The Korea Herald》2026-04-13 明确澄清：6× 内存与 8× 注意力是**两组独立实验，不可相乘**；8× 是 H100 4-bit 下**只算 attention logit**，"如果注意力占总推理时间约一半，**端到端增益更接近 1.5×–2×**"；而且"**短对话里 KV cache 占总内存不到 1 %**，到 200K+ token 才可能超过 60 %"；作者强调的**首要目标是带宽而非省内存**。
- `[S]` AsyncTLS：算子级加速最高 10×，**端到端只有 1.3–4.7×**，因为"即使注意力便宜了 10 倍，MLP、采样和非注意力开销仍占约一半墙钟"。
- `[M]` 本仓库：**深度压缩 decode 收益 0 %**（28→7 层，最好 3.7 % 且在噪声内）；**堆权重 batched matmul 在 n=1 反而慢 29 %**。⚠ 本仓库的"1.46× 并行重叠上限"**不作为机制证据**：其计时区间混入了 k 次 `embed` + k 次 `mx.eval`（`scripts/stage21_bench.py:125-135`），按 A1 独立口径重算只有 **1.06×**；且 A2 用的是 n=64，不是 decode 的 n=1。详见 §0.2 的修正说明。
⇒ **三个独立来源、三个不同硬件、不同模型，指向同一形状的结论。** 这是本报告置信度最高的一条结构性判断。

**发现 5 —— "推理期后处理式稀疏"有一个已被正式命名的根本缺陷：训练/推理不一致。**
`[P]` arXiv:2605.11744（ICML 2026，*Training–Inference Consistent Segmented Execution for Long-Context LLMs*）指出：推理时才施加窗口/稀疏注意力、训练仍用全上下文，会让模型**过度依赖训练时可得、推理时不可得的信息**，从而损害泛化稳定性与长上下文表现。其解法是把片段级执行从"推理优化"提升为**训练与推理共用的建模假设**：固定大小 KV 尾部作为唯一可微跨片段接口 + TBPTT 截断（**K=1 最优**）+ 只读检索前缀 + 头/层稀疏长程头；在 PG19 上 PPL 与全上下文几乎重合，**128K prefill 峰值内存约为 FlashAttention 的 1/6**。
`[P]` arXiv:2504.17768（*The Sparse Frontier*）独立给出训练无关稀疏方法的行为规则：**等算力下，大的稀疏模型打赢小的稠密模型**；**解码比 prefill 更耐压缩**（32B/72B 撑住 17×），**小模型更早崩**（7B 从 16K 的 12× 安全值掉到 128K 的 5×）；**几乎每个配置都有某个任务掉到 5× 以下**。
`[P]` arXiv:2502.11089（DeepSeek NSA）则展示了原生训练路线的上限：它能在 LongBench 上**打赢**全注意力（0.469 vs 0.437），并报告 64K 下 11.6× 解码加速。
⇒ **对"已冻结权重"这一硬约束，这条发现是对后处理式注意力稀疏方法的根本性质疑：它省下的 FLOPs 可能部分被训练/推理失配的质量损失吃掉。**

### 0.4 我的独立判断（不迎合既有方向）

1. **KV cache 压缩在消费级设备上不是主线，而是支线——但支线在 32k 上下文处变成主线。** 文献热度与目标约束严重错配。理由有三：(i) 短上下文（<16k）下 KV 占总内存 <1 %（作者自述 `[S]`），**而按 Qwen3.6-27B 的 64 KB/token 核算，32k 起 KV 就与权重同量级、128k 时 KV 主导**（见 §C.2 拐点表）；容量瓶颈与延迟瓶颈是两种不同的失败，不能混为一谈；(ii) 最好的两个方法（TurboQuant、KVarN）**都没有 Metal 后端**，而 TriAttention 是 vLLM 插件（vLLM 不跑在 Mac 上）；(iii) 本项目硬件上短上下文 decode 是**权重**带宽受限，KV 压缩不触碰它。**例外（必须写清）**：当 KV 大到装不下时（**128k 上下文下 KV 单块约 14 GiB > 本机单 buffer 上限 13.32 GiB**），KV 压缩是唯一能让你"跑得起来"的方法，此时它是**准入条件**而不是**加速手段**——而且它还必须是 **Metal 可实现**的版本，这一点目前只有 TurboQuant 尚未满足。

2. **本项目最该押注的是"减少串行轮次"，不是"减少 FLOPs"，也不是"省内存"。** 唯一同时做到两件事的成熟方法就是推测解码/MTP，而它**已经在目标硬件上被独立实测到 2.6×**，而且**分布严格保持**。它的硬约束是"checkpoint 必须自带 MTP 头"——这与"已冻结权重"冲突，但它**不与"消费级设备"冲突**：Qwen3.6-27B 这种量级正好是消费设备跑得动的规模。

3. **"混合注意力/SSM 架构"是被严重低估的方向，但理由和现有叙事不同。** 现有叙事强调吞吐（Nemotron 3 的 3.3× 是 H200 厂商数字 `[S]`）。消费级设备上真正的理由是**长上下文的 KV 容量**：我按配置核算，Qwen3.6-27B 因为 64 层里只有 16 层是全注意力，**KV 是 64 KB/token，是同宽度全注意力模型的 1/4**；`granite-4.0-h-1b`（36 Mamba-2 + 4 attention）更是 **8 KB/token**。这让"在 24 GB 上跑长上下文"从不可能变成可能。**但 2026 年的旗舰混合模型（Qwen3.8-Flash-Next 80B）根本装不下**（即使 1-bit 也需 72.5 GB `[S]`），所以消费级落点实际上是 **3B–27B 这一档**。

4. **"更高能力 + 更少成本"在 2026 年最可能由"换 checkpoint"而不是"改 checkpoint"实现。** 证据链一致：Qwen3.6-27B（原生 MTP + 75 % 线性注意力）、Qwen3.8-Flash-Next（原生 MTP + 原生 3-gram 预测器 + 原生稀疏 indexer + 层级压缩 + PLE）、granite-4.0-h-*（Mamba-2 混合）。这些能力是**训练进去的**，后处理方法无法凭空造出来。**如果"已冻结权重"是硬约束，就应当承认能力上限被冻结在 checkpoint 里，可做的只有延迟工程。**

5. **质量评估方法本身是本领域最薄弱的环节，而它直接决定上面的结论能否成立。** 证据：压缩失败常常表现为**解码动力学崩坏**（重复环、过度思考、最终答案不输出），而这**在 PPL 与准确率上完全不可见**。`[P]` arXiv:2608.26676（ICML 2026, FOCUS & RePAIR，POSTECH/DGIST）明确写："剪枝会放大文本退化，尤其是重复环，**即使困惑度和任务准确率基本不变**"；他们把退化分解为**进入重复环的风险**与**重复环的持续性**，后者由采样集合中**分配给合理替代项的 escape mass** 决定。`[P]` arXiv:2602.04099 进一步指出 PPL 本身被输入长度系统性污染。**结论：任何只报 PPL/MMLU/AIME 的压缩方法报告，都不足以支撑"可用于强模型"的结论。**

6. **一条被忽视的正向结论：batch=1 的单用户端侧推理天然是确定性的。** `[P][S]` Thinking Machines《Defeating Nondeterminism in LLM Inference》（Horace He, 2025-09-10, DOI 10.64434/tml.20250910）证明推理的不确定性主因**不是**"并发 + 浮点非结合性"，而是**内核缺乏 batch invariance**（内核随 batch/切分策略切换 split-K / stream-k / 不同 tile 与指令，导致归约顺序变化）；Qwen3-235B temp=0 跑 1000 次得到 **80 个不同 completion**（前 102 token 相同，第 103 token 分叉）。他们的修法是 batch-invariant 的 RMSNorm / matmul / attention（`thinking-machines-lab/batch_invariant_ops`），代价是端到端 26 s → 55 s（改进 attention 内核后 42 s）for 1000×100 token。**推论：端侧单用户、batch 恒为 1 时，这个不确定性来源天然被消掉；而"可复现"是端侧相对云端的一个未被宣传的真实优势，一旦转成多用户服务就消失。**

---

## A. 完整的技术分类图

> 分类原则：**按"方法作用在哪一层、改的是什么对象"分**，而不是按论文标题分。每格标注 [L0–L4] 无损等级（见 §C.0）与该族在**消费级 batch=1** 下的主要作用轴。

```
高能力 LLM 的推理成本
│
├─ 0. 权重侧（静态，推理时不再变）
│   ├─ 0.1 低比特量化
│   │   ├─ 通用 PTQ：RTN / GPTQ / AWQ / SmoothQuant / OmniQuant / QuIK   [L4]
│   │   ├─ 旋转/不相干处理：QuaRot / SpinQuant / QuIP# / Hadamard      [L1]
│   │   ├─ 低比特标量：GSQ（Gumbel-Softmax 学网格+尺度）                [L4]  arXiv:2604.18556
│   │   ├─ 预算化混合精度分配：RCO（Riemannian 流形，精确满足总预算）    [L4]  arXiv:2605.00649
│   │   ├─ 格式家族：GGUF / MXFP4-8 / NVFP4 / EXL3 / oQ4 / RadixAttention
│   │   └─ 精确算术模拟（Ozaki / split-float / 3xTF32 / Kahan）        [L0/L1]
│   ├─ 0.2 异常值保护
│   │   ├─ 离群分解（LLM.int8()）/ 逐通道尺度（AWQ）/ 平滑（SmoothQuant）
│   │   └─ 注意力质量保持：QJL 残差修正（TurboQuant 的一环）            [L1]  arXiv:2406.03482
│   ├─ 0.3 低秩与分解
│   │   ├─ 激活感知 SVD / ASVD / LLM-Surgery 风格分解
│   │   └─ 投影共享：W_K = W_V（"Q-K=V"）→ KV 直接减半                [L4]  arXiv:2606.04032
│   └─ 0.4 结构化稀疏
│       ├─ N:M 半结构化（2:4）/ 非结构化（SparseGPT, Wanda）
│       ├─ 深度剪枝（ShortGPT, LayerDrop）/ 宽度剪枝 / head 剪枝
│       └─ 混合精度 × 剪枝：GSQ-RCO Coder（256/512 专家，1.89 bpw）   [L4]  [O]
│
├─ 1. KV cache 侧（随上下文增长）
│   ├─ 1.1 量化（压"每条"）
│   │   ├─ 逐通道/逐 token 非对称：KIVI                                [L4]
│   │   ├─ 校准型：KVQuant（pre-RoPE keys）                             [L4]  arXiv:2401.18079
│   │   ├─ 旋转 + 残差：TurboQuant = PolarQuant + QJL                  [L1]  arXiv:2504.19874
│   │   ├─ Hadamard + 方差归一 + K4/V2：KVarN                         [L1]  arXiv:2606.03458
│   │   └─ 向量量化系：CommVQ / VQKV / VecInfer / RotorQuant
│   ├─ 1.2 驱逐/选择（压"条数"）
│   │   ├─ 注意力分数：H2O / SnapKV / PyramidKV / Ada-KV / FastGen      [L4]
│   │   ├─ 流式与 sink：StreamingLLM
│   │   ├─ 位置稳定打分：TriAttention（pre-RoPE 三角）                  [L4]  arXiv:2604.04921
│   │   └─ 学习型压缩（native）：NSA 的压缩 token 分支                    [L4]  arXiv:2502.11089
│   ├─ 1.3 结构性共享（压"维度"）
│   │   ├─ 跨层共享：CLA / YOCO
│   │   ├─ 低秩潜变量：MLA（DeepSeek-V2/V3）                             [L4]  [O]
│   │   └─ 投影共享：W_K=W_V                                            [L4]  arXiv:2606.04032
│   ├─ 1.4 重建 / 重构（压"条数"但保留信息）
│   │   └─ landmark / 重建注意力 / 块级重算
│   ├─ 1.5 复用
│   │   ├─ 前缀 KV 复用（vLLM/llama.cpp/MLX prefix cache）
│   │   ├─ 跨模型 KV 转换（aither-kvcache 2.4.0 "cross-model KV transfer"）  [S]
│   │   └─ 分支/树解码的 KV 复用
│   └─ 1.6 分级存储
│       ├─ CPU/SSD 卸载（FlexGen / InfiniGen / LMCache / Pensieve）
│       ├─ 异步预取（AsyncTLS 的时间局部性预取）                        [S]
│       └─ 磁盘驻留的 n-gram 草稿表（Qwen3.8-Flash-Next 的 51.2B 参数分片） [O]
│
├─ 2. Attention 计算
│   ├─ 2.1 精确高效算法（不改函数）
│   │   └─ FlashAttention 系 / FlashDecoding / sliding-window+sink      [L0/L1]
│   ├─ 2.2 训练无关动态稀疏
│   │   ├─ 模式分类：MInference（vertical / vertical-slash / block）    [L4]
│   │   ├─ 逐查询稀疏：Quest / SpargeAttn
│   │   └─ 两级（块→token）：AsyncTLS                                    [L4]  [S]
│   ├─ 2.3 原生可训练稀疏
│   │   ├─ NSA（压缩 token + 选中块 + 滑窗，learned gate）              [L4]  arXiv:2502.11089
│   │   └─ DSA（DeepSeek 稀疏注意力）
│   ├─ 2.4 头/层稀疏与混合
│   │   ├─ DuoAttention（按头分检索/流式模式）                            [L4]
│   │   └─ MoBA / HeadKV
│   └─ 2.5 训练-推理一致的长上下文执行                                   [L4]  arXiv:2605.11744
│
├─ 3. FFN / MoE
│   ├─ 3.1 动态执行与跳过（token/专家/FFN 级）
│   ├─ 3.2 专家预测与预取（路由可预测性是前提）
│   ├─ 3.3 卸载与分层（CPU/SSD 专家）
│   ├─ 3.4 专家剪枝/合并/upcycling
│   └─ 3.5 原生稀疏/条件计算训练
│
├─ 4. 推理期控制
│   ├─ 4.1 推测解码（严格 L2 的唯一家族）
│   │   ├─ 独立草稿模型（Leviathan/Chen 修正拒绝采样）                    [L2]
│   │   ├─ 自推测（早退）：LayerSkip / Draft&Verify                     [L2]
│   │   ├─ 多头草稿：Medusa / EAGLE-2 / EAGLE-3                          [L2]
│   │   ├─ 树草稿：SpecInfer / Sequoia / Medusa 树                       [L2]
│   │   ├─ 无模型草稿：prompt-lookup / n-gram / REST                     [L2/L4]
│   │   └─ 原生 MTP 头（需特训 checkpoint）                              [L2]
│   ├─ 4.2 思考预算与提前停止
│   │   ├─ budget forcing / adaptive reasoning length
│   │   ├─ overthinking-marker logit 惩罚（免训练）                      [L4]  arXiv:2606.00206
│   │   └─ 熵/置信度门控
│   └─ 4.3 结果质量控制（自洽性、verifier、答案提取）
│
├─ 5. 架构替换（换 checkpoint）
│   ├─ 5.1 线性/循环注意力（Linformer/Performer 谱系、GLA、Mamba-2）
│   ├─ 5.2 混合 attention+SSM（Jamba / Zamba / MiniMax-01 / Qwen3-Next /
│   │      Qwen3.6 / Qwen3.8-Flash-Next / Nemotron 3 / granite-4.0-h）    [L4]
│   ├─ 5.3 扩散/掩码并行解码（Mercury / LLaDA / Dream）                  [L4]
│   └─ 5.4 低比特原生（BitNet b1.58/b2）                                 [L4]
│
├─ 6. 编译器 / 内核 / 数学等价变换
│   ├─ 6.1 算子融合（RMSNorm+quant、SwiGLU、attention+KV write）
│   ├─ 6.2 精度重排（fp32 累加、Kahan、split-float 模拟）
│   ├─ 6.3 权重布局与 kernel 特化（小 M 的 qmv、查表格式的解码代价）
│   ├─ 6.4 批不变内核（batch-invariant RMSNorm/matmul/attention）          [L1]
│   └─ 6.5 跨阶段中间结果复用（prefix cache、跨分支、跨候选）
│
└─ 7. 跨域可迁移算法
    ├─ 7.1 增量计算（incremental SVD/QR、rank-revealing update）
    ├─ 7.2 缓存理论（Belady/LRU、offline caching、cache-aware routing）
    ├─ 7.3 稀疏数值线性代数（低秩更新、稀疏 Cholesky/QR）
    ├─ 7.4 概率近似（sketching、count-min、ANN）
    └─ 7.5 在线/自适应算法（doubling、ski-rental、对偶拟合并）
```

**分类图之外我主动补上的、原始任务清单未列但重要的家族**：
1. **输出头 / 词表压缩**（见 §G.4）—— vocab 151936–248320 时它是每步字节数的一等公民，无人处理。
2. **确定性 / 批不变内核**（§6.4）——直接决定 §E 的"可靠性"能不能被测量。
3. **消融/归因方法学**：Token Inflation Ratio、escape mass 分解、CTIR——这些是**度量学**而非优化学，但在本领域比多数优化方法更有用。
4. **模型路由 / 级联**（判定为低相关，见 §G.5）。
5. **残差流架构变更（mHC）**——与本仓库原主题直接相关，见 §C.4。

---

## B. 代表性论文与源码证据索引

完整逐条索引见 **`research/notes/X1_evidence_index.md`**（含每条的确切 URL、核验日期、原文数字、来源类型与证据等级）。此处只给"必读清单"与关键数字。

### B.1 消费级硬件上的直接证据（最高优先级）

| 编号 | 来源 | 关键数字 | 等级 |
|---|---|---|---|
| B1 | MTPLX 实测（M4 Pro 48 GB, 2026-05-19）`[M]` | 27B Q4 带宽上限 ~7 tok/s；llama.cpp+MTP 10.5；**MTPLX 18.3（2.6×）**；depth-3 接受率 73/48/32 %；1.84 tok/cycle；16.2/18.6 GB | [M] |
| B2 | 本仓库 Stage 2.1 `[M]` | 881 MB/步；**decode 有效带宽 124.6 GB/s（= 273 GB/s 峰值的 46 %）**；prefill 有效算力 2.06 TFLOP/s；**每 eval 提交/同步成本 0.2–0.4 ms**（链内 0.253 vs 孤立 0.47–0.68 ms/Block）；堆权重 n=1 慢 29 %；深度 28→7 decode 收益 0。⚠ 原报的"0.558 ms/block 固定开销 / ~90 % 固定开销 / 1.46× 重叠上限"**已修正**，见 §0.2 | [M] |
| B3 | TurboQuant 作者澄清（Korea Herald 2026-04-13）`[S]` | 6× 与 8× 是**独立实验不可相乘**；8× 只算 attention logit，**端到端更接近 1.5–2×**；短对话 KV 占内存 <1 %，200K+ 才 >60 %；**首要目标是带宽** | [S] |

### B.2 KV cache 家族

| 方法 | 论文 | 关键数字 | 部署状态（消费级） |
|---|---|---|---|
| **TurboQuant** | arXiv:2504.19874 (ICLR'26) `[P]` | PolarQuant + 1-bit QJL；3.5 bit 绝对质量中性 / 2.5 bit 边际退化；LongBench Llama-3.1-8B **50.06 = FP16 50.06**，2.5-bit 49.44，KIVI 3-bit 48.50；NIAH 0.997 = FP16（KIVI 0.981，SnapKV 0.858）；≥6× KV 压缩；免训练免校准；≈2.7× Shannon 界 | **未合入** vLLM/llama.cpp/Ollama（2026-04-19）；社区实现全 CUDA/Triton；**无 Metal 后端** |
| PolarQuant | arXiv:2502.02617 (AISTATS'26) `[P]` | 旋转后坐标服从已知 Beta 分布 ⇒ Lloyd-Max 标量量化，去掉每块归一化元数据 | — |
| QJL | arXiv:2406.03482 (AAAI'25) `[P]` | 1-bit 符号草图修正残差，恢复**无偏内积估计** | — |
| KVQuant | arXiv:2401.18079 (NeurIPS'24) `[P]` | pre-RoPE keys、逐通道、非均匀类型、稠密+稀疏；**需校准** | — |
| **TriAttention** | arXiv:2604.04921 `[P]` | **AIME25 长推理 2.5× 吞吐、10.7× KV 压缩、精度 40.8 vs 40.8**；pre-RoPE 三角打分；32B agent 可在单张 24GB RTX 4090 上跑完多轮 | vLLM 插件（`pip install triattention`，`--triattention-budget 0.1`）；**非 Apple 友好** |
| **KVarN** | arXiv:2606.03458 `[P]` | Hadamard + Sinkhorn 式方差归一 + **K4/V2**；Qwen3-32B：3–5× 容量、**吞吐高于 FP16**、MATH500/AIME24/HumanEval 达 FP16 水平 | vLLM `kv_cache_dtype="kvarn_k4v2_g128"`，**免校准免改模型**；代码 `huawei-csl/KVarN` |
| **W_K=W_V 投影共享** | arXiv:2606.04032 (ICML'26) `[P]` | KV 直接减半，**PPL +3.1 %**；叠加 GQA-4 → 87.5 %，叠加 MQA → 96.9 %；W_Q=W_K=W_V 失败（Q/K 角色不对称） | **需从零训练** ⇒ 对冻结权重不适用 |

### B.3 权重侧与推理模型的相互作用

| 来源 | 关键数字 | 等级 |
|---|---|---|
| **GSQ** arXiv:2604.18556 `[P]` | Gumbel-Softmax 学 per-coordinate 网格 + per-group 尺度；2–3 bit 逼近向量量化；GGUF 可部署 | [P] |
| **RCO** arXiv:2605.00649 `[P]` | 把总尺寸预算重述为 logit 空间上的光滑 Riemannian 流形，**精确**满足预算且直接优化任务损失 | [P] |
| **GSQ-RCO 位宽扫描**（ISTA-DASLab 模型卡）`[O]` | BF16 16.00 bpw/354 GB/AIME25 100.00/GPQA-D 91.92/LCB v6 87.43/task-avg 93.12/ZS-recovery 100.0 % → **Q2_0 2.40 bpw/66.4 GB/96.67/89.39/81.14/89.07/101.4 %** → IQ2_XS 2.50/68.0/96.67/87.37/83.43/89.16/100.3 % → **IQ3_XXS 3.00/75.8/100.00/91.41/86.29/92.57/100.4 %** → **IQ3_S 3.50/83.6/100.00/92.93/86.86/93.26**。速度：Q2_0 **367.49 prompt t/s / 93.79 decode t/s / 6.70 s**，IQ2_XS 108.19 / 70.30 / 12.68 s | [O] |
| **Quantization Inflates Reasoning** arXiv:2606.25519 `[P]` | CTIR 276.1 %（Qwen3-4B AWQ-INT3）；**BBH 285 s(BF16) → 292 s(INT4)，尽管 W4A16 Marlin 有 1.2–1.4× 单 token 加速**；vLLM 模拟下 1.1× 膨胀 ⇒ p90 TTFT +33 %（+350 s）；QAT(DiscQuant) 68 %@9.5k vs BF16 73 %@8.5k | [P] |
| **Quantized Reasoning Models Think Longer** arXiv:2606.00206 `[P]` | **52 % 失败是"想对了但没输出"**；KL 峰值 ↔ next-token 熵；过量采样 "wait/but/alternatively"；**免训练 logit 惩罚：CoT −12–23 %，准确率持平或更好，过度思考错误 −58 %** | [P] |
| **FOCUS & RePAIR** arXiv:2608.26676 (ICML'26) `[P]` | 剪枝放大文本退化/重复环，**而 PPL 与任务准确率基本不变**；解码视作动力系统，分解为**进入风险**与**持续性**；持续性由采样集合中分给合理替代项的 **escape mass** 决定；FOCUS（高置信区蒸馏重加权）+ RePAIR（onset 中心正负对 + margin loss） | [P] |
| **Rethinking Perplexity** arXiv:2602.04099 `[P]` | PPL 被输入长度与评测协议的隐藏偏置系统性污染；提出 LengthBenchmark | [P] |

### B.4 注意力 / 稀疏

| 来源 | 关键数字 | 等级 |
|---|---|---|
| **NSA** arXiv:2502.11089 `[P]` | 压缩 token + 选中块 + 滑窗，learned gate；27B MoE 上 LongBench **0.469 > 全注意力 0.437**；9.0× 前向 / 6.0× 反向 / **11.6× 解码 @64K**；64K 每步读 ~5.6K 而非 65K 条 KV | [P] |
| **The Sparse Frontier** arXiv:2504.17768 `[P]` | 等算力下大稀疏模型打赢小稠密模型；解码耐压缩优于 prefill（32B/72B 撑 17×）；**7B 从 16K 的 12× 掉到 128K 的 5×**；**几乎每个配置都有任务掉到 5× 以下** | [P] |
| **MInference 1.0** (NeurIPS'24) `[P]` | 按头分类 vertical / vertical-slash / block-sparse；**仅 prefill** | [P] |
| **训练-推理一致片段级执行** arXiv:2605.11744 (ICML'26) `[P]` | 固定 KV 尾部做唯一可微接口 + TBPTT（**K=1 最优**）+ 只读检索前缀 + 头/层稀疏长程头；PG19 PPL ≈ 全上下文（LLaMA2-32K@32K≈5.0；LLaMA2-80K@64K≈4.8）；**128K prefill 峰值内存 ≈ FlashAttention 的 1/6**（FA-2 @64K ≈80 GB，Chunked Prefill >60 GB，本文 <15 GB） | [P] |

### B.5 架构

| 模型 | 配置（已核验 `[O]`） | KV 字节/token | 消费级可跑？ |
|---|---|---|---|
| **Qwen3.6-27B** | `qwen3_5`；64 层 = 48 linear_attention + 16 full_attention；hidden 5120；24 head/head_dim 256/4 KV head；dense；**`mtp_num_hidden_layers: 1`**；vocab 248320；262144 ctx | **64 KB**（全注意力同宽模型为 256 KB，**1/4**） | ✅ 4-bit 约 16 GB `[M]` |
| **Qwen3-Next-80B-A3B** | 48 层 = 36 linear + 12 full；hidden 2048；16 head/head_dim 256/2 KV head；512 experts top-10；81.32 B 参数；262144 ctx | 24 KB | ❌ 4-bit ≈ 40 GB，无 KV 余量 |
| **Qwen3.8-Flash-Next** | `qwen4_exp`；48 层 = 36 linear + 12 full；hidden 2560；24 head/head_dim 256/2 KV head；512 experts top-10；**原生稀疏 indexer（budget 2048, compress_ratio 4, indexer_head_dim 128）**；`hc_count 4 / hc_lowrank 320`；**PLE（layer_ids [2]）**；**原生 3-gram 预测器（vocab_base 20 M, heads_per_ngram 8）**；**MTP 1 层**；**vocab 248320**；另有 **51.2 B 参数的 per-layer n-gram embedding 表**（独立分片，IQ4_NL 28.8 GB，磁盘 mmap） | 24 KB + 查表 | ❌ 即使 1-bit 也 72.5 GB `[S]` |
| **granite-4.0-h-1b** | `granitemoehybrid`；40 层 = **36 Mamba-2 + 4 attention**；hidden 1536；12 head/4 KV head；`mamba_d_state 128 / n_heads 48 / d_head 64 / chunk 256`；vocab 100352；tied embedding；131072 ctx | **8 KB**（全注意力同宽为 80 KB，**1/10**） | ✅ 消费级混合架构，有发布权重 |
| **granite-4.2-8b** | `GraniteForCausalLM` —— **4.2 已回到纯注意力**（40 层/hidden 4096/32 head/8 KV head/vocab 100352） | 80 KB | ✅（但非混合） |
| **Nemotron 3 Nano** arXiv:2512.20848 `[P]` | 52 层 = **6 attention (GQA, 2 组) + 23 Mamba-2 + 23 MoE(128 专家, 6 激活)** → 30B 模型每 token ~3.5 B 激活；厂商称 H200 上 8K-in/16K-out 为 Qwen3-30B-A3B 的 3.3×、1M 上下文 RULER-100 86.3 % | 极小 | 30B 总参，需 4-bit ≈ 15 GB，勉强 |

### B.6 编译器 / 内核 / 确定性

| 来源 | 关键内容 | 等级 |
|---|---|---|
| **Defeating Nondeterminism in LLM Inference**（Horace He, Thinking Machines Lab, 2025-09-10, DOI 10.64434/tml.20250910）`[P]` | 根因是**缺乏 batch invariance**（内核随 batch/切分切换 split-K / stream-k / tile / 指令 ⇒ 归约顺序变），**不是**"并发+浮点非结合性"（前向 pass 固定 shape 下是 run-to-run 确定的）。Qwen3-235B-A22B-Instruct-2507 temp=0 跑 1000 次 ⇒ **80 个不同 completion**，前 102 token 相同、第 103 token 分叉（"Queens, New York" vs "New York City"）；batch-invariant 内核后 **1000/1000 完全一致**。性能：1000×100 token，vLLM 默认 **26 s** → 未优化 deterministic **55 s** → 改进 attention 内核 **42 s**（≈1.6–2.1× 代价）。修法细节：batch-invariant RMSNorm（单核内完成每样本归约）、matmul（禁 split-K，单一 tile 配置，~20 % 吞吐代价）、attention（**固定 split-size 而非固定 split 数** 的 Split-KV + 调用前统一 KV page table，需改 FlexAttention 内部）。代码 `thinking-machines-lab/batch_invariant_ops` | [P] |
| BIO / TBIK（Zhang et al., 2025-11-21）`[S]` | BIO 单独把 12 个 (TP×batch) 配置的唯一输出数降到 ~7–8、最大概率偏差降到 0.006–0.02，**加 TBIK 才到 1 / 0**；**BIO 单独带来 18–99 % 端到端延迟开销** | [S] |
| **BaseRT（Metal 原生运行时）** arXiv:2607.00501 `[P]` | 绕过 MLX / llama.cpp 的框架抽象，**直接写 Metal**；8 种量化格式（Q2–FP16）；Qwen3 / Llama 3.2 / Gemma 4 家族，Q4/Q8，**M3 与 M4 Pro**；**decode 吞吐最高 1.56× llama.cpp、1.35× MLX**；MoE 的 prefill 差距更大；覆盖 sub-1B 到 30B | [P] |
| **BaseRT（M5 张量核）** arXiv:2607.19438 `[P]` | M5 每核一颗 Neural Accelerator，经 **Metal 4 tensor API** 暴露；手写 cooperative-tensor kernel（dense/MoE GEMM、flash-attention prefill）；**M5 Pro 上 prefill 最高 6.4× llama.cpp、3.9× MLX**；**decode 路径明确留在原有专用 kernel 上（1.02–1.33×）** | [P] |
| **BaseRT 的反面评注** `[S]` | 第三方结构化评注独立判定：prefill 增益归因于张量核，但**缺消融、基线是默认配置** ⇒ "归因欠支撑"；同时它自己记录了核验等级仅 T0–T2 | [S] |
| **SGLang on Apple Silicon**（官方文档）`[P]` | 官方有 `docs/hardware-platforms/apple_metal.mdx`：**`SGLANG_USE_MLX=1`** + `--disable-cuda-graph`（文档原文："CUDA graph 与 Apple Metal 无关"）+ 可选 `--disable-overlap-schedule`；前置条件 **macOS 14+、PyTorch 2.13.x、MLX ≥0.32.0**；`--quantization mlx_q4/mlx_q8` 走 `mlx_lm.utils.quantize_model`（group 64）。⚠ **本机 MLX 是 0.31.2，不满足该前置条件** | [P] |
| **SGLang 状态的来源冲突** `[S]` | 第三方 2026-07 评测称"SGLang 无官方 Apple Silicon 支持，只有 main 分支上的实验性 MLX 路径（pre-alpha / roadmap #19137）"，并称其 **5.3× 增益是相对 SGLang 自己的 PyTorch-MPS 路径**、样本很小 ⇒ **该 5.3× 不是对 mlx-lm 的比较，不能用来判断端侧引擎选择** | [S] |

### B.7 组合先例（见 §D 展开）

| 组合 | 出处 | 状态 |
|---|---|---|
| 量化 × 推测解码（互补方案） | **QSpec** arXiv:2410.11305 (EMNLP'25) `[P]` | 已发表并开源；≤1.64× vs 高精度基线；≤1.55× vs SOTA spec-decode（**batched**） |
| 量化 × 推测解码（KV 侧） | NeurIPS 2026「Memory-Efficient Speculative Decoding with Quantized Draft KV Cache」`[S]` | 已发表 |
| 渐进式推测量化 × 长上下文 KV 传输 | **Lynx** arXiv:2607.01831 `[S]` | 已发表 |
| 自推测 × 分层 KV | **QuantSpec** `[S]` | 已发表 |
| 量化 × 混合精度 × 专家剪枝 | **GSQ-RCO Coder**（256/512 专家，1.89 bpw，58.4 GB）`[O]` | 已发布权重 + 推理/代码 benchmark |
| KV 量化 × KV 驱逐 | `aither-kvcache` 2.0.0/2.4.0 `[S]`；ruvLLM 文档给出 30–50× 堆叠目标 `[S]` | **实现存在，但无同行评审的联合测量** |
| 原生 MTP × 混合注意力 | Qwen3.6-27B `[O]`；Qwen3.8-Flash-Next `[O]` | 单 checkpoint 内原生共存 |
| 块扩散 × 推测草稿 | `incoai/Qwen3.8-Flash-Next-DFlash2`（block-diffusion draft，sglang）`[O]` | 已发布模型 |

---

## C. 数学与适用条件对照

### C.0 无损等级（本报告统一口径）

| 等级 | 定义 | 判据 |
|---|---|---|
| **L0 精确算术等价** | 变换后的计算在实数域上完全相同（仅允许代数重排） | 逐位或代数相等 |
| **L1 有限精度数值等价** | 精确算术下相同，有限精度下末位不同 | 差异仅为舍入/累加顺序 |
| **L2 采样分布保持** | 输出分布可证相同 | 精确的接受/拒绝机制 + 残差修正 |
| **L3 能力近似保持** | 改变模型函数，靠训练/校准把能力拉回来 | 质量为实测恢复 |
| **L4 有损** | 显式改变函数/分布，质量只靠经验测量 | 任何"无损"说法都需被降级 |

**几处必须点名的降级**（这些是文献里"lossless"用词的常见漏洞）：

1. **QSpec 的 "without quality degradation"** `[P]` —— 参照系是**权重量化后的基线**，不是原始 bf16 模型；其验证路径本身是 L3/L4（相对原模型），只是相对目标分布 L2。其接受规则是 **greedy top-1 匹配 + 拒绝后从高质量分布重采样**，**不是** Leviathan/Chen 的精确修正拒绝采样，因此**只在 greedy 极限下 L2**。
2. **TurboQuant 的 "3-bit zero loss"** `[S]` —— 论文口径是 **3.5 bit 绝对质量中性、2.5 bit 边际退化**；"3 bit" 是标题简化 `[S]`。同族的 SnapKV 在 NIAH 上只有 **0.858**（对照 FP16/TurboQuant 0.997）——**"驱逐"与"量化"不是同一类损失，驱逐是结构性的、量化是数值性的**。
3. **bf16 下的累加重排不是 L0** —— 任何"重排求和顺序 ⇒ 数值等价 ⇒ 无损"的说法只到 **L1**。这正是本仓库 Stage 1 测到的（bf16 与 fp32 累加最大差 0.0099 nats，结构误差比舍入误差大 3 个数量级）`[M]`。
4. **"免费"推理加速几乎不成立**：TurboQuant 作者自述 8× 注意力 ⇒ 端到端 1.5–2×；AsyncTLS 算子 10× ⇒ 端到端 1.3–4.7×。

### C.1 逐方法：数学对象 / 省什么 / 需要什么 / 适用哪一段

> 省的是哪一栏，直接决定它在消费级 batch=1 下有没有价值。

| 方法 | 数学对象（改了什么） | 省 FLOPs | 省容量 | 省带宽 | 省串行轮次 | 需要预训练/校准/草稿模型 | 适用冻结权重 | Prefill / Decode / 长上下文 / batch=1 |
|---|---|---|---|---|---|---|---|---|
| **MTP 推测解码** | 用 checkpoint 自带 MTP 头从已算出的最后隐状态草拟 k 个 token；一次前向并行验证；按接受规则接受/重采样 | ✗ | ✗ | **✓（每 token 分摊权重流）** | **✓（k+1 token / 1 次权重流）** | **需特训 checkpoint** | ✗ | ✗/✓✓/✓/✓✓ |
| 独立草稿模型推测解码 | 同上，草稿来自另一个模型 | ✗ | ✗ | ✓ | ✓ | 需第二模型常驻 | ✓ | ✗/✓✓/✓/✗（内存贵） |
| 自推测（早退） | 跑前 L 层草拟，再跑全模型验证 | ✗ | ✗ | ✓ | ✓（需 >L/N 接受率才赚） | 无 | ✓ | ✗/✓/✓/✓ |
| **TurboQuant** | 随机正交旋转 R → 坐标服从已知 Beta → Lloyd-Max 标量量化（b−1 bit）+ **1-bit QJL 符号残差** ⇒ 内积估计无偏 | ✓(注意力 logit) | **✓** | **✓** | ✗ | **免训练、免校准** | ✓ | ✓/✓/✓✓/✗ |
| **KVarN** | 通道维 Hadamard 旋转 + 迭代方差归一（Sinkhorn 式）跨量化 tile + **K4/V2 非对称** | ✓ | **✓** | **✓** | ✗ | **免校准、免改模型** | ✓ | ✓/✓/✓✓/✗ |
| **TriAttention** | 在 **RoPE 之前**的 K 上做三角打分（此时 Q/K 聚在固定中心，注意力≈位置的三角函数）⇒ 位置稳定的重要性 | ✗ | **✓✓** | ✓ | ✗ | 无 | ✓ | ✗/✓/✓✓/✗ |
| KV 驱逐 (H2O/SnapKV/…) | 用注意力分数保留 heavy hitter + 最近窗口 | ✗ | ✓ | ✓ | ✗ | 无（部分需校准预算） | ✓ | ✗/✓/✓✓/✗ |
| **W_K=W_V 共享** | 让 key/value 投影矩阵相同 ⇒ 缓存条目少一半 | ✓ | **✓** | ✓ | ✗ | **需从零训练** | **✗** | ✓/✓/✓/✗ |
| MLA | 把 KV 潜变量化到低维再上投影 | ✓ | **✓✓** | ✓ | ✗ | **需特训** | ✗ | ✓/✓/✓✓/✗ |
| **NSA / DSA** | 训练时就有：压缩 token 分支 + 选中块 + 滑窗，learned gate 混合 | **✓✓** | ✓ | ✓ | ✗ | **需原生训练** | ✗ | ✓✓/✓✓/✓✓/✗ |
| **训练无关动态稀疏** (MInference/Quest/AsyncTLS) | 按查询动态选块/选 token | **✓✓** | ✓ | ✗ | ✗ | 无（部分需 per-head 校准） | ✓ | ✓✓/✓/✓✓/✗ |
| **训练-推理一致片段执行** | 固定 KV 尾部作唯一可微跨片段接口 + TBPTT(K) + 只读检索前缀 + 头/层稀疏长程头 | ✓ | **✓** | ✓ | ✗ | **需训练** | ✗ | ✓✓/✗/✓✓/✗ |
| **MoE 专家卸载 + 预取** | 只流过被选中的专家权重；靠路由可预测性提前预取 | ✗ | ✓ | **✓（只流激活专家）** | ✗ | 需路由可预测 | ✓ | ✗/✓✓/✗/✗ |
| **MoE 专家跳过** | 某些 token 不算 FFN | **✓** | ✗ | **✗（同 batch 其他 token 仍要读）** | ✗ | 无 | ✓ | ✓/✗/✗/✗ |
| **权重量化** (GPTQ/AWQ/GSQ/RCO) | 权重低比特 + 尺度/网格；RCO 还精确满足总预算 | ✓ | **✓** | **✓✓** | ✗ | RTN 免校准；GPTQ/AWQ 需校准；RCO 需任务损失 | ✓ | ✓✓/✓✓/✗/✓✓ |
| **格式选择**（Q2_0 vs IQ2_XS） | 同样位宽，不同解码实现（查表 vs 直算） | ✗ | 略差 | **✓✓（实测 1.9× 端到端）** | ✗ | 无 | ✓ | **✓✓**/✓✓/✗/✓✓ |
| **低秩 / SVD 分解** | W ≈ UV | ✓ | ✓ | ✓ | ✗ | 需校准 | ✓ | ✓/✓/✗/✓ |
| **深度剪枝** | 去掉某些 block | ✓ | ✓ | **✓** | ✓ | 通常需恢复训练 | 部分 | ✓/✓/✗/✓ |
| **思考长度控制 / 早停** | 减少生成 token 数 | ✗ | ✓（KV） | **✓（按比例）** | **✓✓** | 无 | ✓ | ✗/✓✓/✓/✓✓ |
| **overthinking-marker logit 惩罚** | 对 "wait/but/alternatively" 等加 logit 惩罚 | ✗ | 间接 | 间接 | **✓（CoT −12–23 %）** | **免训练** | ✓ | ✗/✓✓/✗/✓ |
| **批不变内核** | 固定归约顺序、禁 split-K、固定 split-size | ✗ | ✗ | ✗ | ✗ | 无 | ✓ | ✓/✓/✓/✓（batch=1 天然满足） |
| **扩散并行解码** | 一次前向出整段，8–16 步精修 | ✓ | ✗ | ✓ | **✓✓** | 需特训 | ✗ | ✗/✓✓/✗/✗ |
| **混合 attention+SSM** | 部分层用线性递推代替注意力 | ✓ | **✓✓（KV 降 4–10×）** | ✓ | ✗ | 需特训 | ✗ | ✓/✓/✓✓/✓ |

### C.2 消费级 batch=1 的算术（把方法翻译成"每 token 搬多少字节"）

以 **Qwen3.6-27B 4-bit（~16 GB）** 在 **M4 Pro 48 GB（~120 GB/s）** 为具体算例 `[M][O]`：

| 组件 | 每 token 字节 | 说明 |
|---|---|---|
| 27B 权重 @4-bit | ~16 GB | 一次前向全部流过；这是 decode 的主项 |
| 输出头（vocab 248320 × 5120 × 2 B） | ~2.5 GB (bf16) / ~0.64 GB (4-bit) | **常被忽略的一等公民** |
| KV @ 16k context（64 KB/token） | ~1.0 GB | 混合架构把这个数压到同宽度全注意力模型的 1/4 |
| KV @ 128k context | ~8.4 GB | 此时 KV 才开始和权重同量级 |
| **总计 @16k** | **~17.6 GB** | 理论上限 120/17.6 ≈ **6.8 tok/s**（与 B1 实测 ~7 吻合） |
| 推测解码 depth-3，1.84 tok/cycle | ~9.6 GB/输出 token | ⇒ 上限 ~12.5 tok/s，**实测 18.3**（说明还有格式/内核红利） |

**这张表就是本报告的判据表**：任何方法，如果它不改变"每输出 token 搬多少字节"或"要串行几次"，它在最后一行就原地不动。

#### 长上下文的拐点（**§G.1-G1 与 §C.3 的必要限定**）

上表里有一行必须被单独拎出来：**@16k 时 KV 只有 1.0 GB，而权重是 16 GB——KV 占 6 %。这就是"短上下文下 KV 压缩无用"的算术根据。但它有拐点：**

| 上下文 | KV 大小（64 KB/token） | 与 16 GB 权重之比 | 主导项 |
|---|---|---|---|
| 16k | ~1.0 GB | 6 % | **权重** |
| 32k | ~2.0 GB（实测口径 **3.5 GiB**，含增长峰值） | 12–22 % | **KV 与权重接近** |
| 128k | ~8.4 GB | 52 % | **KV 主导** |

⇒ **在 32k 附近，KV 从"次要项"变成"与权重同量级"，此时 KV 量化从"加速手段"升级为"准入条件"**（不是"跑得更快"，而是"跑得起来"）。

⚠ **两个必须同时记住的限定**：(i) 本项目硬件实测显示 decode 的**有效带宽只有峰值的 46 %**，所以 KV 带宽节省的**端到端转化率低于理论值**；
(ii) 在 mlx-lm 上量化 KV 还要付 §D.6 那条**每 token 56 次小 kernel** 的调度代价。

⚠ **一个容量硬上限（本轮读 `mx.device_info()` 实测）**：`max_buffer_length = 14,302,248,960 B = 13.32 GiB`、
`max_recommended_working_set_size = 19,069,665,280 B = 17.76 GiB`、`resource_limit = 499,000`。
**单个 Metal buffer 的上限 13.32 GiB 小于 17.76 GiB 的建议工作集**——128k 上下文下 KV 单块约 14 GiB 就已经**超过单 buffer 上限**。
这解释了为什么"分页 KV"在端侧不是优化而是**必需**。

#### 推测解码的精确成本账

比常用的简化式 `(W + W_d)/(a+1)` 更精确的形式是：

```
每轮成本 = n·T_d + T_t(n+1)
每轮产出 = a + 1 个 token（a = 平均接受数，n = 草稿长度）
⇒ 加速比 = (a+1) / ( n·T_d/T_t + (n+1) )
```

其中 `T_d` = 一次草稿前向，`T_t` = 一次目标前向（吃掉 n+1 个 token）。
**关键点：`T_t(n+1) ≈ T_t`**，因为 decode 是带宽受限而非算力受限——多喂 n 个 token 几乎不增加目标前向的时间。

⇒ **收益完全由"草稿有多便宜 × 接受率多高"决定，与目标模型的深度无关。** 由此得到两个可直接检验的判据：

- **同尺寸草稿模型**（`T_d ≈ T_t`）：加速比 ≈ `(a+1)/(2n+1)`，**需要 100 % 接受率才打平**。
- **1/10 尺寸草稿**（`T_d ≈ 0.1·T_t`）：加速比 ≈ `(a+1)/(1.1n+1)`，**70 % 接受率、n=3 ⇒ ≈2.1×**（与 `[M]` MTPLX 实测 2.6× 同量级）。

⇒ **"更大的 n（草稿更深）"在没有高接受率时是净亏损**，因为 `n` 在分母里被计了两次。
⚠ 与 `[M]` MTPLX 实测的**唯一口径差异**：该实测报告的是 **1.84 token/cycle**（即 a≈0.84），不是"接受率 84 %"——两者不可混用。

### C.3 哪些方法在消费级"根本不触发"

- **KV 压缩（量化/驱逐/共享）在 context < 4k 时几乎无用**：KV 只占内存 <1 %（作者自述），且完全不触碰权重带宽。`[S]` ⚠ **但这个结论有明确的失效区间**：按 64 KB/token 核算，**32k 起 KV 与权重同量级、128k 时 KV 主导**（见 §C.2 拐点表）——此时它不是「无用」，而是**准入条件**。
- **MoE 专家卸载在模型装得下时是负优化**：把已经在统一内存里的权重搬到 PCIe 路径上只会更慢。
- **FlashAttention 类"精确高效算法"在 decode 侧收益小**：它们优化的是 prefill 的 IO 与 FLOPs；decode 的瓶颈是权重流。
- **batch=1 时"continuous batching / 调度类"优化基本不适用**。
- **KV 驱逐类方法在 4 bit 权重 + 短上下文下没有容量压力可解。**

### C.4 与本仓库原主题的交叉：残差流与 mHC

本仓库研究的是残差流与跨层依赖。**2025 年 12 月残差流本身发生了一次架构级变更**，且有 2026 年的后续工作：

- `[P]` **mHC: Manifold-Constrained Hyper-Connections**（DeepSeek）—— **arXiv:2512.24880**（https://arxiv.org/abs/2512.24880）。把残差混合矩阵 `H_res` 投影到 **Birkhoff 多胞形（双随机矩阵）**上，用 **Sinkhorn–Knopp** 迭代实现；数学上保证信号传播范数有界（≤1）与凸组合性质，从而恢复恒等映射稳定性。系统侧用 TileLang 算子融合 + 分块重计算 + 扩展的 DualPipe 通信重叠，把 **n=4 扩展率下的训练开销控制在 6.7 %**；在 27B 规模上显著降低梯度范数波动、消除 loss 尖峰，并在多个下游任务上优于 baseline 与 HC。
  前作：`[P]` **Hyper-Connections**（ByteDance Seed）—— **arXiv:2409.19606**，把残差流宽度扩展 n 倍并引入 `H_pre / H_post / H_res` 三个可学习线性映射：`x_{l+1} = H_res^l x_l + H_post^{l,T} F(H_pre^l x_l, W_l)`。
- `[S]` 二手复述的数字：复合增益从 **3,000× 压到 1.6×**、相对标准残差 **+7.2 BBH / +6.9 DROP / +7.1 GSM8K**、训练开销 +6.7 %。**机制部分已升为 `[P]`，这些具体 benchmark 数字仍为 `[S]`。**
- **是否已被超越？——是，2026 年有两篇直接跟进：**
  - `[P]` **Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections** —— **arXiv:2603.20896**（Zhaoyi Liu, Haichuan Zhang, Ang Li；UMD + Utah）。指出 Birkhoff 多胞形约束的**三个局限**，首先是 **identity degeneration**。
  - `[P]` **mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations** —— **arXiv:2601.05732**（Yongyi Yang, Jianyang Gao）。指出 mHC 的两个局限：(i) **有限次 SK 迭代并不保证精确双随机**，留下的近似误差会**随网络深度累积**；(ii) 高效 SK 实现需要**高度定制的 CUDA kernel**，工程门槛高。
  - 社区实现（非官方）：https://github.com/thefirehacker/mHC-manifold-constrained-hyper-connections

**对本项目的直接意义**：跨层依赖/残差混合矩阵**可以被代数约束成良态**，而不必重新发明 DAG；Stage 3 若要动残差流结构，**双随机性约束是有先例、有理论、低成本的稳定化手段**（与本仓库 Stage 1.5 观察到的"早层失败是方向错误不是幅度错误"高度相关）。**但 mHC 本身已被两篇 2026 跟进工作部分超越**，且其 CUDA 定制内核的工程门槛说明：**在 Metal 上没有现成实现**——这又一次落到"消费级不可用"的一类里。

---

## D. 组合兼容性矩阵

### D.1 判读规则

1. **"都有效"不等于"组合有效"。** 必须区分三种状态：**已发表联合测量** / **已实现但无联合测量** / **仅工程推断**。
2. 每格必须标注**共同假设**、**潜在冲突**、**互相抵消的开销**。
3. 对消费级 batch=1，还要额外过 §C.2 的判据（是否改变字节数或串行轮次）。

### D.2 主矩阵

图例：`✔✔` 有同行评审的联合测量 · `✔` 有已发布实现/官方组合但**无联合测量** · `△` 仅工程推断（可组合但有明确冲突） · `✘` 不适用/互斥 · `—` 无关

| | 权重量化 | KV 量化 | KV 驱逐 | 稀疏注意力 | MTP 推测解码 | 混合 attn/SSM | 思考长度控制 |
|---|---|---|---|---|---|---|---|
| **权重量化** | — | `✔` KVarN `[P]`、QSpec `[P]`、GSQ-RCO Coder `[O]` | `✔` 互补（不同轴） | `✔` | `✔✔` **QSpec** `[P]`（≤1.64× vs 高精度基线） | `✔` Nemotron/GSQ-RCO `[O]` | `△` **强冲突**（见 D.3-①） |
| **KV 量化** | `✔` | — | `✔` aither-kvcache `[S]`、ruvLLM 30–50× 目标 `[S]`（**无联合测量**） | `✔` | `✔` NeurIPS'26 量化草稿 KV `[S]`、Lynx `[S]` | `✔` MLA 型 | `△` |
| **KV 驱逐** | `✔` | `✔` | — | `△` **双过滤冲突**（见 D.3-③） | `△` 接受率可能受损 | `△` | `△` |
| **稀疏注意力** | `✔` | `✔` | `△` | — | `△` 草稿与验证用不同注意力模式 | `✔` 原生 indexer `[O]` | `△` |
| **MTP 推测解码** | `✔✔` | `✔` | `△` | `△` | — | `✔✔` **Qwen3.6/Qwen3.8 原生共存** `[O]` | `✔` 减少 CoT 轮次 |
| **混合 attn/SSM** | `✔` | `✔` MLA | `△` | `✔` | `✔✔` | — | `△` |
| **思考长度控制** | `△` | `✔` CoT KV 是长上下文内存主项 | `△` | `△` | `✔` | `△` | — |

### D.3 四个必须点名的冲突 / 抵消

**① 权重量化 × 思考长度控制 —— 强冲突，且方向已知。**
`[P]` arXiv:2606.25519：INT4/INT3 量化让推理模型 CoT 变长（CTIR 最高 276.1 %），**在 BBH 上把 285 s 变成 292 s**，尽管单 token 有 1.2–1.4× 加速。`[P]` arXiv:2606.00206 给出机制与对冲手段：量化模型在 KL 峰值位置（高 next-token 熵）过量采样 overthinking marker；**免训练的 marker logit 惩罚把 CoT 压 12–23 % 且准确率不降**。
⇒ **组合建议**：量化 + overthinking-marker 惩罚，**不要**单独用"关掉思考模式"这种粗粒度做法（`[M]` MTPLX 文章里"reasoning off 提速 ~7 % 且多数任务无质量损失"是**未经支撑的质量主张**，本报告不接受）。
⚠ **未解**：arXiv:2606.25519 的作者明确写明，**KV 量化与激活量化对 token 膨胀的复合效应"remains unexplored"**——这正是消费级最可能同时开启的两项（4-bit 权重 + 2–3 bit KV）。**这是本领域最明确、最可做、且直接命中本项目目标的开阔空白。**

**② 推测解码 × 批大小 —— 方向相反。**
`[S]` 行业实践指南明确：spec decode 的延迟收益**随并发上升而缩小**（推测偏爱低并发）。这对"消费级单用户"是**有利**的（batch=1 正是 spec decode 的最佳工况），但对"端侧多用户"是**不利**的。
`[P][S]` 另一条线：批不变内核的引入会把 spec decode / dynamic batching 的确定性收益抵消——因为它们恰恰依赖"同一请求被放进不同 batch"。⇒ **端侧 batch=1 反而天然免疫这类不确定性**（见 §0.4-6）。

**③ 两种"重要性判据"叠加 —— 双过滤冲突。**
`[S]` 2026 组合分析明确点出：TriAttention（pre-RoPE 三角打分）与 AsyncTLS（块级过滤）都减少注意力看到的 token 数，但**判据不同**；AsyncTLS 的块打分可能恰好丢掉 TriAttention 因三角指标高而保留的 token。**没有公开测量。** 这类冲突的通用形式是：**两个各自正确的选择器，串联后不等于各自的交集，而可能是各自的差集。**

**④ "容量收益"与"带宽收益"不是同一件事 —— 抵消的根源。**
`[S]` TurboQuant × TriAttention 在**容量**上可乘（60–100×），但：TurboQuant 减少**每条**的字节，TriAttention 减少**条数**；而在统一内存设备上，**KV 的实际带宽成本由残留项是否被稠密读取决定**——如果驱逐留下的是散布的残片，page 粒度开销会主导，此时"每条更小"几乎不省时间。
**这正是本仓库在另一个方向上实测到的同一形状** `[M]`：A5 显示 4 个 block 权重堆成一个矩阵乘是**逐位等价**的数学，却在 n=1 时**慢 29 %**；A2 显示 8 个独立 block 的重叠上限只有 **1.06×**（⚠ 原报 1.46× 的计时区间混入了 k 次 `embed` + k 次 `mx.eval`，按 A1 独立口径重算；且 A2 用 n=64，不是 decode 的 n=1）。**⇒ 数学上的可组合性与硬件上的可组合性是两回事，而 2026 年的文献基本只论证了前者。**

### D.4 已被明确否证的"组合"（重要）

- **KV 家族 × 扩散解码：✘ 不适用。** `[S]` 扩散模型不用自回归式 KV cache（它每步处理整段），TurboQuant / TriAttention / AsyncTLS 都是为自回归流水线设计的，**不能直接套用**。这不是"效果差"，是"对象不存在"。
- **KV 家族 × 消费级带宽瓶颈：✘ 方向错。** `[M][S]` 两个独立来源都明确说 KV 压缩"不解决限制解码速度的权重带宽瓶颈"。
- **"把 6× 内存和 8× 注意力相乘得到 48×"：✘ 作者本人已否定。** `[S]`

### D.5 运行时互斥矩阵（**源码级**，`mlx-lm 0.31.3` / `MLX 0.31.2`，本轮亲自复核）

> §D.2 的矩阵是**方法层面**的可组合性；这一节是**本项目实际要用的运行时**（mlx-lm）层面的**硬互斥**。两者冲突时以这一节为准——因为它决定"这条组合在本仓库根本跑不起来"。
> 核验方式：对 `mlx_lm.models.cache` 做**运行时内省**（`inspect.getsource` 逐类读取 `is_trimmable`），并直接读 `generate.py` / `server.py` 的分支代码。**不是靠行号推断。**

| 组合 | 能否共存 | 源码依据（mlx-lm 0.31.3） |
|---|---|---|
| KV 量化 + 推测解码 | **✔ 可共存** | `QuantizedKVCache.is_trimmable()` → `True`（`models/cache.py:306`），trim 语义与 `KVCache` 一致 |
| KV 量化 + 连续批处理 | **✘ 互斥** | `_make_cache` 的 `to_batch_cache` 对未知类型落到 `else: raise ValueError(f"{type(c)} does not yet support batching")`（`generate.py:857`）；**且不存在 `BatchQuantizedKVCache` 类**（本轮全模块 `grep` 确认） |
| KV 量化 + 滑动窗口 | **✘ 互斥** | `RotatingKVCache.to_quantized()` 直接 `raise NotImplementedError("RotatingKVCache Quantization NYI")`（`models/cache.py:551-552`） |
| 连续批处理 + 推测解码 | **✘ 互斥** | `server.py:371` `is_batchable = draft_model is None`——一旦配 draft 就退回单请求串行 `generate_step` |
| 连续批处理 + `RotatingKVCache(keep>0)` | **✘ 互斥** | `to_batch_cache` 中 `if c.keep > 0: raise ValueError("RotatingKVCache with keep tokens is not supported.")`（`generate.py:851-852`） |
| 前缀缓存（trie/LRU）+ 滑动窗口 | **△ 部分** | `RotatingKVCache.is_trimmable()` 返回 `self.offset < self.max_size`（`cache.py:542-543`）——**绕回之后即不可 trim**，"更长前缀 → trim 成公共前缀"的路径失效 |
| 前缀缓存 + 推测解码 | **✔ 可共存** | 两者都只要求 cache 可 trim，前缀缓存不改变 trim 语义 |

**推论（对本项目的直接含义）**：**"KV 量化 + 连续批处理 + 滑动窗口 + 推测解码"这四项在 mlx-lm 里两两不兼容。** 端侧 batch=1 本来就用不上连续批处理，所以真正咬人的是 **"KV 量化 + 滑动窗口"**：若目标 checkpoint 是 Gemma-3 / Mistral 这类滑窗模型，KV 量化这条线直接不通。

**一条比文献更硬的限制（本轮源码级确认）**：**推测解码对 SSM/Recurrent 状态不是"不适用"，而是直接报错。**
`generate.py:529-532` 的 `speculative_generate_step` 在进入草稿循环前检查 `cache.can_trim_prompt_cache(model_cache)`（`cache.py:91-92`），不通过就
`raise ValueError("Speculative decoding requires a trimmable prompt cache (got {types}).")`。
而承载 SSM / 循环状态隐状态的 **`ArraysCache` 没有自己的 `is_trimmable`**，继承 `_BaseCache` → **返回 `False`**（本轮运行时内省确认）。
⇒ **对纯 SSM 或 SSM 占多数的混合 checkpoint，mlx-lm 的推测解码路径不可用**；对混合 checkpoint，**只要有一个 SSM 子缓存，整个模型缓存就不可 trim**（见下表）。

### D.6 状态类型 × 能力矩阵（**源码级**，本轮运行时内省逐类确认）

| 状态 / 缓存类型 | 代表模型 | 连续批处理 | KV 量化 | 推测解码（trim） | 说明 |
|---|---|---|---|---|---|
| **MHA / GQA**（每层 K/V） | Qwen3、Llama、Qwen3.6-27B 的 16 个 full-attention 层 | `✔` `BatchKVCache` | `✔` `QuantizedKVCache` | `✔` 可 trim | 功能最完整的一条路 |
| **滑窗 / 局部注意力** | Gemma-2/3、Mistral | `✔`（但 `keep>0` 禁止） | `✘` `NotImplementedError` | `△` 绕回后不可 trim | **量化与滑窗二选一** |
| **Recurrent / SSM**（定长隐状态） | Mamba-2、Gated-DeltaNet、granite-4.0-h-1b 的 36 层 | `✔`（`filter`/`extend`） | `✘` 无"量化状态"概念 | `✘` **不可 trim ⇒ 推测解码直接 raise** | 状态不随长度增长，但**不可回退** |
| **Hybrid**（`CacheList`） | Qwen3-Next、Qwen3.6-27B、granite-4.0-h-1b | `✔` 逐子 cache 转发 | `△` 仅 KV 子缓存部分 | `✘` **任一子缓存不可 trim ⇒ 整体不可 trim** | `CacheList.is_trimmable()` = `all(c.is_trimmable() for c in self.caches)`（`cache.py:821-822`） |

**这张表直接改写 §F.2 的可行性判断**：Qwen3.6-27B 的 64 层里 **48 层是 linear attention**。若这 48 层用 `ArraysCache`（线性注意力 / SSM 状态），则**整个模型缓存不可 trim** ⇒ **mlx-lm 的 `speculative_generate_step` 会直接抛异常**。
这正好解释了 `[M]` MTPLX 为什么必须自己实现 "**GDN linear-attention kernel with innovation tape replay for deterministic rollback**"——**那不是性能优化，那是让不可回滚的状态变得可回滚的必要机制。**

**并且，量化 KV 在 mlx-lm 上不是"免费 4×"**：`QuantizedKVCache.update_and_fetch` 每一步都对**当步新增的 token** 调用 `mx.quantize(keys)` / `mx.quantize(values)`（`models/cache.py:277-278`），
28 层 × K/V ⇒ **每 token 56 次小 kernel**；写入 packed uint32 布局是**标量式 scatter**而非连续追加，并**绕开融合 SDPA**。
⇒ **KV 量化的真实收益 = 容量 ÷ (访存收益 − 量化/打包开销)**，在短上下文、batch=1 下净收益可能接近 0。

### D.7 引擎的三层可移植性判断（对本项目的直接结论）

把 vLLM / SGLang / llama.cpp / MLX 的成果搬到 Apple Silicon 上，要分三层看：

| 层 | 能否借 | 说明 |
|---|---|---|
| **调度与内存管理**（radix cache、分页、连续批处理、预算分配） | **✔ 能借** | 这些是**架构无关的算法**，不依赖 CUDA 指令路径 |
| **张量核与量化布局**（GEMV 特化、packed 格式、查表 kernel） | **✘ 不能借** | 依赖目标指令集（PTX / CUDA 内核），必须用 Metal 重写；`[P]` BaseRT 的全部价值正是"绕过框架抽象直接写 Metal" |
| **数值恒等式**（batch-invariance、量化数学、RoPE 等价变形） | **✔ 能借** | 数学是数学，但**核的归约顺序**仍需在目标后端重新实现（§B.6） |

---

## E. 已解决的问题与仍存在的实际限制

### E.1 2024–2026 已经**解决**的问题（有证据）

| 问题 | 解决状态 | 证据 |
|---|---|---|
| 低比特 KV 能不能在几乎不掉质量的前提下实现 | **是**（3.5 bit 绝对质量中性、2.5 bit 边际退化） | `[P]` arXiv:2504.19874 |
| KV 量化能不能完全免校准 | **是**（TurboQuant 数据无关；KVarN 免校准免改模型） | `[P]` 两篇 |
| 能不能只改训练、不改推理就拿到等质量的长上下文 | **是**（训练-推理一致片段级执行，PPL ≈ 全上下文，128K prefill 内存 ≈ FA 的 1/6） | `[P]` arXiv:2605.11744 |
| 原生稀疏注意力能不能打赢全注意力 | **能**（NSA 在 LongBench 0.469 vs 0.437） | `[P]` arXiv:2502.11089 |
| 混合架构能不能把长上下文 KV 压一个数量级 | **能**（Qwen3.6-27B 64 KB/token；granite-4.0-h-1b 8 KB/token） | `[O]` 配置核算 |
| 推测解码能不能在消费级拿到 2×+ 且保持分布 | **能**（MTPLX 2.6×，L2 接受+残差修正） | `[M]` |
| 推理的不可复现性从哪来 | **已定位**（缺乏 batch invariance，非并发/原子加） | `[P]` tml.20250910 |
| 压缩为什么会破坏输出而不掉 PPL | **已定位**（解码动力学：进入风险 × escape mass 决定的持续性） | `[P]` arXiv:2608.26676 |
| 量化为什么让推理模型变慢 | **已定位**（token 膨胀，CTIR 最高 276 %） | `[P]` arXiv:2606.25519 |

### E.2 仍然存在的实际限制（按对消费级目标的严重度排序）

> ⚠ **在使用本节之前，先看清本仓库自身测量的口径上限**（采纳自同仓库并行审计，本轮已复核）：
> 1. **本仓库的绝对毫秒数不可跨报告引用**——同一段 28-block 代码在 Stage 2.0 记 61.32 ms、Stage 2.1 记 116.92 ms，**相差 1.9×**，来源未定位（候选：热节流、分配器状态、编译标志的全局副作用）。**在定位之前，只有比率与"有效带宽 / 有效算力"这类归一化量可用。**
> 2. **三项证据空白从未被测量**：**能耗 / 温度**（消费级设备的真实约束）、**生成质量**（本仓库全部是数值等价与性能测量，**没有一次质量评测**）、**长上下文 TTFT**。而 §0.4-5 已经证明质量失败在 PPL / 准确率上不可见——**所以"本仓库实测"目前只能支撑"性能"结论，不能支撑"质量无损"结论。**
> 3. **A2 的重叠上限是 1.06× 不是 1.46×**（计时区间混入 `embed` + `mx.eval`），且用的是 n=64 而非 decode 的 n=1。

1. **【最严重】Token 膨胀把量化加速吃掉，且只有 QAT 能根治，QAT 不适用于冻结权重。** 唯一可用的对冲（免训练 marker logit 惩罚）只覆盖 overthinking 这一类失败，**没有覆盖 KV 量化 / 激活量化的复合效应**（作者自述未探索）。
2. **【严重】最好的 KV 方法在 Apple Silicon 上不存在可用实现。** TurboQuant 未合入任何主流框架且无 Metal 后端；TriAttention 是 vLLM 插件。**这不是研究问题，是工程问题**——但它决定了很多论文的结论在目标设备上无法落地。
3. **【严重】"FLOPs 节省 → 端到端加速"的换算在文献里被系统性高估。** 至少三个独立来源给出 1.5–2×、1.3–4.7×、0%（本仓库）的实际落差。
4. **【重要】没有任何公开测量报告"KV 驱逐后的残留访问局部性"在统一内存设备上的表现。** 这直接决定 TurboQuant × TriAttention 的 60–100× 容量收益能否变成延迟收益。
5. **【重要】输出头/词表几乎无人研究。** vocab 248320 时输出头 bf16 约 2.5 GB/步，是 27B 4-bit 权重的 15 %；**没有任何一篇主要论文以它为目标**。
6. **【重要】质量评估体系跟不上方法学。** SEQUOR（`[P]` arXiv:2605.06353, COLM'26）测出 50 轮约束跟随的准确率衰减（单约束 >11 %，多约束 >40 %，中途增改 >9 %），但**其自身评审指出"论文没有分析 persona 一致性、约束顺序、逐轮施加规则如何影响模型行为"**——也就是说，连"角色一致性"这个轴的测量工具本身都在起步阶段。而**没有任何一篇高效推理论文用过 SEQUOR 或任何同类多轮一致性基准**。
7. **【中等】训练无关稀疏方法在小型模型上更早失效**（7B：16K 12× → 128K 5×）`[P]`，这与"用小模型做端侧"的直觉直接冲突。
8. **【中等】"方法是否在消费级硬件上跑得起来"几乎没有被当作论文的评价维度。** 只有 TurboQuant 作者明确区分了"注意力 logit 加速"与"端到端"，以及 `[M]` MTPLX 作者明确指出 Metal 后端缺失。
9. **【中】消费级实测样本极少。** 本轮找到的、真正在 Apple Silicon 上做的端到端推理优化测量只有 1 个（MTPLX），加上本仓库的 1 个（M4 Pro / MLX）。**本领域绝大多数数字来自 H100/A100。**
10. **【已核，2026-10-21 补】mHC 已定位到一手来源**（arXiv:2512.24880），并发现两篇 2026 跟进工作（arXiv:2603.20896、arXiv:2601.05732）**部分超越它**。详见 §C.4。
11. **【重要】在本项目实际要用的运行时里，一批"方法层面可组合"的组合是"运行时层面互斥"的。** 见 §D.5：KV 量化 × 连续批处理、KV 量化 × 滑动窗口、连续批处理 × 推测解码、连续批处理 × `RotatingKVCache(keep>0)` 全部互斥；**SSM/Recurrent 状态不可回退 ⇒ 推测解码在纯 SSM 与多数混合模型上直接抛异常**。这是**工程事实**而非研究问题，但它决定了 §F.2 的路径必须**自带回滚机制**，不能指望现成运行时。
12. **【中】消费级运行时本身在快速演进，且门槛正在抬高。** SGLang 的官方 Apple Silicon 路径要求 **MLX ≥0.32.0 + PyTorch 2.13.x**，而本机是 **MLX 0.31.2**；BaseRT 的 M5 张量核路径对本机（M4 Pro 无 NAX）**不可迁移**，而它在 M4 Pro 上的那篇（2607.00501）报的是 **decode 1.56×/1.35×** 而非 prefill 的 6.4×。⇒ **"等框架支持"是一个会持续失配的赌注。**

### E.3 本轮未能核实的清单（诚实列出）

- TriAttention 论文里"**H2O/SnapKV/PyramidKV 在 10 % 预算下 AIME25 掉约一半准确率**"这句对比的确切措辞与模型条件（我核到的是 `[S]` 二手复述，arXiv:2604.04921 的摘要页抓取超时，未打开全文）。
- AsyncTLS 的一手论文（我只核到 `[S]` 二手解读，未找到其 arXiv id / 官方仓库）。
- Mercury / Mercury 2 的官方页面与 AIME25 91.1、1,009 tok/s 的原始出处（仅 `[S]`）。
- QSpec 之外更多"KV 量化 × 推测解码"的联合测量（已知 NeurIPS'26 一篇，仅核到会议页标题与摘要片段）。
- QuantSpec / Lynx 的原始论文页。
- mHC 的 benchmark 具体数字（+7.2 BBH 等）仍为 `[S]` 二手；机制与论文 id 已是 `[P]`。

---

## F. 值得进一步调查的五个组合机会

> 每个机会都写明：**为什么可能有用** / **为什么可能失败** / **如何以最小代价证伪**。按"证伪成本低、可操作性高"排序。

---

### F.1 原生 MTP 推测解码 × 低比特权重 × 免训练 overthinking-marker logit 惩罚

**为什么可能有用。** 三者分别打三个不同的轴，且都已有独立证据：
- MTP 打**串行轮次**：`[M]` M4 Pro 上 2.6×（7→18.3 tok/s），且 L2 严格保分布；模型侧 `[O]` Qwen3.6-27B 原生带 MTP 头。
- 低比特权重打**带宽**：`[O]` 4-bit ≈ 16 GB（27B），相对 bf16 减半。
- marker 惩罚打**token 膨胀**：因为量化会引入膨胀 `[P]`（CTIR 276 %），这个惩罚是**唯一免训练、且被同一篇论文证明与量化组合后不降准确率**的对冲（CoT −12–23 %，5 模型 × 3 量化 × 5 benchmark）。
三者的失败模式是**已知且不同**的：MTP 接受率、量化膨胀、惩罚误伤。组合后至少可以**解耦测量**。

**为什么可能失败。**
- **正交性可能只是表面的**：如果 marker 惩罚降低了 CoT 的**熵**，MTP 的接受率理论上会**上升**（草稿与目标更一致），这很好；但反过来，惩罚若把"wait/but"这类**必要的自我纠错**也罚掉，会破坏长推理，可能同时降低接受率（草稿质量下降）并降低准确率。**两者的相互作用方向不确定**，而且是"好则更好、坏则双倍坏"的结构。
- **marker 惩罚是 L4**：它改分布。任何"无损"说法都不成立。
- **MTP 接受率在长 CoT 上可能崩**：公开的接受率数据（73/48/32 %）是在 27B **非推理模式**（`--reasoning off`）下测的。**在真正需要长思考的提示上，接受率没有任何公开数据。** 这是最大的未知。
- **冻结权重冲突**：本项目若坚持"已冻结权重"，MTP 整条路被排除。

**如何以最小代价证伪。** 在单个 checkpoint（Qwen3.6-27B 4-bit，MLX/MTPLX）上做 2×2×2 因子实验：{MTP on/off} × {marker 惩罚 on/off} × {温度 0/0.6}，指标用 **AIME25 + LiveCodeBench + token 数 + 每 token 接受率**（而不是 PPL），并把 `[M]` 的 6.14–7.15 ms/step 基线作为固定参照。**成本：单机、单模型、数小时。**

---

### F.2 原生 MTP × 混合注意力/SSM checkpoint —— 以及"循环状态回滚"这个未解问题

**为什么可能有用。** 混合架构把 KV 压 4–10×（`[O]` 配置核算：Qwen3.6-27B 64 KB/token；granite-4.0-h-1b 8 KB/token），这在 24–48 GB 设备上直接决定了"能开多长上下文"。MTP 则是唯一 L2 的串行轮次优化器。二者**在 Qwen3.6-27B 这一个 checkpoint 里已经原生共存** `[O]`——这是"组合先例"最强的形式：不是两个人拼出来的，是**厂商训进去的**。

**为什么可能失败。**
- **【源码级硬约束，本轮亲自核验】在 mlx-lm 上，这条路不是"难"，而是"默认不通"。** `speculative_generate_step`（`generate.py:529-532`）在进入草稿循环前检查 `can_trim_prompt_cache`，不通过直接 `raise ValueError`；而承载 SSM / 循环状态隐状态的 `ArraysCache` **没有自己的 `is_trimmable`**，继承 `_BaseCache` → `False`（本轮运行时内省确认）。**混合模型更糟**：`CacheList.is_trimmable()` = `all(c.is_trimmable() for c in self.caches)`（`cache.py:821-822`）⇒ **任一子缓存不可 trim，整个模型缓存就不可 trim**。**推论：Qwen3.6-27B 的 64 层里有 48 层是 linear attention，它能否走 mlx-lm 的现成推测解码路径，取决于那 48 层用的是什么缓存类——这是必须先查清的第一个事实，而不是先去写内核。** 完整矩阵见 §D.6。
- **循环状态回滚是真正的技术障碍**：Mamba-2/Gated-DeltaNet 的状态是递推的，验证阶段如果拒绝了草稿 token，**状态必须回滚**。`[M]` MTPLX 的实现里明确写了需要一个 "**GDN linear-attention kernel with innovation tape replay for deterministic rollback**"——这说明**回滚不是免费的**，而是一个需要专门内核、且**正确性难以验证**的机制。在 batch=1 解码下，任何回滚误差会**逐步累积**（不像 attention 的 KV 可以整段丢弃重建）。
- **MTPLX 的实测数据里没有"混合架构 + MTP"的隔离测量**：它的 2.6× 是在一个 27B 上取得的，而混合层的存在本身可能改变接受率（线性注意力层的隐状态分布与全注意力不同）。
- **Qwen3.6-27B 是稠密模型**；`[M]` 文章推测 MoE（35B-A3B）会更快但**只是推测**。

**如何以最小代价证伪。** 用**同一提示集**分别测：(a) 纯注意力 checkpoint + MTP（若可得）与 (b) 混合 checkpoint + MTP，比**接受率**而非 tok/s（tok/s 混入了权重流差异）。再单独做**回滚正确性测试**：构造一个已知被拒绝的草稿序列，验证回滚后的隐状态与"从未草拟过"的状态在 fp32 下 bitwise 相等。**这一项不需要训练，只需要内核级的等价性测试。**

---

### F.3 KV 量化 × pre-RoPE 驱逐：**在统一内存上先把"访问局部性"测出来**

**为什么可能有用。** 这是全语料里唯一一个**"容量乘法"看起来最大、而"带宽结论"最不确定**的组合。两个组件都是免训练、免校准、作用于不同轴（每条字节 vs 条数），`[S]` 已有工程实现（aither-kvcache 2.0.0/2.4.0，声称 TurboQuant 3.8–7.1× + TriAttention ~10×，并含跨模型 KV 转换），`[S]` ruvLLM 文档甚至给出了 30–50× 的堆叠目标。**但没有任何同行评审的联合测量，也没有任何人在统一内存设备上测过它。**

**为什么可能失败。**
- **最可能的原因恰恰是本项目已经踩过的那一类**：在统一内存上，驱逐后的残片若不连续，实际搬运的是"页"而不是"条目"，于是"每条从 16 bit 降到 3 bit"几乎不省时间，而"每条更小"还额外引入了 dequant 开销。**本仓库的 A5 提供了这条陷阱的直接同构证据**（逐位等价的堆权重矩阵乘在 n=1 慢 29 %）。
- **两个组件都只有 CUDA 实现**；要在 Metal 上跑，PolarQuant 的旋转 + Lloyd-Max 查表 + QJL 残差 sketch 都要重写为 Metal kernel，**这是一笔真实的工程债**，而且查表格式在 Metal 上的 decode 成本可能正是 GSQ-RCO 那个"Q2_0 比 IQ2_XS 快 1.9×"现象的来源（查表代价与内容相关）。
- **TriAttention 的收益与推理相关**（AIME25），而 KV 驱逐在**长推理**场景里可能与"思考链的引用需求"冲突。

**如何以最小代价证伪。** **不需要 GPU 实验就能先做一半**：在 M4 Pro 上用 MLX 直接测"驱逐后残片的实际读取字节数 vs 理论条目字节数"——即 page/对齐粒度的浪费比。这是一个**纯访存测量**，可以在几小时内给出 yes/no。**这是本清单里我最有信心的一条**，因为它直接连接本仓库已有的 Stage 2.1 测量传统（真实 bench，不用 FLOPs 折算）。

---

### F.4 把"训练-推理一致"当作**设计准则**而不是推理技巧：面向消费级的新 checkpoint 配方

**为什么可能有用。** `[P]` arXiv:2605.11744 给出了本领域最干净的因果解释：**推理期后处理式稀疏之所以掉质量，是因为训练/推理执行语义不一致**，而不是稀疏本身有害；把片段级执行变成训练与推理共用的建模假设后，**PPL 与全上下文几乎重合**，同时 128K prefill 峰值内存只有 FlashAttention 的约 1/6。这条原则可以**同时**套用到 2026 年所有原生能力上：混合注意力（Qwen3.6 类）、原生 MTP 头、原生稀疏 indexer（Qwen3.8-Flash-Next 的 `indexer_budget 2048`）、原生 3-gram 预测器。**这些能力都是训练进去的；如果它们在推理时的执行语义与训练时不一致，就会以"看起来莫名的质量下降"的形式付账。**

**为什么可能失败。**
- **直接违反"已冻结权重"约束**——这条路必须**从零/继续训练**，本轮明确不做。
- **那篇论文只在语言建模（PPL）上验证**，`[S]` 其局限是"未在长文档 QA、精确撤销上下文、复杂推理等场景验证"；**PPL 相等 ≠ 推理能力相等**。而本报告 §0.4-5 已经说明 PPL 是不可信的代理量。
- **TBPTT K=1 最优**的结论是在 PG19（连续书籍文本）上得到的，**对"精确回忆"类任务可能不足**（论文自己指出 TBPTT K=1 在需要跨长距离细粒度对齐的任务上可能不够）。

**如何以最小代价证伪。** **不需要训练**：拿一个已经训好的混合/稀疏 checkpoint（如 Qwen3.8-Flash-Next 的 `indexer_budget 2048`），检查其**训练时的注意力执行语义与推理时的执行语义是否一致**（配置层面即可判定：训练时是否也用 budget 2048 的 indexer、是否也用同样的滑窗/压缩比）。若不一致，则该 checkpoint 的 indexer 正在承受 F.1 中未知的代价。**这是一个纯静态核对，成本极低。**

---

### F.5 "格式优先"而不是"位宽优先"的量化选择 × 专家剪枝 × 磁盘驻留的草稿表

**为什么可能有用。** `[O]` GSQ-RCO 的位宽扫描给出了一个反直觉且**可操作**的结论：**Q2_0（2.40 bpw，66.4 GB）比 IQ2_XS（2.50 bpw，68.0 GB）小，decode 却是 93.79 vs 70.30 t/s（1.34×），端到端延迟 6.70 s vs 12.68 s（1.9×）**，原因不是位宽而是**查表格式的解码代价随内容变化**（Q2_0 跨类别 92.5–94.5 t/s，IQ2_XS 49.4–94.5 t/s）。在统一内存 + Metal 上，这个"查表代价"同样要付。加上：(i) 专家剪枝 × 量化（GSQ-RCO Coder：256/512 专家，1.89 bpw）[O]；(ii) 草稿表分级存储（Qwen3.8-Flash-Next 把 **51.2 B 参数**的 n-gram 表做成独立分片，IQ4_NL 28.8 GB，`-lm mmap --lazy-mode on` 放磁盘）[O]。三者共享同一个优化准则：**在统一内存设备上，访问模式与格式的解码代价比位宽更重要。**

**为什么可能失败。**
- **Q2_0 的 1.9× 是在 llama.cpp 下、很可能不是 Metal 测的**；Metal 上的查表/反量化代价可能完全不同。**这是必须重测而不是外推的。**
- **磁盘驻留草稿表的随机行读是最差的访问模式**：`per_layer_token_embd` 每 token 读一行（8 个 head 各一行），在 SSD 上是随机读。**它可能比它加速的解码还贵。** 这一条我在 §G.2(b) 已经标记为"被吸收进模型的免模型草稿的隐藏成本"。
- **RCO 的"精确预算"在 GGUF 上被格式表达力卡住**（不能按专家分配，只能按层）[O]——**这是"方法受限而非方法不够好"的范例**，在 Metal/自定义格式上这个约束可能不同。
- **2.4 bpw 已经是推理质量的悬崖**：AIME25 100→96.67、LCB 87.43→81.14（`[O]`），**而这还没算 token 膨胀**。

**如何以最小代价证伪。** 在同一台 M4 Pro 上，用**同一模型**分别以 Q2_0 与 IQ2_XS 跑 llama.cpp 与 MLX，测 decode t/s 与端到端延迟；再测"草稿表放 SSD vs 放 RAM"的差值。**不需要新方法，只需要同一台机器上的同模型对比。** 这一条同样不需要 GPU 租约。

---

## G. 反面证据与不能迁移到消费级硬件的结果

### G.1 硬性反面证据（来自原始来源，且指向本项目的目标设备）

| # | 反面证据 | 来源 | 等级 |
|---|---|---|---|
| G1 | **"KV 压缩不解决消费级 decode 的约束"**——KV 只占内存 <1 %（**短对话**），且不触碰权重带宽。⚠ **必须限定在短上下文**：按 64 KB/token 核算，**32k 起 KV 就与权重同量级、128k 时 KV 主导**，此时 KV 量化从"加速手段"升级为**准入条件**（见 §C.2 的拐点表） | TurboQuant 作者自述 + MTPLX 作者独立复述 + 本轮配置核算 | [S]+[M]+[推算] |
| G2 | **最好的 KV 方法在 Apple Silicon 上不可用**：未合入任何主流框架；社区实现全 CUDA/Triton；**无 Metal 后端** | turbo-quant 状态页 + MTPLX 作者 | [S]+[M] |
| G3 | **量化可能让端到端变慢**：Qwen3-4B 在 BBH 上 INT4 292 s > BF16 285 s，尽管单 token 有 1.2–1.4× 加速 | arXiv:2606.25519 | [P] |
| G4 | **压缩会破坏输出而 PPL/准确率看不见**（重复环） | arXiv:2608.26676 (ICML'26) | [P] |
| G5 | **52 % 的量化失败是"想对了但没输出"** —— 准确率指标会把它记成"答错" | arXiv:2606.00206 | [P] |
| G6 | **算子加速 → 端到端加速的落差被系统性高估**：1.5–2×（TurboQuant 作者）、1.3–4.7×（AsyncTLS）、0 %（本仓库深度压缩） | 三处独立 | [P][S][M] |
| G7 | **6× 与 8× 不可相乘**（作者本人明确否定） | The Korea Herald 2026-04-13 | [S] |
| G8 | **训练无关稀疏在小模型、长上下文上更早失效**（7B：12×→5×）；**几乎每个配置都有任务掉到 5× 以下** | arXiv:2504.17768 | [P] |
| G9 | **推理期后处理式稀疏有训练/推理不一致的根本缺陷** | arXiv:2605.11744 (ICML'26) | [P] |
| G10 | **深度/依赖结构压缩在本仓库目标设备上零收益**（decode 28→7：0 %；深度 7 最好也只有 3.7 % 且在噪声内） | 本仓库 Stage 2.1 | [M] |
| G11 | **"堆权重做批量矩阵乘"在 decode 尺度是负优化**（逐位等价却慢 29 %） | 本仓库 Stage 2.1 A5 | [M] |
| G12 | **批不变内核有 18–99 % 端到端延迟开销**（BIO 单独），完整确定性需 TBIK | Zhang et al. 2025-11-21 | [S] |
| G13 | **2026 旗舰混合模型装不进消费设备**（Qwen3.8-Flash-Next 即使 1-bit 也 72.5 GB；80B MoE 4-bit ≈ 40 GB 无 KV 余量） | GSQ-RCO 模型卡 + 第三方内存拆解 | [O][S] |
| G14 | **端侧可用的混合架构在能力上落后一代**（granite-4.0-h-1b 是 1B；granite-4.2 已回退纯注意力） | HF 配置核验 | [O] |

### G.2 不能迁移到消费级硬件的结果（按原因归类）

**(a) 硬件不兼容（无后端）**
- TurboQuant / PolarQuant / QJL 的所有官方与社区实现（CUDA/Triton）。
- TriAttention（vLLM 插件；vLLM 不在 Mac 上运行）。
- KVarN 的 vLLM 集成（`kv_cache_dtype="kvarn_k4v2_g128"`）。
- AsyncTLS 的异步 KV 卸载引擎（以 PCIe 为前提；Apple 统一内存没有 PCIe 这一层）。

**(b) 前提在端侧不成立**
- 所有"模型装不下 ⇒ 卸载到 CPU/SSD"的方案（MoE-Infinity / FlexGen / InfiniGen / LMCache / Pensieve）：**在 16–48 GB 设备上，模型通常装得下**，卸载是纯负优化。
- 所有依赖 **batch>1** 的收益（连续批处理、spec decode 的高并发版、paged attention 的吞吐收益）。
- 所有依赖 **多卡/NVLink/NCCL** 的结果（TP 不变性、NCCL all-reduce 确定性）。
- 依赖 **H100 特有格式**（NVFP4 + Tensor Core 指令路径）的加速数字——Apple GPU 无对应指令路径。
- 依赖**磁盘容量充足**的"51.2 B 参数 n-gram 表驻磁盘"方案：它在 72.5 GB 磁盘上可行，在消费级笔记本的 SSD 上仍可行，但**随机行读是 SSD 的最差模式**，且与"统一内存"设计哲学冲突。

**(c) 目标硬件上已被本仓库实测否证**
- 跨层依赖深度压缩（Stage 2.0/2.1）。
- 层级 block 并行（重叠上限 **1.06×**，按 A1 口径重算；原报 1.46× 的基线被 `embed`+`mx.eval` 污染）。
- 堆权重 batched matmul 作为启动开销解法（n=1 慢 29 %）。

**(d) 只在 H100/A100 上成立、与本项目目标无关**
- FlashAttention-4 类 IO 感知内核在 **prefill** 上的收益。
- 注意力 logit 计算的 8× 加速（TurboQuant，H100 4-bit）。

### G.3 本报告刻意不做的三件事

1. **不把"两个方法都有效"写成"组合有效"。** §D 里每一个"✔"都注明了是"有联合测量"还是"仅有实现"还是"仅工程推断"。
2. **不把论文标题当成结论。** TurboQuant 的"3-bit zero loss"降级为 3.5 bit；QSpec 的"without quality degradation"标注了参照系；AsyncTLS 的 10× 降级为 1.3–4.7×。
3. **不把"推理期质量"当成"任务能力"。** §0.4-5 与 §C.0 的 L 级分类贯穿全文。

### G.4 明确指出的研究空白（文献里没有，但工程上是一等公民）

**输出头 / 词表压缩。** 全库唯一相关的严肃线索是 `[S]` "Embeddings Are Overpriced"（用冻结的 512-bit 词表码替换 Gemma 3 270M 的 168 M 参数嵌入表）与词表裁剪方案 `[S]`。**没有找到同行评审文献。** 而核算显示：Qwen3.6-27B 的输出头 bf16 约 **2.5 GB/步**（27B 4-bit 权重的 15 %），Qwen3.8-Flash-Next 的 vocab 248320 更大。
**这是一个"文献少但成本大"的家族**，且它作用于**带宽轴**（本项目最在意的轴）。⚠ 该判断基于配置核算与两篇博客，**尚未找到任何论文做过头部量化的正式工作**——这本身就是一个"值得先做"的结论。

### G.5 明确判定为低相关（不是"没用"，是"目标不匹配"）

**级联 / 模型路由（FrugalGPT、RouteLLM、零样本分类器路由）。** 该文献的目标函数是 **API 每请求成本**，不是端侧延迟；而且级联要求**两个模型同时驻留**，这恰是 16 GB 设备做不到的。唯一合理的变体是"一个小模型常驻 + 一个大模型磁盘/冷备"，但那已经变成"选择性加载"，与本项目的其他议题正交。**判定：不值得在后续调查中投入。**

---

## H. 方法论、证据分级与可复现说明

### H.1 本轮的做法
1. **先定判据，再查文献**：先用本仓库 Stage 2.1 的实测把"消费级 batch=1 的约束轴"定死（容量 / 带宽 / 串行轮次 / FLOPs），再用它去筛所有方法——避免被论文的 speedup 数字牵着走。
2. **2025–2026 的东西一律打开原始页面**：本轮核验的 2026 一手来源包括 arXiv abs 页、ICML/COLM/NeurIPS 会议页、HF `config.json` 与模型卡、官方 GitHub、以及作者本人对媒体的澄清（Korea Herald）。所有未能打开的一律标 `UNVERIFIED` 或降级为 `[S]`。
3. **区分"论文声称"与"独立验证"**：报告里所有百分比与速度数字都标了来源类型；`[P]` 只表示"论文/官方页面"，**不表示已被独立复现**。特别地，GSQ-RCO 的位宽扫描（`[O]` 官方模型卡）与 2606.25519 / 2606.00206 的关键数字，**目前只有作者侧报告**；`[S]` 的 Pith 结构化评注对 2606.25519 的独立等级标注为 **T0（仅 AI 阅读，未重算）**。
4. **组合分开记三种状态**：已发表联合测量 / 已实现无测量 / 仅工程推断。
5. **负面结果优先**：用户明确要求"宁愿发现没有可行组合，也不要制造创新点"，因此 §F 全部写成"可证伪的机会 + 失败条件 + 最小证伪成本"，而不是"新方法提案"。

### H.2 产物清单

| 文件 | 内容 |
|---|---|
| `research/REPORT.md` | 本报告 |
| `research/PROTOCOL.md` | 研究协议（统一判据、无损等级定义、输出格式） |
| `research/notes/X0_crosscutting.md` | 交叉核验笔记：硬件约束模型、推测解码成本模型、2026 一手来源的详细数字、组合分析与矛盾点、**第 5 节：第二轮新增的运行时互斥发现** |
| `research/notes/X1_evidence_index.md` | 证据索引：逐条 URL / 核验日期 / 原文数字 / 来源类型 / 证据等级；**N 组：第二轮核验的 mlx-lm 源码事实、`mx.device_info()` 硬件事实、BaseRT 两篇、SGLang 官方文档** |
| `research/tools/compact_search.py` | 检索结果紧凑提取器（仅本地工具，无副作用） |
| `research/tools/apply_taskB_corrections.py` | **本轮对报告的 14 处修正的可复现脚本**（每处替换带断言，未命中即中止且不写文件） |
| `research/tools/fix_xrefs.py` | 5 处交叉引用修正脚本（§0.3-N → §0.4-N 的编号错指） |

### H.3 边界声明
- 未运行 GPU 实验、未训练、未下载模型、未修改 `results/` 与 `scripts/`、未启动 Stage 3、未安装任何插件。
- 新增文件全部位于 `research/`（工作树中为未跟踪状态），**未提交、未推送**。
- 仓库原有的两个未跟踪文件（`docs/M1_LITERATURE_MECHANISM_REVIEW.md`、`docs/STAGE3_NATIVE_DAG_DESIGN_DRAFT.md`）**未被本轮触碰**。

### H.4 建议的下一步（供与其它研究者交叉审查时讨论，本轮不执行）
1. 先做 **F.3 的访存局部性测量**（纯访存，无 GPU 租约，结论可证伪）。
2. 再做 **F.5 的同机同模型格式对比**（Q2_0 vs IQ2_XS × llama.cpp vs MLX）。
3. 把 **§E.3 的待核清单**（mHC、TriAttention 一手对比句、AsyncTLS 原论文、Mercury 官方页）补齐。
4. **讨论是否接受"**冻结权重是硬约束**"这一前提**——因为 §0.4-4 的判断是：若接受，可及的能力上限就被冻结在 checkpoint 里，研究就只剩延迟工程；若不接受，F.1/F.2/F.4 才打开。**这是一个关于研究方向的决策，不是技术决策。**

### H.5 与同仓库并行审计（"任务 B"）的关系 —— 以及本报告的修正记录

本仓库中同时存在两份由**不同研究者 / 不同会话**产出的文档，二者**不是同一件工作**：

| 文件 | 归属 | 与本报告的关系 |
|---|---|---|
| `research/REPORT.md`（本文） | **本轮（任务 A）** | 独立完成的全景研究：10 个方向 + 遗漏家族扩展，A–G 七项交付 |
| `docs/INFERENCE_SYSTEM_AUDIT.md` / `docs/INFERENCE_SYSTEM_AUDIT_SUMMARY.md` | **同仓库并行任务 B**，**非本轮所写** | 系统 / 运行时层的源码级审计，产出了本报告初稿**没有**的源码事实 |

**本报告采纳了什么，以及是否自己复核：**

| 审计结论 | 本报告处理 |
|---|---|
| A2 的 "1.46× 重叠上限"基线被 `embed`+`mx.eval` 污染 | **采纳并复核**：重算为 **1.06×**；已在 §0.2、§0.3-4、§B.1、§D.3-④、§G.2(c) 五处同步修正 |
| "0.558 ms/block 固定开销"是外推；"解码 90 % 固定开销"在 n=1 恒真 | **采纳**：§0.2 已改写为三条更稳健的锚点（有效带宽 124.6 GB/s / 有效算力 2.06 TFLOP/s / LM head 311.2 MB） |
| mlx-lm 的能力组合是**互斥**的 | **采纳并独立复核**：本轮用 `inspect.getsource` 对 `mlx_lm.models.cache` 做**运行时内省**，逐类确认 `is_trimmable` 的归属 ⇒ 新增 §D.5 / §D.6。**复核中把审计的一处措辞改严格了**：审计写"SSM 状态不可回退 ⇒ 投机解码 trim 不适用"，源码事实更硬——`ArraysCache` 继承 `_BaseCache.is_trimmable() → False`，`speculative_generate_step` 会**直接 `raise`**，且 `CacheList` 会把不可 trim **传播到整个模型** |
| 量化 KV 在 mlx-lm 上不是免费 4× | **采纳并复核**：确认 `QuantizedKVCache.update_and_fetch` 每步对新增 token 调 `mx.quantize`（28 层 × K/V ⇒ 56 次小 kernel/步），且 packed 布局是 scatter 写入 ⇒ 已写入 §D.6 |
| 32k 上下文 KV 拐点、BaseRT、SGLang Apple Silicon 支持 | **部分采纳并补充**：本轮发现 **BaseRT 是两篇论文**——arXiv:2607.00501（**M3/M4 Pro**，decode **1.56× vs llama.cpp、1.35× vs MLX**）与 arXiv:2607.19438（**M5 张量核**，prefill 6.4×/3.9×，decode 1.02–1.33×）。**审计只引了 M5 那篇，而 M5 的张量核对本项目硬件（M4 Pro 无 NAX）不可迁移**——这会把读者的注意力引向一个用不上的数字。SGLang 方面，官方文档确有 `SGLANG_USE_MLX=1`，但**要求 MLX ≥0.32.0，而本机是 0.31.2** ⇒ 官方路径当前不可用；且其流传的 "5.3×" 是相对 SGLang 自己的 PyTorch-MPS 路径，**不是对 mlx-lm 的比较** |
| 硬件事实（273 GB/s、17.76 GiB 建议工作集、13.32 GiB 单 buffer、M4 Pro 无张量核） | **采纳并复核**：本轮直接读 `mx.device_info()` 得到 `max_recommended_working_set_size = 19,069,665,280 B (17.76 GiB)`、`max_buffer_length = 14,302,248,960 B (13.32 GiB)`、`resource_limit = 499,000` |
| 审计建议"不再投入"的方向（深度压缩 / 稀疏 DAG、ANE 跑 decode、M4 追张量核、现在做专家流式卸载） | **采纳**：其中"深度压缩"已由本仓库 Stage 2.1 实测否证；"M4 追张量核"由 2607.19438 的 **M5 专属性**直接支持 |

**一处本报告对审计的补充（不是分歧）**：审计把 `RotatingKVCache` 的不可 trim 归因于"绕回后"，机制上正确，但**漏掉了更关键的推论**——`CacheList` 的 `all(...)` 语义使得**混合模型里只要有一个滑窗或 SSM 子缓存，整个模型就失去推测解码能力**。这把 §F.2 的判断从"难"改成"默认不通"。

**两者的关系因此是互补而非重复**：**任务 A 回答"哪些方法值得用、哪些组合值得做"，任务 B 回答"在本机这个运行时里它们能不能共存"**；本报告的 §D.5 / §D.6 就是把两者接起来的那一层。
