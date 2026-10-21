# R2 — 证据索引（逐条原始 URL、版本日期、实验口径、证据等级、反面材料）

**任务**：R2 / Issue #1。研究者：MiniMax。所有来源打开日期 **2026-10-21**（除另注）。

**配套文档**：结论见 **`2026-10-r2-report.md`**（主报告），组合矩阵见 **`2026-10-r2-combinations.md`**，一手全文核验记录见 **`fulltext-notes/`**。本文件是**证据层**，不单独下结论。

## 证据等级约定

| 等级 | 含义 |
|---|---|
| `[P]` | **一手来源已打开**：arXiv abs 页 / arXiv LaTeXML HTML 全文 / 论文 PDF / 会议页 / OpenReview 记录 / 官方仓库源码 |
| `[O]` | **官方产物直读**：GitHub API 记录、HF 模型卡 / papers 页、OpenReview API 字段、`mx.device_info()` 本机读数 |
| `[S]` | **二手来源**：第三方聚合站、博客、结构化机器评注（`pith.science`）、Wikipedia 类条目 |
| `[推算]` | **本报告自己的算术**，非任何来源的陈述；给出推导过程供复核 |
| `[UNVERIFIED]` | **未能打开**；写明尝试过什么 |

> ⚠ **`pith.science` 的结构化评注一律标 `[S]`**：它是**机器渲染**的（自报核验等级 T0 = "AI 读，未重算"；其"模拟作者回复"是机器生成）。**它指出的问题只有在论文自身的表格/文字能佐证时，本报告才当作证据使用**；否则仅作"值得攻击的线索"。

---

## A. 极低位宽权重表示

### [R1] QTIP: Quantization with Trellises and Incoherence Processing
- **arXiv**: `2406.11235` **v4**（v1 2024-06-17，v4 2025-06-18）`[P]`
- **作者**：Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa（Cornell）
- **Venue**：**NeurIPS 2024 Spotlight**（arXiv comment 字段）`[P]`
- **机制**：TCQ（trellis coded quantization）替代 VQ——**有状态解码器把码本大小与比特率/有效维度解耦**，突破 VQ"码本随维度指数增长 ⇒ 只能 d≤8"的限制；设计"从纯查表到计算式免查表"的 trellis 码，目标是硬件友好的 **bitshift trellis** `[P]`
- **省什么**：权重容量 + 权重带宽（PTQ 直接压 decode 内存流量）`[P]`
- **训练要求**：PTQ + 校准（需 Hessian）；**不需重训** ⇒ 适用冻结权重 `[P]`
- **无损等级**：**L4**（量化本身改变模型函数）
- **关键运行时代价**（CurveTQ 原文）："rotate each weight matrix by a random orthogonal transform, **which must be undone at every decoding step**" `[P]`
- **2-bit 头条数字**：Llama-2-70B 2-bit 与 FP16 的 PPL 差 <0.7 `[S]` — ⚠ **未从论文正文核到该数字的精确表述与对照表**
- **消费端**：**trellis 在 Metal 上无实现**；llama.cpp 已部署的 2-bit 是**格码系**（IQ2_*）`[S]`
- **2026 状态**：**被 CurveTQ（去旋转）与 Proteus（免查表）超越**；仍是这一族的事实基线 `[P]`

### [R2] Proteus: Lookup-Free Trellis-Coded Quantization by Lattice-Breaking Compute Codes for 2-Bit LLMs
- **Venue**：**ICML 2026** `[P]` — 会议页 `https://aiconfpaper.com/paper/icml-2026-SHYZM21zO7`
- **arXiv id**：**本轮未检索到**。检索记录：arXiv API `ti:"Lookup-Free Trellis-Coded Quantization"` / `abs:"MUL-BAL"` / `ti:"Proteus" AND abs:"trellis"` / `all:"Proteus" AND abs:"quantization" AND abs:"2-bit"` —— 最后一次因 **HTTP 429 限流**失败，其余三次无命中 `[UNVERIFIED]`
- **机制**：**严格免查表 TCQ**。计算式生成器 **MUL-BAL** = 廉价整数混合 + 逐层仿射高斯化器 ⇒ overlap-robust、near-Gaussian 码值，**运行时零表加载** `[P]`
- **它识别的真问题** `[P]`：
  - *"ultra-low-bit weight-only PTQ helps **only if** dequantization avoids irregular codebook or LUT access in the inner loop"*
  - *"existing 2-bit TCQ pipelines either **reintroduce micro-LUTs** or suffer **overlap-amplified artifacts** because incoherence improves global Gaussianity but **does not guarantee overlap-local joint geometry**"*
- **数字**：Llama 2 (7B–70B) 2-bit PTQ，**70B 端到端解码带宽 740 GB/s vs 基线 1020 GB/s（吞吐相当）** `[P]`
- **训练要求**：逐层从"tiny, pre-vetted candidate pool"选择 + 通道补偿 + **可选 few-shot 蒸馏**（只调逐层仿射统计量，不改解码器结构）`[P]`
- **2026 状态**：**仓库未找到** `[UNVERIFIED]`

### [R3] CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search
- **arXiv**: `2610.09212` **v1**，2026-10-06（scirate 记 2026-10-08 发布）`[P]`
- **作者**：Guanhua Ding, Zi Wang, Ruichao Li, Jack Liu（**Black Sesame Technologies**）`[P]`
- **机制** `[P]`：
  1. 误差反馈把损失化为**逐坐标舍入误差的加权和**，权重是 **Hessian 的 LDL 分解对角**；既有量化器**算了但从不读**。
  2. 把这些权重放进 **Viterbi 分支度量** ⇒ 搜索在每个编码块内顺应曲率。
  3. **"weighting in the native basis and rotating are substitutes"** —— 旋转的作用是消除块内变化。
  4. 幅值/边缘形状用**分解式尺度场 + 闭式分位数表**；每编码块存**起始状态**以适配误差反馈残差。
- **数字** `[P]`：三模型上"加权原基底搜索"与"全维随机 Hadamard"差约 1 个点；**2-bit 下平均下游精度比 QTIP 与 Proteus 高 1–3 点**（三个 4–8B Instruct 模型），**即使给基线同款起始状态仍领先 1–3 点**（起始状态本身值 1–3 点）；**35B MoE 上领先，据称是此类模型首个 trellis 编码结果**；**"With no rotation to undo, our decoder is the fastest of the three at all tested batch sizes and bit widths"**
- **⚠ 未能核**：**arXiv 无 HTML 版本**（`arxiv.org/html/2610.09212` 返回 "No HTML for '2610.09212'"）⇒ **正文、硬件、batch、逐模型 PPL 表、trellis 解码器实现细节均未核**；以上仅为摘要级 `[P*]`
- **消费端意义**：把"旋转税"变成可选；若实现有 trellis 内核，去旋转直接省掉每量化层一次在线正交变换

### [R4] XOR-Trellis: Ultra-Low-Complexity Dequantization and Curvature-Aware Hadamard-Free LLM Quantization
- **arXiv**: `2610.00432` **v1**，2026-09-30 `[P]`
- **作者**：Xiaofan Que, Nir Elkayam, Spandan Pyakurel, Shuokai Pan, Dibakar Gope（**Arm Inc.**；Pyakurel 与 Gope 标注 RIT）`[P]`
- **机制（两件互补的事）** `[P]`：
  1. **超低复杂度 trellis 反量化器**：结构化、硬件友好的 state-to-value 映射，保留 trellis 搜索所需的多样重建选择 ⇒ *"inexpensive, highly parallel runtime reconstruction"*
  2. **曲率感知目标**：把离散 trellis 路径优化重构为**在原始坐标空间**中反映模型敏感性的目标
- **明确点名本项目关心的问题** `[P]`：*"maintaining quantization accuracy without **costly incoherence transformations**"*；*"reconstructing compressed weights at sufficient parallel throughput **to avoid making dequantization an inference bottleneck**"*
- **商业意义（独特）**：作者来自 **Arm** ⇒ 这是通向**移动/嵌入式 CPU IP** 的实际部署路径，而非"又一个 CUDA 论文"
- **⚠ 未能核**：模型、硬件、bit-width、全部数字（仅摘要）`[UNVERIFIED]`

### [R5] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization
- **arXiv**: `2605.04738` **v2**，2026-05-06 `[P]`
- **作者**：Zhikai Li, Zhen Dong, Xuewen Liu, Jing Zhang, Qingyi Gu — **ICML 2026**（arXiv comment）`[P]`
- **机制** `[P]`：**加性**权重抑制。观察到 **Hessian 跨输入具有低秩一致性**，某些方向持续表现出消失的曲率 ⇒ 识别 **Hessian 的稳定零空间**，在零空间内线性组合构造**加性权重变换**，抑制权重离群点而**不影响任务损失**
- **部署属性（本报告最看重的一句）** `[P]`：*"This additive transformation can be **absorbed into the weights offline**, requiring **no inter-layer transformations** and introducing **no inference overhead**."*；闭式解，**无资源密集训练、无迭代过程**
- **无损等级**：**L3**（任务能力近似保持；"不影响任务损失"是 Hessian 意义下的局部论证，**不是**端到端质量保证）
- **对比意义**：与 A1 全族的分野是结构性的——**旋转是正交的、必须在线撤销的；OSAQ 是加性的、可离线折叠的**

### [R6] Model-Preserving Adaptive Rounding（YAQA）
- **arXiv**: `2505.22988` **v3**，v1 2025-05-29 — **ICML 2026**（arXiv comment）`[P]`
- **作者**：Albert Tseng, Zhaofeng Sun, Christopher De Sa（Cornell）`[P]`
- **⚠ 归类更正**：**真实标题是《Model-Preserving Adaptive Rounding》**；"YAQA" = "Yet Another Quantization Algorithm"，是正文中的简称。**YAQA 不是一个新的位宽格式，而是自适应舍入（误差反馈）算法。**
- **机制** `[P]`：① 刻画自适应舍入算法的**收敛时间**（经其 Hessian 近似的结构）；② 证明**端到端误差可由该近似与真实 Hessian 的余弦相似度界定**；③ 由此得到 **Kronecker 分解近似**与近最优 Hessian sketch
- **数字** `[P]`：**可证优于 GPTQ/LDLQ**，经验上误差降低 **≈30 %**；**甚至达到比 QAT 更低的误差**；**"all while adding no inference overhead"**
- **搜索成本（本报告另行追查）**：**FastKron**（arXiv:`2608.06291` v2, 2026-08-06, Birnick & Saab）明确指出 *"BoA uses a large number of sequential steps, while **YAQA improves over the sequential depth but still has quartic total cost**"*；FastKron 把总工作量从 `O(m²n²)` 降到 `O(mn(m+n))`、顺序步数降到 `O(m+n)` `[P]`
  ⇒ **本轮第一次见到的"量化器自身搜索成本被量化"的一手陈述。** 对消费级：再好的 PTQ 算法，若校准本身要 `O(m²n²)` 总工作量，**在 24 GB 设备上根本跑不动校准**。这是一个被严重低估的部署约束。

### [R7] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
- **arXiv**: `2502.02631` **v2**，2025-02-04 — **NeurIPS 2025**（arXiv comment）`[P]`
- **作者**：Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang（**Meta**）`[P]`；模型权重在 HF `facebook/mobilellm-6722be18cb86c20ebe113e95` 集合 `[P]`
- **核心发现** `[P]`：
  1. **2 与 3 bit 之间存在"学习相变"**：3 bit 及以上微调后仍**贴近原预训练分布**；**2 bit 及以下表示剧烈变化**
  2. **三值、2-bit、3-bit 在 size-accuracy 权衡上表现相当，且通常优于 4-bit 与二值**
  3. 600M 三值模型**超过此前 SoTA 的 3B 三值模型**（1/5 参数）
- **部署** `[S]`（PyTorch 官方博客转述）：实现 **2-bit GPU 算子**，TritonBench 上**相对 FP16 最高 4.14×、相对 Machete 4-bit 算子 1.24×**；**已集成进 torchao**；HF 发布 MobileLLM 低比特系列（最小 1-bit 125M，≈16 MB）
- **⚠ 致命限制**：ParetoQ 是 **QAT** ⇒ **不适用于"冻结权重"前提**。其价值是**位宽-质量相变的地图**，不是可直接部署的方法
- **⚠ 未能核**：TritonBench 4.14×/1.24× 的硬件与 batch（仅 `[S]`）

### [R8] BCJR-QAT: A Differentiable Relaxation of Trellis-Coded Weight Quantization
- **arXiv**: `2605.10655` **v1**，2026-05-11 `[P]`；代码 `https://github.com/Venugopalan2610/quant-2bit`，权重与轨迹快照在 HF `[P]`
- **机制** `[P]`：把 trellis 的**不可微 Viterbi argmax** 换成 **BCJR 前向-后向 sum-product**（温度 T），产生 trellis 路径上的 Boltzmann 期望码字，**T→0 时精确退化为硬 QTIP 码**
- **含义**：把 2-bit 从"可证明但 PTQ 天花板"推向"可训练"——**但必须训练 ⇒ 同样不适用冻结权重**

### [R9] QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization
- **arXiv**: `2609.00224` **v2**，2026-08-31 — **EMNLP 2026 Main Conference**（arXiv comment）`[P]`
- **机制** `[P]`：三值 + 稀疏残差显著权重 + 按列优化，**sub-2-bit**
- **★ 自述本族的代价** `[P]`：*"Many leverage unstructured sparsity to mitigate this loss, but at the cost of **regularity and GPU-friendly execution**."*
  ⇒ **非结构化稀疏救 2-bit，代价是失去规则访存**——在统一内存设备上几乎必然反噬带宽收益（与 R1 的 A5 观测同构：逐位等价的堆权重矩阵乘在 n=1 慢 29 % `[M]`）

### [R10] GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling
- **arXiv**: `2604.18556` **v2**，2026-04-20 `[P]`（R1 已深入核验）
- **R1 的关键实测** `[O]`：GSQ-RCO 模型卡位宽扫描 —— BF16 16.00 bpw / 354 GB / AIME25 100.00 / GPQA-D 91.92 / LCB v6 87.43 / task-avg 93.12 / ZS-recovery 100.0 % → **Q2_0 2.40 bpw / 66.4 GB / 96.67 / 89.39 / 81.14 / 89.07 / 101.4 %** → IQ2_XS 2.50 / 68.0 / 96.67 / 87.37 / 83.43 / 89.16 / 100.3 % → IQ3_XXS 3.00 / 75.8 / 100.00 / 91.41 / 86.29 / 92.57 / 100.4 % → IQ3_S 3.50 / 83.6 / 100.00 / 92.93 / 86.86 / 93.26。速度：**Q2_0 367.49 prompt t/s / 93.79 decode t/s / 6.70 s** vs IQ2_XS 108.19 / 70.30 / 12.68 s
  ⇒ **同位宽附近 decode 差 1.34×、端到端差 1.9× ⇒ 格式比位宽更重要**

### [R11] Model Compression with Exact Budget Constraints via Riemannian Manifolds（RCO）
- **arXiv**: `2605.00649` **v3**，2026-05-01 `[P]`（R1 已深入核验）
- **机制**：把总尺寸预算重述为 logit 空间上的光滑 Riemannian 流形，**精确**满足预算并直接优化任务损失 `[P]`

### [R12] ★ 本族七篇的部署面（子代理逐篇核出，2026-10-21）

| 论文 | llama.cpp | vLLM | MLX/Metal | 内核产物 |
|---|---|---|---|---|
| QTIP | 否 | 否（#11416 `not_planned`，2026-01-10） | 否 | 有，**仅 CUDA**（`exllamav3` EXL3） |
| Proteus | 否 | 否 | 否 | **否**（`YZW-explorer/Proteus` 只有 README） |
| CurveTQ | 否 | 否 | 否 | **否**（GitHub 0 仓库） |
| YAQA | 否 | 否 | 否 | 有，**仅 CUDA**（复用 qtip 内核） |
| OSAQA | — | — | — | 未核实 |
| **GSQ** | **是**（GGUF K-Quant Q2_K/Q3_K_M，"runs unchanged on llama.cpp / Ollama / LM Studio"） | `compressed-tensors`，**声称**自动检测，未验证 | 未测量 | 有（Apache-2.0） |
| RCO | N/A | N/A | 否 | 有（Apache-2.0） |

**七篇中没有任何一篇有 Apple/Metal/MLX 测量** `[P]`（对 QTIP/YAQA/CurveTQ/RCO 全文做 `Metal`/`Apple`/`MLX`/`llama.cpp`/`vLLM` 字面串检索；GSQ 硬件章节只有 NVIDIA L40s）。

**官方仓库与提交** `[O]`：

| 仓库 | 许可 | 最后提交 | 备注 |
|---|---|---|---|
| `Cornell-RelaxML/qtip` | GPL-3.0 | `e90c6688c8df` (2025-06-22) | 内核 CUDA-only |
| `Cornell-RelaxML/yaqa-quantization` | GPL-3.0 | `f9508723251a` (2025-06-20) | 内核复用 qtip |
| `IST-DASLab/GSQ` | Apache-2.0 | `03fc16484c36` (2026-07-05) | **唯一有可用 llama.cpp 路径** |
| `IST-DASLab/RCO` | Apache-2.0 | `9a1e09c07d46` (2026-06-13) | 分配器 |
| `YZW-explorer/Proteus` | — | `fdf1845d2c1b` (2026-05-29) | **只有 README 占位** |

**OpenReview / 会议身份** `[O]`：Proteus `SHYZM21zO7`（ICML 2026 正会）、YAQA `PKFilPWjMI`（ICML 2026 正会）、CurveTQ `5n4xeaG7bM`（ICLR 2027 投稿）。

### [R12a] ★★ `YanissAmz/escha-port` —— 本轮唯一的端到端 llama.cpp 2-bit 证据
- **是什么** `[O]`："a QTIP-style 2-bit format in llama.cpp"，参考模型 **Qwen3.8-27B @ 2.469 bpw**（GGUF 转换、CPU 参考解码器、CUDA + HIP 内核）
- **与 Q4_K_XL 正面对比（同一张 RTX 3090）** `[O]`：

| | 2-bit trellis (2.469 bpw) | Q4_K_XL |
|---|---|---|
| 生成 tok/s | **34.19** | **41.40** |
| pp512 tok/s | **610.20** | **1382.86** |
| PPL（wikitext-2-raw, 580 chunks） | **7.3679** | **6.9744** |
| 磁盘 | 9.66 GiB | 16.35 GiB |

- **作者原话** `[O]`：*"against a real Q4_K_XL this format **wins on size and loses on everything else**"*；*"the decode lookup table, the obvious optimisation, was generating **13× more L2 traffic than the weights it was meant to save**"*
- **⚠ 口径限制** `[S]`：**不是 Apple 测量**（RTX 3090 + gfx1151 APU）；是**商业格式（Escha Labs / ESCHAM）的第三方移植**，不是 QTIP 本身的移植；ESCHAM↔QTIP 的等价性为 `[S]`，数字原文为 `[O]`

### [R12b] `Azimml/trellis-webgpu` —— "每个 TCQ 解码器都是 CUDA-only"
- `[O]`：*"every TCQ decoder shipped to date is **CUDA-only**, which locks this quality tier to NVIDIA data-center and desktop GPUs"*；该项目在浏览器 WebGPU 上跑 8B 模型，并**复现**了 QTIP 发表的率失真（MSE **0.0739** vs QTIP 报告的 **0.0733**）

### [R12c] vLLM issue #11416
- `[P]`：《[Feature]: QTIP Quantization》2024-12-22 开，**2026-01-10 以 `state_reason: not_planned` 关**；GPL-3.0 许可阻碍 vendoring；2025-04-11 有志愿者表示愿意实现；线程中"相对 BF16 8× 加速是否可达"的提问**从未被回答**

### [R12d] llama.cpp PR #19726
- `[P]`：移植 IQ*_K quants 的 PR 于 **2026-02-23 关闭**

### [R12e] QuIP / QuIP# 谱系（背景）
- QuIP（Chee et al. 2023）与 QuIP#（ICML 2024）是 2-bit 的起点，QTIP 的直接前身 `[S]`
- **独立横向比较**（**本轮找到的唯一第三方统一协议比较**）：**Bielik-Q2-Sharp**（Polish 11B Mistral 基座，比较 **QuIP# / SpinQuant+GPTQ / ButterflyQuant / QTIP / VPTQ / AQLM** 六种 2-bit PTQ，统一校准同一 Hessian）：
  - QuIP# E8P12 达 **71.92 %**（对照 IQ2_XXS 72.07 %），模型 3.26 GB vs ~2.6 GB
  - **QTIP 达到最佳 per-bit 效率**，以 **35 % 更小体积（3.27 GB）**匹配 VPTQ 质量，**~2.4 bpw 下 79.4 % MC 准确率** `[S]`
  - ⚠ 比较对象**全是 PTQ，没有任何 trellis / Metal 相关项** ⇒ **它无法回答"2-bit 在 Apple Silicon 上行不行"**

### [R12f] 2-bit 的质量损失（论文自报）
- **YAQA 自表** `[P]`：2 bit（最佳变体）**GSM8K 84.31 → 66.07**
- **QTIP 自表** `[P]`：Llama-3.1-70B W2 **PPL 3.52 → 5.08**、zero-shot 均值 **66.78 → 63.55**
- **GSQ 自表** `[P][O]`：70B zero-shot **FP16 78.99 → GSQ 2-bit 75.57**，且 **GSQ 仍比 QTIP 低 1.7 点**
- **GSQ 校准域损伤** `[P]`：Kimi-K2.5 2-bit **GPQA Diamond 下降**——OpenThoughts 校准集偏数学/代码
- **CurveTQ 自测** `[P]`：去掉误差反馈 **GSM8K −27**、IFEval −6；换分支度量 **PPL −0.24 但下游 −3.3 点**

---

## B. 冻结权重的 FFN 动态稀疏

> 本组的两篇主论文由**子代理通读全文**（LaTeXML HTML + PDF 表格），条目可回溯到章节号。

### [R13] Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs
- **arXiv**: `2607.27591` **v1**，2026-07-30，cs.LG，CC BY 4.0 `[P]`
- **作者**：Jinyi Liu, Wei Chen, Pengyu Chen, Xinyi Yuan, Minghe Bai, Guoquan Wu, Jun Wei（ISCAS + UCAS + NJUPT）`[P]`
- **打开来源**：`https://arxiv.org/abs/2607.27591` + `https://arxiv.org/html/2607.27591v1`（1661 行提取文本，含附录 A.1–A.11 全读）`[P]`
- **机制（精确）** `[P §2.1, §2.2, §4.1, A.6]`：`FFN(x) = (xW_up ⊙ SiLU(xW_gate))W_down`；`u=xW_up`、`h=SiLU(xW_gate)`、`s=u⊙h`。**up/gate 输出稀疏 + down 输入稀疏，由一个共享的中间通道掩码 `m_s` 驱动**：
  ```
  Stage 1: [m_x]_{ℓ,t,i} = 1[|x_{ℓ,t,i}| ≥ τ_x,ℓ^{s1}]
           ũ = Π^in(x, W̃_up, m_x);  h̃ = SiLU(Π^in(x, W̃_gate, m_x));  s̃ = ũ ⊙ h̃
           [m_s]_{ℓ,t,j} = 1[|s̃_{ℓ,t,j}| ≥ τ_s,ℓ^{s2}]
  Stage 2: u_sp = Π^out(x, W_up, m_s);  h_sp = SiLU(Π^out(x, W_gate, m_s))
           s_sp = u_sp ⊙ h_sp;  y = Π^in(s_sp, W_down, m_s)
  ```
  - **是阈值，不是 top-k**（论文明确拒绝 per-token top-k）；阈值比 top-k 平均高 **+1.79 分**（8 组设置，`[P Table 9]`）；层内 P10–P90 稀疏度跨度 17.33 / 17.51 点 `[P Table 8]`
  - **逐 token、逐层、逐中间通道**；非逐专家；非整块 MLP 跳过
  - **Oracle 上界** `[P §3.1]`：70 % 稀疏下多数模型 PPL 相对上升 **<3 %**
- **selector 成本（完整）** `[P §4.2, §5, A.3]`：**不是**学习探针/门控权重/注意力预测器/路由，而是 **`W_up`/`W_gate` 的第二份 INT4 量化副本**。成本模型 `C_Dense = 3`、`C_Prox = 2α(1−s₁) + 3(1−s₂)`、**有效稀疏度 `e = s₂ − 2α(1−s₁)/3`**。**α 为实测**：INT4/FP16 GEMV 延迟比 **0.234–0.351**（pooled mean 0.285），论文取 **1/3**；跨 A6000/RTX 3090/A100、4 个 Qwen3 规模、4 种输入稀疏度 `[P §5, A.3 Table 5]`
- **内存代价** `[P §5.3]`：*"retaining Stage 1 INT4 proxy weights in GPU memory introduces an **approximate 12% weight-storage overhead**."*（**不在其自身的有效稀疏度指标里** —— α 是延迟比不是字节比）
- **延迟拆解** `[P A.5, Fig. 8]`（Qwen3-8B @70 %，A6000，每 token、36 层）：block **9.88 ms/token**；attention+RMSNorm+残差 4.72；**FFN 5.18**；Stage 2 gate/up 2.64（FFN 的 51.0 %）、**Stage 1 代理 1.03**、Stage 2 down 1.47 ⇒ **代理 ≈ 整个 block 的 10.4 %、FFN 时间的 19.9 %** `[推算]`
- **★ 未处理非连续量化读取** `[P §4.1, §5.1]`：速度测量全在 **FP16**；Fig. 6 是**PPL 图**（AWQ/FP8/W4A16/W8A8）；**全文正则检索 "byte"/"bytes" 0 命中、"group size" 0、"contiguous" 0、"pack" 0**。论文原话：*"realizing these gains requires specialized kernels that jointly support sparse execution and low-bit arithmetic, which we leave to future work."*
- **散布输出通道的字节算术** `[推算]`：`W_up,W_gate ∈ R^{d_model×d_ff}` 行主序、量化分组沿输出通道轴（llama.cpp/GPTQ/AWQ 惯例）。Prox 选 `k=(1−s₂)·d_ff` 个**散布**输出通道 ⇒ 实际触及分组数 `(N/g)·(1−(1−k/N)^g)`。Qwen3-8B（`d_ff=12288`，`k/N=0.3`）：`g=256` ⇒ `48·(1−0.7^256) ≈ 48` ⇒ **每组都被触及**；`g=32` ⇒ `384·(1−0.7^32) ≈ 384` ⇒ **每组都被触及**。⇒ **up/gate 的 Stage-2 稀疏在行主序分组量化布局下 ≈0 % 字节节省，只有 MAC 节省**；`W_down` 因行维连续而能省。合计字节 ≈ `(2.0+0.233+0.075)/3 = 0.77` 倍稠密 ⇒ **带宽受限上限 ≈1.30×，不是 1.99×**
- **测试设置** `[P §5, §5.1, A.5]`：10 模型 6 族 —— Qwen3-1.7B/4B/8B/14B、Qwen3.5-4B/9B、Ministral-3.3B、Mistral-7B、Llama-3-8B、**Gemma-3-12B（GeGLU）**；**硬件 A6000（主）/A100/RTX 3090，无 Apple/Metal/MLX/CPU**；**batch = 1**；速度测全 FP16；质量 = LM Harness 6 任务（8-shot GSM8K、5-shot MMLU、10-shot HellaSwag、0-shot ARC-Easy、25-shot ARC-Challenge、0-shot PIQA）聚合分
- **数字** `[P]`：速度 **1.51–1.99×**（60–70 % 稀疏），头条 **1.99× @70 %**；tok/s：Qwen3-4B A6000 82.0→135.2（1.65×）、Qwen3-8B A6000 46.7→85.0（1.82×）、Qwen3-8B A100 93.5→133.5（1.43×）
- **★ 论文自报负面结果** `[P]`：
  1. **70 % 稀疏仍掉真实质量**：Qwen3-8B **76.1→68.6（−7.5）**、Qwen3-4B **71.1→58.7（−12.4）**、Qwen3-1.7B 61.4→56.5（−4.9）；**60 % 才是实用甜点**（Qwen3-8B 76.1→73.3，−2.8）
  2. **代理精度是陷阱（A1）**：INT4 换 FP16 **反而更差**（70 % 下 −9.0 / −4.7 分）——多出的计算必须从 Stage 2 重新分配
  3. **精确 Stage 2 不可妥协（A3）**：直接把代理值喂给 down 投影，**−24.3 / −18.1 分**
  4. **固定 s₁=0.8 更差（A2）**：最多 −4.4 分
  5. **不支持大 batch** `[P §5.3]`
  6. **+12 % 权重存储** `[P §5.3]`
  7. **量化 × 稀疏的速度明确留作 future work** `[P §5.1]`
  8. **生成任务下 prefill 走稠密** `[P §5]`
  9. **Apple/Metal/MLX 完全未涉及**
  10. **Stage-2 分配被 clamp** `[P A.4, Table 6]`：40 %/70 % 工作点上 s₁ 被 clamp 到 0 %/70 %，s₂ 变成 **62.2 %/76.7 %** ⇒ **头条工作点上代理正运行在保真度最差处**
- **稀疏注意力组合** `[P Table 2]`（Qwen3-8B, 60 % FFN 稀疏）：稠密 33.1 tok/s @16K → **RocketKV+Prox 63.6（1.92×）**；@32K 23.9 → 62.5（**2.62×**）；LongBench/RULER 100.0/100.0 → **96.6/98.0**（RocketKV+Prox @16K，**LongBench 掉 3.4 分**）、→ 92.5/100.0（HShare+Prox）
- **2026 状态** `[O]`：arXiv v1 only；**OpenReview = ICLR 2027 Conference Submission**，forum `8UJzSYuF9s`，submission `ICLR.cc/2027/Conference/Submission11912`，venue 字段 *"ICLR 2027 Conference Submission"*，bibtex note *"under review"*。⚠ **OpenReview 版标题已改**为 *"Prox: Training-Free FFN Activation Sparsification in LLMs via Approximate Intermediate-State Salience"*，摘要称 **2.07×**（arXiv v1 为 1.99×）—— **该数字来自元数据摘要，本报告未读 OpenReview PDF，不采用** `[UNVERIFIED]`。**无公开评审**（invitation 列表无 `-/Official_Review`）⇒ **截至 2026-10-21 无任何独立批评**。**无代码仓库**（GitHub 搜索 0 命中）；**HF papers 页 404** ⇒ 无社区产物、无复现。**未合入 vLLM/llama.cpp/MLX**（llama.cpp 2026 的 sparse 工作全是稀疏注意力/稀疏 MoE）

### [R14] Enabling Dynamic Sparsity in Quantized LLM Inference（系统名 **SpQt**）
- **arXiv**: `2511.04477` **v1**，2025-11-06，cs.DC `[P]`
- **作者**：Rongxiang Wang, Kangyuan Shu, Felix Xiaozhu Lin（**Univ. of Virginia + Zoom Communications**）`[P]`
- **打开来源**：`https://arxiv.org/abs/2511.04477` + `https://arxiv.org/html/2511.04477v1`（全部散文） + **PDF `https://arxiv.org/pdf/2511.04477` 第 3–9 页**（HTML 把 Table 1–3 渲染成 glyph-path SVG，无可抽取文本，故表格数字取自 PDF）`[P]`
- **机制** `[P §2, §3.2, §4.2.3]`：**不做通道选择**，做"激活稀疏"= 对隐藏态**输入侧**幅值阈值化，施加于**除输出嵌入外的每一个 GEMV**：
  > *"any hidden state entry satisfying |x_i| < threshold can be treated as zero, allowing the corresponding **column W[:,i]** in the weight matrix to be skipped."*
  索引收集 = **Blelloch 并行前缀和** + **全局原子计数器**预留偏移，使索引数组连续且合并访存
- **★ zigzag 布局（核心贡献）** `[P §4.1.2]`：
  - **问题**：Q4_K 把 256 个连续权重打包成一个 superblock（8 个 32-权重子块 + 6-bit 子块 scale + 16-bit superblock scale/min），superblock **按行组织**，与**列向**稀疏模式"directly conflicts"；块内跳过任意权重会引入分支条件、造成 warp divergence
  - **作者的实测结论**：*"only skip the entire superblock is practical."*
  - **解法**：**按列分组**（稀疏的输入通道对应的**所有 superblock 作为整体跳过**）**+ 行主序 superblock 存储**（保证连续输出神经元的 GEMV 结果在内存中连续）⇒ 跳过通道 c 即跳过整个 superblock ⇒ **~128 B 粒度的真实字节节省**
  - **反例**：仅按列重分组而不重排存储会"disrupts memory locality"
  - **内核后果** `[P §4.2.1–4.2.3]`：行并行度从 `m/4` 降到 `m/256` ⇒ 引入**混合方案**（每 superblock-row 启 `n₁` threadgroup × `n₂` simdgroup，部分和留寄存器/局部内存，行末归约；实测 `tg=32, sg=2` 最优）+ **负载均衡**把运行时活跃索引数组均匀分给 simdgroup
- **★ 决定性负测量** `[P §6.3, Fig. 9]`：**"Naive sparse（原权重布局）"基线在 50 % 稀疏下只有 <0.2× 加速（约 5× 变慢）**，跨所有测试 GEMV 形状
  ⇒ **已发表文献里"FLOP 节省 ≠ 字节节省"最硬的实证，且是论文自己的结果**
- **selector 成本** `[P §2, §5]`：**无学习选择器**（论文批评前人的低秩预测器"add nontrivial overhead and complicate deployment"）。成本 = 每次 GEMV 一次 **O(n) 阈值扫描** + 写 `n_ns` 个 int 索引，相对 O(n·m) 权重字节 ~10⁻³ 量级。校准离线，复用 **TEAL 已发布的校准产物** + 一个约 100 行 Python 工具
- **★ Apple Silicon 实测（本轮唯一落在目标硬件族上的测量）** `[P Table 2, PDF p.8]`：
  - **硬件**：**A19 Pro**（5 核 GPU, 1.6 GHz, 75.8 GB/s, ~5 W）、**M2**（8 核, 102.4 GB/s, ~20 W）、**M2 Pro**（19 核, 204.8 GB/s, ~45 W）、**M2 Max**（30 核, 819.2 GB/s, ~80 W）
  - **模型**：Llama-2 **7B/13B/70B**、Llama-3 **8B**（**无 20–30B 模型**；Llama-3-70B 因已知 Q4 质量问题被排除）
  - **精度**：**Q4_K**（除 token/output 嵌入）；**单 token 解码、约 10-token prompt、greedy**；**无长上下文、无多 token batch**
  - **软件**：**~8K 行 C/C++**，基于 llama.cpp b5711，**内核用 Metal Shading Language**；CUDA "anticipated" 但未做
  - **⚠ 表题与数据自相矛盾**：表题写 "latency (ms/token)"，但每行括号系数都等于两个数之**比值**，只有读作 **tok/s** 才自洽 ⇒ **表题是错的**
  - **端到端（tok/s，Unified 50 %）**：A19 Pro Llama-2-7B **1.55×** / Llama-3-8B 1.46×；M2 1.40× / 1.39×；**M2 Pro 1.26× / 1.26×**；**M2 Max 1.19× / 1.18×**（Llama-2-13B 1.23×、Llama-2-70B 1.45×）
  - **同一张表 Baseline-F16 行**：M2 Max 上 F16 只有 Q4_K 的 **0.38–0.42×**（即 F16 慢 2.4–2.6×）⇒ **量化的价值远大于稀疏**
  - **微基准** `[P §6.3]`：密集 zigzag GEMV 最多比默认 Q4_K **低 17 % 延迟**；稀疏 GEMV 加速 **1.22× @25 %、1.51× @40 %、1.78× @50 %**，K 越大增益越大
- **★ 质量** `[P Table 3]`（wikitext.test PPL，Q4_K）：F16 5.13/4.62/3.27/5.64；Q4_K 5.28/4.69/3.35/5.93；Unified 25 % 5.27/4.69/3.41/5.94；40 % 5.40/4.79/3.55/6.42；50 % 5.65/4.97/3.77/**6.67**；**65 % 7.39/5.96/4.48/9.59**
  - **65 % 是悬崖**（Llama-3-8B 5.93→9.59，+62 %）；论文自述 *"higher sparsity (65%) shows noticeable quality loss."* **可用上限约 50 %**
  - **50 % 处，稀疏的 PPL 代价大于 Q4_K 的 PPL 代价**（Llama-3-8B：Q4_K 5.93 vs Unified-50 % 6.67 vs F16 5.64）
  - **zigzag 无负载均衡只有 1.02–1.06×**（B2）⇒ 光有布局不够
  - **低稀疏度是净亏**：M2 Max 25 % TEAL = **0.99×**、M2 Pro Unified 25 % Llama-3-8B = **0.99×**
  - **prefill 稠密且未优化**（"does not yet employ a dedicated tiling-based GEMM optimization"）⇒ 不声称 prefill 加速
  - **batch>1 静默回退稠密**（*"automatically dispatches dense kernels when the token count is greater than one"*）
  - **权重文件不兼容 stock llama.cpp / MLX / HF** ⇒ 必须用自定义重排工具（*"quantizing a 70B model takes less than five minutes on a commodity workstation"*）
- **2026 状态** `[O]`：**无公开仓库**（GitHub 搜索 0 命中）；**HF papers 页 404**；**找不到 OpenReview 记录**（PDF 写 *"Preliminary work. Under review."*，并有 ICML 样式违规警告）⇒ **尚未被接收，只是投稿中**；**未合入 llama.cpp**（zigzag 是 fork-only）；**无 MLX、无 vLLM 实现**；未发现被更新工作取代
- **⚠ 未能核** `[UNVERIFIED]`：Figure 2–10（glyph-path SVG 无文本，仅读 PDF 第 3–9 页）；**Semantic Scholar 引用数**（API 429 限流）⇒ **既没能找到、也不能排除存在未被索引的第三方复现**

### [R15] TEAL: Training-Free Activation Sparsity in Large Language Models
- **arXiv**: `2408.14690` v3 `[P]` — **本方向最重要的前身**。全网络隐藏态条目级激活稀疏 + 逐矩阵贪心稀疏搜索。**Prox 用它作质量基线；SpQt 直接复用它发布的校准产物与贪心模式**；它还是"prefill 稀疏化损害质量"这一结论的来源（SpQt 引用）

### [R16] CATS: Contextually-Aware Thresholding for Sparsity in LLMs
- **arXiv**: `2404.08763` v4 `[P]` — 在**稠密算出的 gate 激活** h 上阈值化 → up 输出稀疏 + down 输入稀疏。Prox 三基线之一。**Prox Table 1 显示 CATS 在 60 % 崩塌**（Qwen3-8B 76.1 → **27.6**）—— 单分支通道选择在高稀疏下失效的最清晰已发表证据

### [R17] M-COUNTDOWN: Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection
- **arXiv**: `2505.17701` v3 `[P]` — 在 **up 分支**激活 u 上阈值化；另提供需训练的 D-COUNTDOWN。Prox 第二基线；也是"FFN 加权和视角"（`FFN(x) = Σ_i s_i W_down[i,:]`）的来源

### [R18] R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference
- **arXiv**: `2504.19449` `[P]` — 输入稀疏 + 跨层**低秩补偿**。SpQt 评述：*"R-Sparse … focus on dense weight layouts and full-precision kernels, leaving integration with quantized models unexplored."*

### [R19] WiSparse: Weight-Aware Mixed Activation Sparsity
- **arXiv**: `2602.14452`（2026-02）`[P]` 摘要；OpenReview *"Submitted to ICLR 2026"* `[O]` — 训练自由、**权重感知、逐块非均匀稀疏比**。**对 Prox 前提最接近的 2026 后继**：直接质疑"均匀稀疏比"假设。**全文未核** `[UNVERIFIED]`

### [R20] DAWS: Dual Activation-Weight Sparsity: A Training-Free Framework for Efficient LLM Model Compression
- **ACL 2026 (Long), pp. 8350–8366**, doi `10.18653/v1/2026.acl-long.378` `[O]`（DBLP/OpenReview 记录）— 训练自由，**三层幅值路由**，同时利用激活与权重稀疏；明确主张注意力与 FFN 的激活/权重稀疏模式**互补**。**已正式发表**。**全文未核** `[UNVERIFIED]`

### [R21] Dynamic Input Pruning and Cache-Aware Masking
- **arXiv**: `2412.01380` v2 `[P]` — 动态**输入**剪枝（针对 SwiGLU 无 ReLU 精确零）+ KV 掩码，**明确以"reducing effective DRAM bandwidth per token"为框架** ⇒ 本方向**最接近字节/带宽口径**的表述

### [R22] Universal Properties of Activation Sparsity in Modern LLMs
- **arXiv**: `2509.00454` v2 `[P]` — 分析为什么 exact-zero 方法不迁移到现代 LLM

### [R23] SVD Contextual Sparsity Predictors for Fast LLM Inference
- **arXiv**: `2603.14110`（2026-03）`[P]` 摘要 — SVD 上下文稀疏**预测器**（ReGLU FFN），**需训练**。"训练型选择器"分支，与 Prox 明确回避的选择器成本形成对照。**数字未核**

### [R24] Activation Sparsity as Training–Inference Co-Design
- **OpenReview `ICLR.cc/2027/Conference/Submission14301`** `[O]` 摘要 — 需预训练固定阈值非线性。★ **其中心发现是本方向的警告**：*"more than doubling model-wide sparsity from 12.71 % to 27.48 % **slightly increases latency**"* ⇒ **没有内核支撑的稀疏一文不值**

### [R25] SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference
- **arXiv**: `2411.12692` v2 `[P]`；**DATE 2025** `[O]` — 对 **ReLU-fied** LLM，只比较**输入与权重的符号位**预测稀疏。最廉价的预测器设计；**不适用于冻结的 SwiGLU 模型**

### [R26] GRASP: Group-based Prediction of Activation Sparsity for Fast LLM Inference
- **DAC 2025**, doi `10.1109/dac63849.2025.11132899` `[S]` — 基于幅值、**分组**的免训练预测，声称比 SOTA 多跳 **11 %**。★ **直接命中"取 k/N 通道时分组该多大"这一任务书问题**最接近的已发表工作。**IEEE 付费墙、未上 arXiv、全文未核** `[UNVERIFIED]`

### [R27] SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference
- **arXiv**: `2307.02628` `[P]` — 跳层/跳 token（早退），**不是 FFN 稀疏**。任务书提到它，此处明确区分以免混淆

### [R28] Collapse or Preserve: Data-Dependent Temporal Aggregation for Spiking Neural Network Acceleration
- **arXiv**: `2603.13810`（2026-03）`[P]` 摘要 — ★ **同硬件族的独立警告**：*"five different sparse computation strategies on **Apple M3 Max** all fail to outperform dense convolution, because **SIMD architectures cannot exploit the fine-grained, unstructured sparsity** of i.i.d. binary spikes."* 不是 LLM/FFN 结果，但同硬件、同类

### [R29] 联合测量的搜索结果
- arXiv API 检索 `abs:(FFN OR feed-forward) AND abs:sparsity AND abs:quantization` 与 `abs:sparsity AND abs:quantization AND abs:(bandwidth OR "memory traffic")`（2026-10-21）**未找到任何在量化模型上做动态 FFN 稀疏端到端测速并给出字节核算的工作** `[P]` 摘要级。最接近者均为**架构类**（非稠密 FFN）或需训练：*Hardware-Native Joint Sparse-Quantization for Trillion-Scale MoE*（`2610.02241`）、*SPARQLe*（`2606.00365`，次精度激活）、*SALCA*（`2604.24820`，稀疏注意力加速器）、*Compression Trinity*（`2608.24070`，学位论文）`[UNVERIFIED]` 全文

---

## C. 函数等价重参数化（gauge transformations）

### [R30] Functional Equivalence in Attention: A Comprehensive Study with Applications to Linear Mode Connectivity
- **arXiv**: `2606.17830` **v1**，2026-06-16 — **"Published at the International Conference on Machine Learning (ICML 2026)"**（arXiv comment）`[P]`
- **作者**：Viet-Hoang Tran, Vinh Khanh Bui, Van-Hoan Trinh, Tan Lai Ngoc, Tan M. Nguyen `[P]`
- **机制** `[P]`：对**带位置编码（正弦 / RoPE）**的多头注意力做函数等价的形式化研究。★ **"sinusoidal encodings preserve the equivalence structure of vanilla attention, whereas rotary encodings significantly reduce the symmetry group, thereby enhancing expressivity"** —— 这为 RoPE 的流行给出了原则性解释。进一步考察位置编码如何影响**线性模式连通性（LMC）**，并用**对齐算法**实证证明"连接性的存在与变化程度关键取决于位置编码"
- **⚠ 归类更正（对任务书）**：**该论文本身不做压缩**。它是**参数对称性 / 优化几何**方向的工作 + LMC + 权重匹配对齐算法。**任务书预设的"用它找更易压缩的变换"在当前证据下不成立**；它对本项目的价值在**定理本身（RoPE 缩小对称群 ⇒ 现代模型上 gauge 自由度本来就更少）**
- **代码**：**未提及** `[UNVERIFIED]`

### [R31] When Local Variance Optimality Is Not Enough: RoPE-Aligned Q/K Rotations for Dynamic 4-Bit Quantisation
- **arXiv**: `2608.13365` **v1**，2026-08-13 `[P]`
- **作者**：Shuhan Wang, Yilin Luo, Nan Xu, Chi Wang Cheung `[P]`
- **★ 定理（逆向结果）** `[P]`：*"We state the **converse result** that, for distinct frequencies, **no other single-head orthogonal map commutes with RoPE**."* ⇒ **与 RoPE 交换的正交映射恰好是逐频率对的 2×2 旋转**；整头 Hadamard **不**交换 RoPE
- **★ 否定性结果（范围明确、站得住）** `[P]`：*"The evaluated head-shared pairwise configuration **does not improve accuracy** in the tested dynamic W4A4KV4 setting. **Across four checkpoints, replacing the full-head Hadamard with this configuration increases perplexity at both short and long context lengths.**"*
- **两个有用的正面** `[P]`：`Composing the pairwise rotation with the Hadamard` **满足 ±0.05 PPL 判据**；`Estimating the shared angle from K alone` 在每个 checkpoint 上都优于 pairwise-only，但**没有补上与整头混合的差距**
- **★ 机制（可复用的洞见）** `[P]`：*"The analytic objective controls a **position-averaged second moment of a pooled calibration covariance**, whereas the **dynamic quantiser sets its step from a tokenwise group range**."* ⇒ **变换的优化目标必须与量化器真正的步长统计量对齐，否则"最优变换"是错的目标**
- **实验设置** `[S]`（PaperVerse 索引）：Llama-3.2-1B / Llama-3.2-3B / Mistral-7B-v0.3 等四个 checkpoint，**dynamic per-token W4A4KV4**，WikiText-2 / **Proof-Pile** / PG19。★ **Proof-Pile 属于本项目关心的"程序逻辑"质量轴**
- **反面材料**：`pith.science/p/2608.13365` 自评 **T0（AI 读，未重算）** `[S]` —— "a scoped negative result that holds up"，并给出"surrogate 与量化器统计量错位"的解释

### [R32] SpinQuant: LLM Quantization with Learned Rotations
- **arXiv**: `2405.16406` **v4**，v1 2024-05-26 — **ICLR 2025** `[P]`
- **作者**：Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort（**Meta**）`[P]`
- **★ 范式意义** `[S]`：确立了"**这些旋转参数化方法在全精度 Transformer 架构中能产生相同的输出**"（输出等价的旋转），并发现**不同随机旋转之间的下游零样本性能差异可达 13 个点**。LLaMA-2 7B 4-bit 与全精度的差距缩至 2.9 点（对比 LLM-QAT 19.1 点、SmoothQuant 25.0 点）；LLaMA-3 8B 上相对 QuaRot 差距最多缩 45.1 %。代码 `facebookresearch/SpinQuant`

### [R33] QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **arXiv**: `2404.00456` **v2**，v1 2024-03-30 `[P]`
- **作者**：Saleh Ashkboos (ETH Zürich), Maximilian L. Croci (MSR), Bo Li (ETH), Martin Jaggi (EPFL), Torsten Hoefler (ETH), Amirkeivan Mohtashami (EPFL), Pashmina Cameron (MS), Dan Alistarh (ISTA & Neural Magic), James Hensman (MSR) `[S]`
- **★ 关键措辞** `[S]`：*"fuses randomized Hadamard rotations into pretrained LLM weights and inserts **cheap online rotations** inside attention and FFN blocks, removing activation and KV-cache outliers so weights, activations, and KV cache all quantize to 4 bits **without changing model output**."*
  ⇒ **注意 "cheap online rotations"：在线旋转存在，只是便宜。** QuaRot 自己把它放在"在线"位置，而非折叠进相邻层

### [R34] PeRQ: Pushing the Limits of Block Rotations in Post-Training Quantization
- **Venue**：**ICML 2026** `[P]` — Brevitas 官方页 `https://xilinx.github.io/brevitas/v0.13.0/papers/perq.html`
- **机制** `[P]`：Permute → Rotate → Quantize。旋转前插入**校准过的排列**以在置换等变区域里重分配激活质量。
- **★ 部署属性（也是本报告指出的矛盾源头）** `[P]`：*"The permutations are calibrated once offline using activation statistics and then **merged into surrounding weight tensors before deployment so they do not incur additional inference overhead**."*
  ⇒ **要"零推理开销"就得把变换折叠进相邻层权重；而相邻层若也已被量化，折叠就会改变它的量化误差。这个矛盾在文献里没有被正面处理**

### [R35] HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning
- **arXiv**: `2601.21626` **v2**，2026-01-29 `[P]` — 指出 PTQ 的悖论 *"low-error, high-loss"* ⇒ **逐层误差小不等于任务损失小**（与 YAQA 的端到端误差界是同一问题的一体两面）

### [R36] HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs
- **arXiv**: `2501.02625` `[P]` — ETH Zürich /ISTA / Neural Magic，量化**训练**中的 Hadamard 辅助

### [R37] KurTail: Kurtosis-based LLM Quantization
- **arXiv**: `2503.01483` `[P]` 摘要 — 基于**峰度**的旋转以缓解激活离群点，4-bit 权重+激活+KV；逐层优化

---

## D. 结构感知降维与低秩

### [R38] A³: an Analytical Low-Rank Approximation Framework for Attention
- **arXiv**: `2505.12942` **v4**（v1 2025-05-19，v4 2026-05-12）`[P]`
- **Venue**：**ICML 2026 poster #63042**（Thu 2026-07-09, KST Hall A #2208）`[P]` — `https://www.icml.cc/virtual/2026/poster/63042`
- **作者**：Jeffrey T. H. Wong, Cheng Zhang, Xinye Cao, Pedro Gimenes, Christos-Savvas Bouganis, George A. Constantinides, Wayne Luk, Yiren (Aaron) Zhao（**Imperial College London + Cambridge**）`[P]`
- **代码**：`https://github.com/DeepWok/a3` `[P]`
- **机制** `[P]`：把 Transformer 层拆成 **QK / OV / MLP** 三个功能组件，为每个组件给出**解析解**，**直接缩减组件内部的隐藏维度**（QK/OV 缩 head dim，MLP 缩中间维），同时最小化**该组件的功能损失**（注意力打分误差 / 注意力输出误差 / MLP 输出误差）。
  - **它针对的正是"改变实际矩阵形状"**：既有限秩方法的两个局限是 (1) 只最小化单层输出误差、不考虑架构特性；(2) **把大矩阵分解成两个小低秩矩阵 ⇒ 引入运行时开销（额外 GEMM kernel 启动与访存）**
  - **"This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads."** `[P]`
  - **解的形式** `[S]`：`f*W_qk,i = (R^{1/2}_{XqXq})^{-1} SVD_r(R^{1/2}_{XqXq} W_qk,i R^{1/2}_{XkvXkv}) (R^{1/2}_{XkvXkv})^{-1}` —— 输入自相关矩阵白化后的 SVD。**无学习参数**，唯一自由度是目标秩
- **数字** `[P]`：同等算力与内存预算下，**A³ 的 LLaMA 3.1-70B 在 WikiText-2 上 PPL 4.69，此前 SoTA 为 7.87（差 3.18）**
- **★ 与 RoPE 的关系（重要）** `[S]`：**Theorem 2 / 3 覆盖的是不带 RoPE 的注意力**；**带 RoPE 时改用 CUR 近似**（§3.4）⇒ **对现代 LLM（几乎全 RoPE），A³ 的闭式解退化为近似**
- **★ 联合测量** `[S]`：Fig. 6 = **HQQ 4-bit 量化下，原模型 vs A³-LLaMA-3.1-8B 的 WikiText-2 PPL 对比**。另有 A³ × KV 压缩、A³ × 微调、A³ × 混合秩分配 `[P]`
- **吞吐口径** `[S]`：Fig. 3 = **A³ vs SVD-LLM，LLaMA-2-13B，A100 40GB，batch=2，seq=2048，Eager/SDPA** ⇒ **A100 + batch=2，不是消费级 batch=1**
- **校准** `[S]`：BF16 模型、**FP64 累加外积**、**SlimPajama**（明确不用 WikiText2 以免污染）
- **★ 反面材料** `[S]`（`pith.science/p/2505.12942`，**T0 机器渲染 + 模拟作者回复**，非正式审稿）：
  - **组件独立最小化是承重假设**：QK 输出调制注意力打分，再被 OV 缩放，MLP 又跟在注意力残差之后；**无跨组件联合目标**，也无误差放大/补偿分析
  - **"解析解"的推导与各组件功能损失的精确定义未在正文给出**（作者称在附录）
  - **4.69 vs 7.87 缺跨组件交互消融、逐组件秩分配敏感性消融，以及"基线是否获等价超参调优"的对照**
  - 自由参数 1 个（逐组件目标秩），假设 1 个（组件可独立最小化）
  - **Google Scholar 引用 6 次** `[S]`
- **★ 本报告自行提出的未讨论风险** `[推算]`：**缩减 head_dim 会改变 RoPE 的频率谱**（`ω_f = θ^{-2f/d}`）；head_dim 128→64 同时改变**基频与频率集合** ⇒ **这不是函数等价的形状变换，而是改变了位置编码语义**。文献未处理

### [R39] Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD Compression for LLMs
- **arXiv**: `2606.19993` **v2**，2026-06-18 `[P]` — **"Accepted at the ICML 2026 Workshop on Resource-Adaptive Foundation Model Inference (AdaptFM), Seoul (non-archival)"**（arXiv comment）`[P]`
- **机制** `[S]`：在纯激活感知基线 SVD-LLM(W) 之上，**把反向信号影响项（backward relevance propagation）折进压缩目标**；将重要性编码为逐元素矩阵，通过 **Hadamard 乘**并入 Frobenius 范数损失
- **⚠ 证据权重必须下调**：**非 archival workshop ≠ 主会论文**

### [R40] LQER: Low-Rank Quantization Error Reconstruction for LLMs
- **arXiv**: `2402.02446` **v3** — **ICML 2024** `[P]` — 低秩**量化误差**重建（Zhang / Cheng / Constantinides / Zhao，与 A³ 同一实验室谱系）⇒ **低秩 × 量化的已发表先例**

### [R41] SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression
- **arXiv**: `2503.12340` — **NAACL 2025** `[P]`；代码 `https://github.com/AIoT-MLSys-Lab/SVD-LLM` `[P]`

### [R42] Palu: Compressing KV-Cache with Low-Rank Projection
- **arXiv**: `2407.21118` **v2** `[P]` — KV cache 低秩投影

### [R43] From Low Rank Gradient Subspace Stabilization to Low-Rank Weights
- **Venue**：**ICML 2025** `[S]` — Jaiswal, Wang, Yin, Liu, Chen, Grama, Tian, Wang；从梯度子空间稳定性看权重矩阵的**非均匀低秩性**

---

## E. 消费端与运行时（与 R1 交叉）

### [R44] BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal
- **arXiv**: `2607.00501` **v1**，2026-07-01 `[P]`
- **作者**：Prabod Rathnayaka, Fabian Waschkowski, Lukas Wesemann（**Base Compute, Melbourne**）`[P]`
- **机制** `[P]`：绕过 MLX / llama.cpp 的框架抽象，**直接写 Metal**（无 PyTorch / MLX / CoreML 依赖）；8 种量化格式（Q2–FP16）
- **实验** `[P]`：**Qwen3 / Llama 3.2 / Gemma 4 家族，Q4/Q8，在 M3 与 M4 Pro 上**（⇒ **本项目硬件族相关**）；**decode 吞吐最高 1.56× llama.cpp、1.35× MLX**；MoE 的 prefill 差距更大；覆盖 sub-1B 到 30B

### [R45] BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators
- **arXiv**: `2607.19438` **v1**，2026-07-21，cs.AR/cs.AI/cs.CL/cs.DC/cs.LG/cs.PF `[P]`
- **机制** `[P]`：M5 每核一颗 **Neural Accelerator**，经 **Metal 4 tensor API** 暴露；手写 **cooperative-tensor kernel**（dense + MoE GEMM、flash-attention prefill），**把计算受限的 prefill 路由到 NAX，内存受限的 decode 路径留在原有专用 kernel 上**
- **数字** `[P]`：**M5 Pro 上 prefill 最高 6.4× llama.cpp、3.9× MLX**；**decode 仅 ≈1.02–1.33×**
- **★ 不可迁移性**：**M4 Pro 无 NAX（NAX 自 M5 起）** ⇒ 该论文的数字**对不上本项目硬件**
- **反面材料** `[S]`：`pith.science/p/2607.19438` 自评 **T0**；*"The M5 prefill numbers are real and internally consistent, but **the attribution to tensor cores is under-supported by the missing ablation and default-only baselines**."*

### [R46] SGLang on Apple Silicon（官方文档）
- `docs/docs/hardware-platforms/apple_metal.mdx` 与 `https://docs.sglang.io/references/environment_variables.html` `[P]`
- **官方路径** `[P]`：**`SGLANG_USE_MLX=1`** + `--disable-cuda-graph`（文档原文："CUDA graph is not relevant for Apple Metal"）+ 可选 `--disable-overlap-schedule`；启动 `SGLANG_USE_MLX=1 python -m sglang.launch_server --model <ID> --disable-cuda-graph --host 0.0.0.0`
- **前置条件** `[P]`：**macOS ≥14、PyTorch 2.13.x、MLX ≥0.32.0**（`srt_mps` extra 安装 PyTorch 2.13.0 + MLX ≥0.32.0）；启动时**先校验框架版本与 Metal 可用性**，不兼容则在解析/下载模型之前停止
- **量化** `[P]`：预量化 `mlx-community/*-4bit` 直接经 `mlx_lm.load` 载入；或 `--quantization mlx_q4/mlx_q8` 走 `mlx_lm.utils.quantize_model`（group 64）
- **⚠ 本机 MLX 是 0.31.2 ⇒ 官方路径当前不可用** `[O]`
- **来源冲突** `[S]`：第三方 2026-07 评测称"SGLang 无官方 Apple Silicon 支持，只有 main 分支上的实验性 MLX 路径（pre-alpha / roadmap #19137）"，并称其 **5.3× 增益是相对 SGLang 自己的 PyTorch-MPS 路径**、样本很小 ⇒ **该 5.3× 不是对 mlx-lm 的比较，不能用来判断端侧引擎选择**

### [R47] 本仓库自身的硬件与运行时读数 `[O]`
- `mx.device_info()`（M4 Pro）：`max_recommended_working_set_size = 19,069,665,280 B = 17.76 GiB`；`memory_size = 25,769,803,776 B = 24.0 GiB`；`max_buffer_length = 14,302,248,960 B = 13.32 GiB`；`resource_limit = 499,000`
- `mlx_lm` 0.31.3 / `mlx` 0.31.2；安装于 `/Users/bozhongxiao/code/.venv/lib/python3.12/site-packages/mlx_lm`
- **R1 实测锚点（修正口径）** `[M]`：decode 有效带宽 **124.6 GB/s = 273 GB/s 峰值的 46 %**；prefill 有效算力 **2.06 TFLOP/s**；每 eval 提交/同步成本 **0.2–0.4 ms**；LM head 311.2 MB = 0.6B 模型单步流量的 26 %；堆权重 n=1 慢 29 %；深度 28→7 decode 收益 0 %。⚠ **同一段 28-block 代码两轮相差 1.9× ⇒ 绝对毫秒不可跨报告引用**

### [R48] 第三方能耗实测（低位宽的反面证据）
- 4 种 NVIDIA 架构、360+ 组 NVML 直测（10 Hz 采样、FP16 基线配对、每组 10 轮 decode）`[S]`：**小模型量化后每 token 能耗反而更高**；节能边界 T4 ≈2.1B / RTX 4090 ≈3.7B / RTX 5090 ≈4.8B；**bitsandbytes `LLM.int8()` 在所有受测消费级 GPU 上一��没省过电（+50 % ~ +242 %）**。机制：**反量化是每次矩阵乘都多一遍、与模型大小无关的固定计算开销**。⚠ 未在 Apple Silicon 复测

### [R49] 低位宽的边际收益递减与格式差异（第三方）
- `[S]`：H100 上 8-bit 给 1.3–1.5×，**4-bit 只再多给 1.1–1.2×（带宽已饱和）**；消费级 RTX 4090 上 4-bit 配好内核可达 2–3×
- `[S]`：*"in llama.cpp, 'fewer bits' doesn't automatically mean 'more speed' — different quantization formats can trigger different kernel paths."*（Shapelearn / Qwen3-30B-A3B 经验）
- `[O]`：R1 的 GSQ-RCO 模型卡位宽扫描（见 [R10]）——**Q2_0 vs IQ2_XS 在近等 bpw 下 decode 差 1.34×、端到端差 1.9×**

### [R50] 2-bit 的独立横向比较
- **Bielik-Q2-Sharp**（Polish 11B Mistral 基座，6 种 2-bit PTQ，统一校准同一 Hessian）`[S]`：QuIP# E8P12 **71.92 %**（vs IQ2_XXS 72.07 %，3.26 GB vs ~2.6 GB）；**QTIP 最佳 per-bit 效率**，35 % 更小体积（3.27 GB）匹配 VPTQ 质量，**~2.4 bpw 下 79.4 % MC 准确率**。⚠ 比较对象全为 PTQ，**无 trellis / Metal 项**
- IQ2_XXS 在 llama.cpp 的 2-bit 横向测试中吞吐最高（47.89 tok/s）但**"2-bit 量化已出现事实修正错误"** `[S]`

---

## F. 本轮未能核实清单（集中列出）

| 项 | 尝试过什么 | 状态 |
|---|---|---|
| Proteus 的 arXiv id / 官方仓库 / 硬件与 batch | arXiv API 四组检索（`ti:"Lookup-Free Trellis-Coded Quantization"`、`abs:"MUL-BAL"`、`ti:"Proteus" AND abs:"trellis"`、`all:"Proteus" AND abs:"quantization" AND abs:"2-bit"`）；最后一次因 429 限流失败 | `[UNVERIFIED]`（仅拿到 ICML 2026 会议页摘要） |
| CurveTQ 正文（硬件、batch、逐模型 PPL、trellis 解码器） | `arxiv.org/html/2610.09212` 返回 "No HTML for '2610.09212'" | `[UNVERIFIED]`（仅摘要级 `[P*]`） |
| XOR-Trellis 正文（模型、硬件、bit-width、数字） | 仅 arXiv API 摘要 | `[UNVERIFIED]` |
| SpQt Figure 2–10 | HTML 为 glyph-path SVG；PDF 仅读第 3–9 页 | `[UNVERIFIED]`（1.22×/1.51×/1.78×、<0.2×、17 % 均引自正文叙述） |
| Prox 的 OpenReview PDF 正文 | forum API 需 challenge token | `[UNVERIFIED]`（**2.07× 仅为元数据摘要，未读 PDF；本报告不采用**） |
| Prox 2.07× 与 1.99× 的差异来源 | 两份不同文档，摘要未说明 | `[UNVERIFIED]` |
| Prox 的 FP16 内核在散布通道上的确切字节行为 | 论文描述掩码但未述访问模式 | `[UNVERIFIED]`（§3.3 算术是问题上界的估计） |
| GRASP (DAC 2025) 全文 | OpenAlex 元数据 + 摘要；IEEE 付费墙、未上 arXiv | `[UNVERIFIED]` |
| WiSparse / DAWS / SVD-CSP / Co-Design / 2603.13810 全文 | 仅摘要 | `[UNVERIFIED]` |
| 两篇 FFN 稀疏工作的引用数与 venue 元数据 | Semantic Scholar API 429 限流 | `[UNVERIFIED]`（**既不能确认也不能排除存在未被索引的第三方复现**） |
| SpQt 是否被投给别的会议 | 找不到 OpenReview 记录；只有 PDF 的 "Preliminary work. Under review." | `[UNVERIFIED]` |
| 任何 MLX / vLLM 实现（FFN 稀疏 / zigzag） | GitHub code 搜索需认证（401），仅能做未认证仓库/issue/PR 搜索 | `[UNVERIFIED]` |
| trellis 格式在 llama.cpp / MLX 中是否已合入 | 未见证据，但**未做穷尽性源码审计** | `[UNVERIFIED]` |
| "Do Transformers Need Three Projections?"（ICML 2026） | 仅见二手提及其问 W_Q/W_K/W_V 是否都必要 | `[UNVERIFIED]`（**本报告不作任何主张**） |
| GQA / QK-Norm / 门控 FFN 与 gauge 变换的精确条件 | 系统检索未找到正面处理的论文 | **`[文献空白]`（非"我未找到"）** |
