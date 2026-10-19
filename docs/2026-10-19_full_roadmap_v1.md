# 预训练 Transformer 的离线稀疏依赖图优化：首轮研究实验方案

**版本**：2026-10-19 / 研究方案 v1.0  
**研究定位**：验证性研究，不是发布计划、不是完成的算法、也不是对本地开发者现有工程结构的指令。  
**实验环境**：Apple Silicon M4 Pro、24GB 统一内存，实务上可用内存按 <10GB 预算；优先 BF16 + MLX。  
**核心约束**：冻结预训练权重；依赖发现与搜索均在离线进行；推理只执行已经确定的有向无环计算图（DAG）。不要求模型间的图结构泛化。

---

## 1. 要回答的科学问题

能否对于**一份具体的、已预训练的 decoder-only Transformer**，不重新预训练、不进行大量微调，仅通过校准样本和有限图结构搜索，找到一个比原始逐层串行链**计算关键路径更短**、且在未参与搜索的数据上保留模型能力的**静态跨层依赖 DAG**？

与「两层一组」不同，此处允许：

- 非相邻层从同一个已有状态并行执行，例如 2、3、5、6；
- 某层只等待**它真正需要的多个上游结果**，即上游就绪立即具备启动条件，不等待其余兄弟分支；
- 不同层的残差更新以**可定义、可校验**的规则汇合；
- 第二阶段在维持无环前提下尝试有限层重排；
- 每个模型允许有自己的计算图，暂不要求所有架构通用。

本研究**不声称**任意层都可并行、不声称理论深度下降等于实际推理同倍加速，也不声称 ONNX/MLX 本身能自动发现近似等价的 DAG。

## 2. 基础数学与待验证假设

对预归一化 Transformer，先以**完整 Decoder Block** 为原子节点；定义残差更新（模型实际可能还包含独立 Attention 与 MLP 子残差，实验实现应保持真实语义）：

\[
\Delta_i = F_i(h_{i-1}),\qquad h_i=h_{i-1}+\Delta_i.
\]

对于冻结权重的候选静态 DAG，用原始 embedding 后的状态 \(h_0\) 以及部分前驱的残差更新构造节点输入：

\[
\widehat{\Delta}_i=F_i\!\left(h_0+\sum_{j\in P_i}\widehat{\Delta}_j\right),\qquad
\widehat h_{\rm final}=h_0+\sum_i\widehat{\Delta}_i.
\]

这里的 \(P_i\) 是离线搜索产生的前驱集合；这是一种**候选语义**，不是已被证明最佳的唯一实现。实际开发可研究固定或少量可校准的边权，以及为某些节点设置局部基准状态，但必须先确保在全前驱情形能恢复原模型。最终 LayerNorm、LM Head、attention mask、RoPE、cache 处理保持原模型语义。

**首先验证的不是“能否找到漂亮的图”，而是下面四点：**

1. 从原始模型转写成等价的图，是否没有引入实现误差；
2. 部分上游更新可以被省略或提前的假设是否在真实数据上成立；
3. 组合变换后误差是否依旧受控，且优于同等条件的删层/固定分块；
4. 图的关键路径改善是否能够在目标硬件转为真实加速。

> **特别提醒**：某层与前一层「相似」不是「可以删除依赖」的充分条件；把第 3 层计算结果交给第 4 层，也无法自动补回第 3 层在提前运行时没收到的中间信息。模型可能出现高阶/组合误差。

---

## 3. 模型与环境

| 用途 | 模型 | 权重与结构 | 选择理由 |
|---|---|---|---|
| 首轮主实验 | [`mlx-community/Qwen3-0.6B-Base-bf16`](https://huggingface.co/mlx-community/Qwen3-0.6B-Base-bf16) | BF16，约 1.19GB，28 层 | 本地便宜、多次前向可承受，Base 模型适合 token 级 PPL / Logit 对照 |
| 首轮复核（通过第一道门槛才做） | [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | BF16，约 3.44GB，28 层；**需要验证/执行 MLX 转换** | 同深度、宽度不同；验证方案不是仅在 0.6B 上成立 |

**重要纠正**：现成的 [`mlx-community/Qwen3-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-1.7B-bf16) 是 **post-trained Qwen3-1.7B** 的 MLX 转换版，**不等于** `Qwen3-1.7B-Base`。如果保持 Base 对 Base 的对照，必须从 Base 检查点转换，不能错用该现成仓库。

**推荐底座**：[MLX](https://github.com/ml-explore/mlx) + [MLX-LM](https://github.com/ml-explore/mlx-lm)。锁定包版本、模型 revision/commit、dtype、tokenizer、随机种子、batch、上下文长度、运行温度及内存状况。无需部署新的本地 Agent/Harness；由本地开发者选择既有工作方式。

**不采用** 4-bit GGUF/MLX 量化进行首轮校准；首轮禁止 LoRA、重新训练、量化噪声、无约束自动权重变动。主研究是**改变计算图**。

## 4. 数据集：明确角色、固定抽样与成本

在小内存与有限时间条件下，先按以下「建议起始预算」抽取；它们不是统计上必然充分的样本量。把固定样本 IDs、抽样种子和 tokenized 数据摘要记录下来。校准、候选选择、最后测试不能交叉泄露。

| 数据源 | 建议首轮量 | **唯一角色** | 用法与注意事项 |
|---|---:|---|---|
| [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 校准 128×256 token；另留验证 128×256 token，完全不重叠 | **依赖测量 / 搜索 / 验证** | 流式采样，不下载完整 10B token；避免重复文档。可从 32×256 开始调试 |
| [WikiText-2 `wikitext-2-raw-v1`](https://huggingface.co/datasets/Salesforce/wikitext) | test 最多约 256×256 token（以实际可切出的完整样本为准） | **独立 LM 测试** | 只用 test split；保留原始文本，按既定规则构造连续 token 窗口；不可用来挑选结构 |
| [C4 `en` validation](https://huggingface.co/datasets/allenai/c4) | 最多 128×256 token | **分布外补充测试** | 以 streaming 读取，避免下载海量语料；不纳入校准 |
| [GSM8K `main` test](https://huggingface.co/datasets/openai/gsm8k) | 起步 64 题 | **推理敏感性检查** | 0.6B Base 可能正确率接近地板；同时记录教师强制下的答案/推理 log-likelihood，不能把它当作题目正确率 |
| [MBPP / Google Research](https://github.com/google-research/google-research/tree/master/mbpp) | 起步 64 题，正式采用官方 test ID | **代码敏感性检查** | 可观察给定参考代码的 NLL；实际 pass@1 只在原始 Base 模型有足够通过率时解释，否则是地板指标 |

**数据处理协议**：统一由 Qwen3 tokenizer 编码；记录是否有 BOS/EOS、padding、packing 及滑动窗口边界；对 loss 采用只计预测 token 的 causal shift；每个数据集使用一致的截断与遮罩。不要把不同 tokenizer 计算出的 perplexity 直接比较。输出包括按 token 的 NLL，PPL 只是其指数形式。

**评估精度指标**：

- **首要**：独立测试 NLL/PPL 的相对变化，尤其与删层 baseline 同条件比较；
- **结构保真**：与冻结教师模型的逐 token logits KL（计算在模型真正输出处分布；如用 Top-K 近似则清楚标注近似）、top-1 一致率、正确答案 token 的 log-prob；
- **内部诊断**：相同位置残差更新角度与范数、节点输入分布漂移、最终层 cosine / NMSE，不能以中间 cosine 取代最终质量；
- **生成行为**：固定 16–64 个测试 prompt 的 greedy decode，检查连贯性和重复/失控；与教师文本不完全相同不自动等于失败；
- **后续硬件性能**：分别报告 prefill（多个输入长度）与 batch=1 decode（生成 token），暖机、同步后重复计时，报告中位数、分位数与内存峰值。

为避免 151,936 vocab 的 full-logit 保存占据内存，使用分批/流式聚合；可考虑 Top-K teacher logits + 对数和归一化统计作为**筛选代理指标**，最后复核时计算严格指标。缓存应写盘，且不得把参考模型和候选模型作为两个完整独立权重副本同时装入内存。

---

## 5. 一定要做的四组基线

| 编号 | 基线 | 目的 |
|---|---|---|
| B0 | 原始 BF16 串行 28 层 | 黄金标准：质量、延迟、内存 |
| B1 | **保持语义完全等价**的图执行版 | 验证图系统正确性：不改顺序/不漏任何信息时必须复现 B0 |
| B2 | 跳过/删除对应数量的层（固定层数 & 计算预算） | 防止把「有损删层」误判成 DAG 并行的价值 |
| B3 | CQIL / True Depth 风格的相邻或连续分组并行 | 测试非局部 DAG 是否真正超越简单分块 |

只有本方案候选 `G` 显著优于 B2、B3，才能声称「复杂跨层依赖搜索带来了独立增量」。B2 要至少包含简单中间层删除，以及同样搜索预算的优化删除基线，否则对照不公平。

## 6. 分阶段实验：门槛、产物、停止条件

### 阶段 0：复现原模型与语义等价性（先于一切搜索）

- 固定一个模型版本，用 MLX-LM 运行 BF16 Qwen3-0.6B-Base；得到 token NLL、logits、prefill 与 decode baseline。
- 将 decoder layer 视为可调用冻结模块，用残差更新量表示；将原始串行执行改写为新的图接口，**保持所有原有依赖**。
- 检查 LayerNorm / Attention / MLP / RoPE / causal mask / KV cache / 最终 norm / LM head 全部和参考实现一致。
- 同一输入上，B1 对比 B0 的逐 token logits、NLL、greedy 生成。浮点加法结合律和 MLX 执行顺序可能造成极小差异；用基线数值噪声设容差并报告，不武断要求逐 bit 相同。

**停止条件**：如果等价性失败，**停下来修正测试实现**；禁止进入结构搜索。这一步产生 reproducibility report。

### 阶段 1：8 层窗口中的少量结构干预

- 初始窗口建议选**中间连续 8 层**（例如按 0-based 的 10–17 层）；不要根据 test set 选择窗口。稍后再换第二个中间窗口检查稳定性。
- 固定窗口外层的完整串行执行。对窗口内层做**候选输入替代实验**：让目标层读取窗口中更早的残差状态、或由多个选定前驱的残差更新构成的状态；记录质量退化。
- 先做：① 层跳过；② 两两相邻并行；③ 非相邻节点并行；④ 一节点有多个前驱且无须等待不相干分支。
- 不用余弦相似度直接推断边：要在**真正修改后的前向图**上评分，并对几个组合做交互检查。
- 不必真的「穷举所有 DAG」。先设候选 ≤30–50 张，在 32×256 token 上快速筛选；仅把表现最佳且结构不同的少数候选升级到 128×256 token 验证。

**主要产物**：至少一个能运行的非连续依赖候选图、每条候选边的干预报告、失败类型分析。

### 阶段 2：受控的图搜索（仍只在 8 层窗口）

把结构编码为：节点、前驱集合 `P(i)`、输入汇合规则、可选的有限重排。先保持原始拓扑顺序，降低无环约束成本；确认其收益后再开放相邻交换与少量非相邻交换。

- 初始候选由 CQIL 风格局部敏感度与更大跨度的输入替换探针产生。
- **局部搜索**：加边/删边、交换指定节点输入、合并或拆开并行组；对受影响子图做增量复测。
- **组合搜索**：借鉴 EvoPress 的选择—变异—评估思想；同时保留几个帕累托候选（质量 vs 关键路径），不要把一个质量指标的最佳点当成唯一答案。
- 用真实图运行质量评估，不能把单边效应相加后充当组合质量。
- 图约束：无环、确定性输入、无孤立未使用计算、明确最终残差汇合、错误缓存状态不可混入，图与权重版本必须绑定。
- 限制初轮图评估预算（例如先 50 次，效果明显再提高到 100–200 次）；每一个候选必须记录计算耗时，而不是无限自动重试。

**产物**：候选图 JSON / 可视化、帕累托质量-关键路径表、与 B2/B3 的对照、已用校准 token 与 wall-clock 成本。

### 阶段 3：完整 28 层 + 独立测试

仅当窗口内确有明显优于 B2/B3 的候选再扩展。

- 从中间层区域开始，在原始串行首尾层周围保守搜索；必要时按窗口逐步推广。
- 新图定稿后**冻结**，才在 WikiText-2 test、C4 validation、GSM8K 和 MBPP 上做评测；如测试结果不好，不能再用同一测试集持续挑图。
- 单独分析 prefill/decode，KV cache 的每层状态必须与**新图自己的语义**一致；不能假装串行原模型的 cache 直接适用。
- 在 Mac GPU 上做真正 wall-clock 测速。MLX 的惰性求值与设备同步必须处理好；一个更短的逻辑 DAG 在单 GPU 上不一定更快。

### 阶段 4（可选）：1.7B Base 复核

- 使用 `Qwen/Qwen3-1.7B-Base` BF16 转换成 MLX；与 0.6B 用一致的数据抽样和指标。
- 复用**校准方法**，不强迫复用 0.6B 找到的**图结构**。
- 内存余量与稳定性不足时停止，不要自动触发量化或者交换到巨大的 PyTorch/CUDA 环境。

---

## 7. 明确可以抄哪些：按实际复用级别分类

### A. 能直接用作底座 / 库（不要重写）

| 项目 | 链接 | 直接拿来做什么 | 不做什么 |
|---|---|---|---|
| **MLX / MLX-LM** | https://github.com/ml-explore/mlx 、https://github.com/ml-explore/mlx-lm | Apple GPU 张量计算、BF16 模型加载、tokenizer、模型代码、常规 decode | 其现成模型 forward 不会自动支持我们任意跨层 DAG，**模型执行路径需要研究性改造** |
| **HuggingFace datasets** | https://huggingface.co/docs/datasets/ | 数据集流式读取、切分、缓存、版本化 | 不代表自动避免校准/test 污染 |
| **标准评测定义（lm-eval-harness）** | https://github.com/EleutherAI/lm-evaluation-harness | benchmark task、prompt/scoring 协议 | 它的现成后端未必兼容修改后的 MLX 图；可复用**评测定义**，不必强装完整后端 |
| **mlx-lm-lens（参考性工具）** | https://github.com/adeelahmad/mlx-lm-lens | 在 Apple Silicon 上做 per-layer 观察与消融的示例 | 未核验本项目架构、版本的完全兼容性；不是 DAG 搜索器。若增加集成成本，不要安装 |

### B. 已开源，但主要是**改造/移植思路**；不要硬搬 CUDA/PyTorch 整套环境

| 项目 | 链接 | 抄/改的部分 | 不建议直接移植的部分 |
|---|---|---|---|
| **CQIL (ACL 2024)** | https://github.com/Photooon/CQIL | 用输入替换测准独立性、邻层共享输入、Attention bypassing、准确率/加速对照 | 论文仓库主要支持早期 LLaMA 与 Qwen1.5 的 CUDA/分布式路径；不能直接用于 Qwen3 + MLX |
| **Leveraging the True Depth / Effective Depth** | https://github.com/noctrog/effective-depth | 相邻层并行、结构敏感性验证、冻结权重 benchmark | 原实现偏向双 GPU 的特定 LLaMA 体系 |
| **Transformer Layers as Painters** | https://github.com/floatingbigcat/transformer_layers_as_painters | 层交换、逆序、并行干预矩阵及「不同任务敏感度」实验范式 | 原代码改造较旧 eval/模型，优先移植具体对照测试，不复制整套环境 |
| **EvoPress (ICML 2025)** | https://github.com/IST-DASLab/EvoPress | `evo_drop_search.py` 的候选选择/变异/选择压力、baseline、逐级数据预算与 Top-K KL 代理指标 | 它原生是 PyTorch/CUDA 压缩，不是 MLX DAG；不能原封不动运行 |
| **SNLP (2026)** | https://github.com/phymhan/nanochat-snlp | 残差并行近似的失败/成功边界、前缀累积、可选迭代修正，对比误差/延迟 | 依赖特殊训练配置；完整重训练不是本实验目的 |
| **AutoCircuit** | https://github.com/UFO-101/auto-circuit | 激活 patching、边消融、候选边归因、对组合效应的警示 | 面向解释性、原有框架可能与 Qwen3 GQA/MLX 不兼容；先看算法而非强行绑定运行时 |
| **ACDC** | https://github.com/ArthurConmy/Automatic-Circuit-Discovery | 贪心删边、重要计算回路的发现策略 | 作者仓库自己提醒实现棘手，建议看 AutoCircuit；不作为首轮主依赖 |
| **Delta Attention Residuals** | https://github.com/wdlctc/delta-attention-residuals-code | 跨层残差增量和汇合的思想 | 不要照搬其动态 attention routing、存储所有历史候选状态或额外训练机制 |

**“抄”的边界**：实现复用前核对每个仓库 LICENSE、commit 和第三方依赖；研究思想可借鉴，代码复制要满足许可和标注要求。不要装未经验证的插件来代替研究实现。

### C. 应由本实验自己研究、无法指望现成库直接提供的部分

1. **冻结 Qwen3 的真实 DAG 输入语义**：如何由多个前驱构建合法且质量良好的隐藏状态，并保持残差、Attention/KV、最终输出正确。
2. **跨层候选图表示**：静态无环、权重版本绑定、可序列化、可重放、可检查存活激活。
3. **依赖探针**：测量`目标层接收哪组上游增量`对真实模型 loss / logits / 生成的影响，区别于简单相似度与解释性 patching。
4. **带硬件代价约束的搜索目标**：关键路径、总算子工作量、fan-in、峰值活跃激活、实际 GPU wall-clock 的权衡。
5. **候选 DAG 的真正执行/基准对照**：必须运行被搜索出的新 forward，不能把「理论图深度」当作速度。

这些是研究任务，不是建议你另起一个庞大推理框架。

---

## 8. 评价与停止条件（预设，运行前可讨论调整）

**门槛 0：正确性** — B1 与 B0 在相同 token、模型版本、mask、dtype 下具有可解释的极小数值差异；不达标即停。

**门槛 1：存在可用的局部非串行结构** — 至少一张 8 层窗口图，在独立验证样本上，同时显示（a）关键路径更短，（b）最终 NLL/KL 可接受，（c）相对于删层和简单分块存在明确优势。若没有，记录负结果，不扩张 28 层搜索。

**门槛 2：控制损失并给出帕累托曲线** — 用质量损失预算（例如 NLL 相对增长不超过 1%、3%、5%，作为候选探索档位，**不是普遍接受的质量阈值**）比较图深度收益。必须报告不同 task 的损失，不只报告总体平均。

**门槛 3：真实收益** — 在 warmed-up、同步的 Mac 测试中提供 prefill 与 decode 两项实测。如果质量可接受且图深度大幅缩短但真实延迟无改善，结论是**依赖深度可缩短，单 GPU 性能暂未兑现**，不是宣告实际推理提速。

**不要宣称的成果**：仅有 cosine 很高、层数图看起来很少、删层也能达到类似精度、只在校准集改善、没有端到端延迟测试、单组 0.6B 结果推广到所有大模型。

## 9. 最终交付物与研究记录

第一轮最终要求出现以下研究产物（格式由本地选择，不要求大规模新系统）：

- 数据 manifest：数据集版本、分区、文档 IDs、tokenizer、种子与授权说明。
- 研究环境与基线：硬件、内存、MLX 模型与库版本、B0/B1 等价性报告。
- 干预记录：候选图、改动的依赖、质量变化、计算/时间代价。
- 对照报告：B0/B1/B2/B3 与 DAG 候选在验证和独立测试上的结果。
- DAG 图文件与可视化：前驱集合、汇合规则、关键路径、可执行验证状态。
- 结论必须在 `支持 / 部分支持 / 未支持 / 因实验缺陷无法判断` 中选一种，并写出下一步最便宜的决定性实验。

建议在早期保留精简的失败实验与负结果；这个题目很容易因为长期搜索过拟合而误判成功。

---

## 10. 给明天本地助手的研究边界

**请先检查现有环境、可以复用的模型和工具，再确认阶段 0 的最小实施范围；不要在未确认前引入新 Harness、插件、后台守护进程、隐形会话、付费 API，也不要自行开始数小时的搜索。**

首个验收目标只需回答：

> 在 Qwen3-0.6B-Base BF16 上，能否通过 MLX 编写一个与原始串行模型数值等价的残差更新图执行接口，并以一次简单的共享输入并行变换跑出可解释的最终 NLL 差异？

这一关通过后，才进入非相邻层与多前驱依赖搜索。

---

## 参考论文 / 仓库

1. [CQIL: Concurrent Computation of Quasi-Independent Layers, ACL 2024](https://aclanthology.org/2024.acl-long.394/)
2. [Leveraging the True Depth of LLMs](https://arxiv.org/abs/2502.02790)
3. [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298)
4. [EvoPress: Accurate Dynamic Model Compression via Evolutionary Search](https://github.com/IST-DASLab/EvoPress)
5. [ACDC — Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997)
6. [AutoCircuit](https://github.com/UFO-101/auto-circuit)
7. [SNLP: Layer-Parallel Inference via Structured Newton Corrections](https://github.com/phymhan/nanochat-snlp)
8. [MLX-LM](https://github.com/ml-explore/mlx-lm)
9. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
10. [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
11. [C4](https://huggingface.co/datasets/allenai/c4)
12. [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
13. [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

**研究者立场**：此前相关论文已有冻结模型并行和自动搜索先例；本项目仍需用真实实验决定「离线稀疏 DAG 搜索是否形成额外贡献」。不能把现有研究路线包装成已经证明的大规模层并行突破。
