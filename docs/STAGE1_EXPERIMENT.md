# 第一阶段详细研究方案：串行—并行残差差异测量

版本：2026-10-19 v1.1 · **设计文件，不是已完成的实测报告**

## 0. 决策背景及严格范围

用户的 Plan A 是冻结一个既有 LLM，离线识别层依赖后将线性层链改写为稀疏 DAG。Plan B 则更直接：让原始昂贵层尽量并行，增加一个小型共享误差预测网络，必要时允许该小网络串行累积状态。**本阶段先不训练这个小网络，只回答它到底要补多少、补偿目标是否存在简单统计结构。**

本阶段要回答六个具体问题：

1. 第 `i` 层如果不使用真实的串行输入，而使用共同 anchor，其残差更新改变多少？
2. 误差主要集中在早层/中层/晚层、Attention/MLP，还是分散在全部层？
3. 多层的补偿目标是否方向相近、抵消、强化？简单的低秩/少系数表征是否具有**离线重建潜力**？
4. 与真正可部署的多组逐段推进模型相比，仅用教师 anchor 的局部测试究竟乐观多少？
5. 不训练的纯并行近似，最终 NLL、logits KL、top1 一致性和 token 生成会退化多少？
6. 损失来自取消层间依赖，还是实际上跟删层已经没有明显差别？

**暂不回答：** 自动图搜索能否找到最优图、辅助补偿器是否可训练、校准能否泛化到不同模型、单 GPU 是否能实际加速。后续的真实加速必须用真正的并发执行和完整的 prefill/decode 含 KV cache benchmark 验证。

## 1. 确定模型/精度/硬件

- 主模型：[`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) 的 MLX BF16 转换 `mlx-community/Qwen3-0.6B-Base-bf16`；对照记录各自 Git revision（官方配置含 28 层，hidden size 1024，GQA 16Q/8KV，vocab 151936，tie embeddings）。
- 在苹果 M4 Pro 24GB 统一内存中单份加载；实务上将模型进程及中间缓冲预算控制在约 8GB 内，遇到内存压力减少 token 长度/样本批次，而非改为 4bit。
- `mlx-lm` 的 Qwen3 block 精确形式：`a = Attn(RMSNorm(h))`; `u=h+a`; `b=MLP(RMSNorm(u))`; `Block(h)=u+b`。注意这里不可以把原始 Attention、MLP 两项当成彼此独立。
- LayerNorm 后的最终头、RoPE 位置、causal mask、embedding tie、tokenizer 必须沿用模型参考实现。
- 本阶段一次处理一段文本；**所有实验禁用 KV cache，采用完整前缀的 teacher forcing prefill**。不应声称已验证了自回归 decode 状态正确性。

## 2. 精确定义三种状态与两类误差

给定 token 输入 `X`，嵌入后的 `h_0=Embed(X)`；预训练完整 block 为 `B_i`，i 以 0..27 编号。

原始串行教师：

`h_{i+1}^T = B_i(h_i^T)`；`d_i^T = B_i(h_i^T) - h_i^T`。

**独立共享输入的并行反事实：** 设某并行组为 `[s,e)`，先获得一个共享 anchor `a`，对组内每层计算 `d_i^P(a) = B_i(a) - a`；该层的目标差 `c_i^*(a) = d_i^T - d_i^P(a)`。当 `a = h_s^T` 时，我们称它为 **teacher-conditioned local**；其余节点不必同时真实运行，只诊断该组理论可并行程度。

**可部署的逐组推进：** 输入当前候选状态 `\hat h_s`；组内共享这个状态，所有 block 可独立构图，然后将 `d_i^P(\hat h_s)` 求和形成 `\hat h_e = \hat h_s + \sum_{i=s}^{e-1}d_i^P(\hat h_s)`。下一组从 `\hat h_e` 开始。注意不能在下一组偷用 `h_e^T`，否则会泄漏教师信息。单层组必须调用原始 `B_i(h)` 以还原等价执行。

- **局部误差** `c_i^*(h_s^T)`：衡量在*正确的组入口*上，提前执行本层漏掉的影响。它不是辅助小网络可以在运行时直接读取的真值。
- **自由运行误差**：`\hat h_e - h_e^T` 和最终 `\hat y-y_T`，同时包含前序组误差传播，代表部署风险。
- **差分不等于小函数**：`||c_i||` 小不证明可预测性高。量化其低秩谱只说明离线压缩潜力，不等于给定输入后可被简单网络学习。

每层另可拆出 `d_attn`、`d_mlp` 两部分做扩展诊断；由于 MLP 的输入含 Attention 输出，只有**完整 block** 的共享输入并行是真正的第一轮对照，不允许把内部子层随意视为独立。

## 3. 对照及测试矩阵（严格区分真实与诊断）

| ID | 模式 | 目标 | 是否可直接部署 | 会不会真并发 |
|---|---|---|---|---|
| T0 | 官方 MLX-LM `model(X)`，原始串行 | 模型/质量基准 | 是 | 不涉及 |
| T1 | 用 28 个原模块手工串行重建 | 数值和语义等价验收 | 是 | 不涉及 |
| L2 | 教师 anchor 下相邻两层的 `d^P` | 测局部差值（组 size=2） | 仅局部测量 | 数学上独立 |
| L4/L8 | 教师 anchor 下 4 / 8 层 `d^P` | 测误差增长和层位置 | 仅局部测量 | 数学上独立 |
| P2/P4/P8 | 对完整 28 层按组 2/4/8 推进 | 最小部署可行性 | 是（仍需高效执行器） | 目前代码未实际并发 |
| P28 | 所有 28 层共享 `h0` 一次汇合 | 最激进压力测试 | 数学上可部署 | 目前代码未实际并发 |
| W-early/mid/late | 只把 [0,8)、[10,18)、[20,28) 改造，其他保持串行 | 定位敏感区域 | 是 | 目前代码未实际并发 |
| D | 跳过同一区间中对应数目的层 | 排除“删层也一样好” | 是 | 不涉及 |

非相邻层集合（如 2、3、5、6）需要明确每个节点的输入、汇合规则与拓扑依赖，**本阶段不假装它们已在同一 anchor 下能安全插回原链**。先从组内共享输入开始测，再根据差值决定是否进入 Plan A 的 DAG 原型。

### 组融合细节

从同一个 anchor 获得的是残差更新 `d=B_i(a)-a`，融合时：`a + sum(d_i)`。不准直接 `sum(B_i(a))`，否则共享 anchor 会被加进去多次。

MLX BF16 逐步相加与 float32 中间归约可产生不同舍入结果，须锁定归约次序并将其作为配置记录；正式对照保留两个变体用于分析舍入敏感性。分组为 1 时严格使用参考层直接计算，避免不必要的重新求和。

## 4. 数据集与抽样协议

选小规模、可信、无需大型下载的基准先判定误差。整个 Stage 1 **不训练、也不需要训练/微调数据**；“校准集”只是选图或选并行区间的观察集合。

| 数据集 | 用途 | 起始预算 | 处理 |
|---|---|---|---|
| [`Salesforce/wikitext`, `wikitext-2-raw-v1` train](https://huggingface.co/datasets/Salesforce/wikitext) | 模式筛选/内部诊断 | smoke 8×128；pilot 32×128；最终诊断最多 64×256 | 原始文本拼接后按 tokenizer 切为互不重叠长度 `T+1` 的窗口，固定 seed+起点。取 `T+1` 保证有 `T` 个被预测 token |
| WikiText-2 validation | 用于比较配置，调节选项 | 32×256 | 不与 train 混用 |
| WikiText-2 test | **冻结配置后仅一次最终核验** | 64×256 | 冻结候选和评价代码之前不读取结果 |
| [`openai/gsm8k`, main train](https://huggingface.co/datasets/openai/gsm8k) | 推理风格的文本压力测试，非训练 | 32 题文本（问题与标准推理答案拼接） | 只测 teacher-forced NLL/KL，不以 Base 模型解题准确率作 Stage 1 主要成功条件 |
| GSM8K main test | 最终异常检查（可选） | 32 题 | 严格留作最后阶段，不用于选图 |

不在首日引入 C4/FineWeb-Edu 的大规模 streaming；如局部趋势清晰才跨语域扩展。代码数据准备工具必须记录 dataset repo、config、split、下载时 revision、文本拼接规则、样本 SHA256、tokenizer ID、窗口 token IDs SHA256，防止之后误换数据。

因 WikiText-2 的逐行切片经常短于 256 token，不应仅筛选长段落（会产生长度偏差）；可保留 split 内顺序，使用 `\n\n` 连接原始段落，再从连续 token 流按非重叠窗口抽样。选好的 token IDs 可保存 JSONL；不将原始大段受版权保护文本提交 GitHub。WikiText-2 许可见其 dataset card。

评价的 token NLL 使用 shift：第 t 个 logits 预测第 t+1 个 token，完全保持相同 token IDs 和 causal mask。对所有候选和教师使用相同长度与位置索引；不得把 padding token 纳入 loss。

## 5. 收集量与统计定义

**每 token / 每层内部统计**：

- `||d_i^T||_2`、`||d_i^P||_2`、`||c_i^*||_2`、`||c_i^*||_2 / (||d_i^T||_2 +eps)`；同时报告 *绝对* RMS，避免早期增量接近零时比例虚高；
- `cos(d_i^T,d_i^P)`、均方误差 NMSE、残差符号对齐；统计 token 位置的中位数、P90、P99，不只把所有元素平均；
- 前/中/后分布，以及随组大小和起点变化的误差；
- 层间补偿差值的 Gram 矩阵 `G_ij=Σ_{samples,tokens,features}<c_i,c_j>`，其特征值能衡量**跨层输出方向的冗余**，但不能证明补偿容易从 `X` 预测；
- 组合后的实际残差 `\hat h - h^T`，并分别记录绝对 RMS、相对 RMS、cosine、最大偏差；
- 加法是否发生误差抵消：比较 `||Σ_i c_i||` 与 `Σ_i||c_i||`；用真教师 anchor 的这个比例只是 oracle 统计，不能算最终准确率。

**输出质量**：

- Teacher vs Candidate 的 `KL(P_teacher || P_candidate)`，对语言模型真正的 vocab 维做 float32 logsoftmax（不使用只覆盖 top-k 的近似冒充严格 KL）；
- NLL / PPL（由 causal-shift 计算），Top-1 token 一致率；
- 对个人输出日志保留均值、P50/P95、置信区间与每条样本的观测；样本很少时使用 bootstrap 只能说明不确定性，不能包装为显著性结论。

**性能**：阶段一主要报告理论组大小和关键路径 block 数（如 size4 从28→7 个阶段），以及同样数量的总 block 调用；现有逐层 Python 原型不等于同时运行的硬件并发。实际延迟需在新的 MLX 执行器实现后单独测量。任何时延都要区分“参考模型实际运行耗时”和“未优化原型的实际运行耗时”，不得把 28→7 当作 4x 真实加速。

**补偿难度的代理**：

- `R_i^* = d_i^T - d_i^P` 的能量、方向共线性、层间 Gram 谱、跨样本方差；
- `rank` / 标量加权 oracle 如果做，只能在校准上拟合并在 validation/test 复核，不可用 oracle 分数当作可实现的无训练速度；
- “串行轻量校正比一次性校正更容易”尚未证明，不在 Stage 1 自动得出。

## 6. 分步流程和硬门槛

### S0 — 版本与等价性（必须首先完成）

- 单份 BF16 权重加载，同样的 tokenizer、mask、RoPE，测 `T0=model(X)` 和 `T1=manual_serial(X)`；核对全序列 logits、token NLL。
- 使用 2–4 条短文本，记录 `max_abs_diff`, `rms_diff`, top1 agreement, NLL difference。若数值差异显著，暂停所有并行对照并排除 mask、位置、最终 norm、tied head、dtype、模型 revision 不同等错误。
- 不能把 `max abs` 一项设成跨版本固定 magic threshold；阈值必须结合 16-bit 数值误差和相同进程反复运行的噪声基线设定。

### S1 — 层差异地图（teacher-conditioned oracle）

- 固定若干 8 层区间：早 0–8、中 10–18、晚 20–28；额外从 anchor `h0` 检查 28 层所有共享输入对照。
- 用 teacher `h_s^T` 为组入口构造每层 `d_i^P`，得到 `c_i^*`，逐层保存统计而非 GPU 上持有数千条完整隐藏状态；必要时少量样本的残差矢量落盘用于低秩谱分析。
- 除原生串行教师外，不让任何 candidate 读取被跳过的 teacher 层结果。Oracle 只用来测结构特征。

### S2 — 可部署的分组累积

- 真实从 `h0` 出发，按 size2、size4、size8、size28 分组，绝不再注入教师前驱状态。
- 做三个局部改造，对照相同区间的直接删层；比较 delta 漂移、最终 KL/NLL、序列长度敏感性。
- 可行后，再引入“跨层可独立集合 + 多前驱汇合”作为下一轮 Plan A，当前不抢跑。

### S3 — 独立测试与决策

- 根据 validation 冻结候选组合（至少含一个轻度并行和一个激进并行），只在此时运行 test；后续若反复改变测试选择，必须重新划分独立测试。
- 给出 quality vs effective-depth 点图（横轴只是理论关键路径，非实际加速）；报出让 Plan B 值得尝试的几个校正目标分布。
- 单列 `unknown`: 是否可预测、在真实单 GPU 上速度、长距离 KV cache 稳定性，在本阶段绝不可填“通过”。

## 7. 如何判断要不要进入 Plan B

不是预设“P28 一定需要救活”。从 S3 的真实数据判断：

- **值得做补偿实验：** 至少存在一组非平凡并行结构，尚能保留一定的语言模型能力，或者它与教师的误差呈稳定可压缩/可解释形态，且不完全等同于直接删层。至少两种文本分布、多个长度下趋势一致会增加信心。
- **优先 B1（层级标量系数）：** 目标主要是各层更新的相对幅度错误，方向相近；先比较全局标量和输入条件化标量。
- **优先 B2（一次性低秩向量）：** 纠正方向很重要，但总体低秩谱集中；可以训练共享低秩头，区别于每层单独 LoRA。
- **优先 B3（小型串行递推）：** 如果补偿向量表现出强深度依赖、前一步残差显著改善下一个残差的可预测性，但不得仅凭 Gram 相关性断言“GRU 会成功”。
- **暂缓：** 组 size2 就在独立数据上大幅崩溃、误差随深度快速不可控，或并行结构完全不优于等预算删层；先回头找局部并行区域，不把小网络容量盲目放大。

没有未经验证的固定“PPL 只降 5% 才算成功”门槛：报告 Pareto 数据，并根据原模型基准质量决定门槛。建议基于 3 个种子、不同窗口起点、多个文本长度复核稳定性；这些是实验预案不是已取得统计结论。

## 8. 预期文件与结果规范

- `manifest.json`：系统、包版本、模型 revision、dataset revision、tokenizer、seed、BF16 与约束；
- `data/{calibration,validation,test}.jsonl`：只在本地，记录固定 token IDs / SHA256，不提交到私仓（可上传样本 ID 哈希与采样规则）；
- `results/stage1_per_sample.jsonl`：每个 sample 的 teacher NLL、候选 NLL、KL、Top1、最大误差等；
- `results/stage1_per_layer.jsonl`：每层更新范数、cosine、补偿范数，标识组起点和 anchor 类型；
- `results/REPORT.md`：summary、图表、误差分布、样本数、置信区间、失败 case、环境说明、是否进入 B1/B2/B3；
- 若数据规模太小/未运行，明确写 `NOT MEASURED`，不要给假图或模拟出来的百分比。

## 9. 风险与排错

1. MLX lazy evaluation：强制 `mx.eval` 才能证明数据已计算；延迟计时必须同步。
2. Qwen3 的双残差：以整块 `B_i(h)-h` 为单位对照，不能误把 `MLP(RMSNorm(h))` 当作独立于 Attention 的更新。
3. bf16 误差：主要指标在 float32 求差、归约；候选状态累加按固定规则归约并记录 dtype。
4. 缓存：本阶段 `cache=None`，评估的是 teacher-forced full-prefix，**不是**证明了 decode KV cache 正确。
5. 原始 28 层全部共享 embedding 的前向质量可能崩溃；它是极端压力上限，不应因此否定局部并行与 Plan B。
6. 层差值存在隐藏的任务依赖；不能靠英文 WikiText 单一数据集判定泛化。
7. 这里通过模型权重不同层前向的“独立性”并不意味着 MLX 会同时在 GPU 执行；真实加速需要独立验证。
8. 不允许通过在训练和推理之间混用 teacher 状态或预计算老师输出偷渡补偿结果。

## 10. 复用边界

详细见 [PRIOR_WORK.md](PRIOR_WORK.md)。主要直接用 MLX-LM 的官方 Qwen3 模型实现；对 CQIL 借鉴分块输入与补偿对照，对 True Depth 借鉴有效深度/误差基线，对 EvoPress、ACDC/AutoCircuit 仅留到第二阶段。**现在不安装任何 PyTorch/CUDA 研究框架。**

## 11. 下一轮唯一要做的事（取决于证据）

实验后只提交“是否值得训练、首选 B1/B2/B3 的理由和边界”，不自动执行训练、不自动选择并行化上线，更不声称已获得 x 倍加速。只要 Phase 1 质量/结构统计不充分，就以 `inconclusive` 收尾。
