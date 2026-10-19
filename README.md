# Transformer Residual Lab · Stage 1 / 1.5 / 1.6 / 1.7

**状态：Stage 1、1.5、1.6、1.7 已完成；主模型权重全程冻结，Stage 1.7 只训练了补偿器（最大 278,752 参数）。未做 DAG 搜索，未宣称推理加速。**

对同一个冻结的 `Qwen3-0.6B-Base` BF16 模型，比较逐层串行的真实残差与共享输入的并行近似残差，测量三类误差（层级修正、局部组误差、自由运行累积误差）的分布、累积与结构，并验证 Plan B 的轻量补偿网络能否在独立数据上回收质量。

- [Stage 1.7 报告：补偿有效性验证 + 串行残差网络](results/stage1_7/REPORT.md)
- [Stage 1.6 报告：受控分组边界 + 单点补偿](results/stage1_6/REPORT.md)
- [Stage 1.5 报告：前缀保护、深度扫描、融合标定](results/stage1_5/REPORT.md)
- [Stage 1 报告](results/REPORT.md)
- [详细实验设计](docs/STAGE1_EXPERIMENT.md)
- [既有完整研究路线 v1](docs/2026-10-19_full_roadmap_v1.md)
- [运行方法与验收](docs/RUNBOOK.md)
- [开源复用与核查](docs/PRIOR_WORK.md)
- [结果字段](docs/OUTPUT_SCHEMA.md)
- [结果报告模板](results/REPORT_TEMPLATE.md)

## Stage 1.7 结论（详见报告）

1. **补偿的增益 100% 依赖输入，常数项贡献为零。** 均值补偿在 test_v2 上更差 0.033（不显著）、在 IMDB 上显著更差 0.076；打乱"输入↔预测"对应关系后比不补偿差 0.68 nats（97% 样本都更差）。预测是特异的，不是通用去噪。
2. **跨语料迁移失败。** 同一个 rank-64 MLP：test_v2 关闭 20.7%（0.766 nats），IMDB 只关闭 3.0%（0.094 nats）。增益掉到 1/8。
3. **串行状态没有可测收益。** B3−B2 = −0.011/−0.010/+0.008（val/test/ood），三个集 95% CI 全跨 0，逐样本胜率 39–55%。在 2 站点、间隔 6 层时，1024 维入口 `a` 已经携带了 64 维状态想带的信息。
4. **多站点补偿必须先重采样。** 用未补偿图采集的数据训练 4 站点补偿器，比不补偿差 1.76（B2）– 2.95（B3）nats；改成"补偿生效后再采集"，2 站点方案从 8.0% 跳到 18.1%（配对 0.41 nats，100% 样本改善）。这是本轮最大的单一效应。
5. **修正必须成组上线。** 最终 2 站点模型上：只开站点 4 = +10.0%，只开站点 5 = −8.1%，一起开 = +18.4%。站点 5 的预测器是在"站点 4 已修正"的流上训练的。
6. **普通 MLP 仍然赢。** rank-64 MLP 用 132k 参数 / 0.418 ms 关闭 21.1%；串行单元用 278k 参数 / 0.732 ms 关闭 18.4%。门控残差单元在这个尺度上是负资产。
7. **距教师仍很远。** 最好的方法在 val 上离教师 3.30 nats（PPL 约 27 倍）；全站点 oracle 的上限是 0.428 nats。每站点预测 `rel_err` 0.86–0.91、`cosine` 0.42–0.52。
8. **串行链条占关键路径 4.0%。** 2 站点链条 0.732 ms，`irr-a` 关键路径 7 个 group 步约 18.2 ms（单 block 2.6 ms）。它严格串行、不可与前驱 group 重叠、随站点数线性增长；本轮**没有**实现并发执行。

## Stage 1.6 结论（详见报告）

1. **「保护最后一层」被推翻。** 深度 7 下末组 1/2/3/4/5 层对应验证集 ΔNLL 4.511/4.194/6.587/—/—。`[26,27]` 成组比 `[27]` 单独串行**好** 0.32 nats。
2. **机制是末组放大误差**，不是末组自己算错。放大项 RMS 随末组大小 13.5→95.9（0.86×→12.8×），新产生项只从 0→15.2。末组必须小。
3. **位置价值极不均匀**：只动中间边界，全区间 0.35–0.40 nats；动末边界能造成 5.2 nats。
4. **不规则连续分组确实更好，但收益温和**：`3-10 / 11-19 / 20-25 / 26-27`（`irr-a`）在 28 blocks、depth 7、零训练下比 Stage 1.5 基线省 0.41（cal）/ 0.47（val）nats，约 10%，不是数量级。
5. **单点补偿在全新独立测试集上有效**：`irr-a` 3.981 → 3.410（−0.57 nats，−14.3%），回收 oracle 空间的 16.0%。rank 8/16/32/64 单调改善（5%/11%/16%/20%），未饱和。
6. **泛化不打折**：验证 18.3% → 全新测试 16.0%。训练特征取自自由运行，不用教师态做输入。
7. **隐藏态 MSE 高估能力**：rank-32 把最终隐藏态 rel_rms 改善 25%，ΔNLL 只改善 16%。能力结论以 ΔNLL / KL / top1 为准。
8. **成本必须实测**：rank-32 的 FLOPs 是单 block 的 0.61%，实测耗时是 13.5%（0.350 ms / 256 token）。补偿器读组入口 `a`，与该组 block 并行，**没有额外串行等待**。

## Stage 1.5 结论（详见报告）

1. **深度 16/28（−43%）@ ΔNLL 0.96** 是「几乎无损」的极限；0.5 nats 以内做不到。
2. 保护 3 层是拐点，之后**缩小 group size 比加长前缀更划算**：`prefix3-g2`（d=16, 1.12）优于 `prefix10-g4`（d=15, 1.83）。
3. 同深度下并行全面优于删层，优势 3.4–12.5 nats（64 样本测试集）。
4. 固定缩放（mean/sqrt）基本无效；离线 ridge 标定只回收 6%–25%，因为线性组合的重构上限是 38%–70% 相对 RMS。
5. **早层失败是方向错误，不是幅度错误**：只修幅度更差（9.0→10.5），只修方向回收 79%（9.0→1.9）。
6. 保护 3 层之后，最大单一误差来源变成**最后一组**（46%），不是最前一组。

## Stage 1 结论（详见报告）

1. 误差是**位置**的函数，不是深度的函数：全 28 层两两并行 +9.00 nats，只并行中窗 2 对 +0.17 nats。
2. 误差在 layer 0–4 一次性产生，此后既不抵消也不显著放大。
3. 全局并行的算力效率低于删层（100% block 换到 75% block 的质量），但同深度下并行更好。**两个轴必须分开比。**
4. 修正量在归一化后**不低秩**；原始 Gram 的 rank-1 是早层幅度伪影。
5. bf16 与 fp32 累加的最大差异 0.0099 nats，结构误差比舍入误差大 3 个数量级。

## 环境

`mlx-community/Qwen3-0.6B-Base-bf16`（28 层，hidden 1024，16Q/8KV，vocab 151936，tied embedding），BF16，Apple M4 Pro / 24 GB。模型与数据集 revision 固定在 `data/data_manifest.json` 与各次运行的 `manifest.json` 中。

## 仓库结构

```
scripts/prepare_data.py   固定 revision 的 token 窗口 + manifest
scripts/measure.py        三类误差、plan 定义、bf16/fp32 对照、融合规则、Gram
scripts/calibrate.py      离线 ridge 拟合 per-group / per-layer α（只用 calibration）
scripts/diagnose_b3.py    B3：幅度 vs 方向的 oracle 干预 + 逐组归因
scripts/make_fresh_split.py  与已用样本逐 token 不相交的新测试集（seed 771013）
scripts/compensate.py     Plan B 单点补偿：collect / fit / eval 三阶段
scripts/serial_compensate.py  多站点 / 共享 / 串行状态：collect / fit / eval / bench
scripts/make_stage17_table.py Stage 1.7 统一结果表（报告与图表的唯一数据来源）
scripts/stage17_stats.py      逐样本配对 bootstrap（20000 次）
scripts/summarize.py      聚合、bootstrap CI、Gram 谱、Pareto / 同深度 / 阈值表
scripts/make_figures.py   PNG 图表
tests/                    纯代数与 plan 构造测试（不需要 MLX）
results/smoke/            Stage 1 首轮 smoke 的原始记录与图表
results/stage1_5/         Stage 1.5 全部记录、系数、图表与报告
results/stage1_6/         Stage 1.6 边界扫描、补偿器、成本与报告
results/stage1_7/         Stage 1.7 多站点补偿、串行状态、配对统计、成本与报告
```

`--plans` 支持显式分区语法 `名字:0|1|2|3-10|11-19|20-25|26-27`，构造时断言 `flatten(groups) == range(28)`，所以 **block 数不可能漂移**，depth 由组数直接给出。做边界实验必须用这个语法，不要用 `prefix{k}-g{g}`（它会同时改变深度）。

## 复现

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/prepare_data.py --out data --seq-len 129 --calibration 16 --validation 8 --test 8
python scripts/measure.py --input data/calibration.jsonl --max-samples 16 --output results/smoke
python scripts/summarize.py --input results/smoke
python scripts/make_figures.py --run results/smoke
```

Stage 1.5 用 257 token 窗口与 64/32/64 划分，完整命令见 `docs/RUNBOOK.md` 的「Stage 1.5」一节。
Stage 1.6 的边界扫描与补偿器流程见 `docs/RUNBOOK.md` 的「Stage 1.6」一节。
Stage 1.7 的多站点 / 串行状态流程见 `docs/RUNBOOK.md` 的「Stage 1.7」一节（含 `A.npy` mmap、`--apply-sites`、`chain_ms` 的用法）。

## 研究边界

1. `model(tokens)` 与手工逐层串行必须逐位等价（S0 闸门，当前为 0 差异）。
2. 共享 anchor 只定义**可并行的数学计算图**；当前 Python 循环不代表 GPU 并发，也不构成速度证据。
3. 局部诊断可以用教师状态作 anchor；自由运行**绝不**注入教师前驱状态，两者分开记录。
4. Gram / 低秩只描述修正的结构，不证明小网络可预测。
5. 本仓库不包含模型权重、数据集正文、第三方代码副本或任何模拟图表。
6. **并行与删层必须在两个轴上分别比较**：相同有效深度、相同 block 数。把 `group2`（28 blocks / depth 14）与 `keep21`（21 blocks / depth 21）直接对比是错的。
7. **「保护了 k 层」不是充分描述**，必须同时给出具体分组区间。同深度不同边界的方案质量可以差 2 nats。
8. B3 的 `oracle-*` 干预使用教师态，**只是诊断**，其成绩不能作为可实现方案引用。
9. **反复查看过的 test 集不再是独立测试集。** Stage 1.6 起，结构选择与补偿器训练只看 calibration / validation；最终数字来自 `fresh_test.jsonl`（seed 771013，与旧 test 逐 token 不相交），并记录抽取参数与断言结果。
10. **补偿器只能使用推理时真实可得的输入。** 训练特征取自自由运行的组入口 `a`，教师态只用于生成标签和计算 oracle 上界，绝不进入 `C_θ` 的输入。
11. **成本一律用实测时间，不用 FLOPs 折算。** 小矩阵乘在 GPU 上打不满：rank-32 的 FLOPs 是单 block 的 0.61%，实测耗时是 13.5%。
12. **隐藏态误差不等于模型能力。** 状态 rel_rms 改善 25% 时 ΔNLL 只改善 16%。能力结论只以 ΔNLL / KL / top1 为准。
13. **多站点补偿的输入分布由前驱修正决定。** 不能把独立采集的各组残差拼起来训练，然后假设部署时仍然有效；必须用补偿生效的图重新采集，并重新训练。
14. **按站点逐个加补偿不成立。** 修正必须整组上线：单独开站点 5 会让质量倒退 8.1%，因为它的预测器只在"站点 4 已修正"的流上有效。
15. **"多站点"和"串行状态"必须分开检验。** 同参数量、同 FLOPs、同输入，唯一差别是 `--carry-state`；否则无法区分"多了一个站点"和"状态真的有用"。
16. **小差异必须做逐样本配对检验。** B3−B2 只有 0.01 nats，靠均值差无法判断；配对 bootstrap 才能说明它落在噪声内。
17. **补偿器不一定跨语料。** WikiText-2 上拟合的修正，在 IMDB 上只剩 1/8 的绝对增益。任何"替代串行依赖"的主张都要先过跨语料这一关。
18. **串行补偿是纯增量延迟。** 它读的是被前驱修正过的状态，因此不能与产生该状态的 group 重叠；引用成本时必须用 `chain_ms`（含站点间依赖），不是各站点耗时之和。

## 研究协作

- 仓库：`Atomheart-Father/transformer-residual-lab`（private）
- WorkBuddy / DeepSeek：本地实现、执行、记录、提交。
- GPT：实验设计、结果审查、决定后续实验。
- 用户：重要研究决策。

每个实验阶段完成后提交一次，包含配置、revision、逐层/逐组统计、图表、报告与对应 commit SHA。大型激活张量与 token id 留在本地，仓库保留汇总数据与复现规则。
