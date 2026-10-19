# Transformer Residual Lab · Stage 1 / 1.5

**状态：Stage 1 smoke 与 Stage 1.5（64/32/64 × 257 token）已完成；权重冻结，未训练任何补偿网络，未进入 Plan B。**

对同一个冻结的 `Qwen3-0.6B-Base` BF16 模型，比较逐层串行的真实残差与共享输入的并行近似残差，测量三类误差（层级修正、局部组误差、自由运行累积误差）的分布、累积与结构，判断 Plan B 的轻量补偿网络是否值得训练。**本阶段不训练任何参数、不搜索 DAG、不宣传实际加速。**

- [Stage 1.5 报告：前缀保护、深度扫描、融合标定](results/stage1_5/REPORT.md)
- [Stage 1 报告](results/REPORT.md)
- [详细实验设计](docs/STAGE1_EXPERIMENT.md)
- [既有完整研究路线 v1](docs/2026-10-19_full_roadmap_v1.md)
- [运行方法与验收](docs/RUNBOOK.md)
- [开源复用与核查](docs/PRIOR_WORK.md)
- [结果字段](docs/OUTPUT_SCHEMA.md)
- [结果报告模板](results/REPORT_TEMPLATE.md)

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
scripts/summarize.py      聚合、bootstrap CI、Gram 谱、Pareto / 同深度 / 阈值表
scripts/make_figures.py   PNG 图表
tests/                    纯代数与 plan 构造测试（不需要 MLX）
results/smoke/            Stage 1 首轮 smoke 的原始记录与图表
results/stage1_5/         Stage 1.5 全部记录、系数、图表与报告
```

## 复现

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/prepare_data.py --out data --seq-len 129 --calibration 16 --validation 8 --test 8
python scripts/measure.py --input data/calibration.jsonl --max-samples 16 --output results/smoke
python scripts/summarize.py --input results/smoke
python scripts/make_figures.py --run results/smoke
```

Stage 1.5 用 257 token 窗口与 64/32/64 划分，完整命令见 `docs/RUNBOOK.md` 的「Stage 1.5」一节。

## 研究边界

1. `model(tokens)` 与手工逐层串行必须逐位等价（S0 闸门，当前为 0 差异）。
2. 共享 anchor 只定义**可并行的数学计算图**；当前 Python 循环不代表 GPU 并发，也不构成速度证据。
3. 局部诊断可以用教师状态作 anchor；自由运行**绝不**注入教师前驱状态，两者分开记录。
4. Gram / 低秩只描述修正的结构，不证明小网络可预测。
5. 本仓库不包含模型权重、数据集正文、第三方代码副本或任何模拟图表。
6. **并行与删层必须在两个轴上分别比较**：相同有效深度、相同 block 数。把 `group2`（28 blocks / depth 14）与 `keep21`（21 blocks / depth 21）直接对比是错的。
7. **「保护了 k 层」不是充分描述**，必须同时给出具体分组区间。同深度不同边界的方案质量可以差 2 nats。
8. B3 的 `oracle-*` 干预使用教师态，**只是诊断**，其成绩不能作为可实现方案引用。

## 研究协作

- 仓库：`Atomheart-Father/transformer-residual-lab`（private）
- WorkBuddy / DeepSeek：本地实现、执行、记录、提交。
- GPT：实验设计、结果审查、决定后续实验。
- 用户：重要研究决策。

每个实验阶段完成后提交一次，包含配置、revision、逐层/逐组统计、图表、报告与对应 commit SHA。大型激活张量与 token id 留在本地，仓库保留汇总数据与复现规则。
