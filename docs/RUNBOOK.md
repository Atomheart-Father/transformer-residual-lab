# 运行手册 · Apple M4 Pro 本地

**状态：`scripts/measure.py` 已在 macOS 26.6.2 / MLX 0.31.2 / MLX-LM 0.31.3 上实跑验证，首轮 16 样本 smoke 通过 S0 闸门且逐位等价。它仍然只是测量原型，不是 benchmark。**

## 环境

- Apple Silicon、Python 3.12。本机可用 `/Users/bozhongxiao/code/.venv`（已含 mlx 0.31.2、mlx-lm 0.31.3、transformers 5.9.0、datasets 4.4.1、matplotlib）。
- 若要新建环境：`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`。
- 模型 `mlx-community/Qwen3-0.6B-Base-bf16` 约 1.19 GB；单份加载，不加载第二份权重副本。
- 首阶段不需要 Torch、CUDA、LoRA、EvoPress、AutoCircuit。

## 数据准备（一次）

```bash
python scripts/prepare_data.py --out data --seq-len 129 --calibration 16 --validation 8 --test 8
```

会打印并写入 `data/data_manifest.json` 的 `model_revision` 与 `dataset_revision`。每条窗口 129 个 token，预测 128 个。tokenizer 直接取自模型仓库同一 revision。

## 测量

Smoke（16 样本，全部 plan，fp32+bf16，含层/组诊断）：

```bash
python scripts/measure.py --input data/calibration.jsonl --max-samples 16 --output results/smoke
```

快速子集：

```bash
python scripts/measure.py --input data/calibration.jsonl --max-samples 2 \
  --plans serial,group2,group4 --accum fp32 --diagnose-plans "" --output results/devcheck
```

默认 plan 集合：

| 名称 | 含义 | blocks | depth |
|---|---|---:|---:|
| `serial` / `group1` | 参考串行 / 走分组路径的数值控制 | 28 | 28 |
| `group2,4,8,28` | 全局分组并行 | 28 | 14/7/4/1 |
| `window-{early,mid,late}-g{2,4,8}` | 只改 [0,8)/[10,18)/[20,28) | 28 | 24/22/21 |
| `delete-keep{7,14,21}` | 等距保留 7/14/21 层的删层基线 | 7/14/21 | 同 blocks |

`--accum` 控制残差累加精度（`fp32`、`bf16`），两者都要跑，用于区分结构误差与舍入误差。

## Stage 1.5：前缀保护、深度扫描、融合标定

数据换 257 token 窗口，划分也换（64/32/64）：

```bash
python scripts/prepare_data.py --out data257 --seq-len 257 --calibration 64 --validation 32 --test 64
```

**模型 revision 现在是硬闸门**：`--revision` 默认就是 pinned SHA，传 `None` 会直接 `SystemExit`。Stage 1 曾经记录 pinned SHA 却用 `revision=None` 加载，记录与加载不一致，已修。

`prefix{k}-g{g}` = 前 k 层串行，其余每 g 层一组共享 anchor：

| 名称 | 含义 |
|---|---|
| `prefix0-g2` / `group2` | 全局两两并行（`groupN` 是 `prefix0-gN` 的别名） |
| `prefix3-g8` | 前 3 层串行，其余每 8 层一组 |
| `--preset stage1_5` | 7 个前缀 × 3 个组大小，并为**每个实际达到的深度**自动生成删层基线 |

四步（顺序不能乱，calibration / validation / test 严格分离）：

```bash
# 1) 全网格筛选（calibration）
python scripts/measure.py --preset stage1_5 --input data257/calibration.jsonl \
    --max-samples 16 --output results/stage1_5/screen

# 2) 固定缩放对比（validation）
python scripts/measure.py --input data257/validation.jsonl --max-samples 32 \
    --plans prefix0-g2,prefix3-g8,prefix3-g4,prefix6-g4,prefix10-g4,prefix3-g2 \
    --fusion sum,mean,sqrt --group-records none --output results/stage1_5/val_fusion

# 3) 拟合 α（只用 calibration，拟合完冻结）
python scripts/calibrate.py --input data257/calibration.jsonl --max-samples 64 \
    --plan prefix3-g2 --scheme layer_ls --lambda-rel 0.01 \
    --out results/stage1_5/coeffs/prefix3-g2__layer_ls__lam0.01.json

# 4) 在 validation 上按规则选型，写 selection.json，再跑 test（只跑一次）
python scripts/measure.py --input data257/test.jsonl --max-samples 64 \
    --plans prefix3-g2 --coeffs results/stage1_5/coeffs/prefix3-g2__layer_ls__lam0.01.json \
    --group-records none --output results/stage1_5/test/prefix3-g2
```

B3（幅度 vs 方向）用 oracle 干预，**仅诊断**：

```bash
python scripts/diagnose_b3.py --input data257/validation.jsonl --max-samples 32 \
    --plans prefix0-g2,prefix3-g8,prefix3-g4,prefix3-g2,prefix10-g4 \
    --output results/stage1_5/b3

# 逐组归因：对每个组索引单独给入教师态
python scripts/diagnose_b3.py --input data257/validation.jsonl --max-samples 32 \
    --plans prefix3-g8 --at-group 5 --modes oracle-both --output results/stage1_5/b3_scan/g5
```

`--at-group first-parallel`（默认）会跳过串行前缀，落在第一个真正多层的组上。这一条很重要：`prefix3-g8` 的 group 0 是串行 layer 0，在那里干预没有任何效果，会得出「干预无效」的错误结论。

## Stage 1.6：受控分组边界 + 单点补偿

两件事：**只动组边界**的受控实验（block 数与深度都锁死），以及**一个**补偿器的最小验证。

### 1) 全新独立测试集（必须先做，之后不许再看）

`data257/test.jsonl` 在 Stage 1.5 已被反复查看，不能再当独立测试集。用新 seed 重抽，并显式剔除旧 offset：

```bash
python scripts/make_fresh_split.py --out data257 --split test --count 64 \
    --seq-len 257 --seed 771013 --exclude data257/test.jsonl
```

脚本会断言 offset 集合与旧集合**交集为空**，且每条窗口的 `tokens_sha256` 不重复，写出 `data257/fresh_test.jsonl` 与 `data257/fresh_test_manifest.json`。offset 都是 `seq_len` 整数倍，所以这是逐 token 不相交。

### 2) 自定义分区（保证 block 数不漂移）

`--plans` 支持 `名字:0|1|2|3-10|11-19|20-25|26-27` 语法。构造时断言 `flatten(groups) == list(range(28))`，所以 **blocks 由定义固定为 28**，depth 由 `len(groups)` 直接得出，不会出现「以为在比边界、其实改了深度」。

```bash
# 只动末边界：[19,b) + [b,28)，b ∈ {27,26,25,24,23}
python scripts/measure.py --preset stage1_6 --input data257/calibration.jsonl \
    --max-samples 64 --group-records parallel --output results/stage1_6/screen

# 验证集确认（含 depth 8 的 tail-2serial 与 depth 10 族，比较时必须标明深度）
python scripts/measure.py --input data257/validation.jsonl --max-samples 32 \
    --plans tail-b27,tail-b26,tail-b25,mid-b20,mid-b18,tail-2serial,d10-b24,d10-b26 \
    --group-records parallel --output results/stage1_6/val
```

`--preset stage1_6` 生成的族：`tail-b{27,26,25,24,23}`、`mid-b{10,12,18,20}`、`tail-2serial`（depth 8）、`d10-b{22..26}`（depth 10）、`serial`。

### 3) 补偿器：collect → fit → eval

只训练补偿器，主模型全部冻结。`collect` 用**自由运行**的组入口 `a` 作特征、`h_teacher − h_parallel` 作标签。

```bash
PLAN='irr-a:0|1|2|3-10|11-19|20-25|26-27'

python scripts/compensate.py --stage collect --input data257/calibration.jsonl \
    --max-samples 64 --plan "$PLAN" --group-index 5 \
    --output results/stage1_6/comp/irr-a-g5

python scripts/compensate.py --stage fit --pairs results/stage1_6/comp/irr-a-g5/pairs.npz \
    --output results/stage1_6/comp/irr-a-g5 --rank-linear 8 --rank-mlp 32 --epochs 16

python scripts/compensate.py --stage eval --input data257/fresh_test.jsonl \
    --max-samples 64 --plan "$PLAN" --group-index 5 \
    --compensator results/stage1_6/comp/irr-a-g5/linear_r8.npz \
    --compensator results/stage1_6/comp/irr-a-g5/mlp_r32.npz \
    --output results/stage1_6/comp/eval_fresh
```

**注意**：多个 `--compensator` 必须**显式重复参数**。`C="--compensator a --compensator b"; cmd $C` 在 zsh 下不会被分词，argparse 报错，而管道到 `head` 会把退出码变成 0，错误会被静默吞掉。

线性版是**降秩回归**：先白化 `X`，再截断白化后交叉协方差的 SVD。用「ridge 解 + 截断解矩阵」会发散（近奇异设计下解的最大奇异方向正好是噪声被放大的方向）。

`pairs.npz` 约 129 MB，`.gitignore` 已排除，只提交 `*_r{8,16,32,64}.npz`、`fit_report.json` 与 `eval_summary.json`。

### 4) 冻结选择，然后才允许跑 fresh test

在跑 `eval --input data257/fresh_test.jsonl` **之前**，把候选结构与补偿器的 cal/val 数字写进 `results/stage1_6/selection.json`（含 `fresh_test_used_for_selection: false`）。这个文件一旦写了就不再改；fresh test 只跑一次。Stage 1.5 的 `selection.json` 是同一个约定。

### 5) 出图

```bash
python scripts/make_figures.py --stage16 results/stage1_6
```

生成 `fig16_tail_scan.png`（质量 + 放大倍数 vs 末组大小）、`fig16_group_exit.png`、`fig16_compensation.png`。

## 聚合与出图

```bash
python scripts/summarize.py --input results/smoke
python scripts/make_figures.py --run results/smoke
```

`summarize.py` 会同时从 `gram_*.npy` 派生原始谱与归一化相关谱。**引用「低秩」时必须用归一化谱**，原始谱被幅度主导。

## 测试

```bash
python -m unittest discover -s tests -v
```

纯代数与 plan 构造测试，不需要 MLX 或模型。

## 验收清单

- S0：`model(X)` 与手工串行 `max_abs = 0`、`ΔNLL = 0`、`top1 = 1`。不达标立即停止，不许放宽容差。
- `group1` 必须与 `serial` 完全一致（数值控制）。
- 每个组内所有层从**完全相同**的 anchor 读输入。
- 自由运行绝不注入教师状态；教师 anchor 只用于 `local_rms_diff` 与 `layer` 诊断。
- 每个变体都记录 `executed_blocks` 与 `effective_depth`，不得把 depth 当时延。
- 结果目录包含 `manifest.json`、`samples.jsonl`、`groups.jsonl`、`layers.jsonl`、`gram_*.npy`、`summary.csv`。

## 已知事项

- MLX 惰性求值：所有标量读取前必须 `mx.eval`。
- `group1` 与串行逐位一致，是因为 `B(a)-a` 在 `B(a)≈a` 时精确可表示（Sterbenz），加回后还原 `B(a)`；这不是巧合，但也意味着它不能用来估计一般舍入误差。
- 无 KV cache：只测 teacher-forced full prefix，未验证自回归 decode。
- 当前 plan 只覆盖「组内共享同一 anchor」这一种融合规则。

## Stage 1.5 的已知事项

- **前缀长度不是充分统计量。** `prefix3-g8` 与 `prefix4-g8` 有效深度都是 7，质量差 2.07 nats。报告分组必须给出具体层区间，不能只说「保护了 k 层」。
- **删层质量对保留层数非单调**（test：keep16 9.34 < keep12 9.91 < keep15 12.32 < keep14 12.57）。同深度比较必须逐深度对齐，不能插值。
- **B3 的 `oracle-*` 干预使用教师态，是诊断不是方法。** 任何 `oracle-dir` 的成绩都不能当作可实现结果引用。
- `layer_ls` 在 `prefix3-g8` 上过拟合（validation 4.51 → 7.66，条件数 40，1 个 α 被裁到边界）。组越大、共享 anchor 的 block 越多，修正越共线，线性标定越不可靠。
- `--group-records none` 会跳过组级/层级的教师态对比，快很多，但 `groups.jsonl` 与 `layers.jsonl` 为空，`summarize.py` 的组表/层表也随之空。只在不需要诊断记录时用。

## Stage 1.6 的已知事项

- **「末组必须小」比「27 必须串行」重要。** 深度 7 下末组 1/2/3/4/5 层对应验证集 4.511/4.194/6.587/—/—。27 单独成组反而更差。别把「保护最后一层」当结论。
- **末组的破坏力来自放大，不是来自它自己产生误差。** 放大项 RMS 随末组大小 13.5→95.9，新产生项只从 0→15.2。看到「末组误差大」不要直接归因于「末组算错了」。
- **中间边界几乎不重要**（全区间 0.35–0.40 nats），尾部边界能造成 5.2 nats。位置价值极不均匀，做扫描时不要把样本量平均分配。
- **1 参数族无法解耦「前一组的层数」与「末组大小」。** `[19,b)` + `[b,28)` 里 b 一动两个都动，所以「末组 2 层最优」是平衡点，不是 2 这个数字有特殊性。
- **隐藏态 MSE 会高估补偿能力。** rank-32 把最终隐藏态 rel_rms 从 0.939 降到 0.707（25%），ΔNLL 只改善 16%。能力结论一律以 ΔNLL / KL / top1 为准。
- **FLOPs 不能折算成本。** rank-32 的 FLOPs 是单 block 的 0.61%，实测耗时是 13.5%。小矩阵乘打不满 GPU，必须实测。
- **`oracle` 用教师态，不可部署**，只能当上界。`recovery_vs_oracle` 是相对 oracle 空间的百分比，不是相对串行参考。
- 补偿器只在**一个**组上验证过，多点补偿的相互干扰完全没测。
- rank-64 已出现轻微过拟合（train 0.789 vs val 0.812）。扩容量前先扩 calibration 集，当前只有 64 个窗口 / 14,803 行。

## Stage 1.7：多站点 / 共享 / 串行状态

引擎是 `scripts/serial_compensate.py`，四个阶段 `collect → fit → eval → bench`。

### 1) 新数据划分

```bash
# 同语料独立测试（排除 test.jsonl 与 fresh_test.jsonl 的 offset）
python scripts/make_fresh_split.py --out data257 --name test_v2 --count 64 --seed 90210 \
    --source wikitext2 --purpose "Stage 1.7 same-corpus independent test" \
    --exclude data257/test.jsonl --exclude data257/fresh_test.jsonl

# 跨语料测试（IMDB，不与 wikitext 共享任何窗口）
python scripts/make_fresh_split.py --out data257 --name ood_imdb --count 64 --seed 31337 \
    --dataset stanfordnlp/imdb --config plain_text --split test \
    --text-field text --source imdb --purpose "Stage 1.7 cross-corpus test"
```

`--out` 是**目录**，文件名由 `--name` 决定（`<out>/<name>.jsonl`）。IMDB 的文本字段名也是 `text`，但一定要显式写 `--text-field text` 并给 `--source imdb`，否则 `sample_id` 会和 wikitext 混在一起。

### 2) 采集（注意 `A.npy` / `C.npy` 是 mmap 的）

```bash
PLAN="irr-a:0|1|2|3-10|11-19|20-25|26-27"
python scripts/serial_compensate.py --stage collect --plan "$PLAN" --sites 4,5 \
    --input data257/calibration.jsonl --max-samples 64 \
    --output results/stage1_7/B_r0_45
```

`--plan` 必须是 `名字:分组体` 的完整写法（`irr-a:0|1|2|...`）。只给 `irr-a` 会在 `measure.make_plan` 里报 `unknown plan`——名字本身不是合法的 spec。

`collect` 写 `A.npy` / `C.npy`（`[n,S,T,D]`，float32，每个约 250 MB）。`fit` 用 `mmap_mode="r"` 读，**不要**改回 `np.load` 全量读入：4 站点 r64 时会吃掉 500 MB+ 然后被 OOM kill（exit 137）。

### 3) 训练

```bash
python scripts/serial_compensate.py --stage fit --pairs results/stage1_7/B_r0_45 \
    --hidden 64 --rank 64 --epochs 40 --output results/stage1_7/B_r0_45 --tag b3_r64
```

- `--carry-state` 打开状态传递（B3）；不打开就是 B2。**参数、FLOPs、输入、站点完全相同**，这是唯一的控制变量。
- `--per-site` 给每个站点独立权重（参数量 ×S），默认共享。
- `--site-weight normalized` 按站点方差归一化损失。**对 4 站点崩溃没有用**（−43% → −29%），别指望它修站点干扰。
- `--output` 是必填项，即使和 `--pairs` 同一个目录也要显式写。

### 4) 评估

```bash
python scripts/serial_compensate.py --stage eval --plan "$PLAN" --sites 4,5 \
    --compensator results/stage1_7/B_r2_45/serial_b3_r64.npz \
    --input data257/test_v2.jsonl --max-samples 64 \
    --output results/stage1_7/B_test_r2
```

- `--apply-sites 4` 只让部分站点生效，用来给每个站点定价。**必须**是 `--sites` 的子集。
- 评估同时给出 `irr-a` / `+oracle` / `+oracle_ex_last` / 各补偿器。`gap_closure` 由脚本算好；`recovery_vs_oracle` 只在自己站点集合内可比。
- 大张量留在本地；`serial_*.npz` 合计约 12 MB，可以提交。

### 5) 时延

```bash
python scripts/serial_compensate.py --stage bench \
    --compensator results/stage1_7/B_r2_45/serial_b3_r64.npz \
    --output results/stage1_7/bench_serial_b3_r64.json
```

`per_site_ms` 是单站点中位数，`chain_ms` 是整条链（含站点间的串行依赖），`flops_per_token = 4 × n_params`。**`chain_ms` 才是关键路径上的增量**，`per_site_total_ms` 是各站点耗时之和，二者不等。

### 6) 汇总、统计、出图

```bash
python scripts/make_stage17_table.py                      # -> table.json
python scripts/stage17_stats.py                           # -> paired_stats.json
python scripts/make_figures.py --stage17 results/stage1_7 --figdir results/stage1_7/figures
```

**顺序不能反**：`make_figures.py --stage17` 从 `table.json` 与 `paired_stats.json` 读数据，缺文件时会静默少画点。`--figdir` 不给的话默认写到 `--run` 目录下（容易写错地方）。

`make_stage17_table.py` 只做汇总，不做重算；缺键会进 `missing` 并打印，不会静默补 0。加新 run 就改脚本里的 `ROWS` 表。

## Stage 1.7 的已知事项

- **多站点补偿的输入分布由前驱修正决定。** 用未补偿图采集的数据训练多站点补偿器，部署时会因为前驱改写而失效（4 站点 r64 比不补偿差 1.76 nats，带状态的 B3 差 2.95 nats）。必须用 `--compensator` 在**补偿生效**的图上重采数据，然后重训。这是本轮最大的单一效应（8% → 18%）。
- **修正必须成组上线。** 最终 2 站点模型上只开站点 5 是 −8.1%，只开站点 4 是 +10.0%，一起开是 +18.4%。按站点逐个加补偿的部署策略不成立。
- **最后一组不要修。** 4 站点模型上单独修 group 6 只有 +1.5%，而且它会直接改写进 logits 的那一步。
- **状态传递（B3 − B2）在 2 站点上没有显著收益**，三个集 CI 全跨 0。要检验长链状态需要 8+ 站点且每轮重采，代价高，先不要假设它有用。
- **跨语料迁移失败。** IMDB 上所有方法都只有 2.5–3.0% gap closure，而 test_v2 是 16.8%。补偿器拟合的是 WikiText-2 的残差统计。
- **NLL 与 top-1 会背离。** B3 的 NLL 比 B2 好但 top-1 略低（0.237 vs 0.244）。只报一个指标会误导。
- **`test_v2` 在 Experiment A 里当过诊断集。** B 部分的结构/轮次/站点全在 validation 上定，但严格说它不算"从未见过"。`ood_imdb` 是唯一完全干净的集。
- **本轮没有任何并发执行。** 所有时延都是顺序测的，`eval_summary.json` 里的 `warning` 字段写着这一点。不宣称加速。

