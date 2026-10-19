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
