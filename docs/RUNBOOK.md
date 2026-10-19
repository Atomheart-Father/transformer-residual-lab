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
