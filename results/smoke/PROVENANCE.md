# 运行溯源

本目录（`results/smoke/`）由以下代码版本产生。

| 项 | 值 |
|---|---|
| 代码 commit | `29d18c4` |
| 运行时间 | 2026-10-19 16:11–16:13 (Europe/London) |
| 主机 | Apple M4 Pro / 24 GB / macOS 26.6.2 |
| Python | 3.12.12 |
| mlx | 0.31.2 |
| mlx-lm | 0.31.3 |
| numpy | 2.3.5 |
| 模型 | `mlx-community/Qwen3-0.6B-Base-bf16` @ `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8` |
| 数据集 | `Salesforce/wikitext` `wikitext-2-raw-v1` @ `b08601e04326c79dfdd32d625aee71d232d685c3` |
| 样本 | calibration 16 窗口 × 129 token（预测 128） |
| seed | 20261019 |
| 命令 | `python scripts/measure.py --input data/calibration.jsonl --max-samples 16 --output results/smoke` |

`manifest.json` 中的 `git_commit` 为 `null`，因为该字段在运行时尚无提交可读；以本文件的代码 commit 为准。

token id 与原始窗口留在本地 `data/`（已被 `.gitignore` 排除），仓库只保留 `data/data_manifest.json` 的 revision、规则与每条窗口的 `tokens_sha256`。`samples.jsonl` 中也带有 `tokens_sha256`，可用于核对样本身份。

## 重算

```bash
python scripts/prepare_data.py --out data --seq-len 129 --calibration 16 --validation 8 --test 8
python scripts/measure.py --input data/calibration.jsonl --max-samples 16 --output results/smoke
python scripts/summarize.py --input results/smoke
python scripts/make_figures.py --run results/smoke
```

重跑会覆盖 `results/smoke/` 下的记录；若要保留，先复制目录。
