# Stage 1 — Residual gap experimental report

> **STATUS: NOT MEASURED** · 2026-10-19。此文件是实验报告模板，并非已经获得实验结果。

## A. 复现环境

- Model / revision: `TBD`
- MLX-LM / MLX / Python / macOS versions: `TBD`
- Device / available memory / active tasks: `TBD`
- Tokenizer/dataset revision and hashes / split lengths: `TBD`
- Groups and reduction dtype: `TBD`

## B. S0 等价性

- Reference model vs manual_serial: max abs / RMS / KL / top1 / teacher NLL: `NOT MEASURED`
- Acceptance: `NOT MEASURED`

## C. Teacher-local 残差差值

- Per-layer correction RMS and cosine (early/mid/late): `NOT MEASURED`
- Group-size2/4/8 sensitivity: `NOT MEASURED`
- Error correction Gram spectrum and signs of cancellation: `NOT MEASURED`
- Observed failure cases: `NOT MEASURED`

## D. Free-running compositional quality

| Variant | Effective sequential group count | True model NLL | Candidate NLL | ΔNLL | KL(T‖P) | Top1 | Real GPU speedup |
|---|---:|---:|---:|---:|---:|---:|---|
| Serial reference | 28 | NOT MEASURED | — | — | — | — | NOT MEASURED |
| Parallel groups of 2 | 14 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Parallel groups of 4 | 7 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Parallel groups of 8 | 4 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Parallel group of 28 | 1 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

备注：“effective sequential group count” 只是逻辑关键路径分组数，**不是**速度倍数；同时有总计算量与内存带宽瓶颈。

## E. Plan B 决策

- Does the residual have small magnitude? `UNKNOWN`
- Is the residual low-rank across layers? `UNKNOWN`
- Is it learnably predictable from a cheap network? **NOT TESTED in Stage 1**
- Better B1 scalar/B2 low-rank/B3 tiny recurrent? `INCONCLUSIVE`
- Are there local 2/4-layer groups worth exploring? `INCONCLUSIVE`
- Main scientific uncertainty, proposed next cheapest experiment: `TBD`

## F. Reproducibility

- Raw run manifest + samples JSONL: `TBD`
- Token ids SHA256 (no copyrighted long text): `TBD`
- Plot commands, error bars and exceptions: `TBD`
- Caveats: teacher-forcing only, no KV-cache decode, no actual GPU parallel runtime.
