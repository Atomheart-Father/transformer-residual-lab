# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 32 x 128 predicted tokens
- git: `3c47001db9cd1011d5faf93fd02d2263dbd45906`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `irr-a@fp32` | 28 | 7 | 4.039 | [3.90, 4.18] | 3.988 | 0.166 | 56.79x | 0.939 |
| `irr-b@fp32` | 28 | 7 | 4.084 | [3.96, 4.22] | 4.034 | 0.161 | 59.38x | 0.907 |
| `irr-c@fp32` | 28 | 7 | 4.525 | [4.36, 4.70] | 4.462 | 0.155 | 92.30x | 1.117 |

`blocks` = executed decoder blocks (real compute); `depth` = sequential stages (critical path). Neither is a latency measurement.

## minimum effective depth at a ΔNLL budget

| ΔNLL budget | variant | depth | blocks | ΔNLL |
|---:|---|---:|---:|---:|
| 0.1 | not reached | — | — | — |
| 0.25 | not reached | — | — | — |
| 0.5 | not reached | — | — | — |
| 1 | not reached | — | — | — |
| 2 | not reached | — | — | — |
| 3 | not reached | — | — | — |
