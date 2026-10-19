# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 64 x 128 predicted tokens
- git: `3c47001db9cd1011d5faf93fd02d2263dbd45906`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `d10-b25@fp32` | 28 | 10 | 3.484 | [3.39, 3.58] | 3.467 | 0.190 | 32.58x | 0.783 |
| `d10-b26@fp32` | 28 | 10 | 3.650 | [3.54, 3.77] | 3.613 | 0.198 | 38.46x | 0.863 |

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
