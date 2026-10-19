# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 64 x 128 predicted tokens
- git: `3c47001db9cd1011d5faf93fd02d2263dbd45906`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `irr-a@fp32` | 28 | 7 | 3.961 | [3.85, 4.07] | 3.955 | 0.165 | 52.49x | 0.926 |
| `irr-b@fp32` | 28 | 7 | 4.040 | [3.93, 4.15] | 4.035 | 0.159 | 56.80x | 0.893 |
| `irr-c@fp32` | 28 | 7 | 4.430 | [4.31, 4.56] | 4.435 | 0.154 | 83.97x | 1.076 |

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
