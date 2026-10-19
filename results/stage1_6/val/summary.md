# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 32 x 128 predicted tokens
- git: `3c47001db9cd1011d5faf93fd02d2263dbd45906`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `mid-b18@fp32` | 28 | 7 | 4.838 | [4.70, 4.99] | 4.800 | 0.102 | 126.25x | 1.032 |
| `mid-b20@fp32` | 28 | 7 | 4.486 | [4.34, 4.64] | 4.450 | 0.133 | 88.73x | 1.128 |
| `tail-b25@fp32` | 28 | 7 | 6.587 | [6.31, 6.88] | 6.556 | 0.084 | 725.45x | 1.881 |
| `tail-b26@fp32` | 28 | 7 | 4.194 | [4.06, 4.33] | 4.136 | 0.158 | 66.27x | 0.952 |
| `tail-b27@fp32` | 28 | 7 | 4.511 | [4.37, 4.66] | 4.471 | 0.126 | 91.05x | 1.078 |
| `tail-2serial@fp32` | 28 | 8 | 4.046 | [3.92, 4.18] | 3.983 | 0.163 | 57.16x | 0.841 |
| `d10-b24@fp32` | 28 | 10 | 3.707 | [3.56, 3.85] | 3.668 | 0.185 | 40.72x | 0.935 |
| `d10-b26@fp32` | 28 | 10 | 3.648 | [3.51, 3.79] | 3.592 | 0.204 | 38.39x | 0.876 |

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
