# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 32 x 256 predicted tokens
- git: `afd9d9f2feb541f8c6fa54bde760f25496b02d0d`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `prefix3-g8@fp32` | 28 | 7 | 4.511 | [4.37, 4.66] | 4.471 | 0.126 | 91.05x | 1.078 |
| `prefix3-g8+mean@fp32` | 28 | 7 | 10.658 | [10.47, 10.86] | 10.748 | 0.000 | 42544.41x | 2.647 |
| `prefix3-g8+sqrt@fp32` | 28 | 7 | 8.234 | [7.98, 8.49] | 8.316 | 0.011 | 3768.02x | 2.927 |
| `prefix3-g4@fp32` | 28 | 10 | 3.865 | [3.72, 4.01] | 3.837 | 0.183 | 47.72x | 1.206 |
| `prefix3-g4+mean@fp32` | 28 | 10 | 6.471 | [6.33, 6.61] | 6.518 | 0.032 | 646.30x | 0.979 |
| `prefix3-g4+sqrt@fp32` | 28 | 10 | 3.375 | [3.24, 3.51] | 3.383 | 0.234 | 29.22x | 0.703 |
| `prefix6-g4@fp32` | 28 | 12 | 2.642 | [2.54, 2.74] | 2.577 | 0.273 | 14.04x | 0.764 |
| `prefix6-g4+mean@fp32` | 28 | 12 | 4.456 | [4.26, 4.67] | 4.484 | 0.129 | 86.12x | 0.744 |
| `prefix6-g4+sqrt@fp32` | 28 | 12 | 2.421 | [2.31, 2.55] | 2.401 | 0.296 | 11.25x | 0.606 |
| `prefix0-g2@fp32` | 28 | 14 | 8.919 | [8.63, 9.20] | 8.914 | 0.023 | 7470.34x | 1.139 |
| `prefix0-g2+mean@fp32` | 28 | 14 | 8.853 | [8.59, 9.13] | 8.862 | 0.018 | 6998.56x | 0.954 |
| `prefix0-g2+sqrt@fp32` | 28 | 14 | 8.977 | [8.70, 9.25] | 8.976 | 0.019 | 7921.51x | 1.017 |
| `prefix10-g4@fp32` | 28 | 15 | 1.816 | [1.74, 1.90] | 1.759 | 0.365 | 6.15x | 0.658 |
| `prefix10-g4+mean@fp32` | 28 | 15 | 3.525 | [3.28, 3.81] | 3.523 | 0.192 | 33.94x | 0.746 |
| `prefix10-g4+sqrt@fp32` | 28 | 15 | 1.974 | [1.84, 2.14] | 1.941 | 0.352 | 7.20x | 0.565 |
| `prefix3-g2@fp32` | 28 | 16 | 1.172 | [1.13, 1.22] | 1.142 | 0.489 | 3.23x | 0.530 |
| `prefix3-g2+mean@fp32` | 28 | 16 | 2.536 | [2.45, 2.62] | 2.545 | 0.292 | 12.63x | 0.559 |
| `prefix3-g2+sqrt@fp32` | 28 | 16 | 1.296 | [1.25, 1.34] | 1.288 | 0.473 | 3.65x | 0.467 |

`blocks` = executed decoder blocks (real compute); `depth` = sequential stages (critical path). Neither is a latency measurement.

## minimum effective depth at a ΔNLL budget

| ΔNLL budget | variant | depth | blocks | ΔNLL |
|---:|---|---:|---:|---:|
| 0.1 | not reached | — | — | — |
| 0.25 | not reached | — | — | — |
| 0.5 | not reached | — | — | — |
| 1 | not reached | — | — | — |
| 2 | `prefix10-g4@fp32` | 15 | 28 | 1.816 |
| 3 | `prefix6-g4+sqrt@fp32` | 12 | 28 | 2.421 |
