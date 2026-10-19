# Stage-1 measured summary

- model: `mlx-community/Qwen3-0.6B-Base-bf16` revision `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- samples: 64 x 256 predicted tokens
- git: `afd9d9f2feb541f8c6fa54bde760f25496b02d0d`

| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `delete-keep7@fp32` | 7 | 7 | 15.973 | [15.45, 16.56] | 16.100 | 0.000 | 8648526.48x | 3.940 |
| `prefix3-g8@fp32` | 28 | 7 | 4.566 | [4.45, 4.69] | 4.553 | 0.132 | 96.17x | 1.071 |
| `delete-keep10@fp32` | 10 | 10 | 15.484 | [14.81, 16.18] | 15.575 | 0.000 | 5304091.98x | 3.064 |
| `prefix3-g4@fp32` | 28 | 10 | 3.994 | [3.88, 4.11] | 3.993 | 0.170 | 54.27x | 1.204 |
| `prefix3-g4+layer_ls_lam0.01@fp32` | 28 | 10 | 3.020 | [2.94, 3.10] | 3.034 | 0.247 | 20.49x | 0.793 |
| `delete-keep12@fp32` | 12 | 12 | 9.907 | [9.63, 10.20] | 10.001 | 0.010 | 20070.84x | 2.466 |
| `prefix6-g4@fp32` | 28 | 12 | 2.581 | [2.51, 2.65] | 2.563 | 0.278 | 13.21x | 0.753 |
| `prefix6-g4+layer_ls_lam0.01@fp32` | 28 | 12 | 2.094 | [2.04, 2.15] | 2.102 | 0.307 | 8.12x | 0.681 |
| `delete-keep14@fp32` | 14 | 14 | 12.572 | [12.40, 12.74] | 12.724 | 0.000 | 288453.98x | 2.145 |
| `prefix0-g2@fp32` | 28 | 14 | 9.148 | [8.87, 9.45] | 9.208 | 0.019 | 9399.51x | 1.145 |
| `prefix0-g2+layer_ls_lam0.01@fp32` | 28 | 14 | 6.862 | [6.68, 7.07] | 6.983 | 0.043 | 955.11x | 1.159 |
| `delete-keep15@fp32` | 15 | 15 | 12.317 | [12.14, 12.49] | 12.481 | 0.000 | 223573.86x | 2.223 |
| `prefix10-g4@fp32` | 28 | 15 | 1.827 | [1.77, 1.89] | 1.796 | 0.362 | 6.22x | 0.653 |
| `prefix10-g4+layer_ls_lam0.01@fp32` | 28 | 15 | 1.724 | [1.67, 1.78] | 1.704 | 0.359 | 5.61x | 0.562 |
| `delete-keep16@fp32` | 16 | 16 | 9.336 | [9.03, 9.68] | 9.418 | 0.006 | 11340.66x | 1.798 |
| `prefix3-g2@fp32` | 28 | 16 | 1.118 | [1.08, 1.16] | 1.138 | 0.504 | 3.06x | 0.529 |
| `prefix3-g2+layer_ls_lam0.01@fp32` | 28 | 16 | 0.961 | [0.92, 1.00] | 0.988 | 0.528 | 2.61x | 0.469 |
| `serial@fp32` | 28 | 28 | 0.000 | [0.00, 0.00] | 0.000 | 1.000 | 1.00x | 0.000 |

`blocks` = executed decoder blocks (real compute); `depth` = sequential stages (critical path). Neither is a latency measurement.

## minimum effective depth at a ΔNLL budget

| ΔNLL budget | variant | depth | blocks | ΔNLL |
|---:|---|---:|---:|---:|
| 0.1 | `serial@fp32` | 28 | 28 | 0.000 |
| 0.25 | `serial@fp32` | 28 | 28 | 0.000 |
| 0.5 | `serial@fp32` | 28 | 28 | 0.000 |
| 1 | `prefix3-g2+layer_ls_lam0.01@fp32` | 16 | 28 | 0.961 |
| 2 | `prefix10-g4+layer_ls_lam0.01@fp32` | 15 | 28 | 1.724 |
| 3 | `prefix6-g4+layer_ls_lam0.01@fp32` | 12 | 28 | 2.094 |

## depth-matched comparison against deletion

| depth | parallel | blocks | ΔNLL | delete | blocks | ΔNLL | advantage |
|---:|---|---:|---:|---|---:|---:|---:|
| 7 | `prefix3-g8@fp32` | 28 | 4.566 | `delete-keep7@fp32` | 7 | 15.973 | +11.407 |
| 10 | `prefix3-g4+layer_ls_lam0.01@fp32` | 28 | 3.020 | `delete-keep10@fp32` | 10 | 15.484 | +12.464 |
| 10 | `prefix3-g4@fp32` | 28 | 3.994 | `delete-keep10@fp32` | 10 | 15.484 | +11.490 |
| 12 | `prefix6-g4+layer_ls_lam0.01@fp32` | 28 | 2.094 | `delete-keep12@fp32` | 12 | 9.907 | +7.813 |
| 12 | `prefix6-g4@fp32` | 28 | 2.581 | `delete-keep12@fp32` | 12 | 9.907 | +7.326 |
| 14 | `prefix0-g2+layer_ls_lam0.01@fp32` | 28 | 6.862 | `delete-keep14@fp32` | 14 | 12.572 | +5.710 |
| 14 | `prefix0-g2@fp32` | 28 | 9.148 | `delete-keep14@fp32` | 14 | 12.572 | +3.424 |
| 15 | `prefix10-g4+layer_ls_lam0.01@fp32` | 28 | 1.724 | `delete-keep15@fp32` | 15 | 12.317 | +10.593 |
| 15 | `prefix10-g4@fp32` | 28 | 1.827 | `delete-keep15@fp32` | 15 | 12.317 | +10.490 |
| 16 | `prefix3-g2+layer_ls_lam0.01@fp32` | 28 | 0.961 | `delete-keep16@fp32` | 16 | 9.336 | +8.375 |
| 16 | `prefix3-g2@fp32` | 28 | 1.118 | `delete-keep16@fp32` | 16 | 9.336 | +8.218 |
