# 输出格式与解释边界

## 运行目录

每次运行写一个目录（如 `results/smoke/`），必须包含：

- `manifest.json`：模型与 revision、mlx/mlx-lm 版本、平台、plan 定义（含 `executed_blocks`、`effective_depth`、`groups`、`deleted_layers`）、accum、git commit、状态。
- `samples.jsonl`：每样本一行，`variants` 键为 `plan@accum`。
- `groups.jsonl`：每样本 × 每组一行，记录入口/局部/出口误差与三项精确分解。
- `layers.jsonl`：每样本 × 每层一行，anchor 为 `teacher-group-entry`。
- `gram_<plan>.npy`：原始修正 Gram。
- `summary.csv` / `group_summary.csv` / `layer_summary.csv` / `summary.md` / `summary.json`。
- `figures/*.png`。

## 三类误差（不可混用）

| 字段 | 定义 | 能否部署 |
|---|---|---|
| `teacher_delta_rms` / `parallel_delta_rms` / `correction_rms` | `d_i^T`、`d_i^P(a)`、`c_i = d_i^T - d_i^P(a)`，`a` 为所属组的教师入口状态 | 仅诊断 |
| `local_rms_diff` | `||G_P(h_s^T) - h_e^T||`，在教师入口上跑并行组，不含前序误差 | 仅诊断 |
| `exit_rms_diff` | `||ĥ_e - h_e^T||`，候选完全自跑 | 可部署 |

`entry_rms_diff` 是进入该组时已经携带的误差。三者在组边界满足精确恒等式（fp32 下残差 < 1e-5）：

```
ĥ_e - h_e^T = (ĉ_s - t_s)                        继承
            + [G_P(ĉ_s) - G_P(t_s) - (ĉ_s - t_s)] 放大
            + [G_P(t_s) - h_e^T]                  新产生
```

对应字段 `decomp_inherited_rms`、`decomp_amplified_rms`、`decomp_new_rms`、`decomp_residual_rms`、`decomp_total_rms`。

## 质量指标

- `teacher_nll` / `candidate_nll` / `delta_nll`：causal shift 后的 nats/token。
- `teacher_to_candidate_kl`：全 vocab、float32 的 `KL(P_T || P_C)`，非 top-k 近似。
- `top1_agreement`：argmax 一致率。
- `token_nll_p50` / `token_nll_p95`：token 级分位数。
- `ppl_ratio = exp(mean delta_nll)`：相对困惑度倍数，样本 token 数相同时成立。
- `final_state.*`：`rms_diff`、`rel_rms_diff`（除以教师状态 RMS）、`cosine`、`max_abs_diff`。

## Gram / 谱

`G_ij = Σ_{samples,tokens,features} <c_i, c_j>`，`c_i` 取在**该层所属并行组的教师入口**，不是 embedding。

`gram_<plan>.json` 同时给出：

- `raw`：原始谱。受幅度主导，早层会压过其余层。
- `normalized_correlation`：`G_ij / (||c_i|| ||c_j||)` 的谱，单位对角，**引用方向冗余时用这个**。
- `layer_energy_share`：每层占总能量的比例，用于解释 raw 谱为何低秩。

两者都只描述修正的结构，**不证明**小网络可预测。可预测性需要下一轮学习实验。

## 状态取值

`NOT MEASURED` / `MEASURED` / `INCONCLUSIVE` / `FAIL` 必须显式保留，不得用占位图或推算值替代。
