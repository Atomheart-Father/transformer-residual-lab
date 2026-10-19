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

### Stage 1.5 追加的产物

| 路径 | 内容 |
|---|---|
| `coeffs/<plan>__<scheme>__lam<λ>.json` | 拟合的 α、每组 `rel_reconstruction_rms`、`cond`、`n_clipped`、`fit` 元数据 |
| `selection.json` | validation 上按规则做出的选择（计划 → 方案 → 各方案 val ΔNLL），冻结后不得改动 |
| `b3/samples.jsonl` | 每个 `plan::mode` 变体一条；`boundary` 字段含 `rho`、`cosine`、`magnitude_share`、`direction_share` |
| `b3_scan/g<k>/` | 只在第 k 组出口做 oracle 干预的单独运行 |
| `matched_depth.csv` | 每个实际深度上并行与删层的逐项对齐（不是插值） |
| `threshold_depth.csv` | 每个 ΔNLL 预算下最优方案；达不到的预算写 `NOT REACHED` + NaN |
| `pareto_frontier.csv` | 按 (depth, ΔNLL) 的非支配集 |

`rel_reconstruction_rms` 的含义：用最优 α 线性组合重构真实组输出后剩下的相对 RMS。**它是 B2 类方法的上限，不是拟合误差的噪声。** 组越大，这个值越大（g2 0.41 → g8 0.70）。

### Stage 1.6 追加的产物

| 路径 | 内容 | 是否提交 |
|---|---|---|
| `data257/fresh_test.jsonl` | 全新独立测试集（seed 771013，与旧 test 逐 token 不相交） | 否（`data257/` 已 ignore） |
| `data257/fresh_test_manifest.json` | 抽取参数、seed、旧 offset 剔除记录、不相交断言结果 | 是 |
| `comp/<site>/pairs.npz` | 特征 `a` 与标签 `c*`，每 token 一行 | **否（约 129 MB）** |
| `comp/<site>/linear_r{8,16}.npz` | 线性补偿器权重 `A[D,r]`、`B[r,D]`、`x_mean`、`y_mean` | 是 |
| `comp/<site>/mlp_r{16,32,64}.npz` | MLP 补偿器权重 `W1`、`b1`、`W2`、`b2` | 是 |
| `comp/<site>/fit_report.json` | 每个 rank 的 train/val 相对误差、余弦、解释方差、epoch 曲线 | 是 |
| `comp/eval_*/eval_summary.json` | 每变体的 ΔNLL / KL / top1 / `recovery_vs_oracle` / `comp` 块 / 成本 | 是 |
| `comp/benchmark.json` | 各 rank 的实测延迟与峰值显存（50 次平均） | 是 |
| `comp/block_reference.json` | 单 block 前向参考耗时，作为成本分母 | 是 |
| `selection.json` | 冻结记录：候选结构的 cal/val ΔNLL、选定结构与补偿器、`fresh_test_used_for_selection: false` | 是 |

`eval_summary.json` 的 `comp` 块（每个补偿器一项）：

| 字段 | 含义 |
|---|---|
| `target_rms` | `c*` 的 RMS，补偿器要追的目标尺度 |
| `pred_rms` | `ĉ` 的 RMS。远小于 `target_rms` 说明在收缩，远大于说明发散 |
| `err_rms` | `||ĉ − c*||` 的 RMS |
| `rel_err` | `err_rms / target_rms`。**注意这是向量误差，不是 NLL** |
| `cosine` | `ĉ` 与 `c*` 的平均余弦 |
| `recovery_vs_oracle` | `(ΔNLL_none − ΔNLL_comp) / (ΔNLL_none − ΔNLL_oracle)`，相对 oracle 空间，不是相对串行 |
| `flops_per_token` | `4 × n_params`（两个矩阵乘各算 2 次），仅作量级参考 |
| `predict_ms_per_token` | 实测预测耗时。**引用成本时用这个，不要用 FLOPs** |

`recovery_vs_oracle` 可以 > 1（串行参考那行是 1.11），因为 oracle 本身受 bf16 舍入限制、不等于 0。不要把它当「超过上界」。


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

## Stage 1.7 产物

### `results/stage1_7/<run>/eval_summary.json`

由 `serial_compensate.py --stage eval` 写出。比 Stage 1.6 多了几个字段：

| 字段 | 含义 |
|---|---|
| `sites` | 补偿站点（group 索引列表），例如 `[4, 5]` |
| `site_layers` | 每个站点对应的层区间，例如 `[[11..19], [20..25]]` |
| `aggregate[v].gap_closure` | `(ΔNLL_base − ΔNLL_v) / ΔNLL_base`，**跨方法可比** |
| `aggregate[v].recovery_vs_oracle` | 相对本族 oracle，**跨站点集合不可比** |
| `aggregate[v].recovery_vs_oracle_ex_last` | 除最后一组外全 oracle，4 站点族内可比 |
| `aggregate[v].exit_rel_rms_mean` | 每组出口的 `rel_rms_diff` 均值序列，长度 = group 数 |
| `compensators[k].predict_ms_per_sample` | 该补偿器在评估中的实测耗时（含所有站点） |
| `warning` | 固定写着「顺序执行，不宣称加速」 |

### `results/stage1_7/<run>/fit_<tag>.json`

| 字段 | 含义 |
|---|---|
| `n_params` / `hidden` / `rank` / `carry_state` / `per_site` | 结构 |
| `site_weight` | `mse` 或 `normalized`；`site_weights` 是实际用的权重 |
| `site_c_rms` | 每个站点目标的 RMS，解释为什么需要归一化（可差 17 倍） |
| `final_train_mse` / `final_val_mse` | 注意：**不同架构的 MSE 不可比**（B1 与 Stage 1.6 的 MLP 不同尺度） |
| `per_site_val.siteN.rel_err` / `.cosine` / `.pred_rms` / `.target_rms` | 逐站点预测质量，**这是判断预测器强弱的唯一依据** |

### `results/stage1_7/table.json`

由 `make_stage17_table.py` 汇总，是报告表格与图表的**唯一数据来源**。

| 键 | 内容 |
|---|---|
| `rows[]` | 每行一个 (方法, 评估集)：`delta_nll`、`kl`、`top1`、`gap_closure`、`recovery_vs_oracle_ex_last`、`n_params`、`epochs`、`weights`（相对路径） |
| `site_ablation[]` | 同一个 4 站点 r32 权重 + 不同 `--apply-sites` |
| `final_site_pricing[]` | 最终 2 站点 B3 r64 权重的站点定价（4 / 5 / 4+5） |
| `references{}` | 每个评估集的 no-comp / oracle / oracle_ex_last / serial 参考值 |
| `bench{}` | `bench_serial_*.json` 的合并视图 |
| `missing[]` | 缺文件或缺键的记录；**非空就必须查，不能忽略** |

### `results/stage1_7/paired_stats.json`

逐样本配对 bootstrap。`d = nll(A) − nll(B)`，**d > 0 表示 B 更好**（NLL 更低）。字段：`mean_diff`、`se`、`ci95`、`frac_positive`、`n_paired`、`significant_95`。

配对必须按 `sample_id` 对齐，不能按行序——不同 run 的样本顺序没有保证。

### `results/stage1_7/drift.json`

前驱补偿对后续站点入口分布的影响：`entry_rms_r0/r1`、`entry_rel_token_shift`、`mean_vec_rel_change`、`cstar_rms_r0/r1`、`cstar_rel_change`。**这个文件是"必须重采样"的定量依据**，换站点集合就要重测。

### 大张量

`A.npy` / `C.npy`（`[n,S,T,D]`，float32，每个约 250 MB）与 `seq_pairs.npz` 只留在本地，`.gitignore` 已排除。提交的是 `serial_*.npz`（每个 270 KB–2 MB）与所有 JSON。

