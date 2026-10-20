# M1 原始数据索引

本目录是 M1「真实残差缺失的子模块级追踪」的全部原始产物。
**独立目录，未覆盖任何既有结果**（`results/stage2_1/`、`results/smoke/`、`results/stage2_0/` 均未改动）。

- **运行**：`scripts/m1_submodule_trace.py`（主实验）、`scripts/m1_pos0_diagnostic.py`（事后归因诊断）、
  `scripts/m1_analyze.py`（聚合与作图）
- **模型**：`mlx-community/Qwen3-0.6B-Base-bf16` @ `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`
- **数据**：`data257/s21_cal.jsonl` 前 8 个窗口（**未使用**已耗尽的 `s21_final`）
- **运行时长**：40.9 s（MLX / Apple Silicon）
- **闸门**：全部通过（`manifest.json` → `status: DONE`）

---

## 1. 文件清单

| 文件 | 大小 | 行/条目 | 内容 |
|---|---:|---:|---|
| `PREREGISTRATION.md` | 10.4 KB | — | 运行前定稿的预注册（未修改） |
| `PREREGISTRATION_AMENDMENT.md` | 15.5 KB | — | A1（运行前修订：语义与闸门）/ A2（运行后修正记录：D1–D4 缺陷与事后诊断声明） |
| `edge_selection.json` | 699 B | 4 条边 | 预注册选型结果（规则只用既有 Stage 2.1 数据） |
| `manifest.json` | 1.8 KB | — | 运行清单：模型、revision、样本 id、边、条件、读出点、随机种子、闸门状态、耗时 |
| `gate1_equivalence.json` | 257 B | 4 项 | 闸门 1：读出路径逐位一致性（layer 0/14/27 `max_abs=0.0`；`x_{i+1}==out_i` 0.0） |
| `gate2_semantics.json` | 7.7 KB | — | 闸门 2：b 两种实现一致性（32 行，最大差 0.0）、c S-STAGE2 排序、d `dag_forward`≡`measure_batch`（0.0） |
| `gate2_stage2_replication.json` | 2.8 KB | 3 变体 | 闸门 2a：仓库自带 `measure_batch`（32 样本）复现 `search.json` 的逐位对比 + 逐样本值 |
| `local_readouts.jsonl` | 790 KB | **224 行** | **主数据**：每行 = 一个 (样本 × 边 × 条件) 的 7 个子模块读出 + 扰动元数据 + 端到端指标 |
| `suffix_propagation.jsonl` | 909 KB | **224 行** | 每行 = 同一条件下"block k 之后的状态"相对真链的逐层偏离轨迹 |
| `stage2_contrast.jsonl` | 103 KB | **32 行** | S-STAGE2 语义（仓库执行器 `dag_forward`，未重实现）下的状态轨迹 + 端到端指标 |
| `pos0_diagnostic.json` | 146 KB | 28 层 × 8 样本 + 通道分解 | 事后诊断：逐层 position-0 占比、通道分解、归因测试 |
| `pos0_attribution.jsonl` | 34 KB | 64 行 | 事后诊断：只扰动 position 0 vs 只扰动其余 256 个 token |
| `layer_stats.json` | 82 KB | 28 层 × 8 样本 | 教师链逐层统计（‖x‖、‖h‖、‖r_attn‖、‖r_mlp‖、‖Δ‖、相对占比、cos(x, r_attn)） |
| `summary.json` | 334 KB | — | 全部原始值的聚合（均值/中位数/sd/n），**报告中的表由它生成** |
| `figures/fig1_submodule_profile.png` | 144 KB | — | 同幅度大扰动下的子模块响应剖面（4 条边 × 4 条件） |
| `figures/fig2_eta1_profile.png` | 110 KB | — | η=1% 小扰动下的子模块响应剖面（局部 Jacobian） |
| `figures/fig3_suffix_propagation.png` | 90 KB | — | 后缀传播轨迹：S-PERSIST vs S-STAGE2 |
| `figures/fig4_semantics_damage.png` | 41 KB | — | 端到端代价：Stage 2.1 记录 / S-STAGE2 / S-PERSIST |
| `figures/fig5_local_vs_global.png` | 39 KB | — | 局部响应增益 vs 端到端 ΔNLL（解耦图） |

---

## 2. 数据模式（JSONL 每行的字段）

### 2.1 `local_readouts.jsonl` — 224 行 = 8 样本 × 4 边 × 7 条件

```
sample_id      str    窗口 id，如 "wikitext2:test:5911"
edge           str    "P1" | "P2" | "C1" | "C2"
reader         int    读者层 i（= 施加扰动的 block）
contributor    int    贡献者层 j（= 被移除的更新 Δ_j 所在层）
condition      str    "real" | "matched_radial" | "matched_tan" | "matched_rand"
                     | "small_radial" | "small_tan" | "small_rand"
perturbation   obj    abs_norm_mean / abs_norm_median
                     rel_norm_mean / rel_norm_median / rel_norm_rms_weighted
                     radial_fraction_mean / tangential_fraction_mean / cos_with_x_mean
readouts       obj    7 个读出点（ln1, r_attn, h, ln2, r_mlp, out, delta_new），每个含
                     abs_change_mean          绝对变化 mean_t ‖o'−o‖
                     rel_change_mean          相对变化 mean_t ‖o'−o‖/‖o‖（含分母守卫）
                     rel_change_median
                     rel_change_rms_weighted  rms 口径
                     rel_change_excluded_tokens  被分母守卫排除的 token 数
                     rel_denominator_median
                     response_gain_abs        ‖o'−o‖/‖δ‖
                     response_gain_rel        相对响应增益（无量纲）
                     cos_pert_base_mean       cos(o', o)
                     baseline_norm_mean
final          obj    teacher_nll / candidate_nll / delta_nll
                     teacher_to_candidate_kl / top1_agreement
```

### 2.2 `suffix_propagation.jsonl` — 224 行

同上，但 `suffix` 是一个数组，**每项对应一个 block k（k = reader … 27）**：

```
suffix[i].layer            int    block 序号
suffix[i].rel_change       obj    mean / median / p90 / max / rms_weighted / n_excluded
                                  —— 比较对象是"block k 之后的状态"，
                                     教师对应 states_after[k+1]
suffix[i].abs_change_mean  float
suffix[i].response_gain_abs float
```

### 2.3 `stage2_contrast.jsonl` — 32 行 = 8 样本 × 4 边（S-STAGE2 语义）

```
sample_id / edge / reader / contributor
final      obj    同上（teacher_nll / candidate_nll / delta_nll / KL / top1）
trajectory arr    每项 {layer, rel_change{...}, abs_change_mean}
                  —— S-STAGE2 下"block k 之后的状态"相对真链的偏离
```

### 2.4 `pos0_attribution.jsonl` — 64 行 = 8 样本 × 4 边 × 2 变体（**事后诊断，非预注册**）

```
sample / edge / reader / contributor
variant         str   "pos0_only" | "all_but_pos0"
pert_norm_mean  float 被施加扰动的逐 token 范数均值
rel_change      obj   7 个读出点的相对变化（相对各自教师值）
final           obj   端到端指标（S-PERSIST）
```

### 2.5 `pos0_diagnostic.json`

```
layer_norms[]        {sample, layer, x_norm_pos0, x_norm_median, x_norm_max,
                      x_pos0_over_median, x_argmax_token,
                      d_norm_pos0, d_norm_median, d_norm_max,
                      d_pos0_over_median, d_argmax_token}
channel_decomposition[] {sample, vec("x15_pos0"|"delta2_pos0"), total_norm,
                      top1/top1_ch, top2/top2_ch, top3/top3_ch, top8_share}
attribution[]        同 §2.4
```

---

## 3. 口径约定（读数时务必对照）

| 量 | 定义 | 注意 |
|---|---|---|
| `rel_change_*` | 逐 token 比值的均值 | 与 `rel_change_rms_weighted` **可差一个数量级** |
| `rel_change_rms_weighted` | `rms(Δ)/rms(base)`，整个张量 | 被**单个巨激活 token**主导（本模型为 position 0） |
| `rel_norm_rms_weighted` | `rms(δ)/rms(x)` | 同上，(15,2) 的该值为 0.993 而逐 token 中位数只有 0.105 |
| `response_gain_rel` | `rel_change_mean ÷ mean_t(‖δ(t)‖/‖x(t)‖)` | 无量纲；跨读出点/跨条件可比的局部判据 |
| 分母守卫 | 教师范数 < 1e-3 × 同一样本中位数 ⇒ 该 token 排除 | 同时报告被排除 token 数 |
| bf16 地板 | 被扰动输入存回 bf16 带来 O(2⁻⁸)≈0.2% 的与方向无关的相对扰动 | η=1% 档的**绝对**变化不可直接解读；条件间比较不受影响 |
| 教师 logits | 最后一个 block 的输出（`out_27`），不是 `states[-1]` | 缺陷 D1 的修正，见 A2.1 |

---

## 4. 复现

```bash
cd /Users/bozhongxiao/Desktop/transformer-residual-lab
export TOKENIZERS_PARALLELISM=false
V=/Users/bozhongxiao/code/.venv/bin/python

$V scripts/m1_submodule_trace.py --out results/mech_m1 --n-samples 8 --replicate-stage2
$V scripts/m1_pos0_diagnostic.py
$V scripts/m1_analyze.py
```

主实验约 41 s，诊断约 40 s。全程只读既有权重与数据，不写 `results/` 以外的目录。
