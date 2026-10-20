# HANDOFF_MINIMAX.md — transformer-residual-lab 研究交接

**交接日期**：2026-10-20 · **交接时 Git HEAD**：`d8970ec`（main，已推送，工作树干净）
**仓库**：https://github.com/Atomheart-Father/transformer-residual-lab.git
**读者假设**：你没有任何聊天记录。本文只依赖仓库内容，所有结论都给出可复核的文件路径。

---

## 1. 项目背景与研究目标

### 1.1 最初的研究问题（2026-10-19，`docs/2026-10-19_full_roadmap_v1.md`）

对一份**具体的、已预训练的** decoder-only Transformer（Qwen3-0.6B-Base，28 个 block），
能否不重新预训练、不动权重，仅通过校准样本和图结构搜索，找到一个比逐层串行链
**关键路径更短**、且在未参与搜索的数据上保留模型能力的**静态跨层依赖 DAG**。
动机：串行 Transformer 的依赖是"每层等所有前层"，但真实依赖可能是稀疏的——
如果成立，逻辑深度可以压缩。该文档明确不声称理论深度下降等于实际加速。

### 1.2 两条技术路线

- **Plan A（冻结权重 DAG）**：权重完全冻结，只改依赖结构——block `i` 的输入从
  `h_{i-1}` 换成某个前驱集合的汇合。允许改依赖与时序，不允许删 block、不训练。
- **Plan B（学习残差补偿）**：在 DAG 引入的误差之上，训练小型补偿网络恢复质量。
  Stage 1.x 实际走的是这条线（`results/stage1_5` … `stage1_8`，总入口
  `results/REPORT.md`）。结论：补偿有效但有限（单点补偿 −14.3% ΔNLL，
  rank-64 回收 oracle 空间 20% 未饱和；补偿器不跨语料；普通 MLP 仍是
  成本-质量最优点）。Plan B 目前**保留但不实施**。

### 1.3 Stage 2.0 与 Stage 2.1 各自回答什么

- **Stage 2.0**：冻结权重下，DAG 结构搜索（连续分组、稀疏 DAG、移位等）能走多远？
  产出：执行器 `dag_exec.py`（逐位等价 gate 0）、依赖探针、自由运行筛选、一次性
  盲测、位置效应（窗口 0–7 花 7.57 nats vs 窗口 10–17 只 0.19，40 倍）、
  "同窗口内质量≈只由关键路径深度决定"。
- **Stage 2.1**：两个收尾问题。**Part A**：M4 Pro 上 block 级并行有没有可兑现的
  wall-clock 收益？（回答：没有。）**Part B**：Stage 2.0 的负结果是不是搜索空间
  太窄造成的伪阴性？（回答：在前缀截断族内不是——精确限定见 §3A。）

### 1.4 为什么终止方案 A（实测支撑的决策，非新研究）

`results/stage2_1/REPORT.md` §5 四分类判定：落在「GPU 不能加速 且 找不到更好 DAG」：

1. **质量轴**：98 条窗口边自由运行全测无免费边；前缀截断族穷举 754 个结构后
   前沿单调平滑（盲测 d7 0.042 / d6 0.075 / d5 0.120 / d4 0.189 nats）。
2. **速度轴**：decode ~90% 是每 block 固定开销（0.558 ms 固定 + 0.0058 ms/token；
   每 block 31.5 MB BF16 权重 × 28 ≈ 881 MB/步），前驱集合怎么改都不触碰它；
   独立 block 单图重叠饱和于 ~1.46×（k=8）。
3. 附加教训：32 窗口上 <0.01–0.02 nats 的"改善"不可选型（cal 0.1758 →
   val 0.1864，CI 跨 0）——cal/val/final 隔离纪律正是为此设计。

### 1.5 下一阶段意向（**待讨论，未批准**）

方案 A 的根本约束是"结构是后验加在串行训练出的表示上的"。Stage 3.0 的初步
意向是换假设：层间严格串行依赖在多大程度上是**串行训练形成的 artifact**？
从零用 DAG 结构训练，模型能否学到适配该结构的表示，使更短依赖路径不掉点？
详见 §6。事实基础（已验证）：串行训练模型中每条上游边的缺失都有正代价；
但这**不等于**"原生 DAG 训练也会如此"——那是未来方向，没有任何实验支撑。
原始 Plan B 保留不实施；MoE 暂不扩展。

---

## 2. 当前实验状态与证据

### 2.1 模型与环境（接手前先确认能复现）

- **模型**：`mlx-community/Qwen3-0.6B-Base-bf16`，pinned revision
  `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8`（`scripts/measure.py:42-43`，
  所有脚本默认使用，不要改浮动 tag）。
- **硬件**：Apple M4 Pro，24 GB 统一内存，macOS 26.6.2。
- **软件**：Python 3.12.12（venv 在 `/Users/bozhongxiao/code/.venv`，不在仓库内）、
  mlx 0.31.2、mlx-lm 0.31.3。`requirements.txt` 未 pin 精确版本（已知欠账）。
- 无 GPU profiler 可编程访问（M4 Pro + MLX）；所有利用率数字是 wall-clock +
  FLOP/byte 推断，报告中已标注。

### 2.2 DAG 的数学定义与执行语义

**定义**（`docs/2026-10-19_full_roadmap_v1.md` §2、`scripts/dag_exec.py` 模块头）：
预归一化后以完整 block 为原子节点，`Δ_i = F_i(输入_i)`；串行链即
`输入_i = h_{i-1}`、`h_i = h_{i-1} + Δ_i`。DAG 把 `输入_i` 换成前驱集 `P_i` 的汇合。
图用 `name:spec` 完整表示（`make_dag`，`dag_exec.py:165`；spec 语法见
`docs/OUTPUT_SCHEMA.md`），如 `g2:|0|0,1|0,1,2|...`（28 段，逗号分隔前驱，
`|` 分段）。所有图都可以从 spec 逐位重建。

**执行语义**（`dag_forward`，`dag_exec.py:316`）：
- `merge_mode="materialized"`（默认）：推进一个运行中的 bf16 状态，汇合点 fp32
  累加。"串行延续捷径"（`:414-424`）仅当 `P_i == {0..i-1}` 且前一输出已含完整
  前缀时直接复用 `out_{i-1}`——**逐位正确性所必需**；"丢中边留后边"会被运行
  状态抵消成 no-op 假阴性（Stage 2.0 踩过）。
- `merge_mode="roadmap"`：每个输入都是 `h0 + Σ_{j∈P_i} δ_j`（fp32 累加后回 bf16）。
  权重轴 `weights` 只在此模式的输入汇合生效（`:408`），最终输出汇合（`:454`）不带权。
- 所有 28 个 block 永远执行（不删层），总计算量恒定——结构比较只比依赖不比计算量。
- **decode**：per-layer KV cache（`caches` 参数），DAG 只重组 hidden 输入，任意
  前驱集在 decode 下仍精确。prefill 必须 causal mask（`create_attention_mask(h0,
  None)`），单 token 步用 `None`；缓存解码第一步输入是 `ids[len]`。三对照正确性
  协议见 `results/stage2_1/decode_check/`。
- **深度是逻辑量**：`window_levels`（`stage21_search.py:136`）= 窗口内可调度层数。
  理论深度在 prefill 与 decode 都**零时延收益**（实测）。

### 2.3 Stage 2.0 关键结果（总报告 `results/stage2_0/REPORT.md`，原始数据同目录）

| 结论 | 数值 | 证据目录 |
|---|---|---|
| 执行器逐位等价 | max_abs=0 | `equivalence/` |
| 位置效应 40× | 窗口 0–7 花 7.57 nats，10–17 花 0.19 | `full/`, `screen/` |
| 同深度下连续均匀配对最优 | 盲测深度 27/24/22 = 0.031/0.184/0.550 nats | `blind_test/` |
| 每条上游边都有代价（当时 40/108 边） | 出口 rel 0.02–0.04 | `probe/` |
| OOD（C4）不推翻结论 | 同序 | `blind_ood/` |
| 理论深度零时延收益 | decode lazy 28→14：6.14→6.34 ms | `decode/`, `bench/` |

选型冻结在同目录 `selection.json`。**s20_test、s20_ood_c4 已耗尽，永久不可复用。**

### 2.4 Stage 2.1 关键结果（总报告 `results/stage2_1/REPORT.md`；`PART_A.md` 是 Part A）

Part A（`bench_a12/`、`bench_a345/`、`bench_a67/`，脚本 `scripts/stage21_bench.py`）：
- A1 单 block 成本分解：0.558 ms 固定 + 0.0058 ms/token → decode ~90% 固定开销。
- A2 独立 block 重叠：1.25×（k=2）/1.41×（k=4）/1.46×（k=8），次线性饱和。
- A5 权重堆叠 bmm：与 4 次单独 matmul 逐位等价，但 n=1 慢 29%、n=256 仅 1.25×
  ——"堆成 (4,1,dim) 就能批量"在 decode 尺度为假。
- A6/A6b：decode 计时 + 三对照正确性协议（mask 逐位确定；同图 cache vs 无缓存
  差异 ≈1% rel_rms、ΔNLL≤0.02；deployable 配置=串行 prefill+DAG decode 反而更好，
  g2 2.81 vs 2.93 nats）。

Part B（`search_*` 目录，全部 32 窗口 ΔNLL + 配对 bootstrap，脚本
`scripts/stage21_search.py`）：
- `search_single/`：98 条窗口边全测，距离效应 4×（d1 0.036 → d8 0.0087），无免费边。
- `search_greedy/`：深度受限贪心；单步闭包在 depth 7/6/5/4 各只有 7/15/10/1 个
  结构，唯一 depth-4 成员就是 g2。
- `search_random/`、`search_random_depth/`：不规则对照；**必须匹配深度**而非匹配
  miss（匹配 miss 被深度混淆，旧对照无效，报告已如实标注）。
- `search_exhaustive/`：**754 个可达前缀截断结构**（depth≤5、miss≤6）穷举；
  depth-4 最优 4-miss 结构恰为 g2（0.1788）；整体最优 depth-4（0.1758，5 miss）
  的优势在验证集上消失（→0.1864，vs g2 CI [−0.0028,+0.0176]）。**族限定见 §3A。**
- `search_val/` + `selection.json`：9 finalist 选型冻结。
- `search_blind/`：**一次性盲测**（s21_final 64 窗，`ONESHOT` 守卫文件已写入，
  脚本会拒绝重跑）：前沿 single-pair 0.0422（d7）→ greedy-depth6 0.0752（d6）→
  depth5-3miss 0.1195（d5）→ g2 0.1893（d4）；对 g2 全部配对显著；
  val→blind 漂移 ≤0.011 nats。
- `search_weights/`：γ 合并实验——**先读 §3B 的实现缺陷**，负结果方向可信但
  语义与设计意图不一致。
- `probe_gap/`：补测距离 5/7/8/10/12 的单边探针（`stage20_probe.py --distances`），
  把 40/108 覆盖提升到 78/108；探针→自由运行：单边 Spearman 0.84，多 drop
  （47 结构）0.96，ΔNLL ≈ 0.58 × Σprobe（可排序、不能定阈值）。
- 图：`figures/fig_single_edges.png`、`figures/fig_depth_frontier.png`
  （`scripts/stage21_figures.py`）。

### 2.5 数据划分、盲测规则与已耗尽的测试集

全部由 `data257/data_manifest.json` 登记（role/seed/n_windows/sha256/用途），
153 对 split 两两零 token 重叠（`splits_s21_disjointness` 等键）：

| split | seed | n | role | 状态 |
|---|---|---|---|---|
| s18 系列（train_8、train_8_val、test_v3、ood_imdb_v2） | 8800xx | — | Stage 1.x | 已用 |
| s20_cal / s20_val | 20261020/21 | 64/32 | 测量/选型 | 已用 |
| **s20_test** | 20261022 | 64 | **盲测，已耗尽** | 永不复用 |
| **s20_ood_c4** | 20261023 | 64 | **OOD 盲测，已耗尽** | 永不复用 |
| s21_cal | 20261024 | 48 | Stage 2.1 拟合 | 已用 |
| s21_val | 20261025 | 32 | Stage 2.1 选型 | 已用 |
| **s21_final** | 20261026 | 64 | **盲测，已耗尽**（ONESHOT 守卫） | 永不复用 |

规则：cal 只拟合、val 只选型（冻结进 selection.json）、final 只跑一次。
新盲测集用 `scripts/make_fresh_split.py` 重抽并**断言与全部既有 split 逐 token
不相交**。zsh 下多个 `--exclude` 必须显式重复写出。

### 2.6 可复核性分级

- **直接可复核**（本机重跑即得）：执行器等价性（`tests/test_dag_semantics.py`、
  `tests/test_dag_weights.py`）；A1–A7 计时（`scripts/stage21_bench.py`，分钟级）；
  探针与搜索（需先重建 split）。
- **需重建数据后复核**：所有 split jsonl 未提交（§4），但 manifest 有 sha256 与
  种子，可用同一 wikitext-2 修订版 + 同种子重抽并校验哈希。
- **需额外硬件才能复核**：Part A 时延结论是 M4 Pro/MLX 特定的；换机器或带
  profiler 的环境，速度数字必须重测（质量结论不受影响）。

---

## 3. 独立审查提出的未解决问题（已核实代码，如实记录）

### 3A. 754 个候选**不是**一般稀疏 DAG 的全局穷举

核实结论：**属实**。`enumerate_prefix_truncations()`
（`scripts/stage21_search.py:678`）只枚举**前缀截断**结构 `P_i = {0..t_i}`。
docstring 的论证是"对固定层级模式，前缀集是最大质量前驱集——多丢任何一条边
都有正代价且买不到深度"。该论证**不构成一般证明**：

- 它依赖"每条单边删除误差为正"（实测），但这只说明**固定层级模式下前缀集不劣于
  更稀疏的前缀集**，不能排除**非连续前驱集**更优。
- 非连续依赖（如 block 17 只读 `{0, 16}`——保留浅的远端锚、跳过近端前驱）会产生
  前缀截断**无法表达**的层级模式，这类结构从未被测量。
- 穷举限定：窗口固定 [10,18)、miss ≤6、window_levels ≤5。

因此 Stage 2.1 报告中 "exhaustive" 的正确读法是**"前缀截断族内的穷举"**。
正确结论："在可达前缀截断族中，Stage 2.0 家族最优；前沿单调平滑"，
**不是**"已证明一般 DAG 最优性"。若要挑战前沿，这是最直接的缺口。

### 3B. γ 合并实验存在实现与定义不一致

核实结论：**属实**。设计意图是随目标层与前驱距离变化的 `γ^(i-j)`，但实现：

- `scripts/stage21_search.py:840-844`（`mode_weights`）：
  ```python
  for i in range(LO, HI):
      for j in sorted(d.preds[i]):
          w[j] = float(g ** (i - j))   # 以 j 为键，i 循环覆盖
  ```
  权重字典**只以前驱 `j` 为键**；同一 `j` 被多个目标层读取时**最后一次写入覆盖**
  （消费端 `dag_exec.py:310` 的 `weights.get(j, 1.0)` 也只能按 `j` 查）。
- 对实测结构 g2，最终生效语义是 `w[j] = γ^(17-j)`（对所有 j≤15，以窗口末端为锚的
  全局几何衰减，对**所有**目标层相同），**不是**逐 (i,j) 的 `γ^(i-j)`。
- 影响评估：`search_weights/` 的负结果（γ=0.9/0.8 花 9.3–10.3 nats、top1→0；
  γ=1 逐位复现，单测 `tests/test_dag_weights.py`）仍表明**对残差更新做非 1 缩放
  是破坏性的**，方向可信；但 9–10 nats **不能被引用为"距离加权合并语义已被
  正确测量"**——逐 (i,j) 的距离加权从未被测过。
- 处置：按交接要求不修复、不重测。未来重做时需把权重接口改为 `(i, j) -> w`
  二维键，或明确声明测的是全局衰减语义。

### 3C. 其他已知边界（保留）

- 无 GPU profiler：利用率/带宽为推断值，报告已标注。
- 硬件特定性：Part A 速度结论仅对 M4 Pro/MLX 成立；质量前沿结论硬件无关。
- 有限窗口：Part B 全部在窗口 [10,18)（`stage21_search.py:64`）；位置效应跨窗口
  差异巨大（40×），数字不可跨窗口迁移，机制性结论（距离效应、闭包形状、合并
  语义）是对执行器的陈述。
- 质量指标只用 wikitext-2 的 ΔNLL/KL/top1；Stage 2.1 未在 C4/IMDB 复验。
- 隐藏态 rel_rms 会**高估**能力（状态 MSE 改善 25% 时 ΔNLL 只改善 16%），
  能力只看 ΔNLL/KL/top1。
- Stage 1.x 的补偿器不跨语料（wikitext 关 20.7%，IMDB 只关 3.0%）。

---

## 4. 仓库接手指南

### 4.1 目录地图

```
scripts/            全部实验代码（27 个 py）
  measure.py          模型加载/ pinned revision/ logits 指标（公共底座）
  dag_exec.py         DAG 执行器：Dag 类(:63)、make_dag(:165)、_merge(:299)、
                      dag_forward(:316)；一切结构实验的语义核心
  prepare_data.py     抽取 257-token 窗口（--exclude 显式重复）
  make_fresh_split.py 新独立 split + 逐 token 不相交断言
  stage20_probe.py    单边删除探针（--distances 可扩）
  stage20_screen.py   自由运行筛选
  stage20_bench/decode/eq/full  Stage 2.0 计时/decode/等价/全模型扫描
  stage21_bench.py    Part A：A1–A7 计时
  stage21_search.py   Part B：single/families/greedy/random/random-depth/
                      exhaustive/weights/finalists/blind 九种模式
  stage21_decode_check.py  三对照 decode 正确性
  stage21_figures.py  报告图
  compensate.py / serial_compensate.py / calibrate.py / diagnose_b3.py
                      Stage 1.x Plan B 补偿线
  tiny_serial.py      Stage 1.8 串行单元蒸馏线
  make_stage17_table.py 等：表格生成（报告数字只准来自 table.json）
tests/              三个单测：dag 语义、权重轴、group 语义
data/  data257/     窗口数据（jsonl 不提交，manifest 提交）
results/            每次运行一个目录：manifest + samples/groups/... + REPORT.md
docs/               路线图、RUNBOOK、OUTPUT_SCHEMA、PRIOR_WORK 等
```

### 4.2 重要配置与 checkpoint

- 模型 checkpoint 不在仓库：首次运行由 mlx-lm 从 HF 下载 pinned revision 到
  `~/.cache/huggingface`（仓库 `.gitignore` 排除 `*.safetensors`/`*.mlx`）。
- 无其他配置文件需要关心；所有脚本参数走 CLI（`--help` 可查）。

### 4.3 测试与实验记录怎么查

- 单测：`/Users/bozhongxiao/code/.venv/bin/python tests/test_dag_semantics.py`
  （直接运行式，无 pytest 依赖）。
- 每个结果目录的 `manifest.json` 含 status/model/revision/git_commit/note；
  `search.json`/`table.json` 是汇总数字的**唯一**权威来源，禁止手抄图和表。
- `.workbuddy-ai/memory/`（本地，不提交）有逐日工作日志和踩坑记录。

### 4.4 证据 vs 临时文件

- **证据**：`results/stage2_0/`、`results/stage2_1/`、`results/stage1_5..1_8/`、
  `results/REPORT.md`、各 selection.json、figures。
- **临时/冒烟**：`results/smoke/`、`results/smoke_18/`、`results/devcheck/`
  （后两者 gitignore）。
- 大文件（A.npy/C.npy/pairs.npz/fit 权重）不提交，规则见 `.gitignore` 注释。

### 4.5 未提交到 GitHub 的数据

`data/` 与 `data257/` 的 **所有 jsonl**（30 个窗口文件）本地不提交，只有 5 个
manifest json 在库里。重建方法：同 wikitext-2 修订 + manifest 里的种子 +
`prepare_data.py`，然后校验 sha256。**没有 manifest 对应的本地数据一律不可信。**

### 4.6 如何避免重复消耗旧盲测数据

- s20_test、s20_ood_c4、s21_final 三个盲测集**已耗尽**（§2.5 表）。
- `stage21_search.py --mode blind` 有 ONESHOT 文件守卫
  （`results/stage2_1/search_blind/ONESHOT`），重跑会被拒绝。
- Stage 2.0 的 blind 目录没有脚本级守卫，靠纪律：不要再碰那两个 split。
- 新盲测需求 → `make_fresh_split.py` 重抽 + 断言不相交。

### 4.7 Git 状态

- HEAD = `d8970ec`（main，origin/main 同步），工作树干净，无未提交内容。
- 关键提交链：`3fa7884`（Stage 2.0 执行器+gate0）→ `06fedfd`（盲测+位置效应）→
  `566f26a`（Stage 2.0 报告）→ `1aefea3`（Stage 2.1 Part A+B 机制）→
  `d8970ec`（穷举+选型+盲测+最终报告）。
- 已知欠账（非阻塞）：requirements.txt 未 pin 精确版本；Stage 2.0 blind 目录
  无脚本级 ONESHOT 守卫。

---

## 5. 交接结束状态

本文档为唯一新增文件；未修改任何实验代码、原始结果与数据。
未运行 GPU 实验、未启动新研究、未下载模型、未进入 Stage 3.0。
文中路径、SHA、数字均已对照仓库核实（模型 revision、窗口定义行号、
mode_weights/_merge 源码、data_manifest split 表、git log）。

---

## 6. 下一阶段的研究方向（**待讨论意向，不是已批准方案**）

**Stage 3.0 — 从零训练原生 DAG Transformer（初步意向）**

研究问题：层间严格串行依赖在多大程度上由**串行训练**形成？若从零以 DAG 结构
训练，模型能否学到适配短依赖路径的表示？

初步设想的对照（同训练预算、同数据）：
1. 12 层串行 Transformer（对照基线）
2. 8 层串行 Transformer（深度直接缩减的对照）
3. 12 block、短依赖路径的 DAG Transformer
4. 更激进的并行 DAG Transformer

**明确未定**：模型架构、数据、训练预算、云 GPU、相关论文调研、对照实验设计。
**约束继承**：不做 MoE 扩展；Plan B 保留不实施；报告纪律（三误差分类、
配对 bootstrap、table.json 单一数字来源、cal/val/final 隔离）应延续到 Stage 3.0。
在方案获批前，不要动现有 results/ 与 scripts/。
