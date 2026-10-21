# R2 一手全文核验笔记（full-text verification notes）

本目录是 `2026-10-r2-report.md` / `-evidence.md` / `-combinations.md` 的**底层一手材料**，
保留原始核验记录以便交叉审查时回溯到"每一条结论是从哪一页/哪一段读出来的"。

| 文件 | 范围 | 覆盖 |
|---|---|---|
| `A-extreme-lowbit.md` | 家族 A：极低位宽权重表示 | QTIP `2406.11235`、Proteus（ICML'26）、CurveTQ `2610.09212`、XOR-Trellis `2610.00432`、OSAQ `2605.04738`、YAQA `2505.22988`、GSQ `2604.18556`、RCO `2605.00649`、ParetoQ `2502.02631`、BCJR-QAT `2605.10655`、QTEA `2609.00224`、FastKron `2608.06291` 等 |
| `B-dynamic-sparse-ffn.md` | 家族 B：冻结权重的 FFN 动态稀疏 | Prox `2607.27591`、SpQt `2511.04477` **全文通读**（LaTeXML HTML + PDF 表格），条目可回溯到章节号；另含 TEAL / CATS / M-COUNTDOWN / R-Sparse / WiSparse / DAWS / SVD-CSP / SparseInfer / GRASP / SkipDecode / Loki 等邻近工作 |

## 阅读约定

- 证据等级标记沿用主报告：**`[P]`** 一手已打开 / **`[O]`** 官方产物直读 / **`[S]`** 二手 / **`[UNVERIFIED]`** 未打开。
- 每篇论文都记录了**打开方式**（arXiv abs / HTML render / PDF `pdftotext` / OpenReview API / 会议页 / GitHub API / HF API），
  以便判断某个数字来自正文、图表还是元数据。
- 笔记中**保留了失败记录**（403 / 404 / 429 / 渲染缺字），因为"哪些东西打不开"本身就是本报告 §11 的证据。

## 与主报告的关系

- 主报告是**结论层**，本目录是**证据层**。两者若有冲突，以主报告的谨慎措辞为准。
- 本目录中标记 `[UNVERIFIED]` 的条目，在主报告中**不得**作为结论使用。
- 家族 C（函数等价重参数化）与家族 D（结构感知降维 / A³）的核验记录**未单独成文**：
  该两族由报告作者本人直接经 arXiv API + 官方会议页核验，其条目已直接落在
  `-evidence.md` 中；主报告 §4 / §5 的每条 `[P]` 标注都可在 `-evidence.md` 找到对应 URL 与版本日期。

## 纪律声明

本轮为**纯研究**：未下载模型、未训练、未运行 GPU 基准、未修改 `scripts/` `tests/` `results/`
`docs/` 或任何旧实验产物。笔记中的所有数字均为**他人论文中的自报数字**，
除非明确标注 `[推算]`（本报告自行算术）或 `[O]`（本报告直接读到的官方产物）。
