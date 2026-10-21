# R1 归档（R2 任务 A 的第一轮成果）

本目录是 **R1（任务 A：推理算法与计算数学全景研究）** 产物的**副本**。原件仍在仓库的 `research/` 下
（`research/REPORT.md`、`research/PROTOCOL.md`、`research/notes/`），**未移动、未删除、未改写**。

按 `research/agent-reports/README.md` 的提交契约第 2 条："For previously completed R1 reports currently
only local, copy them into your own `r1-archive/` folder. Do **not** relocate/delete the existing source files."

| 文件 | 内容 | 行数 |
|---|---|---|
| `REPORT.md` | R1 主报告：10 个方向 + 遗漏家族扩展，A–G 七项交付 + 独立判断 | 773 |
| `PROTOCOL.md` | R1 研究协议：统一判据、**L0–L4 无损等级定义**、每方法必填字段、输出格式 | 90 |
| `notes/X0_crosscutting.md` | 交叉核验笔记：硬件约束模型、推测解码成本模型、2026 一手来源详细数字、**第 5 节：第二轮新增的运行时互斥发现** | 约 640 |
| `notes/X1_evidence_index.md` | 证据索引：逐条 URL / 核验日期 / 原文数字 / 证据等级；**N 组：第二轮核验的 mlx-lm 源码事实、`mx.device_info()` 硬件事实、BaseRT 两篇、SGLang 官方文档** | 约 200 |

## ⚠ 阅读本归档前请注意一处已否证的锚点

**`PROTOCOL.md` 第 82–86 行**的"Anchor fact"段落写着：

> decode 约 **90 % 是固定 per-block 开销**；每步权重流量 ~881 MB；**0.558 ms 固定成本 per block** 主导 0.0058 ms/token 边际项。

**这三句话已在 R1 末期被同仓库的独立系统审计否证**（口径问题，不是被推翻的物理结论）：

- "0.558 ms/block 固定开销"是**外推**而非测量（n=1..16 的点全落在 0.47–0.68 ms 且非单调）。
- **"decode 90 % 是固定开销"在 n=1 时是恒真的**（只有一个 token），不构成机制证据。
- 真正的机制证据是**链内 0.253 ms/Block vs 孤立 0.47–0.68 ms/Block** ⇒ 每次 eval 的提交/同步单位成本约 **0.2–0.4 ms**。
- **同一段 28-block 代码在两轮测量间相差 1.9×** ⇒ **绝对毫秒不可跨报告引用**。

**修正后的锚点**（见 `REPORT.md` §0.2）：

| 锚点 | 值 | 口径 |
|---|---|---|
| decode 有效带宽 | **124.6 GB/s = 273 GB/s 峰值的 46 %** | 880.8 MB ÷ 7.07 ms，不含 LM head |
| prefill 有效算力 | **2.06 TFLOP/s** | 241 GFLOPs ÷ 116.9 ms |
| LM head 占单步字节 | **311.2 MB = 0.6B 模型单步流量的 26 %** | vocab 151936 × 1024 × 2 B，tied embedding |
| 每 eval 提交/同步成本 | **0.2–0.4 ms** | 链内 0.253 vs 孤立 0.47–0.68 ms/Block |

**本归档保持原样复制，不修改**——以保留第一轮交付物的原始状态。**引用 R1 数字时，请以 `REPORT.md` §0.2
与 R2 报告 §0.2 为准。**

## R1 与 R2 的关系

R2（Issue #1）**不重复** R1 的内容，只在两处交叉：

1. **R1 §0.4-1 判定"KV 压缩在消费级是支线"** —— R2 从**权重侧**给出同向但更强的一条：
   **2-bit 的真实瓶颈不是编码，是反量化内循环**（Proteus / CurveTQ / XOR-Trellis 三方独立收敛）。
2. **R1 实测"Q2_0 2.40 bpw decode 93.79 t/s vs IQ2_XS 2.50 bpw 70.30 t/s（1.34×）"** ——
   R2 从文献侧独立复现了"**格式比位宽更重要**"这一现象（SpQt 的 Apple Silicon 实测、Proteus 的
   "零查表内循环"、第三方能耗实测），并给出机制解释。

详见 `../2026-10-r2-report.md` §0.3。
