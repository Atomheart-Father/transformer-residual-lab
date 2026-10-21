# R3-W 勘误表：R2 报告的数据与逻辑错误

**范围**：`2026-10-r2-report.md`、`2026-10-r2-kernel-matrix.md`、`2026-10-r2-audit-corrections.md` 中经 GPT 交叉核查确认有误的条目
**性质**：逐条给出「旧判定 → 正确判定 → 可核验来源」。**不撤回** R2 中未被本表列出的结论。
**日期**：2026-10-21

---

## 0. 编号与严重度

| ID | 对应 Issue #4 条目 | 严重度 | 一句话 |
|---|---|---|---|
| E-1 | P0-1 | **致命** | 兑现率公式倒置，理想值算成 3.39 而非 1 |
| E-2 | P0-2 | **致命** | 用 MLX affine 2-bit 的速度论证 IQ2_XS 值不值 |
| E-3 | P0-3 | 严重 | XD-3 归因判据写反 |
| E-4 | P1-4 | 严重 | GB 与 GiB 混用；+12% 代理开销套错基线 |
| E-5 | P1-5 | 严重 | 7.07 ms 与 162.9 tok/s 来自不同轮次，被混在一句话里 |
| E-6 | P1-6 | 严重 | M2 Max 819.2 GB/s 与官方 400 GB/s 矛盾 |
| E-7 | P1-7 | 严重 | 2.77× 是权重字节比，被写成端到端加速预期 |
| E-8 | P2-8 | **事实错误** | "Metal 上没有任何 trellis 实现"——uzu 有，且 2.0 bpw |
| E-9 | 自查新增 | 中 | GSQ-RCO 的 27B 产物是二手博客转述，被当成一手 |
| E-10 | 自查新增 | 中 | R2 §3.1 的 ALU 预算表建立在 124.6 GB/s 上，而该值本身待验证 |

---

## E-1 兑现率公式倒置（P0-1）

**旧判定**（R2 报告 §4 XD-1）：
```
兑现率 = (bpw4 / bpw2) / (t2 / t4)
```
**为什么错**：若理想情况（时间严格正比于字节），`t2/t4 = bpw2/bpw4`，代入得
```
R_old = (bpw4/bpw2) / (bpw2/bpw4) = (bpw4/bpw2)² = 1.84² = 3.39
```
即"完美兑现"被算成 **3.39**，而报告把阈值设在 0.70——**任何正常结果都会通过，判据完全失效**。这是把一个"越小越好"的量当成了"越大越好"。

**正确判定**：
```
R = (t4 / t2) / (B4 / B2)
```
其中
- `B` = **实际**量化后内核读取的 total bytes，含 packed codes、scales、biases、以及任何 metadata（per-group scale / min / zero），由 `q.nbytes + scales.nbytes + biases.nbytes` 逐张量求和，**不得用名义 bpw**；
- `t` = **配对**实测时间（同一进程、交替顺序、N≥40 对、取配对中位数）。

理想 `R = 1.0`。

**重设阈值**：

| R 区间 | 判定 | 动作 |
|---|---|---|
| **R > 1.05** | 测量异常或 4-bit 基线未调优 | **不得下结论**，先重测/重调 4-bit 基线 |
| 0.85 ≤ R ≤ 1.05 | 位宽收益基本兑现 | 进入质量关卡 |
| 0.70 ≤ R < 0.85 | 部分兑现 | 按形状分档判断（见 E-2 的 Stage A 分档） |
| 0.45 ≤ R < 0.70 | 大部分被 ALU 吃掉 | 条件性，只有特定形状值得 |
| **R < 0.45** | 否证 | 停在 4-bit |

**质量关卡必须独立设置**（速度通过不代表可用）：
- 2-bit 的 held-out PPL ≤ **1.15 ×** 同模型 4-bit 的 PPL；
- 一个下游任务（如 5-shot MMLU 子集或 wikitext 上的 top-1 一致率）下降 ≤ **2 分**；
- 任一不通过 → 该位宽直接淘汰，速度数据仅作记录。

**来源**：Issue #4 P0-1；公式推导为本轮自查（可手算复核：1.84² = 3.3856）。

---

## E-2 XD-1 对象不对应命题（P0-2）

**旧判定**：XD-1 用 `mx.quantized_matmul` 的 MLX affine 4-bit vs 2-bit 测"兑现率"，并声称这回答"2.3125 bpw 的 IQ2_XS 是否值得"。且写了"合成 IQ2_XS 若无现成 kernel 则跳过"。

**为什么错**：
1. **MLX affine 2-bit ≠ llama.cpp IQ2_XS**。前者是 group-64 对称仿射（scale+bias），后者是 512 项 8 元素码本 + 符号 + 子块 scale。两者的解码算术差一个量级（仿射 ~2 ops/weight vs 码本 ~2.5–4），**位宽相同不代表 kernel 成本相同**。
2. 若"无 kernel 时跳过 IQ2_XS"，则整个测试里根本没有 IQ2_XS 的数据，**命题无法被回答**。
3. 更根本：**跨 backend 的速度差里混着 kernel 质量差**，不能当作位宽证据。

**正确判定**：拆成四段，每段只回答一个问题。

| 段 | 问题 | backend | 变量 | 允许得出的结论 |
|---|---|---|---|---|
| **Stage A** | 在同一 kernel 家族内，2-bit 相对 4-bit 能兑现多少字节优势 | **MLX only** | bits = 2 / 4 / 8（同一 `quantized_matmul`） | 只能回答"MLX affine 的 2-bit 在这台机器上值不值" |
| **Stage B** | IQ 家族内，2.31 bpw 相对 4.25 bpw 能兑现多少 | **llama.cpp only**（`llama-bench`） | `-ctk/-ctv` 与 `--type` = IQ4_XS / IQ2_XS | 只能回答"IQ2_XS 在 llama.cpp Metal 上值不值" |
| **Stage C** | 各格式的**质量**代价 | 与 backend 无关 | 同一源模型 → 各格式 | 质量门槛，独立于速度 |
| **Stage D** | 真实模型端到端 decode | 选定 backend | 含 head、含采样 | 最终 tok/s |

**硬规则**：
- **Stage A 的速度不得用来支持任何 IQ2_XS 的结论**；Stage B 的速度不得用来支持任何 MLX 的结论。
- 若 Stage B 因环境原因无法执行（如 llama.cpp 未安装），则**必须把"2.3125 bpw 是否值得"标为未回答**，不得用 Stage A 顶替。
- 合成 GEMV（大张量、单一形状）与真实模型 decode **必须分开报告**，不得用前者推算后者。

**来源**：Issue #4 P0-2；`mlx_lm` 与 `llama.cpp` 的格式定义差异见 `2026-10-r2-kernel-matrix.md` §1.1/§1.3。

---

## E-3 XD-3 归因判据写反（P0-3）

**旧判定**（R2 报告 §4 XD-3）：
> "若两者在同一进程内差异 < 5% → 差异来自代码；若仍 1.9× → 差异来自环境。"

**为什么错**：逻辑完全反了。**在同一进程、同一环境里复现出 1.9× 的差距，恰恰证明差距来自代码**（因为环境已被固定为常量）；**差距消失，才说明当初的 1.9× 来自历史环境**（热态、并发进程、wired limit、降频）。

**正确判定**：

| 同环境配对结果 | 归因 | 后续动作 |
|---|---|---|
| 配对 CI 排除 1.0，中位比 ≥ 1.5 | **代码路径差异** | `git diff 73d3379 6a5e6ea -- scripts/` 逐行定位 |
| 配对 CI 包含 1.0 | **历史环境 / 测量条件差异** | 历史数字永久作废；记录当时的 wired limit、并发进程、GPU 频率 |
| CI 排除 1.0，中位比在 1.1–1.5 | 两者都有 | 分别量化，不得只报一个 |

**必须同时处理的三件事**（Issue 明确要求）：
1. **热态**：A 先跑与 B 先跑各占一半，检验顺序效应；若显著，先做热平衡再测。
2. **顺序效应**：交替 A/B/A/B，不用 A×N 再 B×N（否则第一组吃冷缓存、第二组吃热缓存）。
3. **重复配对数据**：N ≥ 40 对，用配对 bootstrap（10 000 次重采样）报中位比与 95% CI，不用均值差。

**来源**：Issue #4 P0-3；配对 bootstrap 方法沿用项目既有 `scripts/stage17_stats.py` 的口径。

---

## E-4 GB / GiB 混用，且 +12% 代理开销套错基线（P1-4）

**旧判定**（R2 报告 §5.3、§3.2）：
> "27B @ 4-bit = 15.2 GB，+12% = 17.0 GB，撞 17.76 GiB 工作集上限。"

**为什么错（两处）**：

**(a) 单位混用。** `17.0 GB`（十进制）= `17.0e9 / 2^30` = **15.83 GiB**，而上限是 `17.76 GiB`。15.83 < 17.76，**没有"撞上限"**。原文把 `GB` 和 `GiB` 当成同一单位直接比大小。

重算（27B，8k 上下文，GQA 64 层 × 8 KV head × 128 dim）：

| 项 | GB | GiB |
|---|---|---|
| 权重 @ 4.5 bpw（MLX affine 4b g64） | 15.19 | 14.14 |
| KV @ 8k | 2.15 | 2.00 |
| **合计** | **17.34** | **16.14** |
| 上限 `max_recommended_working_set_size` = 19,069,665,280 B | 19.07 | **17.76** |
| **余量** | 1.73 | **1.62（9%）** |

结论应改为：**权重+KV 合计 16.14 GiB，余量 1.62 GiB（9%），很紧但未超限**；真正会撞上限的是更长上下文或更大的运行时缓冲。

**(b) +12% 套错基线。** Prox 论文的 "约 12% 额外权重存储" 是**相对 FP16 模型**说的。推导：
- 代理只覆盖 up/gate = FFN 参数的 2/3，INT4 存储 = 0.5 B/参数
- 代理字节 = `0.667 × f × N × 0.5` = `0.333 f N`（f = FFN 参数占比）
- 相对 FP16（2N）：`0.333f/2 = 0.167f`，f = 0.75 时 = **12.5%** ✓ 与论文一致

**但对 4-bit 基座**，分母变成 ~0.56N（4.5 bpw 含 metadata）：
```
0.333 × 0.75 × N / 0.56N = 25% / 56% = 44.6%
```
即 **代理开销从 12% 变成约 +45%**，不是 +12%。

**修正后的结论**（比原来更强，但理由不同）：
- Prox 的 Stage 2 需要 **FP16 原权重**才能精确计算。27B 的 FP16 权重是 54 GB，**在 24 GB 上根本装不下**，所以 Prox 原样在 24 GB 上跑不了 27B——这与代理开销无关，是 Stage 2 的硬需求。
- 若未来做"4-bit + 稀疏"的联合 kernel，代理仍需常驻：27B @ 4.5 bpw = 15.19 GB，代理 = 0.333 × 0.75 × 27e9 = **6.75 GB**，合计 21.9 GB（20.4 GiB）→ **确实超过 17.76 GiB**，但超的是"联合方案"，不是"4-bit + 12%"。

**来源**：Issue #4 P1-4；Prox 论文 §5 存储开销表述（E3）；单位换算与代理字节推导为本轮计算（E2，可手算复核）。

---

## E-5 7.07 ms 与 162.9 tok/s 来自不同轮次（P1-5）

**旧判定**（R1 报告摘要、R2 报告 §0）：
> "7.07 ms / 162.9 tok/s"

**为什么错**：`1 / 0.00707 s = 141.4 tok/s`，不是 162.9。162.9 对应 `1/162.9 = 6.138 ms`。两个数来自**不同轮次**：

| 来源 | 变体 | 时间 | 派生 tok/s | 派生带宽（880.8 MB） |
|---|---|---|---|---|
| `results/stage2_0/decode/decode.json` | `serial-lazy` | **6.141 ms** | **162.9** | 143.4 GB/s |
| `results/stage2_1/bench_a67/bench.json` | `serial-lazy` | **7.074 ms** | **141.4** | 124.5 GB/s |

R1 把 stage2_0 的 tok/s 和 stage2_1 的毫秒数写进同一句话，**违反了自己在 audit C-07 里定的"跨轮不可混用"规则**。

**正确判定**：
- 必须成对引用：`6.141 ms / 162.9 tok/s / 143.4 GB/s`（stage2_0）或 `7.074 ms / 141.4 tok/s / 124.5 GB/s`（stage2_1）。
- R2 全篇用的 124.6 GB/s 对应的是 **stage2_1 的 7.074 ms**，这个配对是对的；但摘要里同时出现 162.9 就是错的。

**关于 head 的可加性**：
- "880.8 + 311.2 = 1192 MB → 9.57 ms" 是**假设带宽线性、head 不享受任何优化**的下界估计（E2），**不能称为"实际完整一步耗时"**。它既没测，也没考虑 head GEMV 的 kernel 效率与 block GEMV 不同（head 是 151936×1024 的长行，形状特性不一样）。
- 正确写法："若 head 与 block 共享同一有效带宽，完整一步的下界估计为 9.6 ms（E2）；实际值未测。"

**来源**：Issue #4 P1-5；`results/stage2_0/decode/decode.json`、`results/stage2_1/bench_a67/bench.json`（E1）。

---

## E-6 M2 Max 819.2 GB/s 与官方规格矛盾（P1-6）

**旧判定**：报告链中出现 "M2 Max 819.2 GB/s"。

**为什么错**：Apple 官方新闻稿（2023-01-17）明确写：

| 芯片 | 官方统一内存带宽 |
|---|---|
| M2 | 100 GB/s |
| M2 Pro | 200 GB/s |
| M2 Max | **400 GB/s** |
| M2 Ultra（同代，另一篇） | 800 GB/s |

`819.2 = 800 × 1.024`，是把 **M2 Ultra 的 800 GB/s** 做了一次错误的单位换算（把 GB 当 GiB 乘 1.024），然后挂到了 M2 Max 名下。两个错误叠加。

**正确判定**：
- **不得**用 819.2 GB/s 作为 M2 Max 的带宽，也不得据此外推任何 Apple 芯片。
- M4 Pro 官方规格是 **273 GB/s**（Apple 公布值）。这是**规格值**，不是可达值；任何以它为分母的"利用率"必须标注 "vs spec"（audit C-12 已定此规则）。
- 若要比较 SpQt 的 M2 系数字与 M4 Pro，只能比较**加速比**，不能比较绝对带宽或绝对 tok/s——因为 SpQt 的加速比是"同芯片 Q4_K dense → 50% 稀疏"的比值，与带宽规格无关。

**SpQt 数字的引用纪律**：
- 只引用其**加速比**（M2 Max 1.47×、M2 1.39×、M2 Pro 1.33×、A19 Pro 1.55×）与**精度表**（PPL 5.28→5.65@50%→7.39@65%）；
- 不引用其任何绝对 GB/s 或 tok/s；
- 明确标注"来自 M2 家族，非 M4，不可直接搬"。

**来源**：Issue #4 P1-6 给出的 Apple 新闻稿链接（本轮已核，400 GB/s 原文为 "Its 400GB/s of unified memory bandwidth is twice that of M2 Pro, 4x that of M2"）。

---

## E-7 2.77× 是权重字节比，不是端到端加速预期（P1-7）

**旧判定**（R2 报告 §0、§3.2、kernel-matrix §4.2）：
> "2.31 bpw + 50% 稀疏 = 2.77×"

**为什么错**：这个数只覆盖 **FFN + attention 的 GEMV 权重字节**，且只在**带宽受限区间**成立。把它写成"端到端加速预期"漏掉了至少七项。

**正确的分解式**：

```
T_end_to_end = T_weights_read + T_head + T_attn_state + T_selector + T_index + T_norm_rope_sample + T_launch
```

| 项 | 在 2.31bpw+50%稀疏 下是否缩小 | 说明 |
|---|---|---|
| 权重读取 | ✅ 是（这就是 2.77× 的来源） | 但只对**被稀疏化的 GEMV 矩阵**成立 |
| **LM head** | ❌ 否 | 151936×hidden，不稀疏、通常不压到 2-bit；27B 上占比小，0.6B 上占 35% |
| attention / SSM 状态 | ❌ 否 | KV 读取与稀疏无关；混合架构的递归状态更无关 |
| **selector / proxy** | ❌ **反向增加** | Prox 要跑 2 个 INT4 代理投影；SpQt 要跑前缀和索引收集 |
| **index 收集 kernel** | ❌ 反向增加 | 每层一次，Blelloch 前缀和 + 原子计数 |
| 布局开销 | ❌ 反向增加 | 列分组破坏行主序，superblock 跨列 → 访存模式变差 |
| GPU cache 效应 | ⚠️ 不确定 | 被跳过的 superblock 可能已被预取；SLC 命中率变化方向不明 |
| 质量损失 | ❌ 是成本 | 50% 稀疏 PPL +7%；70% 稀疏 +40% |

**更根本的一条**：**字节比只是"带宽受限区间内"的加速上界，不是全程硬上界。** 当 bytes/token 压到很低时，kernel 从带宽受限转为 **latency / instruction issue 受限**，实际加速会**低于**字节比，而且越低越多。所以：
- 字节比 = 1.84× 时，实测可能是 1.3×（SpQt 的 1.33–1.55× 就是这个位置）；
- 字节比 = 2.77× 时，实测很可能远低于 2.77×，因为每 token 只剩 ~5.6 GB，27B 在 124.6 GB/s 下是 45 ms，此时 kernel 启动与指令开销占比显著上升。

**正确表述**：
> "2.31 bpw + 50% 稀疏把 **GEMV 权重字节**降到 IQ4_XS dense 的 1/2.77。这是**字节上界**，不是端到端加速预期。端到端还要加上 head、KV、selector/index 收集、布局与 cache 效应，且在实际可达带宽下会因转为 latency-bound 而进一步打折。"

**来源**：Issue #4 P1-7；分解式为本轮构造（E2）。

---

## E-8 "Metal 上没有任何 trellis 实现"是事实错误（P2-8）

**旧判定**（R2 报告 §0、kernel-matrix §1.4、§5 反面证据 #6/#7）：
> "trellis 型（QTIP/Proteus/CurveTQ）无 Metal 实现"；"Proteus 仓库是占位符、CurveTQ 无代码"→ 暗示 Metal trellis 不存在。

**为什么错**：`trymirai/uzu` 有可运行的 Metal trellis，且不止一种格式。详见 `2026-10-r3-metal-trellis.md`。要点：

| 项 | 事实 |
|---|---|
| 文件 | `crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/gemm_trellis.metal`（194 行）+ `trellis.rs`（147 行） |
| 仓库 | `trymirai/uzu`，MIT，2458 star，2026-10-20 仍在推 |
| 格式 | `V4_T8`（2.125 bpw）、`V2_T6`（≈3.00 bpw）、`V2_T4`（≈2.00 bpw） |
| 码本 | **hash 生成**（`state × 0xCFCCB83F + 0x584B4AA3` → 折叠 → SWAR 解 8 个 2-bit 域），**零表加载** |
| 状态 | 16-bit state，`K_STEP = 64`，可选 `restart_columns` |
| 激活 | **int8 对称量化**（`GemmAPrologueKind::Int8Symmetric`），int32 累加 |

**但结论方向不变**——而且理由强得多：

`select_trellis_plan` 硬编码 `engine: GemmEngine::Mxu`；`validate_engine` 在 `!supports_mxu` 时返回 `GemmPlanError::MxuUnavailable`；而 `supports_mxu()` 的实现注释原文：

> `/// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).`
> `/// True on M5+ (Gen18+), false on M1-M4.`

**所以：uzu 的 trellis 在 M4 上直接报错，不可用。**

**修正后的正确表述**：
> "Metal 上**有** trellis 实现（uzu，2.0–2.125 bpw，lookup-free）。但它是 `MatmulB::Trellis` 的**唯一**路径（无 GEMV 变体），且被硬门禁在 `GemmEngine::Mxu`；`supports_mxu()` 在 M1–M4 上为 false。**在 M4 Pro 上它抛 `MxuUnavailable`，不可用。**"

这个订正**强化**了 R2 的路线裁决（不做 trellis 移植），因为现在有更强的证据：**世界上最好的公开 Metal trellis 实现，主动选择要求 M5 的 MXU，而不是退回非 MMA 路径。** 这说明 trellis 解码在没有矩阵单元时确实不划算。

**来源**：`trellis.rs:176`、`selection.rs:84-86`、`device_extensions.rs` 的 `supports_mxu` 注释（E0，本轮直读源码）。

---

## E-9 GSQ-RCO 的 27B 产物是二手转述（自查新增）

**旧判定**（R2 报告 §0 判断 3、kernel-matrix §1.4）：
> "GSQ-RCO 有 27B 公开产物（`Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`，2.50 bpw，8.4 GB）"、"竞赛数学保留 82–85%"

**为什么错**：这些数字来自一篇 CSDN 评测博客（对 `Underdog-Saluki-27B-1.0` 的解读），**不是官方 model card，也不是 GSQ/RCO 论文的实验表**。R2 把它写成了 E3 甚至隐含 E1。

**正确判定**：降级为 **E3-二手**，并标注：
- 产物存在性：博客声称基于 `ISTA-DASLab` 的 `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`（2.50 bpw / 8.4 GB），以及下游 `ConwayResearch/Underdog-Saluki-27B-1.0`（7.89 GB）。**本轮未直接访问这两个仓库核对。**
- 基准数字（BFCL 88 vs 84、AIME 96.7→79.2 等）：**只有博客转述，无官方来源**，不得引用为事实。
- GSQ 仓库描述 "1–3 bit scalar, INT/GGUF-compatible" 与 RCO 论文存在性：**这两个是一手**（GitHub API + arXiv abs 页）。

**结论**："GSQ-RCO 是唯一零 kernel 工作 + 有 27B 产物的路线"这一判断**保留**（因为 GSQ 的 GGUF 兼容性与 RCO 的存在性是一手证据），但**所有具体质量数字必须撤下**，改为"质量代价未独立核实"。

**来源**：`IST-DASLab/GSQ` 仓库描述、`IST-DASLab/RCO` 仓库、arXiv 2604.18556 / 2605.00649（一手）；CSDN 博客（二手，仅记录线索）。

---

## E-10 ALU 预算表建立在未验证的带宽上（自查新增）

**旧判定**（R2 报告 §3.1、kernel-matrix §3.2）：
> "124.6 GB/s → 6.7 ops/weight；190 → 4.4；273（规格）→ 3.1"，并据此判定"2.3 bpw 装得进 ALU 预算"。

**问题**：这张表的**每一行**都以带宽为自变量，而 124.6 GB/s 本身是派生值（880.8 MB ÷ 7.074 ms），且 7.074 ms 只覆盖 28 block、不含 head（audit C-10）、未用计数器验证（audit C-01）。**用一个待验证的量去推一个"够/不够"的结论，是循环论证。**

**正确判定**：把 ALU 预算表改为**条件表**，并显式声明依赖：
> "若本机 decode 的有效权重搬运速率为 X GB/s，则每权重 ALU 预算 ≈ 2.9×10¹² / (X×10⁹ / (2.3125/8)) ops。X 未实测。在 X = 124.6 的假设下预算为 6.7，IQ2_XS 的 2.5–4 ops/weight 有余量；在 X = 273（规格峰值）时预算 3.1，余量消失。"

**这条不改结论，改的是结论的强度**：从"2.3 bpw 装得进"降为"在假设带宽下装得进，需 XD-1 验证"。XD-1 的设计（见 `2026-10-r3-revised-xd1-xd3.md`）正是为了给出 X。

**来源**：audit C-01 / C-10（已定）；本轮自查。

---

## 1. 未受影响的结论（明确保留）

以下 R2 结论本轮复核后**全部维持**，列此以免读者以为全表被推翻：

| 结论 | 依据 |
|---|---|
| llama.cpp Metal 已有 15 个低位 GEMV（IQ1/IQ2/IQ3/IQ4/Q2_K/Q4_K/Q6_K/TQ/MXFP4 + `_id` MoE gather） | E0，内核名逐个核对 |
| IQ2_XS = 74 B/256 权重 = 2.3125 bpw；IQ4_XS = 136 B = 4.25 bpw；字节比 1.84× | E0（`ggml-common.h` 结构体）+ E2 |
| MLX affine 4-bit g64 = 4.5 bpw；2-bit g64 = 2.5 bpw（scale+bias 用 bf16） | E1（`mx.quantize` 输出形状）+ E2 |
| MLX `mxfp4` 必须 g32、`nvfp4` 必须 g16 | E1（`ValueError` 原文） |
| M4 有 `simdgroup_matrix`（`mul_mv_mma.metal` 用 `simdgroup_float8x8`） | E0 |
| `ArraysCache.is_trimmable() == False` → 混合架构在 mlx-lm 上无法投机解码 | E0 + E1 |
| mlx-lm 的 Gated DeltaNet prefill 是 T 串行 | E0 |
| mlx-lm 五处特性互斥（含行号） | E0 |
| 深度并行 / DAG 在 M=1 无收益（两轮一致） | E1 |
| 临界稀疏率 ≈ 0.55–0.56（4-bit 基座赢 2.3125 bpw 所需） | E2（推导，不依赖带宽） |
| 跨轮绝对时间一律不可引用 | 纪律 |

---

## 2. 一句话

R2 的**机制判断**基本站得住；**数字包装**有 8 处必须改，其中 **E-1（公式倒置）与 E-2（对象不对应）使 XD-1 的原判据完全失效**，**E-8（Metal trellis 存在性）是事实错误**，但 E-8 的订正**强化**了"不做 trellis 移植"的裁决。
