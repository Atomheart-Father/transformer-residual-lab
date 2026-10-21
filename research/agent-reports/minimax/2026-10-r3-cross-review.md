# R3 交叉复审（MiniMax）—— Metal Trellis 新实现与量化/稀疏反证复审

**任务**：R3-M / [Issue #3](https://github.com/Atomheart-Father/transformer-residual-lab/issues/3)
**研究者**：MiniMax · **日期**：2026-10-21
**上游交付物**：[`2026-10-r2-report.md`](./2026-10-r2-report.md) / [`2026-10-r2-evidence.md`](./2026-10-r2-evidence.md) / [`2026-10-r2-combinations.md`](./2026-10-r2-combinations.md)
**本轮勘误**：[`2026-10-r2-errata.md`](./2026-10-r2-errata.md)（**不覆盖** R2 原报告，保留审计轨迹）
**一手底层材料**：[`fulltext-notes/R3-U-uzu-trellis.md`](./fulltext-notes/R3-U-uzu-trellis.md)（uzu 源码，我本人逐文件读）、[`fulltext-notes/R3-S-spqt-bandwidth-prox.md`](./fulltext-notes/R3-S-spqt-bandwidth-prox.md)（SpQt/Prox 数值复核）

**证据等级**：`[P]` 一手已打开 · `[O]` 官方产物直读 · `[S]` 二手 · `[推算]` 本轮算术 · `[未核实]` 未打开

**纪律**：本轮**只做源码与文献审查**。未下载模型、未训练、未运行 GPU/CPU 重负载、未编译、未修改 `scripts/` `tests/` `results/` `docs/` 或任何他人目录。

---

## 0. 结论摘要（先给判定）

| # | Issue #3 的质疑 | 本轮判定 |
|---|---|---|
| 1 | "整个 2-bit Trellis 在 Apple Silicon 上没有实现"是否过度绝对？ | **措辞确实过强，须更正**（§2）。但**方向性结论在 M4 Pro 上存活**，且理由比原来更硬 |
| 2 | uzu 的 Metal Trellis 处于"源码存在 / 可编译 / 产品路径 / 完整实测"的哪一级？ | **只到"源码存在"**（§2）。可编译**未核实**；产品路径**否**；完整实测**否** |
| 3 | "与动态稀疏/推测解码硬冲突"是否过强？ | **过强，是错的**（§3.1）。正确表述是**性能代价**，不是正确性冲突 |
| 4 | "旋转必然应被曲率加权替代"是否过强？ | **过强**（§3.2）。"互斥"只在 CurveTQ-vs-旋转这一对具体方法之间成立；**加性 gauge × 低位宽已有联合测量（OSAQ）** |
| 5 | SpQt 的 819.2 GB/s 对 819.2/苹果 400 的质疑 | **论文错，不是转录错**（§4）。但 **M4 Pro 外推 1.2–1.3× 存活**（重算 1.22×） |
| 6 | Prox 的 +12 % 能否直接套到 27B Q4？ | **不能，且方向相反**（§5）。分母是 **FP16**；到 Q4 变成 **≈36–41 %**，R2 低估约 **3×** |
| 7 | 还剩几条值得研究的数学候选？ | **只保留 2 条**（§6），并各附**最强反证** |

> **一句话**：GPT 找到的 uzu 确实推翻了我 R2 的一句**措辞**（"Metal 上没有 trellis 内核"），**没有推翻我 R2 的任何一条判断**；而我 R2 自己在 **Prox 的 +12 % 套用**和**兼容矩阵两处 ✘** 上确实过度自信，本轮一并更正（见 errata）。

---

## 1. 本轮实际打开的东西

| 来源 | 打开方式 | 结果 |
|---|---|---|
| `trymirai/uzu` @ `2968097a` 的 `gemm_trellis.metal` / `trellis.rs` / `quantized/trellis.h` / `selection.rs` / `gemm/kernel.rs` / `gemv/kernel.rs` / `matmul_metal_kernel.rs` / `device_extensions.rs` / `context.rs` / `quantization_method.rs` / `matmul_b.rs` / `simdgroup_multiply_accumulate.h` / `trellis_gemm_test.rs` / `BENCHMARKS.md` / `README.md` | `raw.githubusercontent.com` 直取字节 + GitHub 代码搜索 | ✅ 全部读到 |
| `trymirai.com/local-models`（271,718 B） | HTTP GET | ✅ 全文检索 |
| Apple 官方：M2/M2 Pro/M2 Max newsroom、`support.apple.com/en-us/121553`（MacBook Pro 2024 技术规格） | HTTP GET | ✅ 读到原文 |
| SpQt `2511.04477`：arXiv HTML + PDF（`pdftotext -layout`）+ 直接拉取两张 SVG 表 | HTTP GET | ✅ 表 1/表 2 逐格读出 |
| Prox `2607.27591`：arXiv HTML + PDF | HTTP GET | ✅ §4.3 / §5.2 / §5.3 / A.3 逐节读出 |
| `Qwen3-8B` / `Qwen3.6-27B` 的 HuggingFace `config.json` | HTTP GET | ✅ 用来做 §5.3 的字节算术 |

---

## 2. 焦点一：uzu 的 Metal Trellis —— 四级判定

### 2.1 判定表

| 级别 | 判定 | 一手依据 |
|---|---|---|
| **① 源码存在** | ✅ **是** | 完整 Metal kernel（`KERNEL(GemmTrellis)` + `KERNEL(GemmTrellisReduce)`）+ Rust dispatch + 单元测试。**MIT 许可**（与 QTIP 的 GPL-3.0 相反） |
| **② 可编译** | ⚠️ **未核实** | 仓库有 CI 与 `uzu_test` 驱动的测试；**但我不得编译**，且 **M4 Pro 上该 kernel 是否能跑（`simdgroup_matrix<int8_t>` 的 family 门槛）未核** `[未核实]` |
| **③ 正式产品路径** | ❌ **否** | `QuantizationMethod` 只有 `ScaleBias` / `ScaleZeroPoint` / `ScaleSymmetric` **[O]**；仓库树 `gguf` 路径 **0 命中**（模型格式不是 GGUF）；`trymirai.com/local-models` 全文 **"trellis" 零命中**，量化项只有 **GGUF K-Quants**；README（46,713 B）**trellis/bpw/quant/bit 零命中** |
| **④ 完整实测** | ❌ **否** | 唯一测试用**随机字节**当 codes（`rng.random()`），只校验算术与 CPU 参考在 `1e-2` 相对误差内一致。**无 PPL、无 tok/s、无任何 Apple 芯片实测、无真实模型**。`BENCHMARKS.md` 的基准组列表里**没有 Trellis** |

### 2.2 它编码的东西：**与 QTIP 系完全不同的码**

`quantized/trellis.h` 的常量与函数（逐行可读）：

```c
UZU_CONST uint K_STEP = 64, WEIGHT_BIAS = 54;
UZU_CONST uint STATE_BITS = 16, V4_VECTOR_WIDTH = 4, V2T6_TRANSITION_BITS = 6;
UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;
```

**机制三步**：

1. 每行按 `vector_width` 切"状态"，状态 **16 bit**；相邻状态用 `transition_bits` 位的**转移**连接 ⇒ 整行是一条马尔可夫链（"trellis" 之名由此而来），行首存 16 bit 初始状态，其余靠转移重建。
2. `state → hash → 4 字节 → 每字节 4 个 2-bit 域 → 有偏权重`。注释给出语义：*"Per byte b: weight + WEIGHT_BIAS = 8 * (sum of b's four 2-bit fields) + ((3 * (b & 15)) & 15)."* ⇒ **一个 hash 字节产生一个 4-bit 权重**。
3. 真实权重 = **共享低秩基 + level 项**。`gemm_trellis.metal` 的 `TrellisEpilogue::store`：
   ```c
   const float4 group_sum = group_sums[token];
   const int level_dot = biased_dot - int(trellis::WEIGHT_BIAS) * int(group_sum.x + group_sum.y + group_sum.z + group_sum.w);
   const float4 offsets = float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
   const float dot = float(level_dot) * scale_and_offsets[0] + metal::dot(group_sum, offsets);
   output[...] = bfloat(dot * row_scales[row] * activation_scale);
   ```
   `scale_and_offsets` 就是 `codebook`（Rust 侧 `MatmulB::Trellis { codes, row_scales, codebook, format }`），**共 5 个 float**（1 scale + 4 基向量），全权重矩阵共用；测试里写死为 `const CODEBOOK: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];`。

**码率 `[推算]`**（按 `block_bytes = (STATE_BITS + transition_count * transition_bits + 7) / 8`）：

| 格式（测试中实际出现） | vector_width | transition_bits | restart_columns | 码率 |
|---|---|---|---|---|
| `V4_T8` | 4 | 8 | 64 | **2.125 bpw**（每 64 列 17 字节） |
| `V2_T6` | 2 | 6 | 无 | **≈3.0 bpw** |
| `V2_T4` | 2 | 4 | 无 | **≈2.0 bpw** |

另有每输出行 1 个 f32 `row_scales`、每权重矩阵 5 个 f32 `codebook`（均可忽略）。

### 2.3 ★ 它恰好绕开我 R2 指认的"内循环墙"——**但这只是结构论证**

R2 §1 的核心判断：低比特解码器的**查表/码本访问**会变成内循环瓶颈（escha-port 实测：查表产生 **13× 于权重的 L2 流量**）。

uzu 的解码**没有任何码本查表**：
- 码本只有 **5 个 float**，全矩阵共用；
- 状态→权重是**纯 ALU**：`state * MULT + INC` → `hash ^= hash>>16` → `hash *= AVALANCHE` → `hash ^= hash>>16` ⇒ 约 6–8 条整数指令出 4 个权重，再加 `biased_weights` 约 8 条 ⇒ **约 15–20 条 ALU 指令 / 4 个权重**；
- ⇒ **这是与 Proteus / CurveTQ / XOR-Trellis 不同的答案**：不是"把查表做快"，而是"**把查表变成一个恒定的东西**"。

**但**：零测量 ⇒ **不能据此宣称它解决了内循环问题**。

### 2.4 M=1 Decode：可行，但只有 GEMM 一条路

| 事实 | 依据 |
|---|---|
| **M=1 被测试覆盖** | `trellis_gemm_test.rs`：`const CASES: [(usize,usize,usize);7] = [(1,80,5120), (17,80,5120), (512,80,5120), (16,128,64), (32,128,64), (1,6,64), (2048,2048,128)];` ⇒ 两个 M=1 用例都在 |
| **M 越界被正确处理** | `gemm_trellis.metal`：`const short token_end = short(min(int(Core::SIMDGROUP_BLOCK_M), int(params->M) - int(token_base)));` + `GemmAlignment::M` 模板分派（对齐/非对齐两条路径） |
| **★ 没有 GEMV 路径** | `gemv/kernel.rs`：`MatmulB::Trellis { .. } => { return Err(MatmulMetalError::UnsupportedLayout { path: "Gemv" }); }` ⇒ **decode 必须走 M=1 的 GEMM** |
| **最小 tile 16×32×256 / 1 simdgroup** | `VARIANTS(..., Tile16x32x256_Simdgroups1x1, Tile64x64x256_Simdgroups2x2, Tile128x128x256_Simdgroups4x4)` ⇒ M=1 时 **M 维浪费 15/16 计算槽位**（对带宽受限的 decode 可能无害，**未测量**） |
| **K 必须被 64 整除** | `selection.rs`：`if self.shape.b_is_trellis && !self.shape.k.is_multiple_of(TRELLIS_K_STEP) { return Err(UnsupportedLayout("Trellis K must be divisible by 64")); }` |
| **输出必须 BF16；B 必须转置且连续** | `gemm/kernel.rs`：`output_data_type != BF16` ⇒ Err；`trellis.rs`：`!b_transpose` / `b_leading_dimension.is_some()` / `output_stride < n` ⇒ Err |

### 2.5 ★★ 本轮最决定性的发现：**在 M4 Pro 上它会被引擎自己的门拦下**

`device_extensions.rs`：

```rust
/// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
/// True on M5+ (Gen18+), false on M1-M4.
fn supports_mxu(&self) -> bool {
    optional_selector_value(self, obfstr!("supportsMXU"), false)
}
```

`optional_selector_value` 的语义（同文件）：`if device.respondsToSelector(selector) { raw_msg_send(...) } else { fallback }`，**fallback = `false`**。`context.rs:130` 把它装配进 `GemmProblem`。

`selection.rs`：

```rust
pub fn select_plan(self) -> GemmPlan {
    if self.shape.b_is_trellis {
        return select_trellis_plan(self.shape.m, self.shape.n, self.shape.k);   // ← 提前 return
    }
    let engine = if self.supports_mxu && mxu_is_eligible(self.shape) { GemmEngine::Mxu } else { GemmEngine::Simdgroup };
    ...
}

pub(super) fn validate_engine(&self, engine: GemmEngine) -> Result<(), GemmPlanError> {
    if engine == GemmEngine::Mxu && !self.supports_mxu {
        return Err(GemmPlanError::MxuUnavailable);      // ← 这里拦
    }
```

`select_trellis_plan` **无条件**返回 `engine: GemmEngine::Mxu`；而 `matmul_metal_kernel.rs:97-99` 把 Trellis 分派到 `MatmulDispatch::Gemm(problem.select_plan())`；**生产编码路径** `gemm/kernel.rs:198` 又调用：

```rust
self.problem(shape, command_buffer.context()).validate_engine(plan.engine)?;
```

### 2.6 ⇒ 推论（及其必须同时标注的限制）

> **在 M4 Pro 上，uzu 的 Metal Trellis GEMM 会以 `MxuUnavailable`（"MXU engine is not available for this GEMM"）失败。**
> 链条：`supportsMXU` 私有选择器在 M1–M4 上不存在 ⇒ `respondsToSelector` 为假 ⇒ fallback `false` ⇒ `validate_engine` 判定 `engine == Mxu && !supports_mxu` ⇒ **Err**。

**必须同时标注的两点（否则就是新的过度绝对化）**：

1. **这是"引擎策略门"，不是已证实的"硬件跑不了"。** 内核用的是 `metal::simdgroup_matrix`（Metal 3，family 8+ 即可），见 `simdgroup_multiply_accumulate.h`：
   ```c
   typedef metal::simdgroup_matrix<AccumulatorT, ROWS, COLS> AccumulatorMatrixType;
   ```
   ⇒ **该指令在 M4 Pro 上很可能可用；是 uzu 的策略选择了不用它。**"策略门"与"硬件门"必须分开陈述。
2. **`metal::simdgroup_matrix<int8_t,8,8>`（8-bit 整型矩阵类型）的 family 门槛我未核实** `[未核实]`（本环境 `web_search` 失败、Apple 文档未打开）。**这是本轮最该由下一位补上的一格**：若 int8 矩阵类型本身要求 family 9+/10+，则 M4 Pro 上"策略门"其实是对的、"硬件门"也成立。

### 2.7 重写后的 Trellis 投入性价比结论

**R2 原文**（§8.1 结论段）：*"2-bit 的全部 2026 年创新都在'更好的编码 + 更快的解码器'，而这些解码器目前只在 CUDA 侧存在。…… 在 24 GB 设备上，2-bit 目前不是一条可走的路；4-bit（且格式选对）才是。"*

**本轮修订后的版本**：

| 维度 | R2 说法 | **R3 修订** |
|---|---|---|
| "Metal 上没有 trellis 内核" | 绝对 | **错**。uzu 有公开的 Metal Trellis GEMM（MIT）——**但它是另一种码**（哈希合成 + 共享秩-4 基，无查表），**不是** QTIP/Proteus/CurveTQ 那族 |
| "2-bit 在消费级不成立" | 成立 | **仍成立**，但理由要换：不是"没人实现"，而是 ① **uzu 的实现不是产品路径**（无模型、无编码器、无格式支持）；② **未实测**（连质量都是未知）；③ **在 M4 Pro 上被 `MxuUnavailable` 拦下**；④ **QTIP 族的 Metal 内核仍然不存在** |
| "2-bit 值得投入吗" | 不值得 | **仍不值得**，但理由更精确：**这一族的瓶颈已被重新定位为"解码内循环"**，而 uzu 恰好提供了一个**结构性答案**（把查表变成 ALU）——**这是本轮唯一的正面线索，值得跟踪，但不值得现在投入**（因为没有可测的东西） |

### 2.7bis ★ 与 WorkBuddy 独立审计的交叉确认（两条独立证据链收敛）

WorkBuddy 在 `aaa4e10` 提交了 `research/agent-reports/workbuddy/2026-10-r3-metal-trellis.md`，对**同一份 uzu 源码**做了独立审计。**三条关键事实完全一致**：

| 事实 | 本报告 | WorkBuddy | 收敛 |
|---|---|---|---|
| uzu 的三种格式与码率 | `V4_T8` 2.125 bpw / `V2_T6` ≈3.0 / `V2_T4` ≈2.0 | 同 | ✅ |
| M4 Pro 上被 `MxuUnavailable` 拦下 | ✅ §2.5–§2.6 | ✅ 同结论 | ✅ |
| **没有 trellis 版 GEMV** | ✅ `gemv/kernel.rs` 返回 `UnsupportedLayout` | ✅ 同 | ✅ |

WorkBuddy 的结论先行句与我的一致：*"Metal 上确实有可运行的 trellis（2.00–3.00 bpw，lookup-free），但它被硬门禁在 M5 的 MXU 上，在 M4 Pro 上直接报错不可用。"*

**他们补了两个我漏掉、且对本项目更关键的东西**：

**(1) 字节账：uzu 的 2 bpw 相对 llama.cpp 已有的 IQ2_XS 只省 8.8 %**

| 方案 | bpw | 27B 权重 | 相对 IQ4_XS | 相对 IQ2_XS | M4 Pro 可运行 |
|---|---|---|---|---|---|
| IQ4_XS | 4.25 | 14.34 GB | 1.00× | 0.544× | ✅ |
| **IQ2_XS** | **2.3125** | **7.81 GB** | 1.84× | 1.00× | ✅（llama.cpp `kernel_mul_mv_iq2_xs_f32`） |
| **uzu V4_T8** | 2.125 | 7.17 GB | 2.00× | **1.088×** | ❌（`MxuUnavailable`） |

⇒ **这把 §2.7 的结论从"没有实现"升级为"就算实现了也不划算"**：可用方案（IQ2_XS，M4 Pro 上今天就能跑）比不可用方案（uzu trellis，M5+ 独占、无模型、无质量数据）**只多占 8.8 % 字节**。**与 R2 对 QTIP 算出的 1.16× 是同一个算术必然**（2.0 vs 2.3125）。

**(2) 一个我读到但没讲透的机制：符号位不是免费的**

他们指出 uzu 的 level 解码出来是**非负偏置值**（`weight + 54`），还原有符号权重**不逐权重存符号位**，而是 ① int32 累加后减去 `54 × Σ(group 内激活和)`、② 再加 `dot(group_sum, offsets)` 补偿 4 个列组各自的偏置。代价是要预先算好 `column_group_sums`（由 `activation_transform/trellis.metal` 产出）。
⇒ **收益是省掉每权重 1 bit 的符号位——在 2 bpw 下那 1 bit 是 50 % 的额外开销，这是它能做到 2.0 bpw 的关键之一。**
⇒ **这是一条可迁移的数学洞见**："符号位不是免费的，聪明的代数修正可以把它换成分组统计量。"它独立于 uzu 是否可用。

**(3) 并且他们的 R2 审计 C-04 部分收窄了我 §2.6 的开放问题**

WorkBuddy 的 `2026-10-r2-audit-corrections.md` C-04 一手核过：**`simdgroup_matrix`（`simdgroup_float8x8` / `simdgroup_half8x8`）是 Apple9 家族（含 M4）就有的 MMA 指令**，而 NAX（Metal 4 `MTLTensor` / `matmul2d`）是 M5/A19 起的一代不同的东西。
⇒ 我 §2.6 的开放问题因此**从"该指令族在 M4 上是否存在"收窄为**：
> **"`metal::simdgroup_matrix<int8_t,8,8>` 这一 int8 变体在 Apple9 上是否可用"** —— 已被证实的只有 **fp8 与 half** 变体（llama.cpp `mul_mv_mma.metal` 7 处 `simdgroup_float8x8`、MLX `quantized_nax.h` 族），**int8 矩阵乘仍未核**。
⇒ **这不改变"M4 Pro 上 uzu 不可用"的结论**（那是 `supports_mxu` 的**策略门**，与指令可用性无关），但它决定了"**能不能给 uzu 提个 PR 把门打开**"这个问题是否值得问——**在 int8 变体未核之前，不该改那个门控**。

> **⇒ 给项目的一句话**：**不要因为 uzu 而改变路线；但要把 uzu 当作"内循环墙的一种已被提出的解法"记下来，并在未来评估任何 2-bit 方案时，要求它同时报告"解码内循环的访存量"与"是否使用查表"。**

---

## 3. 焦点二：R2 兼容矩阵中过强的 ❌ —— 逐条重审

Issue #3 要求：**"将数学不可兼容与没有联合测量明确区分"**。这正是 R2 混淆的地方。

### 3.1 (a) 逐 token 动态掩码 × 分布保持的推测解码

**R2 原判**：组合矩阵 `Prox × MTP 推测解码 = ✘`，理由 *"逐 token 动态掩码破坏推测解码"*。

**本轮判定：❌ 判定错误，应为 △。**

**数学事实**：推测采样（Leviathan et al. 2023）/ 推测解码（Chen et al. 2023）的接受规则

```
α(x) = min(1, p_target(x) / q(x))
```

+ 残差修正，**对任意草稿分布 q 都成立**，只要 **p_target 能被精确求值**。⇒

- **草稿侧**逐 token 动态掩码 **不破坏分布保持**：它只让 q 离 p 更远 ⇒ **接受率下降**（性能问题），**不是正确性问题**。
- **目标侧**若用掩码，则 p_target 不再是原模型的分布——但那是"**部署的模型变了**"，是稀疏化本身的性质，**与推测解码无关**。

**关键澄清（这是 R2 漏掉的一环）**：R1 证明的 MTP 是"**分布保持**"——它保持的是"**部署的那个模型**"的分布。稀疏 FFN（Prox）**本来就改变了模型**（R2 §3.7 已记录：70 % 稀疏时 Qwen3-4B 掉 12.4 分）⇒ **不存在"稀疏又额外破坏推测解码保证"这回事**，除非把稀疏当成"原模型"。

**真正的代价（三条，都要写进矩阵）**：

1. **接受率下降**：草稿与目标分叉 ⇒ α 变小 ⇒ 每次验证摊到的收益变薄。
2. **代理成本被草稿 token 数放大**：Prox 的代理占 FFN 时间 **19.9 %** `[P]`；推测解码里**每个草稿 token 都要算一次代理**，而稀疏的收益只在草稿侧兑现 ⇒ **净收益 = 稀疏收益 − 草稿数 × 代理成本**。
3. **若目标是稀疏模型**：验证每个 token 都要精确算 FFN ⇒ **稀疏收益归零**（只剩草稿侧的）。

**⇒ 正确表述**：**△（不是 ✘）。分布保持不构成冲突；冲突在于"稀疏目标 + 精确验证"二者只能选一个，以及接受率与代理成本的复利。**

**附带的自洽性检查**：R2 矩阵里 `Prox × 稀疏注意力 = ✔✔（1.92×，联合测量）`。若 "动态掩码 × 推测解码" 真如 R2 所写是 ✘，那么"动态掩码 × 另一个注意力稀疏方法"更不该是 ✔✔。**R2 自己的矩阵内部就不自洽。**

### 3.2 (b) RoPE 下 gauge/旋转 × 低位宽

**R2 原判**：矩阵格 `△ 互斥冗余`，但 §2.3 总判定写 **"✘ 互斥的替代关系。同时做两个 = 付两次代价买一次收益。"**

**本轮判定：§2.3 的"✘"过强；应为 △（旋转子族）/ ✔（加性子族）。**

**逐层拆开**：

| 层次 | R2 说法 | **本轮判定** |
|---|---|---|
| **数学上** | "互斥" | **错**。旋转 ∘ 量化是完全良定义的算子复合；QuaRot / SpinQuant / QTIP / Proteus **全部**就是"先旋转再量化"。而且 **OSAQ 本身就是"gauge 族变换 × 量化"的联合测量**（加性变换、零推理开销）⇒ **"gauge × 低位宽是 ✘"作为一个一般命题为假** |
| **RoPE 约束** | "与 RoPE 交换的正交映射只有逐频率对旋转" | **保留** `[P]`。这是 RoPE-Aligned 的定理，成立 |
| **该族对量化不利** | "W4A4KV4 下比整头 Hadamard 更差" | **保留** `[P]`，但这是**一个正交族在一个量化器下的经验结论**，**不是"旋转 ⟂ 低位宽"的定理** |
| **"互斥"的真实来源** | "CurveTQ：weighting in the native basis and rotating are substitutes" | **这是方法 vs 方法的比较**，不是定理。成立范围仅限"CurveTQ 的曲率加权搜索 vs 某个具体旋转" |
| **"付两次代价买一次收益"** | 作为总判定 | **只在曲率加权搜索确实完全覆盖旋转收益时成立**——而这本身是经验命题，CurveTQ 的数据只支持"在其解码器下"成立 |

**⇒ 正确表述**：
- **加性 gauge × 低位宽（OSAQ）**：**✔ 已有联合测量**，且零运行时开销、可离线吸收 ⇒ **这是这一族里唯一已被证实的组合**，R2 §10 判断 5 押它是对的，但**矩阵里应标 ✔ 而不是 △**。
- **正交旋转 × 低位宽（RoPE 兼容族）**：**△** —— 数学上可组合、**没有联合测量**、唯一相关经验结果（RoPE-Aligned）**不利**。**不是 ✘。**
- **旋转 × 低位宽的一般命题**：**△** —— "付两次代价"是**经济性论证**，其前提（曲率加权搜索完全替代旋转）未普遍验证。

### 3.3 (c) SpQt zigzag × 低 bpw 重新编码：只是缺联合 kernel 吗？

**R2 原判**：矩阵格 `极低位宽（2-bit trellis） × SpQt = ✘（SpQt 的 layout 需重新量化）`，而 R2 §3.1 同一对却给 **"✔✔ 消费级唯一真正命中带宽轴的一条"**（Q4_K 上完整实现并实测）。**⇒ R2 内部直接自相矛盾。**

**本轮判定：✘ 错；应为 △，且真正的答案比"缺个 kernel"更有意思。**

**先厘清两件事在不同的轴上**：

- **zigzag 是"布局"**（字节怎么排）；**trellis 是"编码"**（每个权重几个比特、状态怎么推）。**两者不作用在同一个轴 ⇒ 原则上可组合。**Q4_K 是分组量化格式，而 SpQt 的 zigzag 已在其上完整实现并实测 ⇒ **"格式无关性"已被证明过一次** `[P]`。

**但真正的相互作用不在"能不能写出来"，而在"码率"**：

1. **trellis 的状态链是"沿行有序"的**：相邻列的状态由 `transition_bits` 转移连接。**任何对列的重排都会打断这条链** ⇒ 转移熵上升 ⇒ **码率上升**。zigzag 的本质就是重排 ⇒ **二者互相破坏**。
   - 量化：uzu 的 `V4_T8` 若不重排，2.125 bpw；**重排后转移不再独立于前序，转移位不能按独立熵编码 ⇒ 码率上升** `[推算]`。
2. **稀疏在熵编码格式里省的是计算，不自动省字节**：
   - 被跳过的权重**仍然要出现在码流里**（状态链要继续），**除非**整个块被跳过；
   - **只有当格式有"重启/分块"结构时，字节跳过才可能发生**。
   - **uzu 有这个结构**：`GemmTrellisFormat { vector_width, transition_bits, restart_columns }`，`V4_T8` 用 `restart_columns = Some(64)` ⇒ **每 64 列独立重启状态链** ⇒ **整块跳过是可能的**。而 `V2_T6`（`restart_columns = None`）**整行一条链 ⇒ 只能从尾部跳**。
3. **⇒ 正确的联合设计约束是**：**稀疏模式必须定义在"码流顺序"上，而不是定义在"权重矩阵"上**；即**块级稀疏**（按 64 列的 trellis 块整块剪枝），而不是 N:M 细粒度稀疏。

**⇒ 正确表述**：**△（不是 ✘），但比"缺一个联合 kernel"多一条硬约束**：*码流顺序与稀疏模式必须共设计；朴素重排会破坏 trellis 的转移熵；字节级跳过只有在格式具备重启/分块结构时才存在。*

**这条约束的实用价值**：它把"要不要做 trellis × 稀疏"从一个工程问题变成了**一个有明确最优答案的数学问题**（块大小 = 码流块大小 = 缓存行/superblock 的对齐）——见 §6 候选 2。

### 3.4 三条重审的汇总：**数学不可兼容 vs 没有联合测量**

| 配对 | R2 原判 | **R3 判定** | 不可兼容的**真正来源** | 有无联合测量 |
|---|---|---|---|---|
| 动态掩码 × 分布保持推测解码 | ✘ | **△** | **不是数学不可兼容**。冲突在"稀疏目标 vs 精确验证"二选一 + 接受率 + 代理成本复利 | Prox × 稀疏注意力 ✔✔（1.92×）；Prox × MTP ✘ |
| RoPE gauge × 低位宽 | △（§2.3 写 ✘） | **旋转子族 △ / 加性子族 ✔** | **不是数学不可兼容**。是"经济性未被普遍验证"+ RoPE 交换定理缩小了可选族 | OSAQ × 量化 ✔（零开销）；正交旋转 × trellis ✘ |
| zigzag × 低 bpw 重编码 | ✘（与 §3.1 自相矛盾） | **△** | **不是数学不可兼容**，但有**真实的码率约束**：重排破坏转移熵；字节跳过需分块结构 | Q4_K 上 ✔✔；trellis 上 ✘ |

---

## 4. 焦点三：SpQt 四芯片数值与 "819.2 GB/s" 溯源

### 4.1 论文表 1 原文（`pdftotext -layout`，PDF p.7）

| Platforms | GPU core | GPU freq | Mem bandwidth | SoC TDP |
|---|---|---|---|---|
| `A19 pro` | `5` | `1.6 GHz` | **`75.8 GB/s`** | `~5 W` |
| `M2` | `8` | `1.4 GHz` | **`102.4 GB/s`** | `~20 W` |
| `M2 Pro` | `19` | `1.4 GHz` | **`204.8 GB/s`** | `~45 W` |
| `M2 Max` | `30` | `1.4 GHz` | **`819.2 GB/s`** | `~80 W` |

**⇒ 论文确实印着 `819.2 GB/s`。R2 是忠实转录，错在论文。**

### 4.2 苹果官方对账

| 芯片 | 论文标注 | **Apple 官方** | 比值 | 官方来源 |
|---|---|---|---|---|
| M2 | 102.4 | **100 GB/s** | 1.024 | newsroom 2023-01 |
| M2 Pro | 204.8 | **200 GB/s** | 1.024 | newsroom 2023-01 |
| **M2 Max** | **819.2** | **400 GB/s** | **2.048** | newsroom 2023-01（"400GB/s … twice that of M2 Pro, 4x that of M2"） |
| **M4 Pro** | — | **273 GB/s** | — | newsroom 2024-10 + `support.apple.com/en-us/121553`（"273GB/s memory bandwidth"） |
| M4 Max | — | 410 / 546 GB/s | — | 同上 |

### 4.3 诊断：论文自身的方法在 M2 Max 行不成立 `[推算]`

论文的三个较小值恰好等于 **"LPDDR5X 速率 × 总线宽度 ÷ 8"**（以 6400 MT/s 计）：

| 芯片 | 公式 | 结果 | 论文 | 一致？ |
|---|---|---|---|---|
| M2 | 6400 × 128 / 8 | 102.4 | 102.4 | ✅ |
| M2 Pro | 6400 × 256 / 8 | 204.8 | 204.8 | ✅ |
| M2 Max | 6400 × **512** / 8 | **409.6** | **819.2** | ❌ **差 2.000×** |
| （若用 1024-bit 总线） | 6400 × 1024 / 8 | 819.2 | 819.2 | 需 1024-bit 总线 = **M2 Ultra** 的宽度 |

⇒ **主假设**：M2 Max 行是 **×2 的算术滑移**（819.2 = 2 × 409.6）。
⇒ **次要假设**：`819.2` 与 Apple **M3 Ultra** 的 "over 800 GB/s" 量级重合，可能是**串行**。**论文未给任何推导 ⇒ 两者均为假设** `[未核实]`。
⇒ **附带结论**：819.2 GB/s 既是官方的 **2.048×**，也是 M2 Max **理论峰值 512 GB/s 的 1.6×** ⇒ **无论哪种解释，它都不可能是该芯片的实测可达带宽。**

### 4.4 ★ M4 Pro 外推重做（结论存活）

用 R2 的口径（speedup = a + b/B，四点最小二乘）**分别用论文标注带宽与官方带宽**拟合：

| 拟合所用的 B | a | b (GB/s) | 四点拟合残差 | **M4 Pro (273 GB/s) 外推** |
|---|---|---|---|---|
| 论文标注（75.8 / 102.4 / 204.8 / 819.2） | 1.1334 | 29.81 | A19 Pro 1.527 vs 1.55；M2 1.425 vs 1.40；M2 Pro 1.279 vs 1.26；M2 Max 1.170 vs 1.19 | **1.243×** |
| **Apple 官方（75.8 / 100 / 200 / 400）** | 1.0984 | 32.79 | A19 Pro 1.531 vs 1.55；M2 1.426 vs 1.40；M2 Pro 1.262 vs 1.26；M2 Max 1.180 vs 1.19 | **1.218×** |

⇒ **R2 的"外推约 1.2–1.3×"存活**；带宽标注错误只把结论从 1.22× 挪到 1.24×。**R2 §8.3 的表述需要更正的是带宽数值本身和"落在 M2 Pro 与 M2 Max 之间"这句话（应改为官方 200 与 400 之间），而不是结论。**

**这个外推成立的条件（必须写全，缺一不可）**：

1. 同一模型 **Llama-2-7B**、同一 **Unified 50 %** 稀疏档、同一 **dense Q4_K** 基线；
2. 加速比与 1/B 近似线性 —— 四点残差 ≤0.023，说明**在这个区间**该线性假设成立；
3. **M4 Pro 的 273 GB/s 落在 M2 Pro（200）与 M2 Max（400）之间 ⇒ 这是内插而非外推** ✔；
4. **计算侧不变**：M4 Pro 与 M2 代的 SIMD 宽度、矩阵单元、内存层级都不同，线性模型隐含"每芯片计算占比相同"。**这是本外推最弱的一环**；
5. SpQt 的稀疏收益是"省字节"（整 superblock 跳过），所以带宽项主导 —— 但**若 M4 Pro 的 L2/片上 SRAM 更大，实际可能更好**（未核）。

### 4.5 顺带修正的口径：SpQt 那四个加速比是"相对什么"

`Table 2` 的列头是 *"speedup over dense **Q4_K**"*（不是相对 FP16）。四个被引用的数字落在**唯一一个切片**：`Unified 50%` × **`Llama-2 7B`** 列：
`22.86 (1.55×)`、`30.69 (1.40x)`、`47.17 (1.26x)`、`68.87 (1.19x)` `[V]`。
⇒ R2 §8.3 的表 **须补上"vs dense Q4_K、Llama-2-7B、Unified 50%"** 这个口径，否则读者会误以为是模型无关的加速。

---

## 5. 焦点四：Prox 代理 "+12 %" 的基准追溯

### 5.1 原文（§5.3 Discussion，全文仅出现一次）

> *"Additionally, retaining Stage 1 **INT4** proxy weights in GPU memory introduces an **approximate 12% weight-storage overhead**. Future work could reduce this overhead through a sliding-window scheme that dynamically loads and evicts proxy weights across Transformer layers."* `[P §5.3]`

§4.3 补充：*"we construct the Stage 1 proxy weights via **symmetric per-row INT4 quantization of the original FP16 W_up and W_gate matrices**, keeping the compact proxies resident in GPU memory."* `[P §4.3]`

⇒ **代理本身是 INT4**（这点 R2 说对了）。

### 5.2 ★ 分母是 **FP16**，不是 4-bit

用 Prox 自己评测的模型配置重算：

```
proxy_bytes = L · 2 · h · inter · (b/8)     # W_up + W_gate，INT4，b=4
```

| 分母 | 代理 INT4 ÷ 该分母 | 与"approximate 12%"是否吻合 |
|---|---|---|
| **FP16 总权重** | **10.2 %–13.0 %**（均值 ≈11.9 %） | ✅ **吻合** |
| **INT4 总权重** | 41 %–52 % | ❌ 完全不像 12 % |
| 仅 Transformer block 的 FP16 权重 | 12.3 %–13.5 % | ✅ 也吻合 |

⇒ **12 % 的分母是全精度（FP16）权重存储**，与 Prox 自身的成本模型一致（§A.3 的 α 也是 `INT4 proxy GEMV ÷ FP16 GEMV`）。

### 5.3 ⇒ 不能套到 27B 的 Q4；精确上/下界

以**已核实**的 `Qwen3.6-27B` 配置（h=5120, inter=17408, L=64, 27.78 B params）：

```
proxy / layer = 64 · 2 · 5120 · 17408 · 0.5 B = 89,128,960 B = 85.00 MiB
```

| 情形 | 额外常驻字节 | 相对 Q4 权重预算（13.89 GB） |
|---|---|---|
| **上界**：所有层代理同时常驻 | 64 × 85.00 MiB = **5.704 GB** | **≈41.1 %** |
| **下界**：论文自述的滑窗方案，W=1 | **85.00 MiB** | ≈0.64 % |
| 单层 / FP16 分母下的占比 | 85.00 MiB ÷ 55.56 GB | 0.16 % |

**⇒ R2 §6.2 组合表写的"27B 4-bit ≈ 16 GB，+12 % ≈ 1.9 GB"低估了约 3×**（真实上界 ≈5.7 GB，占比 ≈36–41 % 而非 12 %）。**这是本轮对 R2 最实质的一处更正。**

### 5.4 与其它工作集占用的关系

- 代理常驻是**纯增量**，与 KV cache、激活、CUDA context **同时存在** ⇒ 在 24 GB 设备上，**5.7 GB 上界会直接吃掉"还能留给 KV cache"的全部余量**；
- 论文自述的缓解（跨层滑窗换入换出）**把上界换成了访存代价**，在 decode 的关键路径上引入**额外的权重流式搬运** —— 而这恰好是 Prox 最想省的东西（**省 FLOPs 不省字节**，R2 §3.3）⇒ **缓解措施与优化目标部分对立**；
- **且**：`+12 %` **不在 Prox 自己的"有效稀疏度"指标里**（α 是延迟比不是字节比）—— 这一点 R2 已经指出，保留。

---

## 6. 仅保留两条仍值得研究的数学候选

> Issue #3 要求：**"只保留不超过两条仍值得研究的数学候选及最强反证。不得强行制造机会或宣传'无人实现过'。"**
> 以下两条都**不是新发明**，而是从已核证据里推导出的、**尚未被联合测量**的数学问题。

### 候选 1：**把"内循环墙"从"查表"改成"ALU"——状态链哈希合成作为低比特解码的结构性解**

**数学内容**：不再用"向量量化 + 大码本"，而用"16-bit 状态链 + 哈希合成 + 共享秩-4 基"，使解码的**访存**退化为常数（5 个 float 的码本），**算术**变成 O(1) 整数运算 `[推算]`。在 2.0–2.125 bpw 码率下，解码器每权重约需 15–20 条 ALU 指令 `[推算]`。

**为什么可能有用**：这一族 2026 年的全部进展（Proteus / CurveTQ / XOR-Trellis）都在回答"怎么让 2-bit 的解码器够快"，而 uzu 给出了**第三条路**：让解码器**不再需要查表**。如果 ALU 成本能被带宽时间掩盖，**2-bit 的"内循环墙"就绕过去了**——这正是 R2 §1 那面墙。

**为什么可能失败（最强反证，按强度排序）**：
1. **★ 率失真完全未验证**：整个格式成立的前提是"真实权重序列能被 16-bit 状态链以 2.125 bpw 表达"。**uzu 从未编码过一个真实权重矩阵**（测试用随机字节），**PPL 未知**。这是一个**未被任何数据触及的前提**。
2. **★ 可运行的实现不存在于目标硬件**：唯一可读的实现被 `supports_mxu` 门挡在 M5+ 之外（§2.6），且**不是产品路径**（无模型、无编码器）。
3. **ALU 成本未被核算**：在 M4 Pro 上，"15–20 条指令 / 4 权重"相对 273 GB/s 的权重流量是否够快，**没有数字**。若不够，结构优势归零。
4. **M=1 走 16×32 GEMM tile**，无 GEMV 路径 ⇒ decode 路径的效率未测。

**证伪判据**：在任一 Apple GPU 上把真实权重编成 `V4_T8`，测 PPL 与 decode tok/s。**若 PPL 相对 FP16 退化 >1 点，或解码器成为瓶颈 ⇒ 候选 1 作废。**

### 候选 2：**稀疏模式必须在"码流顺序"上定义，而不是在"权重矩阵"上定义**

**数学内容**（§3.3 推导）：熵编码格式（trellis）里，**跳过一个权重省的是计算，不自动省字节**——状态链必须连续。要省字节，必须 ① 被跳过的集合在**码流里连续**；② 格式有**重启/分块结构**。uzu 的 `V4_T8` 恰好有（`restart_columns = 64`）⇒ **块级稀疏（在 trellis 块粒度上剪枝）才能同时拿到计算与字节两份收益**。同时，**任何列重排（zigzag 的本质）都会打断转移链、抬高码率** ⇒ 布局与编码必须共设计。

**为什么可能有用**：这是把"trellis × 稀疏"从工程问题变成**有明确最优答案的数学问题**：块大小应当同时对齐 ① trellis 的重启块、② superblock、③ 缓存行。若三者能对齐，块级稀疏 + trellis 可能同时拿到"省字节"和"省 FLOPs"，而 SpQt 在 Q4_K 上的实测表明布局式稀疏在 Apple 上值 1.19–1.26×（**这不是一个无聊的量级**）。

**为什么可能失败（最强反证）**：
1. **★ 块级稀疏比 N:M 粗得多** ⇒ 同等稀疏率下质量损失更大（SpQt 已展示 50 % 稀疏的 PPL 悬崖：Llama-3-8B 5.93→9.59），**块级会更糟**。
2. **★ 唯一的相关实测是负面的**：在 Apple 上，稀疏化的全部收益（1.19–1.26×）**小于**单纯 F16→Q4_K 的 2.4–2.6×。⇒ **即使这条组合成功，它也不是主线。**
3. **trellis 的块大小是否与 llama.cpp / MLX 的 superblock 对齐，无人核过**（CurveTQ/QTIP 的确切分块结构我未核）。
4. **仍需一个可运行的 trellis 解码器**——回到候选 1 的第 2 条反证。

**证伪判据**：在 Q4_K 上先做**块级稀疏**（块 = llama.cpp superblock）的 Metal 实测。**若块级稀疏的字节收益 <10 %，或 PPL 退化超过 N:M 方案 ⇒ 候选 2 作废。**

### 6.1 明确**不**保留的方向（避免制造机会）

| 方向 | 为什么不做 |
|---|---|
| "把 uzu 的 Metal Trellis 接到 llama.cpp" | 许可证（MIT）**没有障碍**，但**没有模型、没有编码器、没有格式支持**，且目标硬件上被门控。**这是工程，不是数学候选**，且当前无可测对象 |
| "给 uzu 提交一个 PR 把 Trellis 打开到 M4 Pro" | 这需要改 `supports_mxu` 的门控语义 —— **它是否在 M4 Pro 上正确运行完全未知**（`simdgroup_matrix<int8_t>` family 门槛未核）。**在核清之前改门控等于制造一个跑不动的路径** |
| "旋转 × trellis 的联合设计" | RoPE-Aligned 已经给该族一个**不利**的经验结果；且 §3.2 显示正确表述是"△、无联合测量"，**不是一个值得投入的候选** |
| "Prox × 量化" | Prox §5.1 自述速度维度的联合测量**不存在**；且 §5.3 表明 27B Q4 下代理成本高达 36–41 %，**这个经济性本身就把方向否掉** |

---

## 7. R2 标题式结论的存活 / 修正表

| R2 结论 | **R3 判定** |
|---|---|
| §1 / §8.1：**"Metal 上没有 trellis 解码内核"** | **✘ 措辞错误**，须更正为"**没有 QTIP 族的 Metal trellis 内核；存在另一种格式（uzu）的实现，但非产品路径、无测量、且在 M4 Pro 上被门控**" |
| §1 / §10 判断 2：**"2-bit 在消费级不是一条可走的路"** | **✅ 存活**，理由更换为更硬的三条（非产品路径 / 无质量数据 / M4 Pro 被门控） |
| §1.1：**escha-port 的 13× L2 流量** | **✅ 存活**（本轮未复核，但也无反证） |
| §1 判断 B/C：**"gauge 变换该被降级"** / **"等价旋转 × 低位宽是互斥的替代关系"** | **△ 部分成立**：gauge 降级 ✔；但"互斥"✘ ⇒ 改为"旋转子族 △、加性子族 ✔" |
| §3.3 / §6.2：**Prox 的字节算术** | **✅ 存活**（§3.3 的 up/gate ≈0 % 字节节省与本轮无关） |
| §6.2 / 组合表：**"27B 4-bit +12 % ≈ 1.9 GB"** | **✘ 低估约 3×** ⇒ 真实上界 ≈5.7 GB，占比 36–41 % |
| §8.3：**"M2 Max 819.2 GB/s"** | **✘ 数值须改为官方 400 GB/s**（论文标注错误，非转录错误） |
| §8.3 / §10：**"M4 Pro 外推 1.2–1.3×"** | **✅ 存活**（官方带宽重算 = 1.218×），**但须补齐 5 个成立条件** |
| §9 假设 2：**"量化压倒稀疏"** | **✅ 存活并被强化**：SpQt 表 2 自身的 Baseline-F16 行就是 2.4–2.6× |
| 组合矩阵 `Prox × MTP = ✘` | **✘ 过强** ⇒ △，理由改为"性能代价"而非"破坏分布保持" |
| 组合矩阵 `trellis × SpQt = ✘` | **✘ 与 §3.1 自相矛盾** ⇒ △，并补"码流顺序共设计"这条硬约束 |
| §0.2 / §11：**"trellis 是否已合入 llama.cpp / MLX 未做穷尽性审计"** | **✅ 存活且现在有正面证据**：uzu 证明"Metal 上可以有 trellis"，但**不改变 llama.cpp / MLX 的结论** |

---

## 8. 本轮未能核实 / 明确不声称

| 项 | 状态 |
|---|---|
| **`metal::simdgroup_matrix<int8_t,8,8>`（int8 变体）在 Apple9/M4 上的可用性** | **未核**。WorkBuddy 的 R2 审计 C-04 已一手核过 **`simdgroup_float8x8` / `simdgroup_half8x8` 在 Apple9（含 M4）可用**（llama.cpp `mul_mv_mma.metal` 7 处、MLX `quantized_nax.h` 族），**int8 变体仍未核**（本环境 `web_search` 失败、Apple 文档未打开）⇒ 该问题已**收窄**，不再是一整族指令的可用性问题 |
| **uzu 的 kernel 在 M4 Pro 上是否编译通过、是否正确** | **未核**（禁止编译 / 禁止跑） |
| **`V4_T8` / `V2_T6` / `V2_T4` 的真实率失真质量（PPL）** | **完全未知**（从未编码过真实权重） |
| **真实权重能否以 2.0–2.125 bpw 进入 16-bit 状态链** | **完全未知**（整个格式的前提，未验证） |
| **`supportsMXU` 私有选择器在 M5 上的实际返回值** | **未核**（"True on M5+" 是仓库作者的说法，非 Apple 文档） |
| **SpQt 论文对 819.2 GB/s 的任何推导** | **论文没给** ⇒ §4.3 的两个诊断均为**假设** |
| **Apple 官方的 A19 / A19 Pro 带宽** | **未核**：iPhone 17 / 17 Pro 的 newsroom 与规格页 **"GB/s" 零命中**；SpQt 的 75.8 GB/s **无法与 Apple 对账** |
| **SpQt 的 batch size、生成 token 数、设备型号、内存/散热状态** | **论文全部未写** ⇒ R2/R3 的所有"端到端"口径都缺这一层 |
| **SpQt 如何在 iOS（A19 Pro）上跑 llama.cpp** | **论文未解释** |
| **Prox 的 12 % 自身推导** | **论文没给**（无表、无图、无字节数）；§5.2 的重建是 `[推算]`，但与 6/6 可核配置吻合在 10–13 % |
| **Prox 的分母是否排除 embedding** | **不可判定**（两种候选都落在 11–13.5 %） |
| **`Llama-3-8B` / `Gemma-3-12B` / `gemma-3-27b` 的 HF config** | **401（gated）**；对 §5.3 结论无影响 |
| **第二个真实的 27B 稠密模型配置** | 未核（`google/gemma-3-27b` 401）⇒ "27 层 × 4"变体已显式标注为**假设** |
| **CurveTQ / QTIP 的确切分块结构**（候选 2 需要它） | **未核** |
| **uzu `main` 分支的当前状态** | **未核**（全部结论锁定在 Issue 指定的 commit `2968097a`） |
| **Apple 官方印有 "819GB/s" 的 M3 Ultra 页面** | **未核**（newsroom 只说 "over 800GB/s"）⇒ 次要假设仅靠二手知识 |

---

## 9. 引用

- **uzu（全部锁定 commit `2968097ab20c66acbf1797b292c29bf18f71882c`）**
  `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/gemm_trellis.metal`
  `.../gemm/trellis.rs` · `.../gemm/common/quantized/trellis.h` · `.../gemm/selection.rs` · `.../gemm/kernel.rs` · `.../gemv/kernel.rs` · `.../matmul_metal_kernel.rs` · `.../metal/metal_extensions/device_extensions.rs` · `.../metal/context.rs` · `.../common/gpu_types/quantization_method.rs` · `.../common/kernel/matmul/matmul_b.rs` · `.../metal/kernel/matmul/common/simdgroup_multiply_accumulate.h` · `.../metal/kernel/matmul/common/mxu_fragment/cooperative_vectors.h` · `.../unit/backends/metal/kernel/matmul/trellis_gemm_test.rs` · `.../unit/BENCHMARKS.md` · `.../README.md`
  GitHub 代码搜索 `repo:trymirai/uzu`：`MatmulB::Trellis`(5)、`is_trellis`(8)、`supports_mxu`(24)、`simdgroup_matrix`(1)、`validate_engine`(2)
- **模型目录**：https://trymirai.com/local-models
- **SpQt**：https://arxiv.org/abs/2511.04477 · PDF `https://arxiv.org/pdf/2511.04477v1`（表 1 p.7、表 2 p.8）
- **Prox**：https://arxiv.org/abs/2607.27591 · §4.3 / §5.2 / §5.3 / A.3
- **Apple**：
  https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/
  `https://support.apple.com/en-us/121553`（MacBook Pro 2024 技术规格：M4 Pro 273GB/s、M4 Max 410/546GB/s）
  M4 Pro newsroom 2024-10（273GB/s）
- **配置**：`https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json`、`Qwen3.6-27B` 的 `config.json`
- **底层笔记**：[`fulltext-notes/R3-U-uzu-trellis.md`](./fulltext-notes/R3-U-uzu-trellis.md) · [`fulltext-notes/R3-S-spqt-bandwidth-prox.md`](./fulltext-notes/R3-S-spqt-bandwidth-prox.md)
