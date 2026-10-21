# R3-U —— `trymirai/uzu` 的 Metal Trellis 实现：一手源码核验

**核验者**：MiniMax（报告作者本人，非子代理）
**核验日期**：2026-10-21
**锁定 commit**：`2968097ab20c66acbf1797b292c29bf18f71882c`（Issue #3 给出）
**约束遵守**：只读源码；未 clone、未编译、未下载模型、未跑任何基准；所有 HTTP GET 均用 `https://` + User-Agent。

---

## 0. 一句话结论

**uzu 确实有一个公开的、真实的 Metal Trellis GEMM 内核——但它不是 QTIP 系的格式，而且在项目目标机 M4 Pro 上会被引擎自身的策略门直接拒绝。**
它达到的是"**源码存在**"这一级，不是"产品路径"，更不是"完整实测"。

---

## 1. 四级区分（Issue #3 要求的口径）

| 级别 | 判定 | 证据 |
|---|---|---|
| **① 源码存在** | ✅ **是** | 完整 Metal kernel + Rust dispatch + 单元测试，MIT 许可，CI 覆盖 |
| **② 可编译** | ⚠️ **未核实** | 我不得编译。仓库有 CI + `uzu_test` 宏驱动的测试，但**该 kernel 是否在 M4 Pro 上编译通过、是否在 family 9/10 上支持所用指令，未核** |
| **③ 正式产品路径** | ❌ **否** | `QuantizationMethod` 无 Trellis；无 GGUF/模型格式支持；公开模型目录 **"trellis" 零命中**；README **从未提及** trellis/bpw |
| **④ 完整实测** | ❌ **否** | 唯一的测试用**随机字节**当 codes，只校验数值正确性；**无 PPL、无 tok/s、无 M4 Pro/M2 实测、无真实模型** |

---

## 2. 格式：它到底编码了什么（**与 QTIP 系完全不同**）

来源：`crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/common/quantized/trellis.h`

```c
UZU_CONST uint K_STEP = 64, WEIGHT_BIAS = 54;
UZU_CONST uint STATE_BITS = 16, BYTE_BITS = 8, V4_VECTOR_WIDTH = 4, V2T6_TRANSITION_BITS = 6;
UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;
```

**机制（`trellis.h` 逐行可读）**：

1. 每行按 `vector_width` 切成"状态"；状态是 **16 bit**。
2. 相邻状态之间用 `transition_bits` 位的**转移**连接 ⇒ 整行是一条**马尔可夫链**（这就是 "trellis" 的来源）。行首存 16 bit 初始状态，其余靠转移重建。
3. 解码路径：`state → hash → 4 字节 → 每字节 4 个 2-bit 域 → 有偏权重`。
   ```c
   static METAL_FUNC uint biased_weights(const uint hash_bytes) {
     const uint nibble_field_sums = hash_bytes - 3 * ((hash_bytes >> 2) & 0x33333333u);
     const uint byte_field_sums = (nibble_field_sums + (nibble_field_sums >> 4)) & 0x0F0F0F0Fu;
     const uint low_nibble_dither = ((hash_bytes & 0x0F0F0F0Fu) * 3) & 0x0F0F0F0Fu;
     return (byte_field_sums << 3) + low_nibble_dither;
   }
   ```
   注释给出语义：*"Per byte b: weight + WEIGHT_BIAS = 8 * (sum of b's four 2-bit fields) + ((3 * (b & 15)) & 15)."*
   ⇒ **一个 hash 字节产生一个 4-bit 权重**（去掉 `WEIGHT_BIAS=54` 的偏置）。
4. 真实权重 = **共享低秩基 + level 项**。见 `gemm_trellis.metal` 的 `TrellisEpilogue::store`：
   ```c
   const float4 group_sum = group_sums[token];
   const int level_dot = biased_dot - int(trellis::WEIGHT_BIAS) * int(group_sum.x + group_sum.y + group_sum.z + group_sum.w);
   const float4 offsets = float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
   const float dot = float(level_dot) * scale_and_offsets[0] + metal::dot(group_sum, offsets);
   output[...] = bfloat(dot * row_scales[row] * activation_scale);
   ```
   `scale_and_offsets` 就是 `codebook`（Rust 侧 `MatmulB::Trellis { codes, row_scales, codebook, format }`，`codebook` 在 `trellis.rs` 中作为 `scale_and_offsets` 传入），**共 5 个 float**：1 个 scale + 4 个基向量偏移。单元测试里写死为 `const CODEBOOK: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];`。

### 2.1 有效 bpw（本报告按代码算术，`[推算]`）

`block_bytes = (STATE_BITS + transition_count * transition_bits + 7) / 8`，`transition_count = block_columns/vector_width - 1`。

| 格式（测试中实际出现的三个） | vector_width | transition_bits | restart_columns | 码率 `[推算]` |
|---|---|---|---|---|
| `V4_T8` | 4 | 8 | 64 | **2.125 bpw**（每 64 列 17 字节） |
| `V2_T6` | 2 | 6 | 无 | **≈3.0 bpw** |
| `V2_T4` | 2 | 4 | 无 | **≈2.0 bpw** |

**另有**：每输出行 1 个 f32 `row_scales`（32 bit / N，可忽略）；每个权重矩阵 5 个 f32 `codebook`（可忽略）。

⇒ **uzu Trellis 的码率与 QTIP/Proteus/CurveTQ 的 2-bit 处在同一量级，但它是完全不同的码**：不是"向量量化 + 码本"，而是"**状态链 + 哈希合成 + 共享秩-4 基**"。

### 2.2 ★ 这个设计恰好绕开了我 R2 指认的"内循环墙"

R2 §1 的核心判断是：低比特解码器的**查表 / 码本访问**会变成内循环瓶颈（escha-port 实测：查表产生 **13× 于权重的 L2 流量**）。

uzu 的解码**没有任何码本查表**：
- 码本只有 **5 个 float**（共享基），全权重矩阵共用；
- 状态→权重的展开是 **纯 ALU**：`state * MULT + INC` → `hash ^= hash>>16` → `hash *= AVALANCHE` → `hash ^= hash>>16` ⇒ 约 6–8 条整数指令产生 4 个权重，再加 `biased_weights` 的约 8 条 ⇒ **约 15–20 条 ALU 指令 / 4 个权重**。
- ⇒ **这是一条与 Proteus / CurveTQ / XOR-Trellis 不同的答案**：不是"把查表做快"，而是"**把查表变成查表不变的东西**"。

**但这只是结构论证，不是测量。** 见 §5。

---

## 3. M=1 Decode 是否可行？

**可行，但只有 GEMM 一条路，且没有专门 decode 内核。**

| 事实 | 证据 |
|---|---|
| **M=1 被测试覆盖** | `trellis_gemm_test.rs`：`const CASES: [(usize,usize,usize);7] = [(1,80,5120), (17,80,5120), (512,80,5120), (16,128,64), (32,128,64), (1,6,64), (2048,2048,128)];` ⇒ **M=1 的两个用例都在** |
| **M 是运行时参数，tile 越界被正确处理** | `gemm_trellis.metal`：`const short token_end = short(min(int(Core::SIMDGROUP_BLOCK_M), int(params->M) - int(token_base)));` + `alignment.contains(GemmAlignment::M)` 模板分派（对齐 / 非对齐 M 两条路径） |
| **★ 没有 GEMV 路径** | `metal/kernel/matmul/gemv/kernel.rs`：`MatmulB::Trellis { .. } => { return Err(MatmulMetalError::UnsupportedLayout { path: "Gemv" }); }` ⇒ **Trellis 权重在 decode 时必须走 GEMM 内核**（M=1 的 GEMM） |
| **最小 tile 是 16×32×256 / 1 simdgroup** | `VARIANTS(..., GemmTiling::Tile16x32x256_Simdgroups1x1, Tile64x64x256_Simdgroups2x2, Tile128x128x256_Simdgroups4x4)` ⇒ M=1 时 **M 维浪费 15/16 的计算槽位**。对 batch=1 decode（带宽受限）这可能无害，但**未测量** |
| **K 必须被 64 整除** | `selection.rs`：`if self.shape.b_is_trellis && !self.shape.k.is_multiple_of(TRELLIS_K_STEP) { return Err(GemmPlanError::UnsupportedLayout("Trellis K must be divisible by 64")); }`，`TRELLIS_K_STEP = 64` |
| **split-K 有约束** | `trellis.rs`：`split_k > 1 && !n.is_multiple_of(4)` ⇒ 不支持；且 `n % 4 == 0` 才能多路 split-K |
| **输出必须 BF16** | `gemm/kernel.rs`：`if self.output_data_type != DataType::BF16 { return Err(MatmulMetalError::UnsupportedDataType(...)); }` |
| **B 布局受限** | `trellis.rs`：`!arguments.b_transpose` / `arguments.b_leading_dimension.is_some()` / `output_stride < n` ⇒ 不支持；输出不得带 mask |

**结论**：M=1 **在代码层面是支持的**（有测试覆盖），但代价是 **M=1 走 16×32 的 GEMM tile、没有 GEMV、没有 decode 专用内核**。

---

## 4. ★★ Metal 硬件门槛 —— **本轮最决定性的发现**

### 4.1 引擎自己的门：M1–M4 一律拒绝

`crates/uzu-engine/src/backends/metal/metal_extensions/device_extensions.rs`：

```rust
/// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
/// True on M5+ (Gen18+), false on M1-M4.
fn supports_mxu(&self) -> bool {
    optional_selector_value(self, obfstr!("supportsMXU"), false)
}
```

- `optional_selector_value` 的语义（同一文件）：`if device.respondsToSelector(selector) { raw_msg_send(...) } else { fallback }`，**fallback = false**。
- `context.rs:130`：`let supports_mxu = device.supports_mxu();` ⇒ 装配进 `GemmProblem::new(..., supports_mxu, apple_gpu_family)`。

### 4.2 Trellis 分支绕过门控，但生产路径另有拦截

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
    ...
}
```

`select_trellis_plan` **无条件**返回 `engine: GemmEngine::Mxu`。
`metal/kernel/matmul/matmul_metal_kernel.rs:97-99`：

```rust
if shape.b_is_trellis {
    return MatmulDispatch::Gemm(problem.select_plan());
}
```

`metal/kernel/matmul/gemm/kernel.rs:198`（**生产编码路径**）：

```rust
self.problem(shape, command_buffer.context()).validate_engine(plan.engine)?;
```

### 4.3 ⇒ 推论

> **在 M4 Pro 上，`uzu` 的 Metal Trellis GEMM 会以 `MxuUnavailable`（"MXU engine is not available for this GEMM"）失败。**
> 链条：`supportsMXU` 私有选择器在 M1–M4 上不存在 → `respondsToSelector` 为假 → fallback `false` → `validate_engine` 判定 `engine == Mxu && !supports_mxu` → **Err**。

**必须同时标注的两点限制**：

1. **这是"引擎策略门"，不是已证实的"硬件不能跑"。** 内核本身用的是 `metal::simdgroup_matrix`（Metal 3，family 8+ 即可），见 `simdgroup_multiply_accumulate.h`：
   ```c
   typedef metal::simdgroup_matrix<AccumulatorT, ROWS, COLS> AccumulatorMatrixType;
   ```
   ⇒ **该指令在 M4 Pro 上很可能可用；是 uzu 的策略选择了不使用它。** 这两件事必须分开陈述。
2. **`metal::simdgroup_matrix<int8_t, 8, 8>`（8-bit 整型矩阵类型）的 family 门槛我未核实**（Apple 官方文档未打开；本环境 web_search 失败）。**这是本轮最该由下一位研究者补上的一格**：如果 int8 矩阵类型本身要求 family 9+/10+，那么 M4 Pro 的"策略门"其实是对的，"硬件门"也成立。

---

## 5. 有没有真实测量？—— **没有**

`crates/uzu-engine/unit/backends/metal/kernel/matmul/trellis_gemm_test.rs` 是唯一的 Trellis 测试：

```rust
const CODEBOOK: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];
const RELATIVE_TOLERANCE: f32 = 1e-2;
const CASES: [(usize, usize, usize); 7] =
    [(1, 80, 5120), (17, 80, 5120), (512, 80, 5120), (16, 128, 64), (32, 128, 64), (1, 6, 64), (2048, 2048, 128)];
```

- `let stored: Vec<u8> = (0..n * row_bytes(format, k)).map(|_| rng.random()).collect();`
  ⇒ **codes 是随机字节**。测试只验证"kernel 的算术与 CPU 参考实现 `decode_level` 在 1e-2 相对误差内一致"。
- **没有**：真实量化权重、perplexity、tok/s、M4 Pro / M2 / M2 Max / A19 Pro 的任何数字。
- `crates/uzu-engine/unit/BENCHMARKS.md` 列出 Criterion 基准组与 `m2_max` / `a19` 的 `CRITERION_HOME` 标签 ⇒ **基础设施存在**，但 **Trellis 不在基准组列表里**（列表为 GEMM / GEMM_MXU / A8W / UnifiedQuantizedGemm / Gemv / Qwen3Layers / RMSNorm / Sampling / ChatSession / Forward pass）。

---

## 6. 模型与格式转换：有没有现成的可用产物？

| 问题 | 答案 | 证据 |
|---|---|---|
| 公开模型目录里有 Trellis 模型吗？ | ❌ **没有** | `https://trymirai.com/local-models`（271,718 B）全文 **"trellis" 零命中**；量化项只有 **GGUF K-Quants（`Q4_K_M` / `Q4_K_XL` / `Q8_K_XL`）** |
| 有格式转换工具吗？ | ❌ **没有找到** | 仓库树中 `convert` 相关路径只有 `crates/legacy/uzu/bindings/typescript/scripts/utils/convert-jsr-readme.cjs`（README 转换脚本）。**没有量化/编码 CLI** |
| `QuantizationMethod` 里有 Trellis 吗？ | ❌ **没有** | `backends/common/gpu_types/quantization_method.rs` 全文只有三个变体：`ScaleBias` / `ScaleZeroPoint` / `ScaleSymmetric` |
| 模型文件格式？ | **不是 GGUF** | 仓库树中 `gguf` 路径 **0 命中**；使用自有 `shoji` 模型类型体系 |
| 谁构造 `MatmulB::Trellis`？ | **只有测试与后端实现** | GitHub 代码搜索 `MatmulB::Trellis` 命中 5 个文件：`metal/.../gemm/trellis.rs`（消费方）、`metal/.../gemv/kernel.rs`（报错）、`metal/.../gemm/kernel.rs`（分派）、`cpu/.../reference.rs`（CPU 参考）、`unit/.../trellis_gemm_test.rs`。**没有任何模型加载代码构造它** |
| README 是否宣传？ | ❌ **完全没有** | README（46,713 B）中 `trellis` / `bpw` / `quant` / `bit` **零命中** |

---

## 7. 其它已核事实（供交叉审查引用）

- **许可**：MIT（README badge + `LICENSE`）。**这与 QTIP 的 GPL-3.0 相反**——R2 记录了 vLLM 因 GPL-3.0 而 `not_planned` 关闭 QTIP 的 issue `#11416`；**uzu 是 MIT**，意味着"把 Metal Trellis 接到 llama.cpp / vLLM"在**许可层面没有障碍**（工程层面仍然是重写）。
- **项目性质**：uzu 是**真实商用推理引擎**（PyPI `uzu` 0.6.11、npm `@trymirai/uzu`、Swift Package.swift、Mirai 产品线、`docs.trymirai.com`、Discord、Discord 上的 benchmark 说明）。**这不是玩具仓库**，这一点应当写进任何"R2 判断过时了"的更正面说法里——但**它同时意味着 Trellis 不是其产品特性**。
- **激活侧有配套 Trellis 变换**：`metal/kernel/activation_transform/trellis.metal` + `backends/common/kernel/trellis.rs`（`mixing_dimension`）⇒ 存在"激活混合矩阵 + 列分组和"的配套实现，用于把 int8 激活与 level 化权重对齐。
- **`restart_columns` 是个关键结构**：`GemmTrellisFormat { vector_width, transition_bits, restart_columns }`；`row_bytes` 里 `restart_columns != 0` 时**每 64 列独立重启状态链**（`V4_T8` 即如此）。**这正是"按码流块跳过"的必要结构**（见交叉审查 §5 候选 2）。

---

## 8. 我未能核实的（交棒清单）

| 项 | 状态 |
|---|---|
| `metal::simdgroup_matrix<int8_t,8,8>` 的 Apple GPU family 门槛 | **未核**（本环境 `web_search` 失败；Apple 文档未打开）⇒ M4 Pro 上"硬件能不能跑"仍是未知 |
| 该 kernel 在 M4 Pro 上是否编译通过 | **未核**（禁止编译） |
| `V4_T8` / `V2_T6` / `V2_T4` 三种格式的**率失真质量** | **完全未知**（无真实编码、无 PPL） |
| 真实权重能否以 2.0–2.125 bpw 无损/低损地编进 16-bit 状态链 | **完全未知**（这是整个格式成立的前提，**未验证**） |
| `supportsMXU` 私有选择器在 M5 上的实际返回值 | **未核**（文档注释是仓库作者的说法，非 Apple 文档） |
| uzu 官方是否在别处发布过 Trellis 模型或基准 | **未核**（只查了 `trymirai.com/local-models` 与仓库树） |
| 仓库 `main` 分支当前状态 | **未核**（全部结论锁定在 Issue 指定的 commit `2968097a`） |

---

## 9. 引用清单（全部一手 URL）

- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/gemm_trellis.metal`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/trellis.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/common/quantized/trellis.h`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/kernel.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/gemv/kernel.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/matmul_metal_kernel.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/metal_extensions/device_extensions.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/context.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/common/gpu_types/quantization_method.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/common/kernel/matmul/matmul_b.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/common/simdgroup_multiply_accumulate.h`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/src/backends/metal/kernel/matmul/common/mxu_fragment/cooperative_vectors.h`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/unit/backends/metal/kernel/matmul/trellis_gemm_test.rs`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/crates/uzu-engine/unit/BENCHMARKS.md`
- `https://github.com/trymirai/uzu/blob/2968097ab20c66acbf1797b292c29bf18f71882c/README.md`
- `https://trymirai.com/local-models`
- GitHub 代码搜索（`repo:trymirai/uzu`）：`MatmulB::Trellis`（5 命中）、`is_trellis`（8 命中）、`supports_mxu`（24 命中）、`simdgroup_matrix`（1 命中）、`validate_engine`（2 命中）
