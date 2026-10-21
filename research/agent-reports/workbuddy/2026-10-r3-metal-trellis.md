# R3-W：Uzu 的 Metal Trellis —— 真实功能边界及与 IQ2_XS 的比较

**对象**：`trymirai/uzu`（MIT，Rust，2458 star，2026-10-20 仍在推）
**方法**：直接读源码（E0），未编译、未运行
**结论先行**：**Metal 上确实有可运行的 trellis（2.00–3.00 bpw，lookup-free），但它被硬门禁在 M5 的 MXU 上，在 M4 Pro 上直接报错不可用。**
**日期**：2026-10-21

---

## 1. 仓库事实

| 项 | 值 |
|---|---|
| 仓库 | `trymirai/uzu` |
| 许可 | MIT |
| 语言 | Rust |
| star | 2458 |
| 最后 push | 2026-10-20T22:11:34Z |
| 平台 | iOS / macOS / Mac Catalyst |
| 定位 | 端侧推理引擎（自带模型分发，模型 ID 形如 `alibaba:qwen3.5:0.8b:mirai:mirai-m:4`） |

`mirai-m:4` 是它自有的量化方案名。trellis 相关代码全部在 `crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/` 下。

---

## 2. Trellis 的真实形态

### 2.1 文件与规模

| 文件 | 行数 | 作用 |
|---|---|---|
| `gemm_trellis.metal` | 194 | 主 kernel + split-K 归约 kernel |
| `trellis.rs` | 147 | `TrellisGemm::encode_plan`，参数校验与 kernel 缓存 |
| `common/quantized/trellis.h` | ~120 | 状态解码、hash、`TrellisCursor` |
| `gemm/selection.rs` | 207 | `select_trellis_plan`、`TRELLIS_K_STEP = 64` |
| `activation_transform/trellis.metal` | ~300 | 激活侧变换（int8 量化 + 分组求和） |

规模上：**核心 trellis 解码不到 150 行 Metal + 120 行头文件**。这本身是个重要信息——trellis 的 Metal 实现并不需要"从零写几千行"。

### 2.2 三种格式与 bpw

格式常量来自单元测试 `unit/backends/metal/kernel/matmul/trellis_gemm_test.rs:23-37`（E0）：

```rust
const V4_T8: TrellisFormat = { vector_width: 4, transition_bits: 8, restart_columns: Some(64) };
const V2_T6: TrellisFormat = { vector_width: 2, transition_bits: 6, restart_columns: None };
const V2_T4: TrellisFormat = { vector_width: 2, transition_bits: 4, restart_columns: None };
```

字节公式（`trellis.h:block_bytes` + 测试里的 `bytes_per_block`）：

```
transitions      = block_columns / vector_width - 1
bytes_per_block  = ceil((16 + transitions × transition_bits) / 8)
bpw              = bytes_per_block × 8 / block_columns
```

代入：

| 格式 | vector_width | transition_bits | restart | 计算 | **bpw** |
|---|---|---|---|---|---|
| **V4_T8** | 4 | 8 | 64 | transitions = 15；`ceil((16+120)/8)` = 17 B / 64 权重 | **2.125** |
| **V2_T4** | 2 | 4 | 无 | K=4096：transitions = 2047；`ceil((16+8188)/8)` = 1026 B / 4096 | **2.004** |
| **V2_T6** | 2 | 6 | 无 | K=4096：transitions = 2047；`ceil((16+12282)/8)` = 1538 B / 4096 | **3.004** |

**所以 uzu 真的做到了 2.00 bpw 的 trellis。** 这与 QTIP / Proteus / CurveTQ 宣称的 ~2.0 bpw 是同一档。

注意 V2_T4 与 V2_T6 **没有 `restart_columns`**，即 trellis 沿整个 K 维不重置，误差会沿行传播。V4_T8 每 64 列重置一次，代价是 2.125 而不是 2.0。**这是一个明确的"质量 vs bpw"旋钮**，比 CurveTQ 的"per-block start state"更朴素（后者只存起始状态，不重置链）。

### 2.3 码本是 hash 生成的，不是查表

这是最值得注意的设计。`trellis.h`：

```metal
UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;
UZU_CONST uint WEIGHT_BIAS = 54;

static METAL_FUNC uint hash_without_final_fold(const uint code_state) {
  uint hash = code_state * HASH_MULTIPLIER + HASH_INCREMENT;
  hash ^= hash >> HASH_FOLD_BITS;
  return hash * HASH_AVALANCHE_MULTIPLIER;
}
```

16-bit state → 32-bit hash → 折叠 → 每字节解出 4 个 2-bit 域 → SWAR 求和得 8 个 level。**没有任何 codebook buffer 被读**（虽然 `MatmulB::Trellis` 结构里有一个 `codebook` 字段，但解码路径不依赖它）。

这正是 Proteus 宣称的 "lookup-free / compute-generated codebook"——**uzu 已经有一个可运行的独立实现**。它对 Metal 的意义很具体：**没有 threadgroup 里的查表，就没有 bank conflict，也不占 32 KB threadgroup memory**。QTIP 式码本查表在 CUDA 上被 Proteus 指为 MIO throttle 来源；hash 方案在这一点上结构性地更好。

### 2.4 有符号处理：`WEIGHT_BIAS` + epilogue 修正

`gemm_trellis.metal` 的 epilogue：

```metal
const int level_dot =
    biased_dot - int(trellis::WEIGHT_BIAS) * int(group_sum.x + group_sum.y + group_sum.z + group_sum.w);
const float4 offsets = float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
const float dot = float(level_dot) * scale_and_offsets[0] + metal::dot(group_sum, offsets);
output[...] = bfloat(dot * row_scales[row] * activation_scale);
```

解码出的 level 全部是**非负偏置值**（`weight + 54`）。要还原成有符号权重，不做逐权重符号位，而是：
1. 在 int32 累加后减去 `54 × Σ(group 内的激活和)`；
2. 再加 `dot(group_sum, offsets)` 补偿 4 个列组各自的偏置。

**代价**：需要预先算好 `column_group_sums`（激活的分组和，由 `activation_transform/trellis.metal` 产出）。
**收益**：**省掉每权重的 1 bit 符号位**。在 2 bpw 下 1 bit 是 50% 的额外开销——这是这套方案能做到 2.0 bpw 的关键之一。

这一条对"2.5bpw 复杂解码 vs 3.5–4bpw 简洁直算"的争论有直接意义：**符号位不是免费的，聪明的代数修正可以把它换成分组统计量**。

### 2.5 激活侧是 int8，不是 fp16

`trellis.rs` 解构：

```rust
MatmulA::Trellis { values: activations, column_group_sums, scales: activation_scales }
```

`gemm_trellis.metal` 的签名：`const device int8_t* activations`，左操作数 `GemmAPrologueKind::Int8Symmetric`，累加 `Fragment<int, ...>`（int32）。

**所以这是 W-trellis × A8 的整数 GEMM，不是 fp16 GEMV。** 与 llama.cpp 的 IQ 路径（fp16/fp32 激活）是完全不同的设计点：
- 好处：MMA 走 int8×int8→int32，dequant 只在 tile 装载时做一次；
- 代价：**引入了激活量化误差**，这是 llama.cpp IQ 路径没有的。质量账要分开算。

### 2.6 分派与 M=1

- `kernel.rs:223-229`：`MatmulB::Trellis { .. } => self.trellis.encode_plan(...)`，**无条件走 trellis GEMM，没有任何 M 阈值判断**。
- `gemv/` 目录下有两个 kernel：`gemv.metal`（通用）与 `gemv_mxu_tiled.metal`（MXU 专用），**但没有 trellis 版 GEMV**。

**所以：即使 MXU 可用，M=1 的 decode 也走 GEMM 路径。**

GEMM 的最小 tile 是 `Tile16x32x256_Simdgroups1x1`（M tile = 16）。M=1 时：
```metal
const short token_end = short(min(int(Core::SIMDGROUP_BLOCK_M), int(params->M) - int(token_base)));
```
→ `token_end = 1`，MMA 只用到 16 行里的 1 行，**MMA 利用率 1/16**；且 `aligned_m` 为 false，走的是 `dispatch_bool` 的未对齐分支。

在带宽受限时这未必致命（字节没多读），但在 M=1 时 kernel 更容易转为 issue/latency 受限。**uzu 没有为 trellis 做 decode 专用路径**，这本身说明作者认为 trellis 的主要价值在 prefill/大 M，或者不值得为 M=1 再写一套。

---

## 3. 关键边界：MXU 门禁 → M4 不可用

这是本节的核心。三处代码构成一条不可绕过的链：

**（1）`selection.rs:152-176`** —— `select_trellis_plan` 硬编码引擎：

```rust
pub(super) fn select_trellis_plan(...) -> GemmPlan {
    ...
    let plan = GemmPlan { engine: GemmEngine::Mxu, ... };
```

**（2）`selection.rs:80-86`** —— 引擎校验：

```rust
pub(super) fn validate_engine(&self, engine: GemmEngine) -> Result<(), GemmPlanError> {
    if engine == GemmEngine::Mxu && !self.supports_mxu {
        return Err(GemmPlanError::MxuUnavailable);
    }
```

错误消息原文：`"MXU engine is not available for this GEMM"`。

**（3）`metal_extensions/device_extensions.rs`** —— `supports_mxu` 的定义与注释：

```rust
/// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
/// True on M5+ (Gen18+), false on M1-M4.
fn supports_mxu(&self) -> bool {
    optional_selector_value(self, obfstr!("supportsMXU"), false)
}
```

**结论**：
- `supportsMXU` 是一个**私有 Objective-C selector**（用 `obfstr!` 混淆字符串，避免出现在二进制里），M1–M4 上不存在 → 走 `fallback = false`。
- 因此 M4 Pro 上 `supports_mxu == false` → `validate_engine` 返回 `MxuUnavailable` → **trellis 路径直接失败，没有回退**。

**这一条同时给出了一个独立证据：NAX/MXU 确实是 M5+ 的私有能力，M4 没有。** 这与 R2 的 audit C-04 一致（M4 有 `simdgroup_matrix` 但没有 MXU/NAX），且现在是**第三方引擎的代码级确认**，不再只是我的推断。

---

## 4. 与 IQ2_XS 的比较

### 4.1 字节

| 方案 | bpw | 27B 权重 | 相对 IQ4_XS | 相对 IQ2_XS |
|---|---|---|---|---|
| IQ4_XS | 4.25 | 14.34 GB | 1.00× | 0.544× |
| **IQ2_XS** | **2.3125** | **7.81 GB** | 1.84× | **1.00×** |
| uzu V4_T8 | 2.125 | 7.17 GB | 2.00× | **1.088×** |
| uzu V2_T4 | 2.004 | 6.76 GB | 2.12× | **1.154×** |
| QTIP/Proteus/CurveTQ（宣称） | ~2.0 | 6.75 GB | 2.13× | 1.157× |

**uzu 的 2.0 bpw 相对 IQ2_XS 只省 13.4% 字节。** 与 R2 对 QTIP 算出的 1.16× 完全一致——这不是巧合，是 2.0 vs 2.3125 的算术必然。

### 4.2 算术成本（估算，E2）

| 方案 | 每权重操作 | 依据 |
|---|---|---|
| IQ4_XS | ~2.0–2.5 | 4 B 取指 + 8 shift/mask + 8 scale-FMA |
| IQ2_XS | ~2.5–4 | 1 次码本索引 + 512×8 表查 + 符号 + 子块 scale |
| uzu V4_T8 | ~2.5–3 | 每 4 权重 1 个 16-bit state；state 解码 ≈ 2 mul + 2 xor-shift + SWAR(~6) ≈ 10–12 ops / 4 权重 |
| uzu V2_T4 | ~5–6 | 每 2 权重 1 个 state → 同样的 10–12 ops / 2 权重 |

**V2_T4 的算术成本约为 IQ2_XS 的 1.5–2 倍，而只省 13.4% 字节。** 按 R2 的 ALU 预算框架（6.7 ops/weight @ 124.6 GB/s），V2_T4 仍在预算内但余量很小；V4_T8 更安全。

**但 uzu 用 int8 激活 + int32 MMA，把 dequant 摊进了 MMA 流水线**，所以它的"ops/weight"不能与 llama.cpp 的 fp32 GEMV 直接相加比较——前者是 MMA 前置一次，后者是逐元素。**这是两套不同的成本结构，不能只比 ops 数。**

### 4.3 可用性矩阵

| 维度 | IQ2_XS（llama.cpp） | uzu trellis |
|---|---|---|
| Metal kernel | ✅ `kernel_mul_mv_iq2_xs_f32`（GEMV）+ `mul_mm` | ✅ `GemmTrellis`（仅 GEMM） |
| **M4 Pro 可运行** | ✅ **是** | ❌ **否**（`MxuUnavailable`） |
| M5+ | ✅ | ✅ |
| M=1 专用路径 | ✅ 有 GEMV | ❌ 无（走 M=16 tile 的 GEMM） |
| 激活精度 | fp16/fp32 | **int8（有量化误差）** |
| 需要额外 buffer | 无 | `column_group_sums` + `activation_scales` |
| 权重来源 | 任意 GGUF 量化器 | **仅 Mirai 自有格式**（`mirai-m:*`），无公开量化器 |
| 质量数据 | 公开（PPL、各类榜单） | **仓库内无公开基准数字**（`BENCHMARKS.md` 只讲怎么跑，`target/criterion` 需自行生成） |
| 代码规模 | ~8 K 行（SpQt 的 fork 量级） | trellis 部分 < 300 行 |

### 4.4 质量

**uzu 仓库内没有公开的 trellis 质量数字。** `crates/uzu-engine/unit/BENCHMARKS.md` 只有运行说明（`cargo bench -- "Metal/Kernel/Matmul"`），示例标签是 `m2_max` 与 `a19`，**没有 m4/m5 的既有结果**。

因此：**无法回答"uzu 的 2.0 bpw 与 IQ2_XS 的 2.3125 bpw 谁的质量更好"**。这是一个明确的空白，不能猜。

另外，`BENCHMARKS.md` 里有一条方法论值得借用：

> "GEMV-class benches cycle through enough quant-buffer copies to cover a **256 MiB weight working set** before reusing one. This avoids ranking kernels on cache-warm weights."

**256 MiB 是他们的 cache-cold 门槛。** 已写入 XD-1 的规则 2。

---

## 5. 对 R2 结论的影响

### 5.1 必须订正的（已写入 errata E-8）

| R2 原文 | 订正 |
|---|---|
| "Metal 上没有任何 trellis 实现" | **错。** uzu 有 `gemm_trellis.metal`，支持 2.00 / 2.125 / 3.00 bpw，lookup-free，MIT |
| "Proteus 是占位符、CurveTQ 无代码 → Metal trellis 不存在" | 这两条**本身成立**（Proteus 确实占位、CurveTQ 确实无代码），但**不能推出"Metal trellis 不存在"** |

### 5.2 强化了的（路线裁决不变）

**"不做 trellis 的 Metal 移植"这个裁决不但保留，而且理由更硬了。**

原来的理由是"没有实现，从零写太贵"。现在的理由是：

> 世界上最好的公开 Metal trellis 实现（uzu），在有能力写非 MMA 路径的前提下，**主动选择把 trellis 硬门禁在 M5 的 MXU 上，并且没有为非 MXU 提供任何回退**。这说明在作者的工程判断里，**trellis 解码离开矩阵单元就不划算**——而这正是 R2 的 ALU 预算分析预测的结论。

这是一个"反向验证"：R2 从 ops/weight 预算推出"trellis 在弱 ALU 预算下不划算"，uzu 从工程实践独立得出同一结论（用门禁而不是降级来表达）。

### 5.3 新增的一条可复用认知

**符号位的代数消除。** uzu 用 `WEIGHT_BIAS + column_group_sums + per-group offsets` 把逐权重符号位换成分组统计量，在 2 bpw 下省掉 50% 的编码开销。这条技术**与具体格式无关**，可以直接搬到任何码本/格点方案的 Metal 实现里。这是 R2 完全没有提到的一个具体优化点。

---

## 6. 可借用 / 不可借用

### 可借用（3 条）

1. **hash 生成码本，不查表。** 16-bit state → 32-bit hash → SWAR 解 8 个 level。零 threadgroup memory、零 bank conflict。若将来要在 Metal 上做任何 2-bit 码本方案，这是首选结构。
2. **`WEIGHT_BIAS` + 分组和 epilogue 修正。** 省掉逐权重符号位，用一次 int32 减法 + 一次 `dot(group_sum, offsets)` 换回。
3. **256 MiB cache-cold 门槛。** 已写入 XD-1。

### 不可借用（3 条）

1. **`GemmEngine::Mxu` 门禁。** 在 M4 上直接失败。要复用必须重写整条 `validate_engine` 链并补一条非 MMA 路径——那就是从零实现，不是复用。
2. **int8 激活。** 引入激活量化误差，质量账与 IQ 路径不可比。若目标是与 IQ2_XS 对照，必须换成 fp16 激活，那 `MxuMmaCore` 的整个 fragment 布局都要改。
3. **`mirai-m` 格式与量化器。** 仓库只有推理侧，没有公开的权重产出流程。**无法用 uzu 复现"自己量化一个 27B 到 2 bpw"**。

### 一个旁证（不作为结论）

`crates/uzu-engine/unit/backends/common/kernel/gdn/tree_verify/` 下有一组单元基准：`tree_gram_bench.rs`、`tree_update_solve_bench.rs`、`prefix_bench.rs`、`out_bench.rs`。

文件名表明 **GDN（gated delta net）的投机解码树验证在 uzu 里被实现过**（Gram 矩阵 + solve + 前缀）。这与 R2 的发现形成对照：mlx-lm 因为 `ArraysCache.is_trimmable() == False` 连投机解码都进不去，而 uzu 已经在做 GDN 的树验证。

**仅凭文件名，不构成实现细节的证据**（E0-弱）。若这条路线重要，需要读 `tree_update_solve_bench.rs` 对应的实现。

---

## 7. 开放问题

1. **uzu 在 M4 上到底能不能跑？** 4-bit 路径有 `gemv.metal`（非 MXU 变体），但整个引擎是否在 M4 上可用，本轮没有验证（未编译）。若能跑，则 uzu 的 4-bit GEMV 可以作为一个**独立于 MLX 与 llama.cpp 的第三个 4-bit 基线**，对 XD-1 有价值。
2. **`mirai-m:4` 的实际 bpw 是多少？** 名字里的 4 可能是"4 weights per state"（V4）而不是 4 bit。若是 V4_T8，则 2.125 bpw。
3. **uzu 是否有 M5 上的公开 tok/s？** 仓库内无。`BENCHMARKS.md` 的示例标签是 `m2_max` / `a19`，暗示他们主要在 M2 Max 与 A19 上测——**恰好都是没有 MXU 的芯片**，那他们测的是 `GEMM` 而非 `GEMM_MXU` 组。这进一步说明 trellis 可能还没有在任何公开硬件上被基准过。

---

## 8. 一句话

uzu 证明了 **2.0 bpw 的 trellis 可以在 Metal 上写得又小（< 300 行）又干净（hash 码本、无查表、无符号位）**；它同时证明了**在 M4 上跑不了**（`supports_mxu` = false on M1–M4），并且**作者选择门禁而不是降级**——这比"没人写过"更强烈地支持了"trellis 在 M4 上不划算"的裁决。
