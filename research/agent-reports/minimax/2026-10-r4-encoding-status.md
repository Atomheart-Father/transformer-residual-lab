# R4-M — Trellis 编码与可用权重：证据状态核查

**任务**：R4 / [Issue #6](https://github.com/Atomheart-Father/transformer-residual-lab/issues/6)
**研究者**：MiniMax · **日期**：2026-10-21
**纪律**：本轮**只读**。未下载模型、未训练、未量化、未运行代码/实验、未安装评测工具、未修改任何旧报告。所有结论来自**公开的 PR、源码、checkpoint 元数据与模型卡**。
**证据等级**：`[P]` 一手已打开 · `[O]` 官方产物直读 · `[S]` 二手 · `[推算]` 本报告算术 · `[未核实]` 未打开

---

## 0. 结论先行

| Issue #6 的问题 | 判定 |
|---|---|
| **是否存在真实权重的 Trellis encoder？** | ⚠️ **内部存在且已用于生产一个 27B 模型，但公开不可用**。第三方可以**消费**，不能**生产** |
| **独立的质量验证？** | ❌ **没有**。公开产物只有**解码器数值一致性**（与稠密包 99.956 % 逐位相同）与**字节账**；**没有任何 PPL / MMLU / 任务分数** |
| **配套 GGUF / MLX 转换？** | ⚠️ **不是 GGUF**。是自有的 `trellis.mirai` + 18 分片 safetensors + 一个 vLLM 插件 wheel（4 个版本）+ 一个预编译 `uzu` 二进制 |
| **实际模型卡与可追溯 bpw？** | ✅ **有，且可完整追溯**：`physical_package_report.json` 给出布局分布、逐类字节数与"每个 leaf 可逐位还原"的证书 |
| **"Mirai Medium ≠ Trellis"？** | ✅ **正确**。`-M` 是 **4-bit**（10 个文件）；**Trellis 是 `-S-experimental`**（2-bit，46 个文件） |

> **并且：本轮推翻了我 R3 的两条结论**（见 §5）。这不影响 R3 的方向性判断，但**措辞与理由必须更正**。

---

## 1. 四个一手来源的精确状态

| 对象 | 状态 | 规模 | 关键内容 |
|---|---|---|---|
| **uzu PR #899** "Support Trellis model" | 🔴 **OPEN / 未合并** | 2026-10-07，CC-Yeh，28 文件，+907/−530 | 正文：*"Support M5 and Specdec first, **AR and pre-M5 in future PRs**"*；链接 lalamo PR #364 |
| **uzu PR #900** "Support simdgroup trellis gemm" | 🔴 **OPEN / 未合并** | 2026-10-07，CC-Yeh，6 文件，+262/−105 | 移除 trellis 的 MXU 特判 ⇒ **M4 落到 Simdgroup 路径**；`gemm_trellis.metal` +155/−53 |
| **lalamo PR #364** "Load packed S checkpoints with trellis and lattice arrays" | 🔴 **OPEN / 未合并** | 2026-09-11，ry2009，18 文件，+1330/−49 | **加载器**，不是编码器；正文自述 *"No fitting, training or KV-cache implementation is added"* |
| **uzu `main` @ `2968097a`** | ✅ 已合并 | — | 只有**加载器 + MXU 内核**；无编码器 |

### 1.1 PR #899 改了什么（28 文件中关键项）

| 文件 | 变更 | 含义 |
|---|---|---|
| `encodable_block/linear/transformed_linear.rs` | **新增 +290** | **加载器 + 运行时装配**，非编码器 |
| `encodable_block/linear/rht_wrapper.rs` | **删除 −214** | RHT 包装层被 `transformed_linear` 取代 |
| `encodable_block/linear/weight_matrix.rs` | +210/−162 | 权重张量装载路径 |
| `config/weight_matrix/qtip_gaussian.rs` | **新增** | `QtipGaussianSpec { vector_width, transition_bits, restart_columns }` |
| `config/weight_matrix/row_stack.rs` | **新增** | `RowStackSpec { parts }` |
| `config/weight_matrix/hybrid_spec.rs` | +8 | 新增 `IncoherenceKind { Hadamard, Kronecker }` |
| `gpu_types/trellis.rs` | +5 | `mixing_dimension` 迁入 |
| `metal/.../gemm/trellis.rs` | +5/−11 | **删掉 `plan.engine != GemmEngine::Mxu` 限制** |

### 1.2 PR #900 的关键语义改动（`selection.rs`）

```diff
 pub fn select_plan(self) -> GemmPlan {
-    if self.shape.b_is_trellis {
-        return select_trellis_plan(self.shape.m, self.shape.n, self.shape.k);
-    }
     let engine = if self.supports_mxu && mxu_is_eligible(self.shape) {
         GemmEngine::Mxu
     } else {
```

⇒ **trellis 不再被强制绑到 MXU 引擎**。在 `supports_mxu == false` 的设备（M1–M4）上，它会落到 `GemmEngine::Simdgroup`。
⇒ **这正是我在 R3 §2.5–§2.6 判定的"M4 Pro 门控"的解除方式。** 且 trellis 的 split-K 另立分支，新增硬约束 `if !shape.n.is_multiple_of(4) { return 1; }`。

### 1.3 lalamo PR #364 的正文要点（逐条抄录级）

> *"Loads packed S checkpoints through `LanguageModel.load` **without refitting their saved weights**. This covers the **Qwen3.8/Qwen3.6 E3, Muse E3 and phone XS package formats**."*
>
> *"**QTIP uses MSB-first trellis codes, one folded float32 scale per output row and a five-float codebook.** V2 supports 4/6/8-bit transitions; V4 supports connected T8 and T6/T7/T8 with 64- or 128-column restarts. Restart blocks are contiguous in bits, with 16-byte row alignment."*
>
> *"**Hybrid owns rotation**; RowStack preserves contiguous output-row order. D4 embeddings export the required `kind: d4`. I4S4 linears use the existing affine MLX INT4 format; **I3 support is removed**."*
>
> *"The loader reads each package once, folds saved row gains, rejects non-unit column gains and unknown gain axes, and checks for unconsumed tensors and specs. **No fitting, training or KV-cache implementation is added.**"*
>
> *"Native integration depends on Uzu #899, including its **#925 packing update**."*
>
> *"Validation: independent Torch-produced decoder fixtures, explicit Kronecker reference, all ten tape layouts, native D4 metadata, mixed-format package load, gain folding, and disk save/reload. … **The opt-in full-package check requires `PACKED_CHECKPOINT_DIRECTORY`; no new full-model, Metal or physical-phone run is claimed.**"*

**三条可直接引用的事实**：
1. **格式就是 QTIP 的 trellis 码**（MSB-first），带**每输出行一个折叠后的 f32 scale** 和**五浮点码本** ⇒ 与 uzu 内核里的 `row_scales` / `codebook` 结构**完全吻合**。
2. **这是加载器，不是编码器**（"without refitting their saved weights"、"No fitting, training"）。
3. **没有任何全模型 / Metal / 真机运行被声称**。

---

## 2. 公开 checkpoint：确凿存在，且 bpw 可完整追溯

### 2.1 `trymirai/Qwen3.8-27B-S-experimental` `[O]`

| 项 | 值 |
|---|---|
| 标签 | `2-bit` · `experimental` · `uzu` · `vllm` · `mirai` · `safetensors` · `apache-2.0` |
| 下载量 / 最后修改 | 1583 / 2026-09-25 |
| 文件数 | **46** |
| 模型卡标题 | *"Qwen3.8-27B, Mirai S, experimental"* |
| **码率自报** | *"quantized with the Mirai S codec: **2.4 bits per weight, 8.45 GB on disk**"* |
| 自述状态 | *"This is a work in progress and **the numbers here are not final**."* |

**关键文件清单（按功能）**：

| 文件 | 作用 |
|---|---|
| `physical_package_report.json` | **字节与布局账** |
| `loader_validation.json` | **解码器 vs 稠密包的数值一致性** |
| `provenance.json` | **溯源**（基包路径、fit 产物路径、逐张量 sha256） |
| `encoding.json` | `[{"name":"qwen3.8","type":"hanashi"}]` ⇒ 编码器标识 |
| `vllm/trellis.mirai` | **权重本体**（vLLM 路径） |
| `vllm/model-0000{1..18}-of-00018.safetensors` + `model.safetensors.index.json` | vLLM 兼容分片 |
| `vllm/mirai_s-0.1.0 / 0.1.1 / 0.2.0 / 0.2.1-py3-none-any.whl` | **vLLM 插件 4 个版本** |
| `uzu/uzu` | **预编译 Apple Silicon 运行时** |
| `speculator/{config.json,model.safetensors}` | **投机解码草稿模型** |
| `speedcheck.py` | 测速脚本（**只测速度，不测质量**） |

### 2.2 ★ bpw 可追溯性：`physical_package_report.json` `[O]`

```json
{
 "candidate": "Qwen3.8-27B p8zm rr1 + QAT 38zmrr2 (q38zmrr2-package-qat-v1), physical S",
 "leaf_count": 272,
 "layouts": { "V2_k6_r0": 97, "V2_k4_r0": 9, "V4_k8_r64": 166 },
 "transformer_payload_bytes": 7469655040,
 "surface_payload_bytes": 815482880,
 "model_safetensors_bytes": 8447545388,
 "model_directory_bytes": 8467719877,
 "leaf_reproduction": "every leaf refolds to the dense package bit for bit (recover_p8zm_codes.py certificate)"
}
```

**★ 一个可独立复算的一致性检查** `[推算]`：

按 leaf 数加权的平均码率 = (97×3.0 + 9×2.0 + 166×2.125) / 272 = **2.433 bpw** ⇒ **与模型卡自报的 "2.4 bits per weight" 吻合**。
（`V2_k6` = vector_width 2 / transition_bits 6 ⇒ 3.0 bpw；`V2_k4` ⇒ 2.0 bpw；`V4_k8_r64` ⇒ 2.125 bpw；`r0` = 无重启块，`r64` = 64 列重启。）

**⇒ 关键事实：一个模型里混用了三种 trellis 布局。** 这说明 **"2-bit trellis" 不是单一格式**，任何"2-bit bpw"的引用都必须注明是哪种布局。

**并且"2-bit"这个标签本身也不唯一**（WorkBuddy R4 `2026-10-r4-experiment-readiness.md` §字节口径）`[P]`：
**MLX affine group-64 的 2-bit = 2.5 bpw**（每 64 权重 2 个 bf16 参数），**IQ2_XS = 2.3125 bpw**（74 B / 256 权重）
⇒ **同一个"2-bit"标签下差 8 % 的字节。** ⇒ **引用 bpw 必须同时给出：格式名 + 载荷口径（是否含 scales / bias / group sums）。**

**字节账** `[推算]`：
- transformer 载荷 7,469,655,040 B + surface（embedding）815,482,880 B = 8,285,137,920 B；
- `AIR_TOTAL_PARAMS = 26,895,998,464` ⇒ 全模型 **2.513 bpw**；若只用 transformer 部分参数则 **≈2.45 bpw**。
- ⇒ 模型卡的 "2.4 bits per weight" 与这些原始字节**不自洽到小数点后一位**（差 ≈0.05–0.11 bpw）。**引用时应同时给出 bpw 与原始字节数**，不要只给 "2.4"。

### 2.3 `loader_validation.json`：唯一的独立数值验证 `[O]`

```json
{ "package": ".../q38zmrr2-s-package-v1",
  "dense":   ".../q38zmrr2-package-qat-v1",
  "checked": 613, "total_mismatches": 4278, "seconds": 1207 }
```

逐张量（节选）：

| 张量 | kind | shape | 逐位不符率 | 超出 1 ULP | max_abs |
|---|---|---|---|---|---|
| `layers.0.mixer.in_proj.weights` | **STrellisMatrix** | 16480×5120 | **4.39e-4** | 17 | 4.88e-4 |
| `layers.0.mixer.out_proj.weights` | STrellisMatrix | 5120×6144 | 4.58e-4 | 9 | 4.88e-4 |
| `layers.0.mlp.up_projection.weights` | STrellisMatrix | 34816×5120 | 4.35e-4 | 27 | 2.44e-4 |
| `decoder.embedding.input_embedding.weights` | SSurfaceMatrix | 5120×248320 | **0.1228** | 0 | **3.66e-8** |
| `decoder.embedding.output_embedding.weights` | SSurfaceMatrix | 248320×5120 | 0.0419 | 0 | 2.44e-4 |
| `ropes.0.cosines` / `.sines` / `norm.scales` / `a_log` / `dt_bias` | array | — | `equal: true` | — | — |

**解读**：
- **Trellis 线性层**：解码结果与稠密包 **99.956 % 逐位相同**，残差在 bf16 量级（max_abs ≈ 2.4e-4 ~ 4.9e-4）⇒ **解码器实现正确**。
- **embedding（surface）**：不符率高但 **max_abs 仅 3.66e-8**（1 ULP 内）⇒ 是 f32 舍入噪声，不是错误；第二个 embedding 的 max_abs 2.44e-4 较大，**成因未核实**。
- **⭐ 这是一次真正的独立验证**（"independent Torch-produced decoder fixtures" + 613 张量 + 1207 秒），但它验证的是 **"解码器 == 稠密包"**，**不是"量化后模型的能力"**。

### 2.4 `provenance.json`：溯源链完整 `[O]`

- `base_pkg` / `model_dir` = `/data/ry2009/naq_research_20260726/sbt-air-q38-model-v1`
- `base_config_sha256` / `model_config_sha256` = `a9bc65ff…756999`；`tokenizer_sha256` = `87a7830d…3db72de4`
- `identity_tensors`：5 个张量的 sha256（`0.mixer.a_log`、`0.mixer.dt_bias`、`0.pre_mixer_norm.scales`、`5.mixer.conv.weights`、`31.pre_mlp_norm.scales`）
- 环境变量（**这是 encoder 存在的直接证据**）：
  `COMPOSE_QTIP_FITS`、`COMPOSE_K3_FITS`、`COMPOSE_K4_FITS`、`COMPOSE_E8P_FITS`、`COMPOSE_I4_FITS`、`COMPOSE_HEAD: i3_s4`、`COMPOSE_EMB: d4_s4`、`AIR_BASIS: fullinc`、`AIR_TOTAL_PARAMS: 26895998464`、`COMPOSE_NO_CANTRY: 1`

### 2.5 ⚠️ 一个必须点明的事实：**这个模型是 QAT 产物，不是纯 PTQ**

三处独立证据：
- `physical_package_report.json` 的 candidate：*"Qwen3.8-27B p8zm rr1 **+ QAT 38zmrr2**"*；
- `loader_validation.json` 的稠密参照包：`q38zmrr2-package-**qat**-v1`；
- `COMPOSE_HEAD: i3_s4` / `COMPOSE_EMB: d4_s4` ⇒ **embedding 与 head 用的是 I3/I4/D4 仿射格式，不是 trellis**。

⇒ **⇒ 对本项目"冻结权重"前提的直接含义**：**目前唯一公开可下载的 2-bit trellis 强模型，是用 QAT 做出来的**。这与我在 R2 §10 判断 7 的方向一致：**"冻结权重"恰好把这一族挡在质量最扎实的部分之外。**

### 2.6 ❌ 仍然缺失的：**没有任何下游任务质量**

- 模型仓 46 个文件里，**没有** PPL / MMLU / GSM8K / HumanEval / KL-vs-size 之类产物。
- `speedcheck.py` 的 docstring：*"Measures **decode speed** on code and on prose, **prefill speed** on a 7k-token prompt, and how fast a follow-up question … starts answering. Thinking is off for these requests so the timings measure the runtime, not the model's reasoning."* ⇒ **只测速度，不测质量**。
- 模型卡自述：*"This is a work in progress and **the numbers here are not final**."*

⇒ **⇒ 2-bit trellis 在 27B 上的任务能力，目前没有任何公开的一手测量。** 这正是本轮 Task 3（质量门槛）要解决的问题。

---

## 3. Encoder 到底存不存在？—— 逐层拆开

| 层 | 存在？ | 公开？ | 证据 |
|---|---|---|---|
| **trellis 码的"拟合/编码"过程** | ✅ **存在** | ❌ **不公开** | `COMPOSE_QTIP_FITS` / `K3` / `K4` / `E8P` / `I4` 等 fit 产物路径；`encoding.json` 的 `"type": "hanashi"` |
| **打包器（bit packing）** | ✅ 存在 | ⚠️ **通用件公开，但不是 trellis 编码器** | `lalamo/compressed/utils/packing.py`（main 分支）只有 `pack_uint_to_uint8` / `unpack_uint8_to_uint` 这类通用位打包 |
| **解码器（trellis → 权重）** | ✅ 存在 | ✅ **公开** | uzu `quantized/trellis.h`、`metal/kernel/activation_transform/trellis.metal`、CPU 参考 |
| **反向还原/证书脚本** | ✅ 存在（`recover_p8zm_codes.py`） | ❌ **不公开** | `physical_package_report.json` 引用它；**GitHub 代码搜索 0 命中** |
| **uzu 侧的"编码器"** | ❌ **不存在** | — | `transformed_linear.rs` 是 `LinearMatmul::load` + 运行时装配；**注意术语陷阱**：uzu 的 `encode/encode_into` 指的是"录制 GPU 命令"，不是量化 |

> **⇒ 一句话：第三方目前只能消费 Trellis 权重，不能生产。** 这不是"没人做过"，而是"**做了但没放出来**"。

### 3.1 ⚠️ 必须同时记录的一条反证：**PR #900 的 diff 目前编译不过**

WorkBuddy 的 R4 上游审查（`research/agent-reports/workbuddy/2026-10-r4-upstream-900.md`）给出了**他们本机 CI 日志**的一手证据（macOS job `112769920094`，`[E0]`）：

```
Error: cannot compile .../backends/metal/kernel/matmul/gemm/gemm_trellis.metal
Caused by:
    0: cannot generate bindings for GemmTrellis
    1: rust expression `TrellisGemmLayout<GEMM_TILING, USE_MXU>::B_SHARED_SIZE` in kernel `GemmTrellis` cannot be parsed
    2: unexpected token
```

读法：uzu 用 Rust 过程宏给 Metal kernel 生成绑定，**宏解析器不认识嵌套模板实参里的 `::` 表达式**；失败的 `B_SHARED_SIZE` 正是 PR #900 为非 MXU 路径新增的 threadgroup 数组长度。
⇒ **"拆掉 gate 之后新写的那条路，一次都没编译过。"**

### 3.2 ★ 两件事必须分开陈述（否则会得出矛盾结论）

| 事实 | 等级 | 出处 |
|---|---|---|
| **PR #900 的 diff 删掉了 trellis→MXU 的强制门** | `[P]` | PR diff（我核） |
| **PR #900 的 diff 在 macOS 上编译失败** | `[E0]` | WorkBuddy 本机 CI 日志（我未复核原始日志） |
| **已发布的预编译 `uzu/uzu` 在 M1–M4 上"run it correctly"（base M4 ≈4 tok/s）** | `[O]` | 模型卡自报（我核） |

⇒ **⇒ 两者并不矛盾**：**发布版里确实有一条能用的 M1–M4 路径，但它不是 PR #900 的当前 diff 产出的**（模型卡让用户自建时指向的是 **`ryan/qtip-s-agent` 分支**，不是 PR 分支）。
⇒ **⇒ 这正是 Issue #6 要求"明确区分发布版 / PR / 源码 / 训练结果"的地方**：
> **"发布版"：能跑，但来源分支未公开、速度慢。**
> **"PR"：意图正确，但编译不通过。**
> **"源码"：可读，可判断结构，不能判断能否运行。**
>
> **⇒ 因此我 R4 初稿里"PR #900 移除门控 ⇒ M4 可用"的说法不完整，应改为**：*门控在 PR 里被移除，但该 PR 不能编译；能跑的 M1–M4 路径存在于已发布二进制中，来源分支未公开。*

---

## 4. 性能事实：这一族的"同一 GPU、不同结果"

### 4.1 Apple Silicon（模型卡自报）`[O]`

| 芯片 | decode | prefill | 备注 |
|---|---|---|---|
| **M4（基础版）** | **≈4 tokens/s** | ≈13 tokens/s | *"M1 to M4 run it correctly but more slowly: about 4 tokens/s on a base M4"* |
| **M5 Pro** | **52 tok/s（代码）/ 17 tok/s（散文）** | 440 tok/s | *"it prints 52 tokens/s decoding code (17 on prose), 440 tokens/s prefill"* |

补充：*"Runtimes downloaded before **September 25** printed gibberish on M1 to M4"* ⇒ **M1–M4 路径曾有 bug，9 月 25 日后才正常**。

### 4.2 NVIDIA RTX 3090（模型卡自报）`[O]`

> *"On an RTX 3090 it answers one chat at **86 (prose) to 141 (code) tokens/s with speculative decoding**, 135 tokens/s while reasoning, serves 620 tokens/s across 64 parallel requests, and has room for about 170k tokens of context."*
> *"**8.2 GiB of GPU memory**, so a 12 GB card runs it and a 16 GB card has room for 54k tokens of context with speculative decoding."*

### 4.3 ★ 与我 R2/R3 引用过的 escha-port 对比（同款 RTX 3090）

| 来源 | 格式 | 栈 | 2-bit decode tok/s | 4-bit decode tok/s |
|---|---|---|---|---|
| **escha-port**（R2 §1.1） | ESCHAM @ 2.469 bpw | llama.cpp 第三方移植 | **34.19** | Q4_K_XL **41.40** |
| **Mirai S**（本轮） | QTIP-Gaussian @ 2.4 bpw | vLLM + `mirai_s` 插件 | **86–141（带投机解码）** / 135（推理中） | — |

⇒ **★ 必须更正我 R2 §1.1 的一般化**：*"2-bit = 更省内存所以更快 这个直觉在唯一一次真机对比里是错的"* —— **这句作为一般结论是错的**。在同款 RTX 3090 上，一个**写对了的** 2-bit trellis 解码器比 Q4_K_XL 快 **2.1–3.4×**；escha-port 的"更慢"是**那一次移植实现的质量问题**（其查表产生 13× 于权重的 L2 流量），**不是 2-bit 的性质**。
⇒ **⚠️ 口径限制**：86–141 tok/s **含投机解码**，与 escha-port 的裸 decode 不可直接同口径比较；"135 tok/s while reasoning"是否含投机解码**模型卡未说明** `[未核实]`。

### 4.4 ★★ 而在 **M4 Pro（本项目目标机）** 上，它其实**不划算**

| 对比 | decode tok/s | 来源 |
|---|---|---|
| **Mirai S 2-bit（uzu/Mirai 路径）** | **≈4**（base M4） | 模型卡自报 `[O]` |
| **GSQ-RCO Q2_0 2.40 bpw** | **93.79** | R1 记录的模型卡位宽扫描 `[O]`（**本轮未重新核对其硬件口径** `[未核实]`） |

⇒ **≈23× 的差距，且完全来自解码器/内核实现质量，不是格式本身。**
⇒ **⇒ 这是本轮对项目最有决策价值的一条**：**在 Apple Silicon 上，"2-bit"这个标签本身几乎不含信息量；决定成败的是解码器能否把位流直接化算成权重。** 与我在 R2 §7.2 "格式比位宽更重要" 的结论**同向**，但现在有了**同一芯片族的 23× 量级对照**。

---

## 5. ⚠️ 本轮推翻的我 R3 结论（必须在 R4 记录，R3 原报告不动）

| R3 位置 | R3 原表述 | **R4 判定** | 依据 |
|---|---|---|---|
| R3 §2.2 | "**它不是 QTIP 系的码**：不是'向量量化 + 码本'，而是'状态链 + 哈希合成 + 共享秩-4 基'" | **✘ 家族归属说错了**。格式**就是 QTIP 的 trellis 码**（MSB-first + 每行折叠 f32 scale + 五浮点码本）。哈希合成是 **QTIP-Gaussian 变体**的实现手段（`QtipGaussianSpec`、`qtip_gaussian.py`），不是另一个族 | lalamo PR #364 正文；uzu `config/weight_matrix/qtip_gaussian.rs` `[P]` |
| R3 §2.5–§2.6 | "**在 M4 Pro 上，uzu 的 Metal Trellis GEMM 会以 `MxuUnavailable` 失败**"（基于 commit `2968097a` 的 main 分支） | **✘ 对已发布 runtime 不成立**。模型卡明写 **"M1 to M4 run it correctly"**（base M4 ≈4 tok/s） | 模型卡 `[O]` |
| R3 §2.6 的"门控"解释 | "M4 Pro 被 `supports_mxu` 策略门拦下" | **△ 机制对、结论要改写**：该门**确实存在且确实生效**（main 分支），但**已发布二进制里有一条能绕过它的路径**；PR #900 虽删掉该门，**其 diff 编译不过**。⇒ 正确表述是"**门在源码里；发布版绕过了它；PR 试图通用化它但没编译过**" | §3.1–§3.2 |
| R3 §2.7 / §1 | "**Metal 上没有 trellis 内核 ⇒ 2-bit 在消费级不可用**" | **△ 需要按载体分述**：**Metal 上现在有了**（uzu，且随模型仓发布预编译二进制）；**真正的问题变成"M4 上只有 4 tok/s"** | §4 |
| R2 §1.1 | "**'2-bit = 更省内存所以更快'这个直觉……是错的**"（一般化） | **✘ 一般化错误**。同款 RTX 3090 上，写对了的 2-bit 解码器比 Q4_K_XL 快 2.1–3.4×；"更慢"是那次移植的性质 | §4.3 |
| R3 §2.2 的"码率" | "V4_T8 = 2.125 bpw / V2_T6 ≈3.0 / V2_T4 ≈2.0" | **✅ 成立**，且被 `physical_package_report.json` 的布局分布独立佐证（加权 2.433 bpw ↔ 模型卡 2.4） | §2.2 |

**R3 仍然成立、未被本轮推翻的部分**：Metal 端 trellis **只到"源码存在"**（无产品路径的论证已被 §2 推翻，但"无独立质量验证"成立）；**没有 trellis GEMV**；**K 必须被 64 整除**；**"省符号位"的说辞需要严格化**（见 `2026-10-r4-quality-gates.md` 之外的 `2026-10-r4-encoding-status.md` §6 与质量门槛文档 §2）。

---

## 6. 仍然是缺口的（未核实清单）

| 项 | 状态 |
|---|---|
| **这个 2-bit 模型的 PPL / 下游任务分数** | **完全缺失**。公开产物只有解码器数值一致性（99.956 % 逐位相同）与字节账 |
| **trellis 编码器的公开实现** | **不公开**。`recover_p8zm_codes.py` GitHub 0 命中；fit 产物路径在私有 `/data/ry2009/...` |
| **uzu PR #899 / #900 / lalamo PR #364 何时合并、是否会合并** | **未知**（三者均 OPEN） |
| **uzu PR #925 "packing update"**（lalamo PR #364 点名依赖） | **本轮未打开** `[未核实]` |
| **`ryan/qtip-s-agent` 分支**（模型卡给出的自建分支） | **本轮未打开** `[未核实]` |
| **vLLM 插件 `mirai_s` 的源码** | **只见到 4 个 `.whl` 二进制**；**源码是否公开未核实** `[未核实]` |
| **`vllm/trellis.mirai` 的实际布局** | **未下载**（禁止下载模型）；格式由 `physical_package_report.json` 间接给出 |
| **M4 Pro（非 base M4）上的确切 tok/s** | **未核实**：模型卡只给 "a base M4" |
| **M1–M4 路径 2026-09-25 修复的具体内容** | **未核实**（PR #900 是候选，但未合并） |
| **escha-port 是否提供公开 encoder** | **未核实**：README 拉取返回空（14 B），本轮未展开 |
| **"Mirai Medium"（`-M`）的具体格式** | **只核实到标签是 `4-bit`**；其 `assets/kl-vs-size.svg` 未下载 |
| **QAT 的训练成本、数据、是否公开** | **未公开**：`provenance.json` 只到 fit 产物路径，无训练配方 |
| **`hanashi` 编码器是否公开** | **未核实**：`encoding.json` 只给出标识符 `hanashi` |

---

## 7. 引用清单

- uzu PR #899：https://github.com/trymirai/uzu/pull/899（OPEN）
- uzu PR #900：https://github.com/trymirai/uzu/pull/900（OPEN）
- lalamo PR #364：https://github.com/trymirai/lalamo/pull/364（OPEN）
- uzu `main` @ `2968097ab20c66acbf1797b292c29bf18f71882c`：
  `crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/gemm_trellis.metal` ·
  `.../gemm/trellis.rs` · `.../gemm/common/quantized/trellis.h` · `.../gemm/selection.rs` ·
  `.../gemm/kernel.rs` · `.../gemv/kernel.rs` · `.../matmul_metal_kernel.rs` ·
  `.../metal/metal_extensions/device_extensions.rs` · `.../backends/cpu/kernel/activation_transform/trellis.rs` ·
  `.../config/weight_matrix/qtip_gaussian.rs`（PR #899 新增）
- 模型：https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental
  （README.md / encoding.json / provenance.json / loader_validation.json / physical_package_report.json / speedcheck.py / config.json）
- 对照模型（4-bit）：https://huggingface.co/trymirai/Qwen3.8-27B-M
- lalamo main：`lalamo/compressed/utils/packing.py`
- escha-port：https://github.com/YanissAmz/escha-port
- 同族其它：https://huggingface.co/trymirai/Qwen3.8-27B-XS-experimental、Qwen3.8-Flash-Next-S-experimental、Qwen3.8-Flash-Next-S-small-experimental、Qwen3.8-Flash-Next-S-strata-experimental
- GitHub 代码搜索：`recover_p8zm_codes`（0 命中）、`MatmulB::Trellis`（5）、`is_trellis`（8）
