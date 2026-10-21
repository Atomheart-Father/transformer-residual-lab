# 任务 B：消费级硬件上的 LLM 推理系统与底层执行审计

**版本** v1 · 2026-10-21 · 供三方交叉审查（WorkBuddy / GPT / 用户）
**范围**：Apple Silicon（M4 Pro 16-core GPU / 24 GB / macOS 26.6.2 / MLX 0.31.2 / mlx-lm 0.31.3）为主要参考，结论按「可移植 / 不可移植」分层给出。
**本轮边界**：只做文献、源码、既有测试记录的审计。**没有新增 GPU 基准**、没有下载模型、没有修改 MLX / mlx-lm / llama.cpp 任何文件、没有 commit / push。文中每个数字都标注证据等级。

---

## 0. 摘要（一页）

### 0.1 七条结论

1. **「多个 Block 理论上可并行」在这台机器上不兑现，原因不是锁，而是「字节数不变 + M=1 时每个 kernel 太小」。** 冻结 DAG 把关键路径从 28 降到 7，prefill 只买到 5–6%（且无置信区间），decode 买到 0（最好 3.7%，落在噪声内）。真正的约束是：decode 每一步必须把 880.8 MB 的 Block 权重从统一内存读一遍，实测折合 **124.6 GB/s，仅为 273 GB/s 峰值的 46%**；DAG 不改变这个字节数。
2. **decode 的 46% 带宽利用率是本机最大的、可量化的低效点。** 它同时说明两件事：设备没有饱和（所以「GPU 忙」是错的），而剩余空间也不能靠「更多并行 Block」拿到（所以「再并行就好了」也是错的）。要拿这 54%，只能改 GEMV 的访存形态或提高 M（批 / 投机解码），不能改依赖图。
3. **mlx-lm 的能力组合是互斥的，不是叠加的。** 源码级确认：KV 量化 ↔ 连续批处理互斥（`_make_cache` 对 `QuantizedKVCache` 直接 `raise`）、KV 量化 ↔ 滑动窗口互斥（`RotatingKVCache.to_quantized` 是 `NotImplementedError`）、连续批处理 ↔ 投机解码互斥（`server.py:371` `is_batchable = draft_model is None`）。任何「四个都要」的方案必须先补这三块拼图。
4. **量化 KV 在 mlx-lm 里不是「免费的 4× 省内存」**：它每步对 1 个 token 重新 `mx.quantize`（28 层 × K/V = 56 次小 kernel），并且绕开了融合 SDPA（走两次 `quantized_matmul`，物化 score 矩阵）。长上下文省的是带宽，短上下文亏的是调度。
5. **唯一能真正减少「每 token 字节数」的机制是投机解码**，因此它是 Apple Silicon 上唯一有结构性收益的 decode 加速路径。用本机实测值做盈亏平衡：draft 与 target 同尺寸时 100% 接受率才打平；draft 小 10× 且接受率 60–70% 时约 1.8–2.1×。M5 之后这一点更重要（论文实测 M5 对 decode 只有 1.02–1.33×，且明确归因于调度与融合，不是张量核）。
6. **本项目的延迟证据存在一个未解释的 1.9× 跨轮差异**：同一段代码（28 blocks / 257 token / 单次 eval）在 Stage 2.0 记录 61.3 ms，在 Stage 2.1 记录 116.9 ms。**在定位它之前，所有绝对延迟数字只能在同一进程内做相对比较，不能跨报告引用。** 另外 A2 的「1.46× 重叠上限」的 k=1 基线被污染（含 k 次 embed + k 次同步 eval，1.053 ms，比独立测得的单 Block 0.762 ms 高 38%），按 A1 口径重算是 **1.06×**，不是 1.46×。
7. **五个值得做的低层改造，按「证据强度 × 成本」排序**：① decode GEMV 带宽/占用率；② KV 分页 + 融合 quantize-append；③ 小 draft 投机解码；④ radix 前缀缓存 + 零拷贝复用；⑤ prefill 侧算子融合（SwiGLU/RoPE/残差）。**不建议**继续在「深度压缩 / 稀疏 DAG」上投入——这条轴已被两轮实测否证，且机制上无法影响字节数。

### 0.2 三个最重要的数字

| 量 | 值 | 口径 |
|---|---|---|
| decode 有效带宽 | **124.6 GB/s**（273 GB/s 的 46%） | 880.8 MB ÷ 7.07 ms，28 blocks，batch 1，257-token KV [E1] |
| prefill 有效算力 | **2.06 TFLOP/s**（单 Block 孤立测 3.37–3.49） | 241 GFLOPs ÷ 116.9 ms [E1] |
| KV 成本 | **112 KiB / token**（bf16） | 2 × 8 KV heads × 128 dim × 2 B × 28 层 [E4] |

### 0.3 三个不能声称的东西

- **不能说「DAG 能省延迟」**：prefill 5–6% 无 CI，decode 0。
- **不能说「GPU 已饱和」**：46% 带宽、单 Block 约 40% 算力（端到端只有 2.06 TFLOP/s，更低），都未饱和。
- **不能说「KV 量化 = 4× 更快」**：它省带宽、加调度；在 257 token 这种短上下文里净收益可能为负（未测）。

---

## 1. 方法与证据分级

### 1.1 证据等级

| 级 | 含义 | 本轮来源 |
|---|---|---|
| **E0** | 本机新测（本轮执行） | 仅 3 个无模型的小实验（`copy.deepcopy` 语义、COW 写、`mx.device_info`），不涉及模型推理 |
| **E1** | 本仓库历史实测，口径已核对 | `results/stage2_0/bench`、`results/stage2_1/bench_a*`、`decode_check` |
| **E1?** | 本仓库历史实测，**口径有疑点** | Stage 2.0 与 2.1 的 prefill 绝对值；A2 的 k=1 基线 |
| **E2** | 源码阅读（版本已固定） | mlx-lm 0.31.3 / MLX 0.31.2 / llama.cpp 目录结构 |
| **E3** | 公开文献 / 官方文档 / 仓库 README | 见 §10 |
| **E4** | 由 E1/E2/E3 算术推导 | 全部标 `[推算]`，给出公式 |

### 1.2 硬件与运行时事实（E0，`mx.device_info()` 实测）

| 项 | 值 | 含义 |
|---|---|---|
| GPU | Apple M4 Pro，16 核，`applegpu_g16s` | 无张量核（NAX 是 M5 起） |
| 统一内存 | 25,769,803,776 B = 24.0 GiB | 与其它开发工作共享 |
| `max_recommended_working_set_size` | 19,069,665,280 B = **17.76 GiB** | Metal 建议工作集上限（≈75% RAM） |
| `max_buffer_length` | 14,302,248,960 B = **13.32 GiB** | **单个 buffer 上限**，长上下文 KV 可能撞到 |
| `resource_limit` | 499,000 | 资源（buffer/纹理）数量上限 |
| 内存带宽 | 273 GB/s（Apple 规格） | M4 Pro 全系；M4 基础款为 120 GB/s [E3] |
| macOS | 26.6.2 | Metal 4 可用 |
| MLX / mlx-lm | 0.31.2 / 0.31.3 | 有 `mx.fast.metal_kernel`、`mx.async_eval`、NAX 内核（M5 专用） |

**关键推论**：24 GB 机器上「可用」不等于 24 GB。mlx-lm 的 `wired_limit()` 会把 wired 上限设成 `max_recommended_working_set_size`（17.76 GiB）——这是**它自己**的上限，不是给其它进程留的余量。用户要求「给其它开发工作留内存」，则实际预算应显式压到 12–14 GiB（见 §3.3）。

### 1.3 被测模型（固定 revision）

`mlx-community/Qwen3-0.6B-Base-bf16` @ `e8e73ff3…`：28 层、hidden 1024、16 Q / 8 KV heads、head_dim 128、FFN 3072、vocab 151936、tied embedding、BF16、磁盘 1.1 GiB、参数 5.96 亿。**这是机制实验载体，不是性能代表**（见 §8）。

---

## 2. 交付 A：推理引擎技术矩阵

### 2.1 主表

| 引擎 | 执行模型 | 权重量化 | KV / 调度 | Apple 可用性 | 可直接借鉴 | 不可移植 | 关键代码 | 证据 |
|---|---|---|---|---|---|---|---|---|
| **MLX / mlx-lm** | 惰性图 + 单 stream，`mx.eval`/`async_eval`，可选 `mx.compile` | affine group-64 2/4/6/8 bit；`mxfp4`/`nvfp4`/`mxfp8`（含激活量化） | `KVCache`（256 步倍增）、`QuantizedKVCache`、`RotatingKVCache`、`BatchKVCache`、`ArraysCache`、`CacheList`；trie + LRU 前缀缓存；`BatchGenerator` 连续批处理 | 原生 | 全部（本机就是它） | — | `mlx_lm/models/cache.py`、`generate.py`、`models/base.py` | E2 |
| **llama.cpp / ggml** | C 图执行，Metal 后端按 op 编码进 command buffer；`-ngl` 分层卸载 | Q2–Q8 block-wise（`q4_0`…）、IQ 系列、K-quants | 每序列独立 KV，`--cache-type-k/v` 量化、`-fa` 融合注意力、`--n-gpu-layers` 分层、slot 级批处理 | 一等公民 | GEMV/GEMM 双内核、`fa_*`/`fa_vec_*` 拆分、量化布局、CPU/GPU 分层 | 调度层弱于 vLLM（无分页/无 radix） | `ggml/src/ggml-metal/ggml-metal-ops.cpp`、`kernels/mul_mv.metal`、`mul_mm.metal`、`fa_*.metal` | E2/E3 |
| **ExLlamaV2** | Python + CUDA C++ 扩展，FlashAttention 2.5.7+ 分页 | **EXL2** 2–8 bit 混合位宽、按列多级 | 动态 generator、KV 去重、Q4 cache（**不支持 FP8 cache**） | **不可用**（CUDA only） | EXL2 的「按误差分配位宽」思想；Q4 cache | 全部 kernel | 仓库已 **archived**，转向 ExLlamaV3 | E3 |
| **ExLlamaV3** | Triton + CUDA，连续动态批处理 | **EXL3 = QTIP 简化变体**，2–8 bit cache 量化 | 投机解码（SAM/Ngram）、MoE CPU offload（AVX2/512） | **不可用**（CUDA 优先，ROCm 在开发） | trellis 量化、cache 量化、offload 策略 | 全部 kernel | `exllamav3/`、`convert.py` | E3 |
| **vLLM** | V1 引擎：调度器 + KV 管理器 + Worker；CUDA/HIP graph、torch.compile | FP8/MXFP8/MXFP4/NVFP4/INT8/INT4/GPTQ/AWQ/GGUF | **PagedAttention**（块表）、continuous batching、chunked prefill、prefix caching、PD 分离 | 通过 `vllm-metal` 插件（社区，MLX 后端） | 分页 KV、调度、前缀缓存、投机（n-gram/suffix/EAGLE/DFlash） | FlashAttention/FlashInfer/CUTLASS/TRTLLM-GEN 内核、CUDA graph | `vllm/v1/`、`csrc/` | E3 |
| **SGLang** | 调度器 + RadixAttention；**已官方支持 Apple Silicon（MLX 运行时）** | 复用 mlx-community 4/8 bit，或 `mlx_q4` 在线量化 | Radix 前缀树、continuous batching、overlap scheduling（用 `mx.async_eval`） | **可用**（`SGLANG_USE_MLX=1`，需 `--disable-cuda-graph`） | **调度层 + MLX 后端的现成组合**；自定义 Metal RoPE、融合 SwiGLU 的开关 | CUDA 专属 kernel、CUDA graph | `docs/hardware-platforms/apple_metal`；`sgl-kernel` 的 Metal 部分 | E3 |
| **TensorRT-LLM** | PyTorch 原生建模 + C++ runtime；按架构分代（blackwell/rubin） | FP8/FP4、blockscaled GEMM、NVFP4 | in-flight batching、KV cache manager V2、PD 分离、宽 EP | **不可用** | 调度/并行策略、KV 保留策略 | 全部（CUDA/TensorRT/NCCL/NVLink） | `cpp/tensorrt_llm/batch_manager/`、`tensorrt_llm/_torch/kernels/` | E3 |
| **MLC-LLM** | TVM/Relax 编译到目标后端（Metal/Vulkan/CUDA/WebGPU/OpenCL） | q4f16 等编译期量化 | 编译期生成 KV 布局；MLCEngine 统一运行时 | **可用**（Metal） | 编译期特化、跨后端代码生成 | 需要重新编译模型；对动态 shape 不友好 | `mlc-llm/`、TVM Unity | E3 |
| **ExecuTorch** | `torch.export` → `.pte`，delegate 分区 | 量化导出（int4/int8，配方随 backend） | 有状态执行（多会话、投机解码在 LLM 能力表内） | **可用**：Core ML / XNNPACK 稳定；MLX、Metal 实验性 | 状态化模型接口、AOT 内存规划 | 实验性后端不建议生产 | `backends/apple/`、`backends/coreml/` | E3 |
| **ONNX Runtime（CoreML EP）** | 图分区 → Core ML（mlprogram/NN） | 依赖 Core ML 侧量化 | 动态 shape 可用但**显著降速**；MatMul 的 B 必须常量 → **KV cache 受限** | 可用 | 图分区 + EP 回退思路 | 不适合自回归 LLM 主路径 | `CoreML-ExecutionProvider` 文档 | E3 |
| **Core ML（原生）** | `mlprogram` + `MLState`（iOS18/macOS15+） | palettization / int4 / fp16 | **KV cache 作为 state buffer**（固定 shape，`make_state()`） | 可用 | 状态化 KV、ANE 路径 | 固定 shape、算子子集、编译慢 | coremltools `stateful-models` 指南 | E3 |
| **手写 Metal / 纯 C 系** | 直接 Metal 4 + MPP/TensorOps；或 C/C++ | 自定义 packed 布局 | 自管 | 可用 | **BaseRT**（M5 NAX 路径）、MetalFlashAttention、tinygrad、candle、mistral.rs、llm.c | — | 见 §2.3 | E3 |

### 2.2 分层判断：什么能借，什么不能

把引擎拆成三层，结论完全不同：

| 层 | 代表实现 | 可移植性 | 说明 |
|---|---|---|---|
| **控制面**（请求调度、批、前缀、抢占、PD 分离） | vLLM V1、SGLang、TRT-LLM | **高**：算法与设备无关 | 但需要一个「KV 块管理器 + 可批处理 attention」的底座；mlx-lm 目前两者都缺（§5） |
| **内存面**（分页 KV、块表、量化 KV、前缀共享） | PagedAttention、RadixAttention | **中**：数据结构可移植，kernel 要重写 | 这是 Apple 侧最值得抄的一层 |
| **计算面**（GEMM/GEMV/FA/MoE kernel） | CUTLASS、FlashAttention、TRTLLM-GEN、Marlin | **低**：与 warp 形状、张量核、TMA、FP8 强绑定 | Metal 的 simdgroup 是 32 宽、无 TMA、M4 无矩阵核；只能借「形状与分相」，不能借代码 |

**一句话**：**能借的是「怎么管内存、怎么排请求」，不能借的是「怎么算矩阵」。**

### 2.3 值得单独记的独立项目

| 项目 | 价值 | 与 MLX/llama.cpp 的关系 | 证据 |
|---|---|---|---|
| **BaseRT**（arXiv 2607.19438） | M5 上把 prefill 的 QK⊤/PV 与 MoE GEMM 换成 Metal 4 `matmul2d`（MPP），prefill 相对 MLX 最高 3.9×、相对 llama.cpp 6.4×；**decode 只有 1.02–1.33×，且作者明确归因于调度/融合而非张量核** | 与 MLX 同为 Metal 路线，但绕开 MLX 直接写 Metal 4 | E3 |
| **MetalFlashAttention** | `simdgroup_async_copy` + 手写 FA，是 Metal 上 FA 的参考实现 | llama.cpp/MLX 的 FA 都可对照 | E3 |
| **tinygrad** | Metal 后端 + 调度器搜索，可看「图级调度」的另一条路 | 与 MLX 同类但更激进 | E3 |
| **candle / mistral.rs** | Rust，Metal + CUDA 双后端，代码干净 | 适合做跨后端对照 | E3 |
| **llm.c / llama2.c** | 纯 C，最小可读实现 | 教学与正确性对照，不是性能参考 | E3 |
| **KTransformers / PowerInfer** | CPU/GPU 混合、专家/热神经元卸载 | 24 GB 跑 100B 级 MoE 的唯一现实路径 | E3 |
| **SwiftLM** | MLX Swift + SSD 流式 + TurboQuant KV | 移动端/大 MoE 的工程参考 | E3 |

### 2.4 逐项目核查表（按第三节要求的九个字段）

| 项目 | 当前可运行 | 真实设备基准 | 收益主要来自 | 与 MLX / llama.cpp 的关系 | 需要的硬件特性 | 移植到 Metal 成本 | 已知 bug / 缺口 |
|---|---|---|---|---|---|---|---|
| **mlx-lm 0.31.3** | ✅ 本机 | ✅ 本仓库 E1 | 实现（惰性图 + 融合 SDPA） | 本体 | 无特殊 | — | §5 的 D1–D11 |
| **llama.cpp（Metal）** | ✅ | ✅ 社区大量 | 实现（GEMV/FA 专用 kernel + 量化布局） | 独立；与 MLX 同走 Metal | 无（simdgroup 即可） | — | 调度层弱：无分页、无 radix；长上下文 KV 管理粗 |
| **ExLlamaV3** | ✅（CUDA） | ✅ 作者基准（4090/3090Ti） | 算法（EXL3/QTIP 位宽分配）+ 实现（Triton） | 与 MLX 无关 | CUDA 12.4+，Triton | **不可移植**（Triton 无 Metal 后端） | Metal 无路径；ROCm 未正式声明 |
| **vLLM + vllm-metal** | ⚠️ 插件实验性 | ⚠️ 插件侧数据少 | 算法（分页/调度） | 控制面在 vLLM，计算下沉 MLX | Metal + MLX 0.32+ | 已有人做（插件） | 分页 attention 标 experimental；默认内存分数 0.9 |
| **SGLang（MLX 后端）** | ✅（`SGLANG_USE_MLX=1`） | ⚠️ 文档未给绝对数 | 算法（Radix）+ 实现（overlap schedule、融合开关） | 直接复用 mlx-lm 的模型与量化 | Metal + MLX 0.32+ + 完整 Xcode | 已有人做 | 需 `--disable-cuda-graph`；自定义 kernel 默认关闭 |
| **TensorRT-LLM** | ✅（NVIDIA） | ✅ | 实现（CUTLASS/TRTLLM-GEN）+ 算法（in-flight batching） | 无关 | SM90+/SM100，FP8/FP4 | **不可移植** | 全栈绑定 NVIDIA |
| **MLC-LLM** | ✅ | ⚠️ 分散 | 算法（编译期特化） | 独立 | Metal / Vulkan / CUDA | 已有 Metal 路径，但需重编译模型 | 动态 shape 不友好；量化方案与 HF 生态脱节 |
| **ExecuTorch + Core ML** | ✅（Core ML 后端） | ⚠️ 有 dashboard | 实现（AOT 内存规划、ANE） | 无关 | ANE / Core ML | 已有 | 固定 shape、算子子集、编译慢；MLX/Metal 后端仍是 experimental |
| **ONNX Runtime CoreML EP** | ✅ | ⚠️ | 实现（图分区） | 无关 | ANE/GPU | 已有 | `MatMul` 的 B 必须常量 → 自回归 KV 受限；动态 shape 降速 |
| **BaseRT（M5）** | ✅（M5） | ✅ 论文 | 实现（Metal 4 MPP matmul2d） | 与 MLX 竞争 | **M5 NAX + Metal 4** | 只能 M5+ | 单用户、无连续批处理、无多卡 |

**读法**：收益来自「算法」的项目（vLLM/SGLang/ExLlamaV3 的量化分配）可以抄设计；收益来自「实现」的项目（llama.cpp/BaseRT 的 kernel）只能抄形状与分相。**没有一个项目的 kernel 能在 Metal 上原样复用。**

---

## 3. 交付 B：Apple Silicon 的计算、内存与调度瓶颈

### 3.1 两个阶段的账本（0.6B，实测口径）

先给公式，再给数。**decode 每 token 的字节数**（与 batch 无关，除非批处理）：

```
B_decode = Σ_layers (W_block) + B_kv(C) + B_head
         = 880.8 MB + C × 112 KiB + 311.2 MB   (bf16, C = 上下文长度)
```

**prefill 的 FLOPs**（C 个 token）：

```
F_prefill = C × 880.6 MFLOPs/token + 0.5 × 8192 × C² × 28   (causal 折半)
```

实测与推算：

| 量 | 实测 | 推算 | 备注 |
|---|---|---|---|
| 单 Block 权重 | 31.46 MB | — | 15,728,640 参数 × 2 B |
| 28 Block 权重 | 880.8 MB | — | 每 token 必读 |
| LM head（tied） | 311.2 MB | — | 每 token 必读；**占 0.6B 的 26%** |
| decode 一步 | **7.07 ms**（28 blocks，257-token KV，lazy） | 880.8 MB ÷ 7.07 ms = **124.6 GB/s** | 46% of 273 GB/s |
| decode 一步（含 LM head，估算） | 未单测 | ≈ 8.6–9.6 ms → 124–139 GB/s | `[推算]` |
| prefill 257 token | **116.9 ms**（lazy）/ 114.1（compiled） | 241 GFLOPs ÷ 116.9 ms = **2.06 TFLOP/s** | 单 Block 孤立 3.37–3.49 |
| 单 Block 257 token | 2.40–2.48 ms | 3.37–3.49 TFLOP/s | A7，四层采样一致 |
| 单 Block n≤16 | 0.47–0.68 ms（**平坦**） | 边际成本在 n<16 时测不出 | A1；见 §3.8 对「0.558 ms 固定开销」的质疑 |

**这张表最重要的一行是 124.6 GB/s。** 它说明 decode 既不是算力问题（M=1 时算术强度 0.5 FLOP/B），也不是「GPU 被锁住」，而是**访存效率只有峰值的一半**。

### 3.2 为什么多个 Block 可并行，却不比串行快

用户要求「区分数学依赖、内存带宽、kernel 竞争、执行队列、同步和 MLX 实现，不允许简单归因为 GPU 有锁」。逐条判定：

| 候选原因 | 判定 | 依据 | 残余不确定性 |
|---|---|---|---|
| **数学依赖** | **不是**（已被 DAG 消除） | full-g4 把关键路径 28→7，prefill 只快 5–6%，decode 0 [E1] | 无 |
| **总工作量不变** | **是必要背景，但不是充分解释** | 28 个 Block 全执行、权重全读；DAG 只改顺序 [E2] | 无 |
| **内存带宽** | **是主因之一（46%）** | 880.8 MB / 7.07 ms = 124.6 GB/s；峰值 273 | 「为什么只有 46%」未定 |
| **kernel 竞争 / 占用率** | **是主因之二（最可能）** | M=1 时每个 kernel 只有 1 行输出，threadgroup 少、延迟无法隐藏；A2 在 n=64 时 8 个独立 Block 只快 6%（按 A1 口径重算） | 需要 profiler 或 M 扫描才能定量 |
| **执行队列 / 同步** | **部分**：eager 已排除，lazy 单流仍有屏障 | eager 14.4 ms vs lazy 7.07 ms（差 2×，说明每 Block 同步极贵）；但 lazy 仍慢于带宽下限 2.2× | in-place KV 写是否触发 `fence.metal` 级屏障，未验 |
| **MLX 实现** | **是**：merge 内核吃掉重叠收益 | window-g4（depth 22）比 serial 慢 12%；full-g4（depth 7）只快 5% | merge 单独计时未做 |
| **「GPU 有锁」** | **否** | A2 显示独立 Block 确有 1.25–1.41× 重叠（n=64），说明存在并发 | 该数值本身口径有疑（§3.8） |

**机制叙述（把上面几条串起来）**：

decode 一步 = 28 个 Block × 约 18 个 kernel ≈ **500 次 dispatch（估算，非实测）**，总字节 880.8 MB。若带宽打满，下限是 3.2 ms；实测 7.07 ms。差额 3.9 ms 只能来自两类东西：**(a) 每个 kernel 的启动/尾部延迟在 M=1 时无法被其它工作掩盖；(b) 张量太小，DRAM 的并发请求数不足以打满。** 这两者都随「同时有多少独立工作」变化——这正是 DAG 想提供的。但 DAG 提供的并发是**同层 Block**，而它们的输入要先 merge：每个 merge 点都是一次 fp32 加 + 一次 cast（g2 约 14 个 merge 点，g4 约 21 个），每次都是小 kernel，且落在关键路径上。于是「多出来的并发」和「多出来的 merge 调度」大致抵消，实测就是 0。

**为什么 eager 差 2×**：eager 在每个 Block 后 `mx.eval`，把「一次提交 500 个 kernel」拆成 28 次提交 + 28 次 CPU/GPU 同步。这证明**提交/同步的单位成本很高（约 0.25 ms/次）**，也解释了 A1 里「孤立单 Block 0.47–0.68 ms」为什么远高于链内 0.253 ms/Block：**孤立测量测的是「一次 eval 的成本 + 一个 Block」，不是「一个 Block 的成本」。**

**这条结论对项目的意义**：`critical path` 在本机是逻辑量，不是延迟量；想让深度变成延迟，必须先把「每 token 500 次 dispatch」压下来（融合），而融合的收益上限受 46% 带宽约束——**最多 2.2×，且拿不满**。

### 3.3 内存层级与真实预算

| 层级 | 容量/带宽 | 谁在用 | 风险 |
|---|---|---|---|
| 统一内存（LPDDR5X） | 24 GiB / 273 GB/s | 权重 + KV + 激活 + 其它进程 | 与开发工作共享 |
| Metal 建议工作集 | **17.76 GiB** | mlx-lm 默认把 wired 设到这里 | 会挤掉其它进程 |
| 单 buffer 上限 | **13.32 GiB** | 单个 KV/权重张量 | 128k 上下文的 KV 会撞上限 |
| GPU L2 / SLC | 未公开（M4 系约数十 MB） | 权重流式复用 | 31.5 MB/Block 勉强可缓存一层 |
| SSD | ~3–6 GB/s | 专家卸载、prompt cache 落盘 | 比 DRAM 慢 50–90× |

**建议的显式预算**（满足「给其它开发工作留内存」）：

```python
mx.set_wired_limit(12 * 1024**3)   # 12 GiB，而不是默认 17.76
mx.metal.set_cache_limit(2 * 1024**3)
```

`mlx_lm.generate.wired_limit()` 与 `BatchGenerator.__init__` 都会把上限拉到 17.76 GiB（源码 E2），**在共享机器上应当显式覆盖**。这是本审计给出的第一条可直接执行的改动，零风险。

### 3.4 调度：`mx.eval` / `async_eval` / stream / compile 的实际作用

| 机制 | 源码位置 | 作用 | 实测效果 |
|---|---|---|---|
| 惰性图 + 单次 `mx.eval` | 全仓库 | 把 500 次 dispatch 合成一次提交 | 相对 eager **2×**（7.07 vs 14.40） |
| `mx.async_eval` 预取下一步 | `generate.py:455` | CPU 侧提前编码下一步，GPU 不空转 | 未单独 A/B（建议做） |
| `mx.compile` | `stage21_bench` | 图级融合 + 减少 dispatch | prefill **2.4%**，decode **0%**（7.15 vs 7.07） |
| `mx.clear_cache()` 每 256 步 | `generate.py:467` | 释放 Metal 缓冲缓存 | 会引入周期性重新分配（未测抖动） |
| `mx.stream(...)` | `generate.py:399` | 生成放到独立 stream | 单请求无收益；多请求才有意义 |

**`mx.compile` 只有 2.4% 这件事很关键**：如果 decode/prefill 是 dispatch-bound，编译融合应当带来两位数收益。它没有 → **prefill 的 GPU 确实在忙（2.06 TFLOP/s 是真实占用，不是空转）**；decode 的瓶颈则更可能在「单 kernel 太小 + 带宽利用率」，而不是「提交次数」。

### 3.5 长上下文账本（0.6B，`[推算]`）

| 上下文 C | KV 大小 | decode 每步字节 | decode 时延 @125 GB/s | decode tok/s | prefill FLOPs（causal） | TTFT @2–4 TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4k | 448 MiB | 1.33 GB | 10.6 ms | 94 | 5.5 TF | 1.4–2.8 s |
| 8k | 896 MiB | 1.78 GB | 14.2 ms | 70 | 14.9 TF | 3.7–7.5 s |
| 32k | 3.5 GiB | 4.47 GB | 35.8 ms | 28 | 152 TF | 38–76 s |
| 128k | 14 GiB | 15.2 GB | 122 ms | 8 | 2.1 PF | 9–18 min |

（FLOPs 按附录 A 的 causal 公式计算；TTFT 按 2–4 TFLOP/s 区间，注意注意力 kernel 的效率未实测，区间可能偏乐观。）

三个直接结论：
1. **32k 之后 decode 由 KV 带宽主导**（KV 3.5 GiB > 权重 0.86 GiB），此时 KV 量化（4-bit → 1.75 GiB）是唯一有效手段。
2. **32k 的 prefill 由注意力的二次项主导**（152 TF 中 123 TF 是注意力），这才是 M5 张量核真正能帮上忙的地方（BaseRT 的 prefill 3.9× 就来自这里）。
3. **128k 在这台机器上不现实**：KV 14 GiB 已超单 buffer 上限（13.32 GiB），且远超 12 GiB 预算。

### 3.6 投机解码在 Apple Silicon 上的真实账

源码（E2）：`speculative_generate_step` 的 draft 阶段是 `for _ in range(num_draft): _step(draft_model, ...)`——**每个 draft token 一次完整前向**；验证阶段一次前向吃 `num_draft+1` 个 token，然后 `_rewind_cache` 按接受数 trim。

设：target 单步 `T_t`（7.07 ms），draft 单步 `T_d`，接受数 `a`（0..n），n 个 draft：

```
每轮时间 = n × T_d + T_t(n+1)
每轮产出 = a + 1 个 token
有效 token 时间 = (n·T_d + T_t(n+1)) / (a + 1)
```

| 场景 | T_d | n | 接受率 | 有效 ms/token | 相对 7.07 |
|---|---:|---:|---:|---:|---:|
| 同尺寸 draft | 7.07 | 2 | 100% | 7.07 | 1.00×（打平） |
| 同尺寸 draft | 7.07 | 2 | 60% | 8.8 | 0.80×（更差） |
| 1/10 尺寸 draft | 0.71 | 4 | 70% | 3.4 | **2.1×** |
| 1/10 尺寸 draft | 0.71 | 4 | 50% | 4.2 | 1.7× |
| 1/20 尺寸 draft | 0.35 | 4 | 50% | 3.8 | 1.9× |

关键点：**验证步的边际成本极低**（`T_t(n+1) ≈ T_t`，因为 decode 是带宽受限，多 4 个 token 只多 0.02 ms 的边际算力）。所以投机解码的收益完全由「draft 有多便宜 × 接受率多高」决定，**与 target 的深度无关**。这与 M5 论文的判断一致（decode 要靠投机把工作转成可批的矩阵乘）。

### 3.7 组合限制矩阵（源码级，E2）

| 组合 | 是否可行 | 证据 |
|---|---|---|
| KV 量化 + 单请求生成 | ✅ | `maybe_quantize_kv_cache`（`generate.py:299`） |
| KV 量化 + 投机解码 | ✅（缓存可 trim） | `QuantizedKVCache.is_trimmable()` 返回 True |
| KV 量化 + 连续批处理 | ❌ | `_make_cache`（`generate.py:838`）对 `QuantizedKVCache` 落到 `else: raise ValueError` |
| KV 量化 + 滑动窗口 | ❌ | `RotatingKVCache.to_quantized` → `NotImplementedError`（`cache.py:551`） |
| 连续批处理 + 投机解码 | ❌ | `server.py:371` `is_batchable = draft_model is None` |
| 连续批处理 + `RotatingKVCache(keep>0)` | ❌ | `_make_cache` 显式 raise |
| 前缀缓存（trie）+ 滑动窗口 | ⚠️ 部分 | `RotatingKVCache.is_trimmable()` 在 `offset >= max_size` 后为 False |
| 前缀缓存 + 连续批处理 | ✅ | `server.py:753` 先 `fetch_nearest_cache` 再 `BatchGenerator` |
| 投机解码 + 前缀缓存 | ✅ | `server.py:965` 同一路径 |

**这张表是「为什么不能把四个特性一起开」的完整答案**，也是 §7 机会 2 的直接依据。

#### 3.7.1 状态类型 × 引擎能力（模型结构决定状态管理）

| 模型结构 | 状态是什么 | mlx-lm 的载体 | 可批处理 | 可量化 | 可前缀复用 | 备注 |
|---|---|---|---|---|---|---|
| **MHA / GQA**（Qwen3、Llama） | 每层 K/V | `KVCache` | ✅ `BatchKVCache` | ✅ `QuantizedKVCache` | ✅（可 trim） | 最完整的一条路 |
| **滑窗 / 局部注意力**（Gemma-2/3、Mistral） | 环形 K/V + keep 前缀 | `RotatingKVCache` | ✅（但 `keep>0` 禁止） | ❌ NYI | ⚠️ 绕回后不可 trim | 量化与滑窗二选一 |
| **MLA**（DeepSeek-V2/V3） | 压缩 latent + rope 分量 | 模型自定义 cache | ⚠️ 取决于是否实现 `merge` | ⚠️ 需自定义 | ⚠️ | 每 token 状态远小于 GQA，KV 压力天然小 |
| **Recurrent / SSM**（Mamba、gated delta） | 固定形状隐状态 | `ArraysCache` | ✅（`filter`/`extend`） | ❌ 无「量化状态」概念 | ❌ 无「前缀」概念 | 状态不随长度增长，但**不可回退**——投机解码的 trim 不适用 |
| **Hybrid**（Qwen3-Next、Llama-4） | 上面几种混合 | `CacheList` | ✅（逐子 cache 转发） | ⚠️ 仅 KV 部分 | ⚠️ | 任一组件的 `is_trimmable` 为 False，整体就不可 trim |
| **Attention sinks**（gpt-oss） | 额外 sink logits | 模型内 | ✅ | ❌ | ✅ | `base.py:118-119` 显式 `raise ValueError("Quantized SDPA does not support attention sinks.")` |
| **MoE**（Qwen3-30B-A3B） | 无额外状态 | `KVCache` | ✅ | ✅ | ✅ | 但**权重**只读激活专家 → decode 字节数由「激活参数」而非总参数决定 |

**结论**：状态类型决定「能不能量化 / 能不能批 / 能不能回退」；三者的交集只有 **MHA/GQA + 无 sinks** 这一格是完整的。任何「通用加速方案」都必须先声明它假设了哪种状态类型。

### 3.8 度量口径问题（审计发现，必须先解决）

三个必须在下一轮修正的度量问题：

**(1) 同一代码路径两轮相差 1.9×。**
Stage 2.0 `bench.json`：`serial-lazy` = 61.32 ms（peak 1672 MB）；Stage 2.1 `bench_a345`：`serial-lazy` = 116.92 ms。两者都是「28 blocks / 257 token / 单次 eval」，`dag_forward` 在 chain 图 + `chain_mode="auto"` 时退化为同一段 `for block in layers` 代码（E2）。差异来源未定位，候选：热节流（interleaved 让 lazy 与更慢的 eager 交替，稳态温度更高）、分配器状态、编译标志的全局副作用。**在定位前，绝对 ms 不可跨报告引用。**

**(2) A2 的 1.46× 重叠上限基线被污染。**
`part_a2` 的计时区间包含 `build(k)`：k 次 `embed` + k 次 `mx.eval(h)`（`stage21_bench.py:125-135`），即**计时里混进了 k 次同步**。k=1 得到 1.053 ms，而 A1 独立测得的同规模单 Block（n=64）是 0.762 ms。按 A1 口径重算：8 个 Block 5.751 ms ÷ (8 × 0.762) = **1.06×**，不是 1.46×。另外 A2 用的是 **n=64**，不是 decode 的 n=1——报告正文没有写这一点。

**(3) A1 的「0.558 ms 固定开销」是外推，不是测量。**
n=1..16 的点全部落在 0.47–0.68 ms 且非单调（n=2 比 n=1 慢），边际斜率在这段测不出来；0.0058 ms/token 是由 n=64/256 两点主导的线性拟合。**「decode 90% 是固定开销」这句话在 n=1 时是恒真的（只有一个 token），不构成机制证据。** 真正的机制证据是「链内 0.253 ms/Block vs 孤立 0.47–0.68 ms/Block」，即 **per-eval 成本约 0.2–0.4 ms**。

**建议的判别实验（低成本，需批准）**：同一进程内先跑 20 次 28-block 链，再跑 20 次单 Block，交替 3 轮，同时用 `sudo powermetrics --samplers gpu_power` 采样频率/功耗；再用 `MPSGraph`/`Metal System Trace` 取一次 kernel 时间线。预计 5–8 分钟 GPU 时间、<2 GB 内存。**这是把「46% 带宽」与「kernel 太小」分开的唯一便宜办法。**

---

### 3.9 性能证据清单（按第五节的十项要求逐条填）

| 要求 | 本机现状 | 口径 / 来源 | 缺口 |
|---|---|---|---|
| **模型文件大小** | 1.1 GiB（BF16） | 本地 HF 缓存 `du` | 4-bit 版未在本机生成 |
| **活跃工作集** | 权重 1.19 GB + KV + 激活 | `[推算]` | 未用 `mx.get_active_memory()` 采样 |
| **实际峰值统一内存** | **1672 MB**（prefill 257，28 blocks，lazy） | `stage2_0/bench` 的 `peak_memory_mb` | decode 与长上下文峰值未记录 |
| **GPU/CPU/SSD 传输** | 统一内存无 H2D/D2H；SSD 仅加载与 prompt cache 落盘 | 架构事实 | 未测 SSD 带宽 |
| **Prefill tok/s** | **2198**（257 / 0.1169 s） | E1 | 只在 257 token 上测过 |
| **Decode tok/s** | **141.4**（7.07 ms/step，**不含 LM head**） | E1 | 含 LM head 的端到端未测 |
| **TTFT** | 257 token ≈ 117 ms；4k/32k 为推算 | E1 + E4 | 真实长 prompt 未测 |
| **每次高质量任务完成时间** | **未定义** | — | 建议定义 `T_correct = (TTFT + N·t_dec) / pass_rate`，在固定 eval 上测 |
| **能耗与温度** | **完全未测** | — | 协议见 §9.4；需 `powermetrics` |
| **模型输出质量** | teacher-forced ΔNLL 有历史数据；**生成质量未测** | 项目历史 | 需固定 eval（如 200 题）与 pass@1 |

**这张表的意义**：它把「哪些结论有证据、哪些只是推算」摊开。**能耗、生成质量、长上下文 TTFT 三项目前是空白**，任何对外表述都不应越过它们。

---

## 4. 交付 C：可直接复用的成熟底层技术清单

按「能否不重写就用」排序。所有条目都给出来源与落点。

### 4.1 内存管理

| 技术 | 来源 | 解决什么 | Apple 侧落点 | 复用方式 | 成本 |
|---|---|---|---|---|---|
| **分页 KV（块表）** | vLLM PagedAttention（SOSP'23, arXiv 2309.06180） | 消除 `concatenate` 增长、消除 2× 峰值、支持非连续序列 | 新 `PagedKVCache`（`cache.py` 加子类）+ 块分配器 | 抄数据结构与分配策略，attention kernel 用 gather 或按块循环 | 中高（要改 attention 路径） |
| **Radix 前缀树** | SGLang RadixAttention（arXiv 2312.07104） | 多会话共享前缀、按 token 级共享而非整段 | mlx-lm 已有 `PromptTrie`（`cache.py:1532`），缺「共享块 + 引用计数」 | 把 trie 的值从「整份 cache」改成「块列表 + refcount」 | 中 |
| **块级量化 KV** | vLLM fp8 KV、ExLlamaV3 2–8 bit cache | 长上下文带宽 | `QuantizedKVCache` 已有，缺「融合 append」与「批处理版」 | 补 `BatchQuantizedKVCache`（`merge`/`filter`/`extend`） | 中 |
| **滑动窗口 + 量化共存** | llama.cpp `--cache-type-k/v` + `-fa` | 滑窗模型的 KV 压缩 | `RotatingKVCache.to_quantized` 目前直接 raise | 让 rotating 的物理布局与 quantized 的 packed 布局对齐 | 中 |
| **零拷贝前缀复用** | vLLM prefix caching、SGLang radix | 重复 prefill | `LRUPromptCache.fetch_nearest_cache` 已用 `deepcopy`（实测为 COW，0.03 ms/64 MB，E0） | 只需避免「命中后首次写入触发整段复制」 | 低（先验证 COW 语义） |

### 4.2 调度

| 技术 | 来源 | 复用价值 | 备注 |
|---|---|---|---|
| **Continuous batching + 分段 prefill** | vLLM / SGLang / mlx-lm `BatchGenerator` | mlx-lm 已有，且支持 `prefill_step_size` 分段 | 与量化 KV、投机解码互斥（§3.7） |
| **Chunked prefill（按 token 预算切）** | vLLM V1 | 降 TTFT 抖动、控峰值 | mlx-lm 的 `prefill_step_size` 是近似物 |
| **Overlap scheduling（CPU 与 GPU 重叠）** | SGLang MLX 后端用 `mx.async_eval` | **已在 Apple 上验证可行** | `--disable-overlap-schedule` 可关 |
| **PD 分离** | vLLM / TRT-LLM | 单机无意义 | 多机才有价值 |
| **投机解码（draft/verify/trim）** | mlx-lm 已实现 | 直接可用 | 需小 draft + 可 trim 缓存 |

### 4.3 计算内核（只能借形状，不能借代码）

| 技术 | 来源 | 可借的部分 | 不可借的部分 |
|---|---|---|---|
| **GEMV 专用 kernel（M=1）** | llama.cpp `kernels/mul_mv.metal`（133 KB，按量化类型特化） | 「按行分块 + simdgroup 归约 + 权重量化布局」 | 具体 `simdgroup_matrix` 用法与量化块布局 |
| **GEMM 模板（M≥64）** | MLX `kernels/steel/`、llama.cpp `mul_mm.metal` | 分块、双缓冲、寄存器压力权衡 | 依赖 TMA/张量核的部分 |
| **FA 的 prefill/decode 双内核** | llama.cpp `fa_*.metal`（prefill）vs `fa_vec_*.metal`（decode）；MLX `sdpa_blocked.h` vs `sdpa_vector.h` | **必须按阶段拆**：decode 用 vector 版，prefill 用 blocked 版 | 具体 tile 尺寸 |
| **量化 FA** | llama.cpp `fa_q4_0/q8_0` 等 | KV 量化后仍走融合 attention | MLX 目前没有对应物（§5-D5） |
| **融合 SwiGLU / RoPE / 残差** | SGLang MLX 后端（`SGLANG_MLX_FUSE_SWIGLU`、`SGLANG_MLX_USE_CUSTOM_ROPE`）、BaseRT 的 fused MoE gate/up | 减少 dispatch 与一次全局往返 | 收益随模型形状变化 |
| **MoE 专家 GEMM + 路由融合** | TRT-LLM、BaseRT、llama.cpp | 按专家分组的小 GEMM 批处理 | 依赖 scatter/gather 效率 |
| **循环状态 kernel** | llama.cpp `ssm.metal`、`gated_delta_net.metal`、`wkv.metal`；MLX `gated_delta_update.metal` | Hybrid/Recurrent 模型的状态更新 | 状态布局与引擎强绑定 |

### 4.4 工程接口

| 技术 | 来源 | 为什么重要 |
|---|---|---|
| **`mx.fast.metal_kernel`** | MLX 0.31.2（E0 已验证签名） | **不 fork MLX 就能写融合 kernel**：GEMV、quantize-append、SwiGLU 都可以在 Python 侧原型化 |
| **`mx.fast.scaled_dot_product_attention`** | MLX | 已有 blocked/vector 两条路径；量化版需自己接 |
| **`mx.compile`** | MLX | 图融合；实测只有 2.4%（prefill），不要期待过高 |
| **`mx.async_eval`** | MLX | 唯一可靠的 CPU/GPU 重叠手段 |
| **`MLState`（Core ML）** | coremltools | 若走 ANE 路线，KV 必须做成固定 shape 的 state |

---

## 5. 交付 D：值得修改的具体实现问题（MLX / mlx-lm / Metal）

每条给：**位置 → 现象 → 影响 → 修复路径 → 是否必须改核心库**。行号对应 mlx-lm 0.31.3 / MLX 0.31.2。

### D1. `QuantizedKVCache` 每 token 重新量化，且是标量式追加

- **位置**：`mlx_lm/models/cache.py:242-283`，核心是 `277-281`。
- **现象**：`update_and_fetch` 每次调用都对**新来的 1 个 token** 做 `mx.quantize(keys)` 与 `mx.quantize(values)`，再逐字段写入 6 个 packed 数组（`weight(uint32)`、`scales`、`biases`，K/V 各一套）。
- **影响**：28 层 × 2 = **56 次小 quantize + 56 次跨 6 个 buffer 的散布写**，每 decode step 都付一次。1 个 token 的量化没有任何「批量摊薄」，纯属调度税。按 7.07 ms/step 估算，若每次 5–10 µs，合计 0.3–0.6 ms（**4–8%，未实测**）。
- **修复**：① 融合 kernel `quantize_append(dst_packed, dst_scale, dst_bias, new_kv, offset)`，一次 dispatch 完成量化 + 写入；② 或改成「块边界批量量化」：每 256 步把整块量化一次，期间用 bf16 暂存。
- **是否改核心库**：不必。可用 `mx.fast.metal_kernel`（E0 已验证接口）在用户侧写一个 `FastQuantizedKVCache`，只替换 `maybe_quantize_kv_cache` 的落点。

### D2. 滑动窗口 + KV 量化互斥

- **位置**：`cache.py:551-552`（`RotatingKVCache.to_quantized` → `raise NotImplementedError`）；`cache.py:1327-1328`（Batch 版同样）。
- **现象**：只要模型用滑窗（Gemma-2/3、Mistral、Qwen3 的 `sliding_window`），就**不能**开 `--kv-bits`。
- **影响**：滑窗模型恰恰是最需要长上下文压缩的一类；现在只能二选一。
- **修复**：`RotatingKVCache` 的物理布局是「环形 + keep 前缀」，量化版只需把 `keys/values` 换成 packed 三元组，并让 `_temporal_order` 在 packed 布局上做 roll（或先 dequant 再 roll，代价高）。更简单的路线是 D-O2 的分页化：分页天然支持环形替换。
- **是否改核心库**：是（或写一个平行的 `QuantizedRotatingKVCache` 子类，约 150 行）。

### D3. 连续批处理不支持量化 KV

- **位置**：`generate.py:838-857`（`_make_cache` 的 `else: raise ValueError(f"{type(c)} does not yet support batching")`）。
- **现象**：`BatchGenerator` 只认 `KVCache / ArraysCache / RotatingKVCache / CacheList`；`QuantizedKVCache` 直接抛错。
- **影响**：**「省内存」和「多会话」不能同时要**。24 GB 机器上这恰恰是最需要的组合。
- **修复**：实现 `BatchQuantizedKVCache`：需要 `merge / filter / extend / prepare / finalize / make_mask`，其中 `merge` 要把不同长度的 packed KV 左填充对齐（`cache.py:1088-1118` 是 bf16 版的参考实现）。
- **是否改核心库**：是（新增类，不改既有语义）。

### D4. 连续批处理与投机解码互斥

- **位置**：`server.py:370-374`。
- **现象**：`is_batchable = draft_model is None`，一旦配 draft 就退回「单请求串行 `generate_step`」。
- **影响**：多用户场景下，投机解码会把并发能力关掉；单用户场景无所谓。
- **修复**：`BatchGenerator` 需要支持「每序列独立的 draft 长度与 trim」——本质是把 `_rewind_cache` 从全局变成 per-sequence。工程量大。
- **是否改核心库**：是。

### D5. 量化 attention 没有融合，且 `queries *= scale` 是原地写

- **位置**：`mlx_lm/models/base.py:64-105`。
- **现象**：量化 KV 的 SDPA 被实现为两次 `mx.quantized_matmul`（QK⊤ 与 PV），中间**物化完整 score 矩阵** `(B, n_kv, n_rep, L, T)`；没有 flash 式的分块与在线 softmax。另外 `queries *= scale`（第 77 行）对传入张量做原地乘。
- **影响**：① 长上下文 prefill 时 score 矩阵是 O(L·T) 显存，32k×32k 不可行（只能靠 chunked prefill 规避）；② 失去 flash 的带宽优势；③ 原地写若 `queries` 是 rope 输出的视图，存在别名风险（未复现，标记为待查）。
- **修复**：写 `quantized_flash_attention`（分块 + 在线 softmax + on-the-fly dequant），或至少对 decode（L=1）走 vector 版。
- **是否改核心库**：是（或 `mx.fast.metal_kernel` 原型）。

### D6. KV 增长用 `concatenate`，峰值 2×

- **位置**：`cache.py:333-355`。
- **现象**：每 256 token 触发一次「裁剪到 offset → `mx.concatenate([旧, 新zeros])` → 写回」。增长时旧 buffer 与新 buffer 同时存在。
- **影响**：32k 上下文单层 KV 128 MiB，28 层 3.5 GiB；增长瞬间峰值 **7 GiB**，直接撞 12 GiB 预算与 13.32 GiB 单 buffer 上限。此外每次增长要复制全部历史（32k 时累计复制量约 224 GiB，摊到每 token ≈ 7 MB，约 1% 时间）。
- **修复**：① 预分配 `max_kv_size`（`--max-kv-size` 已支持，但会改变语义）；② 按 1.5× 几何增长而非固定 +256；③ 分页（D-O2）。
- **是否改核心库**：② 可以子类化覆盖；③ 需新类。

### D7. `mx.clear_cache()` 每 256 步

- **位置**：`generate.py:467-468`；`cache_prompt.py` 的 prefill 每块也调。
- **现象**：周期性释放 Metal 缓冲缓存，下一次分配要走一次系统分配。
- **影响**：长生成会出现周期性延迟尖峰；SGLang 的 MLX 后端把同一参数暴露为 `SGLANG_MLX_CLEAR_CACHE_STEPS=256`，说明这是社区共识的折中。
- **修复**：按 `mx.metal.get_cache_memory()` 动态决定，而不是固定步数。
- **是否改核心库**：否（用户侧循环可控）。

### D8. wired limit 默认拉满

- **位置**：`generate.py:230-266`（`wired_limit`）；`generate.py:1544-1547`（`BatchGenerator.__init__`）。
- **现象**：默认把 wired 上限设为 `max_recommended_working_set_size` = **17.76 GiB**。
- **影响**：在还有其它开发工作的 24 GB 机器上，这等于允许 mlx-lm 把机器吃到只剩 6 GiB。
- **修复**：显式 `mx.set_wired_limit(12 GiB)`；这是本审计**唯一建议立刻执行**的改动，零风险。
- **是否改核心库**：否。

### D9. `KVCache.to_quantized` 量化整段已分配缓冲（含未使用的零尾）

- **位置**：`cache.py:383-391`。
- **现象**：`quant_cache.keys = mx.quantize(self.keys, ...)` 用的是 `self.keys`（完整 256 对齐的 buffer），不是 `self.keys[..., :offset, :]`。
- **影响**：转换那一刻多做最多 255 行的量化；一次性，影响小，但语义上不干净（零尾也被量化并计入 nbytes）。
- **修复**：先切片再量化。
- **是否改核心库**：否（可子类化）。

### D10. 前缀缓存的 trim 语义在滑窗后失效

- **位置**：`cache.py:542-543`（`is_trimmable` 返回 `offset < max_size`）。
- **现象**：`RotatingKVCache` 一旦绕回，就不再可 trim，`LRUPromptCache.fetch_nearest_cache` 的「longer → trim 成公共前缀」路径失效，退化为「只匹配更短前缀」。
- **影响**：滑窗模型的前缀复用率下降。
- **修复**：环形缓冲支持「按 token 数回退」需要保存被覆盖的历史，代价高；更现实的是分页化。
- **是否改核心库**：是。

### D11. `BatchRotatingKVCache` 不支持 `keep > 0`

- **位置**：`generate.py:850-852`。
- **现象**：`make_prompt_cache(max_kv_size=...)` 默认 `keep=4`（`cache.py:37`），而批处理路径显式禁止 `keep>0`。
- **影响**：批处理 + 滑窗时，前 4 个 token（通常是 BOS/系统提示）不再被保留，语义与单请求路径不一致。
- **修复**：在 batch 版实现 keep 语义（左填充偏移即可）。
- **是否改核心库**：是。

### D12. 项目脚本的度量口径（不是库的问题，但影响结论）

- `stage21_bench.py:125-135`：A2 计时区间包含 `build(k)`（k 次 embed + k 次 `mx.eval`），k=1 基线被抬高 38%。
- `stage21_bench.py:96-105`：A1 的线性拟合把 n=1..256 混在一起，n≤16 段非单调，0.558 ms 是外推值。
- `stage20_bench.py` vs `stage21_bench.py`：同一 chain 图，61.3 vs 116.9 ms，未定位。
- **修复**：见 §9.2 的三个实验。

---

## 6. 交付 E：与 NVIDIA / AMD 平台共通或不共通的机制

### 6.1 三层判定

| 机制 | Apple Silicon | NVIDIA | AMD | 手机 GPU/NPU | 判定 |
|---|---|---|---|---|---|
| **decode 是 bytes/token 受限** | ✅ 124.6 GB/s / 273 | ✅ HBM 3.35 TB/s（H100） | ✅ HBM3 | ✅ LPDDR | **完全共通**（roofline 同构） |
| **prefill 是算力受限** | ✅ 2.06 TFLOP/s（无矩阵核） | ✅ 989 TFLOPS bf16（H100） | ✅ Matrix Core | ⚠️ NPU 另算 | **共通但差 2 个数量级** |
| **分页 KV / 块表** | 可移植 | 原生 | 原生 | 可移植 | **算法共通** |
| **连续批处理 / chunked prefill** | 可移植（mlx-lm 已有） | 原生 | 原生 | 可移植 | **算法共通** |
| **Radix 前缀缓存** | 可移植 | 原生 | 原生 | 可移植 | **算法共通** |
| **投机解码** | 可移植（已有） | 原生 | 原生 | 可移植 | **算法共通** |
| **FA 的 prefill/decode 拆分** | 已有（`sdpa_blocked`/`sdpa_vector`） | 已有 | 已有 | 部分 | **设计共通，kernel 不共通** |
| **量化 KV + 融合 FA** | ❌ 缺 | ✅ fp8 KV | ✅ | 部分 | **Apple 的缺口** |
| **张量核 / NAX** | M5+ 才有（`*-nax.metal`） | Volta+ | CDNA3+ | NPU | **不共通** |
| **FP8 算力** | 无 fp8 矩阵指令；`fp8.h` 只做存储/转换 | ✅ | ✅ | 部分 | **不共通** |
| **TMA / cluster / warp specialization** | 无 | Hopper+ | 部分 | 无 | **不共通** |
| **CUDA graph / HIP graph** | 无对应物（惰性图是近似） | ✅ | ✅ | 部分 | **半共通**（Metal 4 显式编码器是替代） |
| **零拷贝 H2D/D2H** | ✅ 统一内存天然零拷贝 | ❌（Grace Hopper 除外） | ❌ | ✅ 部分 | **Apple 独有优势** |
| **MoE 专家卸载** | 统一内存 → 「卸载」= 留在 RAM，无需 PCIe | PCIe 4/5 | PCIe | 无 | **Apple 成本最低** |
| **ANE / NPU 跑 LLM** | 固定 shape、算子子集、编译慢 | — | — | 同 | **不推荐做 decode 主路径** |

### 6.2 一条重要的跨平台推论

因为 **decode 在所有平台上都是 bytes/token 受限**，「压缩关键路径深度」在 NVIDIA 上同样不会带来 decode 加速——只是 NVIDIA 的带宽更高（H100 3.35 TB/s ≈ M4 Pro 的 12×），所以同样的 0.6B 模型在 H100 上 decode 是 ~0.36 ms/token（2778 tok/s），仍然与深度无关。**Stage 2.1 的「深度不是延迟」结论在 NVIDIA 上只会更强，不会更弱**；唯一会变的是 prefill（张量核让「深度」重新有意义，但 prefill 本来就已经被 batch 填满）。

### 6.3 移植成本估计（人日，单人）

| 工作 | 到 CUDA | 到 Metal | 说明 |
|---|---|---|---|
| 分页 KV（块表 + gather attention） | 5–8 | 15–25 | Metal 缺 paged-attention 现成 kernel |
| 连续批处理 + 量化 KV | 3–5 | 8–12 | 主要是 `merge/filter/extend` 的 packed 布局 |
| Radix 前缀缓存 | 2–3 | 3–5 | 数据结构为主 |
| 融合 quantize-append | 2 | 4–6 | Metal 无 `cp.async`，要手工流水 |
| 小 draft 投机解码 | 1–2 | 1–2 | mlx-lm 已有，主要是调参 |

---

## 7. 交付 F + G：五个低层改造机会（含风险、依赖、验证成本、反面证据）

排序依据：**证据强度 × 预期收益 ÷ 成本**。每条都给 kill 条件——先想清楚什么结果会让它死。

### O1. decode 主路径 GEMV 的带宽/占用率

| 项 | 内容 |
|---|---|
| **假设** | M=1 时 124.6 GB/s 不是硬件上限，而是「每 threadgroup 只算一行、并发请求数不足」造成的；改行分块 + 128-bit 向量化 + 预转置布局可到 170–200 GB/s |
| **依据** | E1：880.8 MB / 7.07 ms；E2：MLX 有 `gemv.metal`/`gemv_masked.metal`，llama.cpp 有 133 KB 的 `mul_mv.metal` 按量化类型特化——说明「GEMV 需要专用 kernel」是两家共识；E3：BaseRT 明确 decode 不靠张量核，靠调度与融合 |
| **最小实现** | 用 `mx.fast.metal_kernel` 写一个 bf16 GEMV（4 行/threadgroup、float4 载入、simd 归约），在合成张量上对比 `mx.quantized_matmul`/`nn.Linear` |
| **验证协议** | 形状扫描 M=1，K∈{1024,4096}，N∈{1024,4096,151936}；指标 GB/s；再接入一个 Block 做端到端 A/B |
| **成功判据** | 三个主形状平均 ≥1.3×；端到端 decode ≥1.15× |
| **成本** | 编码 3–5 天，GPU 时间 <15 分钟，内存 <2 GB |
| **风险** | Apple GPU 的 1-row GEMV 可能本就封顶在 ~150 GB/s；若如此，收益只有 1.2× |
| **反面证据** | ① `mx.compile` 只带来 0%，说明 dispatch 不是主要成本 → 剩下的只能是「kernel 内部效率」或「带宽墙」，两者都不好改；② A2 显示独立 Block 重叠收益有限 → 并发补不上占用率 |
| **kill 条件** | 手写 kernel 与现成 `quantized_matmul` 的 GB/s 差距 <10%，或端到端 <5% |

### O2. KV 分页 + 融合 quantize-append

| 项 | 内容 |
|---|---|
| **假设** | 用块表（block=16/32）替代 `concatenate` 增长，并让量化在 append 时一次完成，可同时拿到：峰值内存 −50%、32k decode +20–30%、批处理 + 量化 KV 可共存 |
| **依据** | E1：32k 时 KV 3.5 GiB，增长峰值 7 GiB，超预算；E2：D1/D2/D3/D6 四个缺口同源；E3：vLLM PagedAttention 是这类问题的标准答案 |
| **最小实现** | 新 `PagedKVCache`（块分配器 + 引用计数）+ 基于 `mx.take` 的 gather attention（先不追求 flash） |
| **验证协议** | ① 32k 单请求 decode 的峰值内存与时延；② 4 会话并发下的 TTFT 与总吞吐；③ 与 `--kv-bits 8` 组合是否可用 |
| **成功判据** | 峰值内存 ≤ 现在的 60%；32k decode 不慢于现在；4 会话吞吐 ≥1.5× |
| **成本** | 2–3 周；GPU 时间 1–2 小时；需要改 attention 调用路径（可先只在 decode 用） |
| **风险** | Metal 上 gather 版 attention 可能比连续版慢；batch=1 时分页是纯开销（vLLM 的收益来自多请求） |
| **反面证据** | ① vLLM 的分页收益几乎全部来自多请求吞吐，单请求 latency 反而略差；② 本机是单用户开发机，4 会话并发是假设而非现状；③ 单 buffer 13.32 GiB 的限制可以用「预分配 max_kv_size」绕过，成本低得多 |
| **kill 条件** | 单请求 32k decode 没有改善，且多会话不是真实需求 → 退化为「预分配 + 几何增长」的 2 天方案 |

### O3. 小 draft 投机解码（含零成本 draft）

| 项 | 内容 |
|---|---|
| **假设** | 唯一能减少 bytes/token 的机制；用 1/10 尺寸 draft 或「prompt-lookup / n-gram」零成本 draft，可在不损质量的前提下拿到 1.5–2× |
| **依据** | E1/E4：§3.6 的盈亏平衡表；E2：`speculative_generate_step` 已实现且要求可 trim 缓存；E3：M5 论文把投机解码列为 decode 的主要方向 |
| **最小实现** | 先用**零模型成本**的 prompt-lookup（从 prompt 里找 n-gram 续写作为 draft），~100 行，不需要下载任何模型；再用小 draft 模型 |
| **验证协议** | 在 wikitext-2 窗口与一段代码/JSON 上测：接受率、有效 ms/token、输出与贪心解码的一致性（投机解码在贪心采样下是**精确**的，可逐 token 对齐验证） |
| **成功判据** | 目标负载上接受率 ≥40% 且有效加速 ≥1.4× |
| **成本** | prompt-lookup 1–2 天；GPU 时间 <20 分钟 |
| **风险** | 0.6B 的 target 本身只有 7 ms，draft 相对成本高；自然语言散文的接受率通常低于代码/结构化文本 |
| **反面证据** | ① 同尺寸 draft 在 100% 接受率下才打平（§3.6），说明「draft 必须显著更小」；② 0.6B 上再小就是 0.06B，本地没有现成模型（不得下载）；③ 验证步虽便宜，但 `_rewind_cache` 会触发 KV 重写 |
| **kill 条件** | 目标负载接受率 <30%，或有效加速 <1.2× |

### O4. Radix 前缀缓存 + 零拷贝复用

| 项 | 内容 |
|---|---|
| **假设** | 多会话共享长系统提示时，按 token 级 radix 共享可把 TTFT 从「每次重算」降到「只算新增」，4 会话下总吞吐 ≥1.5× |
| **依据** | E2：`PromptTrie` + `LRUPromptCache` 已有（`cache.py:1532-1763`），但是「整段 cache 为值」，不支持部分共享；E3：SGLang RadixAttention |
| **最小实现** | 把 trie 的值改成「块列表 + refcount」，`fetch` 返回共享块 + 新块；先用现有 `KVCache` 拼装，不做分页 |
| **验证协议** | 4 个会话、共享 2k 系统提示、各自 200 token 输出：TTFT、总时间、峰值内存；对照组 = 现在的 `--prompt-cache-file` |
| **成功判据** | 第 2–4 个会话 TTFT ≤ 首会话的 30%；总时间 ≥1.5× 改善 |
| **成本** | 5–8 天；GPU 时间 <30 分钟 |
| **风险** | 单用户场景命中率低；缓存本身占内存，挤占 KV 预算 |
| **反面证据** | ① 现有 LRU + 精确前缀匹配已经覆盖「同一系统提示」这一最常见场景；radix 的增量只在「部分前缀」；② 24 GB 下缓存与 KV 争内存，可能得不偿失 |
| **kill 条件** | 4 会话基准下 TTFT 改善 <20% |

### O5. prefill 侧算子融合（SwiGLU / RoPE / 残差）

| 项 | 内容 |
|---|---|
| **假设** | 把 `silu(gate)*up`、`rope(q),rope(k)`、`h+x` 各自融成一个 kernel，可减少 20–30% 的 dispatch 与一次全局往返，prefill 提升 3–8% |
| **依据** | E3：SGLang 的 MLX 后端已把这两项做成开关（`SGLANG_MLX_FUSE_SWIGLU`、`SGLANG_MLX_USE_CUSTOM_ROPE`），说明工程上可行且有收益；BaseRT 的 MoE gate/up 融合省掉「每专家一次全局内存往返」 |
| **最小实现** | 用 `mx.fast.metal_kernel` 写 fused SwiGLU（输入 gate/up 两个张量，输出一个），A/B 对比 |
| **验证协议** | 257 / 2048 / 8192 token 三档 prefill，28 blocks，端到端 ms；同时看 `mx.get_peak_memory()` |
| **成功判据** | 2048 token prefill ≥3%；峰值内存不升 |
| **成本** | 3–5 天；GPU 时间 <20 分钟 |
| **风险** | 收益小；M5 上张量核会把瓶颈移走，这项优化在 M5 上价值下降 |
| **反面证据** | ① `mx.compile` 只有 2.4%，说明 prefill 的 dispatch 税本来就不高；② A5 显示融合 matmul 在 n=1 时反而慢 29%——融合不是无条件的 |
| **kill 条件** | 2048 token 下 <2% |

### 7.1 明确不推荐（以及为什么）

| 方向 | 为什么不做 |
|---|---|
| **继续压缩关键路径深度 / 稀疏 DAG** | 两轮实测否证（prefill 5–6% 无 CI、decode 0），且机制上不影响 bytes/token。**这是本审计最重要的「止损」建议。** |
| **ANE / Core ML 跑 decode** | 固定 shape、算子子集、每次预测的调度开销；decode 是带宽受限，ANE 的算力用不上 |
| **M4 上追张量核** | 硬件没有；M5 的 NAX 只能通过 Metal 4 `-std=metal4.0` 访问，且 `MTLGPUFamily` 仍报 apple9（E3），不能靠能力查询分流 |
| **专家流式卸载（现在）** | 0.6B 用不上；24 GB 能装下 4-bit 30B MoE。等真的要用 100B 级模型再立项（那时 KTransformers / ExLlamaV3 的 offload 是现成参考） |
| **多机 PD 分离** | 单机单用户无意义 |

---

## 8. 目标平台与模型：两层实验体系

用户明确要求：**不要用 0.6B 的 tok/s 代表大模型**。因此建立三层：

### 8.1 第一层：0.6B（机制正确性，已完成）

用途：DAG 等价性、KV 正确性、量化误差、投机解码的逐 token 对齐。**不用于性能结论**。

### 8.2 第二层：合成大张量（形状与访存，不需要下载模型）

这是本轮建议新增的核心实验。用 `mx.random` 造权重，只测 kernel：

| 维度 | 取值 | 目的 |
|---|---|---|
| M（token 数） | 1, 2, 4, 8, 16, 32, 64, 256 | 找到「GEMV → GEMM」的转折点；验证 M=1 的带宽墙 |
| K（输入维） | 1024, 2048, 4096, 8192, 16384 | 覆盖 0.6B / 8B / 70B 的 hidden |
| N（输出维） | 1024, 4096, 11008, 151936 | 覆盖 FFN 与 LM head |
| dtype | bf16, q4(group64), q8(group64), mxfp4 | 量化布局对带宽的影响 |
| 指标 | GB/s、TFLOP/s、峰值内存、dispatch 数（用 `mx.metal.start_capture` 或计时差） | 得到 roofline |

**为什么这一步最值**：它能在**不下载任何模型**的前提下，回答「8B/70B 在这台机器上 decode 应该是多少 tok/s」，并把 O1 的成功判据从「感觉快了」变成「GB/s 到没到 roofline」。

### 8.3 第三层：真实模型的代表性（现在不下载，只做评估）

| 模型 | 类型 | 参数量 / 激活 | 4-bit 体积 | 12 GiB 预算内？ | 为什么有代表性 | 状态类型 |
|---|---|---|---|---|---|---|
| Qwen3-8B | Dense GQA | 8B | ~4.6 GB | ✅ | 与 0.6B 同架构放大 13×，最干净的 scaling 对照 | KVCache |
| Llama-3.1-8B | Dense GQA, 128k | 8B | ~4.6 GB | ✅ | 长上下文 + RoPE scaling | KVCache |
| Qwen3-30B-A3B | MoE | 30B / 3B | ~15 GB | ⚠️ 超预算 | 24 GB 上「能跑但挤」的边界；decode 只读激活专家 | KVCache（全层） |
| gpt-oss-20b | MoE MXFP4 | 20B / ~3.6B | ~12 GB | ⚠️ 临界 | 原生 MXFP4，测 MLX 的 `mxfp4` 路径 | KVCache |
| Gemma-3-12B | Dense + 滑窗（5:1 local:global） | 12B | ~6.5 GB | ✅ | **唯一能测 `RotatingKVCache` 的真实模型** | Rotating + 全层混合 |
| DeepSeek-V2-Lite | MLA | 16B / 2.4B | ~9 GB | ✅ | 测 MLA 的压缩 KV（每 token 远比 GQA 小） | 压缩 latent |
| Qwen3-Next-80B-A3B | Hybrid（gated delta + attention） | 80B / 3B | ~40 GB | ❌ | 测 `ArraysCache` 与 `CacheList` 的混合状态管理 | 混合 |
| Llama-4-Scout | Hybrid MoE + iRoPE | 109B / 17B | ~55 GB | ❌ | 未来形态；本轮只记录，不计划 | 混合 |

**建议**：下一轮只碰 **Qwen3-8B + Gemma-3-12B** 两个（覆盖 dense GQA 与滑窗），其余保持「评估过、不下载」。

### 8.4 内存预算表（24 GB，给其它工作留 6 GB）

| 项 | 预算 | 说明 |
|---|---|---|
| 其它开发工作 | 6 GiB | 用户硬约束 |
| 系统 + 窗口服务器 | 3 GiB | macOS 自身 |
| 可用给推理 | **≤13 GiB** | 其中 wired 建议 ≤12 GiB |
| 权重（4-bit） | ≤10 GiB | 对应 ≤20B dense 或 30B MoE |
| KV + 激活 + 临时 | ≥3 GiB | 32k GQA-8B 的 KV 约 4 GiB（bf16）/ 1 GiB（4-bit） |

---

## 9. 证据清单、冲突与待批准实验

### 9.1 与项目既有结论的冲突（必须由三方裁决）

| # | 既有表述 | 审计发现 | 影响 |
|---|---|---|---|
| C1 | 「decode 是 ~90% 固定开销（launch + 权重流）」 | 链内 0.253 ms/Block 是**带宽**（124.6 GB/s）；0.558 ms 是**per-eval** 成本，二者不可混用 | 机制叙述要改；不影响「并行无收益」的结论 |
| C2 | 「GPU 重叠上限 1.46×」 | k=1 基线含 k 次同步 eval，按 A1 口径重算 **1.06×**；且 A2 用的是 n=64 不是 n=1 | 「上限」这个数不能作为硬件属性引用 |
| C3 | 「A7 达到 3.4–4.0 TFLOP/s，约 40% 峰值」 | 端到端 prefill 只有 **2.06 TFLOP/s**；单 Block 孤立值不能代表链内 | 峰值占比应改为「端到端 ~20–30%（峰值本身未文档化）」 |
| C4 | 「full-g4 prefill 快 5–6%」 | 无配对 CI；p10–p90 区间重叠 | 只能写成「方向性、未达显著」 |
| C5 | Stage 2.0 vs 2.1 的 prefill 绝对值 | 61.3 vs 116.9 ms，同代码 | 绝对延迟暂不可引用 |

### 9.2 需要批准才能做的三个实验（按性价比排序）

| # | 实验 | 目的 | 成本 | 风险 |
|---|---|---|---|---|
| X1 | **同进程交替测 28-block 链 / 单 Block，采 `powermetrics`** | 定位 C5 的 1.9× 差异；分离「per-eval 成本」与「kernel 成本」 | 5–8 min GPU，<2 GB | 无（只读） |
| X2 | **合成张量 GEMV/GEMM 扫描（§8.2）** | 建立 roofline，给 O1 定判据 | 15–20 min GPU，<4 GB | 无 |
| X3 | **prompt-lookup 投机解码（零新模型）** | 验证 O3 的接受率 | 10–20 min GPU，<2 GB | 无（不改库） |

三个实验都满足「不下载模型、不改核心库、低负载」；建议一次批准，分三次跑。

### 9.3 未核实项（明确标记，不要当结论用）

- M4 Pro 的 bf16 峰值算力（Apple 不公布；「~9 TFLOPS」是推断）。
- 46% 带宽里，「kernel 太小」与「DRAM 并发不足」各占多少。
- `queries *= scale` 的原地写是否真的会造成别名问题。
- `copy.deepcopy` 对 mx.array 是 COW（实测 0.03 ms/64 MB，E0），但「命中后首次写入是否触发整段复制」未验证。
- 投机解码的真实接受率（本地无小 draft 模型，未下载）。
- 能耗与温度：**本轮完全没有测**。协议见 §9.4。

### 9.4 能耗/温度测量协议（下一轮执行）

```bash
sudo powermetrics --samplers gpu_power,cpu_power,thermal -i 200 -n 300 \
  -o /tmp/pm.txt &
# 同时跑固定 200 token 生成，记录 wall time 与 tokens
```
指标：package power（W）、GPU power、`thermal pressure`、频率；换算 **J/token** 与 **tokens/J**。注意 M4 Pro 的 package TDP 上限约 45 W 量级（规格值，未实测），持续负载 2–3 分钟后应能看到降频——这正是 C5 的候选解释。

---

## 10. 参考

**引擎与文档**
- MLX：https://github.com/ml-explore/mlx ｜ MLX-LM：https://github.com/ml-explore/mlx-lm
- llama.cpp：https://github.com/ggml-org/llama.cpp ｜ Metal 后端：`ggml/src/ggml-metal/`（`ggml-metal-ops.cpp`、`ggml-metal-context.m`、`kernels/mul_mv.metal`、`kernels/fa_*.metal`）
- ExLlamaV2（archived）：https://github.com/turboderp-org/exllamav2 ｜ ExLlamaV3：https://github.com/turboderp-org/exllamav3
- vLLM：https://github.com/vllm-project/vllm ｜ vllm-metal：https://github.com/imclab/vllm-metal
- SGLang：https://github.com/sgl-project/sglang ｜ Apple Silicon 文档：https://docs.sglang.io/docs/hardware-platforms/apple_metal
- TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM
- MLC-LLM：https://github.com/mlc-ai/mlc-llm
- ExecuTorch：https://github.com/pytorch/executorch
- ONNX Runtime CoreML EP：https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- Core ML 有状态模型：https://apple.github.io/coremltools/docs-guides/source/stateful-models.html

**论文**
- PagedAttention（vLLM），SOSP 2023，arXiv:2309.06180
- SGLang / RadixAttention，arXiv:2312.07104
- BaseRT: Advancing Best-in-Class LLM Inference with Apple M5 Neural Accelerators，arXiv:2607.19438
- QTIP（ExLlamaV3 的量化基础），arXiv:2406.11235 ｜ QuIP#，arXiv:2402.04396
- ExecuTorch，MLSys 2026

**Apple 官方**
- Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU：https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Metal 4 / MTLTensor / MPP：https://developer.apple.com/documentation/metal/

**本仓库**
- `results/stage2_0/bench/bench.json`、`results/stage2_0/decode/decode.json`
- `results/stage2_1/PART_A.md`、`bench_a12/a345/a67`、`decode_check`
- `results/stage2_1/REPORT.md`

---

## 附录 A：公式与常数

```
B_block      = 15,728,640 × 2 B = 31.46 MB
B_weights    = 28 × 31.46 MB = 880.8 MB
B_head       = 151,936 × 1024 × 2 B = 311.2 MB
B_kv/token   = 2 × 8 × 128 × 2 B × 28 = 114,688 B = 112 KiB
F_token      = 880.6 MFLOPs（28 层线性部分）
F_attn(C)    = 0.5 × 8192 × C² × 28
t_decode(C)  = (880.8 MB + 311.2 MB + C × 112 KiB) / BW_eff
BW_eff       = 124.6 GB/s（实测，0.6B，257-token KV）
T_prefill(C) = (C × 880.6 MF + F_attn(C)) / 2.06 TFLOP/s
```

## 附录 B：本轮未做但值得记的观察

1. **LM head 占 0.6B 每 token 字节的 26%**（311/1192）。小模型的 tok/s 被 vocab 严重拖累，这正是「0.6B 不能代表大模型」的算术原因之一。
2. **`max_buffer_length` = 13.32 GiB**：任何单张量超过它就会失败。128k 上下文的 KV（14 GiB）正好越线。
3. **`resource_limit` = 499,000**：分页 KV 若用 16-token 块，128k 上下文 × 28 层 = 229,376 块，接近上限的一半——分页块大小不能太小。
4. **SGLang 的 MLX 后端在启动命令里要求 `--disable-cuda-graph`**，说明「图捕获」在 Metal 上没有等价物；MLX 的惰性图是替代品，但不是同一种东西（不能跨请求复用执行计划）。

