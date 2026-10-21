# R4-W / 实验就绪性：现有资源、Stage B 与 Stage C 的受控计划

**一句话：Stage A（MLX 内部位宽 A/B）今天就能跑；Stage B（IQ2_XS vs IQ4_XS）今天跑不了，缺 3 项、其中 2 项是安装；Stage C 三层里 L0 可做、L1 勉强、L2 需下载。本轮只设计，未执行任何一步。**

授权边界：只读清点 + Markdown 设计。**未编译、未下载、未安装、未跑 GPU、未改任何实验代码。**

---

## 1. 环境清点（只读实测）

### 1.1 软件

| 项 | 版本 / 状态 | 对实验的意义 |
|---|---|---|
| macOS | **26.6.2 (25G83)** | 系统够新 |
| CommandLineTools | 已装，`clang 21.0.0` | 只能编 C/C++/ObjC |
| **Xcode.app** | **未安装**（`/Applications` 无 `Xcode*.app`） | — |
| **`xcrun metal`** | **不存在**：`unable to find utility "metal"` | **无法编译任何 Metal shader** |
| **`cargo` / `rustc`** | **不存在** | **无法编译 uzu** |
| mlx | **0.31.2** | Stage A 主力 |
| mlx-lm | **0.31.3** | 含 `mx.quantize` 2/3/4/5/6/8 bit × affine/mxfp4/mxfp8/nvfp4 |
| lm-eval | **0.4.9.2** | **Stage C 的 L2 不需要装包** |
| datasets | 4.4.1 | 但**无缓存**（见 1.3） |
| transformers | 5.9.0 | — |
| torch | 2.9.1 | `convert_hf_to_gguf.py` 需要 |
| gguf | 0.19.0 | 只能读写容器，**不能做 IQ2_XS 量化** |
| numpy / scipy | 2.3.5 / 1.16.3 | — |

### 1.2 llama.cpp

| 项 | 状态 |
|---|---|
| 源码树 | `~/Desktop/ChlomeeTV/game design/revolution-core/tools/llama.cpp-v042` |
| commit | `0d2d9cc`（**grafted 浅克隆**，PR #24787） |
| ggml | **0.15.2** |
| 构建 | `build/`，2026-06-19，`CMAKE_BUILD_TYPE=Release`，`GGML_METAL=ON`，`GGML_METAL_EMBED_LIBRARY=ON`，`GGML_NATIVE=ON` |
| 可执行 | **只有 `llama-server`**（49 KB） |
| 缺 | `llama-bench`、`llama-cli`、`llama-quantize` |
| IQ2_XS Metal kernel | **在**（`ggml-metal.metal` 里 `kernel_mul_mv_iq2_xs_f32` 命中 5 次） |
| 归属 | **别人的项目目录**，只读，不修改 |

**这个构建证明了两件事**：2026-06 时这台机器有可用的 Metal 编译器（否则 `libggml-metal.0.15.2.dylib` 不会存在）；现在没有了。

### 1.3 模型与数据

| 项 | 状态 |
|---|---|
| 模型 | `mlx-community/Qwen3-0.6B-Base-bf16`，rev `e8e73ff38f0eb570eb57ed044ef3f95588c43bc8` |
| HF 缓存 | 只有上面这一个模型；**`~/.cache/huggingface/datasets` 不存在** |
| GGUF 权重 | **无**。桌面那棵树里只有 20 个 `ggml-vocab-*.gguf`（词表，无权重） |
| 本地数据 | `data/`：cal 16 / val 8 / test 8 窗；`data257/`：s20_cal 64 / s20_test 64 / s20_ood_c4 64 / fresh_test 64 / ood_imdb_v2 64 窗 |
| 单窗长度 | 129 token，预测 128 token |
| 语料 | `Salesforce/wikitext` `wikitext-2-raw-v1`，rev `b08601e0…`，**只有窗口切片，无原始语料** |
| 磁盘 | 124 GiB 可用 |

**可用评估 token 上限**：64 窗 × 128 = **8,192 token**（任一 split）。

### 1.4 结论：三档可执行性

| 档 | 内容 | 今天能否跑 | 缺口 |
|---|---|---|---|
| **A** | MLX 内部位宽 A/B（bf16 / affine 2b g64 / affine 4b g64 / affine 8b g64 / mxfp4 g32），合成大张量 | **能** | 无 |
| **B** | llama.cpp IQ2_XS vs IQ4_XS | **不能** | ① Metal 编译器 ② `llama-quantize`/`llama-bench` ③ 一个 GGUF |
| **C-L0** | 数值/重构误差门 | **能** | 无（只需本地权重） |
| **C-L1** | PPL / 退化门 | **勉强** | 8,192 token 只够粗判（见 §4.2） |
| **C-L2** | 配对任务门 | **不能** | GSM8K / HumanEval / BFCL 数据集均无本地副本 |
| **XD-3** | 1.91× prefill 归因 | **不能** | 需运行（本轮未授权） |

**Stage B 的三个缺口，两个是安装**（Xcode 提供 `metal`；构建提供 `llama-quantize`/`llama-bench`），一个是下载（GGUF，或从本地 safetensors 现转）。按 Issue 裁决，**一律不自动补**。

---

## 2. Stage B 受控设计（不执行）

### 2.1 为什么不能省

Issue 已裁定：**MLX affine 2b 与 llama.cpp IQ2_XS 不是同一个性能问题**。补两条更硬的理由：

1. **编码不同**：MLX affine g64 = `w = scale·q + bias`，每 64 权重 2 个 bf16 参数 → 4.5 bpw（4-bit）/ 2.5 bpw（2-bit）。IQ2_XS = 74 B / 256 权重 = **2.3125 bpw**，是块状 E8 格点码 + 每 8 个块一组 scale。**同一"2-bit"标签下差 8% 的字节**。
2. **kernel 不同**：MLX 的 `quantized_matmul` 走 `steel/` 模板；llama.cpp 的 `kernel_mul_mv_iq2_xs_f32` 是独立的 SIMD-group 归约 + 格点解码。**位宽相同不代表 ALU 需求相同。**

所以 Stage A 的速度**不得**支持任何 IQ2_XS 结论。这一点 R3 已写死，本轮不变。

### 2.2 前置条件（全部缺失）

```
P1  xcrun -sdk macosx metal --version      → 需要 Xcode.app（不是 CLT）
P2  build/bin/llama-quantize               → 需要 cmake 构建（可用 GGML_METAL=OFF，CPU 足够量化）
P3  build/bin/llama-bench                  → 需要 cmake 构建，且**必须 GGML_METAL=ON**，否则测的是 CPU
P4  qwen3-0.6b-f16.gguf                    → convert_hf_to_gguf.py（本地有，torch/transformers 齐）
P5  qwen3-0.6b-{iq2xs,iq4xs}.gguf          → llama-quantize
```

**P3 是硬卡点**：没有 `metal` 编译器，`GGML_METAL=ON` 构建会在 shader 编译阶段失败；`GGML_METAL=OFF` 能构建但 `llama-bench` 只测 CPU，对"带宽 vs ALU"这个问题**零信息**。

### 2.3 命令形状（预注册，未执行）

```bash
# 一次性准备
python convert_hf_to_gguf.py <mlx_model_dir> --outfile qwen3-0.6b-f16.gguf --outtype f16
llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-iq2xs.gguf IQ2_XS
llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-iq4xs.gguf IQ4_XS

# 分开测，不合并
llama-bench -m qwen3-0.6b-iq2xs.gguf -m qwen3-0.6b-iq4xs.gguf \
            -p 0   -n 128 -r 10 -ngl 99 -fa on     # 纯 decode
llama-bench -m qwen3-0.6b-iq2xs.gguf -m qwen3-0.6b-iq4xs.gguf \
            -p 512 -n 0   -r 10 -ngl 99 -fa on     # 纯 prefill
```

`-ngl 99`、`-fa on`、`-t`、`-b`、`-ub` 两个模型**必须完全一致**，任何一项不同则该次测量作废。

### 2.4 一个必须先说的污染源（本轮新发现）

`uzu` 的 `BENCHMARKS.md` 定了一条规则：**GEMV 基准的 working set 必须 ≥ 256 MiB，否则测的是 SLC 带宽。**

用这条规则检查 0.6B：

| 编码 | 权重字节 | vs 256 MiB |
|---|---|---|
| IQ2_XS (2.3125 bpw) | 0.6e9 × 2.3125 / 8 = **173 MB** | **低于阈值** |
| IQ4_XS (4.25 bpw) | 0.6e9 × 4.25 / 8 = **319 MB** | 高于阈值 |

**2-bit 落在阈值下、4-bit 落在阈值上。** 这个不对称会**系统性偏袒 2-bit**——因为 2-bit 的那 173 MB 有一部分可能驻留 SLC，而 4-bit 的 319 MB 更彻底地打到 DRAM。

**这正好是"R > 1.05"最可能的来源之一**（见 §5 纠错 1）。处置：

- **方案 B1**：在 0.6B 上接受这个偏差，但**必须**在报告里把它写成已知污染项，且不得用这组数据支持"2.3 bpw 值不值"。
- **方案 B2（若一定要在 0.6B 上做）**：用合成大张量补齐。取 `N=K=30500` 时 2-bit 才到 268 MB，而 fp32 参考张量要 4.6 GB——**装得下，但这不是在测模型，是在测张量**。
- **方案 B3**：直接换一个 1.5–3B 的模型（2 bpw 下 ≥ 400 MB），但那要下载，未授权。

**结论：0.6B 不适合回答"2.3 bpw 值不值"。它能回答的是"2-bit kernel 相对 4-bit kernel 的兑现率"，而这个问题在 Stage A（MLX 内部）已经能问，且不受跨 backend 干扰。**

### 2.5 预注册判据

```
B   = 实测文件字节（GGUF 文件大小，不是名义 bpw）
R_B = (t_4bit / t_2bit) / (B_4bit / B_2bit)
```

| R_B | 判定 | 动作 |
|---|---|---|
| > 1.05 | **触发四项强制对照**（见 §5 纠错 1），**不得直接判"4-bit 基线异常"** | C1–C4 |
| 0.85 – 1.05 | 位宽收益基本兑现 | 进 Stage C-L0 |
| 0.45 – 0.85 | 部分兑现，ALU 开始咬 | 报出 ALU 预算反推 |
| < 0.45 | 否证：2.3 bpw 在 M4 上不划算 | 停在 4.25 bpw |

**并且**：无论 R_B 多少，都必须同时报 prefill 与 decode 两组，不许合并成一个"加速比"。

---

## 3. Stage A（今天可跑，但仍需授权）

保留 R3 的四段结构，此处只补 0.6B 上的实际形状：

| 段 | 对象 | 位宽 | 形状 | 判据 |
|---|---|---|---|---|
| A1 | MLX 内部 | bf16 / affine 2b / 4b / 8b，g64 | 合成 `N=K=29300`（见下方 working set 约束） | `R = (t_4/t_2)/(B_4/B_2)` |
| A2 | MLX 内部 | 同上 | 真实 `q_proj` 5120×5120 等价（0.6B 是 1024×1024） | 同上 |
| A3 | 质量 | 同上 | `rel_rms(W − dequant(quant(W)))` | 相对比较，无绝对阈值 |

**A1 的 working set 问题同样存在**：MLX affine 2b g64 = 2.5 bpw，`N=K=8192` 只有 20 MB。**必须把形状放大到 ≥ 256 MiB**，即 `N×K ≥ 8.6e8` → `N=K≈29300`。这在 24 GB 上可行（2.5 bpw 时 268 MB，bf16 参考 1.7 GB），但要显式分配。

**这条规则 R3 写了、R2 没写，本轮把它变成硬约束：任何 GEMV 计时，working set < 256 MiB 一律无效。**

---

## 4. Stage C：三层质量门

Issue 的裁决原文："不接受单靠 5-shot MMLU 或 ARC/HellaSwag/PIQA/WinoGrande 证明 2b 能力"。据此重设计。

### 4.0 三条总则

1. **参照必须是同 checkpoint 的另一种编码，不是 BF16。** 0.6B 上 BF16 与 2-bit 的差距（R2 记录为 ~0.77 nats / 20.7%）会把编码差异完全淹没。BF16 只用作 L0 的数值真值。
2. **判据是配对差值的 CI，不是绝对分数。** 三层里没有一条是"PPL < X 通过"或"top1 > 90% 通过"。
3. **0.6B 只检验机制，不外推 27B。** 这一条要写进每一层的小标题，不是脚注。

### 4.1 L0 数值门（今天可做）

| 指标 | 定义 | 参照 |
|---|---|---|
| 重构 `rel_rms` | `‖W − dequant(quant(W))‖ / ‖W‖` | 同一 tensor 的另一种编码 |
| 重构 `cos` | 逐输出行余弦的均值 | 同上 |
| kernel 一致性 | `max_abs`、`rel_rms`、**逐元素相对误差的 99.9 分位** | fp32 参考 GEMM |
| 分布尾部 | 每行 `max|level|` 的直方图、饱和比例 | 同上 |

**分组报，不报总平均**：`q/k/v/o`、`gate/up/down`、`embed`、`head` 分开。R2 已知"误差由 layer 0–4 决定"只在特定图成立，所以**按层也要分开**。

**L0 通过不蕴含 L1 通过**——R2 已有硬证据：隐藏态 `rel_rms` 改善 25% 时 ΔNLL 只改善 16%，**状态 MSE 会高估能力**。

### 4.2 L1 分布门（勉强可做）

| 指标 | 说明 |
|---|---|
| ΔNLL（配对） | 同窗口、同 token 数，逐窗配对，报 **bootstrap 95% CI**（10,000 次） |
| PPL | 由 ΔNLL 换算，**只作展示**，不作判据 |
| KL(BF16 ‖ q) | 逐 token，报中位数与 90 分位 |
| top-1 一致率 | 同上，**不设 90% 阈值** |
| 退化 | `distinct-4`、最长重复子串长度、连续相同 token 的最大游程 |

**样本量问题（必须写在前面）**：

```
CI 半宽 = 1.96 × σ_窗 / √n
本地最大 n = 64 窗
若 σ_窗 ≈ 0.2 nats：CI 半宽 ≈ 0.049 nats
要 ±0.02 nats：需 n ≈ 385 窗（本地不足 6 倍）
```

**所以 L1 只能判"有没有 >0.1 nats 的效应"，不能判"0.02 nats 的差异"。** R2 已记录 `<0.01–0.02 nats` 的改善不可选型，这一层恰好覆盖不到那个区间——这是**已知的能力边界**，不是缺陷。

**参照选择**：优先"同 checkpoint 的 IQ2_XS"（依赖 Stage B，未就绪），退而用"同 checkpoint 的 MLX affine 2b g64"（今天可得）。**报告里必须写明用的是哪个参照**，两者不可互换。

### 4.3 L2 配对任务门（需下载）

| 任务 | 测什么 | 为什么选它 | 0.6B 的现实 |
|---|---|---|---|
| GSM8K（子集 200 题） | 多步算术链是否崩塌 | 唯一能测"长链一致性"的低成本任务 | 接近地板，**只能看是否从 ~0 掉到 0** |
| HumanEval+ / MBPP（子集） | 代码格式与语法 | pass@1 之外看**语法可解析率** | 同上 |
| BFCL / 工具调用（格式子集） | JSON schema 遵循率 | 这是**格式能力**，0.6B 有可测信号 | 相对可用 |

**L2 的判据是"配对差值"**：同 prompt、贪心解码、同一 max_tokens，比较 A/B 在同一题上的**结果是否翻转**。报 McNemar 检验或配对比例差的 CI。

**明确写死的限制**：
- 0.6B 在 GSM8K / HumanEval 上接近地板 → 这些任务**不能**用来比较"2-bit 与 4-bit 谁更聪明"，只能用来发现"是否整体崩塌"。
- **绝对分数不构成 27B 的证据。** 任何"2b 在 MMLU 上掉了 X 分所以不可用"的推论都超出数据支持范围。
- `lm-eval 0.4.9.2` 已装，**不需要装包**；但三个任务的数据集本地都没有 → 需下载（未授权）。

### 4.4 三层的关系

```
L0 ──必要不充分──▶ L1 ──必要不充分──▶ L2
```

不是加权打分，是**逐层否决**：L0 显著更差 → 停，不必跑 L1；L1 的 CI 明显劣于参照 → 停，不必跑 L2。反向不成立：L0 好不保证 L1 好，L1 好不保证 L2 好。

---

## 5. 四条纠错（本轮新增，直接改前面的判据）

### 纠错 1：`R > 1.05` 不等于"4-bit 基线异常"

R3 的表写 `> 1.05 → 4-bit 基线未调优 / 测量异常 → 不下结论`。**这个归因是错的**：R 大于 1 也可能因为 2-bit 那条路本身更快，而不是 4-bit 有问题。

R 的定义是"实际加速 ÷ 字节比加速"。`R > 1.05` 意味着**实际比字节比预期还快**，可能来源有四种，**只有第四种是"异常"**：

| 来源 | 机制 | 是否算异常 |
|---|---|---|
| (a) kernel 实现差异 | 2-bit 用了更好的 tiling / 更少指令 | 否，是正当收益 |
| (b) 缓存驻留不对称 | 2-bit working set 落到 SLC 阈值下（§2.4） | **否，但污染结论** |
| (c) 字节口径错 | `B` 只算了 codes，漏了 scales / bias / group sums | 是，测量错误 |
| (d) 噪声 | n 太小，CI 跨 1.0 | 是，不可解释 |

**改为四项强制对照**，全做完才允许下结论：

- **C1 同 kernel 交换**：2-bit 与 4-bit 走**同一个** kernel、同一 tiling，只改 bits 参数。R 回到 ~1.0 → 是 (a)；仍 >1.05 → 是 (b) 或 (c)。
- **C2 working set 扫描**：同一编码在 64 MiB / 256 MiB / 1 GiB / 4 GiB 四档 working set 各测一次。R 随 working set 单调下降 → 确认 (b)。
- **C3 字节口径核对**：`B = q.nbytes + scales.nbytes + biases.nbytes`，与理论 `bpw × 元素数 / 8` 对比，差 > 2% 即口径错。
- **C4 样本量**：n ≥ 30 次重复，配对 bootstrap；CI 跨 1.0 → R 不可解释，只能说"没测出差异"。

**只有 C1–C4 全部排除 (b)(c)(d) 之后，才允许写"4-bit 基线异常"。** 否则结论只能是"2-bit 在该实现下更快，原因未定"。

### 纠错 2：uzu 的 bias 恒等式不能泛化

R3 写"2 bpw 下省掉 50% 的符号位开销"。**这句话没有定义对照编码，而且算错了。**

按 `trellis.h` 的实际常量重算：

- `WEIGHTS_PER_DECODE = 4` —— 一个 uint32 装 4 个权重，**解码时 8 bit/权重**。
- `MAX_BIASED_WEIGHT = 111` —— level ∈ [0, 111]，需要 7 bit。
- 不做 bias 时 level ∈ [−54, 57]，7 bit 幅度 + 1 bit 符号 = 8 bit。

**两种情况都刚好占满一个字节。所以 bias 在解码字里省 0 bit，在存储里也省 0 bit**（存储是 trellis 状态转移，2 bpw 与符号无关）。

真实代价与真实收益：

| | 内容 |
|---|---|
| 代价 | 需要 `column_group_sums`：每 token 每 matmul 读 `K/4` 个 fp32。K=5120 时 **5.1 KB/token** |
| 收益 | 解码出的 level 全非负 → 4 个 level 可以一次 `as_type<uchar4>` 重解释成 `bfloat4`，省掉 4 次有符号 int→float 转换 |

**所以它是指令数收益，不是字节收益，且付了一个额外的 fp32 buffer。** 净效果未测。

**正确的表述**（替换 R3 那句）：

> uzu 的 `WEIGHT_BIAS = 54` 把解码出的 level 从 `[−54, 57]` 平移到 `[0, 111]`，使 4 个 level 能一次 `uchar4 → bfloat4` 重解释，省掉有符号转换；代价是每 token 读一个 `K/4` 的 fp32 组和 buffer，在 epilogue 做 `level_dot = biased_dot − 54·Σgroup_sum`。**相对标准 2-bit 标量/affine 编码，它省 0 bit/权重**——后者没有独立符号平面，不存在"符号位"可省。

### 纠错 3：124.6 GB/s 不是 DRAM 计数器读数

R2 的 C-01 已把它降级为"派生值"，但**后面的文字仍然在用"有效带宽"这个词**，容易被读成实测。本轮彻底改名：

```
BW_derived = 880.8 MB / 7.07 ms = 124.6 GB/s
```

它依赖三个**都成立才有效**的假设：

1. 只算 28 个 block 的 bf16 权重（880.8 MB）；
2. **不含 output head**（311.2 MB，tied embedding）；
3. **不含 KV 读写、不含激活**。

**所以它最多是"权重流式速率的量级估计"，不是 DRAM 吞吐。** 任何"利用率"表述必须是 `BW_derived / 273 GB/s（规格峰值）`，且必须带"派生/规格"两个限定词。**禁止**写成"实测带宽 124.6"、"DRAM 利用率 46%"、"还有 54% 可收回"。

### 纠错 4：SpQt 的 M4 1.218× 是条件拟合，不是实测

说明：`grep -rn "1.218\|SpQt" research/agent-reports/workbuddy/*.md` 在已提交文件里**零命中**，这个数字只出现在 Issue 讨论里。规则仍然成立并从此生效：

- **撤下**：任何"稀疏在 M4 上能带来 1.2×"的预计值。
- **保留**：SpQt 作者报告在移动端 GPU 上有加速（E3，二手）。
- **原因**：1.218× 来自作者对多配置的拟合/外推，不是 M4 上的硬件计时；把它当预计值等于把拟合曲线当测量。
- **连带**：R2 的"临界稀疏率 ≈ 0.56"推导本身不依赖 1.218×（它只依赖字节比），**那条推导保留**；但它给出的"2.77×"必须继续标注为"GEMV 权重字节比，不是端到端"。

---

## 6. XD-3 修订：40 对交替先导

### 6.1 要解释的两个数

| 量 | 值 | 来源 |
|---|---|---|
| prefill | `116.92 / 61.32 = **1.907×**` | `stage2_1/bench_a345` vs `stage2_0/bench` |
| decode | `7.074 / 6.141 = **1.152×**` | `stage2_1/bench_a67` vs `stage2_0/decode.json` |

**这两个数必须分开解释**，不许合成一个"1.9× 位移"。1.907 与 1.152 差 65%，任何"整体变慢 1.9×"的说法都解释不了 decode。

### 6.2 设计

| 项 | 设定 |
|---|---|
| 环境 | **同一进程**内加载两个 commit 的等价闭包（不是两次启动） |
| 配对数 | **40 对先导**；若 CI 半宽 > 0.05× 则扩到 200 对 |
| 顺序 | **ABBA**：偶数对 A 先、奇数对 B 先 |
| 丢弃 | 前 5 对作 warmup，报"含/不含"两版 |
| 计时 | `mx.eval` 后同步；每对内 A、B 各 1 次，不取平均 |
| 统计 | `d_i = t_A,i − t_B,i`，bootstrap 10,000 次，报均值差、95% CI、中位数比 |
| 分离 | **prefill（T=257）与 decode（T=1，步数固定）各算一套** |
| 归因 | 预注册：同环境**复现** 1.91× → 代码；**消失** → 历史环境 |
| 若归因到代码 | `git diff 73d3379 6a5e6ea -- scripts/` 逐行定位，优先查 mask 构造、chunk 大小、cache 初始化 |

**顺序效应检验**：比较"偶数对"与"奇数对"的 `d` 均值。两者差 > CI 宽度 → 有顺序效应，必须先建模再报。

**成本**：40 对 × 2 条件 × (0.117 s prefill + 64×0.007 s decode) ≈ 45 s。200 对 ≈ 226 s。**成本不是问题，授权是。**

### 6.3 预注册的三种结局

| 结局 | 含义 | 后续 |
|---|---|---|
| prefill 1.91× 复现，decode 1.15× 复现 | 代码路径整体位移，两个量同源 | 查 `git diff` 找共同改动 |
| prefill 复现，decode 不复现 | 差异在 prefill 专有路径 | 查 mask / chunk / `prefill_step_size` |
| 两者都不复现 | 历史环境（后台负载、热态、当时 commit 的依赖版本） | 归档为"不可复现"，**永久禁止引用那两个绝对时间** |

---

## 7. 明确不做的三件事

| 不做 | 理由 |
|---|---|
| **uzu 4-bit 第三基线** | 本机**无 cargo、无 metal 编译器** → 连构建都不成立；且它是另一个引擎，引入即违反"不装新引擎"的裁决。**接受 Issue 裁决，不加。** |
| **Pi 插件** | 它是 agent 扩展，不是推理引擎，没有可比的 bench 接口。已核实页面只说 "Qwen3.8 27B Mirai Medium Quantization is recommended"，**无 trellis / bpw / M5 任何字样**（见 `2026-10-r4-upstream-900.md` §6）。 |
| **任何安装/下载/编译** | 本轮未授权。三个缺口（Xcode、llama-bench、GGUF）全部只报告，不补。 |

---

## 8. 需要裁决的问题

1. **Stage B 是否补环境？** 补 = 装 Xcode（≈7 GB）+ 构建 llama.cpp（`GGML_METAL=ON`）+ 转一个 0.6B GGUF。代价约 1–2 小时，收益是能回答"2.3125 bpw 值不值"——但 0.6B 有 §2.4 的 SLC 不对称污染，**答案会带已知偏差**。是否值得？
2. **是否接受"2.3 bpw 值不值"长期无答案？** 若 Stage B 不补，这个问题只能停在"ALU 预算 6.7 ops/weight，IQ2_XS 需 2.5–4，理论上装得进，未验证"。
3. **Stage C-L1 的样本量**：本地 64 窗只够 ±0.049 nats。是否允许从 `Salesforce/wikitext` 重抽更多窗口（下载，约 12 MB）？
4. **Stage C-L2 的任务集**：GSM8K 200 题 + HumanEval+ + BFCL 子集，三份下载合计约 20 MB。是否允许？还是只用本地已有的 IMDB/OOD 做退化检查？
5. **XD-3 是否放行 40 对先导？** 约 45 秒机时，同一进程，不动任何代码。

**Stage A / C-L0 / XD-3 三项都不需要新装任何东西，只需要运行授权。**
