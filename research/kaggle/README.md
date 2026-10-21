# Kaggle 低位宽量化验证台

`trlab_kaggle.ipynb` —— 一个**可直接上传到 Kaggle** 的独立 Notebook，用来在同一份 `Qwen/Qwen3-8B` 上产出 `IQ2_XS` 与 `IQ4_XS`，并比较**真实文件字节**、**配对 NLL**、**有限配对质量**。

当前版本 **v0.2.2**。首版（v0.1）只做环境检查与短时 smoke test；v0.2 修掉了一轮逐 cell 审查提出的 3 个 P0 + 4 个 P1；v0.2.1 又修了 S4 的两个 P0（二次量化门槛、产物复用不看指纹）和 S9 的 rho 符号；v0.2.2 修了 `--allow-requantize` 参数位置（放在位置参数之后等于没传）、S9 快照里 `status=null` 的自相矛盾，并给 S2 加了工具链预检与两个编译期开关。所有修复都有 CPU 可跑的回归测试（见第九、十节）。

S2 → S3 的分步操作清单在 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md)：数据从哪来、每步按哪个按钮、什么时候算成功、失败了看哪个日志、S4 的容量预算。本文件讲"为什么"，那份讲"怎么做"。

---

## 一、这个 Notebook 不做什么

写在前面的原因是：这些不是"还没做"，是**设计上就不做**。

| 不做 | 机制 |
|---|---|
| 自动下载大模型 | `CFG["ALLOW_NETWORK"] = False`；S3 起全部用 `need()` 检查本地路径，缺失即 `GuardError` |
| 自动 `pip install` | 只用 Kaggle 镜像自带的包；llama.cpp 从源码编译；GGUF 解析器是纯 Python，无第三方依赖 |
| 自动租 GPU / 跑实验 | 阶段开关默认全关；用户手动改 `STAGES` 后在网页上逐段执行 |
| 上传私有项目文件 | Notebook 不引用 `transformer-residual-lab` 的任何本地数据或脚本 |
| 在单张 T4 上硬加载 8B BF16 | S8 先算显存，放不下就返回 `status=skipped` 并写明所需/可用 GB，不硬撑到 OOM |

---

## 二、文件

```
research/kaggle/
├── trlab_kaggle.ipynb        # 唯一需要上传的文件（28 cells：16 code + 12 markdown）
├── KAGGLE_S2_S3_RUNBOOK.md   # S2 编译 → S3 imatrix 的分步操作清单（网页 GUI 视角）
├── test_trlab.py             # CPU-only 回归测试，不联网、不需要 GPU / llama.cpp
└── README.md                 # 本文件
```

`test_trlab.py` **不是** notebook 的副本：它把 `trlab_kaggle.ipynb` 里所有 code cell 抽出来 exec，再对函数做断言。改了 notebook 就跑得出来。

---

## 三、三步上手

1. Kaggle → `File` → `Import Notebook` → 上传 `trlab_kaggle.ipynb`
2. 右侧 `Settings`：
   - `Accelerator` = **GPU T4 ×2**（Notebook 只用第 0 张，第二张不会自动用上）
   - `Internet` = **Off**（默认不需要联网）
3. 直接 `Run All`。默认只跑 `S0_ENV` / `S1_SMOKE` / `S9_REPORT`，约 2–4 分钟，**不碰任何大文件**。

跑完后读 `S0_ENV` 的输出，它会列出 `/kaggle/input` 下每个 Dataset 的文件数、大小、以及**所有 `.gguf` 的完整路径**。把这些路径抄进配置 cell 的 `CFG`，再把要跑的阶段改成 `True`，从上到下逐段执行。

---

## 四、阶段表

| 阶段 | 默认 | 前置依赖 | 预算 | 产出 | 能回答什么 |
|---|---|---|---|---|---|
| `S0_ENV` | **开** | — | 120 s | GPU / CPU / RAM / 磁盘 / 工具链 / input 清单 | 这台机器能干什么 |
| `S1_SMOKE` | **开** | — | 240 s | 合成 GGUF 往返、fp16 vs bf16 TFLOP/s、拷贝带宽、磁盘写速 | 解析器和预算算术对不对 |
| `S2_BUILD` | 关 | llama.cpp 源码 | 2400 s | `llama-imatrix` / `llama-quantize` / `llama-perplexity` / `llama-cli` | 工具链能否就位 |
| `S3_IMATRIX` | 关 | S2 + `IMATRIX_BASE` + 校准语料 | 3600 s | `imatrix.gguf` + sha256 | 校准信息是否对等 |
| `S4_QUANT` | 关 | S2 + S3 + `BASE_GGUF`（f16/bf16） | 1800 s | 两个 GGUF + provenance + 逐文件 bpw | 能否编码、字节多少 |
| `S5_BYTES` | 关 | S4（或任意 GGUF） | 300 s | 逐张量类型字节、按 attn/ffn/embed 分类 | 哪些张量没被低位量化 |
| `S6_NLL` | 关 | S2 + S5 + 评测语料 | 5400 s | 分组配对 NLL + bootstrap CI | 2.31 bpw 差多少 nats |
| `S7_QUALITY` | 关 | S2 + S5 | 1800 s | 贪心续写一致性、repeat4 | 有没有整体崩塌 |
| `S8_BF16_ANCHOR` | 关 | 高精度 GGUF + 足够显存 | 3600 s | 锚点 NLL 或明确的 skip 记录 | 锚点能不能上，`rho` 能不能算 |
| `SX_EXPLORE` | 关 | 第三方 GGUF | 1800 s | 字节 + NLL，标 `provenance=third_party` | 这个文件本身如何（**不参与公平比较**） |
| `S9_REPORT` | **开** | — | 120 s | `report.md` + `results.json` | 汇总 |

**S2 的四个编译开关（v0.2.2 新增，都在 `CFG` 里）**

| 键 | 默认 | 为什么 |
|---|---|---|
| `BUILD_CUDA` | `True` | `False` 只构建 CPU 版。三个工具（imatrix / quantize / perplexity）都有 CPU 路径，只是慢；nvcc 不可用时预检会直接建议改这个 |
| `CUDA_ARCH` | `"75"` | T4 = sm_75。留空则用上游默认，而 `ggml/src/ggml-cuda/CMakeLists.txt` 的默认值是 `50-virtual 61-virtual 70-virtual 75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real`（CUDA ≥12.9 再加 `121a-real`）—— **九套架构**，4 核上编译量翻几倍 |
| `BUILD_JOBS` | `4` | Kaggle 只有 4 核 |
| （硬编码） | `-DLLAMA_USE_PREBUILT_UI=OFF` | 不关的话 `tools/ui` 会在**构建期**跑 `file(DOWNLOAD)` 去 HF bucket 取 UI 资产包（TIMEOUT 30/300）。Internet Off 时不会 FATAL_ERROR，只 `message(WARNING)`，但会白等几十秒到几分钟 |

`S2_BUILD` 在调 cmake 之前先跑 `_toolchain_report()`（`cmake --version` / `nvcc --version` / `g++ -dumpversion` / `gcc -dumpversion`，各 20 s 超时，拿不到就写"不可用"而不是抛），再由 `_cuda_preflight()` 判定：`BUILD_CUDA=True` 但 nvcc 不可用时直接 `GuardError`，不让 cmake 在 `FATAL_ERROR` 的几十行输出里把它淹掉。

**一个容易踩的坑**：不要为了省编译时间去关 `LLAMA_BUILD_SERVER`。上游 `tools/CMakeLists.txt` 里 `add_subdirectory(cli)` 写在 `if (LLAMA_BUILD_SERVER)` 内部，关了就连 `llama-cli` 都没有（S7 要用）。

---

## 五、需要准备的 Kaggle Dataset

全部**只读挂载**到 `/kaggle/input/<slug>/`，不占 `/kaggle/working` 的 20 GB。

| # | 内容 | 建议 | 大小 | 谁需要 |
|---|---|---|---|---|
| 1 | **base GGUF** | `Qwen3-8B` 的 f16 或 Q8_0 | f16 = 15.25 GiB；Q8_0 ≈ 8.1 GiB | S3 / S4 / S8 |
| 2 | **校准语料** `calib.txt` | wikitext-2-raw 的 **train** split | ≥ 200 KB（S3 会警告过短） | S3 |
| 3 | **评测语料** `eval.txt` | wikitext-2-raw 的 **test** split | 几十 KB 起 | S6 / S7 / S8 |
| 4 | **llama.cpp 源码** | 任意近期 master | ~50 MB | S2 |
| 5 | （可选）第三方量化 GGUF | 别人的 IQ2_XS / IQ4_XS | 2–5 GiB | SX |

**两个 base 不是同一个东西（v0.2.1）**

| CFG 键 | 谁用 | 要求 |
|---|---|---|
| `BASE_GGUF` / `BASE_KIND` | **S4 量化**、S8 锚点 | **必须是 f16/bf16**。已量化的 base 再量化需要 `--allow-requantize`，官方帮助原文：`WARNING: this can severely reduce quality compared to quantizing from 16bit or 32bit!` |
| `IMATRIX_BASE` / `IMATRIX_BASE_KIND` | **S3 算 imatrix** | 留空 = 同 `BASE_GGUF`。T4 上可以指向 Q8_0（能进显存，快几十倍） |

**T4 上的推荐配置**：

```python
CFG["IMATRIX_BASE"]      = "/kaggle/input/.../qwen3-8b-q8_0.gguf"   # imatrix 要整模型进显存
CFG["IMATRIX_BASE_KIND"] = "q8_0"
CFG["BASE_GGUF"]         = "/kaggle/input/.../qwen3-8b-f16.gguf"    # 量化 base
CFG["BASE_KIND"]         = "f16"
```

`llama-quantize` 是 **CPU 路径**，15.25 GiB 的 f16 走的是内存不是显存，T4 实例的 RAM 够。两份 base 必须来自同一 checkpoint —— S4 会核对 architecture、每个张量的名字与形状、参数量，不一致直接报错。

如果只有 Q8_0 一个文件、又只想做 smoke，必须显式设 `CFG["ALLOW_REQUANTIZE"] = True`；此时命令追加 `--allow-requantize`，产物 provenance 标 `requantized_pilot_only`，报告顶部会出现醒目声明，**不能**当作相对原始 checkpoint 的高保真低比特结论。

关于 Q8_0 当**锚点**，v0.2 起是另一条硬约束：`_anchor_decision()` 把 `BASE_KIND` 分档，`bf16/f16/fp16 → high`，`q8_0/q6_k/q5_k/q4_k_m/q4_k/q4_0 → approx`。`approx` 档默认 `status=skipped`，因为拿二次量化的文件当"原始 checkpoint BF16 真值"是错的。确实只想估个量级时显式设 `CFG["ANCHOR_ALLOW_APPROX"] = True`，结果带 `validity_scope="approx_only"`，S9 不算 `rho`。

---

## 六、为什么 IQ2_XS 必须有 imatrix

不是"通常依赖"，是**源码直接抛异常**：

```cpp
// llama-quant.cpp:882
static bool tensor_requires_imatrix(...) {
    if (tensor_name_match_token_embd(...) || tensor_name_match_output_weight(...)) return false;
    switch (dst_type) {
        case GGML_TYPE_IQ2_XS:            // ← IQ2_XS 在这里
            return true;
        ...
```

```cpp
// llama-quant.cpp:1324
if (!imatrix && tm.requires_imatrix) {
    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", ...);
    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
    throw std::runtime_error(...);
}
```

所以 `llama-quantize f16.gguf out.gguf IQ2_XS` **会直接退出、不产出文件**。S4 因此把 imatrix 设为硬前置，缺失即报错，不做"没 imatrix 也试一下"的降级。

**公平性问题**：`IQ4_XS` **不**要求 imatrix，`IQ2_XS` 要求。只给 Q2 配就是给低比特开小灶。**协议规定两者共用同一份 `imatrix.gguf`。**

还有一个连带事实：imatrix 会改变 `llama_tensor_get_type` 的分层决策（`llama-quant.cpp:688` 对前 `n_layer/8` 层按 `has_imatrix` 分支），所以两个文件的实际张量类型组合不同 —— **字节必须从文件读，不能用名义 bpw 乘参数**。这就是 S5 存在的理由。

---

## 七、为什么第三方量化不能做公平比较

一个现成的 `xxx-IQ2_XS.gguf` 里，以下信息全部未知：

- 用的哪个 base（f16？Q8_0？已经被量化过一次的？）
- imatrix 从哪来、多少 chunk、什么语料
- 量化时 llama.cpp 的 commit（分层类型分配策略会变）
- 有没有额外混入更高精度的张量

所以它的 NLL 只能说明**这个文件本身如何**，不能与自建量化放进同一张表。SX_EXPLORE 的所有结果都带 `provenance.fair_comparison = False`，并且 S9 有一条断言：

```python
assert THIRD_PARTY_IN_COMPARISON is False, "第三方量化禁止进入公平比较表"
```

**允许的用法**：确认格式可加载、量级是否合理、有没有明显崩塌。

---

## 八、预算、断点恢复与产物

### 三重硬上限

```python
CFG["MAX_WALL_SEC"]     = 8 * 3600   # Kaggle GPU 会话 9 h，留 1 h 余量
CFG["MAX_VRAM_GB"]      = 14.0       # 超了主动抛异常，不等 OOM killer
CFG["MIN_FREE_DISK_GB"] = 3.0
CFG["STAGE_BUDGET_SEC"] = {...}      # 每阶段独立预算
```

每次 `run_stage` 进入前都过一次 `guard()`。外部命令（cmake / llama-*）由 `run_cmd` 轮询 deadline，超时就 `kill`。

**`guard()` 的语义边界**（v0.2 写进 docstring）：它只能拦住"进入时已经不满足"的情况。llama.cpp 是外部进程，运行中自己分配显存，Python 侧无法阻止它 OOM。运行中的保护只有墙钟超时 + kill。

### 断点恢复：指纹，不是时间戳

```
trlab/ckpt/state.json      # {"S0_ENV": {"status":"ok","fingerprint":"9ec7..."}, ...}
trlab/out/<STAGE>.json     # 每阶段独立结果，可单独读
trlab/logs/*.log           # 外部命令的完整 stdout+stderr
```

每个阶段有输入指纹：

```
sha256(notebook_version + stage + 该阶段依赖的 CFG 子集
       + 每个输入文件的 (size, mtime_ns, 头 1 MiB 哈希[:16], 尾 1 MiB 哈希[:16])
       + 源码目录的 CMakeLists + git commit)[:32]
```

于是三种结果：

| 情况 | 行为 |
|---|---|
| `status=ok` 且指纹一致 | `RESUME`，直接用旧结果 |
| `status=ok` 但指纹变了 | `INVALIDATE`，重跑（日志里写明旧→新指纹） |
| 阶段在 `ALWAYS_RERUN` 里 | 无条件重跑 |

`ALWAYS_RERUN = {"S9_REPORT"}` —— 汇总永不停留在旧结果上。

指纹的意义在 v0.1 是缺失的：换了 `BASE_GGUF` 但阶段名没变，旧结果会被当成新的。v0.2 起 `BASE_GGUF` / `BASE_KIND` / `IMATRIX_GGUF` / 语料 / `NLL_GROUPS` / `SEED` 全部进指纹。

`status == "ok"` 的阶段重跑时自动跳过。强制全跑改 `CFG["FORCE_RERUN"] = True`。

失败不会中断后续阶段 —— 每个阶段自己捕获异常并写 `status: "error"` + traceback，S9 仍然会产出一份带错误摘要的报告。

### 失败命令不再"看起来成功"

`run_cmd` 没有 `check` 参数了：**rc ≠ 0 一律抛 `CommandFailed`**（带命令、rc、日志尾部 1800 字符）。

写文件走 `.partial` → 验证 → 原子改名：

```
llama-imatrix -o imatrix.gguf.partial     # 先写临时名
  ↓ 检查存在、≥ 100 KB、gguf_scan() 的 n_tensors > 0
os.replace("imatrix.gguf.partial", "imatrix.gguf")
```

这样"上次跑失败留下的旧文件"不可能被当成本次产物。`CODE_FINAL` 会列出所有残留 `.partial`。

### 产物

```
trlab/out/report.md        # 人读
trlab/out/results.json     # 机读，含 config_snapshot 与逐阶段状态
trlab/out/S*.json          # 逐阶段
trlab/models/*.gguf        # imatrix + 两个量化文件（约 7 GiB）
```

---

## 九、v0.2 修了什么

一轮逐 cell 审查提了 3 个 P0 + 4 个 P1。每一条都改在 notebook 里，并且都有一条 CPU 可跑的回归测试钉住。

### P0-A · 源码编译目录写进只读挂载

`build = src / "build"` 会往 `/kaggle/input` 写，而它是只读的 —— S2 必然失败，且失败原因看起来像 cmake 的问题。

改成 `BUILD = WORK / "build-llama"`，`cmake -S <src> -B <BUILD>`。`llama_bin()` 按 `CFG["LLAMA_BIN"] → BUILD/bin → BUILD → 源码树/build → PATH` 找，找不到时的报错直接写出 `BUILD` 的真实路径。

### P0-B · 显存需求被夸大约 9.5 亿倍

```python
need_bytes = ap.stat().st_size + 1 << 30     # 实际是 (size+1) << 30
```

`+` 优先级高于 `<<`。8 GiB 的文件算出来是 **8.59e9 GiB**，于是 S8 永远 `skipped`，而理由是假的。

改成纯函数，括号显式：

```python
def _est_need_bytes(size_bytes, extra_gb=2.0):
    return int(size_bytes) + int(round(float(extra_gb) * (1 << 30)))
```

`S1_SMOKE` 里有 `need_bytes_arithmetic_ok` 断言，测试里还有一条专门比较"旧写法 vs 新写法"的数量级。

### P0-C · T4 上 S9 直接 KeyError

```python
tb["bf16"]["tflops"]      # T4（CC 7.5）没有原生 bf16，这里没有 "tflops" 这个键
```

于是**默认首跑就产不出报告**。抽出 `_bench_summary()`，全部 `.get()`，并且把"bf16 不可用"当成一种正常结论渲染：

```
- bf16：**不可用** —— RuntimeError: ...
  → 这不是缺陷。T4（CC 7.5）没有原生 bf16，所以锚点只能用 f16，且 bf16 相关结论一律不可引用。
```

### P1-D · 断点恢复不看输入

见第八节。另外两处：

- `_fairness.same_base` 原来是硬编码 `True`，现在与 S3 记录的 `base_fp` / imatrix sha256 比对，比不了就给 `None` 并在 S9 里写"无法验证"。
- S9 曾经成功过就 `RESUME`，导致报告描述的是上一次的输入。现在 `ALWAYS_RERUN = {"S9_REPORT"}`。

### P1-E · 返回码被忽略

`run_cmd(..., check=False)` 时失败命令的残留旧文件会被当成功。现在没有 `check` 参数，见第八节。

S6 解析 `llama-perplexity` 日志前也要求 rc = 0，正则匹配不到 `final estimate: PPL = x +/- y` 时直接报错而不是给 `None`。

### P1-F · 分组数与显存语义

`ceil(len/n)` 在 41 段切 8 组时只得到 **7 组**，而 S6 会按 8 组算 CI。改成 `divmod`：

```python
base, rem = divmod(len(paras), n)
sizes = [base + (1 if i < rem else 0) for i in range(n)]
```

组数严格等于 `min(n_groups, 段数)`，组间段落数最多差 1。因为组间 **token 数**仍不严格相等，S6 同时给 `char_weighted_mean` 作为对照。

`MAX_VRAM_GB` 只在入口检查这件事，现在写在 `guard()` 的 docstring 里，不再假装它是运行中保护。

### P1-G · 未知 ggml type 被当 8 bpw

v0.1 的 `GGML_TYPES.get(tid, ("UNKNOWN", 1, 1))` 会按 8 bpw 算出一个**看起来正常**的字节数。现在 fail closed：

```python
class UnknownGGMLType(ValueError): ...

def ggml_type_info(tid, strict=True):
    info = GGML_TYPES.get(tid)
    if info is None:
        if strict:
            raise UnknownGGMLType(f"未知 ggml type_id={tid}。拒绝按默认尺寸假算字节 …")
        return ("UNKNOWN", None, None)
```

`gguf_scan(..., strict_types=True)` 默认抛；显式降级时 `nbytes=None`，并且 **`bpw_effective` 也给 `None`**（0.0 bpw 同样是假数）。S5 捕获后把文件列入 `byte_breakdown_unavailable`，S9 的表里显示"**否**（未知 ggml type）"。

### 三项裁决的落实

| 裁决 | 落实 |
|---|---|
| T4 上可用同源 Q8_0 base 做 pilot，但它是二次量化，不能替代同 checkpoint 的 F16/BF16 真值 | `ANCHOR_TIERS` 分档 + `ANCHOR_ALLOW_APPROX` 默认 `False`；放行时标 `validity_scope="approx_only"`，S9 不算 `rho` |
| `NLL_GROUPS = 8` 只用于 pilot | 默认值 8，S9 第 8 节第 3 条写死"只用于 pilot，正式 CI 需按独立文档重设计" |
| `SX_EXPLORE` 保留、默认关闭、严格隔离 provenance | 默认 `False`；结果带 `fair_comparison=False`；S9 开头有 `THIRD_PARTY_IN_COMPARISON is False` 断言 |

---

## 九之二、v0.2.1 修了什么

第二轮审查（Issue #5 comment 6321656242）指出 S4 还有两个 P0，外加一个 rho 符号问题。

### P0-1 · Q8_0 base 根本走不通 S4

README 原来推荐"T4 上用 Q8_0 当 base"，但 `llama-quantize` 对已经量化的张量有硬门槛。上游 `llama_model_quantize_impl` 的主循环里：

```cpp
if (ggml_is_quantized(tensor->type) && !params->allow_requantize)
    throw std::runtime_error(format("requantizing from type %s is disabled", ...));
```

`allow_requantize` 默认 `false`，所以 `llama-quantize f16.gguf out.gguf IQ2_XS` 换成 `q8_0.gguf` 就直接抛。`tools/quantize/quantize.cpp` 的帮助文本：

```
--allow-requantize
    allow requantizing tensors that have already been quantized
    WARNING: this can severely reduce quality compared to quantizing from 16bit or 32bit!
```

修复分三块：

1. **拆开两个 base**：`IMATRIX_BASE` / `IMATRIX_BASE_KIND` 与 `BASE_GGUF` / `BASE_KIND` 独立。T4 上 imatrix 用 Q8_0（进显存），量化用 f16（CPU 路径，走内存）。
2. **`_base_requant_plan()` 前置拦截**：`BASE_KIND` 是量化类型且 `ALLOW_REQUANTIZE=False` 时，在调用 `llama_bin()` 之前就 `GuardError`，报错里直接给出"BASE_GGUF 指 f16、IMATRIX_BASE 指 Q8_0"的替代方案，并引用上游帮助原文。放行时命令追加 `--allow-requantize`，产物 provenance 标 `requantized_pilot_only`。
3. **`_base_consistency()` 核对同源**：两份 base 分别 `gguf_scan`，比 architecture、张量名集合、每个张量形状、参数量。不一致直接报错——imatrix 是逐张量逐输入通道的统计量，形状对不上就没有意义。

### P0-2 · 指纹变了，S4 仍然复用旧 GGUF

v0.2 的复用条件是 `dst.exists() and size > 0 and not FORCE_RERUN`，跟输入毫无关系。于是换了 imatrix 或 base，`run_stage` 报 `INVALIDATE`、S4 也"重跑"了，但两个 GGUF 还是上一次的，而 `_fairness` 又引用新输入说"已核验"。

现在每个产物旁边有 `<name>.gguf.provenance.json`，里面记 `quant_fp`：

```
sha256(base 指纹 + imatrix sha256 + llama.cpp commit + 格式
       + 二次量化标记 + 量化参数)[:32]
```

只有 `quant_fp` 与当前输入算出来的一致才复用。不一致时旧文件被改名成 `<name>.gguf.stale-<旧fp8>`（同时删掉更早的同类 stale，每种格式最多留一份），然后重新量化、重写 provenance。回归测试 `test_s4_rebuilds_when_imatrix_changes` 连跑四次 S4：第一次造两个、第二次全复用（外部命令零调用）、**改 imatrix 后必须实质重建两个且旧文件被移走**、最后切到 `q8_0 + ALLOW_REQUANTIZE` 检查 flag 与 `base_scope`。

> 操作提示：**从 v0.2 升级到 v0.2.1 后，第一次跑 S4 会重建两个 GGUF** —— v0.2 的产物没有 provenance，无法证明它来自当前输入。旧的会被改名成 `.stale-*`，不会丢。

### 附带 · rho 的符号

v0.2 写的是 `d4 = anchor - Q4`（负向），打印却标成 `dNLL(Q4,anchor)`。比值因为分子分母同号碰巧正确，但标签与代数定义相反。现在统一成"后减前"：

```python
d4 = q4 - anchor          # 锚点 → Q4 的退化，应为正
d2 = q2 - anchor
rho = (d2 - d4) / d4
```

抽成纯函数 `_rho_from_nll()`，并且 `d4 <= 0` 时返回 `rho=None` + 原因（锚点没有比 Q4 更好，此时 rho 没有意义，而不是硬算一个数）。S9 的渲染测试 `test_s9_rho_rendering` 直接断言报告里出现 `dNLL(Q4,anchor)=+0.5000` 且不出现 `-0.5000`。

---

## 九之三、v0.2.2 修了什么

这一轮不是审查驱动的，是**在真实 Kaggle T4 上跑通 v0.2.1 默认 S0/S1/S9 之后**，为 S2 → S3 做准备时发现的四个问题。前两个是真实缺陷，后两个是"知道上游行为之后必须提前处理"。

### P0-A · `--allow-requantize` 追加在位置参数之后，等于没传

v0.2.1 加了这个 flag，但拼命令的方式是：

```python
# v0.2.1（错）
cmd = [qbin, "--imatrix", imatrix, base, out, fmt, threads]
cmd += flags          # ← --allow-requantize 落到了位置参数后面
```

上游 `tools/quantize/quantize.cpp` 的参数循环是：

```cpp
for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++)
```

**遇到第一个不以 `--` 开头的 token 就退出**，之后一律当位置参数读（model / output / ftype / nthreads）。所以写在末尾的 `--allow-requantize` 被当成"多余的位置参数"丢掉，`allow_requantize` 仍然是 `false`，`llama_model_quantize_impl` 里那行 `throw std::runtime_error("requantizing from type ... is disabled")` 照旧触发。

更糟的是 v0.2.1 的测试**把错误行为钉住了**：

```python
assert cmd[-1] == "--allow-requantize"      # ← 断言的就是"在末尾"
```

现在命令形状是 `[qbin] + flags + ["--imatrix", im, base, out, fmt, threads]`，测试改成断言**位置参数之后不允许再出现任何以 `--` 开头的 token**：

```python
i = cmd.index(base)
assert all(not str(c).startswith("--") for c in cmd[i:]), cmd[i:]
assert cmd.index("--allow-requantize") < cmd.index("--imatrix")
```

顺带记一笔：审查给的行号（1332-1333 / 127-130）与当前 master 不符（实测约 954 / 154-157），但**机制描述是对的**——所以这一条照修，只是引用行号时不能照抄。

### P0-B · `results.json` 里 `S9_REPORT.status` 是 `null`，日志却写 OK

S9 在写盘时，`run_stage` 还没把 `S9_REPORT` 自己登记进 `STATE`，于是遍历 `STAGES` 取到的记录是 `None`，落盘成 `"status": null`。日志里那一行 `[S9_REPORT] OK` 是 `run_stage` 在 `stage_fn` 返回之后才打的，两处对不上。

修复是在 S9 内部显式补一条，并且**标明这是自报**，不假装成外部登记：

```python
stage_snapshot["S9_REPORT"] = {
    "status": "ok", "elapsed_sec": None,
    "fingerprint": _fingerprint("S9_REPORT"),
    "error": None,
    "self_reported": True,
    "note": "本阶段在写盘时尚未登记进 STATE；status 由这里显式置为 ok",
}
```

`test_default_stages_complete` 现在断言三件事：`status == "ok"`、`self_reported is True`、S9 返回的 `stage_status["S9_REPORT"] == "ok"`。

### P1-C · S2 在 nvcc 不可用时会淹在 cmake 输出里

`BUILD_CUDA=True` 而镜像里没有 nvcc 时，cmake 的 `FATAL_ERROR` 会吐出几十行 `Could not find CMAKE_CUDA_COMPILER`，用户很难判断"是环境问题还是参数问题"。现在 cmake 之前先跑 `_toolchain_report()`（四个命令各 20 s 超时，失败只记 `rc=N` 或 `不可用: XxxError`，不抛），再 `_cuda_preflight()` 判定，直接给结论：

> `BUILD_CUDA=True 但 nvcc 不可用。改 CFG['BUILD_CUDA']=False —— llama-imatrix / llama-quantize / llama-perplexity 都能走 CPU，只是慢；或换一个带 CUDA 的镜像。`

`test_s2_cmake_flags` 用一个假 `run_cmd` 捕获 cmake 的 argv，断言 nvcc 缺失时抛 `GuardError` 且**一次外部调用都没有**。

### P1-D · 两个编译期开关

- `-DCMAKE_CUDA_ARCHITECTURES=75`：不指定时上游默认九套架构（见第四节表格）。4 核上这不是小差别。
- `-DLLAMA_USE_PREBUILT_UI=OFF`：`tools/ui/CMakeLists.txt` 的 `llama-ui-assets` target 在**构建期**跑 `scripts/ui-assets.cmake`，`HF_ENABLED=ON` 时用 `file(DOWNLOAD ...)` 去 HF bucket 取 `dist.tar.gz`。Internet Off 的环境下它不会失败（只 `message(WARNING)` 然后生成空资产表），但会白等。

这两条都只在 `CFG["BUILD_CUDA"]` / 无条件路径上各加一个 flag，没有改流程。

---

## 十、怎么跑测试

```bash
python3 research/kaggle/test_trlab.py
```

不需要 GPU、不需要 torch、不需要 llama.cpp、不联网、不下载。它在一个临时目录里把 notebook 的 16 个 code cell 按顺序 exec 一遍，然后跑 31 项断言。当前结果：**31/31 通过，约 10 秒**。

| 测试 | 钉住的问题 |
|---|---|
| `test_need_bytes_arithmetic` | P0-B：括号优先级 |
| `test_anchor_decision_t4_skips` | 8B f16 在 15.0 GiB 可用显存上必须 `skipped` |
| `test_anchor_decision_q8_is_approx` | Q8 默认拒绝；显式放行后 `approx_only` |
| `test_anchor_decision_a100_ready` | 40 GB 上同一份锚点 `high_precision` |
| `test_bench_summary_bf16_error` | P0-C：bf16 报错时不能 KeyError |
| `test_bench_summary_degenerate_inputs` | `{}` / `None` / `{"cuda": False}` / `tflops=0` 都不能崩 |
| `test_gguf_unknown_type_fails_closed` | P1-G：writer 和 scanner 两侧都 strict |
| `test_gguf_roundtrip_bytes` | 合成 GGUF 往返字节逐项相等，IQ2_XS=2.3125 / IQ4_XS=4.25 |
| `test_gguf_vocab_array_skip` | 15 万字符串的词表数组必须 seek 跳过（真实模型头才会暴露） |
| `test_split_eval_exact_groups` | P1-F：8 种段落数/组数组合，组数严格、无丢失、组间最多差 1 段 |
| `test_fingerprint_invalidation` | P1-D：改 CFG 指纹必变，改回来必还原 |
| `test_run_cmd_nonzero_raises` | P1-E：rc=3 必须抛 `CommandFailed` 且带日志尾部 |
| `test_base_requant_plan` | **P0-1**：q8_0 默认 blocked、放行后带 `--allow-requantize`、f16 无 flag |
| `test_quant_cmd_shape` | **P0-A**：`--allow-requantize` 必须在位置参数**之前**；位置参数之后不能再出现任何选项 |
| `test_quant_fp_binds_inputs` | **P0-2**：base/imatrix/commit/格式/二次量化/参数任一变化都必须换指纹 |
| `test_s4_blocks_quantized_base` | 量化 base 未放行时，**在碰 llama-quantize 之前**就报错 |
| `test_s4_rebuilds_when_imatrix_changes` | **P0-2**：连跑四次 S4，改 imatrix 后必须实质重建两个 GGUF、旧文件改名 `.stale-*`、provenance 更新 |
| `test_t4_split_base_path` | **P0-1**：imatrix 用 Q8_0、量化用 f16，结构核对通过且不需要 requantize |
| `test_base_consistency` | 架构/张量名/形状/参数量任一不同都要判定为不同源 |
| `test_rho_sign` | rho 的 d4 必须为正；锚点不如 Q4 时 `rho=None` 而不是硬算 |
| `test_s9_rho_rendering` | S9 报告里 `dNLL(Q4,anchor)=+0.5000`，不能是 `-0.5000` |
| `test_s9_marks_requantized_base` | 二次量化的 base 必须在报告顶部醒目声明，高精度 base 时不能出现该声明 |
| `test_cuda_preflight` | **P1-C**：nvcc 缺失 + `BUILD_CUDA=True` → `GuardError`；`BUILD_CUDA=False` 不拦；nvcc 正常时放行 |
| `test_toolchain_report` | 四个探测命令都失败时只记录不抛（假 PATH 下不崩） |
| `test_s2_cmake_flags` | **P1-D**：cmake argv 含 `-DLLAMA_USE_PREBUILT_UI=OFF` / `-DCMAKE_CUDA_ARCHITECTURES=75` / `-DGGML_CUDA=ON`；build 不落源码树；`BUILD_CUDA=False` 时不传架构；nvcc 缺失时零外部调用 |
| `test_paired_bootstrap_edges` | n=0 / n=1 返回 `ci95=None`；全正差值给不含 0 的 CI |
| `test_ggml_type_table` | 7 个关键 type 的块尺寸/bpw |
| `test_default_stages_complete` | **默认 S0/S1/S9 三个都必须 `ok`，S9 报告必须有效**；且 `results.json` 里 `S9_REPORT.status` 不能是 `null`（**P0-B**） |
| `test_s9_report_bf16_both_ways` | bf16 报错 / bf16 正常 / 无 torch 三条渲染分支 |
| `test_t4_mock_end_to_end` | 把 GPU 微基准换成 T4 结果，重跑 S1→S9，报告必须写出"bf16 不可用" |
| `test_a100_mock_end_to_end` | 反例：bf16 可用时不能还打印"不可用"，且要给出比值 |

`test_default_stages_complete` 不是装饰。v0.2 加 P1-G 时，`gguf_write_synth` 变成 strict，S1 里"构造一个含未知 type 的坏文件"那行自己就抛了；而 `run_stage` 会吞掉异常只记 `status=error`，于是默认首跑 S1 静默失败、S9 报告里写 `SMOKE_OK = False`。这条测试就是为这种情况写的。

---

## 十一、已知限制

1. **T4 没有原生 bf16**（CC 7.5）。S1 会实测 fp16 vs bf16 的 TFLOP/s 比值；比值明显小于 1 就说明 bf16 走了模拟路径。锚点因此只能用 **f16**，与论文口径有偏差，S8 会把这条写进 `dtype_deviation`。
2. **单张 T4 装不下 8B 高精度权重**：f16 = 15.25 GiB，可用显存约 15.0 GiB，且未计 KV 与 CUDA context。S8 默认 `status=skipped`，并给出 `required_gb` / `vram_usable_gb` 两个具体数字。
3. **隔离只做到内容级**：S6 用前 200 字符做行级重叠检测。如果校准集与评测集来自同一文档的不同段落，这个方法检测不出来 —— 必须靠 `CALIB_SOURCE` / `EVAL_SOURCE` 声明不同 split。
4. **`/kaggle/working` 约 20 GB**。两个 GGUF 约 2.2 + 4.0 GiB，imatrix 只有几十 MB，余量够；不要把 base 也放进去。指纹不符的旧产物会以 `.stale-*` 形式留一份（见第 12 条）。
5. **S7 样本量极小**（8 prompt × 32 token）。只能发现整体崩塌，不能排序细微质量；`repeat4` 是退化信号，不是质量分数。
6. **NLL 的 CI 跨 0 不等于等价**。`paired_bootstrap` 只给区间，判读在 S9 里写死为"未测出差异"。
7. **不能外推**。单张 T4、8B、单一语料的结果不适用于 27B，也不适用于 M4 Pro。**吞吐不可跨平台迁移，只有格式与质量可以。**
8. **`guard()` 不能阻止运行中 OOM**。它只在阶段入口检查，llama.cpp 的显存分配在外部进程里。
9. **未知 ggml type 的文件没有字节拆分**。这是刻意的：宁可不给数，也不给 8 bpw 的假数。遇到这种情况需要升级 `GGML_TYPES` 表（对照 `ggml-common.h` 的结构体大小）。
10. **量化 base 必须是 f16/bf16**。`llama-quantize` 对已量化张量默认拒绝（`requantizing from type ... is disabled`），S4 会在 Python 侧先拦住。`ALLOW_REQUANTIZE=True` 只用于 smoke，产物标 `requantized_pilot_only`，不能当高保真结论。
11. **imatrix 与量化 base 可以不同文件，但必须同源**。S4 核对 architecture / 张量名 / 形状 / 参数量；这只保证结构一致，不能证明是同一个 checkpoint 的权重（同一模型的两个不同 finetune 也能通过）。`IMATRIX_BASE_KIND` 是量化类型时，报告里会声明"imatrix 的激活是对高精度的近似"。
12. **旧产物会留在 working**。指纹不符的 GGUF 被改名成 `.stale-<fp8>`，每种格式最多留一份。确认新结果没问题后可以手动删，`CODE_FINAL` 会把它们列出来。
13. **不能关 `LLAMA_BUILD_SERVER`**。上游 `tools/CMakeLists.txt` 把 `add_subdirectory(cli)` 写在 `if (LLAMA_BUILD_SERVER)` 里，关了就没有 `llama-cli`，S7 会缺工具。省编译时间请改 `CUDA_ARCH` / `BUILD_JOBS`，别动 SERVER。
14. **S2 的编译时间与峰值内存全是估算**。`2400 s` 预算、`1.5–3.5 GiB` build 目录、`8–16 GiB` 峰值 RAM 都没有在真实 Kaggle T4 上验证过（本轮不允许启动实例）。runbook 第六节给了止损值，超了就降 `BUILD_JOBS` 或改 `BUILD_CUDA=False`。
15. **`-DLLAMA_USE_PREBUILT_UI=OFF` 依赖上游有这个选项名**。它是 `tools/ui/CMakeLists.txt` 里定义的 option；如果将来改名，cmake 会忽略这个未知的 `-D` 变量（只警告），后果回到"构建期尝试下载 UI"——不会失败，只是慢。runbook 的核对清单里有这一项。

---

## 十二、本地 dry-run 与测试记录

本机无 GPU、无 torch、无 llama.cpp。以下全部实跑过：

| 验证项 | 结果 |
|---|---|
| 16 个 code cell 全部 `compile()` 通过 | ✅ |
| 按顺序执行全部 cell（默认阶段） | ✅ 16/16 |
| **默认 S0 / S1 / S9 三个都 `status=ok`** | ✅ S9 产出 2.0 KB `report.md` + 6.0 KB `results.json` |
| 第二次执行 | ✅ 全部 `RESUME`（S9 例外，强制重跑） |
| 全阶段开启但无任何输入文件 | ✅ 8 个阶段干净 `GuardError`，S9 仍出报告，无 traceback 崩溃 |
| T4 mock（bf16 报错）走完 S1→S9 | ✅ 报告写出"bf16：**不可用**"+"这不是缺陷" |
| A100 mock（bf16 正常）走完 S1→S9 | ✅ 报告写出"bf16：195.0 TFLOP/s（相对 fp16 比值 1.026）" |
| 合成 GGUF 往返（F16/IQ2_XS/IQ4_XS/Q4_K/F32） | ✅ 字节逐项相等，bpw = 2.3125 / 4.25 |
| 15 万字符串词表数组的 GGUF 头解析 | ✅ `header_bytes == file_bytes - tensor_bytes`，远低于 20 s 上限 |
| `split_eval` 分组（8 种组合） | ✅ 组数严格 = min(n_groups, 段数)，无重叠、无丢失 |
| `paired_bootstrap` | ✅ n=8 给 CI；n=1 / n=0 返回 `ci95=None` 而不是崩 |
| 未知 type 的 GGUF | ✅ strict 抛 `UnknownGGMLType`；降级时 `nbytes=None` 且 `bpw_effective=None` |
| S4 复用逻辑（4 次连跑） | ✅ 造 → 全复用（外部命令零调用）→ 改 imatrix 必重建两个 + 旧文件改名 `.stale-*` → 切 `q8_0+ALLOW_REQUANTIZE` 带 flag |
| S4 拒绝量化 base | ✅ `GuardError` 在 `llama_bin()` 之前抛出，报错含替代方案与上游原文 |
| T4 分 base 路径（imatrix=Q8_0 / 量化=f16） | ✅ 结构核对通过、`requant=primary`、命令无 `--allow-requantize` |
| S9 rho 渲染 | ✅ `dNLL(Q4,anchor)=+0.5000`，无负号版本 |
| S9 快照自一致性 | ✅ `results.json` 的 `S9_REPORT.status = "ok"` + `self_reported: true` |
| S2 命令形状 | ✅ flag 全在位置参数之前；nvcc 缺失时零外部调用即报错 |
| `test_trlab.py` | ✅ **31/31，约 10 秒** |

**没有**验证的：S2 的实际编译、S3/S4 的真实量化、S6 的 `llama-perplexity` 真实输出解析（正则已用真实格式字符串测过，但没跑过真实二进制）。这三项必须在 Kaggle 上首跑时人工确认。

**没有**验证的还有 S2 的预算数字（编译时长、build 目录大小、峰值内存）、`-DLLAMA_USE_PREBUILT_UI=OFF` 在当前 pin 上的实际效果、以及 `llama-imatrix` 在 T4 上跑 8B Q8_0 的耗时。runbook 第八节列了完整清单。

---

## 十三、如果只想先确认环境

保持 `STAGES` 里只有 `S0_ENV` / `S1_SMOKE` / `S9_REPORT` 为 `True`，其余全 `False`，`Run All`。约 2–4 分钟，不碰大文件、不联网。

**如果 `S0_ENV` 说 `/kaggle/input` 下没有 `.gguf`，那 S3–S7 一定跑不了 —— 这是设计如此，不是 bug。**

确认环境没问题之后，**不要**把 `STAGES` 全改成 `True` 再 `Run All`。下一步按 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md) 走：先只开 `S2_BUILD`，编译过了再开 `S3_IMATRIX`，每一段都有停机点和人工核对清单。
