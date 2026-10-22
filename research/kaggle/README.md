# Kaggle 低位宽量化验证台

`trlab_kaggle.ipynb` —— 一个**可直接上传到 Kaggle** 的独立 Notebook，用来在同一份 `Qwen/Qwen3-8B` 上产出 `IQ2_XS` 与 `IQ4_XS`，并比较**真实文件字节**、**配对 NLL**、**有限配对质量**。

当前版本 **v0.3.1**。首版（v0.1）只做环境检查与短时 smoke test；v0.2 修掉了一轮逐 cell 审查提出的 3 个 P0 + 4 个 P1；v0.2.1 修了 S4 的两个 P0（二次量化门槛、产物复用不看指纹）和 S9 的 rho 符号；v0.2.2 修了 `--allow-requantize` 参数位置（放在位置参数之后等于没传）、S9 快照里 `status=null` 的自相矛盾，并给 S2 加了工具链预检与两个编译期开关；v0.3.0 把 S2 收敛成一个开关 `RUN_S2`（notebook 自己下载固定 commit 的源码并校验 sha256，用户不再需要在 Mac 上下载、解压、传 Dataset）；**v0.3.1 是真实 Kaggle 失败的热修**：`CUDA::cuda_driver` IMPORTED target 找不到导致 CMake Generate 失败，默认改传 `-DGGML_CUDA_NO_VMM=ON`，同时修掉把该错误误判成"找不到 nvcc"的分类器。所有修复都有 CPU 可跑的回归测试（见第九、十节）。

S2 → S3 的分步操作清单在 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md)：现在以**一页三步网页操作**开头（改一个开关 → Internet=On → Run All），手工 tar / Dataset 路线降级为断网备选。本文件讲"为什么"，那份讲"怎么做"。

---

## 一、这个 Notebook 不做什么

写在前面的原因是：这些不是"还没做"，是**设计上就不做**。

| 不做 | 机制 |
|---|---|
| 自动下载大模型 | S3 起全部用 `need()` 检查本地路径，缺失即 `GuardError`。**唯一的联网路径**是 `RUN_S2 = True` 时下载 36.6 MiB 的 llama.cpp 源码 tarball（sha256 预登记并强制校验）；`RUN_S2` 默认 `False` |
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
2. 右侧 `Settings`：`Accelerator` = **GPU T4 ×2**（Notebook 只用第 0 张，第二张不会自动用上）；`Internet` = **Off**（默认不需要联网）
3. 直接 `Run All`。默认只跑 `S0_ENV` / `S1_SMOKE` / `S9_REPORT`，约 2–4 分钟，**不碰任何大文件、不联网**。

**要连 llama.cpp 一起编好**（v0.3.0 起只改一个开关）：

1. 右上角 `Edit` → 找到 `RUN_S2 = False` 这一行，改成 `RUN_S2 = True`
2. 右侧 `Settings` → `Internet` = **On**
3. `Run` → `Run all`。S2 会自己下载固定 commit 的源码（36.6 MiB）、校验 sha256、解压到 `WORK/src/`、编译出四个二进制。约 15–30 分钟（含编译，**估算值**）。

`RUN_S2 = False` 是默认值，且 `_apply_run_switches()` 在它为 `False` 时**一个字段都不改** —— 导入 notebook 或误点 `Run All` 都不会悄悄联网。想下载必须显式改这一行，并由 `test_s2_default_off_no_network` 钉住。打开 `RUN_S2` 也**只**开 `S2_BUILD`，`S3`–`S8` 保持 `False`（`test_s2_run_switch_only_turns_on_s2` 钉住）。

跑完后读 `S0_ENV` 的输出，它会列出 `/kaggle/input` 下每个 Dataset 的文件数、大小、以及**所有 `.gguf` 的完整路径**。把这些路径抄进配置 cell 的 `CFG`，再把要跑的阶段改成 `True`，从上到下逐段执行。

---

## 四、阶段表

| 阶段 | 默认 | 前置依赖 | 预算 | 产出 | 能回答什么 |
|---|---|---|---|---|---|
| `S0_ENV` | **开** | — | 120 s | GPU / CPU / RAM / 磁盘 / 工具链 / input 清单 | 这台机器能干什么 |
| `S1_SMOKE` | **开** | — | 240 s | 合成 GGUF 往返、fp16 vs bf16 TFLOP/s、拷贝带宽、磁盘写速 | 解析器和预算算术对不对 |
| `S2_BUILD` | 关（`RUN_S2=True` 时自动开） | **不需要任何输入**：源码由 S2 自己下载 | 2400 s | `llama-imatrix` / `llama-quantize` / `llama-perplexity` / `llama-cli` | 工具链能否就位 |
| `S3_IMATRIX` | 关 | S2 + `IMATRIX_BASE` + 校准语料 | 3600 s | `imatrix.gguf` + sha256 | 校准信息是否对等 |
| `S4_QUANT` | 关 | S2 + S3 + `BASE_GGUF`（f16/bf16） | 1800 s | 两个 GGUF + provenance + 逐文件 bpw | 能否编码、字节多少 |
| `S5_BYTES` | 关 | S4（或任意 GGUF） | 300 s | 逐张量类型字节、按 attn/ffn/embed 分类 | 哪些张量没被低位量化 |
| `S6_NLL` | 关 | S2 + S5 + 评测语料 | 5400 s | 分组配对 NLL + bootstrap CI | 2.31 bpw 差多少 nats |
| `S7_QUALITY` | 关 | S2 + S5 | 1800 s | 贪心续写一致性、repeat4 | 有没有整体崩塌 |
| `S8_BF16_ANCHOR` | 关 | 高精度 GGUF + 足够显存 | 3600 s | 锚点 NLL 或明确的 skip 记录 | 锚点能不能上，`rho` 能不能算 |
| `SX_EXPLORE` | 关 | 第三方 GGUF | 1800 s | 字节 + NLL，标 `provenance=third_party` | 这个文件本身如何（**不参与公平比较**） |
| `S9_REPORT` | **开** | — | 120 s | `report.md` + `results.json` | 汇总 |

**钉死一个 commit**

上游没有 tag 可用（llama.cpp 只有 build number），所以必须按 commit 固定。本文所有行号、CMake 选项名、默认架构列表都对着下面这个 commit 核对过；`S2_PINNED` 把同一个 commit 连同 tarball 的 sha256、字节数、解压目录名一起写死在 notebook 里：

```
commit  d454a204ef2b63af357744809c05af897611a72e   (2026-10-21)
tarball https://github.com/ggml-org/llama.cpp/archive/d454a204ef2b63af357744809c05af897611a72e.tar.gz
大小    38,419,835 B (36.6 MiB)
sha256  f565ee976503e0b66997a05554d4881a252ced7687811f2a41df771203b54d7f
```

v0.3.0 起 S2 的联网路径**只有一条**：下载这个 tarball → 比对 sha256 → 安全解压（拒绝 `..`、绝对路径、符号链接）→ 校验顶层目录名与 pin 一致 → 编译。**没有** `git clone` 分支 —— 旧版 `ALLOW_NETWORK=True` 走 `git clone --depth 1`，拿的是当时的 HEAD，与"固定 commit"直接冲突，等于让"固定版本"这句话失效，那是设计错误而不是缺功能。

commit 的校验强度分三档，写进结果的 `commit_source` / `commit_verified`：

| 来源 | `commit_source` | `commit_verified` | 证明力 |
|---|---|---|---|
| 目录里有 `.git` 且 `git rev-parse` 与 pin 一致 | `git` | `True` | 最强 |
| tarball digest 与预登记 sha256 一致（v0.3.0 自动路径） | `pinned_tarball` | `True` | 证明"这份 tar.gz 就是当初 pin 的那份"；不等于 git 校验 |
| 只有 `CFG["LLAMA_COMMIT"]`，既无 `.git` 也无 digest（手动挂载的源码） | `declared` | `False` | 只证明"我声称" |

三档都可能在 S9 报告第 2 节看到原文。digest 不符时 S2 **拒绝解压并删掉下载的包**，不会"先解压看看"。

**S2 的开关与编译参数（v0.3.0：一个开关 `RUN_S2`，其余自动）**

| 键 | 默认 | 谁设 | 为什么 |
|---|---|---|---|
| `RUN_S2` | `False` | **用户** | 唯一需要手改的开关。`True` 时 `_apply_run_switches()` 一次填好下面全部；`False` 时一个字段都不改 |
| `S2_AUTO_FETCH_SOURCE` | `False` | `RUN_S2` | `True` 才允许联网下载源码 |
| `LLAMA_COMMIT` | `""` | `RUN_S2` | 自动填成 `S2_PINNED["commit"]` |
| `BUILD_CUDA` | `True` | `RUN_S2` | `False` 只构建 CPU 版。三个工具（imatrix / quantize / perplexity）都有 CPU 路径，只是慢；nvcc 不可用时预检会直接建议改这个 |
| `CUDA_ARCH` | `"75"` | `RUN_S2` | T4 = sm_75。留空则用上游默认，而 `ggml/src/ggml-cuda/CMakeLists.txt` 的默认值是 `50-virtual 61-virtual 70-virtual 75-virtual 80-virtual 86-real 89-real 90-virtual 120a-real`（CUDA ≥12.9 再加 `121a-real`）—— **九套架构**，4 核上编译量翻几倍 |
| `CUDA_NO_VMM` | `True` | `RUN_S2` | **v0.3.1 真实失败热修**：Kaggle T4 上 `find_package(CUDAToolkit)` 解析不出 `CUDA::cuda_driver` IMPORTED target，`ggml/src/ggml-cuda/CMakeLists.txt:181` 的 `target_link_libraries` 直接 CMake Error。上游 `ggml/CMakeLists.txt:205` 有 `option(GGML_CUDA_NO_VMM ... OFF)`，`ggml-cuda/CMakeLists.txt:178-182` 据此跳过 libcuda 直链。**代价**：显存池从 `ggml_cuda_pool_vmm` 回退到 `ggml_cuda_pool_leg`（cudaMalloc 池），算力内核与数值不受影响 |
| `BUILD_JOBS` | `4` | `RUN_S2` | Kaggle 只有 4 核 |
| （硬编码） | `-DLLAMA_USE_PREBUILT_UI=OFF` | — | 这个 option 在顶层 `CMakeLists.txt:138` 声明、默认 `ON`，经 `tools/ui/CMakeLists.txt` 转成 `HF_ENABLED` 传给 `scripts/ui-assets.cmake`。`llama-ui-assets` 是 `ALL` target，只要 `LLAMA_BUILD_SERVER=ON`（默认）就会在**构建期**执行；本地没有 `dist/` 又没走 npm 构建时，会 `file(DOWNLOAD)` 去 HF bucket 取 `dist.tar.gz`（sha256 那次 `TIMEOUT 30`，包体那次 `TIMEOUT 300`）。Internet Off 时**不会** FATAL_ERROR，只 `message(WARNING)` 然后生成空资产表，但会白等 |

`S2_BUILD` 在调 cmake 之前先跑 `_toolchain_report()`（`cmake --version` / `nvcc --version` / `g++ -dumpversion` / `gcc -dumpversion`，各 20 s 超时，拿不到就写"不可用"而不是抛），再由 `_cuda_preflight()` 判定：`BUILD_CUDA=True` 但 nvcc 不可用时直接 `GuardError`，不让 cmake 在 `FATAL_ERROR` 的几十行输出里把它淹掉。

**重复 `Run All` 不会重下**：`WORK/src/` 里已解压的目录 + `src/pinned_source.json` 的 sha256 与当前 pin 一致就完全离线复用；tarball 还在且 digest 一致就只重新解压；包被改坏（哪怕只差一个字节）就删掉重下。三级缓存由 `test_s2_repeat_run_uses_cache` 钉住。

**失败时不让人读编译错误**：`_explain_s2_failure()` 把 `CommandFailed` 的日志尾部翻译成"症状 → 处置"一句话（nvcc 与 host g++ 不兼容 / 找不到 nvcc / 磁盘满 / 内存不足 / cmake 配置阶段 / 其他），并给出日志路径；S9 报告第 2 节单独记 S2 的状态与四个二进制路径。`test_s2_failure_message_actionable` 钉住六种症状。

**一个容易踩的坑**：不要为了省编译时间去关 `LLAMA_BUILD_SERVER`。上游 `tools/CMakeLists.txt` 里 `add_subdirectory(cli)` 写在 `if (LLAMA_BUILD_SERVER)` 内部，关了就连 `llama-cli` 都没有（S7 要用）。

---

## 五、需要准备的 Kaggle Dataset

全部**只读挂载**到 `/kaggle/input/<slug>/`，不占 `/kaggle/working` 的 20 GB。

| # | 内容 | 建议 | 大小 | 谁需要 |
|---|---|---|---|---|
| 1 | **base GGUF** | `Qwen3-8B` 的 f16 或 Q8_0 | f16 = 15.25 GiB；Q8_0 ≈ 8.1 GiB | S3 / S4 / S8 |
| 2 | **校准语料** `calib.txt` | wikitext-2-raw 的 **train** split | ≥ 200 KB（S3 会警告过短） | S3 |
| 3 | **评测语料** `eval.txt` | wikitext-2-raw 的 **test** split | 几十 KB 起 | S6 / S7 / S8 |
| 4 | **llama.cpp 源码** | **不需要**（v0.3.0：`RUN_S2=True` 时 notebook 自己下载） | 断网备选：把源码挂上来，设 `CFG["LLAMA_SRC"]` | ~36 MB | S2 |
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

顺带核对了行号（对着钉死的 commit `d454a204ef2b63af357744809c05af897611a72e`，见第四节）：`src/llama-quant.cpp:1332-1333` 是 requantize 的 throw，`tools/quantize/quantize.cpp:127-130` 是帮助文本，`tools/quantize/quantize.cpp:410` 是那个参数循环 —— **与审查给的行号一致**。行号会随上游漂移，引用时请对着自己 pin 的 commit 复核。

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

### P1-E · tarball 源码的 commit 只能记成"声明值"

从 `archive/<sha>.tar.gz` 解出来的目录没有 `.git`，`_git_commit()` 必然返回 `None`。v0.2.1 在这种情况下只打一行警告，结果里 `commit: null` —— 于是 S4 的 `quant_fp` 里 `llama_commit` 是空，事后**分不清"没记录"和"记录了但没校验"**。

现在 `CFG["LLAMA_COMMIT"]` 非空而 git 拿不到东西时，把它记成声明值并标明来源：

```python
commit, commit_source = git_commit, ("git" if git_commit else None)
if CFG["LLAMA_COMMIT"] and git_commit is None:
    commit, commit_source = CFG["LLAMA_COMMIT"], "declared"
```

有 git 且与预注册值不符时仍然 `GuardError`（这一条没放松）。`test_s2_commit_fallback` 钉住四种组合：无 git + 有预注册 → `declared`；无 git + 无预注册 → `None`（不编造）；有 git 且不符 → 抛；有 git 且相符 → `git`。

---

## 九之四、v0.3.0 修了什么（S2 一键）

| ID | 问题 | 修法 |
|---|---|---|
| **B-1** | 用户要在 Mac 上下载源码、解压、传 36.6 MiB Dataset、填绝对路径 —— 纯人工负担 | 新增 `RUN_S2` 一个开关：自动下载 pinned commit 的 tarball → 校验 sha256 → 安全解压到 `WORK/src/` → 编译。用户不碰本地 |
| **B-2** | 旧 `ALLOW_NETWORK=True` 走 `git clone --depth 1`（HEAD），与固定 `LLAMA_COMMIT` 直接冲突 —— "固定版本"这句话失效 | **删掉该分支**。联网只有一条路径：pinned tarball + 预登记 sha256 |
| **B-3** | 默认行为不明确：既想"一键"又不想"导入就下载" | `RUN_S2 = False` 是默认值；`_apply_run_switches()` 在 `False` 时**一个字段都不改**，`test_s2_default_off_no_network` 钉住；`test_s2_run_switch_only_turns_on_s2` 钉住"只开 S2，不碰 S3–S8" |
| **B-4** | S2 失败时用户面对几十行 cmake 输出 | `_explain_s2_failure()` 给"症状 → 处置"一句话 + 日志路径；S9 报告第 2 节单独记 S2 状态、四个二进制、磁盘剩余 |

### B-2 的细节：为什么不许 `git clone`

`git clone --depth 1` 拿到的是**克隆那一刻的 HEAD**。同一份报告里 `LLAMA_COMMIT` 写着 `d454a204…`，实际编译的却是三天后的一次提交 —— 行号、CMake 选项、默认架构列表全部可能对不上，而报告里看不出来。所以 v0.3.0 把这条路径整个删掉，只留"下载一个字节数已知的 tarball 并校验 sha256"。

`commit_source` 因此从两档变三档：`git`（有 `.git` 且相符）/ `pinned_tarball`（digest 已校验，v0.3.0 自动路径的默认结果）/ `declared`（只有声明值，手动挂载源码时）。三档都如实写进 S9 报告，**不把 `pinned_tarball` 说成 git 校验，也不把 `declared` 说成已验证**。

### B-1 的细节：三级缓存与四个"拒绝"

`_fetch_pinned_source()` 的顺序是：已解压目录 + stamp 一致 → 完全离线复用；tarball 在且 digest 一致 → 只解压；否则才下载。四种情况**拒绝继续**而不是"尽量解压"：sha256 不符、Content-Length 与预登记字节数不符、归档里有 `..`/绝对路径/符号链接/非普通文件、顶层目录名与 pin 不符。

---

## 九之五、v0.3.1 修了什么（真实 Kaggle 失败的热修）

v0.3.0 在真实 Kaggle T4 上跑到 CMake 阶段才失败（下载和工具链都没问题）：`CUDA 12.8.93 FOUND`、`nvcc` host GNU 13.3.0 检测通过、arch=75、`-- Including CUDA backend`，然后

```text
CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:181 (target_link_libraries):
  Target "ggml-cuda" links to: CUDA::cuda_driver
  but the target was not found.
CMake Generate step failed.
```

| ID | 问题 | 修法 |
|---|---|---|
| **C-1** | Kaggle 镜像里 `find_package(CUDAToolkit)` 解析不出 `CUDA::cuda_driver` IMPORTED target（libcuda.so 不在它找的位置） | 默认传 `-DGGML_CUDA_NO_VMM=ON`（`CFG["CUDA_NO_VMM"]=True`）。上游 `ggml/CMakeLists.txt:205` 声明这个 option（默认 OFF），`ggml-cuda/CMakeLists.txt:178-182` 据此跳过 `target_link_libraries(... CUDA::cuda_driver)`。**仍然保留 `-DGGML_CUDA=ON` 与 sm_75**，没有退到 CPU |
| **C-2** | `_explain_s2_failure()` 把上面这条错误误判成"找不到 nvcc"，还劝用户关 GPU | 分类器改成**同行匹配**，并新增 `CUDA::cuda_driver` 专用分支且排在 nvcc 分支之前。误判根因：v0.3.0 用 `"nvcc" in tail and "not found" in tail` 这种全文包含判断，而这条日志里"nvcc"（工具链检测行）和 "not found"（target 行）相隔十几行 |

**`GGML_CUDA_NO_VMM=ON` 的代价是什么**（对着 `d454a204` 源码逐行看过）：`common.cuh:265` 的 `GGML_USE_VMM` 不再定义 → `ggml-cuda.cu:745` 的 `new_pool_for_device` 回退到 `ggml_cuda_pool_leg`（cudaMalloc 池），`:326` 的设备 VMM 属性查询和 `:592` 的 `ggml_cuda_pool_vmm` 整段编译掉。**算力内核、cuBLAS 路径、数值结果都不变**，只是显存池实现从 cuMemCreate/cuMemMap 换成 cudaMalloc。构建产物里会多一条 `NO_VMM=1` 特性标记（`ggml-cuda.cu:6134`）。

**为什么不再生嫌疑**：`CUDA_NO_VMM` 已加进 `STAGE_DEPS["S2_BUILD"]` 的 CFG 键列表，改这个开关会换 S2 指纹、必然重跑；加上 `NOTEBOOK_VERSION`  bump 到 0.3.1 让所有阶段指纹一起失效，不会误 RESUME 上一轮的 build。`test_s2_fingerprint_binds_cuda_no_vmm` 钉住前者。

**没保证的**：`CUDA::cuda_driver` 只是 Generate 阶段的第一道坎，绕过它之后 build 阶段是否还有后续错误，**未实测**（不授权启动实例）。用户网页重跑后把新日志回传即可。

---

## 十、怎么跑测试

```bash
python3 research/kaggle/test_trlab.py
```

不需要 GPU、不需要 torch、不需要 llama.cpp、不联网、不下载。它在一个临时目录里把 notebook 的 16 个 code cell 按顺序 exec 一遍，然后跑 43 项断言。当前结果：**43/43 通过，约 11 秒**。

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
| `test_s2_commit_fallback` | 无 `.git` 时 commit 记成 `declared` 而非 `None`；无预注册值时不编造；有 git 且不符必须抛 |
| `test_s2_default_off_no_network` | **B-3**：默认 `RUN_S2=False` 时 `_apply_run_switches()` 返回 `{"run_s2": False}` 且一个字段都不改；无手动源码时 `_find_llama_src` 抛 `GuardError` 且消息含 `RUN_S2` 与 `Internet`，**不触发下载** |
| `test_s2_run_switch_only_turns_on_s2` | **B-3**：`RUN_S2=True` 只开 `S2_BUILD`，`S3`–`S8` 全保持 `False`；`LLAMA_COMMIT`/`CUDA_ARCH`/`BUILD_JOBS`/`BUILD_CUDA` 自动填好 |
| `test_s2_download_checksum_mismatch` | digest 与预登记值不符 → `GuardError`，下载的包被删掉，不解压 |
| `test_s2_extract_rejects_traversal` | 归档里有 `../../evil.txt` → `GuardError`，不"尽量解压" |
| `test_s2_repeat_run_uses_cache` | 重复运行三级缓存：离线复用 → 只解压 → 包被改坏才重下（下载次数被钉死） |
| `test_s2_dir_name_mismatch_fails` | digest 对但顶层目录名与 pin 不符 → `GuardError`（pin 信息本身有问题） |
| `test_s2_commit_from_pinned_tarball` | 自动路径记 `pinned_tarball` + `commit_verified=True`；有 git 且相符记 `git`；不符仍抛 |
| `test_s2_failure_message_actionable` | 六种编译失败症状各自给出可操作处置（host g++ / nvcc / 磁盘 / 内存 / cmake / 其他） |
| `test_s2_failure_classifier_cuda_driver` | **C-2**：Kaggle 真实日志尾部（含 nvcc 行 + `CUDA::cuda_driver ... not found`）必须判成 driver 分支，**不能**误判成"找不到 nvcc"；真正的 nvcc 缺失仍然判成 nvcc |
| `test_s2_fingerprint_binds_cuda_no_vmm` | **C-1**：`CUDA_NO_VMM` 进 S2 指纹，改开关必然重跑，不会误 RESUME 旧 build |
| `test_s2_network_failure_message` | 真的连不上时消息含"Internet = On"，且不留下 `.partial` |
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
15. **`-DLLAMA_USE_PREBUILT_UI=OFF` 依赖上游有这个选项名**。它在顶层 `CMakeLists.txt:138` 用 `option()` 声明；如果将来改名，cmake 只会对未知的 `-D` 变量警告一句然后忽略，后果回到"构建期尝试下载 UI"——不会失败，只是慢。runbook 的核对清单里有这一项。
16. **tarball 源码没有 `.git`，commit 的校验强度取决于来源**。自动下载路径记 `commit_source="pinned_tarball"` + `commit_verified=True`，它证明"这份 tar.gz 的 sha256 与预登记值一致"，**不等于** git 校验；手动挂载的源码只能记 `declared`。要 git 级校验，需要自己 `git clone` 后保留 `.git` 再挂载（并设 `CFG["LLAMA_SRC"]`）。
17. **预登记 sha256 本身是自证的**。它写在 notebook 里，校验只能防"上游文件被替换 / 网络返回错误页面"，防不了"notebook 里的预登记值被改坏"。第一次使用时应在报告第 2 节核对那串 sha256 与本文第四节列出的值一致。
18. **Kaggle 能否直连 github.com 未验证**。自动下载走标准库 `urllib`（不依赖 curl/wget）。如果 Kaggle 网络策略拦了 GitHub，S2 会给出"检查 Settings → Internet = On"的报错，此时用断网备选：手动下载同一 tarball、挂成 Dataset、设 `CFG["LLAMA_SRC"]`。
19. **`-DGGML_CUDA_NO_VMM=ON` 只跨过了 Generate 阶段的第一道坎**。它让 CMake 不再链接 `CUDA::cuda_driver`，但 build 阶段（nvcc 编译、链接）是否还有后续错误**未实测**。代价已知且有限：显存池从 VMM 回退到 cudaMalloc 池，算力内核不变。真跑通之前，不要在报告里写"Kaggle 编译已通过"。
20. **错误分类器按关键词匹配，永远可能漏**。v0.3.0 的 `"nvcc" in tail and "not found" in tail` 全文包含判断就把 driver 错误误判成了 nvcc 缺失。现在改成同行匹配 + driver 专用分支，但新的日志形态仍可能落到通用分支 —— 通用分支的处置是"看日志最后 30 行"，这是兜底不是终点。

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
| S2 cmake 参数（v0.3.1） | ✅ 默认含 `-DGGML_CUDA_NO_VMM=ON`；`CUDA_NO_VMM=False` 或 `BUILD_CUDA=False` 时不传；结果里 `cuda_no_vmm` 与之一致 |
| S2 错误分类（v0.3.1） | ✅ 真实 Kaggle 的 `CUDA::cuda_driver` 日志判成 driver 分支（含 `GGML_CUDA_NO_VMM` 建议、不含"找不到 nvcc"）；真 nvcc 缺失仍判 nvcc |
| S2 指纹（v0.3.1） | ✅ `CUDA_NO_VMM` 翻转 → S2 指纹变化；改回去还原 |
| S2 commit 记录 | ✅ 无 `.git` 时记 `declared`、无预注册值时不编造、有 git 且不符必抛 |
| S2 自动取源码（mock） | ✅ 默认不联网；只开 S2；digest 不符拒解压并删包；`..` 条目拒解压；三级缓存不重下；目录名不符拒解压；`pinned_tarball` 校验档；六种失败症状各有处置；真断网时消息含 "Internet = On" |
| `test_trlab.py` | ✅ **43/43，约 11 秒** |

**没有**验证的：S2 的实际编译、S3/S4 的真实量化、S6 的 `llama-perplexity` 真实输出解析（正则已用真实格式字符串测过，但没跑过真实二进制）。这三项必须在 Kaggle 上首跑时人工确认。

**v0.3.1 的真实进度（来自用户 2026-10-21 的 Kaggle 运行）**：源码下载与 sha256 校验**已通过**，CMake 检测到 CUDA 12.8.93 与 host GNU 13.3.0、arch=75、`-- Including CUDA backend`，随后在 `ggml-cuda/CMakeLists.txt:181` 因 `CUDA::cuda_driver` target 缺失而 Generate 失败。也就是说：**自动下载链路和工具链预检都被真实环境验证过了，失败的只有 CMake 的 driver 链接这一处**，本版针对它做了热修，但热修本身尚未在 Kaggle 上验证。

**没有**验证的还有：S2 从 Kaggle 实际编译能否走完（绕过 driver 链接后的 build 阶段）、编译真实耗时、build 目录大小与峰值内存、`llama-imatrix` 在 T4 上跑 8B Q8_0 的耗时。runbook 第八节列了完整清单。

---

## 十三、如果只想先确认环境

保持 `STAGES` 里只有 `S0_ENV` / `S1_SMOKE` / `S9_REPORT` 为 `True`，其余全 `False`，`Run All`。约 2–4 分钟，不碰大文件、不联网。

**如果 `S0_ENV` 说 `/kaggle/input` 下没有 `.gguf`，那 S3–S7 一定跑不了 —— 这是设计如此，不是 bug。**

确认环境没问题之后，**不要**把 `STAGES` 全改成 `True` 再 `Run All`。下一步按 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md) 走：把 `RUN_S2` 改成 `True`（`Internet` 记得设 On），编译过了再开 `S3_IMATRIX`，每一段都有停机点和人工核对清单。

> v0.3.0 起 `RUN_S2=True` 会自动把 `STAGES["S2_BUILD"]` 打开，所以这一步只需要改一个开关，不用再手改 `STAGES`。
