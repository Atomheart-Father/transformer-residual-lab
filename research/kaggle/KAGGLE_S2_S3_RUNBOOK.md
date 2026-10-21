# Kaggle T4 部署清单：S2 编译 → S3 imatrix

**状态：方案文档。S2 / S3 均未在任何真实 Kaggle 实例上运行过，本文件不含"已验证编译/量化"的结论。**

这份文档只解决一件事：在你已有的 Kaggle GPU Notebook 里，**分两步**把 `llama-imatrix` 编译出来、并生成 `imatrix.gguf`，每一步都有明确的成功标志和停机点。S4 及之后不在本阶段执行，只在第六节给出容量预算。

所有数字分三类，全文统一标注：

| 标记 | 含义 |
|---|---|
| **实测** | 来自你 v0.2.1 的 S0 输出，或我这边核对过的公开文件大小 / 源码 |
| **估算** | 按公开参数推算，没有在 Kaggle 上跑过 |
| **待核** | 无法从这里确认，需要你在网页上看一眼 |

---

## 一、你的实测环境（来自 S0 输出，**实测**）

| 项 | 值 | 对本阶段的影响 |
|---|---|---|
| GPU | Tesla T4 ×2，单卡 14.56 GiB，可用估算 **13.56 GiB** | 只有第 0 张会被用到 |
| CPU | 4 核 | `BUILD_JOBS=4`，编译是瓶颈 |
| RAM | 31.35 GiB（可用 ≈ **29.67 GiB**） | S2 编译够用；S4 量化要看实际缓冲（第六节） |
| `/kaggle/working` | 剩余 **19.5 GiB** | 装不下 15.26 GiB 的 BF16 加产物（第六节算账） |
| CUDA | 12.8，`nvcc` / `cmake` / `g++` 都在 | 决定了 CUDA 架构默认列表（见 S2） |
| Internet | **Off** | 不能 clone、不能 pip、不能 `file(DOWNLOAD)` |
| `/kaggle/input` | 空 | 所以第一步是准备输入（第三节） |

BF16 1.55 TFLOP/s 是微基准，不能当 LLM 解码性能用；本阶段也不需要它。

---

## 二、本阶段的目标与边界

**做**：编译四个工具 → 用 Q8_0 算 imatrix → 停在 S3。

**不做**：S4 量化（只做容量预算）、S5–S8、任何下载/安装/联网操作。

**为什么停在 S3**：S4 需要 15.26 GiB 的 BF16 作为**只读输入**，而 `/kaggle/working` 只有 19.5 GiB —— 输入一旦落进 working，加上两个产物就必然爆盘。这个前提必须先满足（第三节路径 A），否则 S4 一定失败。

---

## 三、数据从哪来（这是本阶段最容易卡住的地方）

### 3.1 为什么不能"在 Mac 上下载再上传"

BF16 是 15.26 GiB。Kaggle 网页上传大文件要断点续传、耗时以小时计，而且你的上行带宽是瓶颈。这条路直接排除。

### 3.2 已验证的 HF 来源（**实测**，2026-10-21 通过 HF API 读取）

| 仓库 | revision（sha） | license | 文件 | 字节 | GiB |
|---|---|---|---|---|---|
| `unsloth/Qwen3-8B-GGUF` | `a6adef130ffb23ddaf1a62fec9dced968c9bc482` | apache-2.0 | `Qwen3-8B-Q8_0.gguf` | 8,709,519,168 | 8.11 |
| 同上 | 同上 | 同上 | `Qwen3-8B-BF16.gguf` | 16,388,044,384 | 15.26 |
| `Qwen/Qwen3-8B-GGUF` | `7c41481f57cb95916b40956ab2f0b139b296d974` | apache-2.0 | `Qwen3-8B-Q8_0.gguf` | 8,709,518,112 | 8.11 |
| `bartowski/Qwen_Qwen3-8B-GGUF` | `0b69f75b7472688e6808490aa2b85efdb81b5ce7` | apache-2.0 | `Qwen_Qwen3-8B-bf16.gguf` | 16,388,044,064 | 15.26 |

**推荐 unsloth 的 Q8_0 + BF16**：同一个上传者、同一次转换流程，是"同一 checkpoint"能拿到的最强证据。

**但请注意这三个 Q8_0 的字节数**：8,709,518,112 / 8,709,518,880 / 8,709,519,168，相差 768–1056 字节。它们的张量形状必然完全一致（都是 Qwen3-8B），字节却不同 —— 这就是"**张量名与形状一致不能证明是同一个文件**"的直接例子。S4 的 `_base_consistency()` 只能排除形状不符，排除不了这个差异；报告里必须声明"imatrix base 与量化 base 来自同一仓库的同一 revision"，而不是"已证明同源"。

### 3.3 路径 A：把 HF 文件变成只读 Input（推荐，不需要从 Mac 上传）

Kaggle 允许把一个 notebook 的输出当作另一个 notebook 的输入。于是：

```
取数 notebook（Internet=On，下载 Q8_0 8.11 GiB） ──Save Version──┐
取数 notebook（Internet=On，下载 BF16 15.26 GiB）──Save Version──┤→ 主 notebook「Add Input」
                                                                │   /kaggle/input/<slug>/...
```

两个文件必须分两个 notebook：8.11 + 15.26 = 23.37 GiB > 19.5 GiB，同一个 working 装不下。

**取数 notebook 的全部代码**（新建一个空 notebook，Settings 里 Internet 设为 **On**）：

```python
# Cell 1 —— Q8_0（imatrix 用）
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="unsloth/Qwen3-8B-GGUF",
    filename="Qwen3-8B-Q8_0.gguf",
    revision="a6adef130ffb23ddaf1a62fec9dced968c9bc482",   # 钉死 revision
    local_dir="/kaggle/working",                            # 直接放根目录，挂载路径最干净
)
print(p, __import__("os").path.getsize(p))
```

```python
# Cell 2 —— BF16（换一个 notebook 跑）
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="unsloth/Qwen3-8B-GGUF",
    filename="Qwen3-8B-BF16.gguf",
    revision="a6adef130ffb23ddaf1a62fec9dced968c9bc482",
    local_dir="/kaggle/working",
)
print(p, __import__("os").path.getsize(p))
```

跑完核对打印出的字节数：**8,709,519,168** 和 **16,388,044,384**。不一致就是 revision 变了，停下来。

然后 `Save Version`（**用 Quick Save，不要 Save & Run All** —— 后者会从头重跑并可能覆盖输出）。之后在主 notebook 里 `Add Input` → 你的 Work → 这两个 notebook 的输出，挂载成只读。

**待核**：Kaggle 界面里 "notebook 输出当输入" 的具体入口名（可能在 `Add Input → Your Work` 下，也可能叫 "Notebook Output"），以及 Quick Save 是否会把 `/kaggle/working` 完整存进版本。请你在网页上确认一次；如果 Quick Save 不保存 working，就退回"一个会话里连着跑 S2 和 S3"。

### 3.4 校准语料与评测语料（**实测**，文件很小，直接从 Mac 上传即可）

| 用途 | 来源 | revision | license | 文件 | 字节 |
|---|---|---|---|---|---|
| 校准 `calib.txt` | `Salesforce/wikitext` | `b08601e04326c79dfdd32d625aee71d232d685c3` | cc-by-sa-3.0 + gfdl | `wikitext-2-raw-v1/train-00000-of-00001.parquet` | 6,357,543 |
| 评测 `eval.txt` | 同上 | 同上 | 同上 | `wikitext-2-raw-v1/test-00000-of-00001.parquet` | 732,610 |

HF 现在只提供 parquet，`llama-imatrix -f` 要的是纯文本，所以要转一下（**6 MB 级，在取数 notebook 里顺手做**）：

```python
import pandas as pd
df = pd.read_parquet("train-00000-of-00001.parquet")   # 列名通常是 text
open("calib.txt", "w", encoding="utf-8").write("\n\n".join(df["text"].tolist()))
```

test 那份同理写成 `eval.txt`。两个文件都上传成一个小的 Kaggle Dataset（几百 KB 到 2 MB，网页上传毫无压力），或跟 3.3 的取数 notebook 放一起。

**分离要求**：calib 只能用 train，eval 只能用 test。S6 会用"前 200 字符逐行比对"做重叠检测并在重叠时报错，但那只挡得住完全相同的行 —— **不同段落来自同一文档时它检测不出来**，所以 `CALIB_SOURCE` / `EVAL_SOURCE` 必须如实填成不同 split。

**量够不够**：`NLL_GROUPS=8` × `NLL_CHUNKS_PER_GROUP=4` × `NLL_CTX=2048` = 65,536 token。wikitext-2 test 约 245k token（常见统计值，**估算**），够用约 27%；每个分组文件约 137 KB 文本，能覆盖 4 个 2048 chunk。如果你把 `NLL_GROUPS` 调大，先按 `4 字节/token` 反算一下总字符数够不够。

### 3.5 路径 B：直接找现成的 Kaggle Dataset（**待核，检索未找到**）

我用 Kaggle 公开检索 API 查了 `qwen3-8b gguf` / `qwen3 8b q8_0` / `qwen3 8b bf16` / `qwen3-8b-gguf` 等几组关键词（2026-10-21），**没有找到**同时满足"Qwen3-8B + Q8_0（8.11 GiB）或 BF16（15.26 GiB）"的公开数据集。命中的 4.6 GiB 级数据集多是 Q4 量化，不能当 imatrix base，更不能当量化 base。

检索 API 不是穷举，你可以自己在 `Add Input` 里搜一次；找到的话优先用它（省掉路径 A 的两个 notebook）。**找不到就用路径 A，不要为了"用现成的"退而求其次拿 Q4 当 base。**

---

## 四、S2：编译（单阶段，约 15–30 分钟，**估算**）

### 4.1 前置

1. 主 notebook 里 `Add Input` 挂上 llama.cpp 源码。需要**固定 commit** 的源码目录（含 `CMakeLists.txt`）。

   **推荐做法：下载固定 commit 的 tarball，上传成一个 Dataset（36.6 MiB，网页上传几秒）**

   | 项 | 值 |
   |---|---|
   | commit | `d454a204ef2b63af357744809c05af897611a72e`（2026-10-21） |
   | URL | `https://github.com/ggml-org/llama.cpp/archive/d454a204ef2b63af357744809c05af897611a72e.tar.gz` |
   | 大小 | **38,419,835 B**（36.6 MiB，**实测**） |
   | sha256 | `f565ee976503e0b66997a05554d4881a252ced7687811f2a41df771203b54d7f`（**实测**） |

   在 Mac 上：

   ```bash
   curl -L -o llama_src.tar.gz \
     https://github.com/ggml-org/llama.cpp/archive/d454a204ef2b63af357744809c05af897611a72e.tar.gz
   shasum -a 256 llama_src.tar.gz    # 必须等于上面那个 sha256
   tar xzf llama_src.tar.gz          # 得到 llama.cpp-d454a204.../ 目录
   ```

   把这个目录（不是 tarball）上传成 Kaggle Dataset，挂载后路径形如 `/kaggle/input/llama-src/llama.cpp-d454a204ef2b63af357744809c05af897611a72e/`。`CFG["LLAMA_SRC"]` 指到**含 `CMakeLists.txt` 的那一层**。

   选这个 commit 的理由：本文所有行号、CMake 选项名、默认架构列表都是对着它核对过的（见 4.2）。换 commit 不影响流程，但**引用行号前请自己复核**。

   **注意：tarball 解出来的目录没有 `.git`。** `git rev-parse` 拿不到 commit，S2 会把 `CFG["LLAMA_COMMIT"]` 记成声明值并标 `commit_source="declared"`（v0.2.2 起，不再是 `null`）。想拿到真正可校验的 commit，只有两条路：`CFG["ALLOW_NETWORK"]=True` 让它 `git clone`，或把 `.git` 目录一起打包（体积会涨到几百 MB，不推荐）。

   **有没有现成的 Kaggle Dataset（**待核**）**：我在 Kaggle 公开检索 API 查了 `llama.cpp` / `llama-cpp` / `llamacpp` 三组关键词（2026-10-21），命中的大多是**预编译二进制**而不是源码，例如 `nixty2/llama-cpp-turing-sm75-b11342`（132.5 MB）、`chaosong319/llamacpp-t4-full-engine`（148.0 MB）、`smartmichaelma/llamacpp-build-latest`（195.3 MB）。这类数据集的问题是：**commit 不可知、不可 pin**，而且 `-DCMAKE_CUDA_ARCHITECTURES` 已经由别人定死，你没法验证它到底编了哪些架构。文件清单需要登录才能看（API 返回空），所以我**没有**验证它们是否含源码。**不要图省事用它们**：36.6 MiB 的 tarball 自己传一次，换来的是可复现的 commit。
   - 把 `CFG["LLAMA_COMMIT"]` 填成上面那 40 位 sha，S2 会校验；不匹配直接报错，避免"同一份报告混用两个 llama.cpp 版本"。
2. 打开 `STAGES`，**只改 S2 这一项**：

```python
STAGES = {
    "S0_ENV": True, "S1_SMOKE": True,          # 便宜，留着当环境快照
    "S2_BUILD": True,                           # ← 只打开这个
    "S3_IMATRIX": False, "S4_QUANT": False, "S5_BYTES": False,
    "S6_NLL": False, "S7_QUALITY": False, "S8_BF16_ANCHOR": False,
    "S9_REPORT": True,
    "SX_EXPLORE": False,
}
```

改 `STAGES` **不会**让 S2 的断点失效（`STAGES` 不进指纹），所以第二次跑会正常 RESUME。

3. 配置：

```python
CFG["LLAMA_SRC"]    = "/kaggle/input/<源码 slug>/llama.cpp"   # 含 CMakeLists.txt 的那层
CFG["LLAMA_COMMIT"] = "<12 位 commit>"
CFG["BUILD_CUDA"]   = True
CFG["CUDA_ARCH"]    = "75"      # T4 = sm_75
CFG["BUILD_JOBS"]   = 4
```

### 4.2 为什么是这几个参数（都能在上游源码里查到）

| 参数 | 作用 | 依据（行号均指 commit `d454a204`） |
|---|---|---|
| `-DGGML_CUDA=ON` | 编 CUDA 后端 | `ggml/CMakeLists.txt` 的 `option(GGML_CUDA ...)` |
| `-DCMAKE_CUDA_ARCHITECTURES=75` | **只编 T4 一个架构** | 不指定时 `ggml/src/ggml-cuda/CMakeLists.txt:21` 起给出默认 `50-v 61-v 70-v 75-v 80-v 86-r 89-r 90-v 120a-r`，**九套**；4 核上编译量翻几倍 |
| `-DLLAMA_USE_PREBUILT_UI=OFF` | 不下载预构建 UI | option 在顶层 `CMakeLists.txt:138`（默认 `ON`），经 `tools/ui/CMakeLists.txt:53` 转成 `HF_ENABLED` 传给 `scripts/ui-assets.cmake`；`llama-ui-assets` 是 `ALL` target，本地无 `dist/` 时走 `hf_download()` → `file(DOWNLOAD)`（sha256 `TIMEOUT 30` / 包体 `TIMEOUT 300`）。Internet Off 时**不会失败**，只是白等，然后打一条 `UI: no assets available` 警告 |
| `-DLLAMA_CURL=OFF` | 兼容旧 pin | 当前 master 里 `LLAMA_CURL` 已被 `llama_option_depr` 标记为 deprecated 并**忽略**（只打一条警告）。保留这一项只为兼容较旧的源码 |
| `-DLLAMA_BUILD_TESTS=OFF` / `-DLLAMA_BUILD_EXAMPLES=OFF` | 少编一堆东西 | 与 tools 无关；tools 由 `LLAMA_BUILD_TOOLS` 控制（默认 ON） |
| 不加 `-DLLAMA_BUILD_SERVER=OFF` | **别关** | `tools/CMakeLists.txt:24-28`：`add_subdirectory(cli)` 在 `if (LLAMA_BUILD_SERVER)` 里面，关掉就没有 `llama-cli`，S7 会失败。而 `LLAMA_BUILD_SERVER` 默认取 `${LLAMA_STANDALONE}`（顶层构建时为 `ON`） |

四个 target 在 `d454a204` 上都存在（逐个查过 `tools/*/CMakeLists.txt` 的 `set(TARGET ...)`；注意同一文件里还有 `llama-quantize-impl` / `llama-perplexity-impl` / `llama-cli-impl`，那是内部库，`--target` 用下面这四个）：

```
llama-imatrix    llama-quantize    llama-perplexity    llama-cli
```

S2 会先跑一次工具链预检（`cmake --version` / `nvcc --version` / `g++ -dumpversion`），结果写进 `S2_BUILD.json` 的 `toolchain`。**如果 `BUILD_CUDA=True` 但找不到 `nvcc`，S2 会在调用 cmake 之前就 `GuardError`**，不会让你等十分钟才发现。

### 4.3 预算与回滚

| 项 | 估算 | 依据 |
|---|---|---|
| 编译时间 | 15–30 分钟 | 单架构（75）约 40 个 `.cu` + C++ 部分，4 核并行；**未实测** |
| build 目录占用 | 1.5–3.5 GiB | CUDA 目标文件较大；**未实测** |
| 峰值 RAM | 8–16 GiB | 4 个 nvcc 并行；**未实测** |
| 阶段预算 | `STAGE_BUDGET_SEC["S2_BUILD"] = 2400` 秒 | 超时会被 kill |

**回滚方案（按顺序试）**：

1. **超时**（`BudgetExceeded: 命令超预算被杀`）→ 把 `STAGE_BUDGET_SEC["S2_BUILD"]` 改成 `5400`，`FORCE_RERUN=True` 重跑。断点会保留已编好的对象文件，不会从头再来。
2. **CUDA 编译失败**（`nvcc fatal`、`unsupported GNU version` 之类）→ 设 `CFG["BUILD_CUDA"]=False` 重跑。**这是可接受的降级**：`llama-imatrix` / `llama-quantize` / `llama-perplexity` 都有 CPU 路径，只是 S3 会从"几分钟"变成"几十分钟到小时级"。
3. **磁盘不够** → 先删 `build-llama` 里没用的对象再试；仍不够就说明这个会话不该同时留 build 和模型。

**停机点 S2**：编译成功、四个二进制都在、`S2_BUILD.json` 是 `ok` —— 到这里**停下来**，别继续开 S3。

### 4.4 S2 人工核对清单

- [ ] `trlab/out/S2_BUILD.json` 里 `meta.status == "ok"`
- [ ] `trlab/build-llama/bin/` 下有 `llama-imatrix`、`llama-quantize`、`llama-perplexity`、`llama-cli` 四个文件
- [ ] 结果里 `bins` 四项都非空，`version` 不是空串
- [ ] 结果里 `cmake_args` 含 `-DLLAMA_USE_PREBUILT_UI=OFF` 和 `-DCMAKE_CUDA_ARCHITECTURES=75`
- [ ] `toolchain.nvcc` 记录了实际版本（不是"不可用"）
- [ ] `commit` 非空，且 `commit_source` 是 `git` 或 `declared`；如果是 `declared`，**报告里必须写"commit 为声明值、未校验"**
- [ ] 日志 `trlab/logs/s2_cmake.log` 末尾有 `Configuring done` / `Generating done`；`s2_build.log` 末尾有 `Built target llama-cli`
- [ ] 日志里如果有 `UI: no assets available`，**可以忽略**（我们不构建 server UI）
- [ ] `df -h /kaggle/working` 剩余 ≥ 12 GiB
- [ ] **Save Version（Quick Save）**，把 build 存进版本

---

## 五、S3：imatrix（约 5–15 分钟，**估算**）

### 5.1 前置

```python
STAGES["S2_BUILD"]   = False     # 已经完成，关掉避免误重跑
STAGES["S3_IMATRIX"] = True      # ← 只打开这个

CFG["IMATRIX_BASE"]      = "/kaggle/input/<gguf slug>/Qwen3-8B-Q8_0.gguf"
CFG["IMATRIX_BASE_KIND"] = "q8_0"
CFG["CALIB_TXT"]         = "/kaggle/input/<语料 slug>/calib.txt"
CFG["CALIB_SOURCE"]      = "wikitext-2-raw/train"
CFG["IMATRIX_GGUF"]      = ""     # 留空 → 写到 working/trlab/models/imatrix.gguf
CFG["IMATRIX_CTX"]       = 512
CFG["IMATRIX_CHUNKS"]    = 32
```

**`BASE_GGUF` 这时候不用填**（那是 S4 的事），但建议现在就填好，省得后面忘：

```python
CFG["BASE_GGUF"] = "/kaggle/input/<bf16 slug>/Qwen3-8B-BF16.gguf"
CFG["BASE_KIND"] = "bf16"
CFG["ALLOW_REQUANTIZE"] = False   # 保持 False
```

### 5.2 会发生什么

```
llama-imatrix -m <Q8_0> -f <calib.txt> -o imatrix.gguf.partial -c 512 --chunks 32 -ngl 99 -t 4
```

`-ngl` 由显存自动判定：Q8_0 8.11 GiB + 1.0 GiB 余量 < 13.56 GiB → `-ngl 99`（进显存）。如果是 BF16 15.26 GiB 就会退成 `-ngl 0`（纯 CPU，慢几十倍）—— 这正是要单独准备 Q8_0 的原因。

产物先写 `.partial`，用内置 GGUF 解析器验证 `n_tensors > 0` 之后才 `os.replace` 成正式文件名。

预算：32 chunk × 512 ctx = 16,384 token 的前向。T4 上 8B Q8_0 大约几分钟量级（**估算**，未实测）；`STAGE_BUDGET_SEC["S3_IMATRIX"] = 3600` 秒。

**停机点 S3**：imatrix 生成并通过验证 —— 到这里**停下**。S4 需要先解决 15.26 GiB 的只读输入（第六节）。

### 5.3 S3 人工核对清单

- [ ] `trlab/out/S3_IMATRIX.json` 里 `meta.status == "ok"`
- [ ] `trlab/models/imatrix.gguf` 存在，且结果里 `n_tensors > 0`
- [ ] 结果里 `base_kind == "q8_0"`、`imatrix_base_from_approx == True`（这条会进报告，别改）
- [ ] 结果里 `ngl == 99`；如果是 0，把原因记下来（说明 Q8_0 没进显存，要么显存被占，要么文件不是 8.11 GiB）
- [ ] `sha256` 非空（`HASH_GGUF=True` 时）
- [ ] `calib_source` 写的是 `wikitext-2-raw/train`，且 `calib.txt` 确实来自 train
- [ ] 日志 `trlab/logs/s3_imatrix.log` 末尾没有 `bailing out` / `error`
- [ ] **Save Version（Quick Save）**

---

## 六、S4 容量预算（**不运行，只算账**）

### 6.1 `/kaggle/working` 19.5 GiB 怎么分

| 项 | 大小 | 类型 |
|---|---|---|
| `build-llama/` | 1.5–3.5 GiB | 估算 |
| `models/imatrix.gguf` | 0.02–0.05 GiB | 估算（逐张量逐输入通道的 f32 统计量，量级是几十 MB） |
| `models/qwen3-8b-IQ2_XS.gguf` | 2.3–2.6 GiB | 估算（2.3125 bpw × 8.19e9 参数 ≈ 2.36 GiB + embed/head 高精度保留） |
| `models/qwen3-8b-IQ4_XS.gguf` | 4.3–4.6 GiB | 估算（4.25 bpw ≈ 4.34 GiB + 同上） |
| logs + out + eval_groups | < 0.2 GiB | 估算 |
| **合计** | **8.1–11.0 GiB** | |

**结论：15.26 GiB 的 BF16 绝对不能进 working**（8.1 + 15.26 = 23.4 > 19.5）。它必须是 `Add Input` 挂进来的只读文件。

### 6.2 止损值

| 时点 | 要求 | 动作 |
|---|---|---|
| S2 之后 | 剩余 ≥ 12 GiB | 不够就删 build 里没用到的对象，或换会话 |
| S4 之前 | 剩余 ≥ 10 GiB | 不够就不跑 S4，先清 working |
| S4 过程中 | 剩余 < 3 GiB | `guard()` 会在**下一个阶段入口**报 `BudgetExceeded`；本阶段运行中只能靠日志里 `llama-quantize` 的输出判断 |

`guard()` 只在阶段入口检查磁盘，跑起来之后管不了 —— 这条已经写进 docstring，别把它当运行中保护。

### 6.3 31 GiB RAM 够不够量化 F16

**不能说"吃内存就一定够"。** 实际情况取决于 `llama_model_quantize` 的加载方式与写缓冲：

- 输入走 mmap，页缓存可回收，不等于常驻；
- 输出是流式写出，峰值不要求同时持有整份权重；
- `--max-buffer-size` 控制"单个张量在内存里保留多少行"，**默认 8192 MiB**（`quantize.cpp:166` 的帮助文本原文 `default: 8192`）。这是最大的一块可控缓冲。

所以 29.67 GiB 可用 RAM **预期**够跑 15.26 GiB 输入（mmap 的 15.26 + 8 GiB 缓冲 + 进程开销 ≈ 24 GiB 量级），但**未实测**。止损办法：

1. 先跑一次 `--dry-run`（只算大小不写文件）确认文件能被正确解析、类型能识别。**注意**：`--dry-run` 在主循环里 `continue` 掉了（`llama-quant.cpp:1267`），所以**不会**触发二次量化的检查（`:1332`）**也不会**触发 imatrix 缺失的检查（`:1324`）—— 别用它判断 `ALLOW_REQUANTIZE` 或 imatrix 有没有生效。
2. 如果出现 `exit 137`（OOM kill）或长时间 swap，把 `--max-buffer-size` 调小（如 64 MiB）再试；仍不行就说明这个会话不该同时留 build 和两份产物。

### 6.4 S4 的前置配置（写在这里，等你说可以了再跑）

```python
CFG["BASE_GGUF"]        = "/kaggle/input/<bf16 slug>/Qwen3-8B-BF16.gguf"
CFG["BASE_KIND"]        = "bf16"
CFG["IMATRIX_BASE"]     = "/kaggle/input/<q8 slug>/Qwen3-8B-Q8_0.gguf"
CFG["IMATRIX_BASE_KIND"] = "q8_0"
CFG["ALLOW_REQUANTIZE"] = False
```

`ALLOW_REQUANTIZE` 必须保持 `False`：BF16 是高精度 base，不需要二次量化；一旦误开，产物会被标 `requantized_pilot_only`，结论效力就降级了。

---

## 七、失败排查

| 现象 | 去哪看 | 处置 |
|---|---|---|
| `GuardError: 找不到 llama.cpp 源码` | `out/S2_BUILD.json` 的 `error` | `Add Input` 挂源码，或 `CFG["LLAMA_SRC"]` 指到含 `CMakeLists.txt` 的那层 |
| `源码 commit ... 与预注册的 ... 不符` | 同上 | 改 `LLAMA_COMMIT`，或换回你打包的那个 commit |
| `GuardError: BUILD_CUDA=True 但 nvcc 不可用` | 同上 | `CFG["BUILD_CUDA"]=False` |
| `CommandFailed rc=1` + `CMake Error` | `logs/s2_cmake.log` | 看最后一屏；`unsupported GNU version` → 关 CUDA；缺 `CUDAToolkit` → 同上 |
| `BudgetExceeded: 命令超预算被杀` | `out/S2_BUILD.json` | 把 `STAGE_BUDGET_SEC["S2_BUILD"]` 调到 5400，`FORCE_RERUN=True` |
| `编译结束但缺少可执行文件: ['llama-cli']` | `logs/s2_build.log` | 大概率有人关了 `LLAMA_BUILD_SERVER`；恢复默认 |
| `UI: no assets available` | `s2_build.log` | **可忽略**，我们不构建 server UI |
| `CFG['IMATRIX_BASE'] 指向的路径不存在` | `out/S3_IMATRIX.json` | 检查 `Add Input` 是否挂上、路径拼写 |
| `imatrix 产物缺失或过小` | `logs/s3_imatrix.log` | 看是否 `bailing out`；常见是校准文本太短（< 200 KB 会警告） |
| `S3` 跑了很久没动静 | `logs/s3_imatrix.log` | `ngl=0` 说明退 CPU 了；确认 Q8_0 真的 8.11 GiB |

所有阶段失败都**不会中断**后续阶段：每个阶段自己写 `status: "error"` + traceback，S9 仍会出一份带错误摘要的报告。

---

## 八、未验证清单（不要在报告里写成已验证）

1. **S2 从未在 Kaggle 上编译过。** 编译时间、build 目录占用、峰值 RAM 全是估算。
2. **Kaggle 镜像的 nvcc 与 g++ 版本兼容性未验证。** CUDA 12.8 对 host gcc 有上限；不兼容时的降级路径是 `BUILD_CUDA=False`。
3. **S3 的 imatrix 运行时间未实测。**
4. **"notebook 输出当输入" 的界面入口与 Quick Save 是否保存 working 未验证**（3.3 节标了待核）。
5. **没有找到公开的 Qwen3-8B Q8_0 / BF16 Kaggle Dataset**（检索了 5 组关键词，2026-10-21）；检索 API 不穷举。
6. **Kaggle 上那些 llama.cpp 数据集到底装了什么，没有验证**。公开检索 API 返回的 `resources` 是空的（要登录才给文件清单），我只能从体积和标题判断它们多半是预编译二进制。**不要**把它们当源码用。
7. **31 GiB RAM 跑 15.26 GiB 输入的量化为未实测**（6.3 节）。`--max-buffer-size` 默认 8192 MiB 是从帮助文本读到的，实际峰值没有观测。
8. **`_base_consistency()` 只能排除形状不符**，不能证明同一 checkpoint —— 三个仓库的 Q8_0 字节数不同就是证据。
9. **tarball 源码的 commit 是"声明值"**（`commit_source="declared"`），不是校验值。

---

## 九、动态条件（不假定）

Kaggle 的每日 GPU 配额、单会话时长上限、并发会话数、存储配额都是**会变的外部条件**，本文件不写具体数字，也不假设它们够用。每次开跑前在网页上确认当前配额；如果中途被截断，S2 的 build 目录会随会话丢失，需要靠 `Save Version` 保住。

---

## 十、下一步

按第四节跑 S2 → 核对 4.4 清单 → 停下。确认无误后再按第五节跑 S3。

**不要点 `Run All`。** 默认开关下 S4–S8 是关的，`Run All` 不会真的跑量化，但它会一路执行到 S9 并重新生成报告 —— 报告里那些"未运行"的段落很容易被误读成"全部跑完了"。逐段执行，每段对着清单看一眼。
