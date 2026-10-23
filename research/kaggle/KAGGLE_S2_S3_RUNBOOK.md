# Kaggle T4 部署清单：S2 编译 → S3 imatrix → S4 量化

**v1.0.1 更新（2026-10-22）**：加一道**运行前预检**。`RUN_S4 = True` 的 `Run All` 会在**任何 S2 编译、S3 的 8.11 GiB Q8_0 下载、BF16 15.26 GiB 下载之前**先跑几秒的 `S4_PREFLIGHT`（只读）：核查 working 与工作区外暂存盘的挂载标识/容量/写权限、BF16 有没有落点、imatrix 能不能复用，并证明两个产物在 working 存得下。不过就**就地停下**，只给**一步**要做的（通常是 6.1 那条，或者清掉 working 里的旧产物）。只想做检查不动工：执行格最后一行改成 `trlab.run(preflight_only=True)`。同时 `S4_QUANT` 的墙钟预算 1800 → 5400 s，另加单格式 3600 s 帽 —— 那是**保护上限，不是耗时承诺**。

**v1.0.0 更新（2026-10-22）**：**执行方式重构。** notebook 从 253 KB / 28 cells 的巨物缩成 **7.3 KB / 5 cells** 的薄入口，全部逻辑在 `trlab/` 包的 16 个 `.py` 模块里；执行包由引导格按**钉死的 commit** 从公开仓库 sparse fetch（约 200 KB、1–3 秒，`git rev-parse HEAD` 强校验）后 import。**你的操作从六步（取 BF16 notebook → Run All → Save Version → Add Input → 配路径 → 主 notebook）收敛成"一个开关 + Run All"**：改 `RUN_S4 = True`，`Internet = On`，`Run all`。S6–S8 与 SX 默认关闭，要跑就把执行格最后一行改成 `trlab.run(run_s4=True, S6_NLL=True)`。

**v0.5.0 的量化链路本身不变**：`RUN_S4 = True` 自动打开 S2 + S3 + S4 + S5，从与 S3 同仓库同 revision 的**原始 BF16** 量化出 `IQ2_XS` / `IQ4_XS`（禁止 Q8_0 二次量化）。已有的 imatrix 会按 sha256 复用，**整段跳过 S3，不重下 8.11 GiB、不重算**。

**状态说明**：
- **S2 已被真实 Kaggle 运行完整验证**（v0.3.2，27 分 52 秒，四个二进制就位，见用户回传 `S2_BUILD-3.json`）。
- **S3 已被真实 Kaggle 运行完整验证**（2026-10-22，imatrix 10,701,952 B / 1008 张量 / 57.95 s，`-ngl 99` 全上 GPU，产物 sha256 `34062ab6…` 已 pin 进 `IMATRIX_PINNED`）。
- **S4/S5 的一键化链路只有 CPU/mock 测试覆盖（v1.0.1 起 118 项全通过，含 10 项预检），尚未在真实 Kaggle 上跑过。** BF16 下载、`llama-quantize` 的 CPU 耗时与峰值 RAM、S5 的真实字节账都还是**估算/待核**，本文一律如实标注。
- **预检本身也未在真实 Kaggle 上跑过**：`/kaggle/tmp` 到底存不存在、是不是另一个文件系统、有多少空闲，只有真跑才知道。本地只验证了判定逻辑（含"挂载表读不到时只保留 `st_dev`"的非 Linux 分支）。
- **v1.0 的引导链路已本地验证**（干净目录真实冷启动：GitHub fetch → 校验 → import → S0/S1/S9 → 产物落盘），**未在 Kaggle 上实测**；它要求 GitHub 仓库保持 public（`--embed` 内嵌变体可兜底）。

S3 的设计 rationale 与五个决策点的落点在 [`KAGGLE_S3_DESIGN.md`](./KAGGLE_S3_DESIGN.md)：**已实现。**

这份文档只解决一件事：在你已有的 Kaggle GPU Notebook 里，**一步**（改一个开关）把 llama.cpp 二进制、`imatrix.gguf`、`IQ2_XS.gguf`、`IQ4_XS.gguf` 全部备齐，有明确的成功标志和停机点。S6 及之后不在本阶段执行。

v0.3.0 起 S2 **不需要你准备任何输入**；v0.4.0 起 S3 同样不需要 —— 模型与语料都由 notebook 自己按 pinned revision 下载并校验。

所有数字分三类，全文统一标注：

| 标记 | 含义 |
|---|---|
| **实测** | 来自你 v0.2.1 的 S0 输出、v0.3.2 的 S2 输出，或我这边核对过的公开文件大小 / 源码 |
| **估算** | 按公开参数推算，没有在 Kaggle 上跑过 |
| **待核** | 无法从这里确认，需要你在网页上看一眼 |

### Kaggle Session 的三个硬事实（v0.4.0 从官方文档核实）

| 事实 | 对 v0.4.0 设计的影响 |
|---|---|
| 交互 session **空闲 20 分钟**即结束 | 断线是常态，不是异常 |
| `/kaggle/working` ≤ 20 GB **自动保存**，可在未来 Notebook 复用 | 编译产物落在 working 里，有机会跨 session 复用 |
| **Save & Run All = 全新干净状态**（12 小时上限）；Quick Save 只快照不重跑 | 不假设旧状态；启动时探活 |
| Add Input 可以搜「任何 Notebook 的输出文件」并挂上 | 二进制也可以变成只读 Input（可选优化） |

**"掉线后不 commit 是否一定恢复"官方没写** → v0.4.0 按"不保"做最坏假设，**每次启动都重新探活二进制**，探不到就自动重编。这是 28 分钟重编唯一会发生的情形，已如实记录。


---

## 一、你的实测环境（来自 S0 输出，**实测**）

| 项 | 值 | 对本阶段的影响 |
|---|---|---|
| GPU | Tesla T4 ×2，单卡 14.56 GiB，可用估算 **13.56 GiB** | 只有第 0 张会被用到 |
| CPU | 4 核 | `BUILD_JOBS=4`，编译是瓶颈 |
| RAM | 31.35 GiB（可用 ≈ **29.67 GiB**） | S2 编译够用；S4 量化要看实际缓冲（第六节） |
| `/kaggle/working` | 剩余 **19.5 GiB** | 装不下 15.26 GiB 的 BF16 加产物（第六节算账） |
| CUDA | 12.8.93，`nvcc` / `cmake` / host GNU 13.3.0 **都被真实运行验证过** | 决定了 CUDA 架构默认列表（见 S2）；host g++ 13.3 与 nvcc 12.8 兼容 |
| Internet | **Off**（S2 需临时设为 **On**，只下载 36.6 MiB 源码 tarball） | GitHub 直连**已被真实运行验证**；pip 与 `file(DOWNLOAD)` 仍不可用 |
| `/kaggle/input` | 空 | S2 不需要输入；S3 起需要（第三节） |

BF16 1.55 TFLOP/s 是微基准，不能当 LLM 解码性能用；本阶段也不需要它。

---

## 二、本阶段的目标与边界

**做**：编译四个工具 → 用 Q8_0 算 imatrix → 停在 S3。

**不做**：S4 量化（只做容量预算）、S5–S8、下载模型权重、`pip install`、租付费 GPU。

**联网边界（v0.4.0）**：`RUN_S3=True` 时 notebook 会联网下载**三**样东西，全部 pinned revision + 预登记 sha256/字节数：

1. llama.cpp 源码 tarball（36.6 MiB）—— 仅在没有可复用二进制时才下；
2. `unsloth/Qwen3-8B-GGUF` 的 Q8_0（8,709,519,168 B）；
3. `Salesforce/wikitext` 的 train parquet（6,357,543 B），转成纯文本。

默认 `RUN_S2=False` / `RUN_S3=False`，此时 `_apply_run_switches()` **一个字段都不改** —— 导入 notebook 或误点 `Run All` 都不会联网、不会下载 8 GB。**要下载必须显式把开关改成 `True`。**

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
trlab_fetch_bf16.ipynb（Internet=On，下 BF16 15.26 GiB）──Save Version──┤→ 主 notebook「Add Input」
                                                                │   /kaggle/input/<slug>/...
```

两个文件必须分两个 notebook：8.11 + 15.26 = 23.37 GiB > 19.5 GiB，同一个 working 装不下。

**这条六步流程从 v1.0.0 起不再是默认用法**，只在一种情况下需要：Kaggle **没有**提供 working 之外的暂存盘（或暂存盘也装不下 15.26 GiB），S4 会在下载前以 `GuardError` 停下并提示这一条降级路径。`trlab_fetch_bf16.ipynb`（3 个 cell：下载 pinned BF16 → sha256 校验 → Save Version 提示）仍随主 notebook 一起交付，不需要你自己新建；点击步骤见 6.1。Q8_0 那条路由于 S3 已一键化不再需要手动取。

**v1.0.0 优先尝试的零操作路径**：S4 的 `scratch_download` 分支会枚举候选挂载点，只认 `st_dev` 与 `/kaggle/working` **不同**的候选（同设备不同路径不算额外空间，避免把 19.07 GiB 算两遍），探测到足够空间就直接下到那里并校验；暂存盘容量 Kaggle 官方没给数字，所以是运行时探测而不是预设值。

**取数 notebook 的代码**（万一要自己新建，代码长这样）：

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

跑完核对打印出的字节数：**8,709,919,168** 和 **16,388,044,384**。不一致就是 revision 变了，停下来。

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

### 4.1 前置（v0.3.0：只需要改一个开关）

> **只想跑 S3 的话不用看这一节。** `RUN_S3=True` 会自动把 S2 一起打开（见 5.1），二进制探活失败时才真的编译。

**S2 不需要任何输入文件。** 源码由 notebook 自己下载 —— 固定 commit 的 tarball，36.6 MiB，sha256 预登记并强制校验。

| 项 | 值 |
|---|---|
| commit | `d454a204ef2b63af357744809c05af897611a72e`（2026-10-21） |
| URL | `https://github.com/ggml-org/llama.cpp/archive/d454a204ef2b63af357744809c05af897611a72e.tar.gz` |
| 大小 | **38,419,835 B**（36.6 MiB，**实测**） |
| sha256 | `f565ee976503e0b66997a05554d4881a252ced7687811f2a41df771203b54d7f`（**实测**） |

网页上做三件事：

1. 右上角 `Edit` → 找到 `RUN_S2 = False`，改成 `RUN_S2 = True`
2. 右侧 `Settings` → `Internet` = **On**（只此一个阶段需要联网）
3. `Run` → `Run all`

`RUN_S2 = True` 会自动把 `STAGES["S2_BUILD"]` 打开，并一次填好 `S2_AUTO_FETCH_SOURCE=True` / `LLAMA_COMMIT=<pin>` / `CUDA_ARCH="75"` / `BUILD_CUDA=True` / `BUILD_JOBS=4`。`S3`–`S8` **不会**被打开。默认值是 `False`，此时 `_apply_run_switches()` 一个字段都不改 —— 导入 notebook 或误点 `Run All` 都不会联网。

S2 内部顺序：下载 tarball → 比对 sha256（不符就删包报错）→ 安全解压到 `/kaggle/working/trlab/src/llama.cpp-<commit>/`（拒绝 `..`、绝对路径、符号链接）→ 校验顶层目录名与 pin 一致 → `cmake` → `cmake --build`。重复 `Run All` 不会重下：已解压目录 + `src/pinned_source.json` 的 sha256 与 pin 一致就离线复用。

**断网备选（不想开 Internet 时）**：在 Mac 上

```bash
curl -L -o llama_src.tar.gz \
  https://github.com/ggml-org/llama.cpp/archive/d454a204ef2b63af357744809c05af897611a72e.tar.gz
shasum -a 256 llama_src.tar.gz    # 必须等于上面那个 sha256
```

把这个 tarball（**不要解压**）上传成 Kaggle Dataset 并 `Add Input`，然后把 `CFG["LLAMA_SRC"]` 指到含 `CMakeLists.txt` 的那一层。注意这条路径下 S2 拿不到归档 digest，`commit_source` 只能是 `declared`（未校验）；想拿到 `pinned_tarball` 档就用自动下载。

**有没有现成的 Kaggle Dataset（**待核**）**：我在 Kaggle 公开检索 API 查了 `llama.cpp` / `llama-cpp` / `llamacpp` 三组关键词（2026-10-21），命中的大多是**预编译二进制**而不是源码，例如 `nixty2/llama-cpp-turing-sm75-b11342`（132.5 MB）、`chaosong319/llamacpp-t4-full-engine`（148.0 MB）、`smartmichaelma/llamacpp-build-latest`（195.3 MB）。这类数据集的问题是：**commit 不可知、不可 pin**，而且 `-DCMAKE_CUDA_ARCHITECTURES` 已经由别人定死，你没法验证它到底编了哪些架构。文件清单需要登录才能看（API 返回空），所以我**没有**验证它们是否含源码。**不要图省事用它们**：让 notebook 自己下 36.6 MiB，换来的是可复现的 commit。

改 `STAGES` **不会**让 S2 的断点失效（`STAGES` 不进指纹），所以第二次跑会正常 RESUME。

### 4.2 为什么是这几个参数（都能在上游源码里查到）

| 参数 | 作用 | 依据（行号均指 commit `d454a204`） |
|---|---|---|
| `-DGGML_CUDA=ON` | 编 CUDA 后端 | `ggml/CMakeLists.txt` 的 `option(GGML_CUDA ...)` |
| `-DCMAKE_CUDA_ARCHITECTURES=75` | **只编 T4 一个架构** | 不指定时 `ggml/src/ggml-cuda/CMakeLists.txt:21` 起给出默认 `50-v 61-v 70-v 75-v 80-v 86-r 89-r 90-v 120a-r`，**九套**；4 核上编译量翻几倍 |
| `-DGGML_CUDA_NO_VMM=ON` | **v0.3.1 热修**：不链 `CUDA::cuda_driver` | 真实失败：`ggml/src/ggml-cuda/CMakeLists.txt:181` 报 `Target "ggml-cuda" links to: CUDA::cuda_driver but the target was not found`。上游 `ggml/CMakeLists.txt:205` 有 `option(GGML_CUDA_NO_VMM ... OFF)`，`ggml-cuda/CMakeLists.txt:178-182` 据此跳过链接。代价：显存池从 VMM 回退 cudaMalloc 池（`ggml-cuda.cu:745`），算力内核不变。`CFG["CUDA_NO_VMM"]=False` 可关掉 |
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

| 项 | 值 | 依据 |
|---|---|---|
| 编译时间 | **27 分 52 秒**（1672.29 s） | **实测**（v0.3.2，真实 Kaggle T4）|
| build 目录占用 | 未观测 | S2 后 working 剩 19.07 GiB（实测），反推占用 < 0.5 GiB 含源码与 dl 缓存 |
| 峰值 RAM | 未观测 | 31.35 GiB 的实例没被 OOM |
| 阶段预算 | `STAGE_BUDGET_SEC["S2_BUILD"] = 2400` 秒 | 对 27 分 52 秒仍有余量；超时会被 kill |

**回滚方案（按顺序试）**：

1. **超时**（`BudgetExceeded: 命令超预算被杀`）→ 把 `STAGE_BUDGET_SEC["S2_BUILD"]` 改成 `5400`，`FORCE_RERUN=True` 重跑。断点会保留已编好的对象文件，不会从头再来。
2. **CUDA 编译失败**（`nvcc fatal`、`unsupported GNU version` 之类）→ 设 `CFG["BUILD_CUDA"]=False` 重跑。**这是可接受的降级**：`llama-imatrix` / `llama-quantize` / `llama-perplexity` 都有 CPU 路径，只是 S3 会从"几分钟"变成"几十分钟到小时级"。
3. **磁盘不够** → 先删 `build-llama` 里没用的对象再试；仍不够就说明这个会话不该同时留 build 和模型。

**停机点 S2**：编译成功、四个二进制都在、`S2_BUILD.json` 是 `ok` —— 到这里**停下来**，别继续开 S3。

### 4.4 S2 人工核对清单

- [ ] `trlab/out/S2_BUILD.json` 里 `meta.status == "ok"`
- [ ] `trlab/build-llama/bin/` 下有 `llama-imatrix`、`llama-quantize`、`llama-perplexity`、`llama-cli` 四个文件
- [ ] 结果里 `bins` 四项都非空，`version` 不是空串（v0.3.2 起 `--version` 走 stderr 也能取到）
- [ ] 结果里 `cmake_args` 含 `-DLLAMA_USE_PREBUILT_UI=OFF` 和 `-DCMAKE_CUDA_ARCHITECTURES=75`
- [ ] `toolchain.nvcc` 记录了实际版本（不是"不可用"）
- [ ] `source.origin` 是 `downloaded` / `cache` / `cached_archive` / `manual` 之一；`source.archive_sha256` 与本文 4.1 表格里的值**逐字符相同**
- [ ] `commit` 非空，且 `commit_source` ∈ {`git`, `pinned_tarball`, `declared`}；自动下载应得到 `pinned_tarball` + `commit_verified=true`；如果是 `declared`，**报告里必须写"commit 为声明值、未校验"**
- [ ] `cuda_no_vmm` 是 `true`（v0.3.1 默认；cmake_args 里应能看到 `-DGGML_CUDA_NO_VMM=ON`）。关掉它只会让上一轮那个 `CUDA::cuda_driver` 错误回来
- [ ] 报告第 2 节（工具链）写出了源码目录、归档 sha256、四个二进制路径、磁盘剩余
- [ ] 日志 `trlab/logs/s2_cmake.log` 末尾有 `Configuring done` / `Generating done`；`s2_build.log` 末尾有 `Built target llama-cli`
- [ ] 日志里如果有 `UI: no assets available`，**可以忽略**（我们不构建 server UI）
- [ ] `df -h /kaggle/working` 剩余 ≥ 12 GiB
- [ ] **Save Version（Quick Save）**，把 build 与解压的源码存进版本

---

## 五、S3：imatrix（v0.4.0 一键，改一个开关）

### 5.1 唯一操作开关

```python
RUN_S2 = False      # 不用管
RUN_S3 = True       # ← 就改这一个（只想做到 imatrix 为止）
RUN_S4 = False      # 想一路做到 IQ2_XS/IQ4_XS 就改这个（见第六节）
```

网页上做三件事：

1. 找到 `RUN_S3 = False`，改成 `RUN_S3 = True`
2. `Settings` → `Internet` = **On**（三样东西都要联网）
3. `Run` → `Run all`

`RUN_S3 = True` 会自动把 `STAGES["S2_BUILD"]` 和 `STAGES["S3_IMATRIX"]` 都打开，并一次填好 `S3_AUTO_FETCH_MODEL=True` / `S3_AUTO_FETCH_CALIB=True` / `LLAMA_COMMIT=<pin>` / `CUDA_ARCH="75"` / `CUDA_NO_VMM=True` / `BUILD_CUDA=True` / `BUILD_JOBS=4` / `IMATRIX_BASE_KIND="q8_0"` / `CALIB_SOURCE="wikitext-2-raw/train"` / `NGL_MARGIN_GB=2.0` / `S3_DELETE_Q8_AFTER=True`。**S4–S8 与 SX 一律保持 `False`**（测试钉住）。

**你不必、也不需要手动先跑一遍 S2。** S2 阶段自己会先探活二进制。

### 5.2 二进制：存在且能跑才复用，否则自动重编

启动时按顺序找四个可执行文件（`llama-imatrix` / `llama-quantize` / `llama-perplexity` / `llama-cli`）：

| 顺序 | 来源 | 说明 |
|---|---|---|
| 1 | `existing_build` | `/kaggle/working/trlab/build-llama/bin/` —— 上个 session 编好留下的 |
| 2 | `user_bin` | `CFG["LLAMA_BIN"]` 指向的目录 |
| 3 | `attached_input` | 扫 `/kaggle/input`（限时 20 s、限 20000 目录）—— 你把 build 做成 Dataset 挂上时走这条 |
| 4 | `rebuilt` | 都没有 → 完整走 S2：下载 pinned 源码 + 编译（≈ 28 分钟） |

**每个候选都真跑一次 `--version`** —— 文件存在不等于能跑（动态库缺失、架构不匹配都会让"存在但不可执行"的二进制通过文件检查）。探活失败就落到下一档。

`S2_BUILD.json` 的 `build_reuse` 字段记录实际走了哪一档；`commit_source` 三档 `git` > `pinned_tarball` > `declared`。**复用挂载二进制时必然落 `declared`（未校验）**，报告里会明确写出来，不会伪装成已校验。

### 5.3 会发生什么

```
（二进制已在）→ 下载 Q8_0 8.11 GiB → 下载 wikitext train parquet → 转 calib.txt
→ llama-imatrix -m Q8_0 -f calib.txt -o imatrix.gguf.partial -c 512 --chunks 32 -ngl 99 -t 4
→ 验证产物体积与 n_tensors → 写 OUT/S3_IMATRIX.json → 删掉 Q8_0 → 再写一次记录
```

**下载安全**：`.partial` + `.partial.json` sidecar 记 `bytes_done` 与已写内容前缀哈希，重跑从断点续传；四重完整性校验（`Content-Length` / `size` / `x-linked-etag` / sha256）任一不符就拒绝落地。**不用 `huggingface_hub`**，边写边算 sha256，`os.replace` 原子落地 —— 同一模型不会因为 HF 缓存 + working 目录产生两份完整副本。下载前先做磁盘预检，空间不够直接报缺口，不下一半再爆。

**`-ngl`**：`_fit_ngl()` 按「Q8_0 字节数 + 2.0 GiB 余量 ≤ 可用显存」判定。T4 上 8.11 + 2.0 = 10.11 < 13.56 GiB → `-ngl 99`（进显存）。OOM 时按阶梯 `[99, 18, 0]` 逐级降，**每次尝试的 `ngl / ok / rc / oom / 日志尾部` 全量进 provenance**，不只写最终值。降到 0 就是纯 CPU（小时级，日志会警告）。

**imatrix base 是 Q8_0 不是 f16**：激活统计量是对 f16 的近似。这比「从 Q8_0 二次量化」好得多，但报告里会带 `imatrix_base_from_approx=True` 的声明 —— 别改，别在结论里当成 f16 口径。

**Q8_0 回收顺序是硬编码的**：生成 → 验证 → **写 `OUT/S3_IMATRIX.json`** → 才 `_reclaim_q8()` → 再写一次含 `q8_reclaimed` / `q8_reclaimed_bytes` / `q8_reclaim_reason`。imatrix 失败时 Q8_0、校准语料、下载断点**全部保留**，修完重跑直接复用不重下；只有不可续跑的半成品 `imatrix.gguf.partial` 会被删掉（不占 working 的 20 GB）。

预算：32 chunk × 512 ctx = 16,384 token 的前向。T4 上 8B Q8_0 进显存大约几分钟量级（**估算**，未实测）；`STAGE_BUDGET_SEC["S3_IMATRIX"] = 3600` 秒。下载 8.11 GiB 的耗时取决于 Kaggle 出口带宽（**待核**，`S3_AUTO_FETCH` 没有单独的时间预算上限，只有阶段总预算）。

**停机点 S3**：imatrix 生成并通过验证 —— 到这里**停下**。S4 需要先解决 15.26 GiB 的只读输入（第六节）。

### 5.4 S3 人工核对清单

- [ ] `trlab/out/S3_IMATRIX.json` 里 `meta.status == "ok"`
- [ ] `trlab/models/imatrix.gguf` 存在，结果里 `n_tensors > 0`、`bytes` 是几十 MB 量级
- [ ] `base_origin == "auto_download"`，`base_download.sha256` 与预登记值 `0cfbf745…` 一致
- [ ] `base_kind == "q8_0"`、`imatrix_base_from_approx == True`（这条会进报告，别改）
- [ ] `calib_source == "wikitext-2-raw/train"`，`calib_origin == "auto_download"`
- [ ] `ngl == 99`；`ngl_attempts` 里如果有失败项，把 OOM 原因记下来
- [ ] `q8_reclaimed == True`，`q8_reclaim_reason` 是 `S3_DELETE_Q8_AFTER=True 且记录已写盘` 之类
- [ ] `S2_BUILD.json` 的 `build_reuse` 是 `existing_build` / `attached_input` / `rebuilt` 之一；如果是 `rebuilt`，把耗时记下来
- [ ] `S2_BUILD.json` 的 `commit_source`：自动下载应得 `pinned_tarball` + `commit_verified=true`；复用二进制只能是 `declared`，**报告必须写"commit 为声明值、未校验"**
- [ ] 日志 `trlab/logs/s3_imatrix.log` 末尾没有 `bailing out` / `error`
- [ ] **Save Version（Quick Save）**，把 imatrix 与记录存进版本

---

## 六、S4：一键量化（v1.0.1：一个开关 + Run All + 自动预检）

### 6.0 唯一操作开关

```python
RUN_S4 = True
```

它自动打开 S2 + S3 + S4 + S5，S6–S8 与 SX 全部保持关闭。**不设 `BASE_GGUF`** —— 量化 base 由 S4 自己按优先级取。已有的 `imatrix.gguf` 只要 sha256 与 `IMATRIX_PINNED`（`34062ab6…`）一致，S3 就整段跳过：不重下 8.11 GiB 的 Q8_0、不重算 58 秒的 imatrix。

**先跑预检，再动工（v1.0.1）**：`Run All` 的第一件事是 `S4_PREFLIGHT`，几秒钟、只读，输出一段摘要（working 空闲/峰值、BF16 落点、imatrix 状态、可用暂存盘数量、S3 会不会重下 8.11 GiB）。它只回答两个问题：**BF16 有地方放吗？两个产物在 working 放得下吗？** 任一为否就地 `GuardError`，只给一步要做的，不会先花 28 分钟编译再失败。明细落在 `/kaggle/working/trlab/out/S4_PREFLIGHT.json`。

完整操作只有三步：上传 7.3 KB 的 `trlab_kaggle.ipynb` → 把开关格改成 `RUN_S4 = True`、`Settings → Internet = On` → `Run All`。第二个代码格会自己从公开仓库取执行包（约 200 KB、1–3 秒，校验 commit 后 import），不需要 Save Version、Add Input 或配任何路径。

### 6.1 BF16 的一次性操作（**最少点击：6 次**）

BF16 有 15.26 GiB，`/kaggle/working` 只有 20 GB，S2 之后实测只剩 19.07 GiB。算一遍就知道单次 Run All 内同时放三个文件不可能：

| 账 | 数字 |
|---|---|
| BF16 + IQ2_XS + IQ4_XS | 15.26 + 2.36 + 4.34 = **21.96 GiB** > 19.07 |
| 连 BF16 + IQ4_XS | 19.60 GiB，仍然超 |

所以**如果预检说「working 之外没有可用暂存盘」**，第一次跑 `RUN_S4=True` 之前要用附带的 `trlab_fetch_bf16.ipynb` 把 BF16 变成只读 Input（预检会直接点名这一步，不用自己判断）：

| # | 点击 | 说明 |
|---|---|---|
| 1 | Kaggle → `File` → `Import Notebook` → 上传 `trlab_fetch_bf16.ipynb` | 3 个 cell，只做下载 |
| 2 | 右侧 `Settings` → `Internet` = **On** | |
| 3 | `Run All` | 下载 pinned BF16（16,388,044,384 B，sha256 `5e416a20…`）并校验，约 15–30 分钟（**估算**，取决于 Kaggle 出口带宽） |
| 4 | `Save Version`（Quick Save 即可） | 产物进入该 notebook 的 Output |
| 5 | 回到主 notebook → 右侧 `Add Input` → 选那个 notebook 的输出 | 官方文档明确支持搜「任何 Notebook 的输出文件」 |
| 6 | 改 `RUN_S4 = True` → `Run All` | 之后**永久复用、不再下载** |

**如果 Kaggle 提供了 working 之外的暂存盘**，第 1–5 步可以省掉：notebook 启动时会探测 `st_dev` 与 working 不同的候选，够大就直接下到那里（不占 20 GB 配额，代价是会话结束就没了，下次要重下）。探测不到才走上面的 Input 路线，两种情况下报错信息都会说清楚该点哪几下。

### 6.2 量化 base 四级取源（自动，不用配路径）

| # | 来源 | 条件 | 结果记录 |
|---|---|---|---|
| 1 | `CFG['BASE_GGUF']` 手动指定 | 文件存在 | `quant_base_origin=manual` |
| 2 | `/kaggle/input` 已挂载 | 字节数与 pin 一致；开 `HASH_GGUF` 时 sha256 强校验 | `attached_input` |
| 3 | working 之外的暂存盘 | 另一个文件系统且空闲 ≥ BF16 + 4 GiB | `scratch_download` |
| 4 | working 自己 | 空闲 ≥ BF16 + 两个产物 + 保底 | `working_download` |
| — | 都走不通 | — | `GuardError` + 上面那张 6 步表 |

### 6.3 同源性核验：Q8_0 不在了也能核

S3 成功后 Q8_0 会被回收（8.11 GiB），所以 S4 **不再要求**读 imatrix base 原文件：

| 等级 | 何时 | 依据 |
|---|---|---|
| `file` | Q8_0 文件还在 | 逐张量比形状（architecture / 张量名 / dims / 参数量） |
| `provenance` | Q8_0 已回收 | ① imatrix sha256 与记录一致；② 记录的 base repo/revision 与量化 base 相同（`unsloth/Qwen3-8B-GGUF@a6adef13…`，同仓库同 revision ⇒ 同一次转换 ⇒ 同一份权重）；③ imatrix 张量名 ⊆ base 张量名且层数一致；④ 每个 imatrix 张量的 `in_sum` 元素数 = base 对应张量的输入通道数 |

`provenance` 级弱于 `file` 级，S9 报告会显式声明这一条的效力范围（`imatrix_base_consistency_level`）。imatrix 旁边的 `models/imatrix.provenance.json`（schema 2）是"新 Session 恢复"的关键：OUT/ 可能随会话丢，models/ 里的产物和它的 provenance 是一起被保存/挂载的。

### 6.4 容量与 RAM 账

| 项 | 大小 | 类型 |
|---|---|---|
| `build-llama/` | 1.5–3.5 GiB | 估算 |
| `models/imatrix.gguf` | 0.01 GiB | **实测**（10,701,952 B） |
| `models/qwen3-8b-IQ2_XS.gguf` | 2.3–2.6 GiB | 估算（2.3125 bpw × 8.19e9 ≈ 2.36 GiB + embed/head 高精度保留） |
| `models/qwen3-8b-IQ4_XS.gguf` | 4.3–4.6 GiB | 估算（4.25 bpw ≈ 4.34 GiB + 同上） |
| logs + out | < 0.2 GiB | 估算 |
| **合计（不含 base）** | **8.1–11.0 GiB** | |

**止损值**：S2 之后剩余 ≥ 12 GiB；S4 之前剩余 ≥ 10 GiB；`guard()` 只在阶段入口检查，跑起来之后管不了。

**RAM**：`llama-quantize` 是 CPU 路径，输入走 mmap（页缓存可回收），但 `--max-buffer-size` 默认 8192 MiB 是实打实要占的。`_ram_precheck()` 会对比「base + `S4_RAM_MARGIN_GB`(4 GiB)」与实测量，不够时警告并提示两种降压办法；真的 `exit 137` 就调小 `S4_RAM_MARGIN_GB` 或加 `--max-buffer-size`。

### 6.5 禁止二次量化

`RUN_S4` 路线上 `ALLOW_REQUANTIZE` 永远保持 `False`，且量化 plan 带任何 flag 时**直接 `GuardError`** —— 上游帮助原文是 `WARNING: this can severely reduce quality compared to quantizing from 16bit or 32bit!`。base 只可能是 pinned 的 `Qwen3-8B-BF16.gguf`。

### 6.6 S4/S5 人工核对清单

- [ ] `S4_QUANT.json` 的 `_fairness.quant_base_origin` 是 `attached_input` / `scratch_download` / `working_download` 之一，且 `quant_base_repo@revision` = `unsloth/Qwen3-8B-GGUF@a6adef13…`
- [ ] `_fairness.base_scope == "from_high_precision"`，`requant == "primary"`（出现 `requantized_pilot_only` 就说明走错了 base）
- [ ] `_fairness.imatrix_trust` 是 `pinned_verified` 或 `provenance_verified`
- [ ] `_fairness.imatrix_base_consistency_level` 是 `file` 或 `provenance`；`provenance` 时核对 S9 报告里的声明段
- [ ] `IQ2_XS` / `IQ4_XS` 的 `dominant_type` 分别就是 `IQ2_XS` / `IQ4_XS`（不是就要查量化是否真生效）
- [ ] `S5_BYTES.json` 里逐张量字节表完整，没有 `byte_breakdown_available: false` 的行
- [ ] 产物旁各有一个 `.provenance.json`（schema 2，含 base/imatrix sha256 与 `quant_fp`）
- [ ] **Save Version（Quick Save）**，把两个 GGUF 与记录存进版本

---

## 六之二、S6/S7：只评测，不再量化（v1.1.0）

**v1.2.0 起（2026-10-23 系统性审计）**：S6 不再"见 1 行重叠就整轮归零"，改为
取证 → 剔除 → 用同一份 clean holdout；`actual_imatrix_calib_overlap` 无法证明时
如实写 `unknown`。缺二进制仍会自动只编 S2，不需要第二个开关。

**前提**：S4/S5 的产物已经备份在 HF 私有仓库 `Atomheart-Father/trlab-qwen3-8b-quant-pilot`
（commit `097e692`），包含 `qwen3-8b-IQ2_XS.gguf`、`qwen3-8b-IQ4_XS.gguf`、
`imatrix.gguf` 与三份 `*.provenance.json`。所以这一节**不需要 BF16、不需要 S3、
不需要重新量化**。编译是**按需自动**的：新 Session 里 `llama-perplexity` /
`llama-cli` 不存在时，`RUN_EVAL=True` 会**自己只编 S2**（见 6.2.0 后的"两道门"），
你不用额外开开关。

### 6.2.0 唯一操作开关

把开关格里的 `RUN_EVAL = False` 改成 `RUN_EVAL = True`，右侧 `Settings`：
`Accelerator = GPU T4 ×2`、`Internet = On`，把 HF token 放进
`Add-ons → Secrets`（名字 `HF_TOKEN`，勾选 attach 到本 notebook），然后 `Run all`。
就这四步，没有别的页面操作。

`RUN_EVAL=True` 会展开成：`S6_NLL` + `S7_QUALITY` + `S9_REPORT`，并自动置
`HF_FETCH_ARTIFACTS=True`（允许从 HF 取输入）与 `HF_UPLOAD_AFTER_RUN=True`
（每个阶段的小结果出站）。它**不会**打开 `S3/S4/S5`；`S2_BUILD` 只在
"二进制缺失"时被自动打开（v1.1.2，见下）。

#### 两道门（v1.1.2 起，顺序是刻意的）

```text
S0/S1 → ① 二进制探活（本地、免费）→ ② HF 授权检查（1 次只读）
      → ③ 缺二进制则自动只编 S2 → S6 → S7 → S9
```

| 门 | 何时 | 行为 |
|---|---|---|
| ① 探活 | 永远（不发网络） | 缺 `llama-perplexity`/`llama-cli` 只打一行日志，不报错 |
| ② 授权 | 永远（编译前） | 无 token 连请求都不发（`status=no_token`）；不通就写 `FAILURE.json` + 出报告，**3 秒内停**，不会白等 28 分钟编译 |
| ③ 自动 S2 | 仅当 ① 缺二进制 | `_s2_only()`：只开 `S2_BUILD` + pinned 源码 36.6 MiB，**不下载模型、不做 S3/S4、不重量化** |

所以"新 Session + 只开 `RUN_EVAL` + `Run all`"现在**成立**了；v1.1.1 里它不成立
（`eval_preflight()` 第 1 项就要求二进制存在）。S2 若失败，S6/S7 **主动跳过**
（不是"开关没开"），`FAILURE.json` 记 `s2_build_failed`，报告里写明"这是失败的后果"。

**默认用 GPU**：`EVAL_ALLOW_GPU=True`，检测到显存就 `-ngl 99`（整模型卸载）。
装不下才退 CPU，并且**一定**在 `device_by_fmt` / `resources.device_why` 里写原因 ——
不允许静默降级。要强制纯 CPU 就 `--set EVAL_ALLOW_GPU=false`。

**会话可丢**：这一节的前提是 Kaggle 只是可随时关机的算力机器。跑失败就直接
关掉 Session，本地修代码 → commit → 新 Session 重跑。**不需要** Save Version、
不需要抢救工作区、不需要 Add Input。HF 上的输入是长期源，重跑只花一次下载时间。

### 6.2.1 token：三条路，任选一条，别写进仓库

| 方式 | 具体做法 |
|---|---|
| 环境变量 | 本地 shell `export HF_TOKEN=hf_xxx`，再 `python -m trlab --run-eval` |
| Kaggle secret | 网页 `Add-ons → Secrets` → 新建 `HF_TOKEN` → 在该 notebook 上勾选 attach。代码用 `kaggle_secrets.UserSecretsClient` 读 |
| HF 官方缓存 | 本机 `huggingface-cli login` 一次，之后 `~/.cache/huggingface/token` 会被复用 |

三者都没有时，**凭据门在编译之前**就把运行停住（v1.1.2）：不发下载请求、不开始
编译，`FAILURE.json` 里 `cause=hf_no_token`，`causes` 会同时列出
`binaries_missing`（如果你这个 Session 也没二进制），`next_step` 直接写
"`Add-ons → Secrets` 新建 `HF_TOKEN`……**attach** 它，再 `Run all`"。
报告顶部会写 `## ⚠ 本次运行失败` + "这份报告是**失败现场**，不是结果"。
**不要**把 token 贴进 notebook cell、git、或 Issue 评论里；代码本身也不会打印它
（只打印 `token_source`，如 `env:HF_TOKEN`）。

私仓可见性**不需要改**：只要 attach 了带 Read 权限的 token 就能读。代码不会替你
把仓库改成 Public（可见性是仓库所有者的决定）；真要走匿名路线由你自己决定。

一次配置之后就不用再管：同一个 Notebook 只要 token 还 attach 着，之后每次
`Run all` 都直接过门。

### 6.2.2 会下载什么、多大

| 文件 | 用途 | 量级 |
|---|---|---|
| `qwen3-8b-IQ2_XS.gguf` | S6/S7 的输入之一 | ≈2.7 GB |
| `qwen3-8b-IQ4_XS.gguf` | 同上 | ≈4.56 GB |
| `imatrix.gguf` | 只做 provenance 引用，不参与计算 | 10.7 MB |
| `*.provenance.json` ×3 | fairness / 来源记录 | KB 级 |
| wikitext-2-raw **test** parquet | 评测语料（**不是** train） | 732,610 B |

合计 ≈7.3 GB。权威字节数以 HF tree API 与 sidecar 为准，上表只是量级。
`EVAL_VERIFY_HASH=True`（默认）会在评测前把本地 GGUF 与 tree 的 `lfs.oid`
对一遍（4.5 GiB 约 2–4 s/个）。

### 6.2.3 评测语料与校准语料是两份不同的东西

| | 用途 | 来源 | 大小 |
|---|---|---|---|
| 校准集 | S3 算 imatrix | wikitext-2-raw **train** | 6,357,543 B |
| 评测集 | S6/S7 打分 | wikitext-2-raw **test** | 732,610 B |

两者 sha256 都预登记（`CALIB_PINNED` / `EVAL_PINNED`），S6 会做内容级重叠断言，
重叠非零直接拒绝。S6 单独跑时也会把 calib.txt 取来（6.35 MB）只为做这个断言，
失败只 log 不致命。

### 6.2.4 规模、设备与成本

`NLL_CTX=512`、`NLL_CHUNKS_PER_GROUP=1`、`NLL_GROUPS=8` → 8 组 × 1 chunk × 512 ctx
≈ 4k token/模型、共 8k token。**一次只装一个模型**：装 IQ2 → 跑完 → 释放 → 再装 IQ4。

`eval_cost_estimate()` 用**假设速率**（GPU 30 tok/s / CPU 2.5 tok/s，字段
`assumed_rate_note` 写明"不是实测"）估算墙钟，并按 `eval_device()` 选设备：

| 设备 | 默认规模估算 | 是否受 `EVAL_CPU_MAX_WALL_SEC` 约束 |
|---|---|---|
| GPU（T4，装得下） | 分钟级 | **否**（`budget_applies=False`） |
| CPU | ≈0.94 h | 是（5400 s；旧的 8×4×2048 是 14.59 h，会被拦） |

CPU 被拦时预检给三个选项，不是跑一半被杀：缩小 pilot / 开 GPU Session
（代码默认就会用）/ 只跑 S7。

### 6.2.5 6.2 人工核对清单

- [ ] `S6_NLL.json` 的 `provenance.inputs` 三个文件都是 `source` 为 `hf`（或 `local`），
      且 `sha256_verified` 与你的预期一致
- [ ] **先看设备**：`device_by_fmt` 每个格式是 `gpu` 还是 `cpu`，`why` 写的是什么。
      是 `cpu` 而你以为在 GPU 上 → 说明装不下或没读到显存，别把慢当成模型慢
- [ ] `fairness_level` 是 `hf_sidecar_declared`（不是 `s4_runtime`）。**引用结论时必须带这个前缀**
- [ ] `settings.one_model_at_a_time` 为 `True`；日志里能看到"装入 / 释放"成对出现
- [ ] `measured_throughput` 有值 —— 它才是实测，报告里的估算是假设
- [ ] 报告第 6/7 节写的是数字，不是 "not yet evaluated"（写了就说明 S6/S7 没真跑）
- [ ] `S7_QUALITY.json` 的 `settings` 与 S6 一致（同 prompt 数 / 同 n_predict / 同 seed）
- [ ] `EVAL_SUMMARY.json` 已生成（<3 KB，可直接贴聊天），`upload` 里
      `remote_verified` 为 `true`；若 `not_persisted` 非空，说明这些结果**还没落地**

### 6.2.6 失败恢复

先看 `out/FAILURE.json`：**顶层永远是最早那条根因**（`stage` / `cause` / `error` /
`next_step`），`causes` 是去重后的根因代码，`all` 按时间顺序列出全部失败 ——
后面的阶段失败**不会**把最早根因覆盖掉（v1.1.2）。所以"缺二进制"和"私仓没认证"
能在一屏里同时看到。

| 现象 | 去哪看 | 处置 |
|---|---|---|
| `cause=hf_no_token` | `FAILURE.json` | 按 6.2.1 配一次 `HF_TOKEN` 并 **attach**，再 `Run all`。**不要把 token 写进 notebook** |
| `cause=hf_bad_request`（HTTP 400） | 同上（看"服务器原话"） | **不是你的 Secrets，不要重建 token**。400 来自 tree API 参数（v1.1.2 用了 `expand=true` + `limit=1000`，Hub 只允许 expand 时 ≤50）。换 v1.1.3 的 notebook 重跑；若新版仍 400，把 `FAILURE.json` 贴到 Issue #5 |
| `cause=hf_unauthorized`（401/403） | 同上 | token 无效或对该 repo 没权限。`--check-hf` 可以只验授权不下载。注意：**私仓无 token 返回的是 401，不是 404** |
| `cause=hf_not_found`（404） | 同上 | revision 或 repo 路径不对（实测：不存在的 repo 也是 401，只有坏 revision 才是 404，正文 `Invalid rev id: ...`）。核对 `CFG['HF_MODEL_REPO']` 与 revision |
| `cause=hf_rate_limited`（429/5xx） | 同上 | Hub 侧限流或临时故障，等 30–60 秒重跑，不用改配置 |
| `cause=hf_network` | 同上 | 连不上 HF：先确认 `Settings → Internet = On`，再看 Kaggle 出口；不涉及 token |
| `cause=binaries_missing` | 同上 | **你不需要单独开 S2**：v1.1.2 起 `RUN_EVAL=True` 会自动只编 S2。若它反复出现，看 `S2_BUILD` 的日志 |
| `isolation` 报剔除比例超限 / 保留段数不足 | `out/isolation/isolation.json` | 评测集里有 >10% 的段与校准集共享前 200 字符（像把 train 当 eval），或清洗后不足 20 段。**不要**调大 `ISOLATION_MAX_DROP_FRAC` 绕过；核对 `EVAL_TXT` 是否真的指向 test split |
| `normalize_expect` / `size 与 bytes 不一致` | `FAILURE.json` | 调用方混用了两种 schema —— 这是代码缺陷，不是环境问题。把 `FAILURE.json` 贴回 Issue #5 |
| `实际污染`（交集里出现 `full_equal=true`） | `out/isolation/isolation.json` 的 `overlap_raw.details` | 整段完全相同才需要认真对待：换 `EVAL_TXT` 或先重跑 S3 换校准子集，**不要**跳过断言 |
| `cause=s2_build_failed` | `out/S2_BUILD.json` + `logs/s2_*.log` | 编译本身失败，S6/S7 已主动跳过。按第七节修 S2 |
| `估算 X h 超过上限` | `out/S6_NLL.json` 的 `cost_estimate` | 按 6.2.4 的三选一；最省事是开 GPU Session |
| `sha256 与 HF 不符` | 同上 | 本地缓存坏了。删掉那个 GGUF 重跑（`.partial` 会续传，但哈希不符必须整个重来） |
| 下载中断 | 日志里的 `.partial` 提示 | 直接重跑，`Range` 续传；输入指纹没变就会接着下 |
| `probe_only` 时看到 `source: hf_pending` | `out/S6_NLL.json` | 正常：预检不拉 7 GB，真跑时才下载 |
| 跑到一半 OOM | `logs/ppl_*.log` 尾部 | 降 `NLL_CTX`（512 → 256），或 `--set EVAL_ALLOW_GPU=false` 退 CPU |
| 上传失败但评测成功 | `EVAL_SUMMARY.json` 的 `upload.not_persisted` | 结果仍在本地产物目录；可直接把 `EVAL_SUMMARY.json` 贴到聊天，或在同一 Session 里再跑 `--upload` |
| 报告顶部出现 `## ⚠ 本次运行失败` | `report.md` 第 0 节 | 这就是失败现场（不是结果页）。`run()` 返回 `ok=False`，`failed` 列出阶段、`causes` 列出根因 |
| `sha256 与 HF 不符` | 同上 | 本地缓存坏了。删掉那个 GGUF 重跑（`.partial` 会续传，但哈希不符必须整个重来） |
| 下载中断 | 日志里的 `.partial` 提示 | 直接重跑，`Range` 续传；输入指纹没变就会接着下 |
| `probe_only` 时看到 `source: hf_pending` | `out/S6_NLL.json` | 正常：预检不拉 7 GB，真跑时才下载 |
| 跑到一半 OOM | `logs/ppl_*.log` 尾部 | 降 `NLL_CTX`（512 → 256），或 `--set EVAL_ALLOW_GPU=false` 退 CPU |
| 上传失败但评测成功 | `EVAL_SUMMARY.json` 的 `upload.not_persisted` | 结果仍在本地产物目录；可直接把 `EVAL_SUMMARY.json` 贴到聊天，或在同一 Session 里再跑 `--upload` |

### 6.2.7 结果落在哪（会话可丢的前提）

每个阶段跑完就出站，不等全部跑完：

| 阶段 | 出站文件 | HF 路径 |
|---|---|---|
| `S6_NLL` | `S6_NLL.json`、`EVAL_SUMMARY.json` | `results/v1.1.3/<run_id>/` |
| `S7_QUALITY` | `S7_QUALITY.json` | 同上 |
| `S9_REPORT` | `report.md`、`results.json`、`S9_REPORT.json` | 同上 |

`<run_id>` 是 UTC 时间戳（如 `20261023T012345Z`），多轮不互相覆盖。
**不传**：GGUF（模型资产，不是运行产物）、`logs/*.log`（可能带路径）、任何 token。
传完立刻远端复核（size 必比、LFS 再比 `lfs.oid`）；失败会把文件名写进
`not_persisted` 并打印"这些结果**尚未持久化**"。

`EVAL_SUMMARY.json` 是几百字节的摘要（均值 NLL、配对差 + CI、verdict、
fairness_level、逐格式设备、吞吐、S7 的 identical_rate、upload 状态、三条 limits），
**设计上就是为了直接贴到聊天里**。

## 七、失败排查

| 现象 | 去哪看 | 处置 |
|---|---|---|
| `GuardError: 下载源码失败 ... 检查 Settings → Internet = On` | `out/S2_BUILD.json` 的 `error` | Internet 没开，或 Kaggle 网络策略拦了 GitHub。开 Internet 重试；仍不行就走 4.1 的断网备选（Mac 下载 → Dataset → `CFG["LLAMA_SRC"]`） |
| `GuardError: 源码包 sha256 不匹配，拒绝解压` | 同上 | 上游文件被替换，或网络返回了错误页面。核对报错里那两个 sha256；不要手动放行 |
| `GuardError: 归档里有可疑路径/链接，拒绝解压` | 同上 | 包被动过。重新下载，或走断网备选 |
| `GuardError: 归档顶层目录不是预期的 ...` | 同上 | digest 对但目录名不符 → 预登记信息本身有问题，升级 notebook 版本 |
| `GuardError: 找不到 llama.cpp 源码，且没有打开自动获取` | 同上 | 把 `RUN_S2` 改成 `True`（需 Internet=On），或设 `CFG["LLAMA_SRC"]` 走手动源码 |
| `源码 commit ... 与预注册的 ... 不符` | 同上 | 改 `LLAMA_COMMIT`，或换回你打包的那个 commit |
| `GuardError: BUILD_CUDA=True 但 nvcc 不可用` | 同上 | `CFG["BUILD_CUDA"]=False` |
| `CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:181 (target_link_libraries): Target "ggml-cuda" links to: CUDA::cuda_driver but the target was not found.` | `logs/s2_cmake.log` | v0.3.1 起默认 `-DGGML_CUDA_NO_VMM=ON` 已绕开；若你手动把 `CFG["CUDA_NO_VMM"]` 关成了 `False`，改回 `True` |
| `S2 编译失败（rc=N）` + 一句"症状 → 处置" | `logs/s2_build.log` / `s2_cmake.log` 尾部 | S2 已把常见症状翻译成一句话（host g++ 不兼容 / 找不到 nvcc / 磁盘满 / 内存不足 / cmake 配置阶段 / CUDA::cuda_driver），按那句话做 |
| `CommandFailed rc=1` + `CMake Error` | `logs/s2_cmake.log` | 看最后一屏；`unsupported GNU version` → 关 CUDA；缺 `CUDAToolkit` → 同上 |
| `BudgetExceeded: 命令超预算被杀` | `out/S2_BUILD.json` | 把 `STAGE_BUDGET_SEC["S2_BUILD"]` 调到 5400，`FORCE_RERUN=True` |
| `编译结束但缺少可执行文件: ['llama-cli']` | `logs/s2_build.log` | 大概率有人关了 `LLAMA_BUILD_SERVER`；恢复默认 |
| `UI: no assets available` | `s2_build.log` | **可忽略**，我们不构建 server UI |
| `CFG['IMATRIX_BASE'] 指向的路径不存在` | `out/S3_IMATRIX.json` | 检查 `Add Input` 是否挂上、路径拼写 |
| `S3_IMATRIX SKIP —— 前置阶段 S2_BUILD 失败` | `out/S3_BUILD.json` / `out/S3_IMATRIX.json` | v0.4.0 的前置守卫：S2 真的失败了，S3 不会拿一堆残缺文件硬跑。先修 S2 再重跑，Q8_0 不会白下 |
| `GuardError: 磁盘空间不足 ... 缺口 X GiB`（下载前） | `out/S3_IMATRIX.json` | 清 working：`rm -rf trlab/build-llama` 以外的旧产物。**别删 `.partial`**，那是续传断点 |
| `下载得到 N B，预登记 M B` / `sha256 不匹配` / `x-linked-etag 不符` | 日志 | revision 被上游替换，或网络返回了错误页面。核对报错里那两个值；**不要手动放行**。`.partial` 会保留，重跑续传 |
| `imatrix 产物缺失或过小` | `logs/s3_imatrix.log` | 看是否 `bailing out`；常见是校准文本太短（< 200 KB 会警告，至少需要 2×512 = 1024 token） |
| `-ngl 99 显存不足 → 降到 -ngl 18 重试` | 日志 + `ngl_attempts` | 正常降级路径，不是错误。降到 0 会变成纯 CPU，小时级 |
| `imatrix 失败（已试 -ngl [99, 18, 0]）` + 一句"症状 → 处置" | `logs/s3_imatrix.log` | S3 已把常见症状翻译成人话（显存不足 / 模型打不开 / token 数不够 / imatrix 相关）。Q8_0 与断点都还在，修完重跑不重下 |
| `parquet 解析失败：镜像里没有 pyarrow / pandas` | `out/S3_IMATRIX.json` | Kaggle 镜像待核项。退路：把 parquet 自己转成 `calib.txt` 挂成 Dataset，设 `CFG["CALIB_TXT"]` 指过去 |
| `S3` 跑了很久没动静 | `logs/s3_imatrix.log` | `ngl=0` 说明退 CPU 了；确认 Q8_0 真的 8.11 GiB |
| 复用二进制但 `commit_source == "declared"` | `out/S2_BUILD.json` | 这是**预期行为**（没有源码就没法 git 校验），不是 bug。想要 `pinned_tarball` 档就让 S2 自己下源码 |
| `GuardError: BF16 ... 放不进 /kaggle/working ...` | `out/S4_QUANT.json` | 磁盘账不过，且没找到够大的暂存盘。按报错里的 6 步用 `trlab_fetch_bf16.ipynb` 把 BF16 挂成只读 Input，再重跑 |
| `GuardError: CFG['BASE_GGUF'] 未设置，且 S4_AUTO_FETCH_BASE=False` | 同上 | 手动模式下要自己把 `BASE_GGUF` 指到已挂载的 BF16；或改用 `RUN_S4=True` |
| `GuardError: imatrix 的 sha256 与任何已知记录都对不上` | 同上 | working/Input 里的 imatrix 既不是 pin 的那个、也没有 provenance 记录。确认是你有意换的统计量就设 `CFG["IMATRIX_ALLOW_UNPINNED"]=True`，否则把 `IMATRIX_GGUF` 指回已验证文件 |
| `GuardError: BASE_KIND=q8_0 是量化类型，而本路线禁止二次量化` | 同上 | 取 base 时拿到了量化文件。检查 `/kaggle/input` 里挂的是不是 `Qwen3-8B-BF16.gguf`（字节数 16,388,044,384） |
| `GuardError: imatrix 与量化 base 无法证明同源` | 同上 | 按报错里的等级（`structure` / `provenance`）逐项核对：张量名、层数、输入通道、sha256、来源 repo/revision |
| `S4 量化失败（rc=N）` + `requantizing` 提示 | `logs/s4_quant_*.log` | base 是量化类型。本路线不给 `--allow-requantize`，换 BF16 |
| `llama-quantize` 被 kill / `exit 137` | `logs/s4_quant_*.log` | RAM 不够。调小 `CFG["S4_RAM_MARGIN_GB"]`，或在命令里加 `--max-buffer-size` 降压（见 6.4） |
| `IQ2_XS 的主导类型是 ...，不是 IQ2_XS` | `logs/s4_quant_IQ2_XS.log` | 量化没生效或产物被复用错。删掉 `models/qwen3-8b-IQ2_XS.gguf*` 重跑 |
| `S5` 报 `没有可分析的 GGUF` | `out/S5_BYTES.json` | S4 没成功。S5 只分析 S4 实际产出的文件 |

所有阶段失败都**不会中断**后续阶段：每个阶段自己写 `status: "error"` + traceback，S9 仍会出一份带错误摘要的报告。

---

## 八、未验证清单（不要在报告里写成已验证）

1. ~~S2 从未在 Kaggle 上编译过~~ **已被实测取代（v0.3.2，27 分 52 秒通过）**。仍未观测：build 目录大小、峰值 RAM。
2. ~~Kaggle 镜像的 nvcc 与 g++ 版本兼容性未验证~~ **已实测**：CUDA 12.8.93 × host GNU 13.3.0 兼容。
3. ~~**S3 的 imatrix 运行时间未实测**~~ **已被实测取代（2026-10-22）**：57.95 s，产物 10,701,952 B / 1008 张量，`-ngl 99` 全上 GPU，显存总量 14.56 GiB。
4. ~~**S3 的整条一键链路未在真实 Kaggle 上跑过**~~ **已被实测取代**：Q8_0 下载、imatrix 生成与验证、Q8_0 回收全部真实发生过，`S3_IMATRIX-2.json` 为证。
5. **Kaggle 能否直连 huggingface.co 未验证**（github.com 与 HF 的 resolve URL 已实测可达 —— S3 的 Q8_0 就是那么下来的；但 **15.26 GiB 的 BF16 下载未实测**，长连接中断后的续传行为也只有 mock 覆盖）。
6. **Kaggle 镜像里有没有 `pyarrow` / `pandas` 未验证**。设计上不硬依赖（先 pyarrow 再 pandas，都没有就报可操作的错），但第一次跑可能踩到。
7. **`/kaggle/working` 是否跨 interactive session 保留未验证**（官方只说 ≤ 20 GB 自动保存）。v0.4.0 起按"不保"做最坏假设，每次启动探活二进制；探不到就自动重编 ≈ 28 分钟。**首次新 session 重编是可接受代价，已如实记录。**
8. **"notebook 输出当输入" 的界面入口与 Quick Save 是否保存 working 未验证**（3.3 节标了待核）。
9. **没有找到公开的 Qwen3-8B Q8_0 / BF16 Kaggle Dataset**（检索了 5 组关键词，2026-10-21）；检索 API 不穷举。
10. **Kaggle 上那些 llama.cpp 数据集到底装了什么，没有验证**。公开检索 API 返回的 `resources` 是空的（要登录才给文件清单），我只能从体积和标题判断它们多半是预编译二进制。**不要**把它们当源码用。
11. **31 GiB RAM 跑 15.26 GiB 输入的量化为未实测**（6.4 节）。`--max-buffer-size` 默认 8192 MiB 是从帮助文本读到的，实际峰值没有观测。`_ram_precheck()` 只警告不拦。
12. **`llama-quantize` 的 CPU 耗时未实测**（1800 s 预算是按 IQ2_XS/IQ4_XS 各 10–20 分钟估的）。
13. **Kaggle 暂存盘的容量没有官方数字**。官方只说有 "Additional scratchpad disk space (outside /kaggle/working)"，没给大小。v0.5 只认 `st_dev` 与 working 不同的候选且要求空闲 ≥ BF16 + 4 GiB；探测不到就走只读 Input 路线，不会把账算错。
14. **`provenance` 级同源核验未在真实运行中走过**：它靠的三项证据（sha256 / 来源 repo+revision / 张量结构）都有 mock 覆盖，但"Q8_0 已回收 + 新 Session 恢复 + S4 量化"这条真实路径第一次跑才算数。
15. **自动下载路径的 S2 部分已被实测**（GitHub 直连、sha256 校验、安全解压）；**S3 的 HF 下载路径已被实测**（Q8_0 8.11 GiB 真实下载并校验通过）；**BF16 15.26 GiB 的下载路径只有 mock 覆盖**（含断点续传、坏 partial 重启）。
16. ~~`-DGGML_CUDA_NO_VMM=ON` 只跨过 Generate 阶段~~ **已被实测取代（v0.3.2）**：build 阶段同样通过。仍未观测：build 目录大小、峰值 RAM。
17. **tarball 源码没有 `.git`**，自动下载路径记 `commit_source="pinned_tarball"` + `commit_verified=true` —— 它证明"这份 tar.gz 的 sha256 与预登记值一致"，**不等于** git 校验。复用二进制时只能是 `declared`。
18. **预登记 sha256 本身是自证的**：它写在 notebook 里，防"上游文件被替换 / 网络返回错误页面"，防不了"notebook 里的值被改坏"。首次使用请在 S3/S4 结果里核对那串 sha256 与 3.2 表格一致。
19. **v1.0 的引导格未在真实 Kaggle 上跑过**（本地已验证）：GitHub 匿名 sparse fetch、`rev-parse` 强校验、stamp 复用都在干净目录实测通过，但 Kaggle 镜像里 `git` 的版本与 `core.sparseCheckout` 支持情况、以及 Kaggle 出口到 github.com 的带宽/限流未实测。失败时的报错信息会说明原因，`--embed` 内嵌变体可绕过仓库依赖。
20. **仓库转私有后匿名 fetch 会失效**。当前 GitHub 仓库是 public（用户 2026-10-22 设为公开）。若之后转回私有，用 `python3 build_notebook.py --embed` 生成内嵌执行包的版本，或在 Kaggle 环境里配好身份。
21. **Kaggle 上重复 `Run All` 的 stamp 复用未实测**：同一 Session 内第二次执行应命中 `_trlab_src/.trlab_commit` 直接复用、不重新 fetch；这是本地验证过的逻辑，Kaggle 上的 `/kaggle/working` 持久化行为（第 7 条）会叠加影响。
22. **S6/S7 从未在真实 Kaggle 上跑通过**（v1.2.0）。171 项测试全是 CPU/mock，`test_s6_end_to_end_from_hf` 用的是假 HF 服务器桩。2026-10-23 的真实首跑在**评测开始前**就停了（缺二进制 + 私仓无 token），只留下 11 秒的失败现场。**所以"量化对 NLL / 质量的影响"目前没有数据**，报告那两节会写 "S6/S7 not yet evaluated" 或直接标"失败"。
   - 附带：**自动只编 S2 与凭据门只在 mock + 本机 macOS 上走过**（v1.1.2）。真实 Kaggle 的 CUDA 编译（`CUDA_ARCH=75` / `BUILD_JOBS=4`）与"`RUN_EVAL` 自动开 S2"这条路径都**没有真机证据**；本机复现的是"无 token + 无二进制 → 3.1 s fail fast、两条根因、不开始编译"。
23. **HF 真实下载只验到 401 路径**。`python -m trlab --check-hf` 在无 token 时正确报 `token_present: false` + 401 + 三条取 token 的路（真机 macOS 跑过）；**带 token 下载 2.7 / 4.56 GiB 未实测**。断点续传逻辑与 S3 的 Q8_0 下载共用同一份 `_fetch_pinned_blob`，那份是真机跑过的。
24. **`hf_sidecar_declared` 级 fairness 未在真实 sidecar 上走过**。测试里是构造的；真实 sidecar 字段是否齐、跨 Session 会不会误判 `imatrix_base_consistent: false`，第一次真跑才知道。
25. **往 HF 上传没有真的发生过**。v1.1.1 起 `RUN_EVAL=True` 会自动出站 5 个小文件，但真实上传只有 dry-run 与 mock 覆盖。上传时只做远端 size 对比（LFS 再比 sha256），不是逐文件内容复核。
26. **评测成本是假设速率**（GPU 30 / CPU 2.5 tok/s）。不能当承诺，真实数字要等 `measured_throughput`。
27. **"默认用 GPU"不等于"一定在 GPU 上"**：读不到 CUDA 显存就是 CPU，只写 `device_why` 不报错。看结果先看 `device_by_fmt`。
28. **`-ngl 99` 不是 OOM 免疫**：`NGL_MARGIN_GB=2.0` 是静态余量，KV cache + compute buffer 仍可能超。真 OOM 就降 ctx 或退 CPU。

### 已被真实 Kaggle 运行验证的部分（v0.3.0–v0.5.0，2026-10-21/22）

1. **源码自动下载与 sha256 校验通过** —— GitHub 直连在 Kaggle 上可用，digest 比对通过，安全解压到 `WORK/src/` 成功。
2. **工具链预检通过** —— CUDA 12.8.93 FOUND、nvcc host GNU 13.3.0 检测通过、arch=75、`-- Including CUDA backend`。
3. **`CUDA::cuda_driver` 热修有效** —— `-DGGML_CUDA_NO_VMM=ON` 之后 Generate 与 build 全部通过。
4. **S2 完整通过** —— 27 分 52 秒，四个二进制就位，S2 后 working 剩 19.07 GiB，`commit_source=pinned_tarball` + `commit_verified=true`。
5. **S3 完整通过** —— Q8_0（8,709,919,168 B）下载并校验、校准语料转换、`llama-imatrix` 57.95 s 产出 10,701,952 B / 1008 张量、`-ngl 99` 全上 GPU（显存 14.56 GiB）、Q8_0 回收、provenance 落盘。产物 sha256 `34062ab6…` 已 pin 进 `IMATRIX_PINNED`。
6. **v0.4 真实运行暴露的两个真 bug 已修** —— `base_origin` 误记 `manual` 导致 Q8_0 未回收（`_s3_base_origin()` 改为看路径与 stamp）；`_base_consistency()` 要求读已回收文件（改为 file / provenance 两级）。

### 已在本地验证的部分（v1.0.1，2026-10-22，非 Kaggle）

1. **真实冷启动**：干净目录执行 notebook 的两个代码格 —— GitHub sparse fetch 钉 commit（约 200 KB、1–3 秒）→ `git rev-parse HEAD` 强校验 → `import trlab` → S0/S1/S9 全部 `status=ok` → `report.md` / `results.json` 落盘，全程约 3 秒。
2. **取到的包与本地 `trlab/` 逐字节一致**（`diff -r` 无差异）；同一 Session 二次执行命中 stamp 直接复用，不重新 fetch。
3. **内嵌回退**：把 `SRC_URL` 指向不存在路径 → 三次 fetch 失败 → 自动解压 notebook 内嵌包（sha256 校验通过）→ 同样能 import；把 `_BUNDLE_SHA` 改坏 → fail closed，不解压、不 import。
4. **118 项 CPU 回归**（不联网、约 31 秒），含 10 项 v1.0 架构回归 + 10 项 v1.0.1 预检回归；其中 3 项 bootstrap 测试用本地 git 仓当"公开仓库"，保持离线。
5. **预检真跑过**（本机 macOS，`trlab.run(preflight_only=True)`，0.11 s）：working 126.74 GiB 可写；`/tmp` 与 `/var/tmp` 被判「与 working 同一个文件系统」（正确，本机就是同一个卷）；imatrix `missing` 并注明 S3 会重下 8.11 GiB + 重算；BF16 走 `working_download`，峰值账 29.37 GiB 够；`preflight: ok`。
6. **修掉的两个 v1.0 真 bug**：内嵌包解压落点少了 `research/kaggle/trlab` 层级（回退路径 import 失败）；`run(**overrides)` 不写 STAGES 导致 S6–S8/SX 在薄 notebook 里无入口。

**没验证的（如实标注）**：Kaggle 上 `/kaggle/tmp` 是否存在、是否独立文件系统、空闲多少；预检的判定在真实 Kaggle 上会不会误判；S4 的真实 CPU 量化耗时与成功率。

### 已在本地验证的部分（v1.1.0/1.1.1，2026-10-23，非 Kaggle）

1. **S3 的 P0 回归已修并钉住**：`_write_stamp() got multiple values for keyword argument 'bytes'`。
   修法是把 `_parquet_to_txt()` 的返回键改成 `converted_bytes`、`_ensure_calib_txt()`
   改成显式 dict 合并 + 重名 `GuardError`（不再 `**` 摊字典）。
   回归是**跨函数全流程**的（假 pyarrow → 真转换 → 写 stamp → cache hit），
   另加一条 AST 静态不变量禁止"同一调用里既有 `bytes=` 又有 `**`"。
2. **144 项 CPU 回归全绿**（不联网、约 45 秒）：118 项旧 + 2 项 S3 回归 + 19 项 v1.1
   + 5 项 v1.1.1（设备不静默降级 / 预检报设备 / 白名单不含模型与日志 /
   未授权时列出未持久化 / 失败现场 + 可粘贴摘要）
   （token 链 / 绝不打印 token / tree 分页 / 错误码可操作性 / 复用与 etag 双源守卫 /
   `verify_remote` 只声明真核过的 / 上传门槛与 opt-in / 缺 `huggingface_hub` 时拒绝
   `pip install -U` / 本地优先 / `probe_only` 绝不下载 / sidecar fairness 只能到
   `declared` / CPU 不可行 fail closed / 预检缺二进制说唯一一步 / `RUN_EVAL` 作用域 /
   `run_ppl` 默认 `-ngl 0` / S6 从 HF 端到端 / 评测语料是 test split / `_ensure_eval_txt` 全流程）。
3. **`--check-hf` 真机跑过**：对 Private 仓库、无 token 的情况下返回
   `token_present: false` + 401 的可操作提示，且**没有**打印任何 token。
4. **CLI 入口可用**：`python -m trlab --help` 列出 `--run-eval/--eval-only`、
   `--check-hf`、`--upload`、`--preflight-only`。

**v1.1.0 没验证的（如实标注）**：S6/S7 在真实 Kaggle 上的一次完整运行；带 token 的
7.3 GB 真实下载；真实 sidecar 的 fairness 判定；往 HF 的真实上传；真实 CPU 吞吐
（0.94 h 是假设速率算的）。

### 已在本地验证的部分（v1.1.2，2026-10-23，非 Kaggle）

1. **149 项 CPU 回归全绿**（不联网、约 45 秒）：144 项旧 + 5 项 v1.1.2
   （无 token 无二进制同时报两条根因 / 有 token 无二进制自动只开 S2 /
   S2 失败则 S6/S7 不启动 / `FAILURE.json` 最早根因不被覆写 /
   无凭据时绝不说 verified）。
2. **"无 token + 无二进制"在真机 macOS 复现过**（`env -u HF_TOKEN`，干净工作目录）：
   3.1 s 结束，`ok=false`、`failed=["HF_AUTH"]`、
   `causes=["hf_no_token","binaries_missing"]`、`S2_BUILD: null`（**没有开始编译**）、
   `S9_REPORT: ok`（报告顶部是 `## ⚠ 本次运行失败` + "这份报告是**失败现场**，不是结果"）。
   `FAILURE.json` 的 `next_step` 直接写"`Add-ons → Secrets` 新建 `HF_TOKEN`……attach 它，再 `Run all`"。
3. **报告空段不再自相矛盾**：`RUN_EVAL=True` 自动补编时第 2 节说"自动只编 S2"，
   二进制已就绪时说"不需要编译"，被凭据门拦下时说"在编译之前就停了"；
   S6/S7 段落区分"没运行"与"**失败**"。

**v1.1.2 没验证的（如实标注）**：自动 S2 在真实 Kaggle 上的 CUDA 编译；
带 token 的 7.3 GB 真实下载与真实上传；S6/S7 的端到端结果 —— 一条都还没有。

### 已在本地验证的部分（v1.1.3，2026-10-23，非 Kaggle）

1. **155 项 CPU 回归全绿**（不联网、约 50 秒）：149 项旧 + 6 项 v1.1.3
   （tree URL 不许 expand+大 limit / 错误正文留证并脱敏限长 / 400 不归咎 token /
   状态码 → cause 一一对应 / 真机 400 现场复刻 / S0 提示不再自相矛盾）。
2. **tree API 参数 A/B（真 HTTP，只读，匿名）**：`expand=true&limit=1000` → **400**
   `Invalid limit for index tree pagination`；`recursive=1&limit=1000` → **200**，
   26 个文件、11 个 LFS **全部带 `lfs.oid`** —— 内容 sha256 校验没有降级。
3. **凭据三态真机跑过**：无 token → `no_token`（不发请求）；假 token → `unauthorized`
   + `http_status=401` + 服务器原话 `Invalid username or password.`；
   另测出私仓/不存在 repo 都是 **401**，只有坏 revision 是 404。
4. **端到端**：`run(run_eval=True)` + 假 token → 3.0 s 结束、
   `causes=["hf_unauthorized","binaries_missing"]`、`S2_BUILD=None`（没开始编译）。

**v1.1.3 没验证的**：带**真** token 的私仓 tree 读取（本机无 token）；
S6/S7 端到端；自动 S2 的真 CUDA 编译；7.3 GB 下载与上传。

---

## 九、动态条件（不假定）

Kaggle 的每日 GPU 配额、单会话时长上限、并发会话数、存储配额都是**会变的外部条件**，本文件不写具体数字，也不假设它们够用。每次开跑前在网页上确认当前配额；如果中途被截断，S2 的 build 目录会随会话丢失，需要靠 `Save Version` 保住。

---

## 十、下一步

**如果你要的是"把 S6/S7 跑出来"（v1.1.3 的默认下一步）**：上传 `trlab_kaggle.ipynb` →
把 `RUN_EVAL = False` 改成 `RUN_EVAL = True` → 配好 HF token（6.2.1 三条路任选一条，
**不要写进 notebook**，Kaggle 上要 **attach** 到本 notebook）→ `Settings`：
`Accelerator = GPU T4 ×2`、`Internet = On` → `Run all` → 对着 6.2.5 的清单核一遍 → 停下。

新 Session 没有 `llama-perplexity` / `llama-cli` 时，它会**自己只编 S2**
（pinned 源码 36.6 MiB，约 28 分钟），你**不用**再开第二个开关；二进制已存在就直接跳过。
不下载 BF16、不重做 S3、不重新量化；**默认用 GPU**，一次只装一个模型，每阶段结果自动出站。
token 没 attach 的话，它在**编译之前**就停下并写清两条根因，不会让你白等 28 分钟。

**如果你要的是"把量化再做一遍"**：把 `RUN_S4 = False` 改成 `RUN_S4 = True` → `Settings → Internet = On` → `Run all` → 核对 5.4 与 6.6 两份清单 → 停下。

**唯一可能需要的一次性操作**：如果 Kaggle 没有 working 之外的暂存盘，按 6.1 用 `trlab_fetch_bf16.ipynb` 把 BF16 挂成只读 Input（6 次点击，之后每个 Session 零点击）。有暂存盘就全自动，这一步也会被省掉。

**只想做到 imatrix 为止**就改 `RUN_S3 = True`（BF16 的一次性操作可以整个跳过）。

**从零开始的全新 session 也走同一条路**：notebook 会自己发现没有二进制、自己下源码、自己编（约 28 分钟）、自己下 Q8_0、自己算 imatrix（已有可信 imatrix 就跳过）、自己按优先级取 BF16。中间任何一步失败都会停在那一格并给人话，不会带着残缺状态往下跑。

**执行包不在 notebook 里**：第二个代码格按钉死的 commit 从公开仓库 sparse fetch `research/kaggle/trlab/`（约 200 KB）。同一 Session 重复 `Run All` 会命中 stamp 直接复用。**仓库必须保持 public**；否则用 `python3 research/kaggle/build_notebook.py --embed` 生成内嵌执行包的版本。

**不要反复点 `Run All` 去"推进"。** 默认开关下 S4–S8 是关的，`Run All` 不会真的跑量化，但它会一路执行到 S9 并重新生成报告 —— 报告里那些"未运行"的段落很容易被误读成"全部跑完了"。逐段执行，每段对着清单看一眼。S4 成功后建议 **Save Version**，这样两个 GGUF 与记录都进版本，下次开会话不用重量化。
