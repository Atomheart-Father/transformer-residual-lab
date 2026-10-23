# Kaggle 低位宽量化验证台

`trlab_kaggle.ipynb` —— 一个**可直接上传到 Kaggle** 的独立 Notebook，用来在同一份 `Qwen/Qwen3-8B` 上产出 `IQ2_XS` 与 `IQ4_XS`，并比较**真实文件字节**、**配对 NLL**、**有限配对质量**。

当前版本 **v1.1.1**（可丢弃 GPU worker 策略）。**v1.1.1 的一句话**：模型与 provenance 已在 HF 持久保存，所以 **Kaggle 只是一台免费、可随时关机的算力机器** —— 首跑失败就直接关掉 Session，本地修代码 → commit → 新 Session 重跑，输入不用重下。据此改了三处行为：**① 评测默认用 GPU**（Kaggle 上开的就是 GPU Session，不该让 8B 在 4 核 CPU 上磨）；**② 一次只装一个模型**，跑完释放再装下一个；**③ 每个阶段跑完就把小结果出站到 HF 并远端复核**（白名单 5 个文件，不传模型/日志/secrets），失败则写 `FAILURE.json` 并明确列出"哪些结果尚未持久化"。

**v1.1.0 的一句话**：S4/S5 的真实量化产物已经备份在 Hugging Face 私有仓库 `Atomheart-Father/trlab-qwen3-8b-quant-pilot`（commit `097e692`），所以 S6/S7 **不再需要重新下载 15 GiB BF16、不再需要重做 S3、不再需要重新量化**——它们直接从 HF 上那三个 GGUF 取输入。分支原则：**模型与 sidecar 存 HF，代码与小型 JSON/report 存 GitHub，云算力只是可替换的执行端**。同时修掉一个 P0 回归（S3 的 `_write_stamp() got multiple values for keyword argument 'bytes'`，见第九之九）。**S6/S7 尚未在真实 Kaggle 上跑过**，本文件所有评测相关段落都按"未运行"表述。

v1.0 是一次**执行方式重构**（Issue #5）：notebook 本体从 253 KB / 28 cells 的巨物缩成 **7.3 KB / 5 cells** 的薄入口，全部逻辑搬进 `trlab/` 包的 16 个 `.py` 模块；执行包不再靠人工搬运，而是 notebook 的引导格按**钉死的 commit** 从公开仓库 sparse fetch `research/kaggle/trlab/`（约 200 KB、1–3 秒，`git rev-parse HEAD` 强校验）后再 import。用户操作收敛为**一个开关 + Run All**，取代原来的六步人工流程（取 BF16 notebook → Run All → Save Version → Add Input → 配路径 → 主 notebook）。v1.0.1 只加一个**运行前预检** `S4_PREFLIGHT`：`RUN_S4=True` 会在任何编译、任何下载之前用几秒钟核查 working 与工作区外暂存盘的挂载/容量/写权限、BF16 有没有落点、imatrix 能不能复用，不过就地 fail-closed 并只给一步要做的；同时把 `S4_QUANT` 的墙钟预算从 1800 s 提到 5400 s 并加单格式帽，降低 CPU 量化被误杀的风险。版本史：v0.1 只做环境检查与 smoke；v0.2–v0.2.2 修掉三轮逐 cell 审查的问题（S4 二次量化门槛、`--allow-requantize` 参数位置、S2 工具链预检）；v0.3.x 把 S2 收敛成开关 `RUN_S2` 并记录真实 Kaggle T4 通过；v0.4.0 把 S3 收敛成开关 `RUN_S3`；v0.5.0 把 S4 收敛成开关 `RUN_S4`（从原始 BF16 量化 IQ2_XS/IQ4_XS，禁止 Q8_0 二次量化，imatrix 命中即整段跳过 S3，量化 base 四级自取）。所有修复都有 CPU 可跑的回归测试（见第九、十节）。

**S2、S3 都已在真实 Kaggle T4 上通过**（S2 见第九之六；S3 于 2026-10-22 跑出 10,701,952 B / 1008 个张量的 imatrix.gguf，sha256 `34062ab6…`，耗时 57.95 s，产物已备份并 pin 进 `IMATRIX_PINNED`）。

**v1.0 已本地验证 / 尚未验证的边界**：短 notebook 生成器、钉 commit sparse fetch 引导（已在干净目录真实冷启动跑通：GitHub → import → S0/S1/S9 → 产物落盘，全程约 3 秒）、`--embed` 内嵌回退与 sha256 篡改 fail-closed、118 项 CPU 测试（不联网）、`S4_PREFLIGHT` 本机真跑（0.11 s，判定与摘要都正常）。**尚未在真实 Kaggle 上跑过**：S4/S5 一键链路本身，预检在 Kaggle 上的判定（`/kaggle/tmp` 到底存不存在、是不是独立文件系统、有多少空闲），以及 BF16 的 15.26 GiB 下载、`llama-quantize` 的 CPU 耗时与峰值 RAM。BF16 与两个产物（≈ 6.7 GiB）放不进 20 GB 的 `/kaggle/working`（15.26 + 2.36 + 4.34 = 21.96 GiB），所以第一次跑 `RUN_S4=True` 前需要把 BF16 挂成只读 Input —— 最少点击方案见 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md) 第 6 节；若 Kaggle 提供了 working 之外的暂存盘，notebook 会自己探测并直接下到那里。

S2 → S3 → S4 的操作清单在 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md)：以**一页一步网页操作**开头。本文件讲"为什么"，那份讲"怎么做"。

---

## 一、这个 Notebook 不做什么

写在前面的原因是：这些不是"还没做"，是**设计上就不做**。

| 不做 | 机制 |
|---|---|
| 自动下载大模型 | 默认 `RUN_S2=False` / `RUN_S3=False` / `RUN_S4=False`，此时**一个字节都不下载**。`RUN_S3=True` 才允许下载 pinned 的 Q8_0（8,709,519,168 B）+ wikitext 校准语料（6,357,543 B）；`RUN_S4=True` 额外允许下载 pinned 的 BF16（16,388,044,384 B）；`RUN_S2=True` 只下载 36.6 MiB 的 llama.cpp 源码 tarball。四样东西全部 sha256/etag 预登记并强制校验 |
| 自动 `pip install` | 只用 Kaggle 镜像自带的包；llama.cpp 从源码编译；GGUF 解析器是纯 Python，无第三方依赖。parquet 解析优先 pyarrow、回落 pandas，都没有就报可操作的错（**不**偷偷装） |
| 自动租 GPU / 跑实验 | 阶段开关默认全关；用户手动改 `RUN_S3` 后在网页上执行 |
| 上传私有项目文件 | Notebook 不引用 `transformer-residual-lab` 的任何本地数据或脚本 |
| 在单张 T4 上硬加载 8B BF16 | S8 先算显存，放不下就返回 `status=skipped` 并写明所需/可用 GB，不硬撑到 OOM |
| 要求用户手动恢复环境 | 二进制探活三档来源，每个候选真跑一次 `--version`；探不到就自动走 S2 重编。**用户不需要手动先跑 S2，也不需要手动配路径或传模型** |

---

## 二、文件

```
research/kaggle/
├── trlab_kaggle.ipynb        # 主 notebook（5 cells：2 code + 3 markdown）
├── trlab/                    # 执行包：19 个 .py 模块（notebook 只 import 它，不含逻辑）
├── build_notebook.py         # 从 trlab/ 生成短 notebook（--embed 可内嵌执行包）
├── trlab_fetch_bf16.ipynb   # 例外降级：一次性取数 notebook，把 pinned BF16 下成 Kaggle Output（v0.5）
├── KAGGLE_S2_S3_RUNBOOK.md   # S2 编译 → S3 imatrix → S4 量化的操作清单（网页 GUI 视角）
├── KAGGLE_S3_DESIGN.md       # S3 一键化的设计 rationale 与五个决策点的落点（已实现）
├── HF_MODEL_CARD.md          # HF 私有仓库的 README 草稿（含 "S6/S7 not yet evaluated"）
├── test_trlab.py             # CPU-only 回归测试（139 项），不联网、不需要 GPU / llama.cpp
└── README.md                 # 本文件
```

v1.1.0 新增的三个模块：`trlab/hf.py`（HF 凭据链 / tree API / 302 签名地址 /
断点续传 / opt-in 上传与远端复核）、`trlab/artifacts.py`（评测输入解析、
sidecar fairness、成本估算、`eval_preflight`）、`trlab/__main__.py`
（`python -m trlab` 入口）。

`test_trlab.py` **不是** notebook 的副本：它直接 **import `trlab/` 包**（不解析 notebook 字符串），在临时工作目录里补齐默认三阶段后对函数做断言。改了 `trlab/` 就跑得出来。notebook 本身另有结构测试钉住"必须薄"：≤5 cells、<20 KB、钉的 commit 必须在仓库历史里、默认不含内嵌大包。

---

## 三、上手：一个开关 + Run All

1. Kaggle → `File` → `Import Notebook` → 上传 `trlab_kaggle.ipynb`（7.3 KB）
2. 右侧 `Settings`：`Accelerator` = **GPU T4 ×2**（Notebook 只用第 0 张，第二张不会自动用上）；`Internet` = **Off**（默认不需要联网）
3. 直接 `Run All`。第二个代码格会自己从公开仓库取执行包（约 200 KB、1–3 秒，校验 commit 后 import），然后只跑 `S0_ENV` / `S1_SMOKE` / `S9_REPORT`，约 2–4 分钟，**不碰任何大文件**。

**要把 imatrix 一起做出来**（只改一个开关，**不需要手动先跑 S2，不需要准备任何 Dataset**）：

1. 找到 `RUN_S3 = False` 这一行，改成 `RUN_S3 = True`
2. 右侧 `Settings` → `Internet` = **On**
3. `Run` → `Run all`

notebook 会自己按顺序做：探活四个 llama.cpp 二进制（在就复用，丢了就自动下载 pinned 源码重编，约 28 分钟）→ 磁盘预检 → 下载 pinned 的 Q8_0（8,709,919,168 B）→ 下载 wikitext train parquet 并转纯文本 → 按显存余量选 `-ngl`（OOM 会逐级降）→ 跑 `llama-imatrix` → 验证产物 → 写 provenance → 才删掉 Q8_0 回收磁盘。**S4–S8 与 SX 一律保持关闭。**

**要把 IQ2_XS / IQ4_XS 一起做出来**（只改一个开关）：

1. 找到 `RUN_S4 = False` 这一行，改成 `RUN_S4 = True`
2. 右侧 `Settings` → `Internet` = **On**
3. `Run` → `Run all`

S2/S3/S4/S5 会一条龙跑完。**唯一的例外**：BF16 有 15.26 GiB，与两个产物（≈ 6.7 GiB）放不进 20 GB 的 `/kaggle/working`（15.26 + 2.36 + 4.34 = 21.96 GiB），所以第一次跑之前要先用附带的 `trlab_fetch_bf16.ipynb` 把 BF16 变成只读 Input（6 次点击，之后每个 Session 都零点击，见 Runbook 第 6 节）。如果 Kaggle 提供了 working 之外的暂存盘，notebook 会自己探测到并直接下到那里，这一次性操作也可以省掉。

三个开关默认都是 `False`，且 `_apply_run_switches()` 在三个开关都 `False` 时**一个字段都不改** —— 导入 notebook 或误点 `Run All` 都不会悄悄联网，更不会下 8 GB / 15 GB。想下载必须显式改这一行，由 `test_switches_off_change_nothing` / `test_switch_s4_expands_plan` 等测试钉住。

**只想编 llama.cpp、先不碰模型**：把 `RUN_S2 = False` 改成 `RUN_S2 = True`，`Internet = On`，`Run all`。约 15–30 分钟（含编译，**估算值**）。

**只想评测、不再量化**（v1.1.1，**最少操作**）：把开关格里的 `RUN_EVAL = False` 改成 `RUN_EVAL = True`，`Settings → Accelerator = GPU T4 ×2`、`Internet = On`，把 HF token 放进 `Add-ons → Secrets`（名字 `HF_TOKEN`，勾选 attach），`Run all`。它会：

1. 从 HF 私有仓库 `Atomheart-Father/trlab-qwen3-8b-quant-pilot` 取 `IQ2_XS` / `IQ4_XS` / `imatrix.gguf` 三个已存产物（**不下载 15 GiB BF16、不重做 S3、不重新量化、不编译**）；
2. 取 pinned 的 wikitext-2-raw **test** parquet（732,610 B）当评测语料（与 S3 用的 **train** 校准集严格分离，S6 会做内容级重叠断言）；
3. 跑 S6 配对 NLL + S7 质量检查 + S9 报告。**默认用 GPU**（`EVAL_ALLOW_GPU=True`，有显存就 `-ngl 99`；装不下才退 CPU，并把原因写进 `device_by_fmt`）；**一次只装一个模型**，每个格式跑完就释放；
4. **每个阶段跑完就把小结果出站**（`results/v1.1.1/<run_id>/` 下的 `S6_NLL.json` / `S7_QUALITY.json` / `EVAL_SUMMARY.json` / `report.md` / `results.json`），并做远端复核；同时写一份几百字节的 `EVAL_SUMMARY.json` 供直接贴到聊天里。

**`RUN_EVAL=True` 自带小结果回传**（只这 5 个白名单文件，不传模型、不传日志、不传 secrets）。想彻底不外发：`--set HF_UPLOAD_AFTER_RUN=false`，或者 CLI 里不加 `--upload` 并把它关掉 —— 但那样结果会随 Session 一起消失。

要强制纯 CPU（例如只想验证链路）：`trlab.run(run_eval=True, EVAL_ALLOW_GPU=False)`，此时所有 llama 命令一律 `-ngl 0`。

私有仓库需要凭据。取 token 的三条路（**都不需要把 token 写进 notebook / git / issue**）：

| 方式 | 怎么做 |
|---|---|
| 环境变量 | 在本地 shell 里 `export HF_TOKEN=hf_xxx` 再跑 `python -m trlab --run-eval` |
| Kaggle secret | `Add-ons → Secrets` 里加一个名为 `HF_TOKEN` 的 secret，notebook 会通过 `kaggle_secrets` 读（**推荐**，只勾选 attach 到这一个 notebook） |
| HF 官方缓存 | 本机 `huggingface-cli login` 之后，`~/.cache/huggingface/token` 会被自动复用 |

三者都找不到时，`--check-hf` 与 S6 的预检会明确说"没有 token"并给出这三条路，**不会**打印 token、**不会**擅自把仓库改 Public。

**失败了怎么办**：阶段会写 `out/FAILURE.json`（阶段名 / 报错 / traceback 尾 / **一条**下一步），并打印"你可以直接关掉 Session"。**不需要** Save Version、不需要抢救工作区 —— HF 上的输入是长期源，GitHub 上是代码，重跑只花一次下载时间。

**S8（BF16 锚点）与 SX（第三方量化探索）** 仍然默认关闭，打开方式是把执行格最后一行改成 `trlab.run(run_eval=True, S8_BF16_ANCHOR=True)` —— 阶段名直接当关键字参数传（见第十节 `test_overrides_route_stages_vs_cfg`）。S8 需要高精度 GGUF 当输入、SX 需要第三方 GGUF，都不在 HF 那个仓库里。

**执行包从哪来**：引导格按钉死的 commit 从公开仓库 sparse fetch `research/kaggle/trlab/`，只取这一个子目录（约 200 KB）。同一 Session 重复 `Run All` 会直接复用盘上的包（stamp 命中），不重新下载。仓库需要保持 public；`--embed` 变体（`python3 build_notebook.py --embed`）可把同一份包 base64 内嵌进 notebook，仓库不可匿名读时自动回退，sha256 校验不过就 fail closed。

---

## 四、阶段表

| 阶段 | 默认 | 前置依赖 | 预算 | 产出 | 能回答什么 |
|---|---|---|---|---|---|
| `S0_ENV` | **开** | — | 120 s | GPU / CPU / RAM / 磁盘 / 工具链 / input 清单 | 这台机器能干什么 |
| `S1_SMOKE` | **开** | — | 240 s | 合成 GGUF 往返、fp16 vs bf16 TFLOP/s、拷贝带宽、磁盘写速 | 解析器和预算算术对不对 |
| `S2_BUILD` | 关（`RUN_S2=True` 时自动开） | **不需要任何输入**：源码由 S2 自己下载 | 2400 s | `llama-imatrix` / `llama-quantize` / `llama-perplexity` / `llama-cli` | 工具链能否就位 |
| `S3_IMATRIX` | 关（`RUN_S3`/`RUN_S4=True` 时自动开） | S2 + imatrix base + 校准语料，**全部由 S3 自己取**；已有可信 imatrix 时整段跳过 | 3600 s | `imatrix.gguf` + sha256 + provenance sidecar | 校准信息是否对等 |
| `S4_QUANT` | 关（`RUN_S4=True` 时自动开） | S2 + S3 + 量化 base（**BF16，由 S4 自己取**） | 1800 s | IQ2_XS + IQ4_XS 两个 GGUF + provenance + 逐文件 bpw | 能否编码、字节多少 |
| `S5_BYTES` | 关（`RUN_S4=True` 时自动开） | S4（或任意 GGUF） | 300 s | 逐张量类型字节、按 attn/ffn/embed 分类 | 哪些张量没被低位量化 |
| `S6_NLL` | 关（`RUN_EVAL=True` 时自动开） | 二进制 + **HF 上已存的 IQ2_XS/IQ4_XS**（v1.1 起不再要求本轮 S2/S5）+ 评测语料（wikitext **test**，自己取） | 5400 s | 分组配对 NLL + bootstrap CI | 2.31 bpw 差多少 nats |
| `S7_QUALITY` | 关（`RUN_EVAL=True` 时自动开） | 同上（同 prompt / 同 seed / 同 n_predict） | 1800 s | 贪心续写一致性、repeat4 | 有没有整体崩塌 |
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

**S3 的开关与参数（v0.4.0：一个开关 `RUN_S3`，其余自动）**

| 键 | 默认 | 谁设 | 为什么 |
|---|---|---|---|
| `RUN_S3` | `False` | **用户** | 唯一需要手改的开关。`True` 时 `_apply_run_switches()` 一次填好 S2 编译参数 + 下面全部 S3 参数；`False` 时一个字段都不改 |
| `S3_AUTO_FETCH_MODEL` | `False` | `RUN_S3` | `True` 才允许联网下载 pinned 的 Q8_0（8,709,519,168 B，sha256 `0cfbf745…`，revision `a6adef13…`） |
| `S3_AUTO_FETCH_CALIB` | `False` | `RUN_S3` | `True` 才允许联网下载 wikitext train parquet（6,357,543 B）并转纯文本 |
| `IMATRIX_BASE_KIND` | `""` | `RUN_S3` | 自动填成 `q8_0`。手动填了 `IMATRIX_BASE` 就以手动为准，且**不下载也不删**那个文件 |
| `CALIB_SOURCE` | `""` | `RUN_S3` | 自动填成 `wikitext-2-raw/train`，写进 provenance 与 S9 报告 |
| `NGL_MARGIN_GB` | `1.0` | `RUN_S3` | 提到 `2.0`：`_fit_ngl()` 按「模型字节 + 余量 ≤ 可用显存」判定，T4 上 8.11 + 2.0 = 10.11 < 13.56 GiB → `-ngl 99` |
| `S3_DELETE_Q8_AFTER` | `False` | `RUN_S3` | `True` = imatrix 验证成功且记录写盘之后回收 Q8_0（磁盘账要求）。改成 `False` 会保留 Q8_0，working 少 8.11 GiB |
| `IMATRIX_CTX` / `IMATRIX_CHUNKS` | `512` / `32` | 用户 | 32 × 512 = 16,384 token，首轮只作 pilot |

**断点与复用**：Q8_0 与校准语料各有一个 `.stamp.json`（repo / revision / sha256 / bytes），命中就不重下；下载中断时 `.partial` + sidecar 保留断点。**Q8_0 被回收之后，重跑 S3 会重新下载 8.11 GiB** —— 这是"成功后才清理"的直接后果。

**一个容易踩的坑**：不要为了省编译时间去关 `LLAMA_BUILD_SERVER`。上游 `tools/CMakeLists.txt` 里 `add_subdirectory(cli)` 写在 `if (LLAMA_BUILD_SERVER)` 内部，关了就连 `llama-cli` 都没有（S7 要用）。

---

## 五之二、Kaggle Session 与编译产物复用（v0.4.0 新增，重要）

Session 掉线是常态，不是异常。v0.4.0 按"不假设旧状态"设计：

| 官方文档事实 | 对设计的影响 |
|---|---|
| 交互 session **空闲 20 分钟**即结束 | 断线随时可能发生 |
| `/kaggle/working` ≤ 20 GB **自动保存**，可在未来 Notebook 复用 | 编译产物有机会跨 session 复用 |
| **Save & Run All = 全新干净状态**（12 小时上限） | 不能依赖上一次的内存状态 |
| Quick Save 只快照、不重跑 | 不能假设它一定存下 build |
| Add Input 可以搜「任何 Notebook 的输出文件」 | 二进制也能变成只读 Input（可选优化） |

**"掉线后不 commit 是否一定恢复"官方没写** → 每次启动都重新探活，探不到就自动重编。

二进制的查找顺序（`_find_llama_bins()`，每个候选**真跑一次 `--version`**）：

| 顺序 | 来源 | `build_reuse` |
|---|---|---|
| 1 | `/kaggle/working/trlab/build-llama/bin/` | `existing_build` |
| 2 | `CFG["LLAMA_BIN"]` 指向的目录 | `user_bin` |
| 3 | 扫 `/kaggle/input`（限时 20 s / 20000 目录） | `attached_input` |
| 4 | 都没有 → 完整走 S2（下载源码 + 编译 ≈ 28 分钟） | `rebuilt` |

**"存在"不等于"能跑"**：动态库缺失、架构不匹配的二进制过不了 `--version` 这一关，会落到下一档。`test_s3_probe_rejects_unrunnable` 钉住这一点。

**挂不挂都跑得通，区别只是要不要等 28 分钟。** 想省这段时间，就在 S2 成功后 `Save Version`，之后在新 Notebook 里 `Add Input` 挂自己上次的输出 —— 但这是可选优化，不是必要步骤。


---

## 五、需要准备的 Kaggle Dataset

全部**只读挂载**到 `/kaggle/input/<slug>/`，不占 `/kaggle/working` 的 20 GB。

> **v0.4.0 起 S3 不需要这里任何一样东西** —— Q8_0 和校准语料都由 notebook 按 pinned revision 自己下载并校验。本节从 S4 起才需要。

| # | 内容 | 建议 | 大小 | 谁需要 |
|---|---|---|---|---|
| 1 | **base GGUF** | `Qwen3-8B` 的 f16 或 Q8_0 | f16 = 15.25 GiB；Q8_0 ≈ 8.1 GiB | S4 / S8（S3 会自动下自己的 Q8_0） |
| 2 | **校准语料** `calib.txt` | wikitext-2-raw 的 **train** split | ≥ 200 KB（S3 会警告过短） | S3（仅当镜像里没有 pyarrow/pandas 时的退路） |
| 3 | **评测语料** `eval.txt` | wikitext-2-raw 的 **test** split | 几十 KB 起 | S6 / S7 / S8 |
| 4 | **llama.cpp 源码** | **不需要**（`RUN_S2`/`RUN_S3=True` 时 notebook 自己下载） | 断网备选：把源码挂上来，设 `CFG["LLAMA_SRC"]` | ~36 MB | S2 |
| 5 | **llama.cpp 二进制**（可选优化） | 上次 S2 编好的 `build-llama/bin/` 四个文件 | 1.5–3.5 GiB | 让新 Session 零编译；不挂也能跑，只是多等 28 分钟 |
| 6 | （可选）第三方量化 GGUF | 别人的 IQ2_XS / IQ4_XS | 2–5 GiB | SX |

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

## 九之六、v0.3.2：S2 真实通过（2026-10-22）

用户回传 `S2_BUILD-3.json`，**这是实测，不是 mock**：

| 项 | 值 |
|---|---|
| 状态 | `meta.status = "ok"`，`elapsed_sec = 1672.29`（**27 分 52 秒**） |
| 源码 | `d454a204…`，`commit_source = "pinned_tarball"`，`commit_verified = true` |
| 工具链 | cmake 3.31.10、nvcc CUDA 12.8、host GCC 13 |
| cmake 开关 | `-DGGML_CUDA=ON` + `-DCMAKE_CUDA_ARCHITECTURES=75` + `-DGGML_CUDA_NO_VMM=ON` 全部生效 |
| 产物 | 四个二进制都在 `/kaggle/working/trlab/build-llama/bin/` |
| 磁盘 | S2 后剩余 `19.07 GiB` |

到此，v0.3.0/v0.3.1 引入的每一条链路都被真实环境验证过：GitHub 直连、tarball sha256 校验、安全解压、工具链兼容性、CUDA 单架构编译、`NO_VMM` 热修。**第二节"编译时间 15–30 分钟 / build 1.5–3.5 GiB / 峰值 RAM 8–16 GiB"的估算可以退役了**，实际是 27 分 52 秒。

顺带修了一个非阻断缺陷（C-3）：结果里 `version` 是空串。根因在上游 —— `common/build-info.h:13` 的签名是 `void llama_print_build_info(const char *, FILE * = stderr);`，`common/arg.cpp:1457` 的 `--version` 分支没传流，所以 build info 写 **stderr**，而代码只取了 stdout。`_probe_version()` 现在两个流都接，stdout 为空时用 stderr，并渲染进 S9 报告第 2 节。**注意**：这只证明二进制能启动并打印版本，不等于验证过任何模型推理。

**S3 的一键化实现（v0.4.0）**见 [`KAGGLE_S3_DESIGN.md`](./KAGGLE_S3_DESIGN.md)：磁盘账、Kaggle Session 事实、五个决策点的落点、二进制探活三档来源。**代码与测试已完成，真实 Kaggle 运行待用户执行。**

---

## 九之七、v0.4.0：S3 一键化 + Session 掉线恢复

用户在确认五个决策点之后追加了一条关键情况：**Kaggle Session 已掉线，S2 编译成功的环境可能丢失**。v0.4.0 就是为"全新 Session 点一次 Run All 也能跑通"而做的。

### 唯一开关

```python
RUN_S3 = True      # 就改这一个；S2 由它自动带上
```

`_apply_run_switches()` 在两个开关都 `False` 时返回 `{"run_s2": False, "run_s3": False}` 且**一个 CFG 字段都不改**；`RUN_S3=True` 时一次填好 S2 编译参数、`S3_AUTO_FETCH_MODEL` / `S3_AUTO_FETCH_CALIB`、`IMATRIX_BASE_KIND="q8_0"`、`CALIB_SOURCE`、`NGL_MARGIN_GB=2.0`、`S3_DELETE_Q8_AFTER=True`，并**只**开 `S2_BUILD` 与 `S3_IMATRIX`。S4–S8 与 SX 一律不动。

### 二进制：存在 ≠ 能跑

`_find_llama_bins()` 按 `existing_build` → `user_bin`（`CFG["LLAMA_BIN"]`）→ `attached_input`（扫 `/kaggle/input`，限时 20 s / 20000 目录）→ `rebuilt` 顺序找四个可执行文件，**每个候选都真跑一次 `--version`**。动态库缺失、架构不匹配的"存在但不可执行"文件会被这一关拦下，落到下一档。一个都没有才走完整 S2（下载 pinned 源码 + 编译 ≈ 28 分钟）。

没有源码的 session 里 `_resolve_commit_record()` 回落到 `S2_PINNED["commit"]` 并标 `declared` + `commit_verified=false`，日志明说"未校验" —— 复用二进制时**不伪造版本证明**。

### 下载：单副本 + 四重校验 + 断点续传

- 边写 `.partial` 边算 sha256 → `os.replace` 原子落地；**不用 `huggingface_hub`**，杜绝 `~/.cache/huggingface` 里的第二份完整副本；
- `.partial.json` sidecar 记 `bytes_done` 与已写内容前缀哈希，重跑从断点续传；前缀哈希不符就删掉重下（早期版本在这里有真 bug：垃圾被拼到新内容后面，报"下载得到 8231 B，预登记 8192 B"）；
- `Content-Length` / 文件 size / `x-linked-etag` / sha256 四重核对，任一不符拒绝落地；
- 下载前 `_disk_precheck()`，空间不够直接报缺口，不下一半再爆；
- Q8_0 与语料各有 `.stamp.json`（repo/revision/sha256/bytes），命中即复用不重下。

两个 pin 的 digest 是**双源互证**的：`/api/models/<repo>/tree/<rev>` 的 `lfs.oid` 与 resolve URL 响应头 `x-linked-etag` 互相独立且一致，均等于文件内容 sha256。

### `-ngl` 与 OOM 降级

`NGL_MARGIN_GB=2.0`（v0.3 是 1.0）：`_fit_ngl()` 按「模型字节 + 2 GiB ≤ 可用显存」判定，T4 上 8.11 + 2.0 = 10.11 < 13.56 GiB → `-ngl 99`。OOM 时 `_ngl_ladder()` 给 `[99, 18, 0]` 逐级降，**每次尝试的 `ngl / ok / rc / oom / 日志尾部`全量进 provenance**，不只写最终值。`_is_oom_tail()` 认 `out of memory` / `cudaErrorMemoryAllocation` / `failed to allocate` / `bad_alloc` / `ggml_gallocr_reserve_n` 等特征。

### Q8_0 回收顺序是硬编码的

```
生成 → 验证（体积 ≥ 100 KB 且 n_tensors > 0）→ 写 OUT/S3_IMATRIX.json
→ _reclaim_q8() → 再写一次含 q8_reclaimed / q8_reclaimed_bytes / q8_reclaim_reason
```

失败时 Q8_0、校准语料、下载断点**全部保留**（那些才是"修完重跑不重下"的可恢复状态），只有不可续跑的半成品 `imatrix.gguf.partial` 被删掉 —— 它留着只是白占 working 的 20 GB。

### 前置守卫

`STAGE_REQUIRES_OK`：`S3_IMATRIX` 要求 `S2_BUILD`、`S4_QUANT` 要求 `S2_BUILD` + `S3_IMATRIX`、`S6_NLL`/`S7_QUALITY` 要求 `S2_BUILD`。**只在前置阶段真的失败时拦**；前置根本没开（`False`）不拦 —— 否则手动模式会被自己的默认值卡死。

**v1.0.1 起 `S4_PREFLIGHT` 是 S2/S3/S4/S5 的共同前置**：预检失败就不编译、不下载、不量化。它同时进 `ALWAYS_RERUN` —— 磁盘状态每个 Session 都不一样，缓存一个 `ok` 等于没查。

### 测试

新增 24 项 mock 回归，`test_trlab.py` 总计 **68 项**。覆盖用户点名的五类场景：

| 场景 | 测试 |
|---|---|
| 全新 Session 无二进制 | `test_s3_fresh_session_no_binaries` |
| 已有二进制复用 | `test_s3_existing_build_reused` / `test_s3_attached_input_reused` / `test_s3_probe_rejects_unrunnable` |
| S2 失败则 S3 不执行 | `test_s3_s2_failure_blocks_s3` |
| 掉线后重启 | `test_s3_restart_after_disconnect` |
| 磁盘空间不足 | `test_s3_disk_precheck_insufficient` |

外加下载安全（checksum / etag / 单副本 / 续传 / 坏 partial / 网络失败 / 重复运行复用）、`-ngl` 阶梯与 OOM 降级、失败人话、Q8_0 回收三态、手动 base 不被碰、报告行渲染。

**这一轮抓到并修掉 3 个真 bug**：前缀哈希不符时没删 `.partial`（会把垃圾拼到新内容后面）、复用二进制时 `commit_source` 记成 `null`（看不出"我们认为它是哪个版本"）、半成品 imatrix 在失败后留在 working。

---

## 九之八、v0.5.0：S4 一键量化（BF16 → IQ2_XS / IQ4_XS）

### 唯一开关

`RUN_S4 = True` 一个开关打开 S2 + S3 + S4 + S5，S6–S8 与 SX 全部保持关闭。**不设 `BASE_GGUF`** —— base 从哪来由 S4 自己按优先级决定，忘配路径不会失败，也不会盲目往 20 GB 的 working 里塞 15 GiB。

### 磁盘算术是这条路线的第一约束

| 账 | 数字 |
|---|---|
| `/kaggle/working` 配额 | 20 GB（Kaggle 官方文档） |
| S2 之后实测剩余 | 19.07 GiB（v0.3.2 真实运行） |
| BF16 + IQ2_XS + IQ4_XS | 15.26 + 2.36 + 4.34 = **21.96 GiB** |
| 连 BF16 + IQ4_XS | 19.60 GiB，仍然超 |

所以 BF16 **必须**来自不占 working 配额的地方。base 取源四级优先级（`_resolve_s4_base()`）：

| # | 来源 | 点击 | 说明 |
|---|---|---|---|
| 1 | `CFG['BASE_GGUF']` 手动指定 | — | 存在就用，不碰网络 |
| 2 | `/kaggle/input` 已挂载的 pinned BF16 | **0**（一次性操作后） | 字节数比对；开 `HASH_GGUF` 时 sha256 强校验 |
| 3 | working 之外的暂存盘 | 0 | 运行时探测 `st_dev`；同设备不算额外空间 |
| 4 | working 自己 | 0 | 只在磁盘账真的过得去时才下 |
| — | 都走不通 | — | `GuardError` + `trlab_fetch_bf16.ipynb` 的 6 步一次性操作说明 |

第 2 档的一次性操作就是附带的 `trlab_fetch_bf16.ipynb`（3 个 cell：下载 pinned BF16 → sha256 校验 → Save Version 提示）。挂成 Input 之后**永久复用、不再下载**。

### imatrix：Q8_0 不在了也能核验同源性

S3 按设计会回收 8.11 GiB 的 Q8_0，所以 `_base_consistency()` 要求"读 imatrix base 原文件"这条路直接堵死（v0.4 的真实缺陷：S4 会以 `GuardError` 收场，只能重下 8 GiB）。v0.5 换成两级核验：

| 等级 | 条件 | 依据 |
|---|---|---|
| `file` | Q8_0 文件还在 | 逐张量比形状（architecture / 张量名 / dims / 参数量） |
| `provenance` | Q8_0 已回收 | ① imatrix sha256 与记录一致；② 记录的 base repo/revision 与 `S4_PINNED` 相同（同仓库同 revision ⇒ 同一次转换 ⇒ 同一份权重）；③ imatrix 张量名 ⊆ base 张量名且层数一致；④ 每个 imatrix 张量的 `in_sum` 元素数 = base 对应张量的输入通道数 |

第 ③④ 项的依据是**本地实测**的 imatrix GGUF 结构：没有 `general.architecture`，1008 个张量 = 36 层 × 7 种 × 4 个统计量（`.in_sum` / `.in_sum2` / `.counts` / `.stats`），`in_sum` 的元素数就是 n_in（`ffn_down.weight` → 12288 = ffn 宽度，其余 → 4096 = hidden）。dims 方向自动判定（要求所有张量同一方向 100% 匹配），防止 ggml `mul_mat` 约定与写入端相反。

### imatrix 复用：新 Session 也能恢复

- `IMATRIX_PINNED`（sha256 `34062ab6…`，10,701,952 B，1008 张量）是**真实 Kaggle 运行的产物**，直接 pin 进 notebook；
- `_verify_imatrix()` 三态：`pinned_verified` / `provenance_verified` / `unverified`（默认不放行）；
- S3 跑完会写 `models/imatrix.provenance.json`（schema 2）**挨着产物** —— OUT/ 可能随会话丢，models/ 里的产物和它的 provenance 是一起被保存/挂载的，这才是"新 Session 恢复"的关键；
- `S3_SKIP_IF_IMATRIX_OK=True`（`RUN_S4` 自动开）：已有可信 imatrix 就整段跳过，**不重下 8.11 GiB、不重算 58 秒**。

### 修掉的两个真 bug（都由 v0.4 真实运行记录现场确认）

| Bug | 现场 | 修法 |
|---|---|---|
| `base_origin` 误记 `manual` | `_ensure_q8()` 把下载路径写回 `CFG["IMATRIX_BASE"]`，重跑时"手动指定"分支抢先命中 → `_reclaim_q8()` 的 `auto_download` 条件不成立 → 8.11 GiB 永远留在 working | `_s3_base_origin()` 改成比对路径落点与 stamp，不看 CFG 是否被污染；缓存命中也不算"手动" |
| `_base_consistency()` 要求读原文件 | Q8_0 回收后 S4 直接 `GuardError` | 上面的 file / provenance 两级 |

### 禁止二次量化

`RUN_S4` 路线上 `ALLOW_REQUANTIZE` 永远保持 False，且 `plan.get("flags")` 非空时**直接 `GuardError`** —— 不给 `--allow-requantize` 活路。上游帮助原文：`WARNING: this can severely reduce quality compared to quantizing from 16bit or 32bit!`。

### 测试

新增 30 项 mock 回归，`test_trlab.py` 到此为止 **98 项**（v1.0 又加 10 项架构回归，v1.0.1 再加 10 项预检回归，现为 **118 项**）。除上表各分支外还覆盖：开关作用域（只开 S2/S3/S4/S5）、默认全关一个字段都不改、pin 值防手滑、暂存盘 `st_dev` 判定（同设备不算）、磁盘算术拒绝、遗留 Q8_0 回收三态、S3 跳过/重生成、S4 两道门（无 imatrix / 不可信）、S5 用 S4 实际取的 base、报告行渲染。

**这一轮还抓到 1 个真 bug**：`stage_s4_quant` 里 `imatrix_base_kind` 变量名笔误（`imat_base_kind` vs `imatrix_base_kind`）→ `NameError`，由 `test_t4_split_base_path` 抓出。


---

## 九之九、v1.1.0：HF 已存产物直接评测 + S3 回归修复

这一版由 Issue #5 的新评论驱动。前提先摆清楚：**S4/S5 的真实量化产物已经成功备份到
Hugging Face**（`Atomheart-Father/trlab-qwen3-8b-quant-pilot`，commit `097e692`，
**Private**），所以"输出没持久化"这个担忧不成立，也**不需要用户再为保存去折腾 Kaggle**。
分工随之定死：**HF 存模型与 sidecar，GitHub 存代码与小型 JSON/report，
云算力只是可替换的执行端**（不再依赖 Kaggle `Save Version`）。

### 1. 先修的 P0 回归：S3 重复 `bytes`

真实 Kaggle 上 `RUN_S3=True` 直接炸：

```
TypeError: _write_stamp() got multiple values for keyword argument 'bytes'
```

根因是 `stages_imatrix._parquet_to_txt()` 的返回 dict 里有个键叫 `bytes`
（转换后 txt 的大小），而调用方写的是

```python
_write_stamp(stamp, ..., bytes=pin["bytes"], **conv)   # conv 里也有 bytes
```

`**conv` 把 `bytes` 摊进来，和显式的 `bytes=` 撞车。修法两层：

- `_parquet_to_txt()` 的键改名 `converted_bytes`（语义也更准：它是 txt 的字节数，
  不是 parquet 的）；
- `_ensure_calib_txt()` 改成**显式 dict 合并 + 重名检查**（`clash = sorted(set(a) & set(b))`
  非空就 `GuardError`），不再用 `**` 摊字典。`meta` 里 parquet 的 `bytes` / `sha256`
  仍然原样保留。

回归测试是**真正跨函数**的，不是单测某个 helper：`test_calib_txt_full_flow_no_duplicate_bytes`
用假 pyarrow 走完整条 `_ensure_calib_txt`（下载桩 → 真转换 → 写 stamp → 第二次 cache hit），
`test_no_splat_into_write_stamp` 再用 AST 扫全包，禁止任何"调用里同时出现 `bytes=`
与 `**`"的写法。前者会复现原 bug，后者防止有人换个地方再写一遍。

### 2. HF 凭据：三条路，一条都不许写进仓库

`trlab/hf.py` 的 `hf_token()` 按顺序找：`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` /
`HUGGINGFACEHUB_API_TOKEN` 环境变量 → Kaggle secret（`kaggle_secrets.UserSecretsClient`）
→ HF 官方缓存（`$HF_HOME/token` 或 `~/.cache/huggingface/token`）。返回值带 `source`
（`env:HF_TOKEN` / `kaggle_secret` / `hf_cache_file` / `None`），日志只打 source，
**绝不打 token**；`redact()` 再兜一层。找不到 token 时给的三选一里明确写着
"不要把 token 写进 notebook / git / issue"，也**不会**建议改仓库可见性
（可见性是用户的决定，代码不碰）。

### 3. HF 私有仓库的两个反直觉点（都是实测出来的）

- **resolve URL 302 到带签名的 CDN 地址，那个地址不需要 `Authorization`。**
  所以下载要分两步：先用 token 请求 resolve、拿到 302 的 `Location`，
  再对签名地址**裸**下载。把 token 一起发过去反而可能 403。
- **`x-linked-etag` 是 LFS oid，也就是内容 sha256；裸 `etag` 不是。**
  HF 现在默认走 xet 存储，裸 `etag` 是分块 id，拿它当内容哈希会永远对不上。
  所以 `fetch_file` 只认 `x-linked-etag`，并和 tree API 的 `lfs.oid` 双向比对。

非 LFS 的小文件（比如 sidecar JSON）只有 git blob sha1，**不能冒充 sha256** ——
`verify_remote()` 对这类文件明确写 `content_sha256_verified=False`，
只比 size。

### 4. S6/S7 的输入改成"本地优先，其次 HF"

`artifacts.resolve_eval_models()` 的顺序是：本地已有 GGUF + sidecar → 复用（带 stamp 指纹）
→ 否则从 HF 下载（`_fetch_pinned_blob` 的断点续传 + 边写边算 sha256 + `os.replace` 原子落地）。
`probe_only=True` 时**只列 tree 拿 size/oid**，`source="hf_pending"`，
绝不为了"看一眼"去拉 7 GiB —— `eval_preflight(do_download=False)` 走的就是这条路。

`EVAL_VERIFY_HASH=True`（默认）会在评测前把本地 GGUF 与 HF tree 的 `lfs.oid` 对一遍
（4.5 GiB 约 2–4 s/个），对不上就 fail closed。

### 5. fairness 分两级，不许混着说

| 来源 | `verified` | `verified_level` | 能核到什么 |
|---|---|---|---|
| S4 在本轮运行时记录 | `True` | `s4_runtime` | 量化当场记的 base / imatrix / 参数 |
| 从 HF sidecar 重建 | `False` | `hf_sidecar_declared` | 只能核 sidecar 之间的一致性 + GGUF 对 `lfs.oid` |

报告第 6/7 节会把 `fairness_level` 和 `provenance.inputs` 一起渲染出来。
**从 HF 来的产物，任何"同源已核验"的说法都必须带 `hf_sidecar_declared` 这个前缀。**

### 6. 默认纯 CPU，成本先估算再开跑

`EVAL_ALLOW_GPU=False` 是默认值，`_eval_ngl()` 在这个前提下对所有格式返回 `0`
（`llama-perplexity` / `llama-cli` 一律 `-ngl 0`）。`eval_cost_estimate()` 用
**假设速率**（CPU 2.5 tok/s、GPU 30 tok/s，`assumed_rate_note` 里写明"不是实测"）
算墙钟，超过 `EVAL_CPU_MAX_WALL_SEC`（5400 s）就 fail closed，并给三个选项：
缩小 pilot / 你自己显式开 `EVAL_ALLOW_GPU=True` / 只跑 S7。

规模默认也改了：`NLL_CTX=512`、`NLL_CHUNKS_PER_GROUP=1`、`NLL_GROUPS=8`
→ 8×1×512 ≈ 4k token/模型、共 8k token，纯 CPU 估算 **0.94 h**。
旧的 8×4×2048 估算 **14.59 h**，S6 会被阶段预算杀掉。配对比较只要求两个模型
**同设置**，不需要把 ctx 拉到 2048 —— 但这也意味着结论的适用范围只有 512 ctx。

### 7. 上传是单独的 opt-in，默认一个字节都不外发

`HF_UPLOAD_AFTER_RUN=False` 时 `upload_after_run()` 直接返回 `{"skipped": True}`。
显式打开后只传白名单里的五个小文件（`report.md` / `results.json` /
`S6_NLL.json` / `S7_QUALITY.json` / `S9_REPORT.json`）到 `results/v{version}/`，
传完做远端复核（size 必比、LFS 再比 sha256），上传失败**不影响**评测结论。
GGUF 与 sidecar 走人工上传（体积大、且是"模型资产"而不是"每次运行的产物"）。

### 8. CLI / 入口

```bash
python -m trlab --run-eval        # 只评测（= notebook 里 RUN_EVAL=True）
python -m trlab --eval-only       # 同上，别名
python -m trlab --check-hf        # 只列 HF 上的文件与 token 状态，不下载
python -m trlab --preflight-only  # 只跑 S4 预检
python -m trlab --run-s4 --upload # 量化 + 显式 opt-in 上传小结果
```

`trlab.run()` 的签名是 `run(run_s2=None, run_s3=None, run_s4=None, run_eval=None,
preflight_only=False, upload=False, **overrides)`；`preflight_only=True` 时四个开关
全部归位为 `False`，只留 `S4_PREFLIGHT`。

### 9. 这一版**没有**验证的东西（不要在报告里写成已验证）

- **S6/S7 没有在真实 Kaggle 上跑过。** 139 项测试全是 CPU/mock，`test_s6_end_to_end_from_hf`
  用的是假 HF 服务器桩。
- **HF 的真实下载只验到 401 路径**：`python -m trlab --check-hf` 在无 token 时
  正确返回 `token_present: false` + 401 的可操作提示（真机 macOS 跑过）。
  带 token 下载 2.7 / 4.56 GiB 的过程**未实测**。
- **成本估算是假设速率，不是实测吞吐。** 0.94 h 这个数字不能当承诺。
- **`hf_sidecar_declared` 级 fairness 未在真实产物上走过一遍**（测试里是构造的 sidecar）。
- 上传路径只有 dry-run 与 mock 覆盖，**没有真的往 HF 写过东西**。

### 10. 测试

`test_trlab.py` 从 118 项加到 **139 项**（v1.0.2 的 2 项 S3 回归 + v1.1 的 19 项）。
新增 19 项覆盖：token 链与"绝不打印 token"、tree 分页解析、错误码可操作性、
复用与 etag 双源守卫、`verify_remote` 只声明真核过的、上传门槛与 opt-in、
缺 `huggingface_hub` 时拒绝 `pip install -U`（会撞 tokenizers 约束）、
本地优先、`probe_only` 绝不下载、sidecar fairness 只能到 `declared`、
CPU 不可行时 fail closed、预检在缺二进制时说唯一一步、
`RUN_EVAL` 开关作用域、`run_ppl` 默认 `-ngl 0`、S6 从 HF 端到端、
评测语料是 test split（不是 train）、`_ensure_eval_txt` 全流程。

当前结果：**139/139 通过，约 45 秒**（v1.1.1 之后是 **144 项**，见第九之十）。


---

## 九之十、v1.1.1：可丢弃 GPU worker

用户 2026-10-23 明确了一条策略，它推翻了我上一版的一个决定，所以单独记一节：
**模型、imatrix 与 provenance 已经在 HF 持久保存，所以 Kaggle 只是一台免费、
可随时关机的算力机器。** 首跑失败就直接关 Session → 本地修代码 → commit →
新 Session 重跑。**不需要** Save Version、不需要抢救工作区、不需要 Add Input。

### 1. 评测默认用 GPU（推翻 v1.1.0 的"默认纯 CPU"）

v1.1.0 我把 `EVAL_ALLOW_GPU` 默认设成 `False`，理由是"GPU 消耗额度、不擅自用"。
用户指出这是浪费：**你开的就是 GPU Session，让 8B 在 4 核 CPU 上磨是错的**。
现在 `EVAL_ALLOW_GPU` 默认 `True`，语义变成"**允许用，且装得下就用**"：

| 情况 | `-ngl` | 记录 |
|---|---|---|
| 有 CUDA 显存 + 权重+余量装得下 | `99`（整模型卸载） | `device=gpu` |
| 有显存但装不下 | `0` | `device=cpu` + **`why` 写明"装不下"** |
| 没有 CUDA 显存 | `0` | `device=cpu` + `why` 写明"没有检测到" |
| `EVAL_ALLOW_GPU=False` | `0` | `device=cpu` + `why` 写明"显式强制" |

关键点是**不允许静默降级**：`eval_device()` 一定要给出 `why`，并且 S6/S7 把它
逐格式写进 `device_by_fmt`。否则"以为在 GPU 上跑了 3 小时"这种事故迟早发生。
成本估算也跟着设备走（GPU 30 tok/s / CPU 2.5 tok/s，都是**假设值**），
且 `EVAL_CPU_MAX_WALL_SEC` 只对 CPU 生效（`budget_applies`）—— 拿 CPU 的上限
去拦 GPU 会误杀。

### 2. 一次只装一个模型

`IQ2_XS` 和 `IQ4_XS` **绝不**同时压在显存/内存里。S6/S7 都是"装 A → 跑完 A →
释放 A → 再装 B"，日志里能看到 `装入 IQ2_XS（gpu -ngl 99…）；跑完释放再装下一个`。
`llama-perplexity` / `llama-cli` 是独立进程，进程退出即释放，所以这条是结构性
保证而不是靠自觉；结果里 `settings.one_model_at_a_time=True` 是它的标记。

### 3. 每完成一个阶段就出站 + 远端复核

`RUN_EVAL=True` 现在会打开 `HF_UPLOAD_AFTER_RUN`（**只传白名单 5 个文件**）：

| 阶段 | 出站文件 |
|---|---|
| `S6_NLL` | `S6_NLL.json`、`EVAL_SUMMARY.json` |
| `S7_QUALITY` | `S7_QUALITY.json` |
| `S9_REPORT` | `report.md`、`results.json`、`S9_REPORT.json` |

落在 `results/v1.1.1/<run_id>/`（`run_id` 是 UTC 时间戳，这样多轮不会互相覆盖）。
传完立刻 `verify_remote()`：size 必比、LFS 再比 `lfs.oid`。**上传失败不改阶段结论**，
但会把文件名写进 `not_persisted` 并打印"这些结果**尚未持久化**" —— 不许虚报。
不传的东西：GGUF（体积大、是模型资产不是运行产物）、`logs/*.log`（可能带路径）、
任何 token。

### 4. 失败现场：`FAILURE.json`

任一阶段失败会写 `out/FAILURE.json`：

```json
{"stage": "S6_NLL", "run_id": "...", "error": "GuardError: ...401",
 "traceback_tail": "...", "next_step": "配 HF token：...",
 "upload": {...}, "note": "Kaggle 只是可丢弃的算力机：...修代码 → commit → 新 Session 重跑"}
```

`next_step` 只给**一条**（token / 缺二进制 / 超预算 / 哈希不符 / 语料重叠 / 看 traceback）。
同时 `out/S6_NLL.json` 里仍有完整 traceback。

### 5. 可粘贴摘要：`EVAL_SUMMARY.json`

几百字节，专为"贴到聊天里"设计：`nll` 两个格式的均值、`delta_nll_iq2_minus_iq4`
（mean / ci95 / n / char_weighted）、`verdict`、`fairness_level`、逐格式 `device`、
`throughput_tok_s`、S7 的 `identical_rate` / `degenerate`、`upload` 状态、三条 `limits`。
它由 S6/S7 的 JSON 生成，**不手抄数字**。

### 6. 没有新增的东西（克制）

按"简化而非叠加"的要求，这一版**没有**加：跨 Session 状态管理、Kaggle 版本保存
备援、云端运维自动化、页面步骤。notebook 仍是 5 cells，仍然只有 `RUN_EVAL` 一个开关。

### 7. 测试

139 → **144 项**，新增 5 项：`test_eval_device_never_silently_degrades`（三种降级
都要有 `why`）、`test_eval_preflight_reports_device`（默认 gpu、强制 cpu 两条路）、
`test_upload_whitelist_has_no_models_or_logs`（白名单不能有 GGUF/日志/token）、
`test_upload_stage_respects_optin_and_reports_not_persisted`（未授权 → 列出未持久化；
失败 → 不抛给上层）、`test_failure_json_has_stage_and_one_next_step`、
`test_eval_summary_is_small_and_pasteable`（<3 KB 且数字来自 JSON）。
另外把 v1.1 里"默认纯 CPU"的 4 项断言改成"默认 GPU、显式才 CPU"。

### 8. 仍未验证

**S6/S7 依然没有在真实 Kaggle 上跑过**（144 项全是 CPU/mock）；带 token 的真实
下载未实测；真实 sidecar 的 fairness 判定未走；往 HF 的真实上传未发生。
所以"量化对 NLL / 能力的影响"**目前依然零数据**。


---

## 十、怎么跑测试

```bash
python3 research/kaggle/test_trlab.py
```

不需要 GPU、不需要 torch、不需要 llama.cpp、不联网、不下载。v1.0 起它**直接 import `trlab/` 包**（不再解析 notebook 字符串），在临时工作目录里补齐默认三阶段，然后跑 139 项断言。当前结果：**139/139 通过，约 45 秒**（v1.0.1 是 118 项 / 约 31 秒；多出来的时间主要是新增的 19 项里那几条要在干净子进程里单独 import 模块、以及 bootstrap 用本地 git 仓真跑 fetch）。

v1.0 新增的 10 项架构回归（对应 Issue #5 的验收清单）：

| 测试 | 钉住的问题 |
|---|---|
| `test_import_has_no_side_effects` | import trlab 不建目录、不打印、不联网 |
| `test_each_module_imports_alone` | 无循环 import：15 个子模块各自单独 import |
| `test_switches_off_change_nothing` | 三开关全 False 时 CFG/STAGES 一个字段都不改 |
| `test_switch_s4_expands_plan` | `RUN_S4=True` 只开 S2–S5，S6–S8/SX 不开，`BASE_GGUF` 留空 |
| `test_run_returns_plan_and_stages` | 入口返回 plan + 各阶段状态 |
| `test_overrides_route_stages_vs_cfg` | 阶段名进 STAGES、CFG 键不进 STAGES（薄 notebook 打开 S6–S8 的唯一入口） |
| `test_generated_notebook_is_thin` | notebook ≤5 cells / <20 KB / 钉的 commit 在仓库历史里 / 默认无内嵌大包 |
| `test_bootstrap_fetch_verify_reuse` | 冷启动：sparse fetch + `rev-parse` 强校验 + 同 Session 复用 + 逐字节一致 |
| `test_bootstrap_embedded_fallback` | 仓库不可读时回退内嵌包（sha256 校验后解压） |
| `test_bootstrap_embedded_sha_guard` | 内嵌包 sha 标注被篡改 → fail closed，不静默解压 |

后三项用一个本地 git 仓当"公开仓库"（`build_notebook.REPO_URL` 指向本地路径），所以整套测试保持离线。真实 GitHub 冷启动（fetch 9ebf858 → import → S0/S1/S9 → 产物落盘，约 3 秒）已手工验证过，不放进自动测试（避免联网）。

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
| `test_s2_default_off_no_network` | **B-3**：默认三个开关都 `False` 时 `_apply_run_switches()` 返回 `{"run_s2": False, "run_s3": False, "run_s4": False}` 且一个字段都不改；无手动源码时 `_find_llama_src` 抛 `GuardError` 且消息含 `RUN_S2` 与 `Internet`，**不触发下载** |
| `test_s2_run_switch_only_turns_on_s2` | **B-3**：`RUN_S2=True` 只开 `S2_BUILD`，`S3`–`S8` 全保持 `False`；`LLAMA_COMMIT`/`CUDA_ARCH`/`BUILD_JOBS`/`BUILD_CUDA` 自动填好 |
| `test_s2_download_checksum_mismatch` | digest 与预登记值不符 → `GuardError`，下载的包被删掉，不解压 |
| `test_s2_extract_rejects_traversal` | 归档里有 `../../evil.txt` → `GuardError`，不"尽量解压" |
| `test_s2_repeat_run_uses_cache` | 重复运行三级缓存：离线复用 → 只解压 → 包被改坏才重下（下载次数被钉死） |
| `test_s2_dir_name_mismatch_fails` | digest 对但顶层目录名与 pin 不符 → `GuardError`（pin 信息本身有问题） |
| `test_s2_commit_from_pinned_tarball` | 自动路径记 `pinned_tarball` + `commit_verified=True`；有 git 且相符记 `git`；不符仍抛 |
| `test_s2_failure_message_actionable` | 六种编译失败症状各自给出可操作处置（host g++ / nvcc / 磁盘 / 内存 / cmake / 其他） |
| `test_s2_failure_classifier_cuda_driver` | **C-2**：Kaggle 真实日志尾部（含 nvcc 行 + `CUDA::cuda_driver ... not found`）必须判成 driver 分支，**不能**误判成"找不到 nvcc"；真正的 nvcc 缺失仍然判成 nvcc |
| `test_s2_fingerprint_binds_cuda_no_vmm` | **C-1**：`CUDA_NO_VMM` 进 S2 指纹，改开关必然重跑，不会误 RESUME 旧 build |
| `test_s2_probe_version_reads_stderr` | **C-3**：`--version` 输出走 stderr，只取 stdout 会得空串；两个流都空时给 rc 而不是空串 |
| `test_s2_network_failure_message` | 真的连不上时消息含"Internet = On"，且不留下 `.partial` |
| `test_s3_run_switch_scope` | **D-3**：`RUN_S3=True` 只开 S2+S3，两个 `S3_AUTO_FETCH_*` 自动置真，S4–S8 与 SX 全 `False` |
| `test_s3_pins_are_registered` | 两个 pin 的 repo/revision/sha256/bytes/etag_header 都在，且 sha256 是 64 位十六进制 |
| `test_s3_fresh_session_no_binaries` | **全新 Session**：没有任何二进制 → 走完整 S2 流程，`build_reuse="rebuilt"` |
| `test_s3_existing_build_reused` | `build-llama/bin` 有四个能跑的假二进制 → 复用，**零下载零编译**，`build_reuse="existing_build"` |
| `test_s3_attached_input_reused` | `/kaggle/input` 下扫到二进制 → 复用，`build_reuse="attached_input"` |
| `test_s3_probe_rejects_unrunnable` | 二进制文件存在但 `--version` 跑不通 → **不算复用**，落到下一档（存在 ≠ 能跑） |
| `test_s3_s2_failure_blocks_s3` | **S2 失败则 S3 不执行**：`STAGE_REQUIRES_OK` 拦下，不会拿残缺文件硬跑 |
| `test_s3_restart_after_disconnect` | **掉线后重启**：上次的 `.stamp.json` 命中 → Q8_0 与语料都不重下 |
| `test_s3_disk_precheck_insufficient` | **磁盘不足**：下载前就报缺口，一个字节都不下 |
| `test_s3_download_checksum_mismatch` | sha256 不符 → 拒绝落地，删 `.partial`，不解压不量化 |
| `test_s3_download_etag_mismatch_refuses` | `x-linked-etag` 不符同样拒绝（双源校验缺一不可） |
| `test_s3_download_no_duplicate_full_copy` | **单副本**：全程只有一个 `.partial` 在长大，`os.replace` 之后才出现正式文件；不用 `huggingface_hub` |
| `test_s3_download_resume_from_partial` | 已有 `.partial` + sidecar 相符 → 发 Range 从断点续传，不重下已传部分 |
| `test_s3_download_bad_partial_restarts` | sidecar 的前缀哈希与文件内容不符 → 删掉从头下（真 bug：曾把垃圾拼到新内容后面） |
| `test_s3_download_network_failure_keeps_partial` | 网络中断 → 保留 `.partial` + sidecar，提示重跑续传，**不假装成功** |
| `test_s3_repeat_run_uses_cache` | 重复 `Run All` → stamp 命中，Q8_0 与语料都不重下 |
| `test_s3_ngl_ladder` | 阶梯严格递减、以 0（纯 CPU）结尾；1 层模型"折半"仍是 1 层 |
| `test_s3_ngl_fallback_on_oom` | 第一次 OOM → 自动降 `-ngl` 重试，两次尝试连 `oom` 标记与日志尾部都进 provenance |
| `test_s3_imatrix_failure_explains` | 三种失败症状（模型打不开 / token 数不够 / 显存不足）各自给出人话 |
| `test_s3_q8_deleted_only_after_record` | **Q8_0 必须在结果记录写盘之后才删**（用 spy 记录"删除那一刻"记录文件是否已在盘上） |
| `test_s3_q8_kept_when_imatrix_fails` | imatrix 失败 → Q8_0 原样保留、半成品被清掉、不写成功记录 |
| `test_s3_q8_not_deleted_when_configured_off` | `S3_DELETE_Q8_AFTER=False` → 保留 Q8_0，并记录原因 |
| `test_s3_manual_base_not_touched` | 手动指定 `IMATRIX_BASE` 时，S3 既不下载也不删那个文件 |
| `test_s3_report_lines` | S9 第 4 节的摘要行覆盖回收/未回收/声明/OOM 记录各分支 |
| `test_paired_bootstrap_edges` | n=0 / n=1 返回 `ci95=None`；全正差值给不含 0 的 CI |
| `test_ggml_type_table` | 7 个关键 type 的块尺寸/bpw |
| `test_default_stages_complete` | **默认 S0/S1/S9 三个都必须 `ok`，S9 报告必须有效**；且 `results.json` 里 `S9_REPORT.status` 不能是 `null`（**P0-B**） |
| `test_s9_report_bf16_both_ways` | bf16 报错 / bf16 正常 / 无 torch 三条渲染分支 |
| `test_t4_mock_end_to_end` | 把 GPU 微基准换成 T4 结果，重跑 S1→S9，报告必须写出"bf16 不可用" |
| `test_a100_mock_end_to_end` | 反例：bf16 可用时不能还打印"不可用"，且要给出比值 |
| `test_s4_run_switch_scope` | **需求 6**：`RUN_S4=True` 只开 S2/S3/S4/S5，S6–S8 与 SX 全 `False`；不设 `BASE_GGUF`、`ALLOW_REQUANTIZE` 保持 `False`、`BASE_KIND=bf16` |
| `test_s4_pins_are_registered` | BF16 与 imatrix 的 pin 值（sha256/bytes/kind/url）防手滑；**量化 base 与 imatrix base 必须同仓库同 revision** |
| `test_s4_default_off_downloads_nothing` | `RUN_S4=False` 时 `S4_AUTO_FETCH_BASE` / `S4_RECLAIM_Q8_CACHE` 保持 `False` |
| `test_verify_imatrix_three_states` | **需求 3**：pin 一致 → `pinned_verified`；与 provenance 记录一致 → `provenance_verified`；都对不上 → `unverified`；缺文件/解析失败/空张量各自 `ok=False` |
| `test_imatrix_found_in_input_and_candidates` | `/kaggle/input` 里挂着的 imatrix 能被扫到；候选优先级 = `IMATRIX_GGUF` > `models/` > Input |
| `test_imatrix_provenance_records_sidecar_first` | provenance 记录按可信度排序：挨着产物的 sidecar 优先于 `OUT/S3_IMATRIX.json` |
| `test_consistency_provenance_level_ok` | **需求 4**：Q8_0 已回收时，结构 + sha256 + 来源三项全过 → `provenance` 级通过 |
| `test_consistency_structure_failures` | 张量名不存在 / 层数不同 / 输入通道两方向都对不上 → 拒；dims 方向反过来也认 |
| `test_consistency_provenance_failures` | 无记录 / sha256 不符 / 来源 repo 或 revision 不符 / 字节数不符 → 拒；缺 repo 时回落 `S3_PINNED` 仍通过 |
| `test_consistency_file_level_when_q8_present` | Q8_0 还在 → `file` 级逐张量比形状；imatrix GGUF 实测没有 architecture，不断言它有 |
| `test_imatrix_base_does_not_throw_when_reclaimed` | **需求 4**：Q8_0 不在时 `_imatrix_base()` 返回记录路径而不抛（v0.4 就在这里炸） |
| `test_resolve_s4_base_manual` | **需求 2**：`BASE_GGUF` 手动指定 → 零网络零预检 |
| `test_resolve_s4_base_attached_input` | Input 已挂 → 零下载；字节数不符跳过；`HASH_GGUF` 开时 sha256 不符拒绝 |
| `test_resolve_s4_base_refuses_without_auto_fetch` | 没挂 Input 又没开自动下载 → `GuardError` 指向 `RUN_S4`，不瞎试 |
| `test_resolve_s4_base_scratch_download` | 暂存盘路线：下载落暂存盘、预检按暂存盘算、不占 working 配额 |
| `test_resolve_s4_base_working_and_guard` | working 装得下才下；装不下 → `GuardError` 带 6 步一次性操作说明 |
| `test_scratch_dirs_same_device_excluded` | 只认 `st_dev` 与 working 不同的候选；同设备 + 余量不足都不算 |
| `test_s4_output_bytes_estimate` | 两种格式都估且落在 6–9 GiB；未知格式按 6.5 GiB 兜底 |
| `test_reclaim_stale_q8_cache` | imatrix 已验证 → 回收遗留 Q8_0（含 stamp 与 `.partial`）；不可信/找不到/开关关掉 → 不删 |
| `test_s3_base_origin_regression` | **需求 5**：下载路径被写回 CFG 后重跑仍判 `auto_download`（v0.4 真 bug 现场）；手动文件仍判 `manual` |
| `test_s3_rerun_reclaims_q8` | 端到端复现该 bug：CFG 被污染的重跑，Q8_0 也被回收，provenance sidecar 落盘 |
| `test_s3_skips_when_imatrix_verified` | **需求 3**：已有可信 imatrix → 整段跳过，`_ensure_q8` 一次都不调 |
| `test_s3_untrusted_imatrix_reruns` | imatrix 存在但不可信 → 不跳过，重新生成 |
| `test_s4_forbids_requantize_even_when_opted_in` | **需求 1**：`ALLOW_REQUANTIZE=True` 也被拦死，命令里不许出现 `--allow-requantize` |
| `test_s4_requires_imatrix` | 没有 imatrix 直接拒（IQ2_XS 会被上游 bail out），不做降级 |
| `test_s4_rejects_unverified_imatrix` | sha256 对不上 → 默认拒并告知 pin 值；显式放行后才继续 |
| `test_s4_base_for_s5` | **需求 7**：S5 优先分析 S4 实际取的 base，回落 `CFG['BASE_GGUF']`，再回落 `None` |
| `test_ram_precheck` | RAM 预检的算术（base + 余量）与告警分支 |
| `test_s4_helper_hint` | 一次性操作说明含文件名与 6 步，且不超过一屏 |
| `test_s4_report_lines_render` | S9 第 5 节：`provenance` 级必须显式声明"弱于文件级"，`file` 级不声明；两行产物表 |

`test_default_stages_complete` 不是装饰。v0.2 加 P1-G 时，`gguf_write_synth` 变成 strict，S1 里"构造一个含未知 type 的坏文件"那行自己就抛了；而 `run_stage` 会吞掉异常只记 `status=error`，于是默认首跑 S1 静默失败、S9 报告里写 `SMOKE_OK = False`。这条测试就是为这种情况写的。

---

## 十一、已知限制

1. **T4 没有原生 bf16**（CC 7.5）。S1 会实测 fp16 vs bf16 的 TFLOP/s 比值；比值明显小于 1 就说明 bf16 走了模拟路径。锚点因此只能用 **f16**，与论文口径有偏差，S8 会把这条写进 `dtype_deviation`。
2. **单张 T4 装不下 8B 高精度权重**：f16 = 15.25 GiB，可用显存约 15.0 GiB，且未计 KV 与 CUDA context。S8 默认 `status=skipped`，并给出 `required_gb` / `vram_usable_gb` 两个具体数字。
3. **隔离只做到内容级**：S6 用前 200 字符做行级重叠检测。如果校准集与评测集来自同一文档的不同段落，这个方法检测不出来 —— 必须靠 `CALIB_SOURCE` / `EVAL_SOURCE` 声明不同 split。
4. **`/kaggle/working` 约 20 GB，装不下 BF16 + 两个产物**。BF16 15.26 + IQ2_XS 2.36 + IQ4_XS 4.34 = 21.96 GiB > 19.07 GiB（S2 后实测剩余），连 BF16 + IQ4_XS（19.60 GiB）都超。所以 S4 的量化 base 必须来自只读 Input 或 working 之外的暂存盘；working 里只放两个产物（约 6.7 GiB，余量够）。指纹不符的旧产物会以 `.stale-*` 形式留一份（见第 12 条）。
5. **S7 样本量极小**（8 prompt × 32 token）。只能发现整体崩塌，不能排序细微质量；`repeat4` 是退化信号，不是质量分数。
6. **NLL 的 CI 跨 0 不等于等价**。`paired_bootstrap` 只给区间，判读在 S9 里写死为"未测出差异"。
7. **不能外推**。单张 T4、8B、单一语料的结果不适用于 27B，也不适用于 M4 Pro。**吞吐不可跨平台迁移，只有格式与质量可以。**
8. **`guard()` 不能阻止运行中 OOM**。它只在阶段入口检查，llama.cpp 的显存分配在外部进程里。
9. **未知 ggml type 的文件没有字节拆分**。这是刻意的：宁可不给数，也不给 8 bpw 的假数。遇到这种情况需要升级 `GGML_TYPES` 表（对照 `ggml-common.h` 的结构体大小）。
10. **量化 base 必须是 f16/bf16，且 `RUN_S4` 路线不给逃生门**。`llama-quantize` 对已量化张量默认拒绝（`requantizing from type ... is disabled`）。手动模式下 `ALLOW_REQUANTIZE=True` 只用于 smoke，产物标 `requantized_pilot_only`，不能当高保真结论；**`RUN_S4=True` 时即使显式设 `ALLOW_REQUANTIZE=True` 也会被 `GuardError` 拦死**（`test_s4_forbids_requantize_even_when_opted_in`）。
11. **imatrix 与量化 base 可以不同文件，但必须同源**（v0.5 起两级核验）。Q8_0 文件还在时逐张量比形状（architecture / 张量名 / dims / 参数量，`file` 级）；Q8_0 已回收时走 sha256 + 来源 repo/revision + 张量名/层数/输入通道三项（`provenance` 级）。两级都只保证"同一份权重"，**不能证明是同一个 checkpoint 的权重**（同一模型的两个不同 finetune 也能通过结构核对）。`IMATRIX_BASE_KIND` 是量化类型时，报告里会声明"imatrix 的激活是对高精度的近似"。
12. **旧产物会留在 working**。指纹不符的 GGUF 被改名成 `.stale-<fp8>`，每种格式最多留一份。确认新结果没问题后可以手动删，`CODE_FINAL` 会把它们列出来。
13. **不能关 `LLAMA_BUILD_SERVER`**。上游 `tools/CMakeLists.txt` 把 `add_subdirectory(cli)` 写在 `if (LLAMA_BUILD_SERVER)` 里，关了就没有 `llama-cli`，S7 会缺工具。省编译时间请改 `CUDA_ARCH` / `BUILD_JOBS`，别动 SERVER。
14. ~~S2 的编译时间与峰值内存全是估算~~ **已被实测取代（v0.3.2）**：真实 Kaggle T4 上 27 分 52 秒，S2 后 working 剩 19.07 GiB。`STAGE_BUDGET_SEC["S2_BUILD"]=2400 s` 的预算对这个耗时仍有余量；峰值 RAM 与 build 目录大小仍未观测，超时仍可按 runbook 4.3 降 `BUILD_JOBS` 或改 `BUILD_CUDA=False`。
15. **`-DLLAMA_USE_PREBUILT_UI=OFF` 依赖上游有这个选项名**。它在顶层 `CMakeLists.txt:138` 用 `option()` 声明；如果将来改名，cmake 只会对未知的 `-D` 变量警告一句然后忽略，后果回到"构建期尝试下载 UI"——不会失败，只是慢。runbook 的核对清单里有这一项。
16. **tarball 源码没有 `.git`，commit 的校验强度取决于来源**。自动下载路径记 `commit_source="pinned_tarball"` + `commit_verified=True`，它证明"这份 tar.gz 的 sha256 与预登记值一致"，**不等于** git 校验；手动挂载的源码只能记 `declared`。要 git 级校验，需要自己 `git clone` 后保留 `.git` 再挂载（并设 `CFG["LLAMA_SRC"]`）。
17. **预登记 sha256 本身是自证的**。它写在 notebook 里，校验只能防"上游文件被替换 / 网络返回错误页面"，防不了"notebook 里的预登记值被改坏"。第一次使用时应在报告第 2 节核对那串 sha256 与本文第四节列出的值一致。
18. **Kaggle 能否直连 github.com 未验证**。自动下载走标准库 `urllib`（不依赖 curl/wget）。如果 Kaggle 网络策略拦了 GitHub，S2 会给出"检查 Settings → Internet = On"的报错，此时用断网备选：手动下载同一 tarball、挂成 Dataset、设 `CFG["LLAMA_SRC"]`。
19. ~~`-DGGML_CUDA_NO_VMM=ON` 只跨过 Generate 阶段~~ **已被实测取代（v0.3.2）**：真实运行里 build 阶段同样通过，四个二进制就位。该开关的代价不变且有限：显存池从 VMM 回退到 cudaMalloc 池（`ggml-cuda.cu:745`），算力内核与数值不变；构建产物会带 `NO_VMM=1` 特性标记。仍未观测的是 build 目录大小与峰值 RAM。
20. **错误分类器按关键词匹配，永远可能漏**。v0.3.0 的 `"nvcc" in tail and "not found" in tail` 全文包含判断就把 driver 错误误判成了 nvcc 缺失。现在改成同行匹配 + driver 专用分支，但新的日志形态仍可能落到通用分支 —— 通用分支的处置是"看日志最后 30 行"，这是兜底不是终点。
21. **S3 已在真实 Kaggle T4 上通过（2026-10-22）**：imatrix 10,701,952 B / 1008 张量 / 57.95 s，`-ngl 99` 全上 GPU，显存总量 14.56 GiB。**但 S4/S5 的一键化链路仍只有 CPU/mock 测试**：BF16 15.26 GiB 的实际下载耗时、`llama-quantize` 的 CPU 耗时与峰值 RAM、S5 的真实字节账，都还是估算。第一次跑请对照 runbook 第 7 节的清单逐条核。
22. **Kaggle 能否直连 huggingface.co 未验证**（github.com 已实测可达）。HF 的 CDN 域名不同，网络策略可能不同。连不上时 S3 会报可操作的错并保留断点。
23. **Kaggle 镜像里有没有 `pyarrow` / `pandas` 未验证**。设计上不硬依赖（先 pyarrow 再 pandas，都没有就报错），退路是自己把 parquet 转成 `calib.txt` 挂成 Dataset 并设 `CFG["CALIB_TXT"]`。
24. **`/kaggle/working` 是否跨 interactive session 保留未验证**。官方只说 ≤ 20 GB 自动保存；"掉线后不 commit 是否一定恢复"没写。v0.4.0 按"不保"做最坏假设，每次启动探活二进制，探不到就自动重编约 28 分钟 —— **首次新 Session 重编是可接受代价，已如实记录**。
25. **Q8_0 被回收之后，重跑 S3 会重新下载 8.11 GiB**。这是决策 5（成功后才清理）的直接后果。想避免就把 `CFG["S3_DELETE_Q8_AFTER"]` 改成 `False`，代价是 working 少 8.11 GiB。
26. **imatrix base 是 Q8_0，不是 f16**。激活统计量是对 f16 的近似，比"从 Q8_0 二次量化"好，但不等于 f16 口径。`S3_IMATRIX.json` 里 `imatrix_base_from_approx=True`，S9 报告会声明 —— 别在结论里当成 f16。
27. **校准规模 32 × 512 = 16,384 token 只是 pilot**。对 IQ2_XS 这种 2.31 bpw 的格式，imatrix 质量对校准量敏感；要更稳需调 `IMATRIX_CHUNKS`，耗时线性增长。
28. **复用挂载的二进制时，commit 只能是声明值**。`build_reuse` 三档里 `attached_input` 和没有源码的 `existing_build` 都拿不到 git 或 tarball digest，`commit_source="declared"` + `commit_verified=false`。报告里会明说"未校验"，不会伪装成已验证。
29. **Kaggle 暂存盘的容量没有官方数字**。官方只说有 "Additional scratchpad disk space (outside /kaggle/working) that will not be saved outside of the current session"，没给大小。v0.5 因此**运行时探测**：只认 `st_dev` 与 working 不同的候选，且要求空闲 ≥ BF16 + 余量。探测不到就走只读 Input 路线（`trlab_fetch_bf16.ipynb`），不会把账算错。
30. **Kaggle 能否直连 huggingface.co 未验证**（github.com 已实测可达，HF 的 CDN 域名 `xet-bridge-us` 可能不同策略）。15.26 GiB 的 BF16 连不上时 S4 会报可操作的错并给出 helper notebook 说明；届时用手动下载 + 挂 Dataset 也一样（字节数与 sha256 双校验不依赖下载路径）。
31. **S4 的 CPU 耗时与峰值 RAM 未实测**。`llama-quantize` 是 CPU 路径（15.26 GiB 走内存不走显存），输入 mmap 但 `--max-buffer-size` 默认 8192 MiB 是实打实要占的。`_ram_precheck()` 只警告不拦；真的 exit 137 时按提示调小 `S4_RAM_MARGIN_GB` 或加 `--max-buffer-size`。
32. **暂存盘路线不跨 Session**。下到暂存盘的 BF16 会话结束就没了，下次要重下；只读 Input 路线才是一次操作永久复用。这是两条路线的本质差别，报告里会记 `quant_base_origin`。
33. **provenance 级核验弱于文件级**。三项证据（sha256 / 来源 repo+revision / 张量结构）互相独立，且"同仓库同 revision ⇒ 同一次转换"比"形状一致"强，但它仍然不是逐张量比形状。S9 报告会显式声明这一条的效力范围（`imatrix_base_consistency_level`）。
34. **S6/S7 尚未在真实 Kaggle 上跑过**（v1.1.1）。144 项测试全是 CPU/mock；`test_s6_end_to_end_from_hf` 用的是假 HF 服务器桩。**任何"NLL 差值 / 质量差异"目前都不存在**，报告里那两节会直接写 "S6/S7 not yet evaluated"。
35. **HF 的真实下载只验到 401 路径**。`python -m trlab --check-hf` 在无 token 时正确报 `token_present: false` + 401 并给三条取 token 的路；带 token 下载 2.7 / 4.56 GiB 的过程**未实测**（断点续传逻辑与 S3 的 Q8_0 下载共用同一份 `_fetch_pinned_blob`，那份是真机跑过的）。
36. **评测成本是假设速率，不是实测吞吐**。`eval_cost_estimate()` 用 CPU 2.5 tok/s / GPU 30 tok/s 的**假设值**（`assumed_rate_note` 里写明），0.94 h / GPU 路径的小时数都不能当承诺。真实吞吐要在第一次跑完后看 `measured_throughput` 才有。
37. **`hf_sidecar_declared` 级 fairness 未在真实产物上走过**。测试里是构造的 sidecar；真实 sidecar 的字段是否齐、`imatrix_base_consistent` 在跨 Session 时会不会误判 `false`，都要第一次真跑才知道。凡是引用 HF 产物的结论都必须带这个前缀。
38. **上传路径只有 dry-run 与 mock 覆盖**。v1.1.1 起 `RUN_EVAL=True` 会自动出站那 5 个小文件，但**真往 HF 写东西还没有发生过**；`upload_after_run()` 的异常分支只被单测覆盖。
39. **HF 仓库是 Private，token 是硬前提**。`--check-hf` / S6 / S7 在缺 token 时都会停在预检并说清三条取 token 的路；代码**不会**擅自把仓库改 Public，也不会要求把 token 写进 notebook / git / issue。
40. **HF 上传时只做了远端 size 对比，不是逐文件内容 sha256**。HF 对 LFS 文件提供 `lfs.oid`（= 内容 sha256）可以随时复核，但"上传时校验过什么"这件事本身要如实说：`verified_level` 只能是 `hf_sidecar_declared`。
41. **512 ctx 的结论不能外推到长上下文**。默认规模是 8 组 × 1 chunk × 512 ctx，配对比较只要求两模型同设置，所以内部一致性没问题，但"在这个 ctx 下"是结论的一部分，不能省略。
42. **"默认用 GPU"只是意图，不是保证**。`eval_device()` 只有在真的读到 CUDA 显存时才返回 `gpu`；读不到（没装 torch、驱动异常、不是 GPU Session）就是 CPU。这时**不会**报错，只会在 `device_by_fmt` / `resources.device_why` 里写原因 —— 所以看结果时**先看 device**，别假设"我开了 GPU 就一定在 GPU 上跑"。
43. **`-ngl 99` 是"整模型卸载"，不是"保证不 OOM"**。T4 的 15 GiB 装 4.56 GiB 的 IQ4_XS 很宽裕，但如果 KV cache + compute buffer 超了，llama.cpp 会在运行时 OOM；`_fit_ngl` 的余量（`NGL_MARGIN_GB=2.0`）是静态估算。真 OOM 时按 6.2.6 降 ctx 或退 CPU。
44. **`EVAL_CPU_MAX_WALL_SEC` 不拦 GPU**。`budget_applies` 只在 CPU 时为真；GPU 路径没有独立墙钟上限（阶段预算 `STAGE_BUDGET_SEC["S6_NLL"]=5400` 仍然生效）。
45. **每个阶段都出站意味着重复覆盖的成本**。`results/v1.1.1/<run_id>/` 用 UTC 时间戳分目录，所以多轮不互相覆盖；但同一轮里 S6 之后和 S7 之后会各传一次 `EVAL_SUMMARY.json`（后者更新）。这是刻意换来的"随时可以关 Session"。

---

## 十二、本地 dry-run 与测试记录

本机无 GPU、无 torch、无 llama.cpp。以下全部实跑过：

| 验证项 | 结果 |
|---|---|
| 16 个 `.py` 文件全部 `compile()` 通过（v1.0 起对象是 `trlab/*.py`，不再是 notebook cell） | ✅ |
| import 包后执行默认阶段（S0/S1/S9） | ✅ |
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
| **S2 真实 Kaggle 运行** | ✅ **27 分 52 秒通过**（v0.3.1，用户回传 `S2_BUILD-3.json`）：四个二进制就位、`commit_source=pinned_tarball`、三个 cmake 开关生效、磁盘剩 19.07 GiB |
| S2 version 探测 | ✅ stderr 回退 + 双流皆空时给 rc（`_probe_version`） |
| S2 自动取源码（mock） | ✅ 默认不联网；只开 S2；digest 不符拒解压并删包；`..` 条目拒解压；三级缓存不重下；目录名不符拒解压；`pinned_tarball` 校验档；六种失败症状各有处置；真断网时消息含 "Internet = On" |
| **S3 一键化（mock，v0.4.0）** | ✅ 24 项新回归全通过：二进制三档复用 + `--version` 真跑 + 不可执行被拒；S2 失败拦下 S3；掉线重启复用断点；磁盘不足拒下；四重完整性校验；`.partial` 续传 / 坏 partial 重下 / 网络失败保留断点 / 重复运行不重下；`-ngl` 阶梯与 OOM 降级；Q8_0 只在记录写盘后删 / 失败保留 / 关开关时保留；手动 base 不碰不删；S9 报告行渲染 |
| `test_trlab.py` | ✅ **108/108，约 40 秒** |
| **v1.0 执行方式重构（本地）** | ✅ 干净目录真实冷启动：GitHub sparse fetch 钉 commit（9ebf858，约 200 KB）→ `rev-parse` 强校验 → import → S0/S1/S9 全 OK → 产物落盘，全程约 3 秒；同 Session 二次执行 stamp 命中直接复用；取到的包与本地 `trlab/` 逐字节一致 |
| **v1.0 notebook 结构** | ✅ 5 cells / 7,281 B（原 253 KB / 28 cells）；`--embed` 内嵌回退与 sha256 篡改 fail-closed 均有回归 |

**没有**验证的：S3 的真实下载与 imatrix 生成、S4 的真实量化、S6 的 `llama-perplexity` 真实输出解析（正则已用真实格式字符串测过，但没跑过真实二进制）。这三项必须在 Kaggle 上首跑时人工确认。**v1.0 新增的未验证项**：S4/S5 一键链路在真实 Kaggle Session 的首跑、BF16 的 15.26 GiB 真实下载、`llama-quantize` 的 CPU 耗时与峰值 RAM、Kaggle 暂存盘的真实容量（官方未给数字，代码只按 `st_dev` 探测）、以及**仓库转私有后匿名 fetch 会失效**（那时用 `--embed` 变体）。

**v0.3.1 的真实进度（来自用户 2026-10-21 的 Kaggle 运行）**：源码下载与 sha256 校验**已通过**，CMake 检测到 CUDA 12.8.93 与 host GNU 13.3.0、arch=75、`-- Including CUDA backend`，随后在 `ggml-cuda/CMakeLists.txt:181` 因 `CUDA::cuda_driver` target 缺失而 Generate 失败。也就是说：**自动下载链路和工具链预检都被真实环境验证过了，失败的只有 CMake 的 driver 链接这一处**，本版针对它做了热修；v0.3.2 已确认热修后完整通过。

**没有**验证的还有：S2 build 目录大小与峰值内存、`llama-imatrix` 在 T4 上跑 8B Q8_0 的耗时、S3 的 HF 下载实际带宽、`/kaggle/working` 是否跨 session 保留。runbook 第八节列了完整清单。

---

## 十三、如果只想先确认环境

保持 `STAGES` 里只有 `S0_ENV` / `S1_SMOKE` / `S9_REPORT` 为 `True`，其余全 `False`，`Run All`。约 2–4 分钟，不碰大文件、不联网。

**如果 `S0_ENV` 说 `/kaggle/input` 下没有 `.gguf`，那 S4–S8 一定跑不了 —— 这是设计如此，不是 bug。** 注意 v0.4.0 起 **S3 不再需要任何 `/kaggle/input`**：它自己下载 Q8_0 与校准语料。

确认环境没问题之后，**不要**把 `STAGES` 全改成 `True` 再 `Run All`。下一步按 [`KAGGLE_S2_S3_RUNBOOK.md`](./KAGGLE_S2_S3_RUNBOOK.md) 走：把 `RUN_S3` 改成 `True`（`Internet` 记得设 On），notebook 会自己处理 S2 依赖、Q8_0 下载、校准语料与 imatrix 生成，每一段都有停机点和人工核对清单。

> v0.4.0 起 `RUN_S3=True` 会自动把 `STAGES["S2_BUILD"]` 和 `STAGES["S3_IMATRIX"]` 都打开，所以这一步只需要改一个开关，不用再手改 `STAGES`，也**不需要**先单独跑一遍 S2。
