# S3 一键化（v0.4.0，**已实现**）

> **状态：已实现并提交。** notebook `trlab_kaggle.ipynb` v0.4.0，68 项 CPU/mock 测试全通过。
> 用户已在 Issue #5 comment 6330320164 之后的对话里逐条确认了五个决策点，并追加了
> 「Session 掉线后一键恢复」的要求。本文件记录**为什么这么实现**，以及哪些事**仍未实测**。

## 一、已被真实 Kaggle 运行验证的事实（v0.3.1，2026-10-22）

| 项 | 值 | 来源 |
|---|---|---|
| S2 状态 | `ok`，`elapsed_sec = 1672.29`（**27 分 52 秒**） | 用户回传的 `S2_BUILD-3.json` |
| 源码 commit | `d454a204ef2b63af357744809c05af897611a72e`，`commit_source="pinned_tarball"`，`commit_verified=true` | 同上 |
| tarball sha256 | `f565ee976503e0b66997a05554d4881a252ced7687811f2a41df771203b54d7f` | 同上 |
| 工具链 | cmake 3.31.10、nvcc CUDA 12.8、host GCC 13 | 同上 |
| cmake 开关 | `-DGGML_CUDA=ON`、`-DCMAKE_CUDA_ARCHITECTURES=75`、`-DGGML_CUDA_NO_VMM=ON` 全部生效 | 同上 |
| 四个二进制 | 均在 `/kaggle/working/trlab/build-llama/bin/` | 同上 |
| S2 后磁盘剩余 | `disk_free_gb = 19.07` | 同上 |

**结论**：自动下载链路、sha256 校验、安全解压、工具链兼容性、CUDA 编译、`NO_VMM` 热修 —— **全部被真实环境验证**。S2 不再是风险点。

## 二、五个决策点 → 落成了什么

| # | 决策（用户确认） | 实现位置 | 行为 |
|---|---|---|---|
| 1 | Q8_0 在 Notebook 内自动下载，固定 HF revision；只读 Input 作为备用 | `S3_PINNED` + `_ensure_q8()` + `_imatrix_base_for_fp()` | `RUN_S3=True` 时自动下载；用户手填 `CFG["IMATRIX_BASE"]` 优先，自动路径一律不碰也不删那个文件 |
| 2 | imatrix 暂存 `/kaggle/working`，保留验证记录与 provenance | `MDLS/imatrix.gguf` + `OUT/S3_IMATRIX.json` + S9 第 4 节 | 落 `bytes` / `n_tensors` / `sha256` / `base_fp` / `calib_bytes` / `ctx` / `chunks` / `ngl_attempts` |
| 3 | 首轮校准规模维持 32 × 512 tokens，只作为 pilot | `IMATRIX_CHUNKS=32`、`IMATRIX_CTX=512`，报告行标注「pilot」 | `need_tokens = 16384`，并在日志里警告语料不足时的后果 |
| 4 | `-ngl` 自动选择，优先 GPU，预留 ≥ 2 GiB；OOM 允许降层并记录原因 | `NGL_MARGIN_GB=2.0` + `_fit_ngl()` + `_ngl_ladder()` + `_run_imatrix_with_fallback()` | 阶梯 `[99, 18, 0]`；每次尝试的 `ngl / ok / rc / oom / tail` 全量进 provenance |
| 5 | imatrix 验证成功 + 记录写盘后才清理 Q8_0；失败保留可恢复状态 | `stage_s3_imatrix()` 第 5→6 步 + `_reclaim_q8()` | 顺序硬编码为：生成 → 验证 → 写 `S3_IMATRIX.json` → 才 `_reclaim_q8` → 再写一次含回收结果 |

### 2.1 「Session 掉线」追加要求的落点

| 要求 | 实现 |
|---|---|
| 全新 Session 一键启动，二进制**存在且可运行**才复用，否则自动走 S2 | `_find_llama_bins()` 三档来源，每个候选**真跑一次 `--version`**（存在 ≠ 能跑，`_probe_bin_runnable`） |
| 不假设旧工作区 | 启动探活；`existing_build` → `user_bin` → `attached_input`（`/kaggle/input` 自动扫描）→ `rebuilt` |
| 不要求用户手动先跑 S2 | `RUN_S3=True` 自动把 `STAGES["S2_BUILD"]` 一并打开，由 S2 自己按指纹决定要不要真编 |
| 持久化优先 Kaggle 原生机制 | 二进制落在 `/kaggle/working/trlab/build-llama/bin/`（20 GB 自动保存区），并支持 Add Input 挂自己的 Output |
| S3 单开关，S4–S8 全关，默认关闭 | `_apply_run_switches()` 两开关都 False 时**一个字段都不改**（测试钉住） |
| 下载/恢复安全 | `.partial` + sidecar 断点续传；**不用 `huggingface_hub`**（避免 `~/.cache` 第二份完整副本） |
| mock 回归覆盖五类场景 | 见 `test_trlab.py` 的 `test_s3_*` 24 项 |

## 三、磁盘账（按实测数字推算）

| 步骤 | 增量 | working 剩余 |
|---|---|---|
| S2 完成后（实测） | — | **19.07 GiB** |
| 下载 Q8_0 | −8.11 GiB | ≈ 10.96 GiB |
| 写 imatrix | −0.02～0.05 GiB | ≈ 10.9 GiB |
| **删掉 Q8_0**（imatrix 已生成） | +8.11 GiB | ≈ 19.07 GiB |

**关键结论：S3 自己不会爆盘**，前提是 imatrix 生成后立刻回收 Q8_0。反过来说，**Q8_0 绝不能被留在 working 里等 S4**：S4 的 BF16（15.26 GiB）只能是只读 Input，两个产物（IQ2_XS 2.2 + IQ4_XS 4.0 GiB）落 working，`19.07 − 6.2 ≈ 12.9 GiB` 仍够 —— 这个前提就是 Q8_0 已经被删掉。

磁盘预检 `_disk_precheck()` 在下载**之前**就拦：`free − 需要 < 保留下限` 直接 `GuardError`，报出缺口，不下一半再爆。

## 四、三个必须如实回答的问题

### 4.1 会不会因为开 S3 又重编 28 分钟 S2？

只有「新 session 且 working 被清空」会。启动顺序：

1. `existing_build`：`/kaggle/working/trlab/build-llama/bin` 有四个二进制且 `--version` 真跑通 → 复用，零编译；
2. `user_bin`：`CFG["LLAMA_BIN"]` 指向的目录，同样真跑 `--version`；
3. `attached_input`：扫 `/kaggle/input`（限时 20 s、限 20000 目录）找同名二进制；
4. 都没有 → 完整走 S2（下载 pinned 源码 + 编译 ≈ 28 分钟）。

**复用二进制时 commit 只能记 `declared`（未校验）** —— `_resolve_commit_record()` 在没有源码的 session 里回落到 `S2_PINNED["commit"]` 并在日志里明说"未校验"。报告里的 `commit_source` 三档：`git` > `pinned_tarball` > `declared`。

### 4.2 Kaggle 官方 Output/版本机制保不保 build？

已从官方文档核实的事实：

| 事实 | 出处 |
|---|---|
| 交互 session **空闲 20 分钟**即结束 | `kaggle.com/docs/notebooks` → Technical Specifications |
| `/kaggle/working` ≤ 20 GB **自动保存**，可在未来 Notebook 复用 | 同上 |
| **Save & Run All = 全新干净状态**（"a completely clean state"），12 小时上限 | 同上 |
| Quick Save 只快照、不重跑 | 同上 |
| Add Input 可以搜「任何 Notebook 的输出文件」并挂上 | 同上 |

**"掉线后不 commit 是否一定恢复"官方没写** → 设计上按"不保"做最坏假设，启动时探活，探不到就自动重编。**首次新 session 重编约 28 分钟是可接受代价，已在文档里如实记录。**

### 4.3 已核对 / 待核清单

| 项 | 状态 |
|---|---|
| HF resolve URL 可达、字节数精确等于 8,709,519,168 | ✅ 已核对（HEAD 请求，`x-linked-size` 与 `content-length` 都是这个数；`urllib` 自动跟随 302 到 CDN） |
| revision `a6adef130ffb23ddaf1a62fec9dced968c9bc482`（unsloth/Qwen3-8B-GGUF） | ✅ runbook 3.2 已记录 |
| wikitext train/test parquet 的 revision / 字节 / license | ✅ runbook 3.4 已记录 |
| **双源 digest 互证** | ✅ `lfs.oid` 与 resolve 响应头 `x-linked-etag` 互相独立且一致，均等于内容 sha256；两个 pin 都是这么钉的 |
| Kaggle 镜像里有没有 `pyarrow` / `pandas` | **待核**。设计上不硬依赖：`_parquet_to_txt()` 先试 pyarrow，再试 pandas，都没有就报可操作的错 |
| Kaggle 能否直连 huggingface.co | **待核**（github.com 已实测可达；HF 的 CDN 域名不同，策略可能不同） |
| `/kaggle/working` 是否跨 interactive session 保留 | **待核**（设计按不保留做最坏假设，探活兜底） |
| Q8_0 进显存跑 imatrix 的实测耗时 | **待核**（估 5–15 分钟；纯 CPU 则是小时级，日志会警告） |

## 五、provenance 与隔离

`OUT/S3_IMATRIX.json` 的 `result` 字段（与 S4 的 `quant_fp` 对齐）：

```
imatrix / bytes / n_tensors / sha256
base_gguf / base_kind / base_bytes / base_fp / base_is_quantized
imatrix_base_from_approx   # True = 由 Q8_0 算的，激活是对 f16 的近似，报告必须声明
base_origin / base_download{repo,revision,sha256,bytes,etag_checked}
calib_txt / calib_bytes / calib_source / calib_origin / calib_download
ctx / chunks / need_tokens
ngl / ngl_initial / ngl_attempts[] / on_gpu / vram_total_gb / vram_margin_gb
llama_bin
q8_reclaimed / q8_reclaimed_bytes / q8_reclaim_reason
```

隔离：`CALIB_SOURCE="wikitext-2-raw/train"`、评测走 `EVAL_SOURCE`（test split，S3 不下载）。README 已知限制已写明「隔离只做到 split 级 + 前 200 字符行级重叠检测」。

## 六、守卫（fail-fast，全部在下载/执行之前）

| 守卫 | 触发条件 | 行为 |
|---|---|---|
| 开关 | `RUN_S3=False` | S3 完全不执行，一个字节都不下载 |
| 前置 | `S2_BUILD` 状态为 `error` | S3 直接 SKIP（`STAGE_REQUIRES_OK`）；S2 根本没开则不拦 |
| 网络 | 下载失败 | 保留 `.partial` + sidecar，提示重跑续传；**不假装成功** |
| 磁盘（下载前） | `free − 需要 < 保留下限` | `GuardError`，报出缺口 |
| 磁盘（imatrix 后） | 回收 Q8_0 失败 | 明确报错，**不静默继续** |
| 显存 | Q8_0 + 2 GiB > 可用显存 | `_ngl_ladder` 降层；降到 0 就是纯 CPU，报告如实写 |
| 完整性 | sha256 / `x-linked-etag` / `Content-Length` / size 任一不符 | 拒绝落地，删 `.partial`，报可操作的错 |
| 单副本 | 边写 `.partial` 边算 sha256 → `os.replace` 原子落地 | 不用 `huggingface_hub`，杜绝 `~/.cache` 第二份完整副本 |
| 半成品 | imatrix 生成失败 / 产物体检不过 | 删 `.partial`（不可续跑，不占 working 20 GB）；Q8_0 与断点**保留** |

断点重入：Q8_0 与语料各有 `.stamp.json`（repo/revision/sha256/bytes），命中即复用不重下。**Q8_0 被回收后重跑 S3 会重下 8.11 GiB** —— 这是决策 5 的直接后果，已在 README 已知限制里写明。

## 七、实现范围（已完成）

1. `CFG["RUN_S3"] = False` + `_apply_run_switches()` 增加 S3 分支（只开 S2/S3，填 `S3_AUTO_FETCH_*`）；
2. `S3_PINNED` / `CALIB_PINNED`（repo/revision/sha256/bytes/etag_header），双源 digest 互证；
3. 大文件下载：磁盘预检 + `.partial` 续传 + 四重完整性校验 + 单副本原子落地；
4. 二进制探活优先（三档来源 + 真跑 `--version`）+ `STAGE_REQUIRES_OK` 前置守卫；
5. `-ngl` 2 GiB 余量 + 阶梯降级 + 全量尝试记录；
6. imatrix 生成 → 验证 → 记录写盘 → 才回收 Q8_0 → 再写一次；
7. S9 报告新增第 4 节「imatrix（S3_IMATRIX）」，结论边界补 3 条；
8. 68 项 CPU/mock 测试（44 原有 + 24 新增）。

**不在范围内**：S4–S8 的一键化、BF16 下载、任何真实推理。


---

## 八、v0.5.0 追加：S4 一键量化（**已实现**）

S3 已在真实 Kaggle T4 上跑通（2026-10-22：imatrix 10,701,952 B / 1008 张量 / 57.95 s，sha256 `34062ab6…`，`-ngl 99` 全上 GPU，Q8_0 按设计回收）。这一节记 S4 的落点。

### 8.1 七个需求 → 落成了什么

| 需求 | 落点 |
|---|---|
| 1. 从同源的原始 F16/BF16 量化，禁止 Q8_0 二次量化 | `S4_PINNED`（repo/revision/sha256/bytes 与 S3 同源）；`plan.get("flags")` 非空直接 `GuardError`，`ALLOW_REQUANTIZE` 不给活路 |
| 2. 解决 20 GiB working 限制，优先云端下载 + 只读 Input | `_resolve_s4_base()` 四级优先级 + 一次性 helper notebook（6 次点击） |
| 3. 复用现有 imatrix，新 Session 也能恢复 | `IMATRIX_PINNED` + `_verify_imatrix()` 三态 + `S3_SKIP_IF_IMATRIX_OK` + 挨着产物的 `models/imatrix.provenance.json` |
| 4. 修 `_base_consistency()` 依赖已回收的 Q8_0 | file / provenance 两级同源核验（见 8.3） |
| 5. 修 `base_origin` 误记 `manual` | `_s3_base_origin()` 改为看路径落点与 stamp，不看被污染的 CFG |
| 6. 完整来源记录 + 磁盘/RAM 检查 + 断点 + 失败恢复，默认不自动下载 BF16 | `S4_AUTO_FETCH_BASE` 默认 False；provenance schema 2；`_disk_precheck(root=)`；`_ram_precheck()` |
| 7. 顺带打开 S5 真实字节/bpw，不启动 S6–S8 | `RUN_S4` 分支只开 S4_QUANT + S5_BYTES；`_s4_base_for_s5()` 用 S4 实际取的 base |

### 8.2 磁盘算术（这条路线的第一约束）

S2 后 working 实测剩 19.07 GiB；BF16 15.26 + IQ2_XS 2.36 + IQ4_XS 4.34 = 21.96 GiB，连 BF16+IQ4_XS = 19.60 GiB 都超。**BF16 必须来自不占 working 配额的地方。** 因此 base 取源按 `manual → attached_input → scratch_download → working_download` 排序，暂存盘只认 `st_dev` 与 working 不同的候选（同设备上的"另一个目录"不提供额外空间，算进来会导致虚假的磁盘账）。

Kaggle 官方只说有 "Additional scratchpad disk space (outside /kaggle/working) that will not be saved outside of the current session"，**没给容量** —— 所以是运行时探测，不是假设。探测不到就报 `GuardError` 并给出 helper notebook 的 6 步说明，不瞎试。

### 8.3 imatrix 与量化 base 的同源性：两级核验

imatrix 的 base（Q8_0）按设计会被回收，所以"读原文件逐张量比形状"这条路会堵死（v0.4 的真实缺陷：S4 直接 `GuardError`，只能重下 8 GiB）。改成：

- **file 级**（Q8_0 还在）：architecture / 张量名 / dims / 参数量，`_base_consistency()`；
- **provenance 级**（Q8_0 已回收）：① imatrix sha256 与 provenance 记录一致；② 记录的 base repo/revision 与 `S4_PINNED` 相同 —— **同仓库同 revision ⇒ 同一次转换 ⇒ 同一份权重**，这比"张量形状一致"强得多（三个仓库的 Q8_0 字节数都不同，形状必然一致）；③ imatrix 张量名 ⊆ base 张量名且层数一致；④ 每个 imatrix 张量的 `in_sum` 元素数 = base 对应张量的输入通道数。

③④ 的依据是本地实测的 imatrix GGUF 结构：**没有** `general.architecture`，1008 个张量 = 36 层 × 7 种 × 4 个统计量（`.in_sum` / `.in_sum2` / `.counts` / `.stats`），`in_sum` 的元素数就是 n_in（`ffn_down.weight` → 12288 = ffn 宽度，其余 → 4096 = hidden）。dims 方向自动判定（要求所有张量同一方向 100% 匹配），防止 ggml `mul_mat` 约定与写入端相反。报告里会写清核验到了哪一级，`provenance` 级必须显式声明"弱于文件级"。

### 8.4 测试

98 项 CPU/mock 回归（44 原有 + 24 v0.4 + 30 v0.5）。v0.5 抓到并修掉 1 个真 bug：`stage_s4_quant` 里 `imatrix_base_kind` 变量名笔误 → `NameError`（由 `test_t4_split_base_path` 抓出）。

**仍未验证**：BF16 15.26 GiB 的真实下载、`llama-quantize` 的 CPU 耗时与峰值 RAM、暂存盘容量、provenance 级核验在真实"回收 → 新 Session → S4"路径上的首次走通。这些都写进了 README 已知限制与 runbook 未验证清单。
