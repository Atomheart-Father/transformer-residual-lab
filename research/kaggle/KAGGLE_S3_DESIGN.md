# S3 一键化设计（等待授权，**未实现**）

> **状态：设计提案。** 用户在 Issue #5 comment 6330320164 提出方向并明确"尚未授权任何实机 GPU/模型下载或提交新的实现"。本文件只做设计与核对，不构成实现承诺。实现前需要用户对第七节的决策点逐条确认。

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

## 二、S3 要做的四件事（用户原话归纳）

1. 联网拉 **pinned** `unsloth/Qwen3-8B-GGUF` 的 Q8_0（8,709,519,168 B）；
2. 拉 **pinned** 的 wikitext train 作为校准语料；
3. 生成 `imatrix.gguf`，记录准确来源 / revision / checksum / 文本处理方式 / 训练-测试隔离；
4. 默认不自动下载，明确授权开关才进行；不自动启用 S4–S8，不下载 BF16，不跑真实 8B 推理。

## 三、磁盘账（全部按上面实测数字推算）

| 步骤 | 增量 | working 剩余 |
|---|---|---|
| S2 完成后（实测） | — | **19.07 GiB** |
| 下载 Q8_0 | −8.11 GiB | ≈ 10.96 GiB |
| 写 imatrix | −0.02～0.05 GiB（估算，36 层 × 每层约 5.7 万个输入通道 × 每通道 2 个 float） | ≈ 10.9 GiB |
| **删掉 Q8_0**（imatrix 已生成） | +8.11 GiB | ≈ 19.07 GiB |

**关键结论：S3 自己不会爆盘**，前提是 imatrix 生成后立刻回收 Q8_0。反过来说，**Q8_0 绝不能被留在 working 里等 S4**：S4 的 BF16（15.26 GiB）只能是只读 Input，两个产物（IQ2_XS 2.2 + IQ4_XS 4.0 GiB）落 working，`19.07 − 6.2 ≈ 12.9 GiB` 仍够 —— 这个前提就是 Q8_0 已经被删掉。

## 四、三个必须如实回答的问题

### 4.1 会不会因为开 S3 又重编 28 分钟 S2？

分三种情况，只有第三种会重编：

| 情况 | 行为 | 依据 |
|---|---|---|
| 同一 session 内已经编过 | S2 指纹一致 → `RESUME`，**零编译** | 现有 `run_stage` 断点逻辑，已被真实运行 + mock 双重验证 |
| 新 session，`/kaggle/working` 还在（build 目录连同对象文件都在） | cmake 重新 Configure + **增量** build（对象文件命中，只重编改动的） | cmake 的标准行为；**working 是否跨 session 保留 = 待核，见 4.3** |
| 新 session，working 被清空 | 完整重编 ≈ 28 分钟 | 实测耗时 |

设计上 `RUN_S3` 会把 `STAGES["S2_BUILD"]` 一并打开（靠指纹自己决定要不要真编），**不会**跳过 S2 直接找二进制 —— 否则"报告里没有 S2 记录但用了来路不明的二进制"，provenance 就断了。

### 4.2 Quick Save / Save Version 到底保不保 build？

**不能声称它必保。** 已确认的事实只有一条：同一 session 内 working 里的 build 会被复用（实测）。跨 session / 跨版本的行为属于 Kaggle 平台策略，**未验证**，设计上按"不保"做最坏假设，并给两条不需要碰 Mac 的退路：

1. **接受重编**：`RUN_S3=True` 时 S2 自动重跑，无人工动作，代价是约 28 分钟（第一次之后若是增量则更快）。
2. **把二进制变成只读 Input（一次性 UI 动作）**：Kaggle 支持把 notebook 输出挂成 Input。把 `build-llama/bin/` 四个文件打成一个小 Dataset（约 1.5–3.5 GiB，估算）挂上，设 `CFG["LLAMA_BIN"]` 指过去，之后每个 session 零编译。**这一步需要用户在网页上做一次，且"notebook 输出 → Dataset → Add Input"的确切入口名与是否支持二进制文件，待核。**

### 4.3 已核对 / 待核清单

| 项 | 状态 |
|---|---|
| HF resolve URL 可达、字节数精确等于 8,709,519,168 | ✅ 已核对（HEAD 请求，`x-linked-size` 与 `content-length` 都是这个数；`urllib` 自动跟随 302 到 CDN） |
| revision `a6adef130ffb23ddaf1a62fec9dced968c9bc482`（unsloth/Qwen3-8B-GGUF） | ✅ runbook 3.2 已记录 |
| wikitext train/test parquet 的 revision / 字节 / license | ✅ runbook 3.4 已记录 |
| Kaggle 镜像里有没有 `huggingface_hub` / `pyarrow` | **待核**。设计上**不依赖它们**：下载走标准库 `urllib`（复用 S2 的 `_http_download`），parquet 解析优先用镜像里的 `pandas`，取不到再报可操作的错 |
| Kaggle 能否直连 huggingface.co | **待核**（github.com 已实测可达；HF 的 CDN 域名不同，策略可能不同） |
| `/kaggle/working` 是否跨 interactive session 保留 | **待核**（设计按不保留做最坏假设） |
| Q8_0 进显存跑 imatrix 的 `-ngl` 判定 | 现有 `_fit_ngl`：8.11 GiB + 1.0 GiB < 14.56 GiB → `-ngl 99`，能进显存；**实测耗时待核**（估 5–15 分钟，4 核 CPU 版则是小时级） |

## 五、provenance 与隔离（用户明确要求）

`imatrix.gguf` 旁边写一份 `imatrix.provenance.json`，字段与 S4 的 `quant_fp` 对齐：

```
base_repo        = unsloth/Qwen3-8B-GGUF
base_revision    = a6adef130ffb23ddaf1a62fec9dced968c9bc482
base_file        = Qwen3-8B-Q8_0.gguf
base_bytes       = 8709519168
base_sha256      = <下载时实算，与预登记值比对>
calib_repo       = wikitext（parquet）
calib_revision   = <pin>
calib_split      = train
calib_sha256     = <实算>
calib_chars      = <实算>
calib_preprocess = <实况：如何切段、是否去空行、编码>
eval_source      = wikitext-2-raw/test   # 只声明，S3 不下载
ngl              = 99 | 0
chunks / ctx     = 32 / 512
```

隔离检查沿用现有 S1 的做法：calib 与 eval 若同源，至少做**前 200 字符行级重叠检测**并写进报告；真正的隔离靠 `CALIB_SOURCE` / `EVAL_SOURCE` 声明不同 split，这一点在 README 已知限制里已经写明（"隔离只做到内容级"）。

## 六、守卫（fail-fast，全部在下载/执行之前）

| 守卫 | 触发条件 | 行为 |
|---|---|---|
| 网络 | `RUN_S3=True` 但 Internet=Off 或 HF 不可达 | `GuardError`，消息写"Settings → Internet = On；或改用已挂载的只读 Input" |
| 磁盘（下载前） | `free < Q8_0_bytes + MIN_FREE_DISK_GB` | `GuardError`，报出缺口 |
| 磁盘（imatrix 后） | 回收 Q8_0 失败 / 仍低于下限 | 明确报错，**不静默继续** |
| 显存 | Q8_0 + 1 GiB > 可用显存 | `_fit_ngl` 退 `-ngl 0`（CPU），并在报告里写明"退化到 CPU，耗时按小时估" |
| 开关 | `RUN_S3=False` | S3 完全不执行，一个字节都不下载 |
| 范围 | `RUN_S3=True` | 只开 `S2_BUILD` + `S3_IMATRIX`，`S4`–`S8`、`SX` 全部保持 `False`（同 `RUN_S2` 的纪律，并有测试钉住） |

断点重入沿用现有指纹机制：S3 指纹 = base 文件 (size, mtime, 首尾 1 MiB sha) + 校准语料 + `IMATRIX_CTX` / `IMATRIX_CHUNKS` + notebook 版本。**Q8_0 被删掉之后指纹会变**（文件不存在）→ 重跑 S3 会要求重新下载。这是可接受的：imatrix 只有几十 MB，正确做法是把 `IMATRIX_GGUF` 指向一个持久位置，或者干脆接受"重跑 S3 = 重下 8.11 GiB"。这一点需要用户选（见第七节）。

## 七、需要用户决策的点（实现前必须确认）

1. **Q8_0 的获取方式**：(a) notebook 内直接下载 8.11 GiB（单次 Run All，最省事，但每删一次就要重下）；(b) 先用一个只读 Input 挂 Q8_0（复用 runbook 3.3 的取数 notebook 路线，跨 session 不重下，但要多跑一次 notebook）；(c) 两者都要，`CFG["REUSE_INPUT_GGUF"]` 切换。
2. **imatrix 的持久化**：留在 working（随 session 可能丢失）vs. 引导用户把 `models/imatrix.gguf` 也做成只读 Input。
3. **校准语料大小**：`IMATRIX_CHUNKS=32 × 512 tokens ≈ 1.6 万 token`（现有默认）。要更稳可以调到 64–128 chunks，耗时线性增长。是否调整？
4. **`llama-imatrix` 的 `-ngl`**：默认自动（进得去显存就 99）。是否允许在 T4 上用满显存（Q8_0 8.11 + KV/context，余量约 6 GiB）？
5. **S3 成功后是否顺手把 Q8_0 删掉**：建议删（磁盘账要求），但如果你想留着当 S8 的近似锚点，就得接受 working 只剩约 10.9 GiB、S4 之前必须再删一次。

## 八、实现范围（授权后）

小范围变更清单，逐条可单独回滚：

1. `CFG["RUN_S3"] = False` + `_apply_run_switches()` 增加 S3 分支（只开 S2/S3，填 `S3_AUTO_FETCH_*`）；
2. `_http_download` 增加"大文件 + 磁盘预检 + 断点续传味儿的 `.partial` 复用"（现有实现已写 `.partial`，只需加"已存在且 size 相符就跳过"）；
3. `S3_PINNED`（base + calib 的 repo/revision/sha256/bytes），复用 S2 的 pin 模式；
4. imatrix provenance 落盘 + S9 报告第 2 节之后新增/扩展 S3 段落；
5. CPU/mock 测试：默认不开 / 只开 S3 不碰 S4–S8 / digest 不符拒用 / 磁盘不足拒下 / calib 与 eval 同文件时报错 / Q8_0 回收逻辑；
6. README + runbook 同步（三步网页说明加"要跑 S3 就把 `RUN_S3` 也改成 True"）。

**不**在本次授权范围内：S4–S8 的一键化、BF16 下载、任何真实推理。
