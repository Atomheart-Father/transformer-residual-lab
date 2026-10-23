# HF 模型卡（可直接粘贴到 HF 仓库 README）

这份文件是 `Atomheart-Father/trlab-qwen3-8b-quant-pilot` 这个 **Private** 仓库的
README 内容草稿。它只陈述**已经发生过的事实**，不把未运行的评测写成结论。

> 粘贴位置：HF 仓库页 → `Files` → `README.md` → `Edit`。
> 不要在 README 里写 token、Kaggle 账号、或任何访问凭据。

---

## 标题与状态横幅（第一段就写清楚）

```markdown
# trlab · Qwen3-8B IQ2_XS / IQ4_XS pilot（量化产物，未评测）

> **状态：S6（配对 NLL）与 S7（质量检查）not yet evaluated。**
> 本仓库目前只包含**量化产物与 provenance**，不包含任何"量化对能力影响"的结论。
> 两个格式之间的质量差异**尚未测量**，请不要据文件体积推断质量。
```

## 仓库里有什么

| 文件 | 是什么 | 备注 |
|---|---|---|
| `qwen3-8b-IQ2_XS.gguf` | 2-bit 量化权重（IQ2_XS） | 从**原始 BF16** 量化，不是从 Q8_0 二次量化 |
| `qwen3-8b-IQ4_XS.gguf` | 4-bit 量化权重（IQ4_XS） | 同上，与 IQ2_XS 共用同一份 imatrix |
| `imatrix.gguf` | 重要性矩阵 | 由 **Q8_0** 基座算出的**近似**，不是 BF16 真值（见下） |
| `qwen3-8b-IQ2_XS.provenance.json` | 该格式的生成记录 | 参数、llama.cpp commit、字节数、sha256 |
| `qwen3-8b-IQ4_XS.provenance.json` | 同上 | |
| `imatrix.provenance.json` | imatrix 的生成记录 | 基座来源、张量数、sha256 |

**权威字节数与 sha256 以各 `*.provenance.json` 为准**，不要在文档里手抄近似值。

### 实验记录会放在哪（现在还没有）

评测跑起来之后，每个阶段的小结果会自动落到
`results/v1.1.1/<run_id>/`（`<run_id>` 是 UTC 时间戳）：

| 文件 | 内容 |
|---|---|
| `EVAL_SUMMARY.json` | 几百字节的摘要：均值 NLL、配对差 + CI、verdict、设备、吞吐、fairness 级别 |
| `S6_NLL.json` | 逐组 PPL/NLL、bootstrap、输入 provenance |
| `S7_QUALITY.json` | 8 个 prompt 的续写一致性 / repeat4 |
| `report.md` / `results.json` / `S9_REPORT.json` | 汇总报告 |

**当前这几个文件都不存在** —— 因为 S6/S7 一次都没跑过。上传只覆盖这些小文件，
不会往这里放 GGUF 的副本，也不会放日志或凭据。

## 怎么生成的（真实发生过的事）

- 基座：`Qwen/Qwen3-8B` 的**原始 BF16**（`Q8_0` 只用来算 imatrix）。
- 工具：llama.cpp，commit 钉死在 `trlab/config.py` 的 `S2_PINNED["commit"]`。
- 量化器：`llama-quantize`，`IQ2_XS` 与 `IQ4_XS` 各一次，**共用同一份 `imatrix.gguf`**。
- 运行位置：Kaggle 单卡 T4 的一次性 pilot 会话（不是可复现的 CI）。
- 代码：`https://github.com/Atomheart-Father/transformer-residual-lab`
  的 `research/kaggle/`（`trlab/` 包 + 运行手册）。

### imatrix 是近似值（必须写出来）

`imatrix.gguf` 由 **Q8_0** 基座计算，而不是 BF16。Q8_0 与 BF16 的激活分布
**不是同一个东西**，所以这份 imatrix 是**近似**，不是"BF16 真 imatrix"。
它对 IQ2_XS 这种极低位宽的影响方向**没有测量过**，既不能说"影响很小"，
也不能说"影响很大"。三个 sidecar 里都记了 `imatrix_base_from_approx` 这类字段，
读的人可以自己看到这个降级。

## 校验到了什么程度（不要夸大）

上传时做过的事：

- 逐文件**远端 size 对比**（本地 vs HF tree API 的 `size`）。
- HF 对 LFS 文件提供 `lfs.oid`，它**等于内容 sha256**，可随时通过 tree API 复核。

**没有**做过的事：

- 上传后没有做"逐文件下载回来再算一遍 sha256"的独立复核。
- 因此 `verified_level` 只能是 `hf_sidecar_declared`（声明级），不是
  `s4_runtime`（量化当场记录的那一级）。任何引用这份产物的分析都必须
  带着这个前缀说话。

复核方法（可选，不需要真的跑评测）：

```bash
# 只比 size + LFS oid，不下载 4.5 GiB
python -m trlab --check-hf
```

## 局限（照抄，不要删）

1. **pilot 规模**：imatrix 只用 32 × 512 ctx 的校准窗口算出来，
   `IMATRIX_CTX=512`、`IMATRIX_CHUNKS=32`。这不是"完整校准"。
2. **只有两个格式**，没有 Q4_K_M / Q5_K_M / Q8_0 之类的对照。
3. **S6/S7 not yet evaluated**：没有配对 NLL、没有 bootstrap CI、没有质量对比。
   任何"2-bit 还行 / 4-bit 明显更好"的说法在本仓库里**没有依据**。
4. **不是可复现的 CI**：单次 Kaggle 会话产物，上游 llama.cpp 行为、
   Kaggle 镜像版本都会影响结果。
5. **语料只到 Wikitext-2 的量级**：校准集是 wikitext-2-raw **train**，
   评测（未跑）用的是 **test**，两者严格分离。

## 怎么用（llama.cpp）

```bash
# 2-bit
llama-cli -m qwen3-8b-IQ2_XS.gguf -p "..." -n 64 -ngl 0

# 4-bit
llama-cli -m qwen3-8b-IQ4_XS.gguf -p "..." -n 64 -ngl 0
```

`-ngl 0` 是**故意的**：本仓库不假设你有 GPU，也不假设你有额度。
真要上 GPU 就自己改，但别把"我跑通了"当成"质量已验证"。

## 许可与归属

- 权重来自 `Qwen/Qwen3-8B`，遵循其原始许可；本仓库只做量化，**不改变许可**。
- 量化由 llama.cpp 完成，遵循 llama.cpp 的许可。
- 本仓库 **Private**，可见性由仓库所有者决定；本文件不指示任何改可见性的操作。
