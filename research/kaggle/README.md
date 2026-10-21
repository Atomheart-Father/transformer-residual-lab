# Kaggle 低位宽量化验证台

`trlab_kaggle.ipynb` —— 一个**可直接上传到 Kaggle** 的独立 Notebook，用来在同一份 `Qwen/Qwen3-8B` 上产出 `IQ2_XS` 与 `IQ4_XS`，并比较**真实文件字节**、**配对 NLL**、**有限配对质量**。

首版只做环境检查与短时 smoke test；量化与评测阶段全部默认关闭。

---

## 一、这个 Notebook 不做什么

写在前面的原因是：这些不是"还没做"，是**设计上就不做**。

| 不做 | 机制 |
|---|---|
| 自动下载大模型 | `CFG["ALLOW_NETWORK"] = False`；S3 起全部用 `need()` 检查本地路径，缺失即 `GuardError` |
| 自动 `pip install` | 只用 Kaggle 镜像自带的包；llama.cpp 从源码编译；GGUF 解析器是 90 行纯 Python，无第三方依赖 |
| 自动租 GPU / 跑实验 | 阶段开关默认全关；用户手动改 `STAGES` 后在网页上逐段执行 |
| 上传私有项目文件 | Notebook 不引用 `transformer-residual-lab` 的任何本地数据或脚本 |
| 在单张 T4 上硬加载 8B BF16 | S8 先算显存，放不下就返回 `status=skipped` 并写明所需/可用 GB，不硬撑到 OOM |

---

## 二、文件

```
research/kaggle/
├── trlab_kaggle.ipynb   # 唯一需要上传的文件（28 cells：16 code + 12 markdown）
└── README.md            # 本文件
```

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
| `S3_IMATRIX` | 关 | S2 + base GGUF + 校准语料 | 3600 s | `imatrix.gguf` + sha256 | 校准信息是否对等 |
| `S4_QUANT` | 关 | S2 + S3 + base GGUF | 1800 s | 两个 GGUF + 逐文件 bpw | 能否编码、字节多少 |
| `S5_BYTES` | 关 | S4（或任意 GGUF） | 300 s | 逐张量类型字节、按 attn/ffn/embed 分类 | 哪些张量没被低位量化 |
| `S6_NLL` | 关 | S2 + S5 + 评测语料 | 5400 s | 分组配对 NLL + bootstrap CI | 2.31 bpw 差多少 nats |
| `S7_QUALITY` | 关 | S2 + S5 | 1800 s | 贪心续写一致性、repeat4 | 有没有整体崩塌 |
| `S8_BF16_ANCHOR` | 关 | 高精度 GGUF + 足够显存 | 3600 s | 锚点 NLL 或明确的 skip 记录 | 锚点能不能上，`rho` 能不能算 |
| `SX_EXPLORE` | 关 | 第三方 GGUF | 1800 s | 字节 + NLL，标 `provenance=third_party` | 这个文件本身如何（**不参与公平比较**） |
| `S9_REPORT` | **开** | — | 120 s | `report.md` + `results.json` | 汇总 |

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

**base 选 f16 还是 Q8_0？**

| | f16 | Q8_0 |
|---|---|---|
| 大小 | 15.25 GiB | ≈ 8.1 GiB |
| imatrix 在 T4 上 | **放不进显存，退 CPU**（慢几十倍） | 能进显存，正常速度 |
| 与论文口径 | 一致 | 略偏 |

**T4 上推荐 Q8_0**。选哪个都必须写进 `CFG["BASE_KIND"]`，它会进 provenance。

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

### 断点恢复

```
trlab/ckpt/state.json      # {"S0_ENV": {"status":"ok",...}, "S3_IMATRIX": {"status":"error",...}}
trlab/out/<STAGE>.json     # 每阶段独立结果，可单独读
trlab/logs/*.log           # 外部命令的完整 stdout+stderr
```

`status == "ok"` 的阶段重跑时自动跳过。强制重跑改 `CFG["FORCE_RERUN"] = True`。

失败不会中断后续阶段 —— 每个阶段自己捕获异常并写 `status: "error"` + traceback，S9 仍然会产出一份带错误摘要的报告。

### 产物

```
trlab/out/report.md        # 人读
trlab/out/results.json     # 机读，含 config_snapshot 与逐阶段状态
trlab/out/S*.json          # 逐阶段
trlab/models/*.gguf        # imatrix + 两个量化文件（约 7 GiB）
```

---

## 九、已知限制

1. **T4 没有原生 bf16**（CC 7.5）。S1 会实测 fp16 vs bf16 的 TFLOP/s 比值；比值明显小于 1 就说明 bf16 走了模拟路径。锚点因此只能用 **f16**，与论文口径有偏差，S8 会把这条写进 `dtype_deviation`。
2. **单张 T4 装不下 8B 高精度权重**：f16 = 15.25 GiB，可用显存约 15.0 GiB，且未计 KV 与 CUDA context。S8 默认 `status=skipped`，并给出 `required_gb` / `vram_usable_gb` 两个具体数字。
3. **隔离只做到内容级**：S6 用前 200 字符做行级重叠检测。如果校准集与评测集来自同一文档的不同段落，这个方法检测不出来 —— 必须靠 `CALIB_SOURCE` / `EVAL_SOURCE` 声明不同 split。
4. **`/kaggle/working` 约 20 GB**。两个 GGUF（2.2 + 4.0 GiB）+ imatrix 约 7 GiB，余量够；不要把 base 也放进去。
5. **S7 样本量极小**（8 prompt × 32 token）。只能发现整体崩塌，不能排序细微质量；`repeat4` 是退化信号，不是质量分数。
6. **NLL 的 CI 跨 0 不等于等价**。`paired_bootstrap` 只给区间，判读在 S9 里写死为"未测出差异"。
7. **不能外推**。单张 T4、8B、单一语料的结果不适用于 27B，也不适用于 M4 Pro。**吞吐不可跨平台迁移，只有格式与质量可以。**

---

## 十、本地 dry-run 记录

Notebook 在写入仓库前，已在本机（无 GPU、无 torch）实跑过：

| 验证项 | 结果 |
|---|---|
| 16 个 code cell 全部 `compile()` 通过 | ✅ |
| 按顺序执行全部 cell（默认阶段） | ✅ 16/16 |
| 第二次执行 | ✅ 全部 `RESUME` |
| 全阶段开启但无任何输入文件 | ✅ 每个阶段干净 `GuardError`，S9 仍出报告，无 traceback 崩溃 |
| 合成 GGUF 往返（F16/IQ2_XS/IQ4_XS/Q4_K/F32） | ✅ 字节逐项相等，bpw = 2.3125 / 4.25 |
| 真实 GGUF 解析（`ggml-vocab-qwen35.gguf`，5.9 MB 头、15 万词表数组） | ✅ 0.06 s，`header == file_bytes` |
| `split_eval` 分组 | ✅ 8 组等长，40 段无重叠、无丢失 |
| `paired_bootstrap` | ✅ n=8 给 CI；n=1 / n=0 返回 `ci95=None` 而不是崩 |
| `_fit_ngl` | ✅ 2.4 GiB→99，16.4 GiB→0，无 GPU→0 |
| S5 端到端（合成 Qwen3 结构双文件） | ✅ 逐类型分解 + 字节比 1.677 |

**没有**验证的：S2 的实际编译、S3/S4 的真实量化、S6 的 llama-perplexity 输出解析（正则已用真实格式字符串测过，但没跑过真实二进制）。这三项必须在 Kaggle 上首跑时人工确认。

---

## 十一、如果只想先确认环境

保持 `STAGES` 里只有 `S0_ENV` / `S1_SMOKE` / `S9_REPORT` 为 `True`，其余全 `False`，`Run All`。约 2–4 分钟，不碰大文件、不联网。

**如果 `S0_ENV` 说 `/kaggle/input` 下没有 `.gguf`，那 S3–S7 一定跑不了 —— 这是设计如此，不是 bug。**
