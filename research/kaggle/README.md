# Kaggle 低位宽量化验证台

`trlab_kaggle.ipynb` —— 一个**可直接上传到 Kaggle** 的独立 Notebook，用来在同一份 `Qwen/Qwen3-8B` 上产出 `IQ2_XS` 与 `IQ4_XS`，并比较**真实文件字节**、**配对 NLL**、**有限配对质量**。

当前版本 **v0.2**。首版（v0.1）只做环境检查与短时 smoke test；v0.2 修掉了一轮逐 cell 审查提出的 3 个 P0 + 4 个 P1，并把它们全部转成了 CPU 可跑的回归测试（见第九、十节）。

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
├── trlab_kaggle.ipynb   # 唯一需要上传的文件（28 cells：16 code + 12 markdown）
├── test_trlab.py        # CPU-only 回归测试，不联网、不需要 GPU / llama.cpp
└── README.md            # 本文件
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
| 能不能当 S8 锚点 | 能（`tier=high`） | **默认不能**（`tier=approx`） |

**T4 上推荐 Q8_0**。选哪个都必须写进 `CFG["BASE_KIND"]`，它会进 provenance。

关于 Q8_0 当锚点，v0.2 起是**硬约束**：`_anchor_decision()` 把 `BASE_KIND` 分档，`bf16/f16/fp16 → high`，`q8_0/q6_k/q5_k/q4_k_m/q4_k/q4_0 → approx`。`approx` 档默认 `status=skipped`，理由是 Q8_0 是**二次量化**，拿它当"原始 checkpoint BF16 真值"是错的。确实只想估个量级时，显式设 `CFG["ANCHOR_ALLOW_APPROX"] = True`，结果会带 `validity_scope="approx_only"`，S9 里不算 `rho`。

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

## 十、怎么跑测试

```bash
python3 research/kaggle/test_trlab.py
```

不需要 GPU、不需要 torch、不需要 llama.cpp、不联网、不下载。它在一个临时目录里把 notebook 的 16 个 code cell 按顺序 exec 一遍，然后跑 18 项断言。当前结果：**18/18 通过，约 8 秒**。

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
| `test_paired_bootstrap_edges` | n=0 / n=1 返回 `ci95=None`；全正差值给不含 0 的 CI |
| `test_ggml_type_table` | 7 个关键 type 的块尺寸/bpw |
| `test_default_stages_complete` | **默认 S0/S1/S9 三个都必须 `ok`，S9 报告必须有效** |
| `test_s9_report_bf16_both_ways` | bf16 报错 / bf16 正常 / 无 torch 三条渲染分支 |
| `test_t4_mock_end_to_end` | 把 GPU 微基准换成 T4 结果，重跑 S1→S9，报告必须写出"bf16 不可用" |
| `test_a100_mock_end_to_end` | 反例：bf16 可用时不能还打印"不可用"，且要给出比值 |

`test_default_stages_complete` 不是装饰。v0.2 加 P1-G 时，`gguf_write_synth` 变成 strict，S1 里"构造一个含未知 type 的坏文件"那行自己就抛了；而 `run_stage` 会吞掉异常只记 `status=error`，于是默认首跑 S1 静默失败、S9 报告里写 `SMOKE_OK = False`。这条测试就是为这种情况写的。

---

## 十一、已知限制

1. **T4 没有原生 bf16**（CC 7.5）。S1 会实测 fp16 vs bf16 的 TFLOP/s 比值；比值明显小于 1 就说明 bf16 走了模拟路径。锚点因此只能用 **f16**，与论文口径有偏差，S8 会把这条写进 `dtype_deviation`。
2. **单张 T4 装不下 8B 高精度权重**：f16 = 15.25 GiB，可用显存约 15.0 GiB，且未计 KV 与 CUDA context。S8 默认 `status=skipped`，并给出 `required_gb` / `vram_usable_gb` 两个具体数字。
3. **隔离只做到内容级**：S6 用前 200 字符做行级重叠检测。如果校准集与评测集来自同一文档的不同段落，这个方法检测不出来 —— 必须靠 `CALIB_SOURCE` / `EVAL_SOURCE` 声明不同 split。
4. **`/kaggle/working` 约 20 GB**。两个 GGUF（2.2 + 4.0 GiB）+ imatrix 约 7 GiB，余量够；不要把 base 也放进去。
5. **S7 样本量极小**（8 prompt × 32 token）。只能发现整体崩塌，不能排序细微质量；`repeat4` 是退化信号，不是质量分数。
6. **NLL 的 CI 跨 0 不等于等价**。`paired_bootstrap` 只给区间，判读在 S9 里写死为"未测出差异"。
7. **不能外推**。单张 T4、8B、单一语料的结果不适用于 27B，也不适用于 M4 Pro。**吞吐不可跨平台迁移，只有格式与质量可以。**
8. **`guard()` 不能阻止运行中 OOM**。它只在阶段入口检查，llama.cpp 的显存分配在外部进程里。
9. **未知 ggml type 的文件没有字节拆分**。这是刻意的：宁可不给数，也不给 8 bpw 的假数。遇到这种情况需要升级 `GGML_TYPES` 表（对照 `ggml-common.h` 的结构体大小）。

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
| `test_trlab.py` | ✅ **18/18，约 8 秒** |

**没有**验证的：S2 的实际编译、S3/S4 的真实量化、S6 的 `llama-perplexity` 真实输出解析（正则已用真实格式字符串测过，但没跑过真实二进制）。这三项必须在 Kaggle 上首跑时人工确认。

---

## 十三、如果只想先确认环境

保持 `STAGES` 里只有 `S0_ENV` / `S1_SMOKE` / `S9_REPORT` 为 `True`，其余全 `False`，`Run All`。约 2–4 分钟，不碰大文件、不联网。

**如果 `S0_ENV` 说 `/kaggle/input` 下没有 `.gguf`，那 S3–S7 一定跑不了 —— 这是设计如此，不是 bug。**
