# R3-W 修订：XD-1 / XD-3 纯实验设计

**性质**：**设计，不是执行结果**。本轮未运行任何 GPU 基准、未下载模型、未修改任何脚本。
**前置**：本文件取代 `2026-10-r2-report.md` §4 中的 XD-1 与 XD-3 原设计。原设计的两处缺陷见 `2026-10-r3-audit-errata.md` E-1 / E-2 / E-3。
**日期**：2026-10-21

---

## 0. 两条贯穿全部设计的硬规则

**规则 1：实际字节，不用名义 bpw。**
所有 `B` 都必须由量化后的数组实测得出：
```
B = q.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)
```
按张量求和。理由：名义 bpw 会把 metadata 四舍五入掉，而 MLX affine 的 scale+bias 占 0.5 bpw（g64）——在 27B 上是 0.9 GB，足以改变"装得下/装不下"。

**规则 2：working set 必须冷于 cache。**
GEMV 类基准必须让权重 working set **≥ 256 MiB**，否则测到的是 SLC 带宽而不是内存带宽。此门槛借自 `trymirai/uzu` 的 `BENCHMARKS.md`：

> "GEMV-class benches cycle through enough quant-buffer copies to cover a **256 MiB weight working set** before reusing one. This avoids ranking kernels on cache-warm weights."

M4 Pro 的 SLC 容量可观，0.6B 模型的 881 MB 权重已超过门槛；但**合成张量必须显式检查**——一个 `(4096, 4096)` 的 4-bit 张量只有 8.4 MB，会被 cache 完全吃掉，测出的"带宽"会高得离谱。

**规则 3：每个数字必须绑定五项元数据**（沿用 audit C-07）：`git_commit`、`mlx_version`、`model_revision`、`n_tokens`、**是否含 output head**。

---

# 第一部分：XD-1 修订（位宽收益的兑现率）

## 1.1 要回答的问题（拆成四个，每个只由一段回答）

| 段 | 问题 | backend | 唯一自变量 | **允许**得出的结论 | **禁止**得出的结论 |
|---|---|---|---|---|---|
| **A** | 在同一 kernel 家族内，2-bit 相对 4-bit 能兑现多少字节优势 | MLX | bits = 2/4/8 | "MLX affine 的 2-bit 在这台机器上值不值" | 任何关于 IQ2_XS 的结论 |
| **B** | IQ 家族内，2.3125 bpw 相对 4.25 bpw 能兑现多少 | llama.cpp | `--type` = IQ2_XS / IQ4_XS | "IQ2_XS 在 llama.cpp Metal 上值不值" | 任何关于 MLX 的结论 |
| **C** | 各格式的质量代价 | 无关 | 量化格式 | 质量门槛 | 速度 |
| **D** | 真实模型端到端 decode | 选定 backend | 格式 | 最终 tok/s | 单 kernel 的性能归因 |

**为什么必须拆**：`mx.quantized_matmul` 的 MLX affine 2-bit 是 group-64 对称仿射（scale+bias），llama.cpp 的 IQ2_XS 是 512 项 8 元素码本 + 符号 + 子块 scale。**位宽相同不代表 kernel 成本相同**，跨 backend 的速度差里混着 kernel 质量差。

**若 Stage B 无法执行**（如环境无 llama.cpp）：必须把"2.3125 bpw 是否值得"标为**未回答**，**不得用 Stage A 顶替**。这是本设计最重要的一条纪律。

---

## 1.2 Stage A：MLX 内部位宽 A/B

### 对象
- **真实权重**：现有 `mlx-community/Qwen3-0.6B-Base-bf16`（不下载新模型）。逐层取 `q_proj`（2048×1024）、`down_proj`（1024×3072）、`gate_proj`（3072×1024）三个形状，覆盖"方阵/长行/长列"三类。
- **合成张量**：`(8192, 8192)` 与 `(11008, 4096)` 两个形状，bf16 初始化（用 `mx.random.normal` 固定 seed）。用途：把 working set 推到 256 MiB 以上，排除 cache 影响。

### 变量
`mode="affine"`, `group_size=64`, `bits ∈ {2, 4, 8}`。三档都走同一个 `mx.quantized_matmul`。

### 测量
```
对每个 (形状, bits):
  1. q,s,b = mx.quantize(W, group_size=64, bits=bits)
     B[bits] = q.nbytes + s.nbytes + b.nbytes        # 规则 1
  2. 预热 20 次 mx.eval(mx.quantized_matmul(x, q, s, b, ...))
  3. 配对交替：for i in range(200):
        t[2][i] = time(2-bit);  t[4][i] = time(4-bit);  t[8][i] = time(8-bit)
     （三个位宽在同一循环里轮转，避免某一档独占热态）
  4. 每次后 mx.synchronize()
```
- `x` 形状 `(1, K)`，bf16。
- 记录 `mx.get_peak_memory()`。
- 报 `median(t)`、p10/p90、以及**配对中位比** `median(t4[i]/t2[i])`。

### 判据（**已修正**，见 E-1）

```
R_A = median(t4 / t2) / (B4 / B2)
```

| R_A | 判定 | 动作 |
|---|---|---|
| **> 1.05** | 4-bit 基线未调优 / 测量异常 | **不下结论**，先查 4-bit 是否退化到 qmm、是否 cache 命中 |
| 0.85 – 1.05 | 位宽收益基本兑现 | 进入 Stage C |
| 0.70 – 0.85 | 部分兑现 | **按形状分档**：分别报 `q_proj` / `down_proj` / `gate_proj` 的 R_A，不许只报总平均 |
| 0.45 – 0.70 | 大部分被 ALU 吃掉 | 条件性：只有 R_A ≥ 0.85 的形状值得用 2-bit |
| **< 0.45** | 否证 | 停在 4-bit |

**为什么必须分形状**：`down_proj` 是 1024×3072（输出维小、K 大），`gate_proj` 是 3072×1024（输出维大、K 小）。M=1 时前者的行内并行度低，后者高，dequant 成本摊薄方式不同。总平均会掩盖这个差异。

### 这一段的已知局限
- 只测 MLX affine，**不覆盖码本型**。要覆盖码本型必须走 Stage B。
- 0.6B 的权重分布与 27B 不同（离群值比例、scale 分布），Stage A 的 R_A 不能直接搬到 27B。缓解：合成张量用 `mx.random.normal` 之外，再加一组"从 0.6B 真实权重 tile 拼出来的大张量"。

---

## 1.3 Stage B：llama.cpp 内部 IQ 家族 A/B

### 对象
同一个源模型转成两个 GGUF：
- `IQ4_XS`（4.25 bpw，136 B/256 权重）
- `IQ2_XS`（2.3125 bpw，74 B/256 权重）

**用 0.6B 还是更大的模型**：0.6B 太小，working set 881 MB 虽然过 256 MiB 门槛，但 kernel 启动开销占比过高；建议用**能装下的最大模型**（M4 Pro 24 GB 下约 7B @ Q4 ≈ 4.4 GB）。若只能用小模型，必须在报告里注明"kernel 启动占比偏高，R_B 是下界"。

### 测量
```
llama-bench -m model-IQ4_XS.gguf -m model-IQ2_XS.gguf \
            -p 0 -n 128 -r 10 -ngl 99 -fa 1
```
- `-p 0` 只测 decode（generation），不测 prefill——因为稀疏/位宽的收益只在 decode 有意义。
- 两个模型**在同一次 `llama-bench` 调用里交替**（llama-bench 原生支持多 `-m` 轮转），避免热态偏置。
- 记录 `llama-bench` 输出的 `t/s` 与 `model size`。

### 判据
```
R_B = (t_IQ4 / t_IQ2) / (B_IQ4 / B_IQ2)
```
其中 `B` 从 GGUF 文件头读取的**实际张量字节和**（不是 `ls -l` 的文件大小，因为含 metadata/对齐填充）。

| R_B | 判定 |
|---|---|
| > 1.05 | 4-bit 基线异常，先查 `-fa` 与 `-ngl` 设置是否一致 |
| 0.85 – 1.05 | IQ2_XS 值得 |
| 0.70 – 0.85 | 部分兑现 |
| < 0.45 | IQ2_XS 不值得，停在 IQ4_XS |

### 这一段的已知局限
- llama.cpp 的 IQ2_XS kernel 已存在（`kernel_mul_mv_iq2_xs_f32`），所以这一段测的是**真实可用路径**，不是假设。
- 但仍**不能**用它的 R_B 去支持 MLX 的任何结论（kernel 完全不同）。

---

## 1.4 Stage C：质量关卡（独立于速度）

### 设计
- 源模型 → 各格式 → 在**held-out** 文本上算 PPL。held-out 必须与校准集不相交（沿用项目 `make_fresh_split.py` 的断言口径）。
- 一个下游任务：5-shot MMLU 子集（≥ 500 题）或 4 个零样本任务的平均。

### 门槛（任一不过即淘汰该位宽）
| 指标 | 门槛 |
|---|---|
| PPL | ≤ **1.15 ×** 同模型 4-bit 的 PPL |
| 下游 | 下降 ≤ **2.0** 分 |
| top-1 一致率（vs 4-bit，同 prompt） | ≥ **0.90** |

### 为什么门槛设在这里
SpQt 的数据给了一个锚：Llama-2-7B Q4_K 5.28 → 50% 稀疏 5.65（+7%）→ 65% 稀疏 7.39（+40%）。**+15% 的 PPL 容忍度大致对应"还能用"的上限**。若 2-bit 的 PPL 已经 +15%，那么再叠加任何稀疏都会突破。

### 这一段的输出
一张 `格式 × (bpw, 实际字节, PPL, 下游, top1一致率)` 表。**Stage A/B 的速度只有在 Stage C 通过后才有意义。**

---

## 1.5 Stage D：真实模型端到端 decode

### 设计
- 选定 Stage C 通过的格式，跑真实 decode：固定 prompt（≥ 512 token）、生成 128 token、`temperature=0`。
- **必须包含 output head 与采样**（这是 R2 的 C-10 教训）。
- 报告：`tok/s`、`TTFT`、`bytes/token`（= 权重实际字节 + KV 增量）、`mx.get_peak_memory()`。

### 判据
端到端加速比 vs 4-bit 基线，与 Stage A/B 的 R 对比：
- 端到端比 ≈ R → 说明瓶颈确实在 GEMV 权重读取；
- 端到端比 << R → 说明固定成本（head、KV、采样、Python 循环）占比大，需要单独优化那些部分。

### 这一段的纪律
- **不得**用合成 GEMV 的结果推算端到端（R2 犯过的错）。
- **不得**用 0.6B 的端到端加速比外推到 27B（head 占比差 4 倍以上）。

---

## 1.6 XD-1 的执行清单（供批准后照做）

| 步 | 动作 | 预计 | 产出 |
|---|---|---|---|
| 1 | Stage A：3 形状 × 3 位宽 × 200 配对 | 30 min | `xd1_stage_a.json` |
| 2 | Stage B：2 格式 llama-bench 交替 | 40 min | `xd1_stage_b.json` |
| 3 | Stage C：PPL + 下游 | 2–4 h（取决于模型大小） | `xd1_stage_c.json` |
| 4 | Stage D：端到端 | 30 min | `xd1_stage_d.json` |
| 5 | 汇总表 + 判据套用 | 30 min | `xd1_table.md` |

**总计约 4–6 小时**，不下载新模型（Stage B 需要转 GGUF，若已有则跳过）。
**最小可行子集**：只做 Stage A + C（约 1 h），能回答"MLX affine 2-bit 值不值 + 质量过不过关"，但**不能**回答 IQ2_XS。

---

# 第二部分：XD-3 修订（跨轮差异的归因）

## 2.1 先厘清一件事：R2 把两个差异混成了一个

R2 说"61.3 vs 116.9 ms 的 1.9× 差异"。实际有**两个不同的差异**：

| 量 | stage2_0 | stage2_1 | 比值 | 出处 |
|---|---|---|---|---|
| **prefill** `serial-lazy` | 61.32 ms | 116.92 ms | **1.91×** | `bench/bench.json` vs `bench_a345/bench.json` |
| **decode** `serial-lazy` | 6.141 ms | 7.074 ms | **1.15×** | `decode/decode.json` vs `bench_a67/bench.json` |

**prefill 是 1.91×，decode 只有 1.15×。** R2 用 prefill 的 1.91× 去描述"decode 差异"，这是错的。XD-3 必须**分别**解释这两个数。

## 2.2 待检验的假设

| H | 内容 | 若成立则 |
|---|---|---|
| **H1** | 代码路径差异（`73d3379` → `6a5e6ea` 之间 `dag_forward` 或 block 调用方式变了） | prefill 与 decode 的差异应**同时**在受影响的变体上出现 |
| **H2** | 历史环境差异（wired limit、并发进程、热降频、mlx 内部缓存状态） | 差异应在**所有**变体上同比例出现（R2 已观察到 eager 1.81× / lazy 1.91× 的同比位移，这**支持 H2**） |
| **H3** | 测量条件差异（`ids_full` 内容、mask dtype、cache 是否传入） | 差异只出现在特定变体 |

**注意**：R2 观察到的"eager 与 lazy 同比位移"（1.81× 与 1.91×）**在方向上是 H2 的证据**，但当时把它当成了"未解释"。这一步必须显式重做。

## 2.3 设计

### 步骤
1. **取源码**：
   ```
   git show 73d3379:scripts/stage20_bench.py > /tmp/A_bench.py
   git show 6a5e6ea:scripts/stage21_bench.py > /tmp/B_bench.py
   ```
2. **抽闭包**：只保留 `run_serial_eager` / `serial-lazy`（即 `dag_forward(chain_dag)`）两条路径，去掉参数解析、文件 IO、其余 part。
3. **同进程、同模型实例**：
   ```
   载入模型一次（bf16，0.6B）
   ids = 固定的 257 token（两版用同一份）
   预热 10 轮（把热态拉平）
   for i in range(40):
       随机顺序决定先 A 后 B 或先 B 后 A（各 20 对）
       tA[i] = time(A_closure()); mx.synchronize()
       tB[i] = time(B_closure()); mx.synchronize()
       记录 mx.get_peak_memory()
   ```
4. **统计**：配对 bootstrap（10 000 次）算 `median(tB/tA)` 的 95% CI。**不用均值差**（沿用 `scripts/stage17_stats.py` 的配对口径）。
5. **环境快照**：每次运行前后记录
   - `mx.metal.device_info()`（含 `max_recommended_working_set_size`）
   - 是否显式调过 `mx.set_wired_limit`
   - `powermetrics --samplers gpu_power -i 1000 -n 60` 的 GPU 频率/功耗采样
   - 系统负载（`ps aux | grep -c python`）

### 判据（**已修正**，见 E-3）

| 观测 | 归因 | 动作 |
|---|---|---|
| CI **排除** 1.0，中位比 **≥ 1.5** | **H1 代码路径** | `git diff 73d3379 6a5e6ea -- scripts/` 逐行定位，找到具体改动后修掉 |
| CI **包含** 1.0 | **H2 历史环境** | 历史绝对数字永久作废；把当时的环境差异写进报告 |
| CI 排除 1.0，中位比 1.1–1.5 | H1 与 H2 混合 | 分别量化，不得只报一个 |
| prefill 差异复现但 decode 不复现 | H3 或形状相关 | 单独查 `ids_full` 长度与 mask |

### 顺序效应检查（Issue 明确要求）
- A 先 / B 先各 20 对。
- 若"先跑的那一组"系统性更慢（> 5%），说明存在热态效应 → 需要延长预热或改为每组前都做固定热身。
- 顺序效应显著时，**报告必须同时给出"仅后半段"的结果**（热平衡后）。

### 与 commit 变更的关系
即使 H1 成立，也必须回答"改了什么"。预判清单（按可疑度排序）：
1. `dag_forward` 的 `chain_mode="auto"` 是否在两版都退化为串行循环；
2. block 调用签名 `block(h, mask, None)` vs `layer(h, mask, caches[i])` —— 后者多传 caches，可能触发不同的 mask 构造；
3. `create_attention_mask` 的调用位置（每次重建 vs 复用）；
4. 是否在某版里多了 `mx.clear_cache()` 或 `mx.eval` 位置不同。

## 2.4 成本与风险

| 项 | 值 |
|---|---|
| 预计 | 2 小时 |
| 风险 | `73d3379` 的脚本可能依赖已删除模块。缓解：只抽两个闭包，不跑全套 |
| 风险 | 40 对 × 2 变体 × 116 ms ≈ 10 s 纯计算，其余是预热与记录，成本极低 |
| 产出 | `xd3_pairs.json`（逐对原始数据）+ `xd3_verdict.md`（判据结论 + diff 定位） |

## 2.5 XD-3 成功后能得到什么

- 若 H1 成立且定位到行：**修掉后，prefill 与 decode 的绝对毫秒数恢复可引用**，后续所有基准的元数据只需记 commit 即可。
- 若 H2 成立：**永久禁止跨轮绝对时间**，只允许同轮比值；同时得到一条"这台机器上哪些环境因素会造成 1.9× 位移"的清单——这本身对后续所有基准都有价值（比一个数字更重要）。

---

## 3. 与 XD-2 的关系（不在本文件范围，仅说明依赖）

- XD-1 Stage A/B 给出 `R`（位宽兑现率），决定"2-bit 基座"是否成立。
- XD-2（稀疏）**依赖** XD-1 成立：若 2-bit 都兑现不了，2-bit + 稀疏更不可能。
- XD-2 的临界稀疏率判据（s > 0.56）来自 E-2 的字节推导，**不依赖带宽**，因此可以在 XD-1 之前先算；但**是否值得做**依赖 XD-1。

---

## 4. 未决问题（需用户裁决）

1. **Stage B 是否执行**？若环境无 llama.cpp，则"2.3125 bpw 是否值得"无法回答，是否接受这个空白？
2. **Stage C 的下游任务选哪个**？MMLU 子集需要额外依赖；零样本四任务（ARC-e/HellaSwag/PIQA/WinoGrande）可以用 `lm-eval-harness`，但是否允许装包？
3. **XD-3 的 40 对是否够**？若 prefill 单次 116 ms，40 对约 10 s，可以轻松加到 200 对（约 50 s）。建议直接 200 对。
