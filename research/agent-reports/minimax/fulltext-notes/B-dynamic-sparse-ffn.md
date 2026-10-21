# B — Dynamic sparsity in the FFN of **frozen, quantized** LLMs

**Task:** primary-source verification for the R2 "low-bit math, sparse FFN" assignment.
**Goal context:** run 20–30B dense models on 24GB consumer hardware (Apple Silicon / MLX / llama.cpp).
**All sources opened on 2026-10-21** (the "date opened" for every `[P]`/`[O]`/`[S]` claim below).

**Evidence legend**
- `[P]` primary source actually opened (arXiv abs page, arXiv HTML full text, or the paper PDF)
- `[O]` official artifact read directly (OpenReview API record, GitHub API record, HuggingFace page)
- `[S]` secondary source (publisher/aggregator metadata)
- `[UNVERIFIED]` could not open / not found

**Hard constraints honored:** no GPU benchmarks run, no model downloads, no training, no git commands, no modification of any existing repo file. Only this one file was created.

---

## 0. Verification summary (TL;DR)

| Assignment claim | Verdict | Evidence |
|---|---|---|
| "Prox" = arXiv:2607.27591 exists | **CONFIRMED** | `[P]` arxiv.org/abs/2607.27591 — *"Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs"*, Liu Jinyi, Chen Wei, Chen Pengyu, Yuan Xinyi, Bai Minghe, Wu Guoquan, Wei Jun; ISCAS + UCAS + NJUPT; submitted 30 Jul 2026, cs.LG, v1 only, CC BY 4.0 |
| arXiv:2511.04477 = "Enabling Dynamic Sparsity in Quantized LLM Inference" | **CONFIRMED** | `[P]` arxiv.org/abs/2511.04477 — Wang Rongxiang, Shu Kangyuan, Lin Felix Xiaozhu (Univ. of Virginia + Zoom Communications); submitted 6 Nov 2025, cs.DC, v1 only. The system name inside the paper is **SpQt**. |
| The two papers are in the same area | **PARTLY** | Prox = *intermediate-channel* (neuron) selection for SwiGLU. SpQt = *activation/input* sparsity on hidden states, i.e. the TEAL pattern, applied to **every** GEMV. Different mechanisms, complementary. |
| A **joint end-to-end** measurement of (dynamic sparsity × quantization) exists | **NO** | See "Joint-measurement check" below. Neither paper measures speed under quantization; Prox measures *accuracy* under quantization and explicitly defers the speed; SpQt measures speed under Q4_K but with a *different* sparsity mechanism and a *re-laid-out* weight file. |
| Selector overhead accounted for | **Prox: yes (in FLOPs), no (in bytes)**; **SpQt: yes, trivially (a threshold scan)** | Prox's Stage-1 proxy is a real cost and is modelled with a measured constant α. SpQt needs no predictor: it thresholds the hidden state it already has. |
| Non-contiguous / group-wise access problem addressed | **Prox: NOT addressed at all**; **SpQt: this is its central contribution and it empirically shows the problem is fatal** | Prox's paper contains **zero occurrences of the word "byte"** `[P]`. SpQt's "Naive sparse (B1)" baseline measures **<0.2× speedup** at 50% sparsity. |
| Apple Silicon / Metal support | **SpQt: yes, native (Metal, llama.cpp fork)**. **Prox: no (NVIDIA only)** | `[P]` both papers |
| Merged upstream into llama.cpp / vLLM / MLX | **NO for both** | `[O]` GitHub API; no repos, no upstream PRs, no HF papers pages |

**The single most important negative finding:** *Prox reports a speedup but never reports a byte count.* Its "effective sparsity" is a **normalized MAC-cost** metric (`C_Prox = 2α(1−s₁) + 3(1−s₂)`), and its speedups are measured on **FP16** weights with **NVIDIA** GPUs. On a bandwidth-bound 24GB Apple Silicon device, whether the 1.99× survives depends entirely on whether the up/gate projections actually read fewer bytes — and because Prox selects *scattered output channels* of `W_up`/`W_gate`, a row-major group-quantized layout does **not** permit byte savings for those two matrices. Prox does not measure or acknowledge this. SpQt, by contrast, demonstrates the problem quantitatively and solves it with a weight re-layout — but that solution is tailored to *input* sparsity, not to Prox's *output-channel* selection.

---

## 1. Prox — arXiv:2607.27591 `[P]`

Source opened: `https://arxiv.org/abs/2607.27591` and the full LaTeXML HTML `https://arxiv.org/html/2607.27591v1` (1661 lines of extracted text, all sections incl. appendices A.1–A.11 read). PDF not separately needed.

### 1.1 Exact skip condition for SwiGLU FFN

Setup `[P §2.1]`: `FFN(x) = (x W_up ⊙ SiLU(x W_gate)) W_down`, with `u = x W_up`, `h = SiLU(x W_gate)`, `s = u ⊙ h` (the "intermediate SwiGLU state").

Prox uses **output sparsity on the up and gate projections** and **input sparsity on the down projection**, driven by **one shared intermediate-channel mask `m_s`** `[P §2.2, §4.1]`:

```
Stage 1 (proxy) — input mask on x, per token, per layer:
  [m_x]_{ℓ,t,i} = 1[ |x_{ℓ,t,i}| ≥ τ_x,ℓ^{s1} ] ,  i = 1..d_model
  ũ = Π^in(x, W̃_up,   m_x)                 (Π^in(x,W,m) = x_{[I(m)]} W_{[I(m),:]})
  h̃ = SiLU( Π^in(x, W̃_gate, m_x) )
  s̃ = ũ ⊙ h̃
  Stage-1 channel mask, per token, per layer:
  [m_s]_{ℓ,t,j} = 1[ |s̃_{ℓ,t,j}| ≥ τ_s,ℓ^{s2} ] ,  j = 1..d_ff

Stage 2 (exact) — same shared mask m_s applied to all three projections:
  u_sp = Π^out(x, W_up,   m_s)               (Π^out(x,W,m) = x W[:, I(m)])
  h_sp = SiLU( Π^out(x, W_gate, m_s) )
  s_sp = u_sp ⊙ h_sp
  y    = Π^in(s_sp, W_down, m_s)
```

**Granularity answers:**
- **Per-channel** in the *intermediate* dimension (d_ff) for up/gate (output sparsity) and for down (input sparsity) — i.e. one shared channel index set `m_s` for all three matrices. `[P]`
- **Per-token** and **per-layer**: thresholds `τ_x,ℓ^{s1}`, `τ_s,ℓ^{s2}` are layer-wise constants, but the resulting masks are recomputed for every token. `[P §4.1, A.6]`
- **Not per-expert** (dense models only). Not a whole-MLP skip — the down projection always runs, and the up/gate projections always run (on a subset of channels).
- **Threshold-based, not top-k.** Prox explicitly rejects per-token top-k: `[P A.6]` *"We use layer-wise magnitude thresholds for both signals instead of selecting a fixed number of coordinates independently for each token."* Token-adaptivity: within-layer P10–P90 sparsity spread is **17.33 pts** (x mask) / **17.51 pts** (s̃ mask); across-layer averaged spread 12.07 / 11.96 pts `[P Table 8]`. Threshold beats top-k by **+1.79 avg points** over 8 settings `[P Table 9]`.

Oracle bound `[P §3.1]`: an "oracle-s" that knows `s_exact` for free keeps the (1−ρ)·d_ff largest-|s| coordinates; at **70% sparsity most models see <3% relative perplexity increase**.

### 1.2 Selector / proxy cost — **fully** `[P]`

The selector is **not** a learned probe, gate weight, attention predictor, or router. It is:

> **A second, INT4-quantized copy of `W_up` and `W_gate`, evaluated on a magnitude-sparsified input, whose product `ũ ⊙ h̃` is thresholded to produce the mask.**

Quantified cost as the paper models it `[P §4.2]`:
- `C_Dense = 3` (unit cost per up/gate/down projection)
- `C_Prox = 2α(1−s₁) + 3(1−s₂)`, with **α = normalized cost of one INT4 proxy projection relative to a full dense FP16 projection**
- `e = 1 − C_Prox/C_Dense = s₂ − 2α(1−s₁)/3` (the reported "effective sparsity")
- α is **empirically measured**, not assumed: ratios **0.234–0.351**, pooled mean **0.285**; the paper uses the single rounded value **α = 1/3** `[P §5, A.3 Table 5]` — measured across A6000, RTX 3090, A100, 4 Qwen3 scales, 4 input sparsities.

**So yes — Prox explicitly accounts for the selector overhead, in FLOPs.** It is *not* a full extra forward pass: Stage 1 is two projections over only (1−s₁) of the input dimension, in INT4.

**But the cost is paid in memory, and it is significant:** `[P §5.3]`
> *"retaining Stage 1 INT4 proxy weights in GPU memory introduces an **approximate 12% weight-storage overhead**."*

That is 12% of total weight storage *added* to a system whose entire premise is fitting a model into limited memory. The paper's own mitigation is future work (sliding-window load/evict of proxy weights across layers).

Measured latency share `[P A.5, Fig. 8]`, Qwen3-8B @ 70% sparsity on A6000, per generated token over 36 layers:
- Transformer block total: **9.88 ms/token**
- attention + RMSNorm + residual: 4.72 ms/token; **FFN: 5.18 ms/token**
- FFN internals: **Stage 2 gate/up 2.64 ms** (51.0% of FFN), **Stage 1 1.03 ms**, **Stage 2 down 1.47 ms**
- ⇒ the proxy (Stage 1) is **~10.4% of the whole block** and **~19.9% of FFN time** at the 70% operating point (my arithmetic on the paper's numbers).

Proxy quality `[P §3.2, Fig. 3(c,d)]`: exact-vs-proxy top-ranking overlap = **82.77%** for one token, **82.04% mean** over 512 tokens (Qwen3-8B). Analytic justification `[P A.2]`: at 70% input sparsity, idealised proxy/dense correlation ≈ **0.885** while relative projection error η ≈ **0.465**; and a Laplace model says keeping the largest 30% of `s` preserves **87.9%** of squared activation energy.

### 1.3 Top-k index vs. a quantized weight layout — **NOT ADDRESSED** `[P]`

This is the most important gap.

- Prox's speed measurements are on **FP16** weights. Stage 2 explicitly says: *"computes the selected channels with the **original up and gate weights**"* and *"using the original **full-precision** model weights"* `[P §4.1]`. The only INT4 in the system is the Stage-1 *proxy*.
- The paper's **Figure 6** is a **perplexity** plot: *"Perplexity of quantized Qwen3-8B across FFN sparsity levels"* for AWQ / FP8 / W4A16 / W8A8, WikiText. It shows Prox has the lowest PPL across all quantization configs and sparsities, **but it is an accuracy plot, not a speed or byte plot.**
- The paper's own sentence `[P §5.1]`:
  > *"Combining low-bit weights with activation sparsity can further reduce weight traffic and improve inference speed, but **realizing these gains requires specialized kernels that jointly support sparse execution and low-bit arithmetic, which we leave to future work.**"*
- **Searches over the full text return zero hits for "byte"/"bytes", zero for "group size", zero for "contiguous", zero for "pack".** There is no discussion of group-wise quantization layouts, no per-group byte accounting, and no DRAM traffic measurement of any kind.

**My inference (not in the paper) — why this matters for a 24GB Apple target:**

`W_up, W_gate ∈ R^{d_model × d_ff}` are stored row-major with quantization groups along the **output-channel (d_ff)** axis (this is the llama.cpp/GPTQ/AWQ convention: for a `[K, N]` matrix, each row of N weights is split into groups of `g` consecutive N-channels with one scale per group). Prox selects `k = (1−s₂)·d_ff` **scattered** output channels.

- Number of groups actually touched per matrix row, for a uniformly-scattered selection: `(N/g)·(1 − (1 − k/N)^g)`.
- Qwen3-8B numbers (`d_ff = 12288`): with `k/N = 0.3` and `g = 256` (Q4_K superblock) → `48·(1 − 0.7^256) ≈ 48` → **every group is touched**. With `g = 32` (Q4_K sub-block) → `384·(1 − 0.7^32) ≈ 384` → **every group is touched**.
- ⇒ In a row-major group-quantized layout, Prox's up/gate stage-2 sparsity yields **≈0% weight-byte savings**, only MAC savings. (It *does* yield byte savings on `W_down`, because there the sparsity is over the *row* dimension of `[d_ff, d_model]`, so whole contiguous rows of `W_down` are skipped.)
- Corroborating arithmetic: if up/gate bytes are unchanged and down bytes fall to (1−s₂)=0.233, plus a Stage-1 proxy read of ≈0.3×0.25 of the FP16 up+gate bytes, total bytes ≈ (2.0 + 0.233 + 0.075)/3 = **0.77 of dense** → a bandwidth-bound ceiling of only **~1.30×**, not 1.99×. To reach 1.99× the up/gate reads *must* also be reduced, which requires either (a) clustering the selected channels, (b) a gathered/compacted intermediate buffer, or (c) a transposed group layout. **The paper does not say which, and does not report bytes.**
- Corollary: Prox's own FP16 kernels have the same issue. Selecting scattered columns of an FP16 `W_up` means stride-`d_ff` accesses; a mask-aware kernel that reads contiguous 128-B chunks and masks inside the loop will read *all* the bytes and only cut MACs. Whether the A6000 (768 GB/s HBM, 16 GB Qwen3-8B FP16 → bandwidth-bound decode) 1.99× came from bytes or from MACs-plus-occupancy **cannot be determined from the paper**.

### 1.4 FLOPs vs. real bytes `[P]`

- **FLOPs/MACs: yes, quantified and modelled.** `e = s₂ − 2α(1−s₁)/3` `[P §4.2]`.
- **Bytes: not quantified at all.** No byte counts, no DRAM traffic, no memory-footprint measurement other than the "12% weight-storage overhead" caveat for the proxy `[P §5.3]`.
- The **12% proxy overhead is a bytes cost that is not in the cost model** (α is a *latency* ratio, not a byte ratio) — so the paper's own headline "effective sparsity" metric is blind to the 12% memory it adds. My arithmetic: 0.30·(0.25·(W_up+W_gate) + (W_up+W_gate)) + 0.233·W_down ≈ 0.075 + 0.233 = 0.308 of dense FFN bytes, i.e. **~1.7× fewer FFN bytes** in the best case; but ~12% of *total model* bytes are permanently resident for the proxy, and `W_down`'s (1−s₂) bytes are the only clean win.

### 1.5 Prefill vs. decode `[P]`

- **Decode only.** Kernels are *"tailored for single-batch autoregressive decoding"* `[P §4.3]`.
- **Prox is single-batch by design and says so** `[P §5.3]`:
  > *"Prox currently targets single-batch autoregressive decoding … Consequently, its primary limitation is the **lack of support for large-batch serving**. Extending proxy-guided sparsification and its underlying sparse kernels to efficient batched inference remains a promising avenue for future exploration."*
- For **log-likelihood** tasks the whole sequence is sparsified (*"we sparsify the full sequence to ensure hidden states are computed under sparsity"*); for **generation** tasks (GSM8K) *"we maintain a dense prompt prefill and sparsify decoding exclusively"* `[P §5]`. So prefill is dense in the generation setting, and fully sparse in the likelihood setting. Batch size is never stated to be > 1.

### 1.6 Frozen-weight applicability — **fully frozen** `[P]`

- **No training, no fine-tuning, no retrained gate.** Title: *"Training-Free"*.
- Requires **offline calibration only**: layer-wise magnitude thresholds from the **Alpaca training split** — *"the first 10 contiguous sequences of 2,048 tokens"*, i.e. **20,480 token positions** per configuration; 10 models × 4 sparsity levels = 40 threshold configurations; a 200k-sample reservoir per layer and a saved 4,096-bin histogram per layer; thresholds from the empirical quantile `[P A.7]`.
- Uses **sequential layer-wise calibration** so that each layer sees the *sparsified* inputs of all previous layers (Algorithm 1) `[P A.7]`.
- Requires **producing an extra INT4 quantized copy of `W_up` and `W_gate`** (12% storage overhead) `[P §4.3, §5.3]`.
- Calibration is per-model and per-sparsity-level → 40 configurations to ship per model family.

### 1.7 Test setup `[P §5, §5.1, A.5]`

- **Models (10 from 6 families):** Qwen3-1.7B / 4B / 8B / 14B, Qwen3.5-4B / 9B, Ministral-3.3B, Mistral-7B, Llama-3-8B, Gemma-3-12B (GeGLU, to test beyond SwiGLU). Qwen3-32B appears only in the *config* table (A.1) and the α-cost table, not in the accuracy table.
- **Hardware:** NVIDIA **A6000** (primary), **A100**, **RTX 3090**. **No Apple, no Metal, no MLX, no CPU.**
- **Batch size:** 1 (single-batch decoding). **Context length:** not stated for the speed runs; 16K/32K for the sparse-attention table.
- **Precision:** FP16 for all speed measurements; AWQ / FP8 / W4A16 / W8A8 only for the Figure-6 **perplexity** study.
- **Baseline:** dense FP16 (the reported "1.00×"), plus training-free baselines **CATS**, **M-COUNTDOWN**, **TEAL**.
- **Quality benchmark:** EleutherAI LM Harness, 6 tasks — 8-shot GSM8K, 5-shot MMLU, 10-shot HellaSwag, 0-shot ARC-Easy, 25-shot ARC-Challenge, 0-shot PIQA; aggregate score reported.
- **Sparse-attention backends:** RocketKV, HShare.

### 1.8 Real numbers `[P]`

**Speed (A6000, single-batch decode):** Prox **1.51–1.99×** at 60–70% sparsity; headline **1.99× end-to-end at 70% FFN sparsity**. At 70% sparsity Prox differs from TEAL in throughput by **≤2.9%** on all models while improving average downstream score by **+14.4 points**.

Throughput table (A.5, Table 7, tok/s): Qwen3-4B A6000 dense 82.0 → 135.2 at 70% (1.65×); Qwen3-8B A6000 dense 46.7 → 85.0 at 70% (1.82×); Qwen3-8B A100 dense 93.5 → 133.5 at 70% (1.43×); Qwen3-8B RTX 3090 dense 54.7 → (Prox) at 70%.

**Quality (Table 1, aggregate downstream score; higher better):**
| Model | Dense | Prox 40% | Prox 50% | Prox 60% | Prox 70% | TEAL 70% | CATS 60% | COUNTDOWN 60% |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 61.4 | 61.5 | 59.5 | 59.3 | 56.5 | 41.8 | 27.3 | 33.7 |
| Qwen3-4B | 71.1 | 70.8 | 68.8 | 66.4 | 58.7 | 52.8 | 26.3 | 45.0 |
| Qwen3-8B | 76.1 | 75.7 | 74.6 | 73.3 | 68.6 | 63.3 | 27.6 | 56.8 |
| Qwen3-14B | 77.9 | 77.6 | 76.6 | 76.1 | 74.8 | 70.3 | 31.9 | 66.7 |
| Mistral-7B | 71.1 | 69.3 | 67.7 | 67.0 | 67.5 | 59.4 | 26.7 | 50.0 |
| Llama-3-8B | 70.1 | 69.7 | 70.1 | 67.2 | 63.4 | 57.1 | 26.7 | 44.2 |
| Gemma-3-12B | 77.9 | 77.6 | 77.5 | 77.5 | 74.8 | 62.0 | 32.9 | 54.0 |

**The paper's own negative results (these are the ones to quote):**
1. **At 70% sparsity Prox still loses real quality** on most models: Qwen3-8B **76.1 → 68.6** (−7.5), Qwen3-4B **71.1 → 58.7** (−12.4), Qwen3-1.7B 61.4 → 56.5 (−4.9). The 70% point is *not* quality-neutral; it is "best available", and 60% is the practical sweet spot (Qwen3-8B 76.1 → 73.3, −2.8).
2. **Proxy precision is a trap (Ablation A1):** replacing the INT4 proxy with FP16 proxy weights makes things **worse** — "performance deficits reaching **9.0 and 4.7 points** at 70% sparsity on Qwen3-8B and Qwen3-14B" — because the extra compute must be reallocated out of Stage 2. So you *cannot* buy quality by making the selector more accurate; you must make it cheaper. Directly relevant to anyone tempted to replace Prox's proxy with a better probe.
3. **Exact Stage 2 is non-negotiable (Ablation A3):** feeding proxy values straight into the down projection costs **24.3 points** (Qwen3-8B) and **18.1 points** (Qwen3-14B) at 70% sparsity.
4. **Fixed stage-1 sparsity is worse (Ablation A2):** fixing s₁=0.8 loses up to **4.4 points** (Qwen3-8B).
5. **No large-batch support** `[P §5.3]`.
6. **12% extra weight storage** `[P §5.3]`.
7. **Quantization × sparsity speed is explicitly future work** `[P §5.1]`.
8. **Prefill under sparsity is avoided** for generation tasks (dense prefill) `[P §5]`.
9. **Apple/Metal/MLX: not addressed at all.**
10. **Stage-2 allocations are clamped, so the advertised sparsity is not uniform:** at the 40% and 70% operating points s₁ is clamped to 0% / 70% and s₂ becomes **62.2% / 76.7%** `[P A.4, Table 6]`. At 70% effective sparsity the *actual* Stage-2 channel sparsity is 76.7%, and Stage 1 is at its maximum input sparsity (70%) — i.e. the proxy is running at its least faithful setting right at the headline operating point. Consistent with the +14.4-point quality gap vs TEAL at 70% being partly a *different* sparsity profile, not only a better selector.

**Sparse-attention composition (Table 2, Qwen3-8B, 60% FFN sparsity):** Dense 33.1 tok/s @16K → RocketKV+Prox 63.6 (1.92×), 23.9 → 62.5 (2.62×) @32K. LongBench/RULER 100.0/100.0 dense → 96.6/98.0 with RocketKV+Prox; 100.0/100.0 → 92.5/100.0 with HShare+Prox. **A 3.4-point LongBench drop at 16K with RocketKV+Prox.**

### 1.9 2026 status `[O]`

- **arXiv:** v1 only, 30 Jul 2026, cs.LG, CC BY 4.0. No v2 as of 2026-10-21.
- **OpenReview:** live **ICLR 2027 Conference Submission**, forum `8UJzSYuF9s`, submission id `ICLR.cc/2027/Conference/Submission11912`, `venue: "ICLR 2027 Conference Submission"`, bibtex note *"under review"*, primary area *"foundation or frontier models, including LLMs"*. `[O]` openreview.net/forum?id=8UJzSYuF9s
  - **The OpenReview version is a later revision with a different title and better numbers:** title *"Prox: Training-Free FFN Activation Sparsification in LLMs via Approximate Intermediate-State Salience"* (arXiv: "Sparsity via Approximate Intermediate-Channel Salience"); abstract says **2.07×** end-to-end decoding speedup at 70% FFN sparsity vs arXiv v1's **1.99×**. `[O]`
  - **No public reviews yet** — the note's invitations include only `-/Submission`, `-/Edit`, `-/Submission_Change_Before_Bidding`, `-/Full_Submission`, `-/Submission_Change_Before_Reviewing`; no `-/Official_Review` notes found. So **no independent criticism of Prox is publicly available as of 2026-10-21.** `[O]`
- **No code repo.** The arXiv HTML contains no author GitHub/HF link (only LaTeXML boilerplate and a llama-3 model card link). GitHub repository search for `prox ffn activation sparsity LLM`, `SpQt zigzag quantization sparsity`, `zigzag quantization llm` → **0 results**. `[O]`
- **No HuggingFace papers page** (`huggingface.co/papers/2607.27591` → 404). `[O]` ⇒ no community artifact, no votes, no public discussion, no reproduced numbers.
- **Not merged into vLLM / llama.cpp / MLX.** llama.cpp's 2026 "sparse" work is entirely **sparse attention / sparse MoE** (e.g. `#29639 vulkan: sparse flash attention for quantized K/V`, `#30370 ggml-cuda: Add HIP sparse fattn`, `#25971 Support for Sparse MoE models`) — none of it is FFN activation sparsity. `[O]`

---

## 2. "Enabling Dynamic Sparsity in Quantized LLM Inference" (SpQt) — arXiv:2511.04477 `[P]`

Sources opened: `https://arxiv.org/abs/2511.04477`; full LaTeXML HTML `https://arxiv.org/html/2511.04477v1`; **PDF `https://arxiv.org/pdf/2511.04477`** (14 pages) read pages 3–9, because the HTML renders Tables 1–3 as glyph-path SVG (no extractable text) and Table 2/3 numbers were taken from the PDF. Tables 1–3 values below are therefore **[P] PDF-verified**.

### 2.1 Exact skip condition

**SpQt does not do neuron/channel selection. It does "activation sparsity" = magnitude thresholding of the hidden state on the INPUT side of every GEMV.** `[P §2]`

> *"Formally, any hidden state entry satisfying |x_i| < threshold can be treated as zero, allowing the corresponding **column W[:,i]** in the weight matrix to be skipped."* `[P §2, p.3]`

Concretely, for **every** GEMV except the output embedding `[P §3.2]`:
1. A **sparse index collecting kernel** scans the hidden state, and entries with `|x_i| < τ` are dropped; the surviving indices are written into a compact array `idx[0:n_ns]`.
2. The **zigzag GEMV kernel** then computes only the surviving columns.

Mechanism of the index collector `[P §4.2.3]`: a **Blelloch parallel prefix-sum** over the binary mask to compute output positions, then a scatter of active indices, with a **global atomic counter** reserving offsets in the output buffer so the index array stays contiguous and coalesced.

**Granularity:** **per-token, per-matrix, per-hidden-state-entry** (input channels). It is *not* FFN-specific: it applies to attention and FFN matrices alike, and it follows the **TEAL** sparsity pattern (input sparsity on `x` for up/gate, and on `s` for down). SpQt's "customized TEAL mode" reuses TEAL's greedy per-matrix sparsity pattern but averages it into **two shared thresholds**: one for `{W_q, W_k, W_v}` and one for `{W_ffn-up, W_ffn-gate}` — explicitly to *"reduce the overhead of sparse index collection"* `[P §5]`.

### 2.2 Selector / proxy cost — **negligible and fully accounted** `[P]`

- **There is no learned selector.** No probe, no gate weight, no router, no auxiliary low-rank predictor. The paper's own criticism of prior work `[P §2]`: *"Prior work typically employs auxiliary low-rank predictors trained per layer to forecast which neurons will be activated, but these add nontrivial overhead and complicate deployment."* SpQt's answer: *"a threshold can be calibrated once using calibration data, eliminating the need for layer-specific predictors."*
- **Cost = one O(n) threshold scan per GEMV**, i.e. a prefix-sum over the hidden-state vector (d_model or d_ff elements), plus writing `n_ns` int indices. That is O(n) work against O(n·m) weight bytes ⇒ **negligible** (order 10⁻³ of the GEMV).
- Calibrated **offline**: a ~100-line Python tool on top of **TEAL's released calibration results**, with two modes (unified sparsity; customized TEAL). `[P §5]`
- The paper does **not** give an explicit numeric cost for the index-collection kernel, but it is architecturally negligible and is *the* mechanism that makes the method work (it converts a per-weight branch into a per-superblock skip).

### 2.3 Top-k index vs. quantized layout — **this IS the paper's core contribution** `[P §4.1.2, §4.2.1]`

This is the single most valuable section of the paper for our assignment, because it *names and measures* the exact failure mode the assignment is worried about.

**The problem, stated by the authors:**
- K-quantization (llama.cpp Q4_K) packs **256 consecutive weights into one superblock**, split into eight 32-weight blocks, with 4-bit weights + 6-bit block scales + 16-bit superblock scale/min. `[P §2, §4.1.1]`
- Superblocks are organized **row-wise**, which *"directly conflicts with the column-wise sparsity pattern"*: a zeroed hidden state `x_i` corresponds to **column i** of `W`, and sparse columns *"become scattered across multiple quantization blocks"*. `[P §2]`
- Skipping arbitrary weights inside a block *"introduces branch conditions, leading to severe warp divergence and degraded performance."* `[P §2]`
- The paper's own measurement: *"Empirical experiment results show that **only skip the entire superblock is practical.**"* `[P §4.1.2]`

**The solution — "Zigzag" layout** `[P §4.1.2]`:
> *"**Column-based grouping:** We group weights along the column dimension, so that if an input hidden state entry is sparse, **all superblocks corresponding to that column can be skipped as a unit**. This transforms irregular element-level skipping into **structured block-level skipping**."*
> *"**Row-major superblock storage:** Although grouping is column-wise, superblocks are stored in row-major order. This ensures that the output GEMV results corresponding to consecutive output neurons remain contiguous in memory."*

In Q4_K row grouping, a superblock = 256 columns of 1 row. In zigzag, a superblock = **256 rows of 1 column** (128 bytes of Q4 weights + scales ≈ 3–4 cache lines). Skipping input channel `c` skips a whole superblock ⇒ **real byte savings at ~128-byte granularity**, which is DRAM-efficient. The paper is explicit that the naïve alternative — re-grouping blocks column-wise without re-ordering storage — *"disrupts memory locality because all the threadgroups working on different columns must write back to the final result tensor, resulting in excessive memory write overhead."* `[P §2]`

**Kernel consequence** `[P §4.2.1–4.2.3]`: the default Q4_K kernel launches `m/4` threadgroups each with 32 threads (4 rows each); with zigzag, row-wise parallelism drops from `m/4` to `m/256`, so SpQt introduces a **hybrid scheme**: per superblock-row, launch `n₁` threadgroups × `n₂` simdgroups, each keeping partial sums in registers/local memory, reducing only at the end of the row (`tg=32, sg=2` found best empirically). A **load-balancing** pass then partitions the *runtime active index array* evenly across simdgroups, so no simdgroup waits on stragglers.

**The negative measurement that proves the problem is fatal:** the "Naive sparse (original weight layout)" baseline (B1) — fine-grained branching inside the default row-major layout — achieves **less than 0.2× speedup** (i.e. a ~5× *slowdown*) at 50% sparsity, across all tested GEMV shapes. `[P §6.3, Fig. 9]`

### 2.4 FLOPs vs. real bytes `[P]`

- SpQt **does reduce real bytes**, structurally: skipping a column-superblock means the corresponding quantized weights are never fetched. Unlike a FLOP-only scheme, the memory traffic is the thing being saved, which is what matters on a bandwidth-bound Apple GPU.
- **However, SpQt never reports a byte count either.** It reports latency and GFLOPS. So the byte reduction is *argued structurally* (and demonstrated indirectly by the 1.51–1.78× sparse-GEMV speedup, which would be impossible without reduced traffic on a bandwidth-bound device), not tabulated. `[P]`
- **Cost of the layout change:** the model must be **re-quantized into the zigzag layout by a custom tool** — SpQt extends llama.cpp's quantization tool, and the resulting GGUF is **not loadable by stock llama.cpp / MLX / HF** without SpQt's fork. One-time per model; *"quantizing a 70B model takes less than five minutes on a commodity workstation."* `[P §5]`

### 2.5 Prefill vs. decode `[P §5, "Operation details"]`

- **Decode-only, by explicit design.** *"During the prefill stage, the system executes dense kernels rather than sparse ones."* Two reasons given: (1) *"the hidden states across different prompt tokens exhibit low sparsity when aggregated, as different tokens trigger different sets of neurons"*; (2) *"prior work (TEAL) has shown that applying sparse computation during prefill can degrade overall model quality."*
- Runtime dispatch: *"the system automatically dispatches dense kernels when the token count is greater than one, and sparse kernels during decoding."* ⇒ **any batch size > 1 (including prefill of a batch) silently falls back to dense.** This is also why SpQt's dense kernel is *"not yet"* a tiled GEMM: *"The dense kernel currently extends from our GEMV implementation and therefore does not yet employ a dedicated tiling-based GEMM optimization."*
- Expected to support *"beam search decoding with small beam sizes."*

### 2.6 Frozen-weight applicability — **fully frozen** `[P]`

- **No training, no fine-tuning, no retrained gate, no predictor.** The method is explicitly *"training-free, quantized LLM decoding."* `[P §6.2]`
- Requires: (a) an **offline re-quantization** of the model into the zigzag layout (custom llama.cpp quant tool); (b) an **offline calibration** of thresholds, reusing **TEAL's** calibration artifacts; (c) a **hardware profile** of the target device (*"SpQt then profiles each new hardware platform with a set of GEMV shapes and its kernel to determine the optimal hyperparameter configuration"*, §3.2).
- Under review, not peer-reviewed at time of writing (the PDF carries *"Preliminary work. Under review."* and an ICML style-violation warning).

### 2.7 Test setup `[P §6.1, Table 1]`

- **Hardware (Table 1):** A19 Pro (5-core GPU, 1.6 GHz, 75.8 GB/s, ~5 W), M2 (8-core, 1.4 GHz, 102.4 GB/s, ~20 W), M2 Pro (19-core, 1.4 GHz, 204.8 GB/s, ~45 W), M2 Max (30-core, 1.4 GHz, 819.2 GB/s, ~80 W).
- **Models:** Llama-2 **7B / 13B / 70B** and Llama-3 **8B**. **No 20–30B model.** Llama-3-70B excluded (known Q4 quality issue).
- **Precision:** **Q4_K** for all GEMV matrices except token/output embedding.
- **Batch / context:** single-token decoding, **~10-token prompt**, greedy decoding, per-token latency/throughput reported. **No long-context or multi-token-batch evaluation.**
- **Accuracy metric:** **wikitext.test perplexity only** (no downstream task suite).
- **Baselines:** K-quant original (Q4_K, dense reference); Naive sparse (original layout); Ours dense (zigzag, no sparsity); Ours sparse without load balancing; Ours sparse full. Sparsity levels 25/40/50%.
- **Software:** ~**8K SLOC of C/C++** on top of **llama.cpp b5711**, kernels in **Metal Shading Language**. CUDA *"anticipated"* but not done.

### 2.8 Real numbers `[P, PDF Tables 2 & 3]`

**Table 2 — end-to-end decoding.** ⚠️ **Caption/data inconsistency (flagged):** the caption reads *"End-to-end decoding latency (ms/token) and speedup over dense Q4_K"*, but the numbers are **self-consistent only as throughput in tokens/s**: for every row, the parenthesised factor equals the ratio of the two listed numbers (e.g. M2 Max Llama-2-7B: F16 24.43 / Q4_K 57.96 = **0.42×**; A19 Pro Llama-2-7B: Unified-50% 22.86 / Q4_K 14.78 = **1.547 ≈ 1.55×**). If the values were ms/token, every "speedup" would be a slowdown. Read them as **tok/s**, and treat the caption as mislabeled.

| Hardware | Setting | Llama-2 7B | Llama-2 13B | Llama-2 70B | Llama-3 8B |
|---|---|---|---|---|---|
| A19 Pro | Baseline-F16 | OOM | – | – | OOM |
| | Baseline-Q4K | 14.78 (1.00×) | – | – | 13.24 (1.00×) |
| | TEAL 50% | 22.79 (1.54×) | – | – | 19.30 (1.46×) |
| | **Unified 50%** | **22.86 (1.55×)** | – | – | 19.29 (1.46×) |
| M2 | Baseline-Q4K | 21.98 (1.00×) | 11.55 (1.00×) | – | 20.03 (1.00×) |
| | **Unified 50%** | 30.69 (1.40×) | 16.82 (1.46×) | – | 27.78 (1.39×) |
| M2 Pro | Baseline-Q4K | 37.29 (1.00×) | 20.52 (1.00×) | – | 34.64 (1.00×) |
| | **Unified 50%** | 47.17 (1.26×) | 27.00 (1.32×) | – | 43.80 (1.26×) |
| M2 Max | Baseline-F16 | 24.43 (0.42×) | 12.99 (0.38×) | OOM | 22.04 (0.40×) |
| | Baseline-Q4K | 57.96 (1.00×) | 34.26 (1.00×) | 7.83 (1.00×) | 55.20 (1.00×) |
| | TEAL 50% | 68.75 (1.19×) | 42.07 (1.23×) | 11.48 (1.47×) | 64.71 (1.17×) |
| | **Unified 50%** | 68.87 (1.19×) | 42.09 (1.23×) | 11.33 (1.45×) | 65.07 (1.18×) |

⚠️ **Note the "Baseline-F16" row: on M2 Max, F16 is 0.38–0.42× of Q4_K (i.e. ~2.4–2.6× *slower*)** — quantification is worth far more than sparsity on these chips.

**Table 3 — wikitext perplexity under Q4_K:**

| Method | Llama-2 7B | 13B | 70B | Llama-3 8B |
|---|---|---|---|---|
| Baseline-F16 | 5.13 | 4.62 | 3.27 | 5.64 |
| Baseline-Q4K | 5.28 | 4.69 | 3.35 | 5.93 |
| TEAL 25% / Unified 25% | 5.26 / 5.27 | 4.68 / 4.69 | 3.41 / 3.41 | 6.00 / 5.94 |
| TEAL 40% / Unified 40% | 5.39 / 5.40 | 4.77 / 4.79 | 3.50 / 3.55 | 6.17 / 6.42 |
| TEAL 50% / Unified 50% | 5.59 / 5.65 | 4.90 / 4.97 | 3.70 / 3.77 | 6.61 / 6.67 |
| TEAL 65% / Unified 65% | 6.78 / **7.39** | 5.58 / 5.96 | 4.45 / 4.48 | 8.92 / 9.59 |

**The paper's own negative results:**
1. **65% sparsity is a cliff**: Llama-2-7B 5.28 → **7.39** (+40% PPL), Llama-3-8B 5.93 → **9.59** (+62%). The paper's own summary: *"higher sparsity (65%) shows noticeable quality loss."* Usable ceiling is ~50%.
2. **Quantization already costs more than sparsity buys at 50%**: Llama-3-8B Q4_K 5.93 → 6.67 (Unified 50%) vs F16 5.64. Sparsity costs more PPL than Q4_K does.
3. **The "Naive sparse" baseline is a 5× slowdown** (<0.2×) — the empirical proof that scattered skips inside row-major groups are catastrophic.
4. **Zigzag layout without load balancing (B2) yields almost nothing**: only **1.02–1.06×** at 50% sparsity, because *"load imbalance offsets most of the potential time savings."* ⇒ the layout alone is not enough; the index-collection + dynamic work partition is what delivers the 1.51–1.78×.
5. **Low sparsity gives nothing.** At 25% on M2 Max, TEAL = **0.99×** (i.e. *slower* than dense Q4_K); on M2 Pro, Unified 25% on Llama-3-8B = **0.99×**. At M2 Max Llama-2-13B, TEAL 25% = 1.00×.
6. **Prefill is dense and the dense path is unoptimized** — no prefill speedup claimed; explicitly deferred to future work.
7. **The weight file is not llama.cpp-compatible** — a custom re-quantization tool is mandatory. No stock-GGUF support.
8. **No 20–30B model, no long context, no batch > 1, wikitext-only quality metric.**

**Micro-benchmarks** `[P §6.3]`: dense zigzag GEMV reaches **up to 17% lower latency** than the default Q4_K kernel with the right (tg, sg) config; sparse GEMV speedup **1.22× @25%, 1.51× @40%, 1.78× @50%**, largest gains at large K.

### 2.9 2026 status `[O]`

- **No public repository.** GitHub repo search for `SpQt`, `spqt+llama`, `spqt+metal`, `spqt+sparsity`, `zigzag quantization llm` → **0 relevant results** (only unrelated "spqt" usernames). The arXiv HTML contains **no** author repo link (only `github.com/ggerganov/llama.cpp` and the k-quants PR #1684, which is the *original* K-quant merge from 2023-06-05, not SpQt). `[O]`
- **No HuggingFace papers page** (`huggingface.co/papers/2511.04477` → 404). `[O]` ⇒ no public artifact, no votes, no community discussion, no reproduction.
- **No OpenReview record found** under its title, "SpQt", or the authors' names (only unrelated fuzzy matches). The PDF says *"Preliminary work. Under review."* — so it is under submission, not accepted, as far as is publicly visible. `[O]`
- **Not merged into llama.cpp.** llama.cpp's 2026 sparse work is all sparse **attention**/sparse **MoE** (see §1.9). The zigzag layout is a **fork-only** artifact. `[O]`
- **No MLX implementation**, no vLLM implementation found. `[O]`
- **Not superseded by a newer arXiv work I could find** — see §3; the closest 2026 neighbours (WiSparse, SVD contextual sparsity predictors) do not supersede it.

---

## 3. Adjacent / nearby research (named, with mechanism and relevance)

All arXiv IDs below were resolved by querying the arXiv API on 2026-10-21 and reading titles+abstracts. `[P]` for abstracts read directly from the arXiv API; `[S]` where only publisher metadata was opened.

**Training-free FFN/activation sparsity (the direct lineage of both papers)**
| Work | arXiv / venue | Mechanism | Frozen? | Why it matters here |
|---|---|---|---|---|
| **TEAL** — *Training-Free Activation Sparsity in Large Language Models* (Liu et al.) | **arXiv:2408.14690** (v3) `[P]` | Training-free **activation sparsity on hidden-state entries** across the whole network (input sparsity on `x` for up/gate, on `s` for down). Greedy per-matrix sparsity search. | Yes (calibration only) | **The single most important predecessor.** Both Prox and SpQt build on it: Prox uses TEAL as its *quality* baseline and characterises TEAL's error-accumulation weakness; **SpQt literally reuses TEAL's released calibration artifacts and its greedy sparsity pattern.** TEAL is also the source of the "sparsity in prefill degrades quality" finding that SpQt cites. |
| **CATS** — *Contextually-Aware Thresholding for Sparsity in LLMs* (Lee et al.) | **arXiv:2404.08763** (v4) `[P]` | Thresholding on the densely-computed **gate** activation `h` → output sparsity on up, input sparsity on down. | Yes | One of Prox's three baselines. Prox's Table 1 shows CATS **collapsing** at 60% (Qwen3-8B 76.1 → 27.6) — the clearest published evidence that single-branch channel selection fails at high sparsity. |
| **M-COUNTDOWN** — *Contextually Sparse Activation Filtering Out Unnecessary Weights in Down Projection* (Cheon & Kang) | **arXiv:2505.17701** (v3) `[P]` | Thresholding on the **up**-branch activation `u` → output sparsity on gate, input sparsity on down. Also offers a *trained* low-rank-predictor variant (D-COUNTDOWN). | Both (Prox compares against the training-free M- variant) | Prox's second baseline; also the source of the *weighted-sum view* of the FFN (`FFN(x) = Σ_i s_i W_down[i,:]`) that Prox's oracle analysis is built on. Prox shows M-COUNTDOWN also collapses at 60%. |
| **R-Sparse** — *Rank-Aware Activation Sparsity for Efficient LLM Inference* (Zhang et al.) | **arXiv:2504.19449** `[P]` | Input-sparsity pattern (like TEAL) plus **low-rank compensation** across linear layers. | Yes-ish (low-rank compensation) | Cited by both Prox and SpQt as the "similar pattern + compensation" variant. SpQt: *"R-Sparse … focus on dense weight layouts and full-precision kernels, leaving integration with quantized models unexplored."* |
| **WiSparse** — *Weight-Aware Mixed Activation Sparsity* | **arXiv:2602.14452** (Feb 2026) `[P]`; OpenReview: *"Submitted to ICLR 2026"* `[O]` | Training-free activation sparsity with **weight-aware, per-block non-uniform sparsity ratios** (arguing that uniform ratios overlook activation/weight interaction and inter-block sensitivity variation). | Yes | **The closest 2026 successor to Prox's premise** and a direct critique of the *"uniform sparsity ratio"* assumption that both Prox and SpQt make (Prox: one s₁/s₂ per layer; SpQt: one threshold per matrix group). **[UNVERIFIED]** — I read only the abstract; I did not open the full text and do not report its numbers. |
| **DAWS** — *Dual Activation-Weight Sparsity: A Training-Free Framework for Efficient LLM Model Compression* (Sun et al.) | **ACL 2026 (Long), pp. 8350–8366**, doi 10.18653/v1/2026.acl-long.378 `[O]` (OpenReview/DBLP record) | Training-free, jointly exploits activation **and** weight sparsity via a **three-tier magnitude-based routing**; observes activation energy is concentrated in a few neurons and that activation/weight sparsity patterns are **complementary between attention and FFN**. | Yes | Published, peer-reviewed, training-free, and it explicitly claims the complementarity insight. Worth reading for the 24GB goal. `[UNVERIFIED]` beyond the abstract. |
| **Dynamic Input Pruning and Cache-Aware Masking** ("Loki"-style) | **arXiv:2412.01380** (v2) `[P]` | Dynamic **input** pruning for SwiGLU (where ReLU exact zeros no longer exist) + KV-cache masking; explicitly framed around *"reducing effective DRAM bandwidth per token"*. | Yes | The closest **byte-level / bandwidth-motivated** formulation of input sparsity for SwiGLU; a useful third reference point for "does the method target bytes or FLOPs". |
| **Universal Properties of Activation Sparsity in Modern LLMs** | **arXiv:2509.00454** (v2) `[P]` | Analysis of why exact-zero methods don't transfer to modern LLMs; argues for model-specific fragmentation. | N/A (analysis) | Frames the general problem Prox/TEAL operate in. |
| **SVD Contextual Sparsity Predictors for Fast LLM Inference** | **arXiv:2603.14110** (Mar 2026) `[P]` | SVD-based **contextual-sparsity predictors** for ReGLU FFNs. | Predictor must be trained | The "trained selector" branch that Prox explicitly avoids; useful contrast for the selector-cost question. |
| **Activation Sparsity as Training–Inference Co-Design** | OpenReview `ICLR.cc/2027/Conference/Submission14301` `[O]` | *Requires* pretraining with fixed-threshold nonlinearities; finds sparsity alone doesn't reduce latency unless the kernel can exploit it; identifies FFN hidden activation `h` and attention-output input `z` as the exploitable sites. | **No — needs training** | Out of scope for frozen models, but its central finding is a warning for our goal: *"more than doubling model-wide sparsity from 12.71% to 27.48% **slightly increases latency**"* — i.e. sparsity without kernel support can be worthless. `[UNVERIFIED]` beyond the abstract. |
| **MixA-Q** — *Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective* | OpenReview/CoRR 2025 `[O]` | ViT analogue of the sparsity×quantization interaction. | — | Vision, not LLM. Mentioned only because it is one of the very few works that couples activation sparsity with mixed-precision quantization at the layout level. `[UNVERIFIED]` |

**Selector / sparsity-prediction for frozen models (the assignment's "skip-gate / selector-prediction" cluster)**
| Work | ID / venue | Mechanism | Selector cost | Note |
|---|---|---|---|---|
| **SparseInfer** — *Training-free Prediction of Activation Sparsity for Fast LLM Inference* | **arXiv:2411.12692** (v2) `[P]`; OpenReview: **DATE 2025** `[O]` | For **ReLU-fied** LLMs, predicts activation sparsity by comparing **only the sign bits of inputs and weights**; adaptive conservativeness knob. | Extremely cheap (sign-bit compare) but **requires a ReLU-fied model** (i.e. fine-tuning) | The canonical "training-free *predictor*" and the cheapest selector in this space. Not applicable to a frozen SwiGLU model, but it is the design that Prox's INT4-proxy is competing with. |
| **GRASP** — *Group-based Prediction of Activation Sparsity for Fast LLM Inference* | **DAC 2025**, doi 10.1109/dac63849.2025.11132899; OpenAlex `W4414198566` `[S]` | Magnitude-based, training-free prediction that **groups values considering the distribution within vectors** and explicitly handles statistical outliers, building on SparseInfer's sign-bit method. Claims higher prediction accuracy and **11% higher skipping** than the SOTA technique. | Group-based ⇒ the group size directly controls both accuracy and overhead | **Directly on point for the assignment's group-size question** — GRASP is the closest published work to "how big should the group be when you gather k of N channels". Not on arXiv; IEEE/DAC paywalled. `[UNVERIFIED]` full text. |
| **A Sparsity Predicting Approach for LLMs via Activation Pattern Clustering** | Springer, doi 10.1007/978-3-031-99854-6_25 (2025) `[S]` | Clustering-based sparsity prediction. | — | Same cluster; `[UNVERIFIED]`. |
| **SkipDecode** — *Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference* | **arXiv:2307.02628** `[P]` | Skips *layers/tokens* via early exit, not FFN channels. | — | Named in the assignment's "SkipDecode-style" framing; it is **not** FFN sparsity and is a different mechanism (early-exit). Worth stating explicitly so it is not conflated. |

**Verification of the "no joint measurement" claim (i.e. who *does* do sparsity × quantization):**
`arXiv` API searches on `abs:(FFN OR feed-forward) AND abs:sparsity AND abs:quantization` and `abs:sparsity AND abs:quantization AND abs:(bandwidth OR "memory traffic")` (2026-10-21) surfaced **no work that jointly measures dynamic-FFN-sparsity end-to-end speed on top of a quantized model with byte accounting.** The nearest items are all *architectural* (not dense-FFN) or require training: *Hardware-Native Joint Sparse-Quantization for Trillion-Scale MoE* (**arXiv:2610.02241**), *SPARQLe* (**arXiv:2606.00365**, sub-precision activations), *SALCA* (**arXiv:2604.24820**, sparse-attention accelerator), *Compression Trinity* (**arXiv:2608.24070**, a thesis). `[P]` abstracts only. `[UNVERIFIED]` full texts.

---

## Joint-measurement check (Prox × 2511.04477)

**Direct answer to the assignment's explicit question: NO. Neither paper reports a joint end-to-end measurement of its dynamic-sparsity method on top of a quantized model.**

| | Prox (2607.27591) | SpQt (2511.04477) |
|---|---|---|
| Sparsity mechanism | Intermediate-channel selection (SwiGLU `s`), thresholded, per-token | Hidden-state input thresholding on every GEMV, per-token |
| Weights used for the **speed** measurement | **FP16** (Stage 2 uses "the original full-precision model weights") | **Q4_K**, in a **custom zigzag layout** |
| Quantization appears in the paper as | **Accuracy only** — Fig. 6, WikiText PPL of Qwen3-8B under AWQ / FP8 / W4A16 / W8A8 × FFN sparsity | The **only** setting (no FP16 speed run except the "Baseline-F16" reference row) |
| Joint **speed** measurement | **None.** The paper says verbatim: *"realizing these gains requires specialized kernels that jointly support sparse execution and low-bit arithmetic, which we leave to future work."* `[P §5.1]` | N/A — SpQt is quantization-native, but its sparsity is **not** Prox's channel selection |
| Joint **byte** measurement | **None** (no byte counts anywhere in the paper) | **None** (byte savings argued structurally, never tabulated) |
| Model overlap | Qwen3 family, Qwen3.5, Ministral, Mistral, Llama-3, Gemma-3 | Llama-2 7B/13B/70B, Llama-3 8B |
| Hardware overlap | A6000 / A100 / RTX 3090 | A19 Pro / M2 / M2 Pro / M2 Max |
| Apple Silicon | **No** | **Yes, native Metal** |

**Why they cannot simply be multiplied, and why that matters for the 24GB goal:**

1. **They solve different problems.** Prox's mask is over *intermediate channels* and needs a proxy FFN to compute; SpQt's mask is over *input entries* and needs only a threshold. Prox's Fig. 2 explicitly frames TEAL (= SpQt's pattern) as the baseline it beats on **quality** at high sparsity, while SpQt frames TEAL as the sparsity pattern it re-implements **efficiently on quantized weights**. They are two answers to the same *question* ("which FFN work can be skipped") with different *skip units*.
2. **The proxy cost and the layout cost are different currencies.** Prox's overhead is *compute + 12% memory*; SpQt's is *one custom weight file*. Neither can be assumed to survive the other's setting.
3. **A 24GB Apple target breaks both papers' headline numbers:**
   - Prox needs the model + **12% extra INT4 proxy weights** resident. On a 24GB budget that is a direct hit, and Prox offers only "future work" (sliding-window proxy eviction) as mitigation.
   - Prox's headline operating point (70% effective sparsity) forces s₁=70% — the proxy's least faithful setting — and s₂=76.7%, with a real quality loss (Qwen3-8B 76.1 → 68.6; Qwen3-4B 71.1 → 58.7).
   - SpQt's usable ceiling is **50%** sparsity, with PPL already +0.37 (Llama-2-7B) / +0.74 (Llama-3-8B) over Q4_K, and its 65% point is a cliff (7.39 / 9.59).
   - SpQt's 1.55× was measured on **Llama-2-7B on A19 Pro** (75.8 GB/s, ~5 W) — the weakest memory bandwidth of the four chips, where sparsity helps most. On M2 Max (819.2 GB/s) the same method gives only **1.19×** on Llama-2-7B. **The speedup shrinks as bandwidth grows** — the exact opposite of what a 24GB Apple Silicon user needs. (My inference from Table 2; note also that F16→Q4_K is itself worth 2.4–2.6× on M2 Max, far more than sparsity.)

---

## Negative results / counter-evidence

Collected from the papers' own text and from the wider literature. These are the claims a cross-reviewer should attack first.

1. **Prox never reports a byte count.** The string "byte"/"bytes" appears **zero times** in the full arXiv HTML of 2607.27591 (verified by exhaustive regex over the extracted text, 2026-10-21). Its sparsity metric is a normalized *MAC-cost* ratio. The single most important question for a bandwidth-bound device — "do you actually read fewer bytes?" — is **not answered by the paper**. `[P]`
2. **Prox's speedups are FP16 + NVIDIA.** No quantized-weight speed run, no Apple/Metal, no MLX, batch size 1 only, and an explicit statement that batched serving is unsupported. `[P §4.3, §5.3]`
3. **Prox's own memory tax (12%) is excluded from its own sparsity metric.** α is a *latency* ratio; the 12% extra weight storage is not in `C_Prox`. `[P §4.2, §5.3]`
4. **Prox's headline 70% point is not quality-neutral** and its Stage-1 allocation is *clamped* at the extremes (s₁=70% means the proxy runs at its worst fidelity exactly at the headline operating point). `[P A.4, Table 1]`
5. **Prox's ablations prove the selector is on a knife-edge:** FP16 proxy is 9.0/4.7 points *worse* (A1); skipping exact Stage 2 is 24.3/18.1 points worse (A3); fixed s₁ loses up to 4.4 points (A2). Any re-implementation that changes the proxy precision, the stage split, or the mask rule will land outside the paper's operating envelope. `[P §5.2]`
6. **SpQt's own baseline is a 5× slowdown** — the naive "skip scattered weights inside row-major groups" approach yields **<0.2× speedup at 50% sparsity**. This is the strongest published empirical statement that *FLOP savings ≠ byte savings* in a group-quantized layout, and it is the paper's own result. `[P §6.3, Fig. 9]`
7. **SpQt's layout without load balancing gains almost nothing** (1.02–1.06× at 50%). Skipping whole superblocks is necessary but not sufficient. `[P §6.3, Fig. 9]`
8. **SpQt at low sparsity is a net loss** (0.99× on M2 Max / M2 Pro at 25%). `[P Table 2]`
9. **SpQt's 65% sparsity is a quality cliff** (Llama-3-8B PPL 5.93 → 9.59). `[P Table 3]`
10. **SpQt's Table 2 caption contradicts its own data** — the caption says "latency (ms/token)" but every parenthesised factor is the *ratio of the two listed numbers*, which only holds if the numbers are tokens/s. Reading the table as ms/token would invert every conclusion. Treat the caption as mislabeled. `[P, PDF p.8]`
11. **SpQt's dense path is not a GEMM.** Prefill is dense and *"does not yet employ a dedicated tiling-based GEMM optimization"* — so no prefill speedup is claimed, and the paper says so. `[P §5]`
12. **Neither method has a released artifact, a public HF page, an upstream merge, or a public peer review.** Prox is ICLR 2027 under review with no reviews posted; SpQt is "Preliminary work. Under review." with no public OpenReview record. ⇒ **there is zero independent reproduction of either method in existence as of 2026-10-21.** `[O]`
13. **Independent negative result on Apple Silicon:** *Collapse or Preserve: Data-Dependent Temporal Aggregation for Spiking Neural Network Acceleration* (**arXiv:2603.13810**, Mar 2026) reports that *"five different sparse computation strategies on **Apple M3 Max** all fail to outperform dense convolution, because SIMD architectures cannot exploit the fine-grained, unstructured sparsity of i.i.d. binary spikes."* `[P]` abstract. Not an LLM/FFN result, but it is a direct, same-hardware, same-class warning: **on Apple Silicon, unstructured sparsity is not automatically a win** — which is precisely why SpQt had to re-lay-out the weights instead of just skipping.
14. **Independent negative result on "sparsity ≠ latency":** the ICLR 2027 submission *Activation Sparsity as Training–Inference Co-Design* reports that at 14M parameters, *"more than doubling model-wide sparsity from 12.71% to 27.48% slightly **increases** latency"* — sparsity without a kernel that can exploit it is worthless. `[O]` abstract.
15. **The bandwidth/quality trade is unfavourable on Apple:** from SpQt's own Table 2/3, at 50% sparsity on Llama-3-8B/M2 Max, PPL goes 5.93 → 6.67 (+12.6% relative) for a 1.18× throughput gain, whereas plain Q4_K from F16 buys 2.4× for 5.93 → 5.93→5.93 vs F16 5.64 (+5.1% relative). **On Apple Silicon, quantizing harder dominates sparsifying.** (My inference, from the paper's own tables; treat as an ordering-of-magnitude argument, not a measurement.)
16. **Open 2026 risks that neither paper addresses:** (a) batch > 1 and speculative decoding both break per-token dynamic masks; (b) the zigzag GGUF is a fork-only artifact with no HF/MLX/llama.cpp compatibility; (c) Prox's per-model, per-sparsity-level calibration (40 configurations for 10 models) is an operational burden; (d) neither paper reports *determinism* guarantees for the dynamic mask.

---

## What I could NOT verify

- **[UNVERIFIED] Prox's OpenReview PDF.** The ICLR 2027 submission exposes `/pdf/6b1154c2…pdf` and a supplementary zip, but the forum API required a challenge-verification token on 2026-10-21. I verified the **metadata** (title, abstract, TLDR, 2.07× figure, venue, forum id) via the OpenReview **search** endpoint, but I did **not** read the OpenReview PDF body. The 2.07× number is therefore **[O]-sourced metadata**, not a figure I inspected. The 1.99× and all the mechanism/appendix details are from the **arXiv v1 HTML** `[P]`.
- **[UNVERIFIED] Whether Prox's 2.07× comes from a different model/hardware/precision than the 1.99×.** The two numbers are from two different documents; the OpenReview abstract does not say what changed. Do not cite 2.07× without checking the OpenReview PDF.
- **[UNVERIFIED] SpQt's Figures 2–10 (all plots).** The LaTeXML HTML embeds them as **glyph-path SVG** with no extractable text, and I read the PDF only for pages 3–9 (text + Tables 1–3). Figures 7–10's per-bar values are therefore unread; I quote only the numbers stated in the surrounding prose (1.22×/1.51×/1.78× GEMV, 1.51–1.78× full design, ≤0.2× for naive sparse, 17% lower dense latency).
- **[UNVERIFIED] GRASP (DAC 2025) full text.** Only OpenAlex metadata + abstract `[S]`. It is IEEE paywalled and not on arXiv. Its group-size and "11% higher skipping" claims are unverified.
- **[UNVERIFIED] WiSparse (arXiv:2602.14452) and DAWS (ACL 2026) full texts.** Abstracts only. I have not verified their numbers, hardware, or byte accounting.
- **[UNVERIFIED] DAWS, MixA-Q, SVD Contextual Sparsity Predictors, "Activation Sparsity as Training–Inference Co-Design", arXiv:2603.13810** — abstract-level only `[P]`/`[O]`.
- **[UNVERIFIED] Semantic Scholar citation counts / venue metadata** for both papers — the API returned HTTP 429 (rate-limited) on 2026-10-21 for both IDs. I could not establish how many papers cite either work, which means **I could not find, and cannot rule out, a third-party reproduction that might not be indexed where I looked.**
- **[UNVERIFIED] Whether SpQt was ever submitted to a venue I could see.** No OpenReview record under its title or the "SpQt" name; the PDF's "Preliminary work. Under review." and ICML-style warning are the only evidence. It may be under review somewhere I could not query.
- **[UNVERIFIED] Exact byte-level behaviour of Prox's own FP16 kernels** for scattered output-channel selection (whether they read contiguous chunks and mask, or gather). The paper describes the masking but not the memory access pattern. My byte arithmetic in §1.3 is therefore an *upper bound on the problem*, not a measurement.
- **[UNVERIFIED] Any MLX or vLLM implementation of *any* of these methods** (GitHub *code* search requires authentication and returned 401; I could only use unauthenticated repository/issue/PR search, which found nothing).
- **Not attempted, by instruction:** no GPU benchmark, no model download, no training, no git operations, and no modification of any pre-existing file in the repository. Only `research/agent-reports/minimax/fulltext-notes/B-dynamic-sparse-ffn.md` was created.

---

## Source list (all opened 2026-10-21)

**Primary (papers)**
- `https://arxiv.org/abs/2607.27591` — Prox, abs page (title, authors, abstract, cs.LG, 30 Jul 2026, v1) `[P]`
- `https://arxiv.org/html/2607.27591v1` — Prox, full LaTeXML HTML incl. appendices A.1–A.11 `[P]`
- `https://arxiv.org/abs/2511.04477` — "Enabling Dynamic Sparsity in Quantized LLM Inference", abs page (Wang, Shu, Lin; cs.DC; 6 Nov 2025) `[P]`
- `https://arxiv.org/html/2511.04477v1` — SpQt, full LaTeXML HTML (all prose; §4.1.2, §4.2.x, §5, §6 prose) `[P]`
- `https://arxiv.org/pdf/2511.04477` — SpQt PDF, pages 3–9 (Tables 1, 2, 3 verbatim; §2 "Conflict between dynamic sparsity and quantization"; §4.1.1–4.1.2; §5 "Operation details"; §6.1–6.3) `[P]`
- arXiv API (`export.arxiv.org/api/query`) for: 2408.14690 (TEAL), 2404.08763 (CATS), 2505.17701 (COUNTDOWN), 2504.19449 (R-Sparse), 2307.02628 (SkipDecode), 2412.01380 (Dynamic Input Pruning), 2411.12692 (SparseInfer), 2509.00454, 2602.14452 (WiSparse), 2603.14110, 2603.13810, 2607.27591, plus targeted `abs:` searches for sparsity×quantization×bandwidth works `[P]`

**Official artifacts**
- `https://api2.openreview.net/notes/search?...` — Prox ICLR 2027 submission `8UJzSYuF9s` (title, abstract, TLDR 2.07×, venue, bibtex, keywords, invitation list) and *Activation Sparsity as Training–Inference Co-Design* `75OCCR3byr`; DAWS ACL 2026 DBLP record `[O]`
- `https://api.github.com/search/repositories?q=...` (SpQt, Prox, zigzag+quantization+llm, spqt+llama/metal/sparsity) — 0 relevant results `[O]`
- `https://api.github.com/repos/ggml-org/llama.cpp/pulls/1684` — the k-quants PR cited by SpQt; merged 2023-06-05, unrelated to SpQt `[O]`
- `https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+sparse` — 42 hits, all sparse **attention**/sparse **MoE** (2026) `[O]`
- `https://huggingface.co/papers/2511.04477` and `https://huggingface.co/papers/2607.27591` — both **404** `[O]`
- `https://api.openalex.org/works?search=Group-based Prediction of Activation Sparsity` — GRASP, DAC 2025, doi 10.1109/dac63849.2025.11132899, abstract `[S]`
