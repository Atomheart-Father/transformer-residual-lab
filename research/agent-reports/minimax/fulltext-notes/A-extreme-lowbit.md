# A — Extreme low-bit weight representations (20–30B on 24 GB consumer hardware)

**Agent:** Explore (MiniMax) · **Date all sources opened:** 2026-10-21 · **Scope:** 7 assigned papers only.
**Read-only task:** no models downloaded, no GPU runs, no training, no repo files modified except this report.

## Evidence legend (used on every claim)

| Label | Meaning |
|---|---|
| `[P]` | Primary source actually opened (arXiv abs/HTML/PDF, OpenReview note, official conference page, official repo) |
| `[O]` | Official artifact read directly (HF model card / config, repo source, README, GitHub API object) |
| `[S]` | Secondary source (blog, third-party repo, third-party summary) |
| `[UNVERIFIED]` | Could not open the primary source; what I tried is stated |

### Tooling note (method, for auditability)

No `web_search` tool was available in this turn, and the OpenReview `notes?forum=` endpoint returned HTTP 403 from this
environment. Evidence was therefore gathered with: arXiv **export API** (`export.arxiv.org/api/query`), arXiv **HTML render**
(`arxiv.org/html/<id>v<n>`), arXiv **PDF via `pdftotext`** (piped, no files written into the repo), **OpenReview
`notes/search?term=` API** (validated against known hits), **HuggingFace papers/models/collections APIs**, **GitHub REST API**,
the **official ICML 2026 program page** (`icml.cc/virtual/2026/papers.html?filter=titles`), and `raw.githubusercontent.com`.

---

## 0. Existence summary (read this first)

| # | Name in assignment | Real? | Canonical identity | arXiv id | Official code | Venue |
|---|---|---|---|---|---|---|
| 1 | QTIP | **YES** | *QTIP: Quantization with Trellises and Incoherence Processing* | **2406.11235** (v4, 2025-06-18) | `Cornell-RelaxML/qtip` (GPL-3.0) | NeurIPS 2024 **Spotlight** |
| 2 | Proteus | **YES** | *Proteus: Lookup-Free Trellis-Coded Quantization by Lattice-Breaking Compute Codes for 2-Bit LLMs* | **none** | `YZW-explorer/Proteus` (**placeholder only**) | ICML 2026 (regular) |
| 3 | CurveTQ | **YES** | *CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search* | **2610.09212** (v1, 2026-10-06) | **none found** | arXiv preprint + ICLR 2027 submission (under review) |
| 4 | YAQA | **YES** | *Model-Preserving Adaptive Rounding* (YAQA = "Yet Another Quantization Algorithm") | **2505.22988** (v3, 2026-06-03) | `Cornell-RelaxML/yaqa-quantization` (GPL-3.0) | ICML 2026 (regular) |
| 5 | OSAQA | **NO — NOT FOUND** | — | — | — | — |
| 6 | GSQ | **YES** | *GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling* | **2604.18556** (v2, 2026-05-15) | `IST-DASLab/GSQ` (Apache-2.0) | arXiv preprint |
| 7 | RCO | **YES** | *Model Compression with Exact Budget Constraints via Riemannian Manifolds* | **2605.00649** (v3, 2026-09-27) | `IST-DASLab/RCO` (Apache-2.0) | arXiv preprint |

**Two assignment premises are factually wrong and should be corrected in the parent report:**

1. **"Proteus — GPTQ-family with learned/optimized rotations for ultra-low-bit" is incorrect.** Proteus is a **lookup-free
   trellis-coded quantization (TCQ)** method: a *computed* generator (`MUL-BAL`) that replaces QTIP's lookup table with integer
   arithmetic, **keeping** the rotation and the Euclidean search. `[P]` (CurveTQ abstract + §1, which states: "Proteus (Yang
   et al., 2026) replaces its lookup table by a computed code but keeps the rotation and the Euclidean search.")
2. **"YAQA — ultra-low-bit" is only partly right.** YAQA *is* an adaptive-rounding (GPTQ/LDLQ-family) method, but its headline
   contribution is **minimising end-to-end KL with a Kronecker-factored Hessian sketch**; it is *quantizer-agnostic* and is
   demonstrated at 2/3/4 bits. It is **not** a rotation-learning method (that is SpinQuant/QuaRot territory). `[P]`

---

## 1. QTIP — Quantization with Trellises and Incoherence Processing

**Identity `[P]`**
- Title: *QTIP: Quantization with Trellises and Incoherence Processing*
- Authors: Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa (Cornell University)
- arXiv: **2406.11235** — v1 2024-06-17; **v4 2025-06-18** (opened `arxiv.org/html/2406.11235v4` and `arxiv.org/pdf/2406.11235`)
- Venue: **NeurIPS 2024 Spotlight** (arXiv comment + OpenReview venue string "NeurIPS 2024 spotlight", forum `7sdkLVuYCU`) `[P]`
- Code: `https://github.com/Cornell-RelaxML/qtip` — license **GPL-3.0**, last commit **`e90c6688c8df` (2025-06-22, "Update README.md")**,
  197 stars / 24 forks, default branch `main` `[O]` (GitHub API + raw README)
- Also: production implementation exists as **EXL3 in `turboderp-org/exllamav3`** ("EXL3 … based on QTIP") `[O]` (README of
  exllamav3, opened 2026-10-21).

**The precise mathematical object `[P]`**
- **Encoding**: *Trellis Coded Quantization* (TCQ) — a **stateful** decoder that decouples codebook size from bitrate and
  effective dimension, replacing vector quantization whose codebook grows as 2^(kd). The hardware target is a **"bitshift
  trellis"** with `L=16`, `T_x=T_y=16` so a 16×16 MMA tile can be used.
- **Code spectrum**: pure-LUT → hybrid LUT/computed ("HYB", with `Q=9` tunable LUT bits) → **pure computed, lookup-free**
  ("1MAD", "3INST") random-Gaussian codes, each ≤ 4 instructions per weight. The lookup-free codes are what make it fast.
- **Curvature assumption**: incoherence processing with the **randomized Hadamard transform (RHT)** so weights become
  approximately i.i.d. Gaussian; the **Hessian H** enters through **BlockLDLQ error feedback between coding blocks**
  (not inside the block search). Formally the paper defines μ-incoherence for H and W and conjugates both by random
  orthogonal matrices.
- **Is the rotation applied at inference? YES — and it costs runtime compute.** The RHT must be undone at every decoding step
  (activation side or weight-decode side). `[P]` CurveTQ quantifies this cost at **4–11 % of decode throughput** (see §2/§3).
- **Calibration**: Hessians generated from **6144 sequences × 2048 tokens (Llama 1/2), 4096 × 8192 (Llama 3/3.1), 2048 × 8192
  (405B)**, all sampled from **RedPajama**. `[P]` (§A.3.2)
- **Fine-tuning / QAT required? NO.** "We do not perform fine-tuning since the codes themselves are not tunable, but these codes
  are fully compatible with QuIP#-style fine-tuning." `[P]` (§4.1)

**Bits per weight including overhead `[P]`**
- **QTIP's own paper reports only a nominal "Bits" column (16 / 8 / 4 / 3 / 2) — there is no bpw figure.** Exact string
  searches for `bpw` and `bits per weight` in the arXiv HTML render return **0 hits**; there is no scale/zero-point accounting
  table. This is a genuine reporting gap.
- The only published *with-overhead* accounting I found is from the **competitor** CurveTQ, which measures QTIP's published
  rate as **2.00 bpw at two bits**, and **2.19–2.22 bpw** when a 64-group fp16 scale (+0.25 bpw) is added, **3.16–3.20 bpw**
  at three bits. `[P]` (CurveTQ §4.1). *Interpretation of how the 0.00 vs 0.19–0.22 gap arises is my inference, not stated.*
- The repo's shipping config is the HYB code with `L=16, Q=9, V=2` `[O]`, i.e. the "2-bit" build is **not** 2.00 raw bits/weight.

**Decode complexity and measured speed `[P]`** (QTIP §4.3, Table 4 — **batch size 1**, **RTX 6000 Ada, 960 GB/s**):

| Method | Bits | 2–7B tok/s | 2–70B tok/s |
|---|---|---|---|
| FP16 | 16 | 55.9 | OOM |
| AQLM | 2 | 81.5 | 8.78 |
| QuIP# | 2 | 186 | 22.2 |
| **QTIP** | **2** | **188** | **23.5** |
| QTIP | 3 | 161 | 19.1 |
| QTIP | 4 | 140 | 16.3 |

The same table is reproduced in the repo README `[O]`. **No batch-size sweep, no Apple/Metal, no MLX numbers are published.**

**Quality evidence `[P]`** (QTIP Tables 3 and the Llama-3.1 tables; Wikitext-2/C4, ctx 4096)
- Llama-3.1 **405B** Inst: FP8 (16 attn/8 MLP) W2 1.70; QTIP 4 → 1.79; QTIP 3 → 2.05; QTIP 2 → 3.29.
- Llama-3.1 **70B** Inst: BF16 3.52; QTIP 4 → 3.73; QTIP 3 → 4.12; QTIP 2 → **5.08** (≈ +44 % perplexity vs BF16).
  Zero-shot (ArcC/ArcE/Hswag/PiQA) from the same table: BF16 mean **66.78**, QTIP 2-bit mean **63.55** → **−3.2 points**
  (*arithmetic computed by me from the paper's own four columns*).
- Llama-3.1 8B Inst: PV-Tuning 2.01 → 8.45; QTIP rows 4/3/2 → 8.28 / 8.51 / 8.98 (per the same table layout).
- Benchmarks: WikiText-2 + C4 perplexity, and 4 zero-shot tasks (ArcC, ArcE, HellaSwag, PIQA) via LM-Eval 0.4.4.

**Negative results the paper itself reports `[P]`**
- "the computed-code QTIP variants **without fine-tuning** outperforms both QuIP# and AQLM **with fine-tuning** on almost all
  models and sizes, showing that **fine-tuning is not a silver bullet**." (§4.1)
- At 2 bits the quality drop is large (above), i.e. the speed win comes with a real accuracy cost.

**Frozen weights?** **YES** — pure PTQ on frozen weights, no retraining. `[P]`

**2026 status `[P]`/`[O]`**
- **vLLM: NOT merged.** Issue `vllm-project/vllm#11416` "[Feature]: QTIP Quantization" was opened 2024-12-22 and **closed
  2026-01-10 with `state_reason: not_planned`**. In-thread: a volunteer offered to implement it (2025-04-11); another
  commenter noted "Maybe skip QTIP and go straight to yaqa-quantization? Since these are **GPL-V3** I guess they need to be
  implemented from the papers and not from the code"; a third asked whether an 8× speedup over BF16 is achievable at all.
- **llama.cpp: NOT merged.** No trellis/2-bit-trellis support in mainline `ggml/src/ggml-common.h` (case-insensitive search
  finds `IQ2`/`iq2_xxs` but **no `IQ2_K` and no trellis**) `[O]`; PR `ggml-org/llama.cpp#19726` ("Port IQ*_K quants from
  ik_llama.cpp") was **closed 2026-02-23**.
- **Ollama / MLX:** no hits for QTIP in `ollama/ollama` or `ml-explore/mlx-lm` issue searches `[O]`.
- **Metal/Apple backend: NONE.** Every TCQ decoder in the ecosystem is CUDA/ROCm only (§3 counter-evidence).
- **Newer work that supersedes/pressures it:** CurveTQ (2026-10-06, claims rotation-free + faster), XOR-Trellis (ICLR 2027
  submission), GSQ (2026-04-20, scalar + GGUF-deployable), and the Together AI writeup on QTIP `[S]`.

---

## 2. Proteus — Lookup-Free Trellis-Coded Quantization by Lattice-Breaking Compute Codes for 2-Bit LLMs

**Identity `[P]`**
- Authors: Zhengwu Yang, Xunchao Li, Ke Cheng, Kunlong Liu, Jianfeng Yang, Haoshuang Wang, Kaipeng Deng, Qingqing Dang,
  Yanlin Sha, Yanjun Ma, Dianhai Yu (first author contact `2586753502@qq.com`)
- Venue: **ICML 2026, regular track** — OpenReview forum **`SHYZM21zO7`**, Submission 34250, with a `Camera_Ready_Revision`
  invitation; venueid `ICML.cc/2026/Conference`; BibTeX booktitle "Forty-third International Conference on Machine Learning".
  The title also appears on the **official ICML 2026 program page** (`icml.cc/virtual/2026/papers.html?filter=titles`).
- **arXiv: NONE.** arXiv title/keyword/author searches for "Proteus" + trellis/2-bit/quantization return only two unrelated
  2013/2022 papers ("Proteus: A Hierarchical Portfolio of Solvers and Transformations", "Proteus: A Self-Designing Range
  Filter"). `[P]`
- **Code: `https://github.com/YZW-explorer/Proteus`** — the repo's own OpenReview field `link_to_code` points here `[P]`.
  Repo state as of 2026-10-21 `[O]`: created **2026-05-29**, default branch `main`, **only one file in the whole tree:
  `README.md`**, last commit **`fdf1845d2c1b` (2026-05-29, "Update README.md")**, 3 stars, no license.
  README states verbatim: *"🚧 Status: code release in progress … **This repository is a placeholder** and will be populated
  soon."* It promises CUDA kernels, a **fair batch-size-extended QTIP baseline kernel**, Nsight profiles at batch 1/16/32/64,
  and a 3-stage pipeline.
- **PDF body: `[UNVERIFIED]`** — `openreview.net/pdf?id=SHYZM21zO7` and the camera-ready hash URL both return **HTTP 403 with a
  "Verifying your browser" Cloudflare challenge** from this environment; there is no arXiv mirror and **no PMLR volume for
  ICML 2026 exists yet** (checked `proceedings.mlr.press` index; no "Forty-third International Conference on Machine
  Learning" entry). Everything below therefore comes from the OpenReview **abstract/TL;DR/lay-summary fields** `[P]` and from
  **CurveTQ's re-implementation and description of Proteus** `[P]`.

**The precise mathematical object `[P]`**
- Strictly **lookup-free TCQ**. The computed generator **`MUL-BAL`** uses "cheap integer mixing plus a per-layer affine
  Gaussianizer" to emit "overlap-robust, near-Gaussian code values with **zero runtime table loads**".
- Motivation is stated precisely: incoherence processing improves *global* Gaussianity but "does not guarantee overlap-local
  joint geometry", so overlap-amplified artifacts appear — hence a *lattice-breaking* code rather than a lookup table.
- Three stages (from the repo README `[O]`): **Stage I** layer-wise discrete generator search over a fixed 3-candidate pool +
  **Hessian-weighted Viterbi/BlockLDLQ** TCQ assignment; **Stage II** lightweight per-channel scale compensation (discrete
  structure frozen); **Stage III** *optional* global few-shot distillation that tunes **only per-layer affine statistics** while
  keeping packed indices and the bitshift-trellis decoder fixed.
- **Rotation at inference: YES.** CurveTQ states Proteus "keeps the rotation and the Euclidean search" `[P]`. So Proteus does
  **not** avoid the RHT runtime cost that CurveTQ attacks.
- **Calibration: not stated in the accessible fields** `[UNVERIFIED]`.
- **Fine-tuning: OPTIONAL** (Stage III few-shot distillation of affine statistics only) — the base pipeline is PTQ.
- **Frozen weights?** Mostly yes, with an optional distillation stage.

**Bits per weight `[P]`**
- Proteus's own bpw figure is **not in the accessible abstract** `[UNVERIFIED]`. The abstract says "about two bits each"
  (lay summary). CurveTQ's accounting gives Proteus's **published rate as 2.25 bpw at two bits** (vs QTIP 2.00 bpw), and
  2.19–2.22 bpw when the per-block stored start state is added `[P]`.

**Decode complexity and measured speed `[P]`**
- Proteus abstract: "reduces end-to-end decode bandwidth at comparable throughput (**e.g., 740 vs. 1020 GB/s on 70B**)".
- **CurveTQ's own measurements are the only independent kernel numbers**, and they are *CurveTQ's implementation*, since
  "**Proteus, which has no public kernel, is our implementation throughout**" `[P]`.
  **RTX 3090, batch 1, tok/s** — Llama-2-7B: 2-bit **105.9**, 3-bit 103.5, 4-bit 102.6 (QTIP: 127.0 / 119.0 / 109.0; fp16
  52.5). Qwen3-4B: 2-bit **117.9**, 3-bit 115.4, 4-bit 113.2 (QTIP 153.6 / 144.2 / 137.2; fp16 74.3).
  Batch 64 (L2-7B, 2-bit): Proteus 1038 vs QTIP 1011. Batch 128: 1830 vs 1715.

**Quality evidence `[P]`**
- Proteus claims better perplexity and zero-shot accuracy than strong TCQ/PTQ baselines on **Llama 2 (7B–70B) at 2 bits**
  (abstract). Full tables are in the unreadable PDF `[UNVERIFIED]`.
- Independently, CurveTQ measures **CurveTQ ahead of Proteus by +0.6 to +3.3 downstream points** at matched side information
  and by **+2.5 [+1.9, +3.3] paired points on a 35B MoE** (Qwen3.6-35B-A3B, 30 tasks) at 2.00 bpw `[P]`.

**2026 status `[P]`/`[O]`**
- ICML 2026 accepted; **no code, no kernels, no weights, no arXiv, no PMLR proceedings** as of 2026-10-21 (5 months post-
  acceptance, the repo still contains only a README). CurveTQ (2026-10-06) confirms "Proteus, which has no public kernel".
- Not in vLLM, llama.cpp, Ollama or MLX (no such formats exist there).

---

## 3. CurveTQ — Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search

**Identity `[P]`**
- Authors: Guanhua Ding, Zi Wang, Ruichao Li, Jack Liu (**Black Sesame Technologies**; first two marked equal contribution)
- arXiv: **2610.09212**, **v1 submitted 2026-10-06**, cs.LG (opened via arXiv API + `arxiv.org/pdf/2610.09212`; **the arXiv
  HTML render does not exist (HTTP 404)** for this very recent submission, so I read the PDF text with `pdftotext -layout`)
- Second venue: **ICLR 2027 Conference Submission** (OpenReview forum **`5n4xeaG7bM`**, Submission 11798, `venueid
  ICLR.cc/2027/Conference/Submission`, anonymous "under review") `[P]`
- **Code: none found.** GitHub repository search for "CurveTQ" returns **0 repositories** `[O]`. No HF artifact seen. `[UNVERIFIED]`

**The precise mathematical object `[P]`**
- Claim: existing 2-bit trellis/lattice quantizers (QTIP, Proteus) rotate weights by a random orthogonal transform, encode with
  a **Euclidean** search, and use the Hessian **only** in the error-feedback step between coding blocks — so *part of the
  Hessian is unused*.
- Error feedback turns the loss into a weighted sum of per-coordinate rounding errors whose weights are the **diagonal of the
  Hessian's LDL factorization** (`1/d²_i`). CurveTQ puts those weights into the **Viterbi branch metric**, so the search
  follows curvature *inside* each coding block.
- **Proposition 1**: the gain of the weighted search over the Euclidean one (high-resolution limit) is governed by the
  within-block variation of those weights — large in the native basis, ≈0 after rotation. Hence **"weighting in the native
  basis and rotating are substitutes"**, and the rotation becomes unnecessary.
- **Rotation at inference: NO.** CurveTQ is explicitly **rotation-free**; the decoder needs no transform. It replaces the
  rotation's two side effects directly: a **factored scale field** (int4 second-level scales, cost **0.068 bpw on Qwen3-4B,
  0.067 on Llama-3**, vs 0.25 for fp16 scales and 0.125 for int8) and a **closed-form quantile table** (edge-indexed emission,
  128 entries per module) fitted deterministically from the calibration set.
- **Trellis construction change**: instead of QTIP's tail-biting trellis (which stores no start state), CurveTQ lets each block
  start in any trellis state and **stores a start state per coding block** (14 bits per 64-weight block = **0.219 bpw**).
- **Curvature assumption**: per-layer Hessian (one Hessian per expert for MoE, following QMoE); under top-8/256 routing the
  per-expert Hessian is often rank-deficient and damping-dominated, so the weighted metric stays close to Euclidean there.
- **Calibration**: **128 calibration segments of 2048 tokens from C4** for all six checkpoints; the build is deterministic given
  the calibration set `[P]`.
- **Frozen weights?** **YES** — PTQ, calibration only, no training.

**Bits per weight `[P]`** — CurveTQ **does** report bpw *including* side information, and explicitly says "every rate we report
counts all of this side information, the emission table and the start state (Appendix J)":
- **2.29 bpw at two bits**; **3.29 bpw at three bits**. Breakdown at 2 bits: start state 0.219 bpw + factored scale field
  ≈0.068 bpw + emission table (remainder) ≈ 2.29 bpw total.
- **This is a theoretical/computed rate, not a measured on-disk footprint** `[P]` (no file-size table).
- Ablation ladder (Table 4, Qwen3-4B, 2 bits) shows rate-buying steps: int8 second-level scales save 0.124 bpw for 0.41
  perplexity; a flat int4 grid saves a further 0.062 bpw for 0.58 perplexity and 0.8 points; the factored int4 field + per-module
  table is 1.26 perplexity and 0.8 points better at 0.006 bpw more. End-to-end the recipe saves **0.18 bpw, 1.82 perplexity and
  1.4 [+0.5, +2.4] downstream points**.

**Decode complexity and measured speed `[P]`** — **one RTX 3090**, QTIP's own benchmark setting, batch 1 / 64 / 128.
Table 5 (batch 1, tok/s): Llama-2-7B 2-bit — fp16 52.5, QTIP 127.0, Proteus 105.9, **CurveTQ 139.0**; 3-bit 119.0/103.5/**125.7**;
4-bit 109.0/102.6/**114.9**. Qwen3-4B 2-bit — fp16 74.3, QTIP 153.6, Proteus 117.9, **CurveTQ 160.8**; 3-bit 144.2/115.4/**151.7**;
4-bit 137.2/113.2/**146.2**. Table I.1 (L2-7B, 2-bit): batch 64 → QTIP 1011, Proteus 1038, **CurveTQ 1082**; batch 128 → 1715,
1830, **1927**.
Their harness reproduces QTIP's published fp16 figure on Llama-2-7B (52.25 vs 52.5 tok/s). **Context length 4096** for the
evaluated models. **No Apple/Metal/MLX measurements; no llama.cpp/vLLM runtime.**

**Quality evidence `[P]`** — WikiText-2 perplexity (ctx 4096) + mean accuracy over a fixed suite (4–34 tasks per model, via
lm-evaluation-harness with a **vLLM backend**, LiveBench via its official framework, lmms-eval for multimodal):
- Six checkpoints: **Llama-2-7B, Llama-3-8B, Llama-3.1-8B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-32B-Instruct,
  Qwen3.6-35B-A3B (MoE, 40 MoE layers, 256 experts, top-8)** — "to our knowledge, the first trellis-coded result on such a model".
- At 2.29 bpw CurveTQ has lower perplexity than QTIP and Proteus at 2.00/2.25 bpw on all four Instruct models:
  **+1.0 to +3.3 downstream points** over the strengthened baselines; against Proteus+start state **+0.6 [−0.1, +1.3]** (i.e.
  **not always significant**).
- MoE: at 2.29 bpw **+2.1 [+1.4, +3.0] and +2.5 [+1.9, +3.3]** paired points over the 30 tasks.

**Negative results the paper itself reports `[P]`** (valuable for the parent report)
1. **The rotation is not free**: adding QTIP's activation-side Hadamard to CurveTQ's own decoder costs **5–9 tok/s on
   Llama-2-7B and 13–17 tok/s on Qwen3-4B = 4–11 % of throughput**; "with the rotation added our decoder is about as fast as
   QTIP's (**0.93–1.02×**)". ⇒ *the rotation consumes most or all of CurveTQ's decoding advantage.*
2. **Perplexity and downstream accuracy disagree**: replacing the 1/d² branch metric by the Euclidean one costs **−3.3
   [−4.5, −2.3] downstream points while *lowering* WikiText-2 perplexity by 0.24**.
3. **Removing error feedback**: −0.34 perplexity but **−2.5 [−4.3, −1.4] points**, concentrated on **GSM8K (−27)** and IFEval (−6).
4. **Removing the stored start state**: −5.5 [−7.0, −4.2] / −5.1 [−6.9, −3.5] points and +2.09/+4.10 perplexity; even
   lengthening the tail-biting sequence to QTIP's tile length still leaves it −4.2 points behind.
5. **At 3–4 bits the advantage largely disappears**: with baselines carrying the start state, CurveTQ is "ahead of or level with
   the baseline in twelve of the sixteen comparisons, and all eight paired contrasts lie within ±0.7 points".
6. **Margins concentrate on generative tasks**: +3.6 to +8.4 points on GSM8K/IFEval/LiveBench; over **multiple-choice tasks
   only** it is +1.5/+1.2/+2.2 vs QTIP+start and **+0.2/+0.8/+0.8 vs Proteus+start — within noise**.
7. Adding a 128-group scale on top of the start state "changes little".

**2026 status `[P]`** — 9 days old at time of writing. Under review at ICLR 2027; **no code release**, no kernel artifact, CUDA
only. It is the most current (and the only rotation-free) entry in this family, but **nothing is independently reproducible yet**.

---

## 4. YAQA — Model-Preserving Adaptive Rounding

**Identity `[P]`**
- Title: *Model-Preserving Adaptive Rounding*; method name **YAQA = "Yet Another Quantization Algorithm"**
- Authors: Albert Tseng, Zhaofeng Sun, Christopher De Sa (Cornell CS)
- arXiv: **2505.22988** — v1 2025-05-29; **v3 2026-06-03**, cs.LG; arXiv comment "**ICML 2026**"
- Venue: **ICML 2026 (regular track)** — OpenReview forum **`PKFilPWjMI`**, Submission 7460, `Camera_Ready_Revision` present;
  venueid `ICML.cc/2026/Conference`; also present on the **official ICML 2026 program page** `[P]`.
  (Earlier the same work was **rejected from ICLR 2026** — forum `oGlgHjYKBi`, venue "Submitted to ICLR 2026", invitation
  `ICLR.cc/2026/Conference/Rejected_Submission` `[P]`.)
- Code: `https://github.com/Cornell-RelaxML/yaqa-quantization` — **GPL-3.0**, last commit **`f9508723251a` (2025-06-20,
  "index")**, 97 stars / 7 forks `[O]`. Official weights + "Sketch-B Hessians" in the HF collection
  `relaxml/yaqa-6837d4c8896eb9ceb7cb899e` (9 items: `Llama-3.1-70B-Instruct-{2,3,4}Bit-YAQA-QTIP`,
  `Llama-3.1-{70B,8B}-Instruct-Hessians-2Sided`, `Gemma-3-12B-Instruct-Hessians-2Sided`, `Llama-3.2-{1B,3B}-…`,
  `Llama-2-7B-Hessians-2Sided`) `[O]`.

**The precise mathematical object `[P]`**
- **Encoding objective: adaptive rounding (GPTQ/LDLQ family) minimising the *end-to-end* KL**, not the usual layer-local
  activation proxy. Contribution set: (i) convergence-time characterisation of adaptive rounding via the structure of its
  Hessian approximation; (ii) an **end-to-end error bound** in terms of the cosine similarity of the approximation to the true
  Hessian; (iii) a **near-optimal Kronecker-factored Hessian sketch** `H ≈ H_O ⊗ H_I` with two concrete sketches (**Sketch A** and
  **Sketch B**, computed with modified PyTorch backward passes).
- **Curvature assumption**: a Kronecker-factored approximation with **low incoherence and low rank** for both `H_O` and `H_I`;
  incoherence processing (RHT) is applied to the sketches.
- **Quantizer-agnostic**: demonstrated with the **INT2/INT3/INT4 scalar quantizer** *and* with the **QTIP trellis quantizer**.
  "YAQA is quantizer agnostic and works with standard datatypes such as INT4."
- **Rotation at inference: YES in the main results.** Table 1 = "with incoherence processing, **the QTIP quantizer**, and no
  finetuning"; Table 2 = "with incoherence processing, the **INT4** quantizer"; Appendix A.5 = "Incoherence Processing with the
  Random Hadamard Transform"; Appendix A.1 states the RHT base Hadamard comes from Neil Sloane's site. **Table 6 is the
  no-incoherence-processing control** and YAQA still beats LDLQ there (KL 0.021–0.025 vs LDLQ 0.033).
  ⇒ The paper's phrase "**all while adding no inference overhead**" refers to the *rounding algorithm* adding no runtime cost —
  **the RHT rotation still costs runtime compute**, exactly as CurveTQ quantifies for QTIP. *(This reading is my inference
  from the table captions + Appendix A.5; the paper does not spell the caveat out in one sentence.)*
- **Calibration**: Hessians from **RedPajama v1**; evaluation uses OPTQ WikiText-2/C4 splits at **sequence length 8192**.
  **Main results use 64K sequences** for Sketch B; Appendix A.11 / Table 9 shows **as few as 2K sequences (1 GPU-hour)** already
  reaches SOTA KL for Qwen3-8B at 2 bits.
- **Compute/memory cost of the sketch `[P]`**: YAQA stores `O(n²)` (`H_I`) **plus** `O(m²)` (`H_O`) ⇒ **roughly double LDLQ's
  Hessian storage**; FP32 storage needed for positive-definiteness (**TF32 is not sufficient**); Sketch B costs
  `O(b·t·m·n + b·n²·m + b·m²·n)` on top of a forward+backward pass; Sketch B is empirically slower than Sketch A.
- **Fine-tuning / QAT required? NO.** §4.3 covers *optional* recovery finetuning; main results are "no finetuning".

**Bits per weight `[P]`** — The paper states an explicit with-overhead figure for its INT4 configuration:
"a 16 bit groupwise absmax scale shared across 32 contiguous elements, giving an **effective 4.5 bits per weight**".
For the QTIP-quantizer rows it reports nominal 2/3/4-bit rates (the trellis side information is QTIP's, not counted again).

**Decode complexity and measured speed** — **YAQA publishes no decode-throughput table at all** `[P]`. It is a *rounding*
algorithm, so it inherits the runtime of whichever quantizer it is paired with (QTIP trellis ⇒ RHT + trellis decode; INT4 ⇒
standard kernels). Its only runtime claim is "no inference overhead" over the quantizer it is used with.

**Quality evidence `[P]`**
- Table 9 (Qwen3-8B, **2 bits**, QTIP quantizer, incoherence processing, no finetuning): original 16-bit W2 PPL **9.00**, MMLU
  72.99, GSM8K(4-shot CoT) **84.31**; LDLQ (4K seqs, 1.5 GPU-h) → 10.81, 63.94, **56.31**; **YAQA-B 64K (30 GPU-h)** → 10.16,
  66.75, **66.07**; YAQA-B 2K (1 GPU-h) → 10.33, 65.74, 62.14.
  ⇒ **Even the best 2-bit YAQA loses ~18 points of GSM8K vs FP16** and ~0.8 PPL. This is the paper's own negative evidence.
- INT4, Llama-3.1-8B Instruct: BF16 KL 0 / PPL 6.50 (W2) / 8.02 (C4) / 69.82 avg zero-shot; LDLQ 0.038 / 6.76 / 8.26 /
  67.99; DiscQuant 0.061; **YAQA-A 0.028**, **YAQA-B 0.029** / 6.71–6.72 / 8.21–8.23 / 68.92–69.11.
- INT4, Gemma-3-12B: **Google's Gemma 3 QAT recipe has a *higher* KL to the original model than YAQA** (Google QAT 4.5 bits KL
  0.089 vs YAQA-A 4 bits KL 0.058, YAQA-B 0.056) — a notable claim, but the QAT reference is a *released 4-bit QAT checkpoint*,
  not a re-run QAT.
- Llama-3.1-70B Instruct with the QTIP quantizer (Table 1): 2-bit KL LDLQ 0.302 → YAQA-B **0.266**; 3-bit 0.101 → **0.091**;
  4-bit 0.036 → **0.029**; 8B 2-bit 0.185 → 0.147.

**Frozen weights?** **YES** — PTQ on frozen weights, no retraining (finetuning optional). `[P]`

**2026 status `[P]`/`[O]`** — ICML 2026 accepted; code public but **last touched 2025-06-20** (unchanged for 16 months).
**No runtime integration**: vLLM issue #11416 explicitly suggests "skip QTIP and go straight to yaqa-quantization" but notes the
**GPL-V3** licence means vLLM (Apache-2.0) would have to reimplement from the paper. No llama.cpp / Ollama / MLX support; no
Metal backend. **Newer work that supersedes it on cost**: **FastKron — *FastKron: Efficient Quantization with Kronecker-Factored
Hessians*, arXiv:2608.06291 (v2, 2026-08-06), Johann Birnick et al.**, which states that "BoA uses a large number of sequential
steps, while **YAQA improves over the sequential depth but still has quartic total cost**" and introduces a cubic-time two-sided
variant `[P]` (arXiv API listing + abstract). A related **GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding** (2026-07-28) also
appears in HF's paper index `[S]`.

---

## 5. OSAQA — claimed ICML 2026 — **NOT FOUND / APPEARS NOT TO EXIST**

**Verdict: no paper named OSAQA could be located in any major index. I am recording this as a confirmed negative result.**

Searches performed on 2026-10-21 and their outcomes:

| Index | Query | Result |
|---|---|---|
| OpenReview notes/search API (`term=`) | `OSAQA` | **count = 0** |
| OpenReview notes/search API | `OSA-QA`, `O-SQA`, `osqa` | **count = 0** each |
| OpenReview notes/search API | `one-shot activation quantization` | count = 10000, but the returned notes are unrelated ICLR reviews — **no OSAQA** |
| HuggingFace papers API | `OSAQA` | **0 results** |
| arXiv export API | `all:"OSAQA"`, `ti:"Proteus" AND cat:cs.LG`, `all:"Lattice-Breaking Compute Codes"` etc. | **no matching entry** |
| GitHub repo search | `OSAQA`, `OSAQA quantization LLM` | 1 unrelated 2017 QA-test project; **0 relevant repos** |
| Official ICML 2026 program page (`icml.cc/virtual/2026/papers.html?filter=titles`) | `OSAQA` | **0 title matches** |

**Endpoint validation** (so the zeros are meaningful): the same OpenReview `notes/search?term=` endpoint returned
**129 hits for "QTIP"**, the exact Proteus title for "Proteus lookup-free trellis", and the exact YAQA title for
"Model-Preserving Adaptive Rounding" — so the endpoint does work, and it *does* find both other ICML 2026 papers in this
assignment. `[P]`

**Nearest loosely-related things I did find** (none of them is OSAQA, none is ICML 2026):
- *"INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats"* and *"TWLA: Achieving Ternary Weights and
  Low-Bit Activations for LLMs via Post-Training Quantization"* — both on the ICML 2026 program page `[P]`, but I did not
  investigate them (out of scope).
- *"Tied Trit-Planes"* (arXiv:2608.08910, 2026-08-09) — a 2026 trellis/ternary weight format paper `[P]`.
- The open GitHub repo `lixzhao/osaqa` is a 2017 QA-test project, unrelated to quantization `[O]`.

**Caveat:** a non-indexed, renamed, withdrawn, or ultra-obscure workshop paper called OSAQA cannot be formally ruled out by
index search. But there is **no arXiv id, no repo, no OpenReview forum, and no ICML 2026 program entry** for anything named
OSAQA. **Recommendation for the parent report: drop OSAQA, or replace it with a real, verified paper.**

---

## 6. GSQ — Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling

**Identity `[P]`**
- Authors: Alireza Dadgarnia (ISTA), Soroush Tabesh (ISTA), Mahdi Nikdan (ISTA), **Michael Helcig (ETH Zürich)**, Eldar Kurtić
  (ISTA & Red Hat AI), Maximilian Kleinegger (TU Wien & ISTA), Dan Alistarh (ISTA & Red Hat AI)
- arXiv: **2604.18556** — v1 2026-04-20; **v2 2026-05-15**, cs.CL (opened `arxiv.org/html/2604.18556v2`)
- Venue: **arXiv preprint** (the repo README explicitly labels it "(preprint)"); **not on the ICML 2026 program** `[P]`
- Code: `https://github.com/IST-DASLab/GSQ` — **Apache-2.0**, last commit **`03fc16484c36` (2026-07-05)**, repo pushed
  2026-07-11, **107 stars**, default branch `main` `[O]`
- Official artifacts: HF collection `ISTA-DASLab/gsq` (10 items) including
  `ISTA-DASLab/Llama-3.1-70B-Instruct-2Bit-GSQ`, `-3Bit-GSQ`, `Kimi-K2.5-2Bit-GSQ`, `Kimi-K2.6-2Bit-GSQ`,
  `Qwen3.6-35B-A3B-2Bit-GSQ`, `Qwen3.8-27B-3Bit-GSQ`, and three **GGUF** ones (`Qwen3-4B-GGUF-GSQ`,
  `Qwen3-8B-GGUF-GSQ`, `Qwen3.5-4B-GGUF-GSQ`) `[O]`

**The precise mathematical object `[P]`**
- **Encoding objective: Gumbel-Softmax relaxation of a discrete scalar grid** — *not* GPTQ's rounding, *not* a trellis code.
  Per-weight learnable **logits over the small grid** (`D` of size 2 for ternary mask/sign, 4 for 2-bit, …) plus **per-group
  scales**; a **Gumbel-Softmax sample** produces a soft one-hot, with **temperature τ annealed high→low** and **logit scale κ
  annealed low→high** so gradients flow early and collapse to a hard discrete choice. A custom autograd op **replays the RNG
  state in the backward pass** to give exact gradients through the sample. **Lion** optimizer + cosine LR, bf16/fp32.
  **Hard-rounding at the end** → symmetric, group-wise, b-bit scalar weights on a uniform grid.
- **Pipeline**: GPTQ initialization (or RTN ablation) → block-wise Gumbel-Softmax refinement → **layer offloading** to `meta`
  device (so models bigger than VRAM can be quantized) → reassembly into an HF checkpoint with a **`compressed-tensors`
  quantization config for "vLLM auto-detection"**.
- **Hessian/curvature assumption**: inherited from the GPTQ initializer (second-order, calibration activations); GSQ itself
  optimizes a **layer-wise reconstruction error** `‖f(x; w̃) − f(x; w)‖_F²` s.t. `w̃ ∈ C`. It does **not** use incoherence
  processing / Hadamard rotation ⇒ **no inference-time rotation** `[P]`.
- **Calibration + optimization budget**: **FineWeb-Edu** (Llama), **OpenThoughts** (Kimi); **4096 sequences of length 4096**;
  **20 epochs for Llama, 10 epochs for Kimi**; scale-only fine-tuning for the 2-bit Llama runs; higher bit-widths use a
  **local-shift** formulation to keep overhead small. ⇒ **This is gradient-based optimization of the quantization variables, not
  "one-shot PTQ"**; it is still *post-training* (no model retraining).
- **Requires fine-tuning/QAT of the model? NO** — but it does require a multi-epoch optimization stage per block.

**Bits per weight including overhead `[P]`/`[O]`** — group size **128**; these are *with* the 16-bit per-group scale:
- uniform **2-bit = 2.13 bpp** (official model card: "produced with **GSQ** … at **≈2.13 bpp**") `[O]`;
  uniform **3-bit = 3.13 bpp**; non-uniform allocations **2.37** and **2.62 bpp**; **ternary = 1.58 bpp** `[P]`.
- These are **computed/analytic rates** (the paper reports them as "avg bit/param"), and the GGUF artifacts are genuinely
  K-Quant files whose on-disk size can be measured directly `[O]`.

**Decode complexity and measured speed `[P]`** — Table 2: **Llama-3.1-70B-Instruct on NVIDIA L40s GPUs, vLLM + "Humming"
kernels, ShareGPT (short conversational) workload, TPS normalised per GPU**:

| Method | Avg bit/param | TPS/GPU | Speedup |
|---|---|---|---|
| BF16 (4 GPU) | 16.00 | 60.3 | 1.00× |
| Uniform 3-bit | 3.13 | 289.6 | 4.80× |
| Non-uniform | 2.62 | 301.1 | 4.99× |
| Non-uniform | 2.37 | 329.2 | 5.46× |
| **Uniform 2-bit** | **2.13** | **374.0** | **6.20×** |

**No Apple Silicon / Metal / MLX measurement anywhere in the paper** (hardware is NVIDIA L40s) `[P]`. The mechanism of the win is
explicitly that a scalar format "can directly leverage highly optimized low-precision GEMM kernels", i.e. it inherits llama.cpp /
Ollama-class kernel support — which is exactly the property trellis formats lack.

**Quality evidence `[P]`/`[O]`** — official README table (paper-table data), zero-shot average over ARC-C/E, HellaSwag, PIQA,
Winogrande:

| Method | Llama-3.1-8B | Llama-3.1-70B |
|---|---|---|
| FP16 | 73.71 | 78.99 |
| GPTQ | 37.53 | 57.38 |
| QuIP | 39.20 | 61.57 |
| EfficientQAT | 63.79 | 71.43 |
| **QTIP (VQ)** | **69.88** | **77.25** |
| PV-Tuning (VQ) | 69.83 | 76.27 |
| **GSQ (ours)** | 68.55 | 75.57 |

⇒ **GSQ is still ~1.3 points below QTIP at 8B and ~1.7 below at 70B, and 3.4 points below FP16 at 70B.** Its claim is to close
*most* of the gap, not all of it.
- GGUF K-Quant refinement: starting from public Unsloth checkpoints, Qwen3-8B Q2_K average over AIME25 / GPQA-Diamond /
  MMLU-Pro improves **50.03 → 56.28** `[O]`.
- Kimi-K2.5 (1T MoE) at 2-bit on non-shared expert weights: "remains close on AIME25, improves on LiveCodeBench-v6 and MATH500,
  but **drops on GPQA Diamond**" `[P]`.

**Negative results / degradations the paper itself reports `[P]`**
- **Calibration-domain damage**: the Kimi calibration set (OpenThoughts) is math/code-skewed, so **GPQA Diamond degrades**;
  the authors attribute this to calibration diversity and show a more diverse calibration mixture shrinks the gap (Appendix D).
- **Limitations (Appendix F)**: (i) auxiliary logit memory grows with the number of candidate grid values — temporary, but it
  "makes full-model end-to-end optimization impractical for large LLMs" and motivates block-wise/expert-wise/local-shift
  procedures; (ii) **depends on a good initialization** (warm-start from GPTQ or an existing GGUF checkpoint) and the local-shift
  formulation "assumes that most useful assignment changes remain near the initialized grid point"; (iii) results do not cover
  some settings.

**Frozen weights?** The *model* is frozen (no retraining), but quantization is **not** a one-shot closed-form step — it needs a
multi-epoch gradient-based discrete search. `[P]`

**2026 status `[P]`/`[O]`** — **This is the only paper in the set with a working, Apple-Silicon-relevant deployment path today.**
- The GGUF artifacts are "**bit-for-bit standard GGUF K-Quant**", a "drop-in replacement for the Unsloth files", and the model
  card documents `llama-cli -m Qwen3-8B-Q2_K.gguf` and `ollama run hf.co/ISTA-DASLab/Qwen3-8B-GGUF-GSQ:Q2_K` — i.e. they run
  unchanged on **llama.cpp / Ollama / LM Studio**, which is the Metal path on Apple Silicon `[O]`.
- The non-GGUF artifacts use a `compressed-tensors` config "for vLLM auto-detection" `[O]`; **I did not verify vLLM actually
  loads these** `[UNVERIFIED]`.
- GSQ also **uses RCO** (§7) to search per-layer bit assignments, tying the two papers together `[P]`.
- No Metal-specific or Apple-Silicon-specific benchmark exists in the paper.

---

## 7. RCO — Model Compression with Exact Budget Constraints via Riemannian Manifolds

**Identity `[P]`**
- Authors: **Michael Helcig (ETH Zürich, corresponding)**, **Dan Alistarh (ISTA)**
- arXiv: **2605.00649** — v1 2026-05-01; **v3 2026-09-27**, cs.LG (opened `arxiv.org/html/2605.00649v3`)
- Venue: **arXiv preprint** — not on the ICML 2026 program `[P]`
- Code: `https://github.com/IST-DASLab/RCO` — **Apache-2.0**, last commit **`9a1e09c07d46` (2026-06-13, "fix remove weight
  normalization")**, 54 stars / 7 forks `[O]`

**The precise mathematical object `[P]`**
- This is **not a weight format** — it is a **budget-constrained discrete-assignment optimizer** for model compression.
- Under softmax relaxation `p_i = softmax(α_i)`, the budget constraint `C(α) = Σ_i w_i Σ_k p_ik c_k = B` defines a **smooth
  Riemannian manifold** in logit space (codim 1). The **normal vector is closed-form**
  `(∂C/∂α)_ik = w_i p_ik (c_k − E_{p_i}[c])`; **shifting logits along the cost vector changes expected cost monotonically**;
  **vector transport reduces to a single inner product**.
- **RCO** = Adam + (i) tangent projection, (ii) **retraction by shifting α along c and binary-searching t so that C(α+tc) = B**
  (monotone ⇒ O(log) to machine precision), (iii) momentum transport. **Gumbel-STE** for the discrete forward pass, plus an
  **exact multiple-choice-knapsack (MCKP) DP solver** for discrete feasibility ⇒ the budget holds **exactly at every iterate**,
  with **no constraint-related hyperparameters**; multiple simultaneous budgets need no extra coefficients.
- **LLM applications `[P]`**:
  1. **Mixed-precision quantization**: groups = linear layers, options = bitwidths in **{2,…,8}** with `c_k = k`, `w_i` = layer
     parameter count, `B` = total-bit budget. Weights are assembled **via GPTQ residuals** (i.e. RCO sits on top of a per-layer,
     per-bitwidth **GPTQ database**, group size 128, asymmetric, **no Hadamard**) — so RCO is a *bit-allocation* method, not a
     codec.
  2. **MoE expert pruning**: groups = experts, options = keep/prune; **router weights are frozen** and pruned experts are masked
     at inference exactly as during optimization ("no approximation gap between the STE training objective and deployment").
- **Calibration / compute `[P]`**: the objective is the **KL divergence to the FP16 model on a calibration set**; optimization
  needs **forward + backward passes** (STE), so it is *not* a zero-compute frozen-weight operation. Protocol in the baselines
  reproduction: **FineWeb-Edu, 256 calibration samples**; MxMoE used 64×2048 tokens. **Hardware: RTX A6000** for matched
  wall-clock vs EvoPress; **4× A100** (batch 1, 4096-token sequences) for GPU/DP timing; **pure-Python DP** costs
  **270–1783 ms per call** (8.3 % of a 3000 ms step for Qwen3-8B); one Qwen3-Coder-Next step (N = 24 576 experts, g = 4)
  ≈ **47 s**.

**Bits per weight `[P]`** — RCO targets **average bitwidths of 2.25 / 2.5 / 3.5 / 4.0 bits** per parameter across layers
(mixed-precision); the discrete solver's grid rounding is visible in the paper (e.g. "2.251 vs. 2.25 bits").

**Decode complexity and measured speed `[P]`** — RCO measures **search** cost, not decode throughput:
- **Qwen3-8B mixed-precision**: at 2.25 bits, RCO (g=16, T=200) FW PPL **20.47** in **215 min**; RCO (g=32, T=50) 20.66 in
  117 min; RCO (g=4, T=200) 20.60 in 62 min; **EvoPress (100 generations) 21.40 in 11–14 h**; IMPQ 24.18; **HIGGS 32.07**.
  At 2.5 bits RCO **15.43** vs EvoPress **15.64** at ~6× lower cost. FP16 reference: FW = 10.96, C4 = 17.20.
  ⇒ "**~10× lower wall-clock**" than EvoPress at 2.25 bits, "~6×" at 2.5 bits.
- **MoE expert pruning**: OLMoE-1B-7B at 25 % sparsity — RCO at 50 steps (~10 min) scores **0.581**, beating EvoESAP's
  **0.597** (searched 5.8 h); at 300 steps RCO reaches **0.608** (+2.7 points, 5× faster).
- **Qwen3-Coder-Next** (80B3A, 512 experts/layer) expert pruning; **Qwen1.5-MoE-A2.7B** mixed-precision (24 layers, 61
  experts/layer, bitwidths 2–8): at 2.5 and 3.5 bits RCO beats MxMoE on **every** aggregate metric; at 3.5 bit RCO matches
  uniform 4-bit quality.

**Negative results / limitations the paper itself reports `[P]`**
- "**By 3.5–4.0 bits surrogate methods already solve the problem well** (HIGGS within 3 % of RCO), and RCO's edge is
  wall-clock time." ⇒ the advantage is confined to the extreme-low-bit regime.
- "the **additive-cost retraction leaves non-additive costs such as fused-operator latency to costlier root-finding**."
- The DP solver is a **pure-Python** implementation ("no compiled version was measured").
- **Calibration domain matters**: coding calibration preserves code generation but hurts general knowledge, and vice versa
  (Appendix C.4).

**Frozen weights?** The model weights are not retrained, but the **search requires gradient optimization** (forward+backward,
Gumbel-STE) — an offline cost of 62 min – 11–14 h on datacenter GPUs. `[P]`

**2026 status `[P]`/`[O]`** — Apache-2.0 (vLLM-compatible licence), 54 stars, last commit 2026-06-13. **No runtime format, no
Apple/Metal/MLX presence** (0 mentions of Apple/Metal/MLX in the paper text). It is already **consumed by GSQ** for non-uniform
per-layer bit allocation, which is the natural production use.

---

## 8. Cross-cutting 2026 deployment status (Apple Silicon / 24 GB specifically)

| Paper | Format deployable in llama.cpp? | vLLM? | MLX / Metal backend? | Kernel artifact public? |
|---|---|---|---|---|
| QTIP | **No** (trellis format absent from mainline) | **No** (issue #11416 closed `not_planned`) | **No** | Yes, **CUDA-only** (`exllamav3` EXL3) |
| Proteus | No | No | No | **No** (repo is a README) |
| CurveTQ | No | No | No | **No** (no repo at all) |
| YAQA | No (2-bit models are QTIP-format) | No | No | Yes, **CUDA-only** (`qtip` repo kernels) |
| **GSQ** | **YES** — GGUF K-Quant Q2_K/Q3_K_M, "runs unchanged on llama.cpp / Ollama / LM Studio" `[O]` | `compressed-tensors` config, vLLM auto-detection *claimed*, **not verified** `[UNVERIFIED]` | **Not measured** (no Apple numbers) | Yes (Apache-2.0) |
| RCO | N/A (allocator) | N/A | No | Yes (Apache-2.0) |
| OSAQA | — | — | — | **paper does not appear to exist** |

**No Metal / Apple Silicon / MLX measurement exists in any of the seven papers** `[P]` — verified by literal string search for
`Metal`, `Apple`, `MLX`, `llama.cpp`, `vLLM` in the QTIP, YAQA, CurveTQ and RCO full texts, and by reading GSQ's hardware
section (NVIDIA L40s only).

---

## 9. Counter-evidence and negative results

**The single most important counter-evidence finding:** a working third-party port of a QTIP-style 2-bit trellis format into
**llama.cpp** is **smaller but slower and worse** than plain 4-bit K-quant, and its decode lookup table *increased* memory
traffic rather than reducing it.

> `YanissAmz/escha-port` — "a QTIP-style 2-bit format in llama.cpp", reference model **Qwen3.8-27B at 2.469 bits per weight**
> (GGUF conversion, CPU reference decoder, CUDA and HIP kernels). Head-to-head against Q4_K_XL on one RTX 3090:
> **generation 34.19 t/s vs 41.40 t/s**, **pp512 610.20 t/s vs 1382.86 t/s**, **perplexity (wikitext-2-raw, 580 chunks) 7.3679 vs
> 6.9744**, in exchange for **9.66 GiB vs 16.35 GiB** on disk. The author's own summary: *"against a real Q4_K_XL this format
> wins on size and loses on everything else."* And: *"the decode lookup table, the obvious optimisation, was generating **13×
> more L2 traffic than the weights it was meant to save**."* `[O]` (repo README, opened 2026-10-21)
> Note also: this is **not** an Apple measurement (RTX 3090 + gfx1151 APU), and it is a *third-party* port of a *commercial
> format* (Escha Labs / ESCHAM), not of QTIP itself — but it is the closest existing end-to-end llama.cpp evidence for this class
> of format. `[S]` for the ESCHAM↔QTIP equivalence, `[O]` for the numbers as written.

Supporting counter-evidence, all primary:

1. **The QTIP authors' own admission, quoted by GSQ**: "Tseng et al. observed that, although VQ and trellis methods yield major
   memory savings, **they lead to only small decoding speedups vs BF16 due to format complexity**." `[P]` (GSQ §1)
2. **Every trellis decoder in the ecosystem is CUDA-only**: `Azimml/trellis-webgpu` — "every TCQ decoder shipped to date is
   **CUDA-only**, which locks this quality tier to NVIDIA data-center and desktop GPUs"; it also *reproduces* QTIP's published
   rate–distortion (MSE **0.0739** vs QTIP's reported **0.0733**) while running an 8B model in a browser on WebGPU. `[O]`
3. **vLLM declined QTIP**: issue #11416 closed **not_planned** on 2026-01-10; licence (GPL-3.0) blocks vendoring; an in-thread
   question about whether an **8× speedup over BF16 is even achievable** was never answered. `[P]`
4. **llama.cpp 2-bit ecosystem friction**: PR #19726 (porting IQ*_K quants) was closed on 2026-02-23, with the maintainer
   stating *"I cannot review, let alone merge any code written by Iwan Kawrakow unless and until the conflict between him and
   Georgi Gerganov has been resolved."* `[P]`
5. **The rotation is the hidden tax**: CurveTQ measures that adding QTIP's activation-side Hadamard to its own decoder costs
   **4–11 % of throughput** and reduces it to **0.93–1.02× of QTIP** — i.e. most of the "rotation-free" win is really the
   *removal of the rotation*, and the rotation itself is not free. `[P]`
6. **Perplexity and accuracy disagree** (so bpw-optimised tables can mislead): swapping CurveTQ's 1/d² branch metric for the
   Euclidean one **lowers** WikiText-2 perplexity by 0.24 while costing **−3.3 downstream points**. `[P]`
7. **The trellis advantage shrinks as bits rise**: at 3–4 bits with matched side information, CurveTQ is level with the
   baselines in 12 of 16 comparisons, all contrasts within ±0.7 points. `[P]`
8. **Margins are concentrated in generative tasks**: over multiple-choice tasks only, CurveTQ is +0.2/+0.8/+0.8 vs
   Proteus+start — **within noise**; the +3.6…+8.4 headline comes from GSM8K/IFEval/LiveBench. `[P]`
9. **2-bit quality losses are large and are reported by the papers themselves**: YAQA's own Table 9 — GSM8K **84.31 → 66.07** at
   2 bits (best variant); QTIP's own table — Llama-3.1-70B W2 **3.52 → 5.08** and zero-shot mean **66.78 → 63.55** at 2 bits;
   GSQ's own table — 70B zero-shot **78.99 (FP16) → 75.57 (GSQ 2-bit)**, and GSQ is still **1.7 points below QTIP**. `[P]`/`[O]`
10. **Calibration-domain damage is real and paper-reported**: GSQ's Kimi-K2.5 2-bit **drops on GPQA Diamond** because the
    OpenThoughts calibration set is math/code-skewed. `[P]`
11. **Cost, not just accuracy, is a limitation**: YAQA needs **64K sequences / 30 GPU-hours** for Sketch B (2K / 1 GPU-h is the
    cheap end), needs **≈2× LDLQ's Hessian memory**, requires **FP32** (TF32 insufficient), and its Sketch B cost is *quartic*
    — which is exactly what **FastKron (arXiv:2608.06291, 2026-08-06)** was written to fix. `[P]`
12. **RCO's advantage is confined to the extreme-low-bit regime and needs real GPUs**: at ≥3.5 bits "surrogate methods already
    solve the problem well"; the discrete solver is pure Python (270–1783 ms/call); search costs 62 min – 14 h. `[P]`
13. **Proteus cannot be independently verified at all** — accepted to ICML 2026 but, 5 months on, the repository contains only a
    README, and CurveTQ (Oct 2026) states Proteus "has no public kernel", so *all* published Proteus decode numbers come from
    the authors themselves or from a competitor's re-implementation. `[O]`/`[P]`

---

## 10. What I could NOT verify

1. **Proteus's full paper text.** `openreview.net/pdf?id=SHYZM21zO7` and the camera-ready hash URL both return **HTTP 403 with a
   Cloudflare "Verifying your browser" challenge**; there is no arXiv mirror and **no PMLR volume for ICML 2026 exists yet**
   (`proceedings.mlr.press` has no "Forty-third International Conference on Machine Learning" entry). Consequence: Proteus's
   **own** bpw accounting, per-model perplexity tables, calibration-set size, GPU model, batch sizes and context lengths are
   `[UNVERIFIED]`; the numbers I report for Proteus come from its OpenReview abstract and from **CurveTQ's re-implementation**.
2. **OSAQA** — nothing to verify; five spellings across OpenReview, arXiv, HuggingFace papers, GitHub and the ICML 2026 program
   page all return zero. A non-indexed/renamed/withdrawn work cannot be formally excluded.
3. **CurveTQ code.** No GitHub repository exists ("CurveTQ" → 0 repos); I could not determine whether a repo exists elsewhere
   (e.g. a private/anonymous ICLR link or a HuggingFace space). The arXiv **HTML render does not exist** (404) for 2610.09212;
   I read the PDF text instead, so I have the paper body but not any figures/tables as rendered images (numbers were read from
   `pdftotext -layout` output, which can garble table columns — I cross-checked the key tables against surrounding prose).
4. **vLLM actually loading GSQ's `compressed-tensors` checkpoints** — the paper and repo *claim* "vLLM auto-detection"; I did
   not run or verify vLLM (out of scope: no downloads/runs).
5. **Any Apple Silicon / Metal / MLX numbers** for any of the seven papers — none exist in the sources; and I did not audit
   `exllamav3` source for a Metal target beyond its README (it advertises CUDA/ROCm + AVX2/AVX-512 CPU paths).
6. **FastKron (arXiv:2608.06291)** — only its arXiv API listing + abstract snippet were obtained (the API was returning HTTP 429
   on re-query); I did not open its PDF, so its claims about YAQA's "quartic total cost" are quoted from its abstract only.
7. **QTIP's own bits-per-weight figure** — QTIP does not publish one; the 2.00 / 2.19–2.22 bpw numbers attributed to QTIP are
   CurveTQ's accounting, and *my* explanation of the gap between them is inference.
8. **Proteus's calibration-data size and hardware** — not in any accessible field.
9. **Exact last-commit SHA for `IST-DASLab/GSQ`** was obtained (`03fc16484c36`, 2026-07-05); for the others: QTIP `e90c6688c8df`
   (2025-06-22), YAQA `f9508723251a` (2025-06-20), RCO `9a1e09c07d46` (2026-06-13), Proteus `fdf1845d2c1b` (2026-05-29).
10. **Nothing was run**: no models downloaded, no inference executed, no GPU benchmarked, no code executed. Every number above is
    quoted from a source document or computed by hand from a paper table (the two hand-computed averages are labelled as such).
