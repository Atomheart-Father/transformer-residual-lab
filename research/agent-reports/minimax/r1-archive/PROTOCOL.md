# Research protocol — high-capability LLM inference on consumer devices

**Date: 2026-10-21.** Literature horizon: 2024 → 2026 (through today). Trace older mature work backward
as needed. The current date is 2026-10-21, so papers from 2025 and 2026 exist that are NOT in your
parametric knowledge. **Never cite from memory for 2025/2026 work — verify by opening the source page.**

## Environment / tool use

- `web_search` returns large artifacts; extract with bounded reads, do not dump whole artifacts.
- `web_fetch` on an absolute URL gets raw text (arxiv abs pages, GitHub READMEs, blog posts).
- Prefer primary sources in this order: arXiv abs/HTML page → official GitHub repo README/source →
  Hugging Face model card → reputable secondary (paper blog, conference page).
- Record the **exact URL** for every claim. If you could not open a page, mark the item `UNVERIFIED`.

## What counts as a "method"

Any concrete, published or shipped mechanism for reducing the cost of running a high-capability LLM
at inference time, on a fixed set of weights, without retraining from scratch — including
architectures that bake savings into the model, and cross-domain algorithmic transfers.

## Required per-method fields (use a table or fixed sub-headings; do not skip fields)

1. **Exact mathematical object** — what is computed, what is replaced by what. Give the actual
   formula/objective where the paper states it (e.g. what the quantized KV is, what the error
   criterion is, what the draft acceptance rule is).
2. **What it actually saves** — be specific and separate: FLOPs, KV/weight memory *capacity*,
   memory *bandwidth*, and *serial rounds* (the count of sequential dependent steps, which is what
   actually governs decode latency on bandwidth-bound hardware). If the paper only reports one of
   these, say which.
3. **Does it change the model function or the sampling distribution?** Answer precisely.
4. **Training requirement** — none / calibration only / PTQ / QAT / fine-tuning / extra draft model /
   requires a model trained specially (e.g. MTP heads, native sparse models). State whether it
   applies to **already-frozen existing weights** (e.g. an off-the-shelf Qwen3 / Llama checkpoint).
5. **Prefill vs Decode vs Long-context vs batch=1** — separate the four.
6. **Experimental setup** — models, hardware, sequence lengths, batch sizes, precision. Note
   explicitly if the paper does not report one of these (a missing baseline is itself a finding).
7. **Claimed vs independently verified** — separate "authors report" from "third party reproduced".
   Name the independent reproduction if one exists.
8. **Failure conditions / quality degradation** — what breaks, when, and how it was measured.
9. **Superseded?** — is there newer work (2025–2026) that beats or replaces it?

## Losslessness taxonomy (use these exact labels, do not invent others)

- **L0 exact arithmetic** — the transformed computation is mathematically identical (up to reassociation
  only). Bitwise or algebraically equal.
- **L1 finite-precision numeric** — equal in exact arithmetic, but differs at finite precision
  (e.g. bf16 rounding, block-wise accumulation order changes the last bits).
- **L2 distribution-preserving** — sampling distribution is provably identical (e.g. a lossless
  transform of the logits, an exact rejection/acceptance scheme).
- **L3 capability-approximate** — training/approximation changes the model function; quality is
  restored only approximately.
- **L4 lossy** — explicitly changes the function/distribution; quality measured empirically.

Never write "lossless" without one of these labels. If a paper claims "lossless", determine *which*
label actually holds and note any gap between the claim and the formal result.

## Priority 1 — combination precedents

Find methods that have been **actually combined** in a single paper or shipped system, not just
methods that could plausibly compose. For each precedent record: which methods, what the
combination is called, whether the authors measured the combination jointly (vs each in isolation),
the joint result, and any statement about interference. Examples to hunt for: KV quantization ×
speculative decoding; MoE offloading × prefetch; KV sharing × compression; attention sparsity ×
memory layout; MTP × low-bit compute; reasoning-length control × reliability.

## Priority 2 — counter-evidence and negative results

Actively look for: replication failures, papers that retract or fail to reproduce others, methods
that only work at large batch size, results that do not hold on small models, on consumer hardware,
or at long context, and claimed speedups that are actually just better kernels. Negative results
are high-value in this task — the user explicitly prefers finding that no viable combination
exists over manufacturing novelty.

## Consumer-device relevance (the project's actual goal)

The target is **consumer hardware** (Apple Silicon / MLX / llama.cpp class): a single device,
limited unified memory (16–64 GB), batch = 1, memory bandwidth on the order of 100–800 GB/s.
Weigh every method against these constraints. A method that only helps at batch 64 on an H100 is
low relevance no matter how good its speedup number is. Note explicitly when a paper's headline
setting is out of scope for the target device.

Anchor fact already established in this repository (Stage 2.1, M4 Pro / MLX): for a 28-layer
0.6B bf16 model, decode is **~90% fixed per-block overhead**; per-step weight traffic is
~881 MB; and 0.558 ms of fixed cost per block dominates a 0.0058 ms/token marginal term. So on
this class of device, decode latency is dominated by **weight streaming**, not FLOPs. Any method
must be judged on whether it reduces bytes moved or the number of sequential dependent passes.

## Output contract

Write a single Markdown file to the path given in your task prompt, using these sections:

```
# <Domain> — evidence notes
## 0. Scope & what I searched
## 1. Method-by-method evidence  (one fixed-format block per method)
## 2. Losslessness ledger  (table)
## 3. Combination precedents actually found  (table)
## 4. Counter-evidence & negative results
## 5. Consumer-device applicability notes
## 6. Gaps / things I could not verify
## 7. Reference list  (numbered, with exact URLs and access date)
```

Reference list format: `[R1] Title — arXiv:XXXX.XXXXX (year) — https://... — verified 2026-10-21`
Mark `UNVERIFIED` when the page could not be opened.

## Style

- Be dense and factual. No filler.
- Quote actual numbers from papers (with the model and setting they came from).
- When a paper's claim is not supported by its own experiments, say so plainly.
- Distinguish "this paper claims" from "this is what the numbers show".
- Do not invent papers, authors, venues, or numbers. If unsure, say so.
- If you find a technical family NOT in your assigned list that looks important, add a section
  `## 8. Out-of-scope discoveries` and describe it briefly.
