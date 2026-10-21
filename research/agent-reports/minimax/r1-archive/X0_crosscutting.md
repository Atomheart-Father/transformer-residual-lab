# Cross-cutting notes — orchestrator's own findings (2026-10-21)

These are findings I verified directly from primary sources while the nine domain
researchers (R1–R9) were running. Everything here has an exact URL and was opened
on 2026-10-21.

---

## 1. The consumer-device constraint model (the spine of the report)

### 1.1 Measured baseline from THIS repository (M4 Pro 24 GB, MLX 0.31.2, Qwen3-0.6B-Base bf16, 28 blocks, batch 1)

Source: `results/stage2_1/PART_A.md` (repo, not external).

| Quantity | Measured value |
|---|---|
| Per-block fixed cost (launch + weight streaming) | **0.558 ms/block** |
| Per-block marginal cost | 0.0058 ms/token |
| Fraction of decode that is fixed overhead | **~90 %** |
| Decode wall clock, 28 blocks, batch 1, 257-token KV | 7.07 ms (lazy) / 6.14–7.15 ms across runs |
| Per-block decode step | 0.253 ms/block |
| BF16 weight bytes per block | 31.5 MB (measured from pinned checkpoint) |
| Total transformer-block weight traffic per decode step | 28 × 31.5 MB = **881 MB** |
| Inferred achieved bandwidth | ~125 GB/s vs ~270 GB/s class machine (~46 % of peak) |
| Independent-block overlap ceiling | **1.46× at k = 4–8**, saturating (A2) |
| "Stack 4 blocks into one (4,out,in) matmul" | bit-identical math, but **29 % SLOWER at n=1**, 1.25× at n=256 (A5) |
| Critical path 28 → 7 (full g4 DAG) | prefill **+5–6 %**, decode **0 %** (best DAG case 3.7 %, inside noise) |

**Consequence (this is the report's central filter).** On this class of device, batch-1
decode latency is set by (a) *bytes of weight streamed* and (b) *number of sequential
dependent passes*, not by FLOPs. Therefore:

- A method that reduces **FLOPs only** is worth ~nothing at batch-1 decode.
- A method that reduces **bytes moved** is worth roughly linearly.
- A method that reduces **serial rounds** (the number of dependent weight-streaming passes
  per output token) is worth roughly linearly, and is the *only* thing that helps
  prefill-and-decode simultaneously.

This is the rubric every method in the report is scored against. It is also why the
repository's own headline negative result (logical depth reduction ⇒ zero wall-clock gain)
is a *device-level* fact rather than a universal one: the depth was reduced but the
**weight bytes per step did not change**.

### 1.2 An under-appreciated term: the output head

For a tied-embedding model the lm_head/embedding matrix is a first-order term in
per-step weight traffic, and it is read *fully* every decode step:

- Qwen3-0.6B: vocab 151936 × hidden 1024 × 2 B (bf16) = **311 MB**, i.e. ~26 % of the
  1.19 GB total per-step traffic, and it is *not* addressed by any of the ten surveyed
  method families.
- Qwen3.8-Flash-Next: vocab **248320** (config verified 2026-10-21) — the output head
  alone is 248320 × 2560 × 2 B ≈ **1.27 GB** per step at bf16, plus a **51.2 B-parameter**
  per-layer n-gram embedding table (≈ 102 GB at bf16, 28.8 GB at IQ4_NL) that is
  *sparsely read, one row per token* and is therefore intended to live on SSD.

**Finding (mine, not in the literature corpus):** the "model-free draft" family
(n-gram / prompt-lookup drafting) has been absorbed into the model weights themselves in
2026 releases, at a storage cost that exceeds the entire transformer stack, and with a
memory-access pattern (sparse random row reads from a disk-resident table) that is
*worse* for a bandwidth-bound device than the thing it is accelerating. This is a
first-order, largely unremarked-in-the-abstracts cost of the technique.

---

## 2. Verified 2026 primary sources (opened 2026-10-21)

### 2.1 Qwen3.8-Flash-Next — the current flagship hybrid (and its surprising config)

- Model card / weights: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
  (created 2026-08-24, 2.53 M downloads, 6360 likes at check time).
- Config verified: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json
  - `model_type: qwen4_exp`, arch `Qwen4ExpForConditionalGeneration` (vision-language).
  - `num_hidden_layers: 48`; `layer_types` = 3 × `linear_attention` + 1 × `full_attention`
    repeating ⇒ **75 % linear attention / 25 % full attention**, i.e. 12 full-attention layers.
  - `full_attention_interval: 4`; `hidden_size 2560`; `num_attention_heads 24`;
    `head_dim 256`; `num_key_value_heads 2` (GQA 12:1); `partial_rotary_factor 0.25`.
  - `num_experts 512`, `num_experts_per_tok 10`, `moe_intermediate_size 640`,
    `shared_expert_intermediate_size 640`.
  - **Sparse-attention indexer**: `indexer_budget 2048`, `indexer_compress_ratio 4`,
    `indexer_head_dim 128`, `indexer_kv_heads 1`, `indexer_n_heads 4` ⇒ a learned
    top-k selector over 2048 compressed KV slots. This is the NSA/DSA-family mechanism
    **natively embedded**, i.e. it is a model change (L3/L4), not a post-hoc method.
  - **Hierarchical compression**: `hc_count 4`, `hc_lowrank 320`.
  - **PLE block**: `ple_layer_ids [2]`, `ple_conv_kernel_size 4`, `ple_embed_dim 2560`.
  - **Native n-gram predictor**: `ngram_size 3`, `ngram_vocab_size_base 20000000`,
    `heads_per_ngram 8`, `make_ngram_vocab_size_divisible_by 128`,
    `split_ngram_parts 128` ⇒ a 3-gram lookup predictor trained into the model.
  - **MTP head**: `mtp: {hybrid: true, num_hidden_layers: 1, layer_types: [full_attention]}`,
    `mtp_num_hidden_layers 1`.
  - `mamba_ssm_dtype float32`; `output_gate_type sigmoid`; `max_position_embeddings 262144`;
    `vocab_size 248320`; `tie_word_embeddings false`.

- Prior generation for comparison (config verified 2026-10-21):
  https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
  → `Qwen3NextForCausalLM`, 48 layers, `full_attention_interval 4`, hidden 2048,
  16 heads, head_dim 256, 2 KV heads, 512 experts top-10, `shared_expert_intermediate_size 512`,
  81.32 B params (HF safetensors index), `max_position_embeddings 262144`,
  `tie_word_embeddings false`, `use_sliding_window false`.

### 2.2 GSQ + RCO — a published *combination* with a bit-width sweep on a reasoning model

Sources opened 2026-10-21:
- Model card with full results table: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- GSQ paper: https://arxiv.org/abs/2604.18556 (Gumbel-Softmax Quantization; Dadgarnia,
  Tabesh, Nikdan, Helcig, Kurtic, Kleinegger, Alistarh) — code https://github.com/IST-DASLab/GSQ
- RCO paper: https://arxiv.org/abs/2605.00649 (Riemannian Constrained Optimization;
  Helcig, Alistarh) — code https://github.com/IST-DASLab/RCO

Method: GSQ learns per-coordinate grid assignments and per-group scales via a Gumbel-Softmax
relaxation (post-training, scalar, GGUF-deployable). RCO then assigns one quantization type
per tensor under a *hard total-size budget*, by reformulating the budget constraint as a
smooth Riemannian manifold in logit space, optimising the task loss directly.

Measured bit-width sweep on Qwen3.8-Flash-Next (llama.cpp, 55 prompts, 11 categories):

| Variant | bpw | Total GB | AIME25 | GPQA-D | LCB v6 | task avg | zero-shot recovery |
|---|---|---|---|---|---|---|---|
| BF16 | 16.00 | 354 | 100.00 | 91.92 | 87.43 | 93.12 | 100.0 % |
| GSQ-RCO Q2_0 | 2.40 | 66.4 | 96.67 | 89.39 | 81.14 | 89.07 | 101.4 % |
| GSQ-RCO IQ2_XS | 2.50 | 68.0 | 96.67 | 87.37 | 83.43 | 89.16 | 100.3 % |
| GSQ-RCO IQ3_XXS | 3.00 | 75.8 | 100.00 | 91.41 | 86.29 | 92.57 | 100.4 % |
| GSQ-RCO IQ3_S | 3.50 | 83.6 | 100.00 | 92.93 | 86.86 | 93.26 | n/a |

**Three findings I will headline in the report:**

1. **Standard benchmarks are blind to the reasoning damage.** At 2.40–2.50 bpw the
   5-task zero-shot average (arc_easy/arc_challenge/hellaswag/winogrande/piqa) goes *up*
   (100.3–101.4 % "recovery") while AIME25 falls 100.00 → 96.67 and LiveCodeBench v6 falls
   87.43 → 81.14. The card itself says the >100 % recoveries "reflect benchmark variance,
   not a model that is better". This is direct, quantitative proof that PPL/MMLU-style
   evaluation cannot certify a quantization for a reasoning-capable model.
2. **Format choice dominates bit-width choice for speed.** Q2_0 (2.40 bpw, *smaller* file)
   is **3.4× the prompt throughput** and **1.9× lower end-to-end latency** than IQ2_XS
   (2.50 bpw), because IQ2_XS uses lookup-table formats whose decode cost varies with
   content: IQ2_XS swings 49.4→94.5 t/s across categories (45.1 t/s spread) while Q2_0
   stays 92.5→94.5 t/s (2.0 t/s spread). **Half a bit per weight is worth less than
   removing a lookup table.** This is the single most transferable finding to a
   consumer device, where the same lookup-table decode cost is paid on Metal.
3. **A hardware file-format constraint silently caps the method.** RCO can only assign a
   quantization type *per tensor*, not per expert, "which GGUF cannot express" — so the
   95 % of searchable weight mass that lives in the routed experts can only be handled
   per layer. The search covers 352 tensors (304 dense + 48 fused expert matrices).
   This is a clean example of "the method is not the constraint; the interchange format is."

Also from that card:
- A **code-specialised variant** keeps 256 of 512 routed experts per layer at 1.89 bpw
  over the original parameters, 58.4 GB total ⇒ **expert pruning × quantization** combination,
  published with reasoning/code benchmarks.
- The n-gram table (51.2 B params) is a *separate shard*, read sparsely one row per token,
  intended to stay memory-mapped on SSD with `-lm mmap --lazy-mode on`. Only shard 1 must
  be resident in VRAM. This is a real, shipped **tiered-storage design** for a
  model-free drafting table.
- A third-party engine (Strata) runs these GGUFs on a single NVIDIA GPU from 8 GB VRAM.

### 2.3 QSpec — quantization × speculative decoding, a published combination

https://arxiv.org/abs/2410.11305 (EMNLP 2025; v1 2024-10-15, v3 2025-10-02;
code https://github.com/hku-netexplo-lab/QSpec). Opened 2026-10-21.

- Mechanism: low-precision **activation–weight joint** quantization for the *drafting* pass,
  high-precision **weight-only** quantization for the *verification* pass; both weights and
  KV cache are reused across the two stages, "enabling near-zero-cost switching without
  retraining or auxiliary models".
- Claim: up to 1.64× speedup vs high-precision baselines "without quality degradation";
  up to 1.55× over SOTA speculative decoding "in batched settings".
- **Taxonomy note (mine):** "without quality degradation" is measured *relative to the
  weight-only-quantized baseline*, not to the original fp16/bf16 model. The verification
  path is itself L3/L4 relative to the unquantized model, so the composite is **L2 relative
  to the quantized target distribution**, not L2 relative to the original model. Papers that
  reuse this phrasing must be checked for which baseline they mean. The paper uses a
  **greedy** acceptance rule (top-1 match) with resampling from the high-quality
  distribution on rejection — which is *not* the exact modified-rejection-sampling rule of
  Leviathan/Chen, so it is L2 only in the greedy limit.

### 2.4 Other combination precedents found by name (to be verified by R6/R2)

- "Memory-Efficient Speculative Decoding with Quantized Draft KV Cache" — NeurIPS 2026,
  https://nips.cc/virtual/2026/156970
- "QuantSpec: Hierarchical KV Cache for Self-Speculative Decoding" — a self-speculative
  method that combines hierarchical KV cache with speculative decoding.
- "Lynx: Progressive Speculative Quantization for accelerating KV Transfer in Long-Context
  Inference" — arXiv 2607.01831.
- Qwen3.8-Flash-Next ships: MTP head **+** native 3-gram predictor **+** sparse indexer
  **+** hybrid attention ⇒ a four-way native combination in one checkpoint.
- `incoai/Qwen3.8-Flash-Next-DFlash2` — a block-diffusion draft model for the same base
  (tags: dflash2, block-diffusion, speculative-decoding, sglang) ⇒ draft-model ×
  diffusion-parallel-decode combination.

### 2.5 Granite 4.x — the hybrid family that IS consumer-sized

IBM's `granite-4.0-h-*` family is tagged `granitemoehybrid` (Mamba-2 + attention + MoE) and
includes consumer-scale members: `granite-4.0-h-350m`, `granite-4.0-h-1b`,
`granite-4.0-h-tiny`, `granite-4.0-micro-base`, plus `granite-4.0-350m/1b`.
Later generations `granite-4.1-*` and `granite-4.2-*` (3b/8b/30b) are tagged plain
`granite` and ship official GGUF + llama.cpp builds with Q2_K…Q8_0.
`granite-4.1-8b-base` cites arXiv 2603.17074.
`ibm-granite/granite-switch-4.1-{3b,8b,30b}` is a *composable* (adapter-based) system
with per-task adapters and an `io_configs/` tree — a different kind of efficiency
(share one base, specialise by adapter) that avoids holding several models in memory.

**This matters for the report:** Qwen3-Next/3.8-Flash-Next at 80 B does not fit a 16–64 GB
device even at 4-bit (≈ 40 GB weights, leaving no room for KV). Granite's 3B/8B hybrids
and Qwen3-0.6B/4B do. So the "hybrid architecture" answer to the consumer question is
available *in principle* at 3–8 B, but the 2026 flagship hybrids are not.

---

## 3. Gaps I identified that no assigned domain covers

1. **Output-head / embedding / vocabulary compression.** For vocab 151936–248320 models the
   output head is 26 %–40 % of per-step weight traffic. No surveyed family addresses it
   except "frozen vocabulary codes" (a blog-level write-up:
   https://www.llms.blog/posts/embeddings-are-overpriced-a-frozen-16-mb-vocabulary-lets-a-46-mb-ternary-gemma-style-model-run-5x-faster-on-cpu)
   and vocabulary trimming (https://aicassandra.com/blogs/slm/slm_tokenizer_vocab_trimming.html).
   Primary-source literature here is thin — itself a finding.
2. **Cascades / model routing** (FrugalGPT, RouteLLM). Objective mismatch: that literature
   optimises *API cost per request*, not on-device latency, and a cascade needs two models
   resident (or a disk/offload tier), which is exactly what a 16 GB device cannot afford.
   Judged **low relevance** for this target, with the exception of "one small model + one
   disk-resident large model".
3. **Tiered storage of the n-gram/draft table** (shipped in Qwen3.8-Flash-Next as a separate
   mmap'd shard) — bridges R2 and R9 and is directly consumer-relevant.
4. **Attention-sink / massive-activation phenomena** as the enabling condition for KV
   eviction — assigned to R3 but should be cross-referenced from R2.

---

---

## 3b. The one measurement that matters most: native MTP on Apple Silicon

Source opened 2026-10-21 (blog, not peer-reviewed — treat as a practitioner measurement,
not a literature claim):
https://vinoth12940.github.io/blog/articles/genai-20260519-local-mtp-speculative-decoding
(2026-05-19). Runtime: https://github.com/youssofal/MTPLX
Model config verified 2026-10-21: https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json

### Hardware arithmetic stated by the author
- M4 Pro 48 GB, **~120 GB/s** memory bandwidth (vs ~936 GB/s on an RTX 3090).
- Dense 27B at Q4 ≈ **17 GB per forward pass** ⇒ 120 GB/s ÷ 17 GB ≈ **7 tok/s ceiling**.
- Measured baseline: ~7 tok/s. ⇒ *the model is exactly at the bandwidth roofline.*

### Measured results on that hardware (batch 1, temp 0.6, top_p 0.95, top_k 20, MTP depth 3)

| Configuration | Decode tok/s | MTP acceptance |
|---|---|---|
| Baseline (llama.cpp, no MTP) | ~7.0 | — |
| llama.cpp `--spec-type draft-mtp --spec-draft-n-max 3` + `--cache-type-k q8_0 --cache-type-v q8_0`, thinking ON | 8.9 | ~84 % |
| same, thinking OFF | 9.5 | ~86 % |
| same + optimized flags | **10.5** | ~86 % |
| MTPLX (MLX-native, custom Metal kernels), depth 3, thinking OFF | **18.3** | per-position ~73 % / 48 % / 32 % |
| MTPLX over LAN | 14.3 | — |

Other numbers: TTFT 27-token prompt 0.8 s → 0.66 s; memory active/peak 16.2 / 18.6 GB;
over 25 verify calls: 39 accepted drafts, 17 corrections, 7 bonus tokens (1.84 tokens/cycle);
"first-32 sliding" 20.1 tok/s. llama.cpp used `--cache-type-k q8_0 --cache-type-v q8_0`
(i.e. **8-bit KV** was already in the loop).

### Why this is the most important data point in the report
On a bandwidth-bound device, speculative decoding is not a FLOPs optimisation. It is a
**weight-bytes-per-output-token** optimisation: the k drafted tokens are all verified in
one weight-streaming pass, so the pass that would have cost `W` bytes per token instead
costs `W` bytes per (a+1) tokens. The mechanism attacks exactly the axis that is binding.

### The author's own counter-evidence about the KV-compression family
The same post evaluates **TurboQuant** (2–4-bit KV cache compression) and concludes:
- the mainline llama.cpp PR is **CPU-only, not merged**; existing forks are **CUDA-focused**;
- "It only compresses KV cache — doesn't help with the **weight bandwidth bottleneck** that
  limits decode speed";
- "**No Metal/Apple Silicon backend exists**."
⇒ **Independent, on-target-hardware evidence that the KV-compression family does not
attack the binding constraint of consumer-device decode.** This is the clearest
counter-evidence I have found against a major method family, and it comes from a
practitioner on the exact hardware class in scope, not from a competitor.

### Also from that post (used with caution)
- "Turning reasoning off gave an immediate ~7 % speed boost **with no quality loss for
  most tasks**." — an *unsupported* quality claim; the user's brief explicitly demands
  scrutiny of reasoning-length control. I flag it as contested, not established.
- "Math-correct sampling is non-negotiable": MTPLX uses Leviathan–Chen probability-ratio
  acceptance **with residual correction**, i.e. it is genuinely **L2 (distribution-preserving)**
  at T>0, whereas greedy-argmax fast paths are not. Direct, on-target confirmation of the
  L2/L3 distinction that the report needs.
- MoE is named as the next frontier: "Qwen 3.6 35B-A3B (only 3B active params) could
  theoretically hit 25–40+ tok/s on the same hardware" — *theoretical*, not measured.

### 3b.1 A first-order cost model for speculative decoding (mine — derived, not a literature claim)

Let `W` = model-weight bytes read in one full forward pass; `W_d` = extra bytes read to
produce the draft; `a` = expected accepted draft tokens per verify cycle.
Then

    cost per output token = (W + W_d) / (a + 1)
    speedup vs baseline   = (a + 1) / (1 + W_d / W)

Substituting the three drafter families:

| Drafter | `W_d / W` | Break-even acceptance `a*` | Needs a specially trained checkpoint? | Extra resident memory |
|---|---|---|---|---|
| Native MTP head (1 extra layer, reads the *already computed* final hidden state) | ≈ k/N ≈ 0.05 (k=3, N=64) | **≈ 0.05** — essentially any acceptance wins | **Yes** (checkpoint must ship MTP heads) | ~none |
| Early-exit self-spec (exit at layer L) | L/N | **1 + L/N − 1 = L/N** (e.g. L=8,N=28 ⇒ 29 %) | No | none |
| Separate draft model (size W_m) | W_m/W | ≈ W_m/W (e.g. 1B vs 27B ⇒ 3.7 %) | No | **full draft model resident** |

**Derived conclusion:** native MTP dominates on a memory-constrained device for two
independent reasons — (i) its break-even acceptance rate is ~0, and (ii) it needs no
second resident model, which is the binding resource on a 16–64 GB device. Early-exit
self-speculation is the only variant that works on an *arbitrary frozen* checkpoint, but it
must clear an acceptance rate of L/N, which is why it is fragile. Separate-draft-model
speculation is the most general but is the most memory-expensive and the least attractive
on this hardware class.

**Caveats I will state honestly:** this model is first-order. It ignores (i) that the
verify pass over k+1 tokens may reuse some weight-cache residency, (ii) runtime/kernel
overhead differences (the 7→18.3 comparison spans two different runtimes, so part of the
2.6× is llama.cpp-vs-MLX, not MTP), and (iii) the 1.84 tokens/cycle figure implies only
~1.7–1.9× from the model, so the residual is plausibly kernel efficiency. I will present
this as a *design-rule* tool, not a validated predictor.

### 3b.2 Qwen3.6-27B — the consumer-relevant hybrid with native MTP (config verified 2026-10-21)

- `model_type qwen3_5`, arch `Qwen3_5ForConditionalGeneration` (vision-language).
- **64 layers**; `layer_types` = 3 × `linear_attention` + 1 × `full_attention` ⇒
  **48 Gated-DeltaNet layers + 16 full-attention layers (75 % / 25 %)**.
- `hidden_size 5120`; `num_attention_heads 24`; `head_dim 256`; `num_key_value_heads 4`;
  `attn_output_gate true`; `output_gate_type swish`; `partial_rotary_factor 0.25`.
- **Dense** (no MoE): `intermediate_size 17408`.
- **`mtp_num_hidden_layers: 1`** ⇒ native MTP head, no second model needed.
- `vocab_size 248320`; `tie_word_embeddings false`; `max_position_embeddings 262144`;
  `mamba_ssm_dtype float32`.
- KV arithmetic: only the 16 full-attention layers hold a KV cache ⇒
  16 × 2 × 4 × 256 × 2 B = **64 KB/token**, vs 256 KB/token for a 64-layer all-attention
  model of the same width — a **4× KV reduction from the hybrid alone**. At 32 k context
  that is 2.1 GB instead of 8.4 GB.
- `Qwen3.6-27B-UD-Q4_K_XL` GGUF is what the practitioner benchmark used ⇒ 16 GB resident,
  which fits a 24 GB M4 Pro with headroom but not a 16 GB device.

**This is the single most decision-relevant model fact in the report:** the 2026
consumer-viable frontier is a 27B dense hybrid with native MTP, not an 80B MoE hybrid.
Qwen3.8-Flash-Next (80B, 512-expert MoE) needs ≥72.5 GB even at 1-bit per
https://blog.mushroom.cv/blog/qwen3-8-flash-next-gguf-mac-unified-memory-quant-guide
(2026), and the GSQ-RCO builds need 37.6–54.8 GB resident for shard 1 alone.

---

## 3c. The decisive counter-evidence: quantization inflates reasoning tokens

Two 2026 papers, both verified from primary sources on 2026-10-21. Together they are the
strongest evidence in this report against "quantization is a free speedup for a
reasoning-capable model on a consumer device".

### (i) arXiv:2606.25519 — "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models"
Lian, Krichene, Huang, Tanaka, Ruwase, Zhang, Zhang. v1 2026-06-24, v2 2026-06-29.
https://www.arxiv.org/abs/2606.25519

- **Mechanism / object:** weight-only PTQ perturbs the output distribution; the perturbed
  model takes a *different reasoning trajectory* that is longer. New metric: **CoT Token
  Inflation Ratio (CTIR)** = reasoning tokens(quantized) / reasoning tokens(BF16), averaged
  over benchmarks.
- **Headline numbers:**
  - Qwen3-4B with **AWQ-INT3: CTIR = 276.1 %**.
  - Inflation is **strongest on problems the full-precision model answers correctly** —
    the quantized model adds steps to *recover* the correct path, rather than rambling on
    already-failed problems.
  - The extra length is **near-duplicate / repetitive steps**, not extra useful reasoning
    (measured per-instance step repetitiveness).
- **The number that matters most for this project:**
  > "Running the BBH benchmark with Qwen3-4B took **285 seconds in full precision (BF16) but
  > jumped to 292 seconds with INT4 quantization**. This occurred despite using highly
  > optimized W4A16 Marlin kernels that provide a **1.2–1.4× per-token speedup**,
  > proving that token inflation can completely negate hardware-level efficiency gains."
  ⇒ A quantized model was **2.5 % SLOWER end-to-end** than the unquantized one, on a
  highly optimized kernel, with a 1.2–1.4× per-token win. The per-token saving was more
  than consumed by the extra tokens.
- Serving effect (vLLM continuous-batching simulation): a modest **1.1× token inflation
  raised p90 TTFT by 33 % (+350 s)** and reduced overall throughput.
- **Mitigations:** prompting and decoding-time sampling give inconsistent accuracy/length
  trade-offs and *severely degrade accuracy*; **QAT (DiscQuant) is the only Pareto-dominant
  mitigation** (68 % accuracy @ 9.5k tokens vs BF16 73 % @ 8.5k tokens). QAT requires
  retraining ⇒ **not available for a frozen checkpoint**.
- **Scope / models:** RTN, GPTQ, AWQ, HQQ, ParoQuant at INT4/INT3; Qwen3-4B and Qwen3-30B
  (dense + MoE); AIME25, GSM8K, LCB, MBPP, GPQA-Diamond, BBH, BFCLv4-Multiturn, HellaSwag,
  PIQA, MMLU-Pro; calibration from UltraChat200k / Wikitext / C4 / OpenR1-Math-220k /
  Codeforces-CoTs.
- **Author-stated limitation (an explicit open gap I will quote):** the study is
  **weight-only**; "the compounding effects of **KV cache quantization or activation
  quantization** on token inflation remain unexplored."

### (ii) arXiv:2606.00206 — "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not"
Lotfi, Kirichenko, Li, Liu. Submitted 2026-05-29, revised 2026-06-02.
https://arxiv.org/abs/2606.00206

- **The failure mode is answer *emission*, not reasoning:**
  > "in up to **52 % of the quantized models' failures, models reach the right answer in
  > intermediate reasoning steps but do not output it as a final answer**."
  This is a *termination / answer-extraction* failure induced by quantization. It is invisible
  to accuracy-only evaluation that inspects intermediate steps, and invisible to PPL.
- **Mechanism:** token-level **KL divergence** between the quantized and full-precision
  output distributions; high-KL positions correlate strongly with **high next-token
  entropy**; at those positions the quantized model disproportionately samples
  **overthinking markers** — "wait", "but", "alternatively".
- **Published, training-free COMBINATION (a real combination precedent):** a **logit penalty
  on a curated set of overthinking markers** reduces CoT length by **12–23 %** while
  preserving or improving accuracy, across **5 models (1.5B–32B), 3 quantization methods,
  5 benchmarks**; overthinking errors reduced by **up to 58 %**. The paper explicitly reports
  a better accuracy-vs-reasoning-cost Pareto frontier than penalising other token sets.
- **Taxonomy:** the logit penalty changes the sampling distribution ⇒ **L4** in the strict
  sense; its *accuracy* effect is empirically neutral-to-positive. It is training-free and
  works on a frozen checkpoint.

### Why these two papers change the report's conclusions
They establish, with numbers, that:
1. **Per-token speed and end-to-end latency are different objectives** for a reasoning model,
   and a quantization method can improve the first while worsening the second.
2. **The "hidden cost" is token count, which is exactly the axis a consumer device cares
   about** (serial rounds). Any method that reduces bytes per token but increases token
   count can be net-negative on this hardware class — and the measured BBH case (285 s →
   292 s) is exactly that.
3. The two papers' mitigations are *different*: QAT (not available on frozen weights) vs a
   training-free logit penalty (available). The second is directly actionable for a
   frozen-checkpoint, on-device setting, and it **composes with quantization** — a
   published combination.

### Honest caveat on the provenance of the "wiped-out speedups" number
The 285 s → 292 s BBH figure, the CTIR 276.1 % figure, and the 52 % overthinking-error
figure are as reported by the papers' abstracts and by a secondary structured summary
(https://lacuna.tiptreesystems.com/work/quantization-inflates-reasoning-token-inflation-as-a-hidden,
reviewed at "T0 AI review" level, i.e. not independently recomputed). I therefore present
them as **author-reported, independently un-reproduced**, and I mark the one qualitative
check I *can* do myself: the direction of the effect is consistent with the independent
GSQ-RCO bit-width sweep (§2.2), which independently shows reasoning-benchmark degradation
at 2.4–2.5 bpw while zero-shot average does not move.

---

## 3d. 2026 methods I verified directly, and the combination analysis (with one crucial correction)

Primary-source index for this section (all opened 2026-10-21):
- TurboQuant (Google, ICLR 2026) — polar-coordinate KV quantization; described in
  https://www.danilchenko.dev/posts/llm-inference-efficiency-guide/ (2026-05-13, upd. 2026-09-25).
  Community impls incl. **MLX** and **llama.cpp** per that source (primary arXiv id not yet confirmed).
- TriAttention (2026) — pre-RoPE trigonometric KV eviction; vLLM plugin
  `pip install triattention`, `--triattention-budget 0.1`;
  https://www.danilchenko.dev/posts/2026-04-11-triattention-kv-cache-compression-long-reasoning/
- AsyncTLS (2026) — two-level sparse attention + async KV offloading;
  https://www.danilchenko.dev/posts/asynctls-sparse-attention/
- NSA — https://arxiv.org/abs/2502.11089 · The Sparse Frontier — https://arxiv.org/abs/2504.17768
- MInference 1.0 (NeurIPS 2024) — https://openreview.net/forum?id=fPBACAbqSN
- H2O (NeurIPS 2023) — https://huggingface.co/papers/2306.14048
- Nemotron 3 Nano — https://arxiv.org/abs/2512.20848 · https://research.nvidia.com/labs/nemotron/Nemotron-3/
- Mamba-2 — https://arxiv.org/abs/2405.21060
- "Do Transformers Need Three Projections?" (ICML 2026) — https://arxiv.org/abs/2606.04032
- KVarN (Huawei) — https://arxiv.org/abs/2606.03458 · https://github.com/huawei-csl/KVarN
- Blog index of the above: https://ai-beat.github.io/news/2026/06/kv-cache-two-ways/ (2026-06-05)

### The 2026 method set (numbers as reported by the sources above)

| Method | Attacks | Headline | Needs special training? | Calibration? |
|---|---|---|---|---|
| **TurboQuant** (PolarQuant + QJL) | KV bytes | 6× compression, 2.5–3 b/v, "zero accuracy loss" on NIAH/LongBench/RULER/ZeroSCROLLS | No | **No** |
| **TriAttention** | KV *entries* | 10.7× memory reduction; "matches full attention exactly" at 10 % budget | No | No (pre-RoPE geometry) |
| **KVarN** (Hadamard + Sinkhorn-like variance norm, K4/V2) | KV bytes | 3–5× capacity, **throughput above FP16**, FP16-level on MATH500/AIME24/HumanEval, Qwen3-32B | No | **No** |
| **Q-K=V sharing** (W_K = W_V) | KV entries (structural) | halves KV; **+3.1 % perplexity**; with GQA-4 87.5 %, with MQA 96.9 % | **Yes — train from scratch** | n/a |
| **AsyncTLS** | attention compute + offload | 1.3–4.7× end-to-end (operator speedup up to 10×) | No | No |
| **NSA** (DeepSeek) | attention compute | 9.0× fwd / 6.0× bwd / **11.6× decode @64K**; beats full attention on LongBench 0.469 vs 0.437 | **Yes — native** | n/a |
| **Nemotron 3 Nano** | attention layers removed | 6 attention + 23 Mamba-2 + 23 MoE of 52 layers; ~3.5 B active for 30 B; 3.3× Qwen3-30B-A3B (H200, vendor) | **Yes — native** | n/a |
| **Mercury 2** (Inception, diffusion) | serial token generation | 1,009 tok/s, 1.7 s e2e, AIME25 91.1 | **Yes — different model** | n/a |
| **mHC** (DeepSeek) | *training* (residual stream) | Sinkhorn-constrained doubly-stochastic mixing; stable 3B–27B; **+7.2 BBH / +6.9 DROP / +7.1 GSM8K**; 6.7 % training overhead | **Yes — train from scratch** | n/a |

### Three findings from this set that I will treat as headline

**(a) The "operator speedup ≠ end-to-end speedup" gap is large and generalisable.**
- TurboQuant: the "up to 8× on H100" figure is for the **attention logit computation kernel only**,
  explicitly not end-to-end.
- AsyncTLS: operator speedup up to 10×, **end-to-end only 1.3–4.7×**, because
  "MLP layers, sampling, and non-attention overhead together account for roughly half the
  wall-clock time even after attention gets dramatically cheaper."
  ⇒ *This independently corroborates this repository's own Stage 2.1 measurement* (0.558 ms
  fixed per-block overhead; ~90 % of decode is fixed cost; the arithmetic part is marginal).
  Two independent sources, different models, different hardware, same shape of result.
  **This is the single most robust cross-check in the report.**

**(b) The Sparse Frontier gives the "when does it break" rule the user asked for.**
https://arxiv.org/abs/2504.17768 (training-free sparse attention, Qwen 2.5, 7B–72B, ≤128K):
- **Budget rule:** at equal compute, a *bigger* model running sparse attention beats a
  *smaller dense* one.
- **Decoding tolerates compression better than prefill** (32B/72B held 17×).
- **Small models break earlier:** 7B fell from 12× safe compression at 16K to **5× at 128K**.
- **"No method wins every task, and almost every configuration had some task that failed
  below 5×, so check the tasks your pipeline depends on, not the average."**
⇒ On a consumer device the realistic target is a *larger* model at 4-bit with modest
sparsity, not a smaller model with aggressive sparsity. That inverts the "make it small
and fast" instinct and is a direct constraint on any plan that combines pruning with
quantization for a small model.

**(c) TriAttention is the method that most directly addresses the user's quality brief.**
Its claim is specifically that **at a 10 % KV budget, H2O / SnapKV / PyramidKV lose roughly
half their accuracy on AIME25, while TriAttention matches full attention exactly** — i.e.
the earlier eviction methods fail on *reasoning* even when perplexity looks fine.
Its mechanism is the one that makes this possible: scoring keys **before RoPE rotates
them**, where Q/K cluster around fixed centres so attention becomes a trigonometric
function of positional distance, giving a position-stable importance signal.
(Reported use case: a 32B coding agent on a single RTX 4090 OOMs mid-conversation with
full attention, and completes the multi-turn session at 10 % budget.)
⚠ **Verification debt:** I have this from a *secondary* 2026 blog, not from the primary
paper. The arXiv id, the exact "matches full attention exactly" wording, and the
"H2O/SnapKV/PyramidKV lose half their AIME25 accuracy at 10 % budget" comparison MUST be
confirmed against the primary source before it appears in the report as a claim.

### The combination analysis — and one correction I must make to the source

The blog's "How These Approaches Interact" section is the most explicit combination
analysis I found anywhere in 2026 material, and it is the closest thing to a published
combination matrix. But its own FAQ contains a statement that contradicts its own
recommendation, and I am recording the contradiction rather than repeating the conclusion:

| Combination | Blog's assessment | My verification of that assessment |
|---|---|---|
| TurboQuant + TriAttention | "directly complementary"; 60–100× KV reduction in theory; **"the TriAttention authors mention this combination as future work but haven't benchmarked it yet"**; FAQ: "**nobody has published a combined benchmark yet**" | **UNMEASURED.** The blog's own "practical takeaway" ("the stack that's most ready today is TriAttention on top of TurboQuant") is therefore **not supported by a measurement** — it is an engineering inference from the fact that the two operate at different levels. Per the user's instruction not to infer that combinations work merely because their components do, I will record this as a *hypothesis with no benchmark*, and it becomes one of the report's combination opportunities **with an explicit note that no published benchmark exists**. |
| TriAttention + AsyncTLS | could compose, but risk of **double-filtering**: AsyncTLS's block scoring may miss exactly the tokens TriAttention kept for scoring high on the trigonometric metric | Plausible conflict, and a good example of **two different importance criteria applied to the same set**. No published measurement. |
| TurboQuant + AsyncTLS | should compose (format-level compression vs scheduling/selection); prefetch predictions do not depend on cached precision | Lowest-risk combination of the three. Still unmeasured. |
| Mercury + KV/sparsity methods | **inapplicable** — diffusion does not use a KV cache the same way | Agrees with the substitution-argument: a method family can be *inapplicable* to another, not merely ineffective. |
| mHC + everything | orthogonal (training-time) | Agrees. |

**My independent addition to this matrix:** the same "different level" argument that makes
TurboQuant × TriAttention *composable* also makes it *non-additive*: TurboQuant reduces the
bytes of each surviving entry, TriAttention reduces the number of entries. Their savings
multiply in *capacity*, but on a bandwidth-bound device the binding constraint may be the
**KV read traffic**, and reducing entries by 10× while keeping entries at 2.5–3 bits gives
the full 60–100× **only if the surviving entries are read densely**. If eviction leaves a
scattered residue, the *effective* bandwidth cost of the residual cache is dominated by
page-granularity, not by entry size — in which case TurboQuant's extra compression buys
little and TriAttention's scattered access costs more than it saves. **This is exactly the
"scattered vs dense access" trap that this repository measured in the opposite direction
for layer parallelism** (A2: independent blocks overlap only 1.46×; A5: a batched matmul
that is bit-identical is 29 % slower at n=1). I have not found any published source that
measures KV access locality after eviction on a unified-memory device. This is a genuine,
specific, testable gap — and it is the single strongest candidate for the report's
"combination worth investigating" list.

### The output-head / embedding gap, now with one concrete data point
The "Embeddings Are Overpriced" write-up (frozen 512-bit vocabulary codes replacing a
168 M-parameter embedding table in Gemma 3 270M) and vocabulary-trimming work remain
blog-level. However, `Qwen3.8-Flash-Next` shows what the industry does when it does take
the vocab seriously: it ships a **51.2 B-parameter per-layer n-gram embedding table** as a
separate, mmap'd, disk-resident shard, and a **vocab of 248 320**. The output head for
Qwen3.6-27B alone is 248 320 × 5120 × 2 B ≈ **2.54 GB per decode step at bf16** — larger
than all 64 transformer layers' MLP weights combined would be at 4-bit. Nothing in the
ten surveyed families attacks it, and it is a *weight* cost, i.e. exactly the cost that
dominates on this hardware.

---

## 4. Deliberate non-findings (things I checked and am not claiming)

- No verified evidence yet that any *depth-reducing* or *layer-parallel* scheme delivers
  wall-clock gain at batch 1 on a bandwidth-bound consumer device. The repository's own
  measurement says the opposite for its family. I will not claim a positive result for
  this family without a primary source that measures on consumer hardware.
- No verified evidence yet that speculative decoding *reduces memory* — it does not; it adds
  a second KV cache (which is why "Memory-Efficient Speculative Decoding with Quantized
  Draft KV Cache", NeurIPS 2026, exists).

---

## 5. Round-2 additions (2026-10-21, after the Task-B system audit)

### 5.1 The layer the audit exposed: *runtime* mutual exclusion is not the same as *method* combinability

`§D.2` of the report scores method pairs. But the actual runtime (mlx-lm 0.31.3) has hard
exclusions that no paper mentions. Verified by **runtime introspection** (`inspect.getsource`
per class), not by reading line numbers:

| Pair | Runtime verdict | Mechanism |
|---|---|---|
| KV quantisation × speculative decoding | **compatible** | `QuantizedKVCache.is_trimmable()` → True |
| KV quantisation × continuous batching | **excluded** | `to_batch_cache` else-branch `raise`; no `BatchQuantizedKVCache` class exists |
| KV quantisation × sliding window | **excluded** | `RotatingKVCache.to_quantized()` → `NotImplementedError` |
| continuous batching × speculative decoding | **excluded** | `server.py:371` `is_batchable = draft_model is None` |
| continuous batching × `RotatingKVCache(keep>0)` | **excluded** | `generate.py:851-852` `raise ValueError` |
| prefix cache × sliding window | partial | `is_trimmable()` = `offset < max_size` ⇒ not trimmable once wrapped around |

**The expensive one is not in that table.** `ArraysCache` — the cache class that carries
recurrent / SSM hidden state — **does not override `is_trimmable`**, so it inherits
`_BaseCache` → **False**. And `speculative_generate_step` *raises* when the cache is not
trimmable. Therefore:

> **Speculative decoding on an SSM-backed checkpoint is not "degraded" — it is refused at
> runtime.** And because `CacheList.is_trimmable()` is `all(...)`, **one** non-trimmable
> sub-cache disables the whole model.

This is the mechanical reason `MTPLX` had to hand-write a *"GDN linear-attention kernel
with innovation tape replay for deterministic rollback"*: that is not an optimisation, it is
the minimum needed to make a non-rewindable state rewindable. It also means **§F.2's first
action is a lookup, not an implementation**: which cache class do the 48 linear-attention
layers of Qwen3.6-27B actually use?

### 5.2 Quantised KV is not "free 4×" in mlx-lm

`QuantizedKVCache.update_and_fetch` quantises only the newly appended tokens
(`mx.quantize(keys)`, `cache.py:277-278`) but does so **once per layer per K/V** ⇒
**56 small quantise kernels per generated token on a 28-layer model**. The result is then
written into a **packed uint32** layout by scatter (not a contiguous append), which also
**bypasses fused SDPA**. Net: capacity ÷ (bandwidth saving − quantise/scatter overhead).
At short context and batch 1 that quotient can plausibly approach 1, which is a *different*
conclusion from the "KV compression is useless" framing and needs to be stated as such.

### 5.3 The long-context inflection (this materially qualifies §0.4-1)

Qwen3.6-27B has 64 KB KV/token against a 16 GB 4-bit weight budget:

| context | KV | vs weights | dominant |
|---|---|---|---|
| 16k | ~1.0 GB | 6 % | weights |
| 32k | ~2.0 GB (3.5 GiB measured) | 12–22 % | comparable |
| 128k | ~8.4 GB | 52 % | **KV** |

⇒ **"KV compression does not help" is a short-context statement.** At ≥32k it becomes an
**admission requirement**, and at 128k it is forced by a hard machine limit: this machine's
`max_buffer_length = 14,302,248,960 B = 13.32 GiB`, while the 128k KV block is ≈14 GiB —
**one contiguous buffer cannot hold it**, so paging is mandatory, not optional.

### 5.4 A sharper speculative-decoding cost model

```
per round   = n·T_d + T_t(n+1)
per round   = a + 1 tokens
speedup(a,n) = (a+1) / ( n·T_d/T_t + (n+1) )
```

`T_t(n+1) ≈ T_t` because decode is bandwidth-bound, not compute-bound: feeding the target
model n+1 tokens costs almost the same as feeding it 1. Consequences:

- speedup depends **only** on *how cheap the draft is* and *how high acceptance is* —
  **not on the target model's depth**;
- equal-size draft ⇒ needs **100 % acceptance** just to break even;
- 1/10-size draft at 70 % acceptance, n=3 ⇒ **≈2.1×**, consistent with the measured 2.6×;
- **deeper drafts are a net loss** unless acceptance is high, because `n` is charged twice.

⚠ Do not conflate "1.84 tokens/cycle" (measured) with "84 % acceptance" — they are different
quantities.

### 5.5 BaseRT is two papers, and only one of them is about this hardware

- **arXiv:2607.00501** — native Metal runtime, evaluated on **M3 and M4 Pro** (Qwen3 /
  Llama 3.2 / Gemma 4, Q4/Q8): **decode up to 1.56× llama.cpp, 1.35× MLX**; bigger prefill
  margins for MoE. ← *this is the M4-relevant one.*
- **arXiv:2607.19438** — M5 **Neural Accelerators** via the Metal 4 tensor API:
  **prefill up to 6.4× llama.cpp / 3.9× MLX**, but **decode only ≈1.02–1.33×** because the
  decode path deliberately stays on the old kernels. ← **M5-only; M4 Pro has no NAX.**
  Its own claim is challenged by a third-party structured reading ("attribution to tensor
  cores is under-supported by the missing ablation and default-only baselines").

Citing only the M5 paper would point the reader at a number this hardware cannot use.

### 5.6 SGLang's official Apple path exists but is not satisfiable here

Official docs (`apple_metal.mdx`, `references/environment_variables.html`) document
`SGLANG_USE_MLX=1` + `--disable-cuda-graph`; prereqs are **macOS ≥14, PyTorch 2.13.x,
MLX ≥0.32.0**. **This machine has MLX 0.31.2 ⇒ the official path cannot be used as-is.**
A third-party July-2026 assessment instead calls it pre-alpha / roadmap #19137 and quotes
**≈5.3× over SGLang's own PyTorch-MPS path** — a comparison against *itself*, not against
mlx-lm, so it says nothing about runtime choice on this device.

