# R3-S — SpQt `2511.04477` bandwidth claims & Prox `2607.27591` "+12% weight storage"

**Purpose:** first-hand numerical verification of two disputed numbers from the R2 report, for adversarial cross-review.
**Date of verification:** 2026-10-21 · **Agent:** explore (read-only) · **Method:** HTTP GET + `pdftotext -layout` only. No model downloads, no builds, no benchmarks.
**Evidence grades:** `[V]` VERIFIED (read directly, quoted) · `[I]` INFERRED (my arithmetic / reasoning from `[V]` facts) · `[NF]` NOT FOUND (explicitly absent or unreachable, with status).

---

## 0. Claim / verdict / evidence (master table)

| # | Claim under dispute | Verdict | Primary evidence |
|---|---|---|---|
| A1 | SpQt reports end-to-end speedups A19 Pro 1.55×, M2 1.40×, M2 Pro 1.26×, M2 Max 1.19× | **CONFIRMED as a faithful transcription of the paper** — but only for **one specific cell of Table 2**: Llama-2-**7B**, **Unified 50 %**, speedup vs **dense Q4_K** | SpQt PDF p.7–8, Table 2 (rows `Unified 50%`, column `Llama-2 7B`): `22.86 (1.55×)`, `30.69 (1.40x)`, `47.17 (1.26x)`, `68.87 (1.19x)` `[V]` |
| A2 | SpQt lists M2 Max memory bandwidth as **819.2 GB/s** | **CONFIRMED — the paper really prints `819.2 GB/s`**, in **Table 1** ("Hardware configurations for evaluation") | SpQt PDF p.7, Table 1: `M2 Max   30   1.4 GHz   819.2 GB/s   ~80 W` `[V]` |
| A3 | Apple says M2 Max = 400 GB/s, so the paper's 819.2 is wrong | **PAPER IS WRONG (the M2 Max cell is erroneous).** Apple's official M2 Max = **400 GB/s**; the paper's number is **2.048× Apple's** and **1.6× the chip's theoretical peak** | Apple newsroom 2023-01: "M2 Max … Its **400GB/s** of unified memory bandwidth is twice that of M2 Pro, 4x that of M2" `[V]` |
| A4 | Prior report's **273 GB/s** for **M4 Pro** | **CORRECT.** Apple's official M4 Pro bandwidth is **273 GB/s** (two independent Apple pages) | Apple newsroom 2024-10 + `support.apple.com/en-us/111907` `[V]` |
| A5 | 819.2 GB/s "means something else" (theoretical peak? typo? different chip?) | **Most likely a 2× arithmetic slip on the M2 Max row**; a secondary hypothesis is that it is **Apple's M3 Ultra figure** (819 GB/s) mis-filed. **The paper gives no derivation, so this is a HYPOTHESIS** | see §3.3 `[I]` / `[V]` for inputs |
| B1 | Prox's proxy overhead is "+12 % of weight storage" | **CONFIRMED verbatim, §5.3 Discussion — but it is a bare, underived assertion** with no table, no figure, no derivation, and no byte count anywhere in the paper | Prox HTML §5.3 + PDF p.4 `[V]` |
| B2 | Is that 12 % measured against **FP16** weights or against 4-bit? | **Against FP16 (full-precision) weights.** Reconstructing the ratio for Prox's own 6 verifiable models: proxy-INT4 ÷ **FP16** weights = **10.2 %–13.0 %** (mean ≈ 11.9 % ⇒ "approximate 12 %"); the same numerator over **INT4** weights = **41 %–52 %**, which is nowhere near 12 % | see §4.2 `[I]` from `[V]` configs |
| B3 | Can the 12 % be transferred to a **27B model in Q4**? | **NO — not as a 12 % multiplier.** The 12 % is FP16-denominated; at Q4 the same INT4 proxy is a **≈36 %–41 %** weight-storage tax. The prior report's "15.2 GB × 1.12 = 17.0 GB" **understates** the true extra bytes by ≈3× | see §4.3 `[I]` |

---

# PART A — SpQt (arXiv:2511.04477)

## 1. How the paper was actually read (provenance)

* `https://arxiv.org/html/2511.04477v1` → **HTTP 200**, 135 761 B. `v2` → **HTTP 404**, `v3` → **HTTP 404** (only **v1** exists; arXiv id line in the PDF: `arXiv:2511.04477v1 [cs.DC] 6 Nov 2025`).
* **The HTML render has no extractable table text.** Confirmed twice:
  * `grep` for `819.2` / `GB/s` over `spqt.html` → **0 hits**.
  * The three tables are `<figure class="ltx_table">` wrapping `<object type="image/svg+xml" data="2511.04477v1/{platformspecs,e2e,perplexity}.svg">` → **0 `<text>` elements**, 180 / 863 / … `<path>` elements ⇒ **pure glyph paths**.
  * Fetched `platformspecs.svg` (HTTP 200, 103 186 B) and `e2e.svg` (HTTP 200, 443 501 B) directly: **0 `<text>` elements**, `grep` for `819.2|75.8|102.4|204.8` → **0 hits**. `[V]`
* ⇒ **All numbers below come from the arXiv PDF via `pdftotext -layout`** (`https://arxiv.org/pdf/2511.04477v1` → HTTP 200, 653 527 B, 11 pages, PDF 1.7).
* **Table ↔ SVG identity mapping is confirmed** by the HTML figure ids: `S6.T1` = Table 1 = `platformspecs.svg`; `S6.T2` = Table 2 = `e2e.svg`; `S6.T3` = Table 3 = `perplexity.svg`.
* Paper identity: title **"Enabling Dynamic Sparsity in Quantized LLM Inference"**, authors Rongxiang Wang, Kangyuan Shu, Felix Xiaozhu Lin (`pdfinfo` metadata + PDF p.1 title block) `[V]`.

> **Prose vs table provenance, stated explicitly:** every bandwidth number (75.8 / 102.4 / 204.8 / **819.2**) comes from **Table 1** (a table). The prose (§6.1 "Test platforms") only says *"the Apple M2 series (5 nm) and A19 Pro (3 nm), as listed in Table 1"* — **no bandwidth number appears in the prose**. Every speedup number quoted below comes from **Table 2**; the prose §6.2 supplies only the qualitative statements *"up to 1.47× on M2 Max, 1.33× on …"*. The headline *"up to 1.55× faster decoding throughput"* is in the **abstract** (prose) and the **conclusion** (prose).

## 2. Table 1 — hardware, **exact printed strings** `[V]`

`pdftotext -layout` output, SpQt PDF **page 7**, right column. Caption: `Table 1. Hardware configurations for evaluation.`

| Platforms | GPU core | GPU freq | Mem bandwidth | SoC TDP |
|---|---|---|---|---|
| `A19 pro` | `5` | `1.6 GHz` | **`75.8 GB/s`** | `~5 W` |
| `M2` | `8` | `1.4 GHz` | **`102.4 GB/s`** | `~20 W` |
| `M2 Pro` | `19` | `1.4 GHz` | **`204.8 GB/s`** | `~45 W` |
| `M2 Max` | `30` | `1.4 GHz` | **`819.2 GB/s`** | `~80 W` |

So the paper prints **`819.2 GB/s`** for M2 Max — **exact string, confirmed, not a transcription error by the prior report.** `[V]`

## 3. Table 2 — end-to-end speedups, and what exactly was timed `[V]`

Caption (verbatim, PDF p.8): *"Table 2. End-to-end decoding latency (ms/token) and speedup over dense Q4 K across Llama models and Apple SoCs. Our method achieves up to 1.55× speedup on A19 Pro, 1.46× on M2, 1.32× on M2 Pro, and 1.45× on M2 Max."*

Column header (verbatim): `hardware | sparsity | Llama-2 {7B, 13B, 70B} | Llama-3 {8B}`.

### 3.1 The four disputed numbers — they ARE in the paper, in one specific slice

Row `Unified 50%`, column **Llama-2 7B** (the only slice that reproduces the disputed quadruple exactly):

| hardware | Table 2 cell (verbatim) | latency ms/token |
|---|---|---|
| A19 Pro | `22.86 (1.55×)` | 22.86 |
| M2 | `30.69 (1.40x)` | 30.69 |
| M2 Pro | `47.17 (1.26x)` | 47.17 |
| M2 Max | `68.87 (1.19x)` | 68.87 |

> ⚠ **Two different "per-chip speedup" series coexist in the paper**, and the prior report used the *Llama-2-7B* one:
> * **Llama-2-7B series** (what the prior report quotes): 1.55× / 1.40× / 1.26× / 1.19×.
> * **"up to" series in the Table 2 caption** (max over all models): A19 Pro **1.55×**, M2 **1.46×**, M2 Pro **1.32×**, M2 Max **1.45×**.
> The M2 Max "1.45×" in the caption is the **Llama-2-70B** cell (`11.33 (1.45x)`), not the 7B cell. `[V]`
> The prior report's other M2 Max figure `1.18×` is the **Llama-3-8B** cell (`65.07 (1.18x)`) — also present in the table. `[V]`

### 3.2 Full Table 2, transcribed (PDF p.8) `[V]`

| hardware | config | Llama-2 7B | Llama-2 13B | Llama-2 70B | Llama-3 8B |
|---|---|---|---|---|---|
| **A19 Pro** | Baseline-F16 | OOM | - | - | OOM |
| | Baseline-Q4K | `14.78 (1.00×)` | - | - | `13.24 (1.00×)` |
| | TEAL 25 % | `17.04 (1.15×)` | - | - | `15.13 (1.14×)` |
| | TEAL 40 % | `19.92 (1.35×)` | - | - | `17.4 (1.31×)` |
| | TEAL 50 % | `22.79 (1.54×)` | - | - | `19.3 (1.46×)` |
| | Unified 25 % | `17.22 (1.17×)` | - | - | `15.08 (1.14×)` |
| | Unified 40 % | `19.37 (1.31×)` | - | - | `17.23 (1.30×)` |
| | **Unified 50 %** | **`22.86 (1.55×)`** | - | - | `19.29 (1.46×)` |
| **M2** | Baseline-F16 | OOM | OOM | - | OOM |
| | Baseline-Q4K | `21.98 (1.00x)` | `11.55 (1.00x)` | - | `20.03 (1.00x)` |
| | TEAL 25 % | `23.91 (1.09x)` | `13.18 (1.14x)` | - | `21.90 (1.09x)` |
| | TEAL 40 % | `27.32 (1.24x)` | `15.11 (1.31x)` | - | `25.05 (1.25x)` |
| | TEAL 50 % | `31.16 (1.42x)` | `17.05 (1.48x)` | - | `27.93 (1.39x)` |
| | Unified 25 % | `24.06 (1.09x)` | `13.12 (1.14x)` | - | `21.83 (1.09x)` |
| | Unified 40 % | `27.69 (1.26x)` | `15.00 (1.30x)` | - | `24.85 (1.24x)` |
| | **Unified 50 %** | **`30.69 (1.40x)`** | `16.82 (1.46x)` | - | `27.78 (1.39x)` |
| **M2 Pro** | Baseline-F16 | OOM | OOM | - | OOM |
| | Baseline-Q4K | `37.29 (1.00x)` | `20.52 (1.00x)` | - | `34.64 (1.00x)` |
| | TEAL 25 % | `37.28 (1.00x)` | `21.42 (1.04x)` | - | `34.79 (1.00x)` |
| | TEAL 40 % | `42.33 (1.14x)` | `24.48 (1.19x)` | - | `39.16 (1.13x)` |
| | TEAL 50 % | `46.88 (1.26x)` | `27.37 (1.33x)` | - | `44.50 (1.28x)` |
| | Unified 25 % | `37.13 (1.00x)` | `21.36 (1.04x)` | - | `34.27 (0.99x)` |
| | Unified 40 % | `41.81 (1.12x)` | `24.27 (1.18x)` | - | `38.65 (1.12x)` |
| | **Unified 50 %** | **`47.17 (1.26x)`** | `27.00 (1.32x)` | - | `43.80 (1.26x)` |
| **M2 Max** | Baseline-F16 | `24.43 (0.42x)` | `12.99 (0.38x)` | OOM | `22.04 (0.40x)` |
| | Baseline-Q4K | `57.96 (1.00x)` | `34.26 (1.00x)` | `7.83 (1.00x)` | `55.20 (1.00x)` |
| | TEAL 25 % | `57.54 (0.99x)` | `34.39 (1.00x)` | `8.78 (1.12x)` | `54.42 (0.99x)` |
| | TEAL 40 % | `64.16 (1.11x)` | `38.57 (1.13x)` | `10.24 (1.31x)` | `60.13 (1.09x)` |
| | TEAL 50 % | `68.75 (1.19x)` | `42.07 (1.23x)` | `11.48 (1.47x)` | `64.71 (1.17x)` |
| | Unified 25 % | `57.59 (0.99x)` | `34.22 (1.00x)` | `8.68 (1.11x)` | `53.65 (0.97x)` |
| | Unified 40 % | `63.08 (1.09x)` | `38.17 (1.11x)` | `10.04 (1.28x)` | `59.83 (1.08x)` |
| | **Unified 50 %** | **`68.87 (1.19x)`** | `42.09 (1.23x)` | `11.33 (1.45x)` | `65.07 (1.18x)` |

### 3.3 Experimental setup — the exact answers the reviewer needs `[V]`

| Item | Answer | Source |
|---|---|---|
| **What is being timed** | **End-to-end decoding latency, ms/token** (Table 2). Full llama.cpp inference: prefill of a short prompt, then greedy autoregressive decode; latency normalised per generated token. Separate **microbenchmarks** (GEMV latency in µs, Figures 8–10) are a *different* measurement. | Table 2 caption; §6.1 "Operation details"; §6.2 "Latency reduction" |
| **Baseline for the speedup** | **`Baseline-Q4K`** = llama.cpp's default **Q4_K** dense kernel. Speedups are **relative to dense Q4_K**, *not* to FP16. (`Baseline-F16` is a separate reference row.) | Table 2 caption: "speedup over dense Q4 K"; §6.1 Baselines |
| **Sparsity level** | Row label **`Unified 50%`** = the paper's *unified sparsity mode* at **50 % activation sparsity** (a single global threshold applied to all components, as opposed to the "customized TEAL mode"). 25 % / 40 % / 50 % rows also exist. | §6.1 "Calibration threshold calculation"; Table 2 row labels |
| **Precision** | Weights quantized to **Q4_K (4-bit K-quant)** in the **zigzag layout**, for **all weight matrices involved in GEMV during decoding — except the token embedding and the output embedding**. `Baseline-F16` = full precision. | §6.1 "Customized quantization" |
| **Sequence length / prompt** | *"we conduct inference using a short prompt of approximately **10 tokens** and report the resulting per-token decoding latency."* **Number of generated tokens is NOT stated.** | §6.1 "Models and Datasets" |
| **Batch size** | **NOT STATED anywhere in the paper.** `grep -i "batch"` over the full PDF text yields only 2 hits, both in Related Work ("batched LLM serving"). No batch size, no throughput-vs-batch curve, no statement that it is 1. | `[NF]` |
| **Models** | Llama-2 (7B, 13B, 70B) and Llama-3 (8B). Llama-3-70B excluded (quantization accuracy issue, Qin 2024). | §6.1 |
| **Runtime** | llama.cpp **b5711** + a custom Metal/GPU GEMV kernel; ~8K SLOC C/C++; zigzag layout on top of TEAL's calibration. | §5 Implementation |
| **Devices** | **Only SoC classes are named** (Table 1). **No specific Mac model, no RAM configuration, no clock/thermal state** is given anywhere in the paper (checked pp. 1–11). | `[NF]` |
| **Accuracy** | Table 3: perplexity under Q4_K; ≤50 % sparsity ⇒ marginal degradation, 65 % ⇒ noticeable loss. | Table 3, PDF p.9 |

### 3.4 Internal inconsistencies found while reading (all `[I]`, flagged not asserted as fact)

1. **Prose contradicts Table 2 on M2 Max.** §6.2 prose: *"quantization alone already provides significant acceleration, with 4 bit quantization (Q4 K) achieving more than 2× speedup over the full-precision baseline."* But Table 2's M2 Max row shows `Baseline-F16 24.43 (0.42x)` vs `Baseline-Q4K 57.96 (1.00x)` ⇒ **Q4_K is 2.37× SLOWER than F16 on M2 Max**; the parenthetical is Q4_K-relative-to-itself, so the "more than 2× speedup" claim is the *reverse* of the table on that chip. `[V]` for the cells, `[I]` for the contradiction.
2. **M2 Max latencies are physically implausible on a real M2 Max.** Llama-2-7B FP16 weights ≈ 6.74 B × 2 B ≈ 13.5 GB; `13.5 GB / 24.43 ms ≈ 552 GB/s` ⇒ **138 % of Apple's official 400 GB/s peak** — impossible; it only becomes arguable under the paper's inflated 819.2 GB/s. *Assumption stated: 6.74 B params for Llama-2-7B, 2 B/param FP16, 100 % memory-bandwidth-bound decode (optimistic for the paper).* `[I]`
3. **Same check on the A19 Pro row.** Llama-2-7B Q4_K ≈ 3.8 GB; `3.8 GB / 14.78 ms ≈ 257 GB/s` vs the paper's own claimed `75.8 GB/s` ⇒ **3.4× over its own stated peak**. *Same optimistic assumptions.* `[I]`
4. **The A19 Pro row does not match Apple's A19 Pro.** Apple's iPhone 17 Pro newsroom: *"A19 Pro chip is faster with a **6-core GPU** …"*; the paper's Table 1 says `A19 pro / 5` GPU cores, `1.6 GHz`, `~5 W`. Also, the paper never explains how llama.cpp (a desktop/macOS runtime) was run on an A19 Pro. `[V]` for the 6-core statement, `[I]` for the mismatch, `[NF]` for the paper's explanation.

## 4. Apple's official memory bandwidth (all `[V]`, all direct reads of Apple pages)

| Chip | **Official bandwidth** | Apple source | HTTP |
|---|---|---|---|
| M2 | **100 GB/s** | newsroom 2023-01, Mac mini w/ M2 — *"Mac mini with M2 also powers through even more demanding workloads. With up to 24GB of unified memory and **100GB/s of bandwidth**…"* — <https://www.apple.com/newsroom/2023/01/apple-introduces-new-mac-mini-with-m2-and-m2-pro-more-powerful-capable-and-versatile-than-ever/> | 200 |
| M2 Pro | **200 GB/s** | newsroom 2023-01 — *"M2 Pro features 40 billion transistors, **200GB/s of unified memory bandwidth** … It features 200GB/s of unified memory bandwidth — twice that of M2 —"* — <https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/> | 200 |
| M2 Max | **400 GB/s** | same page — *"M2 Max features 67 billion transistors, **400GB/s** of unified memory bandwidth … Its 400GB/s of unified memory bandwidth is twice that of M2 Pro, **4x that of M2**"* | 200 |
| M2 Ultra | 800 GB/s | <https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/> — *"**800GB/s** of system memory bandwidth"* / *"…**800GB/s** of memory bandwidth — twice that of M2 Max"* | 200 |
| M3 Ultra | "over 800 GB/s" (newsroom wording) | <https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/> — *"…**over 800GB/s** of memory bandwidth"* | 200 |
| M4 | 120 GB/s | newsroom 2024-10 — *"M4 supports up to 32GB of unified memory and has higher memory bandwidth of **120GB/s**"* | 200 |
| **M4 Pro** | **273 GB/s** | newsroom 2024-10 — *"**M4 Pro** supports up to 64GB of fast unified memory and **273GB/s of memory bandwidth**, which is a massive 75 percent increase over M3 Pro"* — <https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/> | 200 |
| M4 Pro (2nd source) | **273 GB/s** | Apple support, Mac Studio tech specs — *"…20-core GPU … **273GB/s memory bandwidth**"* — <https://support.apple.com/en-us/111907> | 200 |
| M4 Max | 546 GB/s (410 GB/s for the 14-core/32-core config) | newsroom 2024-10 + support page | 200 |
| **A19 / A19 Pro** | **NOT PUBLISHED by Apple** | <https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/> — page read in full: **0 occurrences of "GB/s", 0 of "bandwidth", 0 of "memory bus"**; <https://www.apple.com/iphone-17-pro/specs/> — **0 occurrences of "GB/s"** | 200 both |

**Verdict on A4 / PART A item 5:** the previous report's **273 GB/s for M4 Pro is CORRECT** and matches Apple's official spec exactly (two independent Apple pages). No correction needed to that number. `[V]`

> **Side observation, flagged for the review corpus (not mine to edit):** `research/agent-reports/workbuddy/2026-10-r2-report.md:64` and `research/REPORT.md:64` describe the hardware as *"M4 Pro 48 GB (~120 GB/s)"* — **120 GB/s is the base M4, not the M4 Pro (273 GB/s)**. The minimax R2 report's "M4 Pro 是 273 GB/s" is the correct one. `[V]` for both Apple figures, `[I]` for the attribution.

## 5. Diagnosis of `819.2 GB/s` (the core of the dispute)

### 5.1 Established facts `[V]`

| Fact | Value |
|---|---|
| Paper's Table 1 M2 Max cell | `819.2 GB/s` |
| Apple's official M2 Max | **400 GB/s** |
| Ratio paper/official | 819.2 / 400 = **2.048×** |
| Paper's own M2 and M2 Pro cells | `102.4`, `204.8` (vs Apple 100, 200 → ×1.024) |

### 5.2 The arithmetic that nails it `[I]`

The paper's M2 / M2 Pro numbers are *exactly* `bus_width_bits × 6400 MT/s ÷ 8`:

* M2: 128-bit × 6400 × 10⁶ ÷ 8 = **102.4 GB/s** ✔ matches the paper
* M2 Pro: 256-bit × 6400 × 10⁶ ÷ 8 = **204.8 GB/s** ✔ matches the paper
* M2 Max: 512-bit × 6400 × 10⁶ ÷ 8 = **409.6 GB/s** ✘ the paper prints **819.2**

⇒ **`819.2 = 2 × 409.6`**, i.e. the paper's own rule applied to the M2 Max's 512-bit bus yields 409.6, and the paper printed **2×** that. `819.2` is also exactly `12800 MT/s × 512 bit ÷ 8` (or `1024 bit × 6400 MT/s ÷ 8`).

⇒ **The paper's M2 Max figure is inconsistent with the paper's own M2 / M2 Pro figures.** This is self-contained evidence that the cell is wrong, independent of Apple.

### 5.3 Two competing explanations — **HYPOTHESES** (the paper explains nothing)

**H1 (primary) — 2× arithmetic slip on the M2 Max row.** The author computed `bus × 6400 MT/s ÷ 8` for M2 (128-bit) and M2 Pro (256-bit) correctly, then mis-multiplied M2 Max (512-bit) by 2 (e.g. double-counted the two memory-controller halves / copied the 1024-bit figure). **Correct value: 400 GB/s (Apple official) or 409.6 GB/s (theoretical at 6400 MT/s).**
*Supporting:* the 2.048× and 4.096× relations to Apple's official figures; the exact 2× relation to the rule's own prediction. *Against:* nothing — but the paper gives no derivation to falsify it.

**H2 (alternative) — a different chip's figure was filed under M2 Max.** `819.2 GB/s` coincides with Apple's **M3 Ultra** memory bandwidth (`819 GB/s` on Apple's tech-spec pages; the M3 Ultra newsroom wording is "over 800GB/s", verified this session). M2 Ultra is 800 GB/s (verified). A row-swap/copy from the M3 Ultra spec sheet would produce exactly `819.2`.
*Supporting:* `819.2` is suspiciously exactly the M3 Ultra number rounded to a decimal; M2 Max's row is the only one that breaks the paper's own bus-width pattern. *Against:* I could **not** re-read an Apple page printing the exact string "819GB/s" in this session (Apple's *current* Mac Studio specs page lists M5 parts; the M3 Ultra newsroom says only "over 800GB/s") — see §7.

**H3 (rejected) — "819.2 GB/s is the M2 Max's real theoretical peak".** ❌ Rejected: even the true M2 Max theoretical peak (LPDDR5X-8000 × 512-bit) = **512 GB/s**, and Apple's official figure is 400 GB/s. `819.2` exceeds the theoretical peak by 1.6×. It is also **not** "2× of a real Apple figure" for M2 Max (2 × 400 = 800 ≠ 819.2), except trivially as "2 × 409.6".

**Bottom line (PART A item 4):** the paper's `819.2 GB/s` is an **erroneous cell**, most likely H1 (2× slip), possibly H2 (M3 Ultra figure mis-filed). The prior report transcribed it **faithfully from the paper** — the error originates upstream, in SpQt's Table 1. `[V]` for the paper's cell, `[I]` for the diagnosis.

### 5.4 Does the error damage the prior report's conclusion? **No.** `[I]`

The prior report's inference *"speedup shrinks as bandwidth grows"* and its extrapolation *"M4 Pro (273 GB/s) ⇒ ≈1.2–1.3×"* survive the correction, because **the ordering is unchanged** once M2 Max's bandwidth is corrected to 400 GB/s:

`75.8 → 1.55×  |  102.4 → 1.40×  |  204.8 → 1.26×  |  400 (corrected) → 1.19×`

M4 Pro's 273 GB/s still lies between M2 Pro (204.8 → 1.26×) and M2 Max (400 → 1.19×), so the interpolated range **≈1.2–1.3× is unchanged**. **Only the anchor value `819.2` must be corrected to `400` (official) / `409.6` (theoretical), and the statement "M4 Pro falls between M2 Pro (204.8) and M2 Max (819.2)" must be re-anchored to "…and M2 Max (400)".**

---

# PART B — Prox (arXiv:2607.27591)

## 6. Where the 12 % appears — verbatim, with section number `[V]`

* `https://arxiv.org/html/2607.27591v1` → **HTTP 200**, 496 109 B. Title: *"Prox: Training-Free FFN Activation Sparsity via Approximate Intermediate-Channel Salience in LLMs"*. PDF: `https://arxiv.org/pdf/2607.27591v1` → **HTTP 200**, 1 522 744 B.
* The number appears **exactly once in the entire paper** (`grep -c "12%"` on the PDF text = 1 hit; HTML = 1 hit).

**Section: §5.3 "Discussion"** (last numbered subsection before §6 Conclusion). Verbatim sentence:

> *"Additionally, retaining Stage 1 **INT4** proxy weights in GPU memory introduces an **approximate 12% weight-storage overhead**. Future work could reduce this overhead through a sliding-window scheme that dynamically loads and evicts proxy weights across Transformer layers."*

Adjacent sentence in the same paragraph: *"Prox currently targets single-batch autoregressive decoding … Extending proxy-guided sparsification and its underlying sparse kernels to efficient batched inference remains a promising avenue for future exploration."*

**There is NO table, figure, appendix entry, or byte count behind the 12 %.** The paper's *only* quantitative proxy-cost analysis is **Appendix A.3 "Empirical Proxy-Cost Ratios"**, and that is a **latency** ratio, not a byte ratio:

> §A.3: *"The allocation model uses **α** to approximate the cost of one **INT4 proxy GEMV relative to an FP16 GEMV** at the same input sparsity. We measure this latency ratio across NVIDIA A6000, GeForce RTX 3090, and A100 GPUs for four Qwen3 scales and four input sparsity levels, as reported in Table 5. The measured ratios range from 0.234 to 0.351, with a pooled mean of 0.285. We use the single rounded value **α = 1/3**."*  (Table 5 = measured α ∈ [0.234, 0.351]) `[V]`

⇒ **The 12 % is an underived, whole-model, approximate byte ratio.** It is *not* a per-layer figure, *not* a per-tensor figure, *not* a measurement, and *not* tied to a sparsity level (it is stated with no sparsity qualifier at all).

**Supporting definitions (verbatim):**

* §4.3: *"we construct the Stage 1 proxy weights via **symmetric per-row INT4 quantization of the original FP16 W_up and W_gate matrices**, keeping the compact proxies resident in GPU memory."* `[V]`
* §4.1: *"Stage 2: exact sparse computation. Using the channel mask m_s from Stage 1, Prox performs exact SwiGLU computations exclusively over the selected channels using the **original full-precision model weights**."* `[V]`
* §5.2 A1: *"A1 replaces the INT4 proxy weights with FP16 weights, reallocating the two-stage compute under the same target budget to accommodate the increased proxy cost."* `[V]`
* §5.1 "Compatibility with model quantization" evaluates **AWQ / FP8 / W4A16 / W8A8 — but only WikiText perplexity**, never end-to-end speed. The speedup experiments (§5.1, Figure 5, A6000) use **full-precision weights**. `[V]`

## 7. What is the 12 % *of*? — reconstruction `[I]`

**Hypothesis (strongly supported, not stated by the paper): 12 % = (INT4 bytes of the W_up+W_gate proxy, all layers) ÷ (FP16 bytes of the model's total weight storage).**

Reconstruction formula:

```
proxy_bytes        = L · 2 · h · inter · (b/8)          # W_up + W_gate, INT4, b = 4
weight_bytes(prec) = N_total · (p/8)                     # N_total = all params incl. embeddings & lm_head
overhead_%         = 100 · proxy_bytes / weight_bytes(prec)
```

Applying it to Prox's **own evaluated models** (configs fetched live from HuggingFace, §8):

| model | L | h | inter | params | proxy INT4 | **÷ FP16 weights** | ÷ INT4 weights | ÷ block-only FP16 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 28 | 2048 | 6144 | 1.72 B | 0.352 GB | **10.24 %** | 40.96 % | 12.50 % |
| Qwen3-4B | 36 | 2560 | 9728 | 4.02 B | 0.897 GB | **11.14 %** | 44.58 % | 12.34 % |
| Qwen3-8B | 36 | 4096 | 12288 | 8.19 B | 1.812 GB | **11.06 %** | 44.25 % | 13.04 % |
| Qwen3-14B | 40 | 5120 | 17408 | 14.77 B | 3.565 GB | **12.07 %** | 48.28 % | 13.49 % |
| Qwen3-32B | 64 | 5120 | 25600 | 32.76 B | 8.389 GB | **12.80 %** | 51.21 % | 13.44 % |
| Mistral-7B-v0.3 | 32 | 4096 | 14336 | 7.25 B | 1.879 GB | **12.96 %** | 51.85 % | 13.46 % |

⇒ **FP16 denominator ⇒ 10.2 %–13.0 %, mean ≈ 11.9 % ⇒ "approximate 12 %". ✔**
⇒ **INT4/4-bit denominator ⇒ 41 %–52 % ⇒ could not be described as "12 %". ✘**
⇒ **Transformer-block-only FP16 denominator ⇒ 12.3 %–13.5 % ⇒ also consistent.** The denominator is therefore "all weights" or "block weights"; both land on 11–13 %.

**Answer to the reviewer's question: the 12 % is measured against FP16 (full-precision) weight storage.** The proxy is INT4; the model it is compared against is FP16. If the model itself is quantized to 4-bit, the same numerator sits on a 4× smaller denominator.

**Also note:** the paper's cost model is *itself* FP16-denominated (α = INT4-proxy-GEMV ÷ **FP16** GEMV, §A.3), so an FP16-denominated storage ratio is internally consistent with the paper's framing. `[I]`

**Sparsity level:** the 12 % sentence carries **no sparsity qualifier**; the proxy exists at every sparsity level and its *bytes* do not depend on sparsity (only Stage-1 *compute* does). So it is a sparsity-independent, whole-model, whole-precision-matrix figure. `[V]` (absence of any qualifier) / `[I]` (the invariance argument).

## 8. Transferring to a 27B model in Q4 — exact arithmetic

### 8.1 The verified target model config `[V]`

`https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json` → **HTTP 200** (this is the repo's project model, `research/REPORT.md:65,271`).
Text config: `hidden_size 5120`, `intermediate_size 17408`, `num_hidden_layers 64`, `num_attention_heads 24`, `head_dim 256`, `num_key_value_heads 4`, `vocab_size 248320`, `tie_word_embeddings false`, `full_attention_interval 4` (48 linear_attention + 16 full_attention), `mtp_num_hidden_layers 1`, `intermediate_size` for the **vision tower = 4304**, vision `depth = 27`.
Total parameters: **27 781 427 952** (`https://huggingface.co/api/models/Qwen/Qwen3.6-27B?expand[]=safetensors` → HTTP 200, `"BF16": 27781427952`).

### 8.2 The proxy bytes — formula and bounds `[I]` (from §8.1 `[V]` config)

```
proxy_bytes_per_layer = 2 · h · inter · (4/8)         # W_up + W_gate, INT4
                      = 2 · 5120 · 17408 · 0.5
                      = 89,128,960 B  = 85.00 MiB per layer

UPPER BOUND (Prox today: all layers' proxies resident)
  proxy_total = L · 89,128,960
  L = 64 (verified text layers)  →  5,704,253,440 B = 5.704 GB = 5.312 GiB
  L = 27 (the briefing's literal "27 layers" variant)  →  2,406,481,920 B = 2.406 GB = 2.241 GiB

LOWER BOUND (the paper's own proposed mitigation: sliding-window load/evict)
  only W layers resident at a time:  W · 89,128,960 B
  W = 1 →  85.00 MiB   (0.089 GB)      W = 8 →  680.66 MiB (0.713 GB)
  (W = 0 is not usable: Stage 1 needs the proxy of the layer it is masking)

per-layer proxy as % of that layer's weights:
  proxy INT4 / layer  ÷  FP16 total bytes of the whole model  =  85.00 MiB / 55.56 GB  =  0.16 %
  proxy INT4 / layer  ÷  Q4 total bytes of the whole model      =  85.00 MiB / 13.89 GB  =  0.64 %
```

### 8.3 What the overhead becomes at Q4 — the decisive comparison `[I]`

Denominators for Qwen3.6-27B (total params 27.78 B):

| denominator | bytes | proxy 5.704 GB ÷ denominator |
|---|---|---|
| **FP16** weights (= Prox's own denominator) | 55.56 GB | **10.27 %** ← this is the "≈12 %" world |
| INT8 weights | 27.78 GB | 20.53 % |
| **INT4 / Q4 weights, ideal (no scales)** | **13.89 GB** | **41.07 %** |
| **Q4 weights, repo-measured budget (`research/REPORT.md:271,360` "≈16 GB")** | 16 GB | **35.65 %** |
| Q4 budget per `research/REPORT.md:64` ("~17 GB / forward") | 17 GB | 33.55 % |

⇒ **The 12 % does NOT transfer to a 27B model at Q4.** At Q4 the proxy tax is **≈ 34 %–41 % of the weight budget**, i.e. **≈ 3× the 12 %** the prior report applied. The prior report's arithmetic

> *"27B @ 4-bit = 15.2 GB, +12% = 17.0 GB"*

**is methodologically wrong**: it multiplies a **4-bit** weight budget by a ratio that is **FP16-denominated**. Corrected for the verified Qwen3.6-27B: **Q4 weights ≈ 13.9–16 GB + proxy ≈ 5.70 GB ⇒ ≈ 19.6–21.7 GB of resident weights**, i.e. **far** above the 17.0 GB the prior report derived and **above the 17.76 GiB working-set limit** the repo measured on the M4 Pro (`research/REPORT.md:34`). Under the paper's own sliding-window mitigation (upper bound → 85 MiB–713 MiB), it would fit.

### 8.4 The generic formula to reuse `[I]`

For any SwiGLU dense model with `L` layers, hidden `h`, FFN intermediate `inter`, `N_total` params:

```
proxy_bytes(L)  = L · 2 · h · inter · 0.5          # INT4 proxy of W_up + W_gate
overhead_vs(p) = 100 · proxy_bytes(L) / (N_total · p/8)      # p = 16 (FP16) or 4 (Q4)
```

⇒ **Rule of thumb: overhead(Q4) ≈ 4 × overhead(FP16).** For a "27B-class" dense model with a typical FFN:hidden:intermediate ratio, overhead(FP16) ≈ 10–13 % ⇒ **overhead(Q4) ≈ 40–52 %**.

## 9. Model configs used (all fetched live, HTTP 200 unless noted)

| URL | HTTP | Used for |
|---|---|---|
| `https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json` | 200 | 12 % reconstruction (11.06 %) |
| `https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json` | 200 | 10.24 % |
| `https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json` | 200 | 11.14 % |
| `https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json` | 200 | 12.07 % |
| `https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json` | 200 | 12.80 % |
| `https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json` | 200 | 12.96 % |
| `https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json` | 200 | 27B transfer arithmetic |
| `https://huggingface.co/api/models/Qwen/Qwen3.6-27B?expand[]=safetensors` | 200 | total params 27 781 427 952 |
| `https://huggingface.co/google/gemma-3-27b/raw/main/config.json` | **401** | wanted as a second 27B; gated |
| `https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main/config.json` | **401** | gated |
| `https://huggingface.co/google/gemma-3-12b-it/raw/main/config.json` | **401** | gated |

---

## 10. What I could **NOT** verify (explicit list)

1. **The SpQt M2 Max `819.2 GB/s` derivation** — the paper gives none; H1/H2 in §5.3 are labelled hypotheses, not findings. `[NF]`
2. **An Apple page printing the exact string "819 GB/s" for M3 Ultra.** Apple's M3 Ultra *newsroom* says "over 800GB/s"; Apple's *current* Mac Studio specs page lists M5 parts (M5 Pro 153/307 GB/s, M5 Max 460/614 GB/s) and contains no "819". The exact "819GB/s" tech-spec string is therefore **not re-verified in this session** and H2 rests on secondary knowledge. `[NF]`
3. **Apple's official memory bandwidth for A19 / A19 Pro** — **not published** on the pages I read (iPhone 17 / 17 Pro newsroom + tech specs: 0 occurrences of "GB/s" / "bandwidth"). The paper's `75.8 GB/s` for "A19 pro" therefore **cannot be checked against any Apple source**. `[NF]`
4. **SpQt batch size** — never stated. `[NF]`
5. **SpQt number of generated tokens / measured decode length** — never stated (only "a short prompt of approximately 10 tokens"). `[NF]`
6. **SpQt device models, RAM configs, thermal/clock state** — never stated (only SoC classes in Table 1). `[NF]`
7. **SpQt Table 1 "A19 pro" as a real, benchmarkable A19 Pro device** — unexplained by the paper; my §3.4 checks are `[I]`, not `[V]`. `[NF]`
8. **Prox's own derivation of the 12 %** — absent from the paper; §7's reconstruction is `[I]`, though it matches 6/6 verified configs to within 10–13 %. `[NF]`
9. **Whether Prox's 12 % denominator excludes embeddings** — not determinable from the paper; both candidate denominators land in 11–13.5 %. `[NF]`
10. **Llama-3-8B / Gemma-3-12B configs** (both HF-gated, HTTP 401) — their individual ratios were not added to the §7 table; this does not affect the conclusion (6/6 ungated models agree). `[NF]`
11. **A real 27B-class *dense* HF model besides Qwen3.6-27B** — `google/gemma-3-27b` is 401-gated. The §8 arithmetic therefore uses the project's own verified Qwen3.6-27B, plus an explicitly-stated **assumption** for the briefing's literal "27 layers" variant (that a 27-layer model would use the same `h`/`inter` per layer as the verified 64-layer one). `[NF]`
12. **`819.2 GB/s` as "2× of a real Apple figure"** — 2 × 400 = 800 ≠ 819.2; 2 × 409.6 = 819.2, but 409.6 is *not* an Apple-published M2 Max figure. `[NF]`

## 11. Action items for the R2 report (evidence-level, no edits made here)

| # | Location | Current | Correct |
|---|---|---|---|
| 1 | `2026-10-r2-report.md:405` / `:794`, `-combinations.md:111`, `tools/apply_r2_b_fixes.py:178,334,338,380,396` | M2 Max = `819.2 GB/s` | **`400 GB/s` (Apple official)**; note the paper prints `819.2` and that this cell is erroneous (§5) |
| 2 | same files | "M4 Pro falls between M2 Pro (204.8) and M2 Max (819.2)" | re-anchor to **M2 Max (400)**; **the ≈1.2–1.3× extrapolation is unchanged** (§5.4) |
| 3 | `2026-10-r2-report.md:276`, `:392`; `-kernel-matrix.md:291,321`; `tools/apply_r2_b_fixes.py:126` | "27B @ 4-bit = 15.2 GB, +12% = 17.0 GB" | **12 % is FP16-denominated**; at Q4 the proxy tax is **≈34–41 %** ⇒ **≈5.70 GB extra** for Qwen3.6-27B (upper bound) ⇒ **≈19.6–21.7 GB** resident weights |
| 4 | `2026-10-r2-report.md:794` | "M2 Max 1.19× / 1.18×" | **correct**, but state that these are the **Llama-2-7B** and **Llama-3-8B** cells of the `Unified 50%` row, vs **dense Q4_K** (§3.1–3.3) |
| 5 | `2026-10-r2-report.md` "M4 Pro 是 273 GB/s" | 273 GB/s | **CORRECT — no change** (§4) |

---

### Discipline statement

Read-only. HTTP GET only (`curl` with a browser User-Agent; `https://` only). No cloning, no builds, no model downloads, no benchmarks. The only file created is this one:
`research/agent-reports/minimax/fulltext-notes/R3-S-spqt-bandwidth-prox.md`.
Scratch artefacts (`spqt.pdf/.txt/.html/.svg`, `prox.pdf/.txt/.html`, Apple pages, HF configs) were written to `/tmp` only. No other repo file was read-modified or written.
