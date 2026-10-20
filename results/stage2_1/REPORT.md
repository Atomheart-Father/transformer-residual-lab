# Stage 2.1 report — is there anything left worth researching in the frozen-weight DAG?

Scope: Qwen3-0.6B-Base BF16, 28 blocks, weights frozen, nothing trained, nothing
pruned, every candidate always executes all 28 blocks. M4 Pro 24 GB. Two
questions: **A** does block-level parallelism buy wall-clock on this machine,
and **B** is the *real* design space of the frozen DAG wider than Stage 2.0's
predefined families — i.e. was the Stage 2.0 negative result an artifact of a
too-narrow generator?

Figures: `figures/fig_single_edges.png` (the 98-edge cost map, 2.1) and
`figures/fig_depth_frontier.png` (depth/quality frontier with the val/blind
points, 2.5).

Split discipline: `s21_cal` (48 windows) fits, `s21_val` (32) selects, and
`s21_final` (64, seed 20261026, never seen by any search step) is the blind
test, run once and guarded by a marker file. All 153 pairs of split files share
zero tokens (`data257/data_manifest.json`). Quality numbers below are ΔNLL
against the teacher-forced reference on the same tokens.

---

## 1. Does the GPU gain anything from block-level parallelism? (Part A)

**No.** Measured, not assumed (`results/stage2_1/PART_A.md`, raw data in
`bench_a12` / `bench_a345` / `bench_a67`):

| evidence | result |
|---|---|
| A1 per-block cost vs length | 0.558 ms fixed + 0.0058 ms/token: decode is ~90% fixed overhead |
| A2 k independent blocks, one graph | overlap saturates: 1.25× (k=2), 1.41× (k=4), **1.46× (k=8)** |
| A4 DAG prefill vs best serial | depth 28→7 buys **5-6%**; deployable window DAGs buy 0 |
| A5 weight-stacked batched matmul | bit-identical math, but **29% slower at n=1**, 1.25× at n=256 |
| A6 decode per step | serial 7.07 ms; win-g2 6.81 (inside noise); win-g4 **7.92 (slower)** |
| A7 achieved per block | 3.4-4.0 TFLOP/s, uniform across depth (~40% of peak, inferred) |

The "(4,1,dim) stack = free batch" premise was tested rather than assumed (A5):
the math is exactly equivalent, and the timing says it does not pay where it
would have to. Bottleneck statement: decode is dominated by fixed per-block
cost — 28 blocks' weights must stream every step whatever the graph (31.5 MB
of BF16 weights per block, measured from the pinned checkpoint, 881 MB per
step), and the measured pace is already ~125 GB/s of a ~270 GB/s class
machine; the remaining
GPU capacity (A2/A7) overlaps independent blocks by at most ~1.46×, and the
merge kernels consume what is left. No profiler was available programmatically
in this environment; every utilization number here is inferred from wall-clock
plus FLOP/byte accounting and is labelled as such.

**Consequence: critical path is a logical quantity in this project, not a
latency one.** A structure can only be a quality/latency trade if the machine
changes (or a fused level kernel is hand-written), which is out of scope.

## 2. Is the search space wider than Stage 2.0's predefined families? (Part B)

**Yes — but not in the direction that matters, and the reason is now precise.**

Formulation used throughout: a graph is fully described by its *dropped edges*
relative to the serial chain. Dropping edge *(i, j)* costs exactly **one**
missed update, whatever the distance; depth only drops when the dropped edge is
the one carrying the critical path. That turns the vague "regular groups vs
irregular graphs" debate into a measurable map.

### 2.1 What one dropped edge costs (all 98 window edges, 32 cal samples)

* **Distance effect, ~4×**: mean ΔNLL per drop falls from 0.036 nats at
  distance 1 to 0.0087 at distance 8, then flattens at a 0.004-0.008 floor.
  Remote ancestors are cheap to lose — but they also buy **no depth**.
* **Position effect**: perturbing later blocks is cheaper (block 11 mean 0.0215
  → block 17 mean 0.0113) — less suffix left to amplify the error.
* **Outliers**: dropping *(15,2)* costs 0.070 nats, more than *any*
  depth-reducing drop; dropping *(16,1)* costs 0.0007. Distance is not the
  whole story.
* **Depth-reducing drops (distance 1)**: 0.019 (drop 14,13) … 0.050 (drop
  17,16) — the cheapest place to compress the window is its middle.
* The teacher-materialized probe (Stage 2.0 + the 40 new distant edges) predicts
  this whole map with **Spearman 0.84** — the local probe is a usable
  *ranking* instrument, not just a diagnostic.

### 2.2 The predefined families, re-measured on fresh calibration data

Same structures, fresh `s21_cal`: g2 (depth 4, 4 misses) 0.179, mix3 (3, 7)
0.346, g4 (2, 12) 0.567 nats; a single early pair (depth 7, 1 miss) 0.031.
Stage 2.0's ranking reproduces exactly.

### 2.3 What the wider search found

* **Greedy depth-constrained search** (each generation must cut the window's
  schedulable depth by one): depth 8→7→6 and then it stops, because no single
  further edge drop compresses the window. Its depth-6 structure
  (drops 14,13 and 16,15) costs 0.066.
* **The single-move closure is tiny**: the number of distinct structures
  reachable by depth-reducing single drops is 7 / 15 / 10 / **1** at depths
  7 / 6 / 5 / 4. The unique depth-4 member *is* g2. Inside that search model,
  contiguous even pairing is not merely good, it is the only option — Stage
  2.0's answer to "which depth-4 graph" was correct, and the negative result
  was not caused by the family choice.
* **Depth-matched irregular controls** (every drop must compress the window,
  random choice among the legal ones): at depth 6, random structures span
  0.066-0.100 around the greedy point; at depth 5 (3 misses) 0.118-0.179,
  i.e. irregular structures at depth 5 cost *more* than the best depth-6
  structure. Nothing irregular beats the depth/quality Pareto points of the
  closure. (An earlier matched-*miss* comparison was invalid and is reported as
  such: 4 random drops mostly land on remote edges, which cost almost nothing
  while saving no scheduling step — depth must be matched, misses must not.)
* **Exhaustive enumeration of the reachable space** (all prefix-truncation
  graphs — the maximal-quality predecessor set for every dependency-level
  pattern, including the non-monotone patterns only multi-edge drops can
  create — with ≤6 missed updates, window depth ≤5: **754 structures**, all
  measured on the same 32 cal samples):

| window depth | n | best ΔNLL (misses, level sizes) | median | p90 |
|---|---|---|---|---|
| 5 | 648 | 0.1131 (5, [1,1,2,3,1]) | 0.1800 | 0.2290 |
| 4 | 106 | 0.1758 (5, [2,2,1,3]) | 0.2213 | 0.2704 |

  At depth 4, the **unique best 4-miss structure (0.1788) is exactly g2** —
  among all 106 depth-4 graphs measured, nothing with g2's budget beats it.
  The overall best depth-4 candidate (0.1758) spends one extra missed update
  for −0.003 nats; its apparent advantage did not survive validation (2.5).

### 2.4 Training-free merge semantics: closed, negative

Scaling predecessor updates by γ^(distance) inside the input merge (γ<1 favours
recent updates) is catastrophic: γ=0.9 costs 9.28 nats on g2 and 9.72 on g4
(top-1 agreement → 0.004), γ=0.8 costs 10.0-10.3. γ=1 reproduces the plain
fp32 sum exactly, both in a unit test and in these runs. The residual stream's
plain additivity is load-bearing; the merge rule is not a free knob.

### 2.5 Selection freeze and one-shot blind test

Nine finalists on `s21_val` (32 samples), ranking frozen into
`results/stage2_1/selection.json`; then the five selected structures on
`s21_final` (64 samples), run exactly once (`search_blind/ONESHOT`):

| structure (depth, misses) | val ΔNLL | blind ΔNLL | blind vs g2 (paired ci95) |
|---|---|---|---|
| single-pair (7, 1) | 0.0430 | 0.0422 | −0.147 [−0.156, −0.138] |
| greedy-depth6 (6, 2) | 0.0672 | 0.0752 | −0.114 [−0.126, −0.102] |
| best-depth5-3miss (5, 3) [1,1,2,2,2] | 0.1082 | 0.1195 | −0.070 [−0.077, −0.063] |
| g2 (4, 4) | 0.1786 | 0.1893 | — |

Two things this decides. (1) **The depth/quality frontier is monotone and
smooth**: 0.042 → 0.075 → 0.120 → 0.189 as depth goes 7 → 6 → 5 → 4; every
step costs ~0.03-0.07 nats, and no measured structure anywhere in the space
beats the frontier. The depth-5 irregular structure is a *new* frontier point
(Stage 2.0 had no depth-5 window family; naive interpolation between
single-pair and g2 would predict ~0.134 on val, the measured point is 0.108),
but it extends the curve — it does not break its shape. (2) **Selection noise
is exactly the scale of the "wins" the wider search produced**: the best
exhaustive depth-4 candidate (0.1758 on cal) fell to 0.1864 on val, its
advantage over g2 vanishing inside the paired CI [−0.0028, +0.0176]. Any
claimed improvement below ~0.01-0.02 nats on 32 windows is not selectable.
Val → blind drift is small (≤0.011 nats) and preserves the ranking exactly.

## 3. Verified results vs. explanatory hypotheses

Verified by measurement: every claim in section 1; the single-edge cost map and
its distance/position structure (2.1); the closure sizes (2.3); the merge-weight
failure (2.4); the probe→free-running rank correlation (0.84 single-edge,
0.96 over 47 multi-drop structures against Σ probe semi_free, near-linear at
ΔNLL ≈ 0.58 × Σ probe); the selection-freeze outcome (2.5) including the
rejected phantom improvement.

Still explanatory (mechanism, not measurement): the weight-bandwidth model of
decode (31.5 MB/block × 28 blocks = 881 MB per step vs ~125 GB/s inferred
  pace); the claim that
merge overhead is what consumes the overlap headroom (consistent with A4/A6 but
not isolated); why distance-8 is the floor of the distance effect; why specific
edges such as (15,2) are expensive.

## 4. Which Stage 2.0 conclusions are strengthened / corrected / overturned

* **Strengthened**: "every upstream update matters" — now 78/108 edges probed
  directly at every distance 1-12 (was 40/108, distances ≤6), and all 98
  single-edge drops measured free-running: no free edge exists.
* **Strengthened**: quality ≈ function of critical-path depth — the closure
  result (2.3) shows why the predefined families were the right shape.
* **Corrected**: "the GPU is saturated, so no parallelism is possible" — the
  device is *not* saturated (A7 ≈ 40% of peak); whole-block overlap is merely
  capped at ~1.46× by weight bandwidth and occupancy, and merge kernels eat the
  rest. The right bottleneck statement is fixed-cost + weight streaming, not
  "the GPU is busy".
* **Corrected**: the Stage 2.0 decode check covered only the serial chain. The
  DAG cache path is verified correct here (three controls, `decode_check`), and
  the deployable serial-prefill + DAG-decode configuration is identified as
  better than running the DAG in prefill as well (g2: 2.81 vs 2.93 nats).
* **Overturned**: nothing in the qualitative conclusions. The Stage 2.0 claim
  "contiguous even pairing is miss-minimal at every depth" survives as a
  closure fact, not as a sampled observation — but it was never a global
  optimality claim, and the exhaustive enumeration now tests exactly that
  (2.3).

## 5. Continue / shrink / stop?

The instruction asked us to place the project in one of four categories and we
can now do so from measurement, not opinion:

1. *approximation unacceptable* — **No**: free-running ΔNLL on fresh data
   tracks the frontier smoothly; the serial path itself is exact (S0).
2. *approximation acceptable but too slow* — **No**: nothing in the DAG
   executor is slow relative to what it buys; the cost is the missed updates,
   not the machinery.
3. *GPU can accelerate but no better DAG exists* — **This one, with the
   stronger form**: the GPU does **not** accelerate (Part A is negative at the
   root — the premise "parallel-ready blocks ⇒ wall-clock gain" fails at
   launch overhead and weight streaming), *and* the exhaustive search shows no
   meaningfully better DAG exists (754 structures; the frontier is smooth; at
   matched depth the Stage 2.0 families are optimal once selection noise is
   priced in).
4. *both have continuing value* — No.

**Decision: stop (方案 A 终止), in the strong sense that both premises of the
program are now measured to be false on this hardware and this model scale.**
Not "paused for better hardware" as a hedge — the two blocking facts are
structural: (a) decode latency is fixed per-block weight streaming plus launch
overhead, which no predecessor-set change touches; (b) the quality cost of
every dropped update is positive, position- and distance-dependent, and the
reachable depth/quality frontier is already traced by simple structures. The
one honest caveat: (a) is hardware-specific (M4 Pro, MLX, no custom kernels);
a machine or runtime where concurrency is free at the block level would reopen
question 3 but not question 2 — the frontier result is hardware-independent.

What survives from Stage 2.1 as durable instruments: the drop-formulation
search space (each edge = one miss, exact), the 98-edge free-running cost map,
the probe predictor (Spearman 0.84 single-edge, 0.96 multi-drop vs probe-sum),
the closure argument (g2 is the unique best 4-miss depth-4 graph), the
three-control decode-correctness protocol, and the cal/val/final + one-shot
blind discipline that correctly rejected a −0.003 nat phantom improvement.

---

## Limitations

* No profiler in this environment: GPU utilization, bandwidth, and peak
  figures are inferred from wall-clock accounting, explicitly marked.
* All Part B measurements use the window [10,18) — the same first-stage window
  as Stage 2.0. Position effects between windows are large (Stage 2.0 measured
  a 40× spread), so the *numbers* do not transfer to other windows; the
  *mechanisms* (distance effect, closure shape, merge weight failure) are
  window-independent statements about the executor.
* The exhaustive enumeration covers prefix truncations with ≤6 missed updates;
  structures with ≥7 misses are strictly worse than their 6-miss projections
  by the same measured cost map, but were not individually measured.
* The probe→multi-drop validation (Spearman 0.96 over 47 structures, linear
  map ΔNLL ≈ 0.58 × Σ probe semi_free) is an additivity observation on
  measured multi-drop points, not a proof for arbitrary graphs; interactions
  between simultaneous drops exist (e.g. the depth-5 structures beat naive
  single-edge predictions) but are second-order on this window.
* ΔNLL is measured on wikitext-2 windows; C4/IMDB corpora were not re-run in
  Stage 2.1 (the Stage 2.0 blind set is exhausted and may not be reused).
