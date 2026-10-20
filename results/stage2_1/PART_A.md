# Stage 2.1 Part A — does the M4 Pro GPU gain anything from block-level parallelism?

Question asked: 2-4 blocks that are already independent, with different frozen
weights, run concurrently or fused — is there a *reproducible* wall-clock
advantage in prefill and in batch-1 decode? If not, what is the bottleneck?

Everything below is measured on the pinned Qwen3-0.6B-Base BF16 (28 blocks),
weights frozen, batch 1, `mx.eval` + synchronize around every measurement,
round-robin interleaved timing inside each rep, medians over reps after
warmup. Same weights, same dtype, same total block work in every comparison:
only the execution schedule differs. Raw numbers: `results/stage2_1/bench_a12`,
`bench_a345`, `bench_a67`; decode correctness: `decode_check`.

## A1 — per-block cost vs sequence length (fixed vs marginal)

Block 10 forward, batch 1, no cache:

| tokens | median ms | approx TFLOP/s |
|---:|---:|---:|
| 1 | 0.472 | 0.067 |
| 2 | 0.678 | 0.093 |
| 4 | 0.654 | 0.193 |
| 8 | 0.675 | 0.373 |
| 16 | 0.612 | 0.824 |
| 64 | 0.762 | 2.666 |
| 256 | 2.075 | 4.011 |

Linear fit over all points: **fixed overhead 0.558 ms/block, 0.0058 ms/token.**
At decode length the per-block cost is ~90% fixed overhead (launch + weight
streaming) and ~10% actual work. Any block-level parallelism scheme has to
attack that fixed part; the marginal part is already tiny.

## A2 — k independent blocks in one lazy graph (does the GPU overlap at all?)

k *different* blocks on k *different* inputs, one lazy graph, one `mx.eval`.
This is the direct measurement of GPU overlap capacity — no profiler needed.

| k | median ms | speedup vs one-at-a-time |
|---:|---:|---:|
| 1 | 1.053 | 1.00 |
| 2 | 1.679 | 1.254 |
| 4 | 2.978 | 1.414 |
| 8 | 5.751 | 1.464 |

Reproduced in an earlier run (1.245 / 1.394 / 1.475). **The GPU does overlap
independent blocks, but sub-linearly, and it saturates by k≈4-8 at ~1.46×.**
That is the hard ceiling for any block-level parallel scheme on this machine:
8 different-weight blocks buy 1.46×, not 8×.

## A3 — serial baselines, prefill 257 tokens

| variant | median ms |
|---|---:|
| serial-eager | 126.7 |
| serial-lazy | 116.9 |
| serial-compiled | 114.1 |

Lazy/compiled beat the historical eager measurement style by ~10%. All DAG
comparisons below are against the *best* serial baseline (compiled), not eager.

## A4 — DAG prefill vs best serial (same work, only the schedule changes)

| variant | critical path | lazy ms | compiled ms | vs serial-compiled |
|---|---:|---:|---:|---:|
| serial | 28 | 116.9 | 114.1 | 1.000 |
| win10-17-g2 | 24 | 118.0 | 118.4 | 0.96 / 0.96 |
| win10-17-g4 | 22 | 115.4 | 113.7 | 1.01 / 1.00 |
| full-g2 | 14 | 114.4 | 113.5 | 1.02 / 1.01 |
| full-g4 | 7 | 110.4 | 108.4 | **1.06 / 1.05** |

Halving the critical path from 28 to 7 buys **5-6%**. Window DAGs (the only
structured ones worth deploying) buy nothing. Prefill is work-bound, not
depth-bound: each block is already ~4 TFLOP/s and the level structure cannot
remove work, it can only overlap it — and A2 says the overlap ceiling is 1.46×
with the merge kernels eating the rest.

## A5 — the "(4,1,dim) stack = free batching" idea, tested instead of assumed

Stacking the 4 *real* q_proj weights of blocks 10,12,14,16 into `(4,out,in)`
and doing one batched matmul is **bit-identical** to 4 separate matmuls
(`max_abs_diff = 0.0`) — the math is validated. The timing is not:

| n | separate ms | batched ms | batched speedup |
|---:|---:|---:|---:|
| 1 | 0.337 | 0.436 | 0.77× (slower) |
| 256 | 1.510 | 1.208 | 1.25× |
| 256, compiled | 1.574 | 1.261 | 1.24× |

At decode scale batching is **29% slower**; at 257 tokens it buys 1.25× on one
op. Even if every op of a 4-block level fused that well, the prefill gain
would stay in the ~1.1× range. The assumption behind "stack and batch" is
false where it would matter most.

## A6 — decode, batch 1, 257-token KV cache (per-step wall clock)

| variant | critical path | lazy ms | eager ms |
|---|---:|---:|---:|
| serial | 28 | 7.07 | 14.40 |
| serial-compiled | 28 | 7.15 | — |
| win10-17-g2 | 24 | **6.81** | 23.13 |
| win10-17-g4 | 22 | 7.92 | 14.36 |

Best DAG case is 3.7% faster than serial-lazy (inside rep-to-rep noise), and
the deeper window DAG is 12% *slower* because of its merge overhead. 0.253 ms
per block per step at batch 1.

## A6b — decode correctness for DAG paths (closes the Stage 2.0 gap)

The earlier check compared cached DAG decode against a full forward and found
max_abs differences of 24 / 54 vs 3.0 for the serial chain — either a cache bug
or expected numerics. Three controls decide it (`decode_check`, 3 samples,
32 decode steps each):

* **C1 mask-path determinism**: re-running the DAG over the growing prefix with
  the full causal mask is **bit-identical** to the one-shot full forward. The
  mask path is deterministic, so the comparisons below are meaningful.
* **C2 same graph everywhere** (DAG prefill + DAG decode vs same-graph full
  forward): every variant sits at the serial chain's own noise floor —
  rel_rms ≈ 1.0-1.2%, ΔNLL ≤ 0.02 nats over 32 steps. **The cached DAG decode
  computes the same function; there is no cache bug.**
* **C3 deployable configuration** (serial prefill + DAG decode vs DAG-everywhere
  reference): hidden states diverge by up to 143 max_abs (rel_rms 0.11-0.27) —
  that is the *free-running error* the graph is supposed to have, not a defect.
  It is also **better** than running the DAG in prefill too (g2: 2.81 vs 2.93
  nats; g4: 3.40 vs 3.71).

So the 24/54 numbers were the deployable path's real free-running error against
a different-graph reference, with an absolute threshold that ignored hidden
magnitudes up to ~5000. The review comment is closed with evidence, and the
deployable decode configuration is identified as the better one.

## A7 — achieved throughput per block (257 tokens)

Blocks 0/10/20/27: 2.46 / 2.40 / 2.41 / 2.48 ms, 3.37-3.49 TFLOP/s each,
uniform across depth. Against the M4 Pro's advertised ~9 TFLOPS-class bf16
peak that is roughly 40% — **the device is not saturated by one block**, which
is consistent with A2's 1.46× overlap headroom. Caveat: no profiler was
available programmatically in this environment; the peak figure and every
utilization number here are inferred from wall-clock + FLOP accounting, not
read off a profiler. The report marks this as an explicit limitation.

## Answer to Part A's question

**There is no realizable wall-clock advantage from block-level parallelism on
this machine for this model.** Prefill: at best +5-6% for a depth-7 full DAG,
nothing for deployable window DAGs. Decode: nothing (best 3.7% inside noise;
deeper windows slower). The bottlenecks, in order:

1. **Fixed per-block cost dominates decode** (A1: 0.558 ms fixed vs 0.0058
   ms/token). Every one of the 28 blocks' weights must be streamed every step
   regardless of the graph — parallelism cannot remove bytes. Measured
   decode pace is ~0.25 ms/block with ≈43 MB of BF16 weights per block
   (inferred ≈170 GB/s against a ~270 GB/s class machine).
2. **Whole-block overlap saturates at ~1.46×** (A2) for 8 independent blocks,
   reached already at k=4. The device is not idle (A7 ≈40% of peak), but the
   remaining capacity does not scale with more parallel blocks.
3. **Merge overhead consumes the headroom** (A4): depth 28→7 gains only 5-6%,
   and window DAGs gain nothing because their merge points cost as much as the
   overlap they enable.
4. **The launch-overhead fix is not available** (A5): stacking weights into one
   batched matmul is 29% slower at decode scale and at most 1.25× on a single
   op at 257 tokens.

Consequence for the rest of the study: critical path is a *logical* quantity
here, not a latency one. Any performance claim for a DAG structure on this
hardware would have to come from a different machine or a fused-kernel
implementation, neither of which is in scope (no training, no cloud GPU).
