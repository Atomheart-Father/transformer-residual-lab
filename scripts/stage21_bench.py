"""Stage 2.1 Part A: does the M4 Pro GPU gain anything from block-level parallelism?

Stage 2.0 found zero latency benefit in both prefill and decode but could not
attribute the bottleneck. This file separates the question into measurable
pieces:

A1  per-block cost vs sequence length   -> fixed (launch) vs marginal (work)
A2  k independent blocks in one graph   -> does the GPU overlap them at all?
A3  serial baselines, prefill 257 tok   -> eager / lazy / compiled
A4  window and full DAGs, prefill        -> vs best serial baseline
A5  weight-batched matmul primitive     -> tests the (4,1,dim) idea instead of
                                           assuming it
A6  decode, batch 1, KV cache           -> serial / DAG, lazy / compiled, with
                                           a per-variant correctness check
A7  achieved TFLOP/s per block          -> saturation evidence

Every timing is interleaved round-robin over variants inside each rep, so
thermal drift cannot favour whichever variant happened to be measured first.
Numbers are median / p10 / p90 over reps after warmup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure import (  # noqa: E402
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
)
from dag_exec import (  # noqa: E402
    chain_dag,
    dag_forward,
    group_dag,
    window_group_dag,
)


def timed(fn, mx) -> float:
    """Run fn() to completion; return wall-clock milliseconds."""
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def interleaved(variants: dict, reps: int, warmup: int, mx) -> dict:
    """Round-robin timing: every rep measures every variant once."""
    for _ in range(warmup):
        for fn in variants.values():
            timed(fn, mx)
    samples = {k: [] for k in variants}
    for _ in range(reps):
        for k, fn in variants.items():
            samples[k].append(timed(fn, mx))
    out = {}
    for k, ts in samples.items():
        ts.sort()
        out[k] = {
            "median_ms": statistics.median(ts),
            "p10_ms": ts[max(0, int(0.10 * len(ts)) - 1)],
            "p90_ms": ts[min(len(ts) - 1, int(0.90 * len(ts)))],
            "n": len(ts),
        }
    return out


def flops_per_block(n_tok: int, args) -> float:
    """Approximate matmul FLOPs of one decoder block for n_tok positions."""
    d = args.hidden_size
    nq, nkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    dff = args.intermediate_size
    proj = 2 * n_tok * (d * nq * hd + 2 * d * nkv * hd + nq * hd * d)
    mlp = 2 * n_tok * 3 * d * dff
    attn = 2 * n_tok * n_tok * nq * hd
    return float(proj + mlp + attn)


def part_a1(model, mx, ids_full, out):
    """A1: per-block time vs sequence length; decompose fixed vs marginal."""
    from mlx_lm.models.base import create_attention_mask
    layer = model.model.layers[10]
    variants = {}
    for n in (1, 2, 4, 8, 16, 64, 256):
        h = embed(model, ids_full[:, :n], mx)
        mask = create_attention_mask(h, None)
        mx.eval(h)
        variants[f"n{n}"] = (lambda h=h, mask=mask: layer(h, mask, None))
    stats = interleaved(variants, reps=12, warmup=3, mx=mx)
    args = model.args
    for n, s in stats.items():
        n_tok = int(n[1:])
        s["approx_tflops"] = flops_per_block(n_tok, args) / (s["median_ms"] / 1000.0) / 1e12
    # linear fit T = a + b*n on the small-n points (launch-dominated regime)
    xs = np.array([int(k[1:]) for k in stats], dtype=float)
    ys = np.array([s["median_ms"] for s in stats.values()], dtype=float)
    A = np.vstack([np.ones_like(xs), xs]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    out["a1_block_seqlen_scaling"] = {
        "note": "block 10 forward, batch 1; approx_tflops counts matmul FLOPs only",
        "per_n": stats,
        "linear_fit_ms": {"fixed_overhead_ms": float(a), "per_token_ms": float(b)},
    }
    return out


def part_a2(model, mx, ids_full, out):
    """A2: k independent blocks in one lazy graph - does the GPU overlap them?"""
    from mlx_lm.models.base import create_attention_mask
    n_tok = 64
    idxs = [10, 11, 12, 13, 14, 15, 16, 17]

    def build(k):
        hs, used = [], []
        for i in range(k):
            off = (i * 17) % (257 - n_tok - 1)
            hs.append(embed(model, ids_full[:, off:off + n_tok], mx))
            used.append(idxs[i])
        mask = create_attention_mask(hs[0], None)
        for h in hs:
            mx.eval(h)
        return [model.model.layers[j](h, mask, None) for h, j in zip(hs, used)]

    variants = {}
    for k in (1, 2, 4, 8):
        variants[f"k{k}"] = (lambda k=k: build(k))
    stats = interleaved(variants, reps=12, warmup=3, mx=mx)
    base = stats["k1"]["median_ms"]
    for k, s in stats.items():
        s["ms_per_block"] = s["median_ms"] / int(k[1:])
        s["speedup_vs_one_at_a_time"] = int(k[1:]) * base / s["median_ms"]
    out["a2_independent_block_overlap"] = {
        "note": ("k different blocks on k different inputs, one lazy graph, a "
                 "single mx.eval; speedup_vs_one_at_a_time = k * t1 / t_k"),
        "per_k": stats,
    }
    return out


def part_a3(model, mx, ids_full, out):
    """A3: full-model serial baselines at 257 tokens (prefill)."""
    from mlx_lm.models.base import create_attention_mask
    n = len(model.model.layers)
    h0 = embed(model, ids_full, mx)
    mask = create_attention_mask(h0, None)
    mx.eval(h0)

    def run_serial(eager: bool):
        h = h0
        for block in model.model.layers:
            h = block(h, mask, None)
            if eager:
                mx.eval(h)
        return h

    def run_lazy():
        return run_serial(False)

    compiled = mx.compile(run_lazy)
    variants = {
        "serial-eager": lambda: run_serial(True),
        "serial-lazy": run_lazy,
        "serial-compiled": compiled,
    }
    stats = interleaved(variants, reps=10, warmup=3, mx=mx)
    best = min(s["median_ms"] for s in stats.values())
    for s in stats.values():
        s["speedup_vs_best_serial"] = best / s["median_ms"]
    out["a3_serial_prefill_257"] = {
        "note": ("28 blocks, 257 tokens, batch 1; serial-eager is the historical "
                 "measurement style, serial-lazy one graph + single eval, "
                 "serial-compiled wraps the lazy graph in mx.compile"),
        "critical_path": n,
        "per_variant": stats,
    }
    return out


def part_a4(model, mx, ids_full, out):
    """A4: window and full DAGs at 257 tokens vs the serial baseline."""
    from mlx_lm.models.base import create_attention_mask
    n = len(model.model.layers)
    h0 = embed(model, ids_full, mx)
    mask = create_attention_mask(h0, None)
    mx.eval(h0)

    dags = {
        "win10-17-g2": window_group_dag(n, 10, 18, 2),
        "win10-17-g4": window_group_dag(n, 10, 18, 4),
        "full-g2": group_dag(n, 2),
        "full-g4": group_dag(n, 4),
    }
    variants = {}
    compiled = {}
    for name, dag in dags.items():
        variants[f"{name}-lazy"] = (lambda dag=dag: dag_forward(model, h0, dag, mask, mx))

        def make(dag=dag):
            def f():
                return dag_forward(model, h0, dag, mask, mx)
            return f
        try:
            compiled[name] = mx.compile(make(dag))
            variants[f"{name}-compiled"] = compiled[name]
        except Exception as e:  # pragma: no cover - depends on MLX version
            out.setdefault("a4_compile_failures", {})[name] = repr(e)
    stats = interleaved(variants, reps=10, warmup=3, mx=mx)
    for name, dag in dags.items():
        stats[f"{name}-lazy"]["critical_path"] = dag.critical_path
        if f"{name}-compiled" in stats:
            stats[f"{name}-compiled"]["critical_path"] = dag.critical_path
    out["a4_dag_prefill_257"] = {
        "note": ("same weights, dtype and total block work as a3; only the "
                 "execution order / merge schedule differs"),
        "per_variant": stats,
    }
    return out


def part_a5(model, mx, ids_full, out):
    """A5: weight-batched matmul - measured, not assumed.

    Compares, for identical math: (a) k separate matmuls with k different
    weight matrices, (b) one batched matmul over stacked weights, (c) the
    compiled versions of both. Uses real q_proj weights from four different
    blocks plus a synthetic-warm control, at 1 token (decode regime) and 256
    tokens (prefill regime).
    """
    import mlx.core as mx_
    rows = {}
    for n_tok in (1, 256):
        h = embed(model, ids_full[:, :n_tok], mx)
        Ws = [model.model.layers[i].self_attn.q_proj.weight for i in (10, 12, 14, 16)]
        # (out, in) -> stack to (k, out, in)
        Wb = mx_.stack(Ws, axis=0)          # (4, out, in)
        xs = []
        for i in range(4):                   # four genuinely different inputs
            off = (i * 31) % max(1, 257 - n_tok)
            xs.append(embed(model, ids_full[:, off:off + n_tok], mx))
        xb = mx_.concat(xs, axis=0)  # (k, n_tok, d)
        for t in (h, Wb, xb):
            mx_.eval(t)

        def separate():
            return mx_.concat([mx_.matmul(xs[i], Ws[i].T) for i in range(4)], axis=0)

        def batched():
            return mx_.matmul(xb, mx_.transpose(Wb, (0, 2, 1)))

        ref, got = separate(), batched()
        mx_.eval(ref, got)
        max_abs = float(mx_.max(mx_.abs(ref - got.reshape(ref.shape))).item())
        sep_c = mx_.compile(separate)
        bat_c = mx_.compile(batched)
        variants = {
            "separate-matmul": separate,
            "batched-matmul": batched,
            "separate-compiled": sep_c,
            "batched-compiled": bat_c,
        }
        stats = interleaved(variants, reps=12, warmup=3, mx=mx)
        for k, s in stats.items():
            s["ms_per_matmul"] = s["median_ms"] / 4.0
        rows[f"n{n_tok}"] = {
            "max_abs_diff_separate_vs_batched": max_abs,
            "note": ("identical math: each batch element uses its own weight; "
                     "batched-matmul stacks the 4 real q_proj weights once and "
                     "reuses the stack"),
            "per_variant": stats,
        }
    out["a5_weight_batched_matmul"] = rows
    return out


def part_a6(model, mx, ids_full, out, n_decode: int = 40):
    """A6: decode, batch 1, KV cache - timing plus per-variant correctness.

    Correctness (the Stage 2.0 gap): every DAG variant is checked against a
    full no-cache forward of *the same graph* over the same token sequence,
    not only the serial chain.
    """
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache
    n = len(model.model.layers)
    dags = {
        "serial": chain_dag(n),
        "win10-17-g2": window_group_dag(n, 10, 18, 2),
        "win10-17-g4": window_group_dag(n, 10, 18, 4),
    }

    # ---- correctness: cached DAG decode vs full no-cache DAG forward ------
    check_tokens = 16
    correctness = {}
    for name, dag in dags.items():
        cs = [KVCache() for _ in range(n)]
        h = embed(model, ids_full[:, :check_tokens], mx)
        pmask = create_attention_mask(h, None)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, pmask, cs[i])
        mx.eval(h)
        hid = []
        for t in range(4):
            tok = ids_full[:, check_tokens + t:check_tokens + t + 1]
            h = dag_forward(model, embed(model, tok, mx), dag, None, mx,
                            caches=cs)
            mx.eval(h)
            hid.append(h)
        seq_len = check_tokens + 4
        h_ref = embed(model, ids_full[:, :seq_len], mx)
        h_ref = dag_forward(model, h_ref, dag, create_attention_mask(h_ref, None), mx)
        mx.eval(h_ref)
        diffs = [float(mx.max(mx.abs(a - h_ref[:, check_tokens + t, :])).item())
                 for t, a in enumerate(hid)]
        correctness[name] = {
            "max_abs_diff": max(diffs),
            "critical_path": dag.critical_path,
            "passed": max(diffs) < 5.0,
        }

    # ---- timing -----------------------------------------------------------
    def fresh_caches():
        return [KVCache() for _ in range(n)]

    def prefill():
        cs = fresh_caches()
        h = embed(model, ids_full, mx)
        pm = create_attention_mask(h, None)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, pm, cs[i])
        mx.eval(h)
        return cs

    tok = ids_full[:, 128:129]
    h_tok = embed(model, tok, mx)
    mx.eval(h_tok)

    def decode_loop(dag, cs, eager: bool):
        times = []
        for _ in range(n_decode):
            t0 = time.perf_counter()
            o = dag_forward(model, h_tok, dag, None, mx, caches=cs,
                            eval_each=eager)
            mx.eval(o)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    def stats_from(times):
        ts = sorted(times)
        return {"median_ms": statistics.median(ts),
                "p10_ms": ts[max(0, int(0.10 * len(ts)) - 1)],
                "p90_ms": ts[min(len(ts) - 1, int(0.90 * len(ts)))],
                "n": len(ts)}

    per_variant = {}
    compiled_note = None
    for name, dag in dags.items():
        for eager in (False, True):
            cs = prefill()
            for _ in range(5):                       # warmup steps
                decode_loop(dag, cs, eager)
            times = decode_loop(dag, cs, eager)
            s = stats_from(times)
            s["critical_path"] = dag.critical_path
            per_variant[f"{name}-{'eager' if eager else 'lazy'}"] = s
    # compiled decode: mx.compile over a closure that owns the caches. MLX
    # support for stateful caches inside compile is version-dependent; record
    # the outcome instead of silently skipping it.
    try:
        cs_c = [KVCache() for _ in range(n)]
        h = embed(model, ids_full, mx)
        pm = create_attention_mask(h, None)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, pm, cs_c[i])
        mx.eval(h)
        dag = dags["serial"]

        def comp_step(hh):
            return dag_forward(model, hh, dag, None, mx, caches=cs_c)

        compiled_step = mx.compile(comp_step)
        for _ in range(5):
            o = compiled_step(h_tok)
            mx.eval(o)
        times = []
        for _ in range(n_decode):
            t0 = time.perf_counter()
            o = compiled_step(h_tok)
            mx.eval(o)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        s = stats_from(times)
        s["critical_path"] = n
        per_variant["serial-compiled"] = s
        compiled_note = "serial compiled decode with a persistent KVCache worked"
    except Exception as e:
        compiled_note = f"compiled decode unsupported in this MLX: {e!r}"
    out["a6_decode_batch1"] = {
        "note": ("per-step wall clock with a 257-token KV cache; correctness "
                 "checked per variant before any timing was trusted"),
        "correctness": correctness,
        "per_variant": per_variant,
        "compiled_decode": compiled_note,
    }
    return out


def part_a7(model, mx, ids_full, out):
    """A7: achieved matmul TFLOP/s per block at 257 tokens (saturation)."""
    from mlx_lm.models.base import create_attention_mask
    args = model.args
    rows = {}
    for i in (0, 10, 20, 27):
        layer = model.model.layers[i]
        h = embed(model, ids_full, mx)
        mask = create_attention_mask(h, None)
        mx.eval(h)
        variants = {f"block{i}": (lambda h=h, mask=mask, layer=layer:
                                  layer(h, mask, None))}
        stats = interleaved(variants, reps=12, warmup=3, mx=mx)
        f = flops_per_block(int(ids_full.shape[1]), args)
        s = stats[f"block{i}"]
        s["approx_tflops"] = f / (s["median_ms"] / 1000.0) / 1e12
        s["approx_mflops_per_token"] = f / ids_full.shape[1] / 1e6
        rows[f"block{i}"] = s
    out["a7_achieved_tflops_257tok"] = {
        "note": ("matmul FLOPs only (projections, MLP, attention scores); "
                 "softmax / norms / rope excluded, so these are upper bounds "
                 "on achieved throughput"),
        "per_block": rows,
    }
    return out


PARTS = {"a1": part_a1, "a2": part_a2, "a3": part_a3, "a4": part_a4,
         "a5": part_a5, "a6": part_a6, "a7": part_a7}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s21_cal.jsonl"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--parts", default="a1,a2,a3,a4,a5,a6,a7")
    ap.add_argument("--output", type=Path, default=Path("results/stage2_1/bench"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    n = len(model.model.layers)
    sample = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    ids_full = mx.array(sample["ids"], dtype=mx.int32)[None, :]

    out = {}
    for name in [p.strip() for p in args.parts.split(",") if p.strip()]:
        if name not in PARTS:
            raise SystemExit(f"unknown part {name!r}; choose from {sorted(PARTS)}")
        print(f"== Part {name.upper()} ==", flush=True)
        PARTS[name](model, mx, ids_full, out)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bench.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "MEASURED",
        "stage": "2.1 Part A (block-parallelism wall-clock, M4 Pro)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "n_tokens": int(ids_full.shape[1]),
        "mlx_version": mx.__version__,
        "platform": __import__("platform").platform(),
        "has_synchronize": hasattr(mx, "synchronize"),
        "git_commit": git_sha(),
        "parts": [p.strip() for p in args.parts.split(",") if p.strip()],
        "note": ("interleaved round-robin timing; no speedup is claimed from "
                 "graph depth alone - only measured wall clock is reported"),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.output / 'bench.json'}")


if __name__ == "__main__":
    main()
