"""Stage 2.0: does the theoretical critical path turn into real wall-clock?

Measures, on this machine, per mode:

  serial-eager   one block at a time, mx.eval after every block
                 (the historical baseline: teacher_pass / run_group style)
  serial-lazy    all 28 blocks built lazily, a single mx.eval at the end
  g2-eager       production g2 schedule, mx.eval after every group
  g2-lazy        g2 as a DAG, level-synchronous, single mx.eval
  g2-lazy-sync   same graph, mx.eval after every dependency level
  win-g2/g4      window-mid variants as DAGs, single mx.eval
  sparse-a/b     hand-built sparse window DAGs (non-contiguous preds,
                 multi-predecessor merges), single mx.eval

The eager baselines are the honest "what we ran so far" numbers; the lazy
variants are what a real runtime could do with the same graph. Everything
else (input, mask, weights, dtype) is identical across modes.

No speedup is claimed from the graph alone; this file exists to find out
whether the graph buys anything measurable.
"""

from __future__ import annotations

import argparse
import json
import platform
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
    g2_reference,
    make_dag,
    window_group_dag,
)

SPARSE_A = "spa@10-18:0-9|0-9|10|11|10,12|11,13|14|15"       # window depth 4
SPARSE_B = "spb@10-18:0-9|0-9|0-9|0-9|10,11|12,13|10,13|11,12"  # window depth 2


def run_serial_eager(model, h0, mask, mx):
    h = h0
    for block in model.model.layers:
        h = block(h, mask, None)
        mx.eval(h)
    return h


def run_g2_eager(model, h0, mask, mx):
    return g2_reference(model, h0, mask, mx)


def bench(fn, reps: int, warmup: int, mx) -> dict:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p10_ms": times[max(0, int(0.10 * len(times)) - 1)],
        "p90_ms": times[min(len(times) - 1, int(0.90 * len(times)))],
        "min_ms": times[0],
        "reps": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s20_cal.jsonl"))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_0/bench"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)

    sample = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
    h0 = embed(model, ids, mx)
    mask = create_attention_mask(h0, None)
    n_tokens = int(ids.shape[1])

    chain = chain_dag(n)
    g2 = group_dag(n, 2, name="prefix0-g2")
    g4 = group_dag(n, 4, name="prefix0-g4")
    win2 = window_group_dag(n, 10, 18, 2, name="win-mid-g2")
    win4 = window_group_dag(n, 10, 18, 4, name="win-mid-g4")
    spa = make_dag(n, SPARSE_A)
    spb = make_dag(n, SPARSE_B)

    modes = [
        ("serial-eager", lambda: run_serial_eager(model, h0, mask, mx), chain),
        ("serial-lazy", lambda: dag_forward(model, h0, chain, mask, mx), chain),
        ("g2-eager", lambda: run_g2_eager(model, h0, mask, mx), g2),
        ("g2-lazy", lambda: dag_forward(model, h0, g2, mask, mx), g2),
        ("g2-lazy-sync", lambda: dag_forward(model, h0, g2, mask, mx,
                                             sync_levels=True), g2),
        ("g4-lazy", lambda: dag_forward(model, h0, g4, mask, mx), g4),
        ("win-g2-lazy", lambda: dag_forward(model, h0, win2, mask, mx), win2),
        ("win-g4-lazy", lambda: dag_forward(model, h0, win4, mask, mx), win4),
        ("sparse-a-lazy", lambda: dag_forward(model, h0, spa, mask, mx), spa),
        ("sparse-b-lazy", lambda: dag_forward(model, h0, spb, mask, mx), spb),
    ]

    results = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fn, dag in modes:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        r = bench(fn, args.reps, args.warmup, mx)
        r["critical_path"] = dag.critical_path
        r["executed_blocks"] = dag.executed_blocks
        r["graph"] = dag.name
        if hasattr(mx, "get_peak_memory"):
            r["peak_memory_mb"] = float(mx.get_peak_memory()) / 1e6
        r["tokens_per_s"] = n_tokens / (r["median_ms"] / 1000.0)
        results[name] = r
        print(f"{name:16s} depth={dag.critical_path:2d} median={r['median_ms']:7.2f}ms "
              f"p10={r['p10_ms']:7.2f} p90={r['p90_ms']:7.2f} "
              f"peak={r.get('peak_memory_mb', float('nan')):8.1f}MB "
              f"tok/s={r['tokens_per_s']:9.0f}", flush=True)

    base = results["serial-eager"]["median_ms"]
    for name, r in results.items():
        r["speedup_vs_serial_eager"] = base / r["median_ms"]

    payload = {
        "stage": "2.0 concurrency bench",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "n_tokens": n_tokens,
        "input": str(args.input),
        "reps": args.reps,
        "warmup": args.warmup,
        "mlx_version": getattr(mx, "__version__", "UNKNOWN"),
        "platform": platform.platform(),
        "has_synchronize": hasattr(mx, "synchronize"),
        "graphs": {name: {"critical_path": dag.critical_path,
                          "preds": [sorted(p) for p in dag.preds]}
                   for name, _, dag in modes},
        "modes": results,
        "note": ("Wall-clock only. A shorter critical path is not a speedup "
                 "until it shows up here."),
        "git_commit": git_sha(),
    }
    (args.output / "bench.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.output / 'bench.json'}")


if __name__ == "__main__":
    main()
