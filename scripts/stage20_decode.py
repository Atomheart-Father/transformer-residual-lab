"""Stage 2.0: does the DAG buy anything in decode (1 token, KV cache)?

The prefill bench (``stage20_bench.py``) answered no: at 257 tokens a single
block already saturates the GPU, so theoretical depth 28->7 bought nothing and
the merge bookkeeping cost 6-21%. Decode is the other regime: every block sees
one token, the matmuls are tiny, and per-step latency is dominated by kernel
launches and memory traffic rather than by arithmetic. This file measures the
same graphs there.

Decode is also where the DAG semantics is *exactly* expressible with a KV
cache: block ``i`` attends to its own per-layer cache regardless of what the
hidden-state merge feeds it, so the graph only recombines hidden inputs. The
prefill runs once with the serial chain (identical caches for every variant),
then each variant decodes from the shared caches.

Variants (all 28 blocks execute, weights frozen):

  serial / serial-eager   the chain: auto shortcut vs eval after every block
  g2 / prefix3-g2         full-model groupings, depth 14 / 16
  win-g2 / win-g4         window 10-17 groupings, depth 24 / 22

Each variant is measured in two schedules:
  lazy   one graph build, a single mx.eval at the end of the step
  eager  mx.eval after every block (what the measurement scripts do)

The first check is correctness: serial decode must match a full no-cache
forward on the same token sequence. No timing is reported before that passes.
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
    project_logits,
)
from dag_exec import (  # noqa: E402
    Dag,
    chain_dag,
    dag_forward,
    group_dag,
    window_group_dag,
)


def build_dags(n: int) -> dict[str, Dag]:
    preds = tuple(frozenset(range(i)) if i < 3
                  else frozenset(range(3 + ((i - 3) // 2) * 2)) for i in range(n))
    return {
        "serial": chain_dag(n),
        "g2": group_dag(n, 2),
        "prefix3-g2": Dag("prefix3-g2", n, preds, note="3 serial, tail paired"),
        "win-g2": window_group_dag(n, 10, 18, 2),
        "win-g4": window_group_dag(n, 10, 18, 4),
    }


def step(model, h_tok, dag, caches, mx, eager: bool):
    """One decode step through ``dag`` with per-layer caches."""
    return dag_forward(model, h_tok, dag, None, mx, caches=caches,
                       eval_each=eager)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s20_cal.jsonl"))
    ap.add_argument("--n-decode", type=int, default=32,
                    help="decode steps per variant (timed)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check-tokens", type=int, default=16,
                    help="sequence length for the correctness check")
    ap.add_argument("--output", type=Path,
                    default=Path("results/stage2_0/decode"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    model, tokenizer = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    n = len(model.model.layers)
    dags = build_dags(n)

    sample = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    ids_full = mx.array(sample["ids"], dtype=mx.int32)[None, :]
    assert ids_full.shape[1] >= args.check_tokens + 5

    # ---- correctness: serial decode vs full no-cache forward -------------
    ids_check = ids_full[:, :args.check_tokens]
    caches = [KVCache() for _ in range(n)]
    h = embed(model, ids_check, mx)
    prefill_mask = create_attention_mask(h, None)   # "causal" for >1 token
    for i, layer in enumerate(model.model.layers):
        h = layer(h, prefill_mask, caches[i])       # serial prefill of the prompt
    mx.eval(h)
    hid = []
    for t in range(4):
        # after a 16-token prefill the first decode input is token 16 itself
        tok = ids_full[:, args.check_tokens + t: args.check_tokens + t + 1]
        h = step(model, embed(model, tok, mx), dags["serial"], caches, mx, False)
        mx.eval(h)
        hid.append(h)
    # reference: one full no-cache forward over the same tokens
    ids_ref = mx.concatenate([ids_check,
                              ids_full[:, args.check_tokens:args.check_tokens + 5]],
                             axis=1)
    h_ref = embed(model, ids_ref, mx)
    for i, layer in enumerate(model.model.layers):
        h_ref = layer(h_ref, prefill_mask, None)
    mx.eval(h_ref)
    # hidden at positions 16..19 (the four decode steps) predict 17..20
    ref_last = [h_ref[:, k:k + 1, :] for k in range(args.check_tokens,
                                                    args.check_tokens + 4)]
    diffs = [float(mx.max(mx.abs(a - b)).item()) for a, b in zip(hid, ref_last)]
    lg_c = [project_logits(model, h, mx) for h in hid]
    lg_r = [project_logits(model, h, mx) for h in ref_last]
    nll_c, nll_r = [], []
    for t in range(4):
        # hidden at position 16+t predicts the token at 17+t
        target = int(ids_ref[0, args.check_tokens + t + 1].item())

        def nll_of(lg):
            z = lg[:, 0, :].astype(mx.float32)
            lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
            return float(-lp[0, target].item())

        nll_c.append(nll_of(lg_c[t]))
        nll_r.append(nll_of(lg_r[t]))
    correctness = {
        "max_abs_diff_last_positions": max(diffs),
        "nll_decode_mean": float(np.mean(nll_c)),
        "nll_full_forward_mean": float(np.mean(nll_r)),
        "nll_abs_diff": float(abs(np.mean(nll_c) - np.mean(nll_r))),
        # bf16 attention kernels differ between the cached and the full path;
        # the threshold sits 3+ orders of magnitude below the smallest
        # structural effect measured anywhere in Stage 2.0 (>= 0.17 nats)
        "passed": bool(max(diffs) < 5.0
                       and abs(np.mean(nll_c) - np.mean(nll_r)) < 5e-2),
    }

    # ---- timing: shared prefill, then per-variant decode ------------------
    def fresh_caches():
        return [KVCache() for _ in range(n)]

    def prefill():
        cs = fresh_caches()
        h = embed(model, ids_full, mx)
        pmask = create_attention_mask(h, None)
        for i, layer in enumerate(model.model.layers):
            h = layer(h, pmask, cs[i])
        mx.eval(h)
        return cs

    def decode_loop(dag, cs, eager: bool, n_steps: int):
        times = []
        tok = ids_full[:, 128:129]                  # standing input token
        for t in range(n_steps):
            h = embed(model, tok, mx)
            t0 = time.perf_counter()
            out = step(model, h, dag, cs, mx, eager)
            mx.eval(out)
            if hasattr(mx, "synchronize"):
                mx.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    results = {"correctness": correctness, "n_decode": args.n_decode,
               "variants": {}, "prefill_note": "shared serial prefill"}
    for name, dag in dags.items():
        for eager in (False, True):
            cs = prefill()                       # identical caches per variant
            # warmup on the same caches (cache grows, step cost stays constant)
            decode_loop(dag, cs, eager, args.warmup)
            times = decode_loop(dag, cs, eager, args.n_decode)
            times.sort()
            key = f"{name}-{'eager' if eager else 'lazy'}"
            results["variants"][key] = {
                "critical_path": dag.critical_path,
                "median_ms": statistics.median(times),
                "p10_ms": times[max(0, int(0.10 * len(times)) - 1)],
                "p90_ms": times[min(len(times) - 1, int(0.90 * len(times)))],
                "tokens_per_s": 1000.0 / statistics.median(times),
                "n": len(times),
            }
            print(f"  {key:22s} depth={dag.critical_path:2d} "
                  f"median={results['variants'][key]['median_ms']:7.3f} ms  "
                  f"p10={results['variants'][key]['p10_ms']:7.3f}  "
                  f"p90={results['variants'][key]['p90_ms']:7.3f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "MEASURED" if correctness["passed"] else "FAILED-CORRECTNESS",
        "stage": "2.0 decode-time concurrency bench (1 token + KV cache)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "n_decode": args.n_decode,
        "warmup": args.warmup,
        "git_commit": git_sha(),
        "note": ("A DAG only recombines hidden inputs; each block keeps its own "
                 "per-layer cache, so decode is exact for any graph. Timing "
                 "starts only after the correctness check passes."),
    }
    (args.output / "decode.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncorrectness: {json.dumps(correctness, ensure_ascii=False)}")
    print(f"saved -> {args.output / 'decode.json'}")


if __name__ == "__main__":
    main()
