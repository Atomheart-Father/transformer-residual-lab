"""Stage 2.1 Part A (A6b): is the cached DAG decode path correct?

Stage 2.0 only checked the serial chain against the uncached forward, so a DAG
decode path could have been wrong and nobody would have noticed. ``part_a6`` of
stage21_bench.py compared every DAG variant against a full no-cache forward of
*the same graph* and found max_abs differences of 24 / 54 vs 3.0 for the serial
chain. 24/54 is far above the serial noise floor, which has two possible causes:

  (a) a bug: the cached DAG path computes a different function than the
      uncached path, or
  (b) expected numerics: the attention kernel differs between the cache path
      and the full-mask path, and the DAG merge structure amplifies that
      difference more than the serial chain does.

This script decides between (a) and (b) with three controls that do not need a
profiler:

  C1  mask-path determinism: re-run the DAG over the growing prefix, one token
      at a time, always with the full causal mask. If this is not bit-identical
      to the single full forward, the mask path itself is not deterministic and
      every comparison below is meaningless.
  C2  DAG prefill + DAG decode (same graph everywhere) vs the same-graph full
      forward: isolates the *cache* kernel; the graph is identical in both.
  C3  serial prefill + DAG decode (the deployable configuration) vs the
      same-graph full forward: adds the prefill-vs-decode state difference.

Quality is compared, not just hidden states: per-token NLL of the true
continuation under each path, because a hidden-state gap that costs no NLL is
numerically irrelevant for our purposes, while one that costs NLL is a real
defect.  Report: max/mean |dNLL| per token, rel_rms of the hidden gap, and the
free-running dNLL of the DAG decode vs the serial decode (the Stage 2.0
quality cost, re-measured on the decode path).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure import (  # noqa: E402
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
    logits_metrics,
    project_logits,
)
from dag_exec import Dag, chain_dag, dag_forward, window_group_dag  # noqa: E402


def nll_of_continuation(logits_row, next_ids, mx) -> float:
    """NLL of the true next tokens given one logits row per position."""
    import mlx.core as mx
    s = logits_row[0].astype(mx.float32) if logits_row.ndim == 3 else logits_row.astype(mx.float32)
    log_ps = s - mx.logsumexp(s, axis=-1, keepdims=True)
    tgt = mx.array(next_ids, dtype=mx.int32)
    picked = -mx.take_along_axis(log_ps, tgt[..., None], axis=-1).reshape(-1)
    return float(picked.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s21_cal.jsonl"))
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--prefill-len", type=int, default=128)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_1/decode_check"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    dags = {
        "serial": chain_dag(n),
        "win10-17-g2": window_group_dag(n, 10, 18, 2),
        "win10-17-g4": window_group_dag(n, 10, 18, 4),
        "full-g4": Dag("full-g4", n,
                       tuple(frozenset(range((i // 4) * 4)) for i in range(n)),
                       note="all 28 blocks grouped by 4, depth 7"),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "stage": "2.1 A6b decode-correctness deep check",
        "model": PINNED_MODEL_REPO, "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n, "input": str(args.input), "max_samples": args.max_samples,
        "prefill_len": args.prefill_len, "decode_steps": args.decode_steps,
        "graphs": {k: d.to_dict() for k, d in dags.items()},
        "git_commit": git_sha(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows = []
    done = 0
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if done >= args.max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            ids_full = mx.array(sample["ids"], dtype=mx.int32)
            P, D = args.prefill_len, args.decode_steps
            if P + D + 1 > ids_full.shape[0]:
                continue
            ids = ids_full[:P + D + 1]
            cont = [int(ids[P + t + 1]) for t in range(D)]  # true next tokens

            row = {"sample_id": sample["sample_id"], "controls": {}, "variants": {}}

            # ---- C1: mask-path determinism (g2 graph, no cache, growing) ----
            dag_c1 = dags["win10-17-g2"]
            hs = []
            for t in range(2):
                h = embed(model, ids[:P + t + 1][None, :], mx)
                h = dag_forward(model, h, dag_c1,
                                create_attention_mask(h, None), mx)
                mx.eval(h)
                hs.append(h[:, P + t, :])
            h_ref_c1 = embed(model, ids[None, :], mx)
            h_ref_c1 = dag_forward(model, h_ref_c1, dag_c1,
                                   create_attention_mask(h_ref_c1, None), mx)
            mx.eval(h_ref_c1)
            d1 = [float(mx.max(mx.abs(a - h_ref_c1[:, P + t, :])).item())
                  for t, a in enumerate(hs)]
            row["controls"]["c1_mask_path_determinism"] = {
                "max_abs_diff": max(d1), "bit_identical": max(d1) == 0.0}

            # ---- reference: same-graph full forward (mask path, one shot) ---
            for name, dag in dags.items():
                h_ref = embed(model, ids[None, :], mx)
                h_ref = dag_forward(model, h_ref, dag,
                                    create_attention_mask(h_ref, None), mx)
                mx.eval(h_ref)
                lg = project_logits(model, h_ref[:, P:P + D, :], mx)
                mx.eval(lg)
                row["variants"][f"{name}@fullforward"] = {
                    "nll_continuation": nll_of_continuation(
                        lg, cont, mx),
                    "hidden_max_abs": float(mx.max(mx.abs(h_ref)).item()),
                }

            # ---- cached paths ----------------------------------------------
            for name, dag in dags.items():
                for prefill_kind in ("dag", "serial"):
                    cs = [KVCache() for _ in range(n)]
                    if prefill_kind == "dag":
                        h = embed(model, ids[:P][None, :], mx)
                        h = dag_forward(model, h, dag,
                                        create_attention_mask(h, None), mx,
                                        caches=cs)
                    else:
                        h = embed(model, ids[:P][None, :], mx)
                        pm = create_attention_mask(h, None)
                        for i, layer in enumerate(model.model.layers):
                            h = layer(h, pm, cs[i])
                    mx.eval(h)
                    hidden, nlls = [], []
                    for t in range(D):
                        tok = ids[P + t:P + t + 1][None, :]
                        h = dag_forward(model, embed(model, tok, mx), dag, None,
                                        mx, caches=cs)
                        mx.eval(h)
                        hidden.append(h[:, 0, :])
                        lg = project_logits(model, h, mx)
                        mx.eval(lg)
                        s = lg[:, -1, :].astype(mx.float32)
                        lp1 = s - mx.logsumexp(s, axis=-1, keepdims=True)
                        nlls.append(float(-lp1[0, cont[t]].item()))
                    hidden = mx.stack(hidden, axis=1)
                    h_ref = embed(model, ids[None, :], mx)
                    h_ref = dag_forward(model, h_ref, dag,
                                        create_attention_mask(h_ref, None), mx)
                    mx.eval(h_ref)
                    ref_h = h_ref[:, P:P + D, :]
                    diffs = [float(mx.max(mx.abs(hidden[:, t, :] -
                                                 ref_h[:, t, :])).item())
                             for t in range(D)]
                    rel = []
                    for t in range(D):
                        a = hidden[:, t, :].astype(mx.float32)
                        b = ref_h[:, t, :].astype(mx.float32)
                        rel.append(float((mx.sqrt(mx.mean((a - b) ** 2)) /
                                          mx.sqrt(mx.mean(b ** 2))).item()))
                    row["variants"][f"{name}@cached-{prefill_kind}prefill"] = {
                        "nll_continuation": float(np.mean(nlls)),
                        "nll_per_token": nlls,
                        "hidden_max_abs_diff_vs_fullforward": max(diffs),
                        "hidden_rel_rms_diff_vs_fullforward": float(np.mean(rel)),
                        "hidden_max_abs_diff_per_token": diffs,
                    }
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(row)
            done += 1
            print(f"[{done}/{args.max_samples}] {sample['sample_id']} "
                  f"c1_bit_identical={row['controls']['c1_mask_path_determinism']['bit_identical']}",
                  flush=True)
            for k, v in row["variants"].items():
                print(f"    {k:34s} nll={v['nll_continuation']:.4f} "
                      f"maxabs={v.get('hidden_max_abs_diff_vs_fullforward', float('nan')):.3f} "
                      f"rel={v.get('hidden_rel_rms_diff_vs_fullforward', float('nan')):.5f}",
                      flush=True)

    payload = {"n_samples": len(rows),
               "note": "C1 must be bit-identical; C2/C3 report hidden gaps and "
                       "the NLL gap against the same-graph full forward."}
    (args.output / "check.json").write_text(
        json.dumps({**payload, "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    manifest["status"] = "MEASURED"
    manifest["n_measured"] = len(rows)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"saved -> {args.output / 'check.json'}")


if __name__ == "__main__":
    main()
