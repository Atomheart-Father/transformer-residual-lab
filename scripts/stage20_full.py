"""Stage 2.0 experiment C part 2: full-model and window-position exploration.

Part 1 (``stage20_screen.py``) established, inside one window (blocks 10-17),
that free-running quality is essentially a function of the critical-path depth
and that no sparse DAG beats the fixed contiguous grouping at matched depth.
This script tests the same claims at full scale and across window position,
still on calibration data only (s20_cal). Selection and blind testing happen
later, on their own splits, and are not part of this file's output.

Candidate families (all 28 blocks always execute; nothing is deleted except in
the labelled deletion controls):

  win{lo}-{hi}-g2   an 8-wide window grouped in pairs at five positions
                    (all share depth 24 and 4 missed updates, so the only
                    varying factor is *where* the parallel damage happens)
  g2 / g3 / g4      uniform contiguous groupings over the whole model
  irr-a / irr-a-t3  the irregular groupings that won the Stage 1.5/1.6 scans
  prefix3-g2        three serial blocks then pairs (Stage 1.5 sweet spot)
  keep{N}           deletion controls: only N blocks execute (a different
                    axis, kept separate by label and by effective depth)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure import (  # noqa: E402
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    Plan,
    embed,
    even_keep,
    git_sha,
    logits_metrics,
    project_logits,
    run_plan,
    teacher_pass,
)
from dag_exec import Dag, chain_dag, dag_forward, group_dag, window_group_dag  # noqa: E402

MODES = ("materialized", "roadmap")

# the irregular contiguous groupings carried over from the Stage 1.5/1.6 scans
IRR = {
    "irr-a": [[0], [1], [2], list(range(3, 11)), list(range(11, 20)),
              list(range(20, 26)), [26, 27]],
    "irr-a-t3": [[0], [1], [2], list(range(3, 11)), list(range(11, 20)),
                 list(range(20, 25)), list(range(25, 28))],
}


def groups_to_preds(groups: list[list[int]]) -> dict[int, set[int]]:
    """Shared-anchor predecessors for a contiguous grouping.

    Block ``i`` in group ``[a, b]`` reads every update strictly before ``a``
    (the merge of all earlier groups), which is exactly the production
    ``group_forward`` semantics that gate 0 verified bit-for-bit.
    """
    preds: dict[int, set[int]] = {}
    start = 0
    for g in groups:
        for i in g:
            preds[i] = set(range(start))
        start = g[-1] + 1
    return preds


def candidates(n: int) -> dict[str, Dag]:
    out: dict[str, Dag] = {"serial": chain_dag(n)}
    # window position scan at matched depth (24) and matched misses (4)
    for lo in (0, 5, 10, 15, 20):
        out[f"win{lo}-{lo + 8}-g2"] = window_group_dag(n, lo, lo + 8, 2)
    # uniform and irregular full-model groupings
    out["g2"] = group_dag(n, 2)
    out["g3"] = group_dag(n, 3)
    out["g4"] = group_dag(n, 4)
    for name, groups in IRR.items():
        flat = [i for g in groups for i in g]
        assert flat == list(range(n)), (name, flat)
        preds = tuple(frozenset(groups_to_preds(groups)[i]) for i in range(n))
        out[name] = Dag(name, n, preds, note=f"irregular {groups}")
    # prefix-protected pairing (Stage 1.5 sweet spot)
    preds = tuple(frozenset(range(i)) if i < 3
                  else frozenset(range(3 + ((i - 3) // 2) * 2)) for i in range(n))
    out["prefix3-g2"] = Dag("prefix3-g2", n, preds,
                            note="3 serial blocks, tail paired")
    return out


def deletion_plans(n: int) -> dict[str, Plan]:
    out = {}
    for k in (7, 9):
        keep = even_keep(n, k)
        out[f"keep{k}"] = Plan(f"keep{k}", [[i] for i in keep], "delete", k, k,
                               sorted(set(range(n)) - set(keep)),
                               note=f"keep {k} blocks (deletion control)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s20_cal.jsonl"))
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_0/full"))
    ap.add_argument("--only", default="",
                    help="comma-separated candidate names (default: all)")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)

    dags = candidates(n)
    dels = deletion_plans(n)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        missing = only - (set(dags) | set(dels))
        if missing:
            raise SystemExit(f"unknown candidates: {sorted(missing)}")
        dags = {k: v for k, v in dags.items() if k in only}
        dels = {k: v for k, v in dels.items() if k in only}

    def dag_stat(d: Dag) -> dict:
        return {"critical_path": d.critical_path,
                "missed_updates_total": sum(i - len(d.preds[i]) for i in range(n)),
                "executed_blocks": n}

    def plan_stat(p: Plan) -> dict:
        return {"critical_path": p.effective_depth, "missed_updates_total": None,
                "executed_blocks": p.executed_blocks}

    graph_stats = {k: dag_stat(d) for k, d in dags.items()}
    graph_stats.update({k: plan_stat(p) for k, p in dels.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "stage": "2.0 experiment C part 2 (full model + window position, calibration)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "merge_modes": list(MODES),
        "input": str(args.input),
        "max_samples": args.max_samples,
        "graphs": {k: d.to_dict() for k, d in dags.items()},
        "deletions": {k: {"deleted_layers": p.deleted_layers,
                          "executed_blocks": p.executed_blocks}
                      for k, p in dels.items()},
        "graph_stats": graph_stats,
        "git_commit": git_sha(),
        "note": ("Calibration exploration only. Window scan entries share depth "
                 "24 and 4 missed updates; deletion controls execute fewer "
                 "blocks and are never compared as if they were DAGs."),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    rows = []
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if len(rows) >= args.max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
            ref_logits = model(ids)
            mx.eval(ref_logits)
            h0 = embed(model, ids, mx)
            mask = create_attention_mask(h0, None)
            states, _ = teacher_pass(model, ids, mask, mx)

            row = {"sample_id": sample["sample_id"], "variants": {}}
            for name, d in dags.items():
                for mode in MODES:
                    h = dag_forward(model, h0, d, mask, mx, merge_mode=mode)
                    lg = project_logits(model, h, mx)
                    mx.eval(lg)
                    row["variants"][f"{name}@{mode}"] = \
                        logits_metrics(ref_logits, lg, ids, mx)
            for name, p in dels.items():
                out, _ = run_plan(model, ids, mask, p, states, "fp32", mx,
                                  keep_group_records=False)
                lg = project_logits(model, out, mx)
                mx.eval(lg)
                row["variants"][name] = logits_metrics(ref_logits, lg, ids, mx)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(row)
            print(f"[{len(rows)}/{args.max_samples}] {sample['sample_id']}",
                  flush=True)

    variants = sorted({k for r in rows for k in r["variants"]})
    agg = {}
    for v in variants:
        dn = np.array([r["variants"][v]["delta_nll"] for r in rows])
        kl = np.array([r["variants"][v]["teacher_to_candidate_kl"] for r in rows])
        t1 = np.array([r["variants"][v]["top1_agreement"] for r in rows])
        agg[v] = {"n": len(dn), "delta_nll_mean": float(dn.mean()),
                  "delta_nll_median": float(np.median(dn)),
                  "delta_nll_p90": float(np.percentile(dn, 90)),
                  "kl_mean": float(kl.mean()), "top1_mean": float(t1.mean()),
                  **graph_stats.get(v.split("@")[0], {})}

    rng = np.random.default_rng(20261024)
    have = {k for r in rows for k in r["variants"]}

    def paired(v, ref):
        if v == ref or ref not in have:
            return None
        d = np.array([r["variants"][v]["delta_nll"] - r["variants"][ref]["delta_nll"]
                      for r in rows])
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        boots = d[idx].mean(axis=1)
        return {"mean_diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))]}

    for v in variants:
        agg[v]["vs_serial"] = paired(v, "serial@materialized")
        if v.endswith("@materialized") or "@" not in v:
            agg[v]["vs_g2"] = paired(v, "g2@materialized")

    payload = {"n_samples": len(rows), "variants": agg,
               "note": ("Window scan: all five windows share depth 24 and 4 "
                        "missed updates; only the position varies.")}
    (args.output / "screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "MEASURED"
    manifest["n_measured"] = len(rows)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    def line(v):
        a = agg[v]
        c = a.get("vs_serial")
        cs = "" if not c else f" vs_serial[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
        return (f"  {v:22s} dNLL={a['delta_nll_mean']:7.4f} KL={a['kl_mean']:7.4f} "
                f"top1={a['top1_mean']:.4f} depth={a.get('critical_path')} "
                f"miss={a.get('missed_updates_total')} "
                f"blocks={a.get('executed_blocks')}{cs}")

    print(f"\n== full-model / window position on {args.input} ==")
    for v in variants:
        print(line(v))
    print(f"saved -> {args.output / 'screen.json'}")


if __name__ == "__main__":
    main()
