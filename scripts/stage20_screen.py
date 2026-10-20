"""Stage 2.0 experiment C part 1: free-running screening on calibration data.

The probes (experiment B) showed that *every* upstream update a window block
waits for matters, at a roughly uniform cost (~0.03 rel_rms at the window
exit, amplified ~2.5x by the real suffix). That makes the design space
collapsible: for a block at dependency level L, the largest legal predecessor
set is "every earlier block at a lower level", so the only freedom is the
multiset of level sizes, and contiguous even pairing is miss-minimal at every
depth. This script tests that prediction against the real free-running
forward, together with deletion controls at matched depth (roadmap B2: a
lossy deletion must not be mistaken for DAG parallelism).

Candidates (window = blocks 10-17, everything else serial):

  serial        window depth 8, 0 missed updates
  p1/p2/p3      one/two/three parallel pairs, then serial  (depth 7/6/5)
  g2            four pairs                                     (depth 4)
  g4            two quads                                      (depth 2)
  mix3          levels of 3,3,2                                 (depth 3)
  hyp-shift     shifted pairing (11,12)(13,14)(15,16)+17       (depth 5)
  noncontig     same level sizes as g2 but non-contiguous      (depth 4)
  sparse-x1     non-contiguous multi-predecessor probe graph   (depth 4)
  del-w{1..4}   delete 1-4 blocks inside the window            (depth 27..24)

Selection is NOT here: this ranks on calibration data only. Candidates that
survive go to s20_val in the next step, and the blind splits run once.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure import (  # noqa: E402
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    Plan,
    embed,
    git_sha,
    logits_metrics,
    make_plan,
    project_logits,
    run_plan,
    teacher_pass,
)
from dag_exec import Dag, dag_forward  # noqa: E402

LO, HI = 10, 18
MODES = ("materialized", "roadmap")


def levels_window(sizes):
    """sizes = blocks per level inside the window, contiguous, prefix-maximal.

    Every block of a level reads the merge of all strictly earlier levels
    (the shared anchor), so a level of ``k`` blocks costs ``k(k-1)/2`` missed
    updates and contributes one dependency level.
    """
    p = {}
    start = LO
    for size in sizes:
        blocks = list(range(start, start + size))
        for b in blocks:
            p[b] = set(range(start))     # same anchor for the whole level
        start += size
    assert start == HI, (sizes, start)
    return p


def candidates() -> dict[str, dict[int, set[int]]]:
    out: dict[str, dict[int, set[int]]] = {}
    out["serial"] = {i: set(range(i)) for i in range(LO, HI)}
    out["p1"] = levels_window([2, 1, 1, 1, 1, 1, 1])  # identical to pair-first
    out["p2"] = levels_window([2, 2, 1, 1, 1, 1])
    out["p3"] = levels_window([2, 2, 2, 1, 1])
    out["g2"] = levels_window([2, 2, 2, 2])
    out["g4"] = levels_window([4, 4])
    out["mix3"] = levels_window([3, 3, 2])
    out["mix4"] = levels_window([3, 2, 2, 1])
    # pair-position controls at matched misses=1, depth=27 (p1 IS pair-first)
    out["pair-mid"] = levels_window([1, 1, 2, 1, 1, 1, 1])
    out["pair-last"] = levels_window([1, 1, 1, 1, 1, 1, 2])
    # shifted pairing: (11,12)(13,14)(15,16)(17) - same miss count as g2,
    # one level deeper
    sh = {i: set(range(i)) for i in range(LO, HI)}
    for a in (11, 13, 15):
        sh[a] = set(range(a)); sh[a + 1] = set(range(a))
    out["hyp-shift"] = sh
    # non-contiguous alignment with the same level sizes as g2: levels are
    # {10,12} {11,13} {14,16} {15,17}. Prefix-maximal predecessors force
    # deeper waits for the out-of-order blocks, so the same depth costs
    # more missed updates than the contiguous pairing.
    nc = {i: set(range(i)) for i in range(LO, HI)}
    nc[10] = set(range(10)); nc[12] = set(range(10))
    nc[11] = set(range(11)); nc[13] = set(range(11)) | {12}
    nc[14] = set(range(14)); nc[16] = set(range(14))
    nc[15] = set(range(15)); nc[17] = set(range(15)) | {16}
    out["noncontig"] = nc
    # the sparse multi-predecessor graph used in the bench
    out["sparse-x1"] = {
        10: set(range(10)), 11: set(range(10)), 12: {10}, 13: {11},
        14: {10, 12}, 15: {11, 13}, 16: {14}, 17: {15},
    }
    return out


def window_dag(name: str, n: int, pw: dict[int, set[int]]) -> Dag:
    preds = tuple(frozenset(range(i)) if not (LO <= i < HI) else frozenset(pw[i])
                  for i in range(n))
    return Dag(name, n, preds, note=f"window [{LO},{HI}) candidate")


def deletion_plan(name: str, n: int, deleted: list[int]) -> Plan:
    keep = [i for i in range(n) if i not in set(deleted)]
    return Plan(name, [[i] for i in keep], "delete", len(keep), len(keep),
                sorted(deleted), note=f"delete {deleted} in window")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s20_cal.jsonl"))
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_0/screen"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)

    cands = candidates()
    dags = {k: window_dag(k, n, v) for k, v in cands.items()}
    dels = {f"del-w{k}": deletion_plan(f"del-w{k}", n, [10 + 2 * t for t in range(k)])
            for k in (1, 2, 3, 4)}

    def stat(d: Dag) -> dict:
        sizes: list[int] = []
        lvl = d.levels()
        for L in range(max(lvl) + 1):
            cnt = sum(1 for i in range(LO, HI) if lvl[i] == L)
            if cnt and L >= 10:
                sizes.append(cnt)
        misses = sum(i - len(d.preds[i]) for i in range(LO, HI))
        return {"critical_path": d.critical_path, "window_level_sizes": sizes,
                "missed_updates_in_window": misses}

    graph_stats = {k: stat(d) for k, d in dags.items()}
    graph_stats.update({k: {"critical_path": p.effective_depth,
                            "window_level_sizes": None,
                            "missed_updates_in_window": n - p.executed_blocks}
                        for k, p in dels.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "stage": "2.0 experiment C part 1 (free-running screening, calibration)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "window": [LO, HI],
        "merge_modes": list(MODES),
        "input": str(args.input),
        "max_samples": args.max_samples,
        "graphs": {**{k: d.to_dict() for k, d in dags.items()},
                   **{k: {"name": k, "kind": "delete",
                          "deleted_layers": p.deleted_layers,
                          "critical_path": p.effective_depth,
                          "executed_blocks": p.executed_blocks}
                      for k, p in dels.items()}},
        "graph_stats": graph_stats,
        "git_commit": git_sha(),
        "note": ("Calibration screening only; selection happens on s20_val and "
                 "the blind splits run exactly once."),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows = []
    n_done = 0
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if n_done >= args.max_samples:
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
                    mx.eval(h)
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
            n_done += 1
            base = row["variants"]["serial@materialized"]["delta_nll"]
            g2v = row["variants"]["g2@materialized"]["delta_nll"]
            print(f"[{n_done}/{args.max_samples}] {sample['sample_id']} "
                  f"serial={base:.4f} g2={g2v:.4f} g4={row['variants']['g4@materialized']['delta_nll']:.4f}",
                  flush=True)

    # ---- aggregation ----------------------------------------------------
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

    # paired bootstrap vs g2 and vs serial (same samples)
    rng = np.random.default_rng(20261020)
    def paired(v, ref):
        if v == ref:
            return None
        d = np.array([r["variants"][v]["delta_nll"] - r["variants"][ref]["delta_nll"]
                      for r in rows])
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        boots = d[idx].mean(axis=1)
        return {"mean_diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))]}

    for v in variants:
        agg[v]["vs_serial_materialized"] = paired(v, "serial@materialized")
        if v.endswith("@materialized") or "@" not in v:
            agg[v]["vs_g2_materialized"] = paired(v, "g2@materialized")

    payload = {
        "n_samples": len(rows),
        "variants": agg,
        "note": ("missed_updates_in_window = sum over window blocks of "
                 "(i - |P_i|): how many upstream updates the graph forgoes. "
                 "Deletion controls execute fewer blocks; DAGs always run "
                 "all 28."),
    }
    (args.output / "screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "MEASURED"
    manifest["n_measured"] = len(rows)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def line(v):
        a = agg[v]
        c = a.get("vs_serial_materialized")
        cg = a.get("vs_g2_materialized")
        cs = "" if not c else f" vs_serial[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
        cgs = "" if not cg else f" vs_g2[{cg['ci95'][0]:+.3f},{cg['ci95'][1]:+.3f}]"
        return (f"  {v:24s} dNLL={a['delta_nll_mean']:7.4f} KL={a['kl_mean']:7.4f} "
                f"top1={a['top1_mean']:.4f} depth={a.get('critical_path')} "
                f"miss={a.get('missed_updates_in_window')}{cs}{cgs}")
    print("\n== free-running on s20_cal (mean over samples) ==")
    for v in variants:
        print(line(v))
    print(f"saved -> {args.output / 'screen.json'}")


if __name__ == "__main__":
    main()
