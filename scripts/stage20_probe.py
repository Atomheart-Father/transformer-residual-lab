"""Stage 2.0 experiment B: which upstream updates do blocks 10-17 need?

Local probes only - they screen candidate edges, they never select a final
graph. Everything here runs with the window entry taken from the teacher:
blocks outside the window are replaced by their exact teacher updates
(``delta_override``), so no inherited error enters the window and the
measured effect of dropping an edge is not contaminated by earlier groups.

Two references are reported for every probe:

  vs full    the same construction with all edges present, in the same merge
             mode - cancels the construction rounding, so this isolates the
             dropped edge itself
  vs teacher the absolute error of the whole construction, which includes
             the machinery's own bf16 noise floor

Two merge modes are run side by side, because they answer differently:

  materialized  production-exact; an update already inside the running state
                cannot be declined by a later block, so dropping a *middle*
                edge while keeping later ones is a no-op here
  roadmap       literal h0 + sum_{j in P_i} delta_j; every predecessor set
                is honored exactly, so the same probe measures the real
                sensitivity to that edge

Plus a suffix-amplification run: the window keeps its probed structure while
blocks 18..27 execute for real on the degraded state. The ratio of final to
window-exit error is how much the suffix amplifies a local perturbation.

Final quality judgement is NOT here; it is the free-running stage
(experiment C) on s20_cal -> s20_val -> blind once.
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
    embed,
    git_sha,
    state_metrics,
    teacher_pass,
)
from dag_exec import Dag, dag_forward  # noqa: E402

LO, HI = 10, 18
MODES = ("materialized", "roadmap")


def window_dag(name: str, n: int, preds_window: dict[int, set[int]]) -> Dag:
    preds = tuple(frozenset(range(i)) if not (LO <= i < HI) else frozenset(preds_window[i])
                  for i in range(n))
    return Dag(name, n, preds, note=f"window [{LO},{HI}) probe")


def full_window(n: int) -> dict[int, set[int]]:
    return {i: set(range(i)) for i in range(LO, HI)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data257/s20_cal.jsonl"))
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_0/probe"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)

    # ---- probe families -------------------------------------------------
    probes: dict[str, dict[int, set[int]]] = {}

    def add(name, f):
        probes[name] = f()

    add("full", lambda: full_window(n))
    for i in range(LO, HI):
        for d in (1, 2, 3, 4, 6):
            j = i - d
            if j < 0:
                continue
            def mk(i=i, j=j):
                p = full_window(n)
                p[i] = p[i] - {j}
                return p
            add(f"drop1_b{i}_m{j}", mk)
    for i in range(LO, HI):
        for k in (1, 2, 4):
            def mk(i=i, k=k):
                p = full_window(n)
                p[i] = set(range(max(0, i - k), i))
                return p
            add(f"recent{i}_k{k}", mk)
    for i in range(LO, HI):
        def mk(i=i):
            p = full_window(n)
            p[i] = p[i] - {i - 1, i - 2}
            return p
        add(f"drop2_b{i}", mk)
    add("skipA_12to17", lambda: {i: (set(range(i)) - {i - 1} if i >= 12
                                     else set(range(i))) for i in range(LO, HI)})
    add("skipB_all", lambda: {i: set(range(i)) - {i - 1} for i in range(LO, HI)})
    add("skipC_alt", lambda: {i: (set(range(i)) - {i - 1} if (i - LO) % 2 == 0
                                  else set(range(i))) for i in range(LO, HI)})
    add("neighbor_11_13", lambda: {i: (set(range(i)) - {i - 1} if i in (11, 13)
                                       else set(range(i))) for i in range(LO, HI)})

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "stage": "2.0 experiment B (local dependency probes)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n,
        "window": [LO, HI],
        "merge_modes": list(MODES),
        "input": str(args.input),
        "max_samples": args.max_samples,
        "n_probes": len(probes),
        "probe_names": sorted(probes),
        "graphs": {k: window_dag(k, n, v).to_dict() for k, v in probes.items()},
        "git_commit": git_sha(),
        "note": ("Local triage only. Every number here screens candidate "
                 "edges; final judgement is the free-running stage."),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows = []
    n_done = 0
    with (args.output / "probes.jsonl").open("w", encoding="utf-8") as sf:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if n_done >= args.max_samples:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
            h0 = embed(model, ids, mx)
            mask = create_attention_mask(h0, None)
            states, _ = teacher_pass(model, ids, mask, mx)
            td = {j: states[j + 1] - states[j] for j in range(n)}
            outside_ov = {j: td[j] for j in range(n) if not (LO <= j < HI)}

            def run_local(preds_window, mode):
                d = window_dag("probe", n, preds_window)
                h, recs = dag_forward(model, h0, d, mask, mx,
                                      delta_override=outside_ov, record=True,
                                      merge_mode=mode)
                mx.eval(h)
                acc = h0.astype(mx.float32)
                for j in range(HI):
                    acc = acc + recs["delta"][j].astype(mx.float32)
                exit_state = acc.astype(h0.dtype)
                mx.eval(exit_state)
                return d, recs, exit_state

            refs = {}
            for mode in MODES:
                _, recs_f, exit_f = run_local(full_window(n), mode)
                refs[mode] = (recs_f, exit_f)

            row = {"sample_id": sample["sample_id"], "probes": {}}
            for name, pw in probes.items():
                if name == "full":
                    continue
                entry = {}
                for mode in MODES:
                    d, recs, exit_state = run_local(pw, mode)
                    recs_full, exit_full = refs[mode]
                    vs_full = state_metrics(exit_state, exit_full, mx)
                    vs_teacher = state_metrics(exit_state, states[HI], mx)
                    changed = [i for i in range(LO, HI) if pw[i] != set(range(i))]
                    per_block = {}
                    for i in changed:
                        m = state_metrics(recs["delta"][i], recs_full["delta"][i], mx)
                        per_block[f"b{i}"] = {"rel_rms": m["rel_rms_diff"],
                                              "cosine": m["cosine"]}
                    semi_ov = {j: td[j] for j in range(LO)}
                    h_semi = dag_forward(model, h0, d, mask, mx,
                                         delta_override=semi_ov, merge_mode=mode)
                    mx.eval(h_semi)
                    semi = state_metrics(h_semi, states[-1], mx)
                    entry[mode] = {
                        "exit_vs_full": {"rel_rms": vs_full["rel_rms_diff"],
                                         "cosine": vs_full["cosine"]},
                        "exit_vs_teacher": {"rel_rms": vs_teacher["rel_rms_diff"],
                                            "cosine": vs_teacher["cosine"]},
                        "per_block_delta": per_block,
                        "changed_blocks": changed,
                        "critical_path": d.critical_path,
                        "semi_free_final_vs_teacher": {
                            "rel_rms": semi["rel_rms_diff"], "cosine": semi["cosine"]},
                    }
                row["probes"][name] = entry
            row["full_exit_vs_teacher"] = {}
            for mode in MODES:
                _, exit_f = refs[mode]
                m = state_metrics(exit_f, states[HI], mx)
                row["full_exit_vs_teacher"][mode] = {
                    "rel_rms": m["rel_rms_diff"], "cosine": m["cosine"]}
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(row)
            n_done += 1
            print(f"[{n_done}/{args.max_samples}] {sample['sample_id']} "
                  + " ".join(f"{m}_noise={row['full_exit_vs_teacher'][m]['rel_rms']:.4f}"
                             for m in MODES), flush=True)

    # ---- aggregation ----------------------------------------------------
    def agg(mode, name, path):
        vals = []
        for r in rows:
            node = r["probes"].get(name)
            if not node or mode not in node:
                continue
            node = node[mode]
            for k in path:
                node = node[k]
            vals.append(node)
        if not vals:
            return None
        return float(np.median(vals)), float(np.max(vals))

    edge_table = {m: {} for m in MODES}
    for i in range(LO, HI):
        for d in (1, 2, 3, 4, 6):
            j = i - d
            if j < 0:
                continue
            name = f"drop1_b{i}_m{j}"
            for m in MODES:
                e = agg(m, name, ["exit_vs_full", "rel_rms"])
                t = agg(m, name, ["exit_vs_teacher", "rel_rms"])
                s = agg(m, name, ["semi_free_final_vs_teacher", "rel_rms"])
                if e is None:
                    continue
                edge_table[m][name] = {
                    "exit_rel_vs_full_median": e[0], "exit_rel_vs_full_max": e[1],
                    "exit_rel_vs_teacher_median": t[0],
                    "semi_free_final_rel_median": s[0],
                    "suffix_amplification": (s[0] / e[0] if e[0] > 1e-9 else None),
                }
    combos = {m: {} for m in MODES}
    for name in ("skipA_12to17", "skipB_all", "skipC_alt", "neighbor_11_13"):
        for m in MODES:
            e = agg(m, name, ["exit_vs_full", "rel_rms"])
            t = agg(m, name, ["exit_vs_teacher", "rel_rms"])
            s = agg(m, name, ["semi_free_final_vs_teacher", "rel_rms"])
            if e is None:
                continue
            combos[m][name] = {
                "exit_rel_vs_full_median": e[0], "exit_rel_vs_full_max": e[1],
                "exit_rel_vs_teacher_median": t[0],
                "semi_free_final_rel_median": s[0],
            }
    noise = {m: float(np.median([r["full_exit_vs_teacher"][m]["rel_rms"]
                                 for r in rows])) for m in MODES}

    summary = {
        "n_samples": len(rows),
        "full_construction_noise_rel_median": noise,
        "single_edge_drops": {m: dict(sorted(v.items(),
                                             key=lambda kv: -kv[1]["exit_rel_vs_full_median"]))
                              for m, v in edge_table.items()},
        "combinations": combos,
        "reading": ("exit_rel_vs_full isolates the dropped edge; the "
                    "semi_free number adds the real suffix (blocks 18-27) on "
                    "top of the probed window and is still not a free run. "
                    "materialized mode cannot decline an update already "
                    "inside the running state; roadmap mode honors every "
                    "predecessor set literally."),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "MEASURED"
    manifest["n_measured"] = len(rows)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n== full construction noise (machinery floor) ==")
    for m in MODES:
        print(f"  {m:14s} rel_rms vs teacher: median={noise[m]:.4f}")
    for m in MODES:
        print(f"\n== single-edge drops [{m}], ranked by effect (median exit rel_rms vs full) ==")
        for k, v in list(summary["single_edge_drops"][m].items())[:24]:
            amp = v["suffix_amplification"]
            print(f"  {k:18s} exit={v['exit_rel_vs_full_median']:.4f} "
                  f"vs_teacher={v['exit_rel_vs_teacher_median']:.4f} "
                  f"semi_free={v['semi_free_final_rel_median']:.4f} "
                  f"amp={'n/a' if amp is None else f'{amp:.2f}'}")
        print(f"  == combinations [{m}] ==")
        for k, v in combos[m].items():
            print(f"  {k:18s} exit={v['exit_rel_vs_full_median']:.4f} "
                  f"vs_teacher={v['exit_rel_vs_teacher_median']:.4f} "
                  f"semi_free={v['semi_free_final_rel_median']:.4f}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
