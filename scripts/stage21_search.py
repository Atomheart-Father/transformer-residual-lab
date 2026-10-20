r"""Stage 2.1 Part B: re-check the real search space of the frozen DAG.

Stage 2.0's candidates came from *predefined* families (contiguous groups of
equal size, a few hand-picked irregular controls). That cannot show the
predefined families are optimal inside the real design space, and the review
recorded exactly this objection. This script searches the space instead of
assuming its shape.

Formulation.  All 28 original blocks always execute with frozen weights; only
the predecessor sets change, so the total work is constant by construction and
any comparison is about structure, not about pruning.  A graph is fully
described by the *dropped edges* relative to the serial chain:

    P_i = {0..i-1} \ {j}    costs exactly 1 missed update, for any j < i.

So every dropped edge costs one missed update regardless of distance, while the
*quality* cost of that miss may depend on the distance and on the position
(the Stage 2.1 probe table now covers 78 of the 108 upstream edges at every
distance 1..12).  Depth only drops when the dropped edge is the one carrying
the critical path.  The space is therefore a clean "which edges to drop"
trade-off, and "contiguous even pairing" is one point in it, not a theorem.

Modes
  single     every single dropped edge inside the window (all positions x all
             distances): the local sensitivity map of the real design space.
  greedy     depth-constrained greedy: each generation must reduce the window
             depth by at least 1, so the trajectory is directly comparable to
             the regular families at matched depth (g2 = depth 4, 4 misses).
  random     random walks of the same length, as the control distribution for
             "irregular structures at matched miss count".
  weights    training-free merge semantics: scale predecessor updates
             (gamma<1 favors recent, >1 favors remote) - a separate axis,
             never mixed with the structure search.
  finalists  validation split only (selection freezes here).
  blind      the fresh Stage 2.1 blind split, run once, guarded by a marker.

Split discipline: cal fits, val selects, final tests. Nothing that has been
looked at is reused as the final blind test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

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
from dag_exec import Dag, dag_forward  # noqa: E402

LO, HI = 10, 18
SPLIT_FILES = {
    "cal": "data257/s21_cal.jsonl",
    "val": "data257/s21_val.jsonl",
    "final": "data257/s21_final.jsonl",
}


# ---------------------------------------------------------------------------
# graph construction from dropped edges
# ---------------------------------------------------------------------------


def preds_from_drops(drops: list[tuple[int, int]]) -> dict[int, set[int]]:
    """Window predecessor sets implied by ``drops`` = list of (i, j) pairs.

    Block ``i`` no longer waits for update ``j``. Drops outside the window are
    rejected; dropping an edge that is not present raises (the caller must use
    the current preds, not the original ones).
    """
    p = {i: set(range(i)) for i in range(LO, HI)}
    for i, j in drops:
        if not LO <= i < HI:
            raise ValueError(f"block {i} outside window [{LO},{HI})")
        if j not in p[i]:
            raise ValueError(f"edge ({i},{j}) is not a current predecessor")
        p[i].discard(j)
    return p


def dag_from_drops(name: str, drops: list[tuple[int, int]], n: int = 28,
                   note: str = "") -> Dag:
    pw = preds_from_drops(drops)
    preds = tuple(frozenset(range(i)) if not (LO <= i < HI) else frozenset(pw[i])
                  for i in range(n))
    return Dag(name, n, preds, note=note or f"window drops {sorted(drops)}")


def dag_from_window(name: str, pw: dict[int, set[int]], n: int = 28,
                    note: str = "") -> Dag:
    preds = tuple(frozenset(range(i)) if not (LO <= i < HI) else frozenset(pw[i])
                  for i in range(n))
    return Dag(name, n, preds, note=note)


def spec_of(dag: Dag) -> str:
    """Full-spec string reproducing this dag exactly (make_dag round-trip)."""
    body = "|".join(",".join(str(j) for j in sorted(dag.preds[i])) for i in range(dag.n))
    return f"{dag.name}:{body}"


def stats_of(dag: Dag) -> dict:
    lvl = dag.levels()
    wlvl = [lvl[i] for i in range(LO, HI)]
    sizes: list[int] = []
    for L in range(min(wlvl), max(wlvl) + 1):
        cnt = sum(1 for i in range(LO, HI) if lvl[i] == L)
        if cnt:
            sizes.append(cnt)
    misses = sum(i - len(dag.preds[i]) for i in range(LO, HI))
    merge_pts = sum(1 for i in range(LO, HI) if dag.preds[i] != frozenset(range(i)))
    return {
        "critical_path": dag.critical_path,
        "window_depth": max(wlvl) - min(wlvl) + 1,   # == window_levels
        "window_levels": max(wlvl) - min(wlvl) + 1,
        "window_level_sizes": sizes,
        "missed_updates_in_window": misses,
        "merge_points": merge_pts,
        "n_edges": dag.n_edges,
    }


def window_levels(pw: dict[int, set[int]]) -> int:
    """Schedulable steps inside the window: distinct dependency levels used.

    This - not the global critical path - is the deployable window depth: the
    serial tail after the window is unaffected by window drops, so a drop that
    only lowers the *first* window level shortens the global path without saving
    a single scheduling step inside the window.
    """
    d = dag_from_window("tmp", pw)
    lvl = d.levels()
    w = [lvl[i] for i in range(LO, HI)]
    return max(w) - min(w) + 1


def move_depth_delta(pw: dict[int, set[int]], i: int, j: int) -> int:
    """Reduction in window schedulable steps caused by dropping edge (i, j)."""
    before = window_levels(pw)
    pw2 = {k: (v - {j} if k == i else set(v)) for k, v in pw.items()}
    return before - window_levels(pw2)


def all_moves(pw: dict[int, set[int]]) -> list[tuple[int, int]]:
    out = []
    for i in range(LO, HI):
        for j in sorted(pw[i]):
            out.append((i, j))
    return out


# ---------------------------------------------------------------------------
# probe table (Stage 2.0 family + Stage 2.1 gap) as a predictor
# ---------------------------------------------------------------------------


def load_probe_table(paths: list[Path]) -> dict[tuple[int, int], dict]:
    table: dict[tuple[int, int], dict] = {}
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d["single_edge_drops"]["materialized"].items():
            m = re.fullmatch(r"drop1_b(\d+)_m(\d+)", k)
            b, mj = int(m.group(1)), int(m.group(2))
            table[(b, mj)] = {
                "exit_rel": v["exit_rel_vs_full_median"],
                "semi_free": v["semi_free_final_rel_median"],
            }
    return table


def probe_prediction(drops: list[tuple[int, int]],
                     table: dict[tuple[int, int], dict]) -> dict:
    ex = [table[(i, j)]["exit_rel"] for (i, j) in drops if (i, j) in table]
    sf = [table[(i, j)]["semi_free"] for (i, j) in drops if (i, j) in table]
    return {
        "n_drops_in_table": len(ex),
        "probe_exit_rel_sum": float(sum(ex)),
        "probe_semi_free_sum": float(sum(sf)),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(dags: dict[str, Dag], samples: list[dict], out: Path,
             role: str, probe_table: dict, merge_mode: str = "materialized",
             weights_by_name: Optional[dict[str, dict]] = None) -> dict:
    """Free-running measurement of every candidate on ``samples``.

    One teacher pass per sample, then one DAG forward per candidate. No cache:
    this is the quality axis, kept separate from the timing axis.
    """
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    n = len(model.model.layers)

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "stage": f"2.1 Part B ({role})",
        "role": role,
        "model": PINNED_MODEL_REPO, "model_revision": PINNED_MODEL_REVISION,
        "n_layers": n, "window": [LO, HI], "merge_mode": merge_mode,
        "graphs": {k: {**d.to_dict(), **stats_of(d), "spec": spec_of(d)}
                   for k, d in dags.items()},
        "graph_stats": {k: stats_of(d) for k, d in dags.items()},
        "n_samples": len(samples),
        "git_commit": git_sha(),
        "note": ("cal fits, val selects, final tests exactly once; every "
                 "candidate executes all 28 frozen blocks."),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    rows = []
    done = 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for sample in samples:
            ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
            ref = model(ids)
            mx.eval(ref)
            h0 = embed(model, ids, mx)
            mask = create_attention_mask(h0, None)
            row = {"sample_id": sample["sample_id"], "variants": {}}
            for name, d in dags.items():
                w = (weights_by_name or {}).get(name)
                h = dag_forward(model, h0, d, mask, mx, merge_mode=merge_mode,
                                weights=w if merge_mode == "roadmap" else None)
                mx.eval(h)
                lg = project_logits(model, h, mx)
                mx.eval(lg)
                row["variants"][name] = logits_metrics(ref, lg, ids, mx)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(row)
            done += 1
            base = row["variants"].get("serial", {})
            print(f"[{done}/{len(samples)}] {sample['sample_id']} "
                  f"serial={base.get('delta_nll', float('nan')):.4f} "
                  + " ".join(f"{k}={v['delta_nll']:.4f}"
                             for k, v in row["variants"].items()
                             if k != "serial"), flush=True)

    variants = sorted({k for r in rows for k in r["variants"]})
    agg: dict[str, dict] = {}
    for v in variants:
        dn = np.array([r["variants"][v]["delta_nll"] for r in rows])
        kl = np.array([r["variants"][v]["teacher_to_candidate_kl"] for r in rows])
        t1 = np.array([r["variants"][v]["top1_agreement"] for r in rows])
        agg[v] = {"n": len(dn), "delta_nll_mean": float(dn.mean()),
                  "delta_nll_median": float(np.median(dn)),
                  "delta_nll_p90": float(np.percentile(dn, 90)),
                  "kl_mean": float(kl.mean()), "top1_mean": float(t1.mean()),
                  **manifest["graph_stats"].get(v, {})}
    rng = np.random.default_rng(20261027)

    def paired(v, ref_name):
        if v == ref_name or ref_name not in agg:
            return None
        d = np.array([r["variants"][v]["delta_nll"] - r["variants"][ref_name]["delta_nll"]
                      for r in rows])
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        boots = d[idx].mean(axis=1)
        return {"mean_diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))]}

    for v in variants:
        agg[v]["vs_serial"] = paired(v, "serial")
    if "g2" in agg:
        for v in variants:
            agg[v]["vs_g2"] = paired(v, "g2")

    payload = {"n_samples": len(rows), "role": role, "merge_mode": merge_mode,
               "variants": agg,
               "note": ("dropped edges relative to the serial chain; each drop "
                        "costs exactly one missed update, at any distance.")}
    (out / "search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "MEASURED"
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# candidate builders
# ---------------------------------------------------------------------------


def build_single(probe_table) -> dict[str, Dag]:
    """Every single dropped edge in the window: position x distance map."""
    cands = {"serial": dag_from_drops("serial", [])}
    for i in range(LO + 1, HI):
        for j in sorted(preds_from_drops([])[i]):
            dist = i - j
            cands[f"d{i}_{j}"] = dag_from_drops(f"d{i}_{j}", [(i, j)],
                                                note=f"drop edge ({i},{j}), d={dist}")
    return cands


def build_greedy_starts() -> dict[str, Dag]:
    """The Stage 2.0 regular families, as depth-matched reference points."""
    out = {"serial": dag_from_drops("serial", [])}
    pair = [(11, 10), (13, 12), (15, 14), (17, 16)]      # contiguous pairs
    out["g2"] = dag_from_drops("g2", pair, note="contiguous pairs = g2")
    out["pair-first"] = dag_from_drops("pair-first", [(11, 10)])
    out["pair-last"] = dag_from_drops("pair-last", [(17, 16)])
    # g4: levels {10,11,12,13} and {14,15,16,17}; every block of a level reads
    # the anchor of the level, so within a level block b drops the edges to all
    # earlier blocks of the same level.
    g4 = []
    for base in (10, 14):
        for k, b in enumerate(range(base + 1, base + 4)):
            g4 += [(b, base + t) for t in range(k + 1)]
    out["g4"] = dag_from_drops("g4", g4, note="levels of 4 = g4")
    # mix3: levels {10,11,12}, {13,14,15}, {16,17}
    mix3 = [(11, 10), (12, 10), (12, 11),
            (14, 13), (15, 13), (15, 14),
            (17, 16)]
    out["mix3"] = dag_from_drops("mix3", mix3, note="level sizes 3,3,2")
    return out


# ---------------------------------------------------------------------------
# measurement core (model loaded once, reused across candidates/generations)
# ---------------------------------------------------------------------------


def load_split(split: str, max_samples: int) -> list[dict]:
    path = Path(SPLIT_FILES[split])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
        if len(out) >= max_samples:
            break
    return out


def precompute_refs(model, mx, samples):
    """Teacher logits + embedding + mask per sample, computed once."""
    from mlx_lm.models.base import create_attention_mask
    refs = []
    for s in samples:
        ids = mx.array(s["ids"], dtype=mx.int32)[None, :]
        ref = model(ids)
        mx.eval(ref)
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        mx.eval(h0)
        refs.append({"sample_id": s["sample_id"], "ids": ids, "ref": ref,
                     "h0": h0, "mask": mask})
    return refs


def measure_batch(model, mx, refs, dags: dict[str, Dag],
                  merge_mode: str = "materialized",
                  weights_by_name: Optional[dict] = None) -> dict:
    """Free-running quality of every candidate on every reference sample."""
    rows = []
    for r in refs:
        row = {"sample_id": r["sample_id"], "variants": {}}
        for name, d in dags.items():
            w = (weights_by_name or {}).get(name)
            h = dag_forward(model, r["h0"], d, r["mask"], mx,
                            merge_mode=merge_mode,
                            weights=w if merge_mode == "roadmap" else None)
            mx.eval(h)
            lg = project_logits(model, h, mx)
            mx.eval(lg)
            row["variants"][name] = logits_metrics(r["ref"], lg, r["ids"], mx)
        rows.append(row)
    return rows


def aggregate(rows: list[dict], graph_stats: dict, ref_name: str = "serial") -> dict:
    variants = sorted({k for r in rows for k in r["variants"]})
    agg: dict[str, dict] = {}
    for v in variants:
        dn = np.array([r["variants"][v]["delta_nll"] for r in rows])
        kl = np.array([r["variants"][v]["teacher_to_candidate_kl"] for r in rows])
        t1 = np.array([r["variants"][v]["top1_agreement"] for r in rows])
        agg[v] = {"n": len(dn), "delta_nll_mean": float(dn.mean()),
                  "delta_nll_median": float(np.median(dn)),
                  "delta_nll_p90": float(np.percentile(dn, 90)),
                  "kl_mean": float(kl.mean()), "top1_mean": float(t1.mean()),
                  **graph_stats.get(v, {})}
    rng = np.random.default_rng(20261027)

    def paired(v, ref_n):
        if v == ref_n or ref_n not in agg:
            return None
        d = np.array([r["variants"][v]["delta_nll"] - r["variants"][ref_n]["delta_nll"]
                      for r in rows])
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        boots = d[idx].mean(axis=1)
        return {"mean_diff": float(d.mean()),
                "ci95": [float(np.percentile(boots, 2.5)),
                         float(np.percentile(boots, 97.5))]}

    for v in variants:
        agg[v]["vs_serial"] = paired(v, ref_name)
        if "g2" in agg and v != "g2":
            agg[v]["vs_g2"] = paired(v, "g2")
    return {"variants": agg}


def spearman(x, y) -> Optional[float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return None

    def rank(a):
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1)
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def write_run(out: Path, role: str, dags: dict[str, Dag], payload: dict,
              merge_mode: str, extra: Optional[dict] = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "MEASURED",
        "stage": f"2.1 Part B ({role})",
        "role": role,
        "model": PINNED_MODEL_REPO, "model_revision": PINNED_MODEL_REVISION,
        "n_layers": dags[next(iter(dags))].n,
        "window": [LO, HI], "merge_mode": merge_mode,
        "graphs": {k: {**d.to_dict(), **stats_of(d), "spec": spec_of(d)}
                   for k, d in dags.items()},
        "graph_stats": {k: stats_of(d) for k, d in dags.items()},
        "git_commit": git_sha(),
        "note": ("cal fits, val selects, final tests exactly once; every "
                 "candidate executes all 28 frozen blocks."),
    }
    if extra:
        manifest.update(extra)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    (out / "search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# mode drivers
# ---------------------------------------------------------------------------


def mode_single(model, mx, refs, out, probe_table):
    dags = build_single(probe_table)
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    payload["merge_mode"] = "materialized"
    # probe predictor correlation over the single-edge candidates
    xs, ys = [], []
    for name, st in stats.items():
        if name == "serial":
            continue
        m = re.fullmatch(r"d(\d+)_(\d+)", name)
        if not m:
            continue
        pred = probe_prediction([(int(m.group(1)), int(m.group(2)))], probe_table)
        if pred["n_drops_in_table"] == 0:
            continue
        xs.append(pred["probe_exit_rel_sum"])
        ys.append(payload["variants"][name]["delta_nll_mean"])
    payload["probe_predictor"] = {
        "n": len(xs),
        "spearman_exit_rel_vs_dnll": spearman(xs, ys),
        "note": ("single-edge candidates only; probe exit_rel (teacher-"
                 "materialized sensitivity) vs measured free-running dNLL."),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (single-edge map)", dags, payload, "materialized")
    print(f"saved -> {out / 'search.json'}")
    return payload


def mode_families(model, mx, refs, out, probe_table):
    dags = build_greedy_starts()
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (regular families)", dags, payload, "materialized")
    print(f"saved -> {out / 'search.json'}")
    return payload


def mode_greedy(model, mx, refs, out, probe_table, generations: int):
    """Depth-constrained greedy edge dropping, from the serial window."""
    pw = preds_from_drops([])
    drops: list[tuple[int, int]] = []
    history = []
    all_rows: list[dict] = []
    all_dags: dict[str, Dag] = {}
    all_stats: dict[str, dict] = {}

    def name_for(ds):
        return "g" + "_".join(f"{i}-{j}" for i, j in ds) if ds else "serial"

    for gen in range(generations + 1):
        cur = dag_from_window(name_for(drops), pw, note=f"greedy gen {gen}")
        all_dags[cur.name] = cur
        all_stats[cur.name] = stats_of(cur)
        gen_dags = {cur.name: cur}
        moves: list[tuple[int, int]] = []
        if gen < generations:
            moves = [(i, j) for (i, j) in all_moves(pw)
                     if move_depth_delta(pw, i, j) >= 1]
            for (i, j) in moves:
                d2 = drops + [(i, j)]
                nd = dag_from_drops(f"m{i}_{j}", d2, note=f"gen {gen} move ({i},{j})")
                gen_dags[nd.name] = nd
                all_dags[nd.name] = nd
                all_stats[nd.name] = stats_of(nd)
        if not gen_dags or (gen < generations and not moves):
            print(f"[gen {gen}] depth={window_levels(pw)}: no depth-reducing "
                  f"move remains; search ends", flush=True)
            history.append({"generation": gen, "drops": list(drops),
                            "structure": all_stats[cur.name],
                            "note": "search ended: no move reduces window depth"})
            break
        rows = measure_batch(model, mx, refs, gen_dags)
        all_rows.extend({"gen": gen, **r} for r in rows)
        agg = aggregate(rows, {k: all_stats[k] for k in gen_dags},
                        ref_name=cur.name)
        best = min((k for k in gen_dags if k != cur.name),
                   key=lambda k: (agg["variants"][k]["delta_nll_mean"],
                                  -all_stats[k]["window_levels"])) \
            if moves else None
        rec = {"generation": gen, "drops": list(drops),
               "structure": all_stats[cur.name],
               "measured": {k: {"delta_nll_mean": agg["variants"][k]["delta_nll_mean"],
                                "ci95_vs_cur": (agg["variants"][k]["vs_serial"] or {}).get("ci95")}
                            for k in gen_dags},
               "current_dnll": agg["variants"][cur.name]["delta_nll_mean"],
               "chosen_move": best}
        history.append(rec)
        print(f"[gen {gen}] depth={all_stats[cur.name]['window_depth']} "
              f"misses={all_stats[cur.name]['missed_updates_in_window']} "
              f"dNLL={agg['variants'][cur.name]['delta_nll_mean']:.4f} "
              f"best_move={best}", flush=True)
        if gen < generations:
            if best is None:
                print(f"[gen {gen}] no depth-reducing move left; stopping early",
                      flush=True)
                break
            i, j = (int(x) for x in best[1:].split("_"))
            drops.append((i, j))
            pw = preds_from_drops(drops)

    payload = {"role": "calibration", "merge_mode": "materialized",
               "history": history,
               "final_drops": drops,
               "note": ("each generation must reduce the window depth by >=1; "
                        "the chosen move minimises free-running dNLL on cal.")}
    # flat aggregate: every evaluated structure vs the serial baseline, with the
    # paired bootstrap over the same samples (each candidate appears in exactly
    # one generation batch, and every batch reuses the same refs).
    names = sorted({k for r in all_rows for k in r["variants"]})
    agg_all: dict[str, dict] = {}
    for nm in names:
        per = [(r["sample_id"], r["variants"][nm]["delta_nll"])
               for r in all_rows if nm in r["variants"]]
        serial_per = {r["sample_id"]: r["variants"]["serial"]["delta_nll"]
                      for r in all_rows if "serial" in r["variants"]}
        dn = np.array([v for _, v in per])
        d_serial = np.array([v - serial_per[sid] for sid, v in per])
        rng = np.random.default_rng(20261027)
        idx = rng.integers(0, len(d_serial), size=(20000, len(d_serial)))
        boots = d_serial[idx].mean(axis=1)
        agg_all[nm] = {"n": len(dn), "delta_nll_mean": float(dn.mean()),
                       **all_stats.get(nm, {}),
                       "vs_serial": {"mean_diff": float(d_serial.mean()),
                                     "ci95": [float(np.percentile(boots, 2.5)),
                                              float(np.percentile(boots, 97.5))]}}
    payload["all_candidates"] = agg_all
    write_run(out, "calibration (greedy depth-constrained)", all_dags, payload,
              "materialized",
              extra={"greedy_history": history, "final_drops": drops,
                     "final_spec": spec_of(dag_from_drops("greedy-final", drops))})
    with (out / "generations.jsonl").open("w", encoding="utf-8") as sf:
        for r in all_rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved -> {out / 'search.json'}")
    return payload


def mode_random(model, mx, refs, out, probe_table, n_random: int, length: int,
                seed: int):
    rng = np.random.default_rng(seed)
    dags: dict[str, Dag] = {"serial": dag_from_drops("serial", [])}
    g2 = build_greedy_starts()["g2"]
    dags["g2"] = g2
    for k in range(n_random):
        pw = preds_from_drops([])
        drops = []
        for _ in range(length):
            moves = all_moves(pw)
            i, j = moves[int(rng.integers(0, len(moves)))]
            drops.append((i, j))
            pw = preds_from_drops(drops)
        d = dag_from_drops(f"r{k:02d}", drops, note=f"random walk seed {seed} #{k}")
        dags[d.name] = d
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    payload["merge_mode"] = "materialized"
    # matched comparison: among structures with the same miss count as g2
    g2_miss = stats["g2"]["missed_updates_in_window"]
    same_miss = {k: v for k, v in stats.items()
                 if v["missed_updates_in_window"] == g2_miss and k != "g2"}
    if same_miss:
        best = min(same_miss, key=lambda k: payload["variants"][k]["delta_nll_mean"])
        payload["matched_miss_comparison"] = {
            "target_misses": g2_miss, "n_random_at_target": len(same_miss),
            "best_random": best,
            "best_random_dnll": payload["variants"][best]["delta_nll_mean"],
            "g2_dnll": payload["variants"]["g2"]["delta_nll_mean"],
            "n_random_better_than_g2": sum(
                1 for k in same_miss
                if payload["variants"][k]["delta_nll_mean"]
                < payload["variants"]["g2"]["delta_nll_mean"]),
        }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (random irregular controls)", dags, payload,
              "materialized", extra={"seed": seed, "n_random": n_random,
                                     "walk_length": length})
    print(f"saved -> {out / 'search.json'}")
    return payload


def enumerate_prefix_truncations(max_levels: int, miss_cap: int) -> list[tuple]:
    """All reachable prefix-truncation graphs inside the miss budget.

    ``P_i = {0..t_i}`` is the *maximal-quality* predecessor set for any fixed
    dependency-level pattern: every additional dropped edge costs quality
    (all 98 single-edge drops measured positive) and cannot buy depth, because
    keeping any later predecessor pins the level. Prefix truncations therefore
    cover every level pattern, including the non-monotone ones that only
    multi-edge drops can create, without enumerating dominated variants.
    Returns sorted drop lists, one per structure.
    """
    from collections import defaultdict
    found: dict[tuple, int] = {}

    def dfs(i, ts, misses):
        if i == 18:
            pw = {b: set(range(ts[b - 10] + 1)) for b in range(LO, HI)}
            if window_levels(pw) <= max_levels:
                key = tuple(sorted((b, j) for b in range(LO, HI)
                                   for j in range(b) if j > ts[b - LO]))
                found[key] = window_levels(pw)
            return
        b = i
        for t in range(b - 1, LO - 2, -1):   # t = b-1 (serial) .. LO-1 (anchor)
            m = misses + (b - 1 - t)
            if m > miss_cap:
                break                        # deeper truncation only adds misses
            ts.append(t)
            dfs(i + 1, ts, m)
            ts.pop()

    dfs(LO, [], 0)
    return sorted(found.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))


def mode_exhaustive(model, mx, refs, out, probe_table, max_levels: int,
                    miss_cap: int):
    """Exhaustive evaluation of the reachable prefix-truncation design space."""
    enum = enumerate_prefix_truncations(max_levels, miss_cap)
    dags = {"serial": dag_from_drops("serial", [])}
    for k, (drops, wl) in enumerate(enum):
        nm = f"e{wl:02d}m{len(drops):02d}_{k:04d}"
        dags[nm] = dag_from_drops(nm, list(drops), note=f"exhaustive wdepth {wl}")
    stats = {k: stats_of(d) for k, d in dags.items()}
    print(f"exhaustive: {len(enum)} structures (levels<={max_levels}, "
          f"misses<={miss_cap}) x {len(refs)} samples", flush=True)
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    payload["merge_mode"] = "materialized"
    # Pareto frontier: per window depth, the best few by dNLL
    from collections import defaultdict
    by = defaultdict(list)
    for nm in stats:
        if nm == "serial":
            continue
        by[stats[nm]["window_levels"]].append(nm)
    pareto = {}
    for wl in sorted(by, reverse=True):
        grp = sorted(by[wl], key=lambda k: payload["variants"][k]["delta_nll_mean"])
        pareto[str(wl)] = [{
            "name": k, "delta_nll_mean": payload["variants"][k]["delta_nll_mean"],
            "missed_updates_in_window": stats[k]["missed_updates_in_window"],
            "window_level_sizes": stats[k]["window_level_sizes"],
            "merge_points": stats[k]["merge_points"],
            "drops": [[i, j] for (i, j) in
                      sorted({(b, j) for b in range(LO, HI)
                              for j in range(b) if j not in set(dags[k].preds[b])})],
        } for k in grp[:5]]
    payload["pareto_by_depth"] = pareto
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (exhaustive reachable space)", dags, payload,
              "materialized",
              extra={"max_levels": max_levels, "miss_cap": miss_cap,
                     "n_structures": len(enum)})
    print(f"saved -> {out / 'search.json'}")
    return payload


def mode_random_depth(model, mx, refs, out, probe_table, per_length: int,
                      seed: int):
    """Depth-matched irregular controls.

    Random walks where *every* drop must reduce the window schedulable depth by
    one, so each structure sits at a window depth a regular family also reaches
    (length 2 -> depth 6, length 4 -> depth 4, length 6 -> depth 2). This is the
    control that answers "at matched depth, is the regular family optimal among
    irregular graphs?" - the earlier random mode could not, because its 4-drop
    walks mostly stayed at depth 8 (remote edges are nearly free but save no
    scheduling step).
    """
    rng = np.random.default_rng(seed)
    dags: dict[str, Dag] = {"serial": dag_from_drops("serial", [])}
    fams = build_greedy_starts()
    dags["g2"] = fams["g2"]
    dags["g4"] = fams["g4"]
    trials = []
    for length in (2, 4, 6):
        for k in range(per_length):
            pw = preds_from_drops([])
            drops: list[tuple[int, int]] = []
            for _ in range(length):
                moves = [(i, j) for (i, j) in all_moves(pw)
                         if move_depth_delta(pw, i, j) >= 1]
                if not moves:
                    break
                i, j = moves[int(rng.integers(0, len(moves)))]
                drops.append((i, j))
                pw = preds_from_drops(drops)
            nm = f"rd{length}_{k:02d}"
            d = dag_from_drops(nm, drops, note=f"depth-matched random len {length} #{k}")
            dags[nm] = d
            trials.append(nm)
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    payload["merge_mode"] = "materialized"
    matched = {}
    for length in (2, 4, 6):
        grp = [k for k in trials if k.startswith(f"rd{length}_")]
        if not grp:
            continue
        vals = sorted(((payload["variants"][k]["delta_nll_mean"], k) for k in grp))
        matched[f"depth_{8 - length}"] = {
            "n": len(grp),
            "misses": sorted({stats[k]["missed_updates_in_window"] for k in grp}),
            "best_random": {"name": vals[0][1], "dnll": vals[0][0]},
            "median_random": float(np.median([v for v, _ in vals])),
            "worst_random": {"name": vals[-1][1], "dnll": vals[-1][0]},
        }
    matched["regular_at_depth_4"] = {
        "name": "g2", "dnll": payload["variants"]["g2"]["delta_nll_mean"],
        "misses": stats["g2"]["missed_updates_in_window"]}
    matched["regular_at_depth_2"] = {
        "name": "g4", "dnll": payload["variants"]["g4"]["delta_nll_mean"],
        "misses": stats["g4"]["missed_updates_in_window"]}
    payload["depth_matched_comparison"] = matched
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (depth-matched irregular controls)", dags,
              payload, "materialized",
              extra={"seed": seed, "per_length": per_length})
    print(f"saved -> {out / 'search.json'}")
    return payload


def mode_weights(model, mx, refs, out, probe_table, gammas, structures):
    """Training-free merge semantics, roadmap mode, separate from structure."""
    dags: dict[str, Dag] = {}
    weights_by_name: dict[str, dict] = {}
    for sname, drops in structures.items():
        d = dag_from_drops(sname, drops)
        dags[sname] = d
        for g in gammas:
            nm = f"{sname}-g{g}"
            dags[nm] = d
            w = {}
            for i in range(LO, HI):
                for j in sorted(d.preds[i]):
                    w[j] = float(g ** (i - j))
            weights_by_name[nm] = w
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags, merge_mode="roadmap",
                         weights_by_name=weights_by_name)
    payload = aggregate(rows, stats)
    payload["role"] = "calibration"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, "calibration (merge semantics)", dags, payload, "roadmap",
              extra={"gammas": gammas, "structures": structures,
                     "weights": weights_by_name})
    print(f"saved -> {out / 'search.json'}")
    return payload


def parse_specs(text: str, n: int) -> dict[str, Dag]:
    """``name:spec,name:spec`` -> Dag objects (exact reproduction)."""
    from dag_exec import _expand
    out: dict[str, Dag] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, body = chunk.split(":", 1)
        parts = body.split("|")
        if len(parts) != n:
            raise SystemExit(f"{name}: spec needs {n} parts, got {len(parts)}")
        out[name] = Dag(name, n, tuple(frozenset(_expand(p)) for p in parts),
                        note="replayed from spec")
    return out


def mode_finalists(model, mx, refs, out, probe_table, specs, split_role):
    dags = parse_specs(specs, 28)
    if "serial" not in dags:
        dags = {"serial": dag_from_drops("serial", []), **dags}
    stats = {k: stats_of(d) for k, d in dags.items()}
    rows = measure_batch(model, mx, refs, dags)
    payload = aggregate(rows, stats)
    payload["role"] = split_role
    ranked = sorted((k for k in stats if k != "serial"),
                    key=lambda k: payload["variants"][k]["delta_nll_mean"])
    payload["ranking"] = ranked
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for r in rows:
            sf.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_run(out, f"{split_role} (finalists)", dags, payload, "materialized")
    print(f"saved -> {out / 'search.json'}; ranking = {ranked}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["single", "families", "greedy", "random",
                             "random-depth", "exhaustive", "weights",
                             "finalists", "blind"])
    ap.add_argument("--max-levels", type=int, default=5)
    ap.add_argument("--miss-cap", type=int, default=6)
    ap.add_argument("--split", default="cal", choices=["cal", "val", "final"])
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=24)
    ap.add_argument("--walk-length", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--gammas", default="1.0,0.9,0.8")
    ap.add_argument("--specs", default="",
                    help="name:spec,... full specs for finalists/blind")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.mode in ("single", "families", "greedy", "random", "random-depth",
                     "exhaustive", "weights"):
        if args.split != "cal":
            raise SystemExit(f"{args.mode} is a fitting step: use --split cal")
    if args.mode == "finalists" and args.split != "val":
        raise SystemExit("finalists is selection: use --split val")
    if args.mode == "blind":
        if args.split != "final":
            raise SystemExit("blind must run on --split final")
        marker = args.output / "ONESHOT"
        if marker.exists():
            raise SystemExit(f"{marker} exists: the final blind split may be "
                             "evaluated exactly once. Refusing to run again.")

    import mlx.core as mx
    from mlx_lm import load

    samples = load_split(args.split, args.max_samples)
    model, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    refs = precompute_refs(model, mx, samples)
    probe_table = load_probe_table([
        Path("results/stage2_0/probe/summary.json"),
        Path("results/stage2_1/probe_gap/summary.json"),
    ])

    if args.mode == "single":
        mode_single(model, mx, refs, args.output, probe_table)
    elif args.mode == "families":
        mode_families(model, mx, refs, args.output, probe_table)
    elif args.mode == "greedy":
        mode_greedy(model, mx, refs, args.output, probe_table, args.generations)
    elif args.mode == "random":
        mode_random(model, mx, refs, args.output, probe_table, args.n_random,
                    args.walk_length, args.seed)
    elif args.mode == "random-depth":
        mode_random_depth(model, mx, refs, args.output, probe_table,
                          args.n_random, args.seed)
    elif args.mode == "exhaustive":
        mode_exhaustive(model, mx, refs, args.output, probe_table,
                        args.max_levels, args.miss_cap)
    elif args.mode == "weights":
        gammas = [float(g) for g in args.gammas.split(",") if g.strip()]
        structures = {
            "g2": [(11, 10), (13, 12), (15, 14), (17, 16)],
            "g4": [(11, 10), (12, 10), (12, 11), (13, 10), (13, 11), (13, 12),
                   (15, 14), (16, 14), (16, 15), (17, 14), (17, 15), (17, 16)],
        }
        mode_weights(model, mx, refs, args.output, probe_table, gammas, structures)
    elif args.mode in ("finalists", "blind"):
        if not args.specs:
            raise SystemExit(f"{args.mode} needs --specs")
        role = "validation" if args.mode == "finalists" else "final blind test"
        mode_finalists(model, mx, refs, args.output, probe_table, args.specs, role)
        if args.mode == "blind":
            (args.output / "ONESHOT").write_text(
                f"final blind split evaluated once at {git_sha()}\n",
                encoding="utf-8")


if __name__ == "__main__":
    main()
