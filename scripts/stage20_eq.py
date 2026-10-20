"""Stage 2.0 gate 0: DAG executor equivalence, before any search.

Checks on the real frozen model, all on calibration-role data (no test split is
touched, no selection happens here):

G0.1a  full-predecessor graph executed as the serial chain reproduces the
       reference model bit-for-bit: delta_nll == 0, top1 == 1, max_abs == 0.
G0.1b  the same graph through the merge machinery without the serial
       continuation shortcut stays within bf16 double-rounding noise.
G0.2a  g2-mimic DAG == production g2 (``group_forward`` semantics) bit-for-bit.
G0.2b  the in-process production g2 reference == ``measure.run_plan(group2)``
       bit-for-bit (pins the reference implementation itself).
G0.2c  prefix-k g2 DAG delta_nll equals the stored Stage 1.5 numbers for the
       same samples.
G0.2d  window-mid-g2 DAG == ``run_plan(window-mid-g2)`` in the same process.
G0.2e  vs the monolithic fp32 merge (no intermediate materialization):
       informational - quantifies what materializing at merge points costs.

Pass/fail uses the tolerances recorded in the manifest *before* the run; the
run never relaxes them afterwards.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure import (  # noqa: E402
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
    logits_metrics,
    make_plan,
    project_logits,
    run_plan,
    state_metrics,
    teacher_pass,
)
from dag_exec import (  # noqa: E402
    chain_dag,
    dag_forward,
    group_dag,
    g2_reference,
    monolithic_reference,
    prefix_group_dag,
    window_group_dag,
)

PREFIXES = (0, 2, 3, 4, 6, 8, 10)

# a-priori tolerances; written to the manifest before anything is executed
TOL = {
    "g01a_delta_nll": 1e-9,
    "g01a_max_abs": 0.0,
    "g01b_rel_rms": 0.02,
    "g01b_delta_nll": 0.05,
    "g02a_max_abs": 0.0,
    "g02b_max_abs": 0.0,
    "g02c_delta_nll": 0.01,
    "g02d_max_abs": 0.0,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("data/calibration.jsonl"),
                    help="calibration-role data only; never a test split")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("results/stage2_0/equivalence"))
    ap.add_argument("--cross-check", type=Path,
                    default=Path("results/stage1_5/screen/samples.jsonl"),
                    help="stored Stage 1.5 prefix-g2 numbers for the same corpus")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    stored = {}
    if args.cross_check.exists():
        for line in args.cross_check.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stored[row["sample_id"]] = {
                k: v for k, v in row.get("variants", {}).items()
                if v.get("fusion") == "sum" and k.startswith("prefix") and k.endswith("@fp32")
            }

    model, tokenizer, config = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    args.output.mkdir(parents=True, exist_ok=True)

    chain = chain_dag(n)
    g2 = group_dag(n, 2, name="prefix0-g2")
    win = window_group_dag(n, 10, 18, 2, name="window-mid-g2")
    prefix_dags = {k: prefix_group_dag(n, k, 2) for k in PREFIXES}

    manifest = {
        "status": "RUNNING",
        "stage": "2.0 gate 0 (executor equivalence)",
        "model": PINNED_MODEL_REPO,
        "model_revision": PINNED_MODEL_REVISION,
        "model_config": config,
        "n_layers": n,
        "mlx_version": getattr(mx, "__version__", "UNKNOWN"),
        "mlx_lm_version": getattr(mlx_lm, "__version__", "UNKNOWN"),
        "python": sys.version,
        "platform": platform.platform(),
        "input": str(args.input),
        "max_samples": args.max_samples,
        "cross_check_file": str(args.cross_check),
        "tolerances": TOL,
        "graphs": {
            "chain": chain.to_dict(),
            "prefix0-g2": g2.to_dict(),
            "window-mid-g2": win.to_dict(),
            **{f"prefix{k}-g2": d.to_dict() for k, d in prefix_dags.items()},
        },
        "git_commit": git_sha(),
        "note": ("Correctness gate only. No speedup is claimed from the graph "
                 "and no result here is used for structure selection."),
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
            if ids.shape[1] < 2:
                continue

            ref_logits = model(ids)
            mx.eval(ref_logits)
            h0 = embed(model, ids, mx)
            mask = create_attention_mask(h0, None)
            states, _ = teacher_pass(model, ids, mask, mx)

            def logits_of(h):
                out = project_logits(model, h, mx)
                mx.eval(out)
                return out

            # ---- G0.1a: structural serial reduction, must be bit-exact ----
            h_chain = dag_forward(model, h0, chain, mask, mx)
            mx.eval(h_chain)
            g01a = {
                **logits_metrics(ref_logits, logits_of(h_chain), ids, mx),
                "state": state_metrics(h_chain, states[-1], mx),
            }

            # ---- G0.1b: merge machinery without the serial shortcut ----
            h_delta_chain = dag_forward(model, h0, chain, mask, mx, chain_mode="off")
            mx.eval(h_delta_chain)
            g01b = {
                **logits_metrics(ref_logits, logits_of(h_delta_chain), ids, mx),
                "state": state_metrics(h_delta_chain, states[-1], mx),
            }

            # ---- G0.2a/b: g2 mimic vs production, production vs run_plan ----
            h_g2 = dag_forward(model, h0, g2, mask, mx)
            mx.eval(h_g2)
            h_prod = g2_reference(model, h0, mask, mx)
            out_plan_g2, _ = run_plan(model, ids, mask, make_plan(n, "group2"),
                                      states, "fp32", mx, keep_group_records=False)
            h_mono = monolithic_reference(model, h0, g2, mask, mx)
            mx.eval(h_mono)
            g02a = state_metrics(h_g2, h_prod, mx)
            g02b = state_metrics(h_prod, out_plan_g2, mx)
            g02e = state_metrics(h_g2, h_mono, mx)
            g02_dag = logits_metrics(ref_logits, logits_of(h_g2), ids, mx)
            g02_plan = logits_metrics(ref_logits, logits_of(out_plan_g2), ids, mx)

            # ---- G0.2c: prefix-k g2 vs stored Stage 1.5 numbers ----
            g02c = {}
            for k, d in prefix_dags.items():
                h = dag_forward(model, h0, d, mask, mx)
                mx.eval(h)
                m = logits_metrics(ref_logits, logits_of(h), ids, mx)
                key = f"prefix{k}-g2@fp32"
                ref_v = stored.get(sample["sample_id"], {}).get(key)
                g02c[f"prefix{k}-g2"] = {
                    "dag_delta_nll": m["delta_nll"],
                    "stored_delta_nll": (ref_v or {}).get("delta_nll"),
                    "abs_diff": (abs(m["delta_nll"] - ref_v["delta_nll"])
                                 if ref_v else None),
                    "dag_top1": m["top1_agreement"],
                    "critical_path": d.critical_path,
                }

            # ---- G0.2d: window DAG vs run_plan in the same process ----
            out_plan_win, _ = run_plan(model, ids, mask, make_plan(n, "window-mid-g2"),
                                       states, "fp32", mx, keep_group_records=False)
            h_win = dag_forward(model, h0, win, mask, mx)
            mx.eval(h_win)
            g02d = {
                "state": state_metrics(h_win, out_plan_win, mx),
                "dag_delta_nll": logits_metrics(ref_logits, logits_of(h_win), ids, mx)["delta_nll"],
                "plan_delta_nll": logits_metrics(ref_logits, logits_of(out_plan_win), ids, mx)["delta_nll"],
                "critical_path": win.critical_path,
            }

            row = {
                "sample_id": sample["sample_id"],
                "n_tokens": int(ids.shape[1]),
                "g01a_chain_vs_reference": g01a,
                "g01b_merge_no_shortcut_vs_reference": g01b,
                "g02a_g2dag_vs_production": g02a,
                "g02b_production_vs_run_plan": g02b,
                "g02_g2_logits_dag": g02_dag,
                "g02_g2_logits_plan": g02_plan,
                "g02e_g2dag_vs_monolithic_fp32": g02e,
                "g02c_prefix_vs_stored": g02c,
                "g02d_windowdag_vs_run_plan": g02d,
            }
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            rows.append(row)
            n_done += 1
            print(f"[{n_done}/{args.max_samples}] {sample['sample_id']} "
                  f"g01a dNLL={g01a['delta_nll']:.2e} max_abs={g01a['state']['max_abs_diff']:.2e} | "
                  f"g01b dNLL={g01b['delta_nll']:.4f} rel={g01b['state']['rel_rms_diff']:.4f} | "
                  f"g02a max_abs={g02a['max_abs_diff']:.2e} | "
                  f"g02b max_abs={g02b['max_abs_diff']:.2e} | "
                  f"g02c max|diff|={max((v['abs_diff'] or 0.0) for v in g02c.values()):.2e} | "
                  f"g02d max_abs={g02d['state']['max_abs_diff']:.2e}", flush=True)

    # ---- summary with a-priori pass/fail ----
    s_maxabs = max(r["g01a_chain_vs_reference"]["state"]["max_abs_diff"] for r in rows)
    s_top1 = min(r["g01a_chain_vs_reference"]["top1_agreement"] for r in rows)
    s_dnll = max(abs(r["g01a_chain_vs_reference"]["delta_nll"]) for r in rows)
    b_rel = max(r["g01b_merge_no_shortcut_vs_reference"]["state"]["rel_rms_diff"] for r in rows)
    b_dnll = max(abs(r["g01b_merge_no_shortcut_vs_reference"]["delta_nll"]) for r in rows)
    a_maxabs = max(r["g02a_g2dag_vs_production"]["max_abs_diff"] for r in rows)
    b2_maxabs = max(r["g02b_production_vs_run_plan"]["max_abs_diff"] for r in rows)
    c_max = max((v["abs_diff"] for r in rows for v in r["g02c_prefix_vs_stored"].values()
                 if v["abs_diff"] is not None), default=None)
    c_n = sum(1 for r in rows for v in r["g02c_prefix_vs_stored"].values()
              if v["abs_diff"] is not None)
    d_maxabs = max(r["g02d_windowdag_vs_run_plan"]["state"]["max_abs_diff"] for r in rows)
    d_dnll = max(abs(r["g02d_windowdag_vs_run_plan"]["dag_delta_nll"]
                     - r["g02d_windowdag_vs_run_plan"]["plan_delta_nll"]) for r in rows)
    e_rel = max(r["g02e_g2dag_vs_monolithic_fp32"]["rel_rms_diff"] for r in rows)

    checks = {
        "G0.1a_chain_bit_exact": {
            "metric": {"max_abs_delta_nll": s_dnll, "min_top1": s_top1, "max_state_abs": s_maxabs},
            "tolerance": {"delta_nll": TOL["g01a_delta_nll"], "top1_min": 1.0,
                          "max_abs": TOL["g01a_max_abs"]},
            "pass": (s_dnll <= TOL["g01a_delta_nll"] and s_top1 >= 1.0
                     and s_maxabs <= TOL["g01a_max_abs"]),
        },
        "G0.1b_merge_noise": {
            "metric": {"max_rel_rms": b_rel, "max_abs_delta_nll": b_dnll},
            "tolerance": {"rel_rms": TOL["g01b_rel_rms"], "delta_nll": TOL["g01b_delta_nll"]},
            "pass": b_rel <= TOL["g01b_rel_rms"] and b_dnll <= TOL["g01b_delta_nll"],
        },
        "G0.2a_g2dag_vs_production": {
            "metric": {"max_abs": a_maxabs},
            "tolerance": {"max_abs": TOL["g02a_max_abs"]},
            "pass": a_maxabs <= TOL["g02a_max_abs"],
        },
        "G0.2b_production_vs_run_plan": {
            "metric": {"max_abs": b2_maxabs},
            "tolerance": {"max_abs": TOL["g02b_max_abs"]},
            "pass": b2_maxabs <= TOL["g02b_max_abs"],
        },
        "G0.2c_prefix_vs_stored": {
            "metric": {"max_abs_delta_nll_diff": c_max, "n_compared": c_n},
            "tolerance": {"delta_nll": TOL["g02c_delta_nll"]},
            "pass": (c_max is not None and c_max <= TOL["g02c_delta_nll"]),
        },
        "G0.2d_windowdag_vs_run_plan": {
            "metric": {"max_abs": d_maxabs, "max_abs_delta_nll_diff": d_dnll},
            "tolerance": {"max_abs": TOL["g02d_max_abs"]},
            "pass": d_maxabs <= TOL["g02d_max_abs"],
        },
    }
    overall = all(c["pass"] for c in checks.values())
    summary = {
        "n_samples": len(rows),
        "checks": checks,
        "informational": {
            "g02e_g2dag_vs_monolithic_fp32_max_rel_rms": e_rel,
            "note": ("merge-point materialization vs one fp32 sum: both valid "
                     "bf16 schedules; the executor pins the production one"),
        },
        "overall_pass": overall,
        "gate": ("PASS: executor may be used for dependency experiments"
                 if overall else "FAIL: fix the executor before any search"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["status"] = "MEASURED"
    manifest["n_measured"] = len(rows)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v["pass"] for k, v in checks.items()}, indent=2))
    print(f"overall_pass={overall} -> {args.output}")


if __name__ == "__main__":
    main()
