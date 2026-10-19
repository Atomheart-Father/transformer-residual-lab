"""Stage-1.5 B3: is the early-layer parallel failure a magnitude or a direction error?

The question matters because it decides what a compensation mechanism would have
to do. If the group output is roughly right but too large / too small, a scalar
is enough (B1-style). If the direction is wrong, no scalar can help.

Three oracle interventions are applied at the *exit of the first parallel group*,
after which the rest of the graph keeps running freely:

  oracle-norm  h <- h * (||h^T|| / ||h||)        fixes magnitude, keeps direction
  oracle-dir   h <- ||h|| * (h^T / ||h^T||)      fixes direction, keeps magnitude
  oracle-both  h <- h^T                          upper bound: no error at all

These use the teacher state, so they are diagnostics only. Nothing here is a
deployable method and nothing is trained. Weights stay frozen.

We also report the exact Pythagorean split of the squared relative error at the
same boundary:

  r^2 = ||h_hat - h||^2 / ||h||^2 = (rho - cos)^2 + (1 - cos^2)
      rho = ||h_hat|| / ||h||

  magnitude share = (rho - cos)^2 / r^2
  direction share = (1 - cos^2)   / r^2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from measure import (
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
    logits_metrics,
    make_plan,
    project_logits,
    teacher_pass,
)


def first_parallel_group(plan) -> int:
    """Index of the first group that actually has more than one layer."""
    for gi, group in enumerate(plan.groups):
        if len(group) > 1:
            return gi
    raise ValueError(f"{plan.name} has no parallel group")


def interventional_run(model, ids, mask, plan, teacher_states, mode, at_group, mx):
    """Free-running plan, with one oracle intervention at a group exit."""
    h = embed(model, ids, mx)
    stats = None
    for gi, group in enumerate(plan.groups):
        s, e = group[0], group[-1] + 1
        t_e = teacher_states[e]
        if len(group) == 1:
            out = model.model.layers[group[0]](h, mask, None)
            mx.eval(out)
        else:
            deltas = [model.model.layers[i](h, mask, None) - h for i in group]
            acc = h.astype(mx.float32)
            for d in deltas:
                acc = acc + d.astype(mx.float32)
            out = acc.astype(h.dtype)
            mx.eval(out)

        if gi == at_group:
            hh = out.astype(mx.float32)
            tt = t_e.astype(mx.float32)
            nh = mx.sqrt(mx.sum(hh * hh))
            nt = mx.sqrt(mx.sum(tt * tt))
            dot = mx.sum(hh * tt)
            diff = hh - tt
            nd = mx.sqrt(mx.sum(diff * diff))
            mx.eval(nh, nt, dot, nd)
            nh_f, nt_f = float(nh.item()), float(nt.item())
            nd_f, dot_f = float(nd.item()), float(dot.item())
            rho = nh_f / max(nt_f, 1e-12)
            cos = dot_f / max(nh_f * nt_f, 1e-12)
            r = nd_f / max(nt_f, 1e-12)
            r2 = r * r
            mag2 = (rho - cos) ** 2
            dir2 = max(1.0 - cos * cos, 0.0)
            stats = {
                "group_index": gi, "start": s, "end": e, "size": len(group),
                "rel_rms_diff": r, "rho": rho, "cosine": cos,
                "magnitude_share": mag2 / r2 if r2 > 1e-12 else 0.0,
                "direction_share": dir2 / r2 if r2 > 1e-12 else 0.0,
            }
            if mode == "oracle-norm":
                # keep direction, take the teacher's norm
                out = (hh * (nt / mx.maximum(nh, 1e-12))).astype(out.dtype)
            elif mode == "oracle-dir":
                # keep magnitude, take the teacher's direction
                out = (tt * (nh / mx.maximum(nt, 1e-12))).astype(out.dtype)
            elif mode == "oracle-both":
                out = tt.astype(out.dtype)
            elif mode == "none":
                pass
            else:
                raise ValueError(f"unknown intervention: {mode}")
            mx.eval(out)
        h = out
    return h, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PINNED_MODEL_REPO)
    ap.add_argument("--revision", default=PINNED_MODEL_REVISION)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/stage1_5/b3"))
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--plans", default="prefix0-g2,prefix3-g8,prefix3-g4,prefix10-g4")
    ap.add_argument("--at-group", default="first-parallel",
                    help='"first-parallel" (default) or an integer group index')
    ap.add_argument("--modes", default="none,oracle-norm,oracle-dir,oracle-both",
                    help="comma separated subset of the intervention modes")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    if args.revision is None:
        raise SystemExit("refusing to run with an unpinned model revision")

    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    args.output.mkdir(parents=True, exist_ok=True)
    plans = [make_plan(n, p.strip()) for p in args.plans.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    at_group = {p.name: (first_parallel_group(p) if args.at_group == "first-parallel"
                         else int(args.at_group)) for p in plans}

    manifest = {
        "status": "RUNNING", "model": args.model, "model_revision": args.revision,
        "n_layers": n, "mlx_version": getattr(mx, "__version__", "UNKNOWN"),
        "mlx_lm_version": getattr(mlx_lm, "__version__", "UNKNOWN"),
        "input": str(args.input), "max_samples": args.max_samples,
        "plans": [{"name": p.name, "executed_blocks": p.executed_blocks,
                   "effective_depth": p.effective_depth, "groups": p.groups,
                   "intervention_group": at_group[p.name]}
                  for p in plans],
        "at_group": args.at_group, "modes": modes,
        "warning": "Oracle diagnostics only. Teacher state is used at the "
                   "intervention point. No trained component, no speedup claim.",
        "git_commit": git_sha(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    done = 0
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as sf:
        for line in args.input.open(encoding="utf-8"):
            if done >= args.max_samples:
                break
            sample = json.loads(line)
            ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
            if ids.shape[1] < 2:
                continue
            ref_logits = model(ids)
            mx.eval(ref_logits)
            h0 = embed(model, ids, mx)
            mask = create_attention_mask(h0, None)
            states, _ = teacher_pass(model, ids, mask, mx)

            row = {"sample_id": sample["sample_id"], "n_tokens": int(ids.shape[1]),
                   "variants": {}}
            for plan in plans:
                for mode in modes:
                    out, stats = interventional_run(
                        model, ids, mask, plan, states, mode, at_group[plan.name], mx)
                    logits = project_logits(model, out, mx)
                    mx.eval(logits)
                    key = f"{plan.name}::{mode}"
                    row["variants"][key] = {
                        **logits_metrics(ref_logits, logits, ids, mx),
                        "executed_blocks": plan.executed_blocks,
                        "effective_depth": plan.effective_depth,
                        "intervention": mode,
                        "boundary": stats,
                    }
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            done += 1
            print(f"[{done}/{args.max_samples}] {sample['sample_id']} " + " ".join(
                f"{k.split('::')[0]}/{k.split('::')[1]}={v['delta_nll']:.2f}"
                for k, v in row["variants"].items()), flush=True)

    manifest["status"] = "MEASURED" if done else "NO_SAMPLES"
    manifest["n_measured"] = done
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"saved {done} samples -> {args.output}")


if __name__ == "__main__":
    main()
