"""Offline ridge calibration of residual fusion coefficients.

For one plan and one scheme, solve per parallel group

    min_alpha || (h_e^T - a) - sum_i alpha_i * (B_i(a) - a) ||^2 + lam ||alpha - 1||^2

where a is the *teacher* state at the group entry. This is an offline
reconstruction fit; the teacher state is never available at run time, and the
fitted alpha is frozen before any free-running evaluation.

Schemes
  group_ls  one scalar per group, multiplying sum_i d_i
  layer_ls  one scalar per layer

Usage
  python scripts/calibrate.py --input data257/calibration.jsonl \
      --plans prefix3-g4,prefix4-g4 --schemes group_ls,layer_ls \
      --lambda-grid 0,1e-3,1e-2,1e-1 --out results/stage1_5/coeffs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import PINNED_MODEL_REPO, PINNED_MODEL_REVISION, make_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PINNED_MODEL_REPO)
    ap.add_argument("--revision", default=PINNED_MODEL_REVISION)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--schemes", default="group_ls,layer_ls")
    ap.add_argument("--lambda-grid", default="0,1e-3,1e-2,1e-1")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--clip", type=float, nargs=2, default=(0.0, 2.0))
    ap.add_argument("--out", type=Path, default=Path("results/stage1_5/coeffs"))
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, _, _ = load(args.model, revision=args.revision, return_config=True)
    model.eval()
    n = len(model.model.layers)
    plans = [make_plan(n, p.strip()) for p in args.plans.split(",") if p.strip()]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    lams = [float(x) for x in args.lambda_grid.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    # per plan -> per group index -> normal equations
    stats = {p.name: {} for p in plans}
    samples = 0
    t0 = time.time()

    for line in args.input.open(encoding="utf-8"):
        if samples >= args.max_samples:
            break
        sample = json.loads(line)
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h = model.model.embed_tokens(ids)
        mask = create_attention_mask(h, None)
        states = [h]
        for block in model.model.layers:
            h = block(h, mask, None)
            mx.eval(h)
            states.append(h)

        for plan in plans:
            for gi, group in enumerate(plan.groups):
                if len(group) == 1:
                    continue
                s, e = group[0], group[-1] + 1
                a = states[s]
                y = (states[e] - a).astype(mx.float32).reshape(-1)
                X = mx.stack([(model.model.layers[i](a, mask, None) - a)
                              .astype(mx.float32).reshape(-1) for i in group], axis=1)
                mx.eval(X, y)
                A = np.asarray(X.T @ X, dtype=np.float64)
                b = np.asarray(X.T @ y, dtype=np.float64)
                yty = float((y.astype(mx.float32) * y).sum().item())
                d = stats[plan.name].setdefault(
                    gi, {"start": s, "end": e, "layers": group, "A": 0.0, "b": 0.0,
                         "yty": 0.0, "n": 0})
                d["A"] = d["A"] + A
                d["b"] = d["b"] + b
                d["yty"] += yty
                d["n"] += int(y.size)
        samples += 1
        if samples % 8 == 0:
            print(f"  fitted {samples} samples", flush=True)

    fit_seconds = time.time() - t0
    print(f"normal equations from {samples} samples in {fit_seconds:.1f}s")

    lo, hi = args.clip
    written = []
    for plan in plans:
        for scheme in schemes:
            for lam in lams:
                alpha, diag = {}, {"groups": [], "n_clipped": 0}
                for gi, d in sorted(stats[plan.name].items()):
                    A, b = d["A"], d["b"]
                    k = len(d["layers"])
                    scale = max(np.trace(A) / k, 1e-30)
                    if scheme == "group_ls":
                        A1 = np.array([[A.sum()]])
                        b1 = np.array([b.sum()])
                        lhs, rhs = A1 + lam * scale * np.eye(1), b1 + lam * scale * np.ones(1)
                        try:
                            sol = np.linalg.solve(lhs, rhs)
                        except np.linalg.LinAlgError:
                            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                        a_vec = np.full(k, sol[0])
                    elif scheme == "layer_ls":
                        lhs, rhs = A + lam * scale * np.eye(k), b + lam * scale * np.ones(k)
                        try:
                            sol = np.linalg.solve(lhs, rhs)
                        except np.linalg.LinAlgError:
                            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                        a_vec = sol
                    else:
                        raise ValueError(f"unknown scheme {scheme}")
                    clipped = np.clip(a_vec, lo, hi)
                    diag["n_clipped"] += int(np.sum(clipped != a_vec))
                    a_vec = clipped
                    # reconstruction residual of the fitted group
                    res = d["yty"] - 2 * a_vec @ b + a_vec @ A @ a_vec
                    base = max(d["yty"], 1e-30)
                    diag["groups"].append({
                        "group_index": gi, "start": d["start"], "end": d["end"],
                        "layers": d["layers"],
                        "alpha": [float(x) for x in a_vec],
                        "rel_reconstruction_rms": float(np.sqrt(max(res, 0.0) / base)),
                        "cond": float(np.linalg.cond(A + lam * scale * np.eye(k))),
                    })
                    for i, v in zip(d["layers"], a_vec):
                        alpha[str(i)] = float(v)
                out = {
                    "plan": plan.name,
                    "scheme": scheme,
                    "lambda_rel": lam,
                    "alpha": alpha,
                    "clip": [lo, hi],
                    "n_clipped": diag["n_clipped"],
                    "groups": diag["groups"],
                    "fit": {
                        "n_samples": samples,
                        "n_elements": int(sum(d["n"] for d in stats[plan.name].values())),
                        "seconds": round(fit_seconds, 2),
                        "model_revision": args.revision,
                        "note": "alpha fitted on teacher group-entry states, then frozen; "
                                "teacher states are never used during free running",
                    },
                }
                name = f"{plan.name}__{scheme}__lam{lam:g}.json"
                (args.out / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
                written.append(name)
                print(f"  {name}  clipped={diag['n_clipped']}  "
                      f"rel_recon={np.mean([g['rel_reconstruction_rms'] for g in diag['groups']]):.3f}")

    print(f"wrote {len(written)} coefficient files -> {args.out}")


if __name__ == "__main__":
    main()
