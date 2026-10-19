"""Stage 1.6 Plan B: can a tiny network predict the serial-minus-parallel gap?

One compensation site only. At the exit of a single parallel group we define

    c* = h_teacher - h_parallel_group_output

and fit a small map C_theta that sees *only* the group entry state that the
candidate actually has at inference time:

    c_hat = C_theta(a),   h_fixed = h_parallel + c_hat

The teacher state is used to build the label c* and for the oracle upper bound.
It is never an input to C_theta. Training features are collected while the
candidate runs *freely*, so the input distribution matches deployment; using
teacher entries instead would leak a distribution the model never sees.

Stages (run in this order):

    collect  -> dump (a, c*) per token for one plan and one group
    fit      -> rank-r linear (closed form) and rank-r GELU MLP (Adam)
    eval     -> apply the frozen compensator and measure end-to-end quality

Nothing here trains the transformer. All base weights stay frozen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from measure import (
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
    logits_metrics,
    make_plan,
    resolve_plan,
    project_logits,
    state_metrics,
    teacher_pass,
)


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def run_group(model, h, group, mask, mx):
    if len(group) == 1:
        out = model.model.layers[group[0]](h, mask, None)
        mx.eval(out)
        return out
    deltas = [model.model.layers[i](h, mask, None) - h for i in group]
    acc = h.astype(mx.float32)
    for d in deltas:
        acc = acc + d.astype(mx.float32)
    out = acc.astype(h.dtype)
    mx.eval(out)
    return out


def advance_to_group(model, h, plan, mask, mx, upto):
    for group in plan.groups[:upto]:
        h = run_group(model, h, group, mask, mx)
    return h


def run_with_patch(model, h, plan, mask, mx, patch_gi, patch_fn):
    """Free-running plan, with patch_fn(entry) -> replacement exit at patch_gi."""
    for gi, group in enumerate(plan.groups):
        if gi == patch_gi and patch_fn is not None:
            a = h
            h = patch_fn(a, group)
        else:
            h = run_group(model, h, group, mask, mx)
    return h


# ---------------------------------------------------------------------------
# compensator forms
# ---------------------------------------------------------------------------


class LinearLowRank:
    """c_hat = (x - x_mean) @ A @ B + y_mean, with A [D,r] and B [r,D]."""

    def __init__(self, A, B, x_mean, y_mean):
        self.A, self.B = A, B
        self.x_mean, self.y_mean = x_mean, y_mean
        self.r = A.shape[1]

    @property
    def n_params(self):
        return int(self.A.size + self.B.size)

    def predict_np(self, X):
        return ((X - self.x_mean) @ self.A) @ self.B + self.y_mean

    def predict_mx(self, x, mx):
        z = (x.astype(mx.float32) - mx.array(self.x_mean)) @ mx.array(self.A)
        return z @ mx.array(self.B) + mx.array(self.y_mean)

    def save(self, path):
        np.savez(path, A=self.A, B=self.B,
                 x_mean=self.x_mean, y_mean=self.y_mean, kind="linear")


class MLPCompensator:
    """c_hat = (gelu((x - x_mean) W1^T + b1)) W2^T + b2 + y_mean."""

    KEYS = ("W1", "b1", "W2", "b2")

    def __init__(self, params, x_mean, y_mean):
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        self.x_mean, self.y_mean = x_mean, y_mean
        self.r = self.W1.shape[0]

    @property
    def n_params(self):
        return sum(int(p.size) for p in (self.W1, self.b1, self.W2, self.b2))

    def predict_np(self, X):
        import mlx.core as mx
        out = self.predict_mx(mx.array(X.astype(np.float32)), mx)
        mx.eval(out)
        return np.asarray(out)

    def predict_mx(self, x, mx):
        import mlx.nn as nn
        z = x.astype(mx.float32) - mx.array(self.x_mean)
        h = nn.gelu(z @ mx.array(self.W1.T) + mx.array(self.b1))
        return h @ mx.array(self.W2.T) + mx.array(self.b2) + mx.array(self.y_mean)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 x_mean=self.x_mean, y_mean=self.y_mean, kind="mlp")


def load_compensator(path: Path):
    z = np.load(path)
    kind = str(z["kind"])
    if kind == "linear":
        return LinearLowRank(z["A"], z["B"], z["x_mean"], z["y_mean"])
    return MLPCompensator({k: z[k] for k in MLPCompensator.KEYS},
                          z["x_mean"], z["y_mean"])


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def cmd_collect(args, mx, load):
    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    plan = resolve_plan(n, args.plan)
    gi = args.group_index if args.group_index >= 0 else \
        max(i for i, g in enumerate(plan.groups) if len(g) > 1)
    group = plan.groups[gi]

    from mlx_lm.models.base import create_attention_mask
    args.output.mkdir(parents=True, exist_ok=True)
    As, Cs, meta = [], [], []
    done = 0
    for line in args.input.open(encoding="utf-8"):
        if done >= args.max_samples:
            break
        sample = json.loads(line)
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states, _ = teacher_pass(model, ids, mask, mx)

        h = advance_to_group(model, h0, plan, mask, mx, gi)
        a = h
        out = run_group(model, h, group, mask, mx)
        h_teacher = states[group[-1] + 1]
        c = (h_teacher.astype(mx.float32) - out.astype(mx.float32))
        mx.eval(a, out, c)

        As.append(np.asarray(a.astype(mx.float32)).reshape(-1, a.shape[-1]))
        Cs.append(np.asarray(c).reshape(-1, c.shape[-1]))
        meta.append({"sample_id": sample["sample_id"],
                     "tokens_sha256": sample.get("tokens_sha256")})
        done += 1
        print(f"[collect {done}/{args.max_samples}] {sample['sample_id']} "
              f"c_rms={float(mx.sqrt(mx.mean(c * c)).item()):.4f}", flush=True)

    A = np.concatenate(As, axis=0).astype(np.float32)
    C = np.concatenate(Cs, axis=0).astype(np.float32)
    np.savez(args.output / "pairs.npz", A=A, C=C)
    manifest = {
        "stage": "collect", "plan": plan.name, "groups": plan.groups,
        "group_index": gi, "group_layers": group,
        "effective_depth": plan.effective_depth,
        "executed_blocks": plan.executed_blocks,
        "model": args.model, "model_revision": args.revision,
        "input": str(args.input), "n_samples": done,
        "n_rows": int(A.shape[0]), "hidden": int(A.shape[1]),
        "input_is": "free-running candidate entry state (deployable)",
        "label_is": "teacher end state minus candidate parallel group output",
        "c_rms": float(np.sqrt((C ** 2).mean())),
        "a_rms": float(np.sqrt((A ** 2).mean())),
        "git_commit": git_sha(),
    }
    (args.output / "collect_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"saved {A.shape} pairs -> {args.output}")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def fit_linear(X, Y, rank, lam_rel, rng):
    """Reduced-rank regression, done properly.

    Ridge followed by an SVD truncation of the *solution* keeps the largest
    singular values of W, which for a near-singular design are exactly the
    noise-amplified directions. That blows up (we measured pred_rms 167 against
    a target rms of 13). Whitening X first and then truncating the whitened
    cross-covariance keeps the directions that actually explain Y.
    """
    x_mean = X.mean(0)
    y_mean = Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean
    G = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(G)
    w = np.maximum(w, 0.0)
    keep = w > max(w.max() * lam_rel, 1e-12)
    w, V = w[keep], V[:, keep]
    Wh = V / np.sqrt(w)                       # [D, k], whitening
    Z = Xc @ Wh                               # [N, k]
    Mz = (Z.T @ Yc) / len(Z)                  # [k, D]
    U, S, Vt = np.linalg.svd(Mz, full_matrices=False)
    r = min(rank, len(S))
    A = Wh @ (U[:, :r] * S[:r])               # [D, r]
    B = Vt[:r, :]                             # [r, D]
    return LinearLowRank(A, B, x_mean, y_mean), S


def fit_mlp(X, Y, rank, epochs, batch, lr, seed, verbose=True):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    x_mean = X.mean(0)
    y_mean = Y.mean(0)
    Xc = mx.array(X - x_mean)
    Yc = mx.array(Y - y_mean)
    n, d = X.shape
    mx.random.seed(seed)
    params = {
        "W1": mx.random.normal((rank, d)) * (1.0 / np.sqrt(d)),
        "b1": mx.zeros((rank,)),
        "W2": mx.random.normal((d, rank)) * (1.0 / np.sqrt(rank)),
        "b2": mx.zeros((d,)),
    }

    def forward(p, x):
        h = nn.gelu(x @ p["W1"].T + p["b1"])
        return h @ p["W2"].T + p["b2"]

    def loss_fn(p, xb, yb):
        return mx.mean((forward(p, xb) - yb) ** 2)

    grad_fn = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        order = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = mx.array(order[s:s + batch].astype(np.int32))
            xb, yb = Xc[idx], Yc[idx]
            loss, grads = grad_fn(params, xb, yb)
            params = opt.apply_gradients(grads, params)
            mx.eval(params, loss)
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        rmse = float(np.sqrt(tot / cnt))
        history.append({"epoch": ep, "rmse": rmse})
        if verbose:
            print(f"  epoch {ep:3d} rmse={rmse:.5f}", flush=True)
    out = {k: np.asarray(v) for k, v in params.items()}
    return MLPCompensator(out, x_mean, y_mean), history


def cmd_fit(args, mx, load):
    z = np.load(args.pairs)
    A, C = z["A"], z["C"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(A))
    n_val = max(1, int(round(len(A) * args.val_frac)))
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, Ytr = A[tr], C[tr]
    Xva, Yva = A[va], C[va]

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"pairs": str(args.pairs), "n_train_rows": len(tr),
              "n_val_rows": len(va), "rank_linear": args.rank_linear,
              "rank_mlp": args.rank_mlp, "models": {}}

    def score(model, X, Y, tag):
        P = model.predict_np(X)
        err = P - Y
        rms_err = float(np.sqrt((err ** 2).mean()))
        rms_y = float(np.sqrt((Y ** 2).mean()))
        num = float((P * Y).sum())
        den = float(np.sqrt((P ** 2).sum() * (Y ** 2).sum())) + 1e-12
        return {"split": tag, "pred_rms": float(np.sqrt((P ** 2).mean())),
                "target_rms": rms_y, "err_rms": rms_err,
                "rel_err": rms_err / max(rms_y, 1e-12),
                "cosine": num / den}

    # zero predictor, as the honest floor
    zero = {"split": "val", "pred_rms": 0.0,
            "target_rms": float(np.sqrt((Yva ** 2).mean())),
            "err_rms": float(np.sqrt((Yva ** 2).mean())), "rel_err": 1.0,
            "cosine": 0.0}
    report["models"]["zero"] = {"n_params": 0, "val": zero}

    lin, spec = fit_linear(Xtr, Ytr, args.rank_linear, args.lam_rel, rng)
    lin.save(args.output / f"linear_r{args.rank_linear}.npz")
    report["models"]["linear"] = {
        "n_params": lin.n_params, "rank": lin.r,
        "val": score(lin, Xva, Yva, "val"),
        "train": score(lin, Xtr, Ytr, "train"),
        "spectrum_top8": [float(v) for v in spec[:8]],
        "spectrum_sum": float(spec.sum()),
        "predictable_energy_top8": float((spec[:8] ** 2).sum() / (spec ** 2).sum()),
    }
    print("linear:", json.dumps(report["models"]["linear"]["val"], indent=1))

    mlp, hist = fit_mlp(Xtr, Ytr, args.rank_mlp, args.epochs, args.batch,
                        args.lr, args.seed)
    mlp.save(args.output / f"mlp_r{args.rank_mlp}.npz")
    report["models"]["mlp"] = {
        "n_params": mlp.n_params, "rank": mlp.r,
        "val": score(mlp, Xva, Yva, "val"),
        "train": score(mlp, Xtr, Ytr, "train"),
        "history": hist,
    }
    print("mlp:", json.dumps(report["models"]["mlp"]["val"], indent=1))

    (args.output / "fit_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args, mx, load):
    from mlx_lm.models.base import create_attention_mask
    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    plan = resolve_plan(n, args.plan)
    gi = args.group_index if args.group_index >= 0 else \
        max(i for i, g in enumerate(plan.groups) if len(g) > 1)

    comps = {}
    for path in args.compensator or []:
        p = Path(path)
        comps[p.stem] = load_compensator(p)

    args.output.mkdir(parents=True, exist_ok=True)
    variants = ["serial", f"{plan.name}", f"{plan.name}+oracle"]
    variants += [f"{plan.name}+{k}" for k in comps]

    rows, done = [], 0
    t_comp = {k: 0.0 for k in comps}
    for line in args.input.open(encoding="utf-8"):
        if done >= args.max_samples:
            break
        sample = json.loads(line)
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states, _ = teacher_pass(model, ids, mask, mx)
        ref_logits = model(ids)
        mx.eval(ref_logits)
        row = {"sample_id": sample["sample_id"], "variants": {}}

        serial_out = states[-1]
        row["variants"]["serial"] = logits_metrics(
            ref_logits, project_logits(model, serial_out, mx), ids, mx)

        base_out = run_with_patch(model, h0, plan, mask, mx, -1, None)
        row["variants"][plan.name] = {
            **logits_metrics(ref_logits, project_logits(model, base_out, mx), ids, mx),
            "final_state": state_metrics(base_out, states[-1], mx)}

        # what the group entry looks like, and how good the prediction is there
        a = advance_to_group(model, h0, plan, mask, mx, gi)
        group = plan.groups[gi]
        par = run_group(model, a, group, mask, mx)
        truth = states[group[-1] + 1]
        c_star = truth.astype(mx.float32) - par.astype(mx.float32)
        mx.eval(c_star)
        cs = np.asarray(c_star)
        comp_diag = {}

        def make_patch(fn, sink):
            def patch(entry, grp):
                t0 = time.perf_counter()
                pred = fn(entry, mx)
                mx.eval(pred)
                sink["predict_seconds"] = sink.get("predict_seconds", 0.0) + \
                    (time.perf_counter() - t0)
                out = (run_group(model, entry, grp, mask, mx).astype(mx.float32)
                       + pred.astype(mx.float32))
                mx.eval(out)
                return out.astype(entry.dtype)
            return patch

        def make_oracle_patch():
            def patch(entry, grp):
                return truth
            return patch

        row["variants"][f"{plan.name}+oracle"] = {
            **logits_metrics(ref_logits, project_logits(
                model, run_with_patch(model, h0, plan, mask, mx, gi,
                                      make_oracle_patch()), mx), ids, mx)}

        for name, comp in comps.items():
            sink = {}
            out = run_with_patch(model, h0, plan, mask, mx, gi, make_patch(
                lambda e, m, c=comp: c.predict_mx(e, m), sink))
            t_comp[name] += sink.get("predict_seconds", 0.0)
            pred = comp.predict_np(np.asarray(a.astype(mx.float32)).reshape(-1, a.shape[-1]))
            err = pred - cs.reshape(-1, cs.shape[-1])
            num = float((pred * cs.reshape(-1, cs.shape[-1])).sum())
            den = float(np.sqrt((pred ** 2).sum() * (cs ** 2).sum())) + 1e-12
            row["variants"][f"{plan.name}+{name}"] = {
                **logits_metrics(ref_logits, project_logits(model, out, mx), ids, mx),
                "final_state": state_metrics(out, states[-1], mx),
                "comp": {
                    "target_rms": float(np.sqrt((cs ** 2).mean())),
                    "pred_rms": float(np.sqrt((pred ** 2).mean())),
                    "err_rms": float(np.sqrt((err ** 2).mean())),
                    "rel_err": float(np.sqrt((err ** 2).mean())
                                     / max(np.sqrt((cs ** 2).mean()), 1e-12)),
                    "cosine": num / den,
                }}
            comp_diag[name] = row["variants"][f"{plan.name}+{name}"]["comp"]
        row["boundary"] = {"group_index": gi, "layers": group,
                           "c_star_rms": float(np.sqrt((cs ** 2).mean())),
                           "comp": comp_diag}
        rows.append(row)
        done += 1
        print(f"[eval {done}/{args.max_samples}] {sample['sample_id']} " + " ".join(
            f"{k.split('+')[-1]}={v['delta_nll']:.3f}"
            for k, v in row["variants"].items()), flush=True)

    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = {}
    for v in variants:
        ds = [r["variants"][v]["delta_nll"] for r in rows if v in r["variants"]]
        if not ds:
            continue
        agg[v] = {"n": len(ds), "delta_nll_mean": float(np.mean(ds)),
                  "delta_nll_std": float(np.std(ds, ddof=1)),
                  "kl_mean": float(np.mean([r["variants"][v]["teacher_to_candidate_kl"]
                                            for r in rows if v in r["variants"]])),
                  "top1_mean": float(np.mean([r["variants"][v]["top1_agreement"]
                                              for r in rows if v in r["variants"]]))}
    base = agg.get(plan.name, {}).get("delta_nll_mean")
    orc = agg.get(f"{plan.name}+oracle", {}).get("delta_nll_mean")
    if base is not None and orc is not None and base > orc:
        for v, a in agg.items():
            a["recovery_vs_oracle"] = (base - a["delta_nll_mean"]) / (base - orc)

    manifest = {
        "stage": "eval", "plan": plan.name, "groups": plan.groups,
        "group_index": gi, "group_layers": plan.groups[gi],
        "effective_depth": plan.effective_depth,
        "executed_blocks": plan.executed_blocks,
        "model": args.model, "model_revision": args.revision,
        "input": str(args.input), "n_samples": done,
        "compensators": {k: {
            "n_params": c.n_params, "rank": c.r,
            "flops_per_token": 4 * c.n_params,
            "predict_seconds_total": t_comp[k],
            "predict_ms_per_sample": 1000.0 * t_comp[k] / max(done, 1),
            "predict_ms_per_token": 1000.0 * t_comp[k] / max(done, 1) / 256.0,
        } for k, c in comps.items()},
        "block_reference": {"params_per_block": 21_700_000,
                            "note": "Qwen3-0.6B has ~0.6B params over 28 blocks"},
        "aggregate": agg,
        "git_commit": git_sha(),
        "warning": "Compensation cost is measured, not hidden. No speedup is "
                   "claimed: the base graph is still executed sequentially.",
    }
    (args.output / "eval_summary.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(agg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PINNED_MODEL_REPO)
    ap.add_argument("--revision", default=PINNED_MODEL_REVISION)
    ap.add_argument("--stage", choices=["collect", "fit", "eval"], required=True)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plan", default="tail-b27")
    ap.add_argument("--group-index", type=int, default=-1,
                    help="-1 = last parallel group")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--pairs", type=Path, help="pairs.npz for --stage fit")
    ap.add_argument("--compensator", action="append", default=[],
                    help="npz files for --stage eval (repeatable)")
    ap.add_argument("--rank-linear", type=int, default=8)
    ap.add_argument("--rank-mlp", type=int, default=16)
    ap.add_argument("--lam-rel", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    if args.revision is None:
        raise SystemExit("refusing to run with an unpinned model revision")

    if args.stage == "collect":
        if args.input is None:
            raise SystemExit("--input is required for collect")
        cmd_collect(args, mx, load)
    elif args.stage == "fit":
        if args.pairs is None:
            raise SystemExit("--pairs is required for fit")
        cmd_fit(args, mx, load)
    else:
        if args.input is None:
            raise SystemExit("--input is required for eval")
        cmd_eval(args, mx, load)


if __name__ == "__main__":
    main()
