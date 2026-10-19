"""Stage 1.7: multi-site residual compensation, shared vs serial.

Stage 1.6 fitted one compensator at one group exit. This script does two
things that the single-site experiment could not:

  1. compensate *several* group exits at once (default: every parallel group
     of `irr-a`, i.e. groups 3..6), so the corrections can interact;
  2. carry a low-dimensional state across those sites, so the compensator
     accumulates information the way the original residual stream does.

The control is exact. `--no-carry` zeroes the state at every site, which turns
the same network into a plain per-site function of (entry, site). Parameters,
FLOPs, inputs, training data and sites are then *identical* to the carried
version; the only difference is whether s survives from one site to the next.
That isolates recurrence, not capacity.

Inputs are always the free-running group entry, which is what deployment has.
Teacher states only build the label and the oracle upper bound.

Stages:
    collect -> dump A[n,S,T,D] and C[n,S,T,D] for the chosen sites
    fit     -> train one compensator (shared or per-site, carried or not)
    eval    -> free-run the plan with every site compensated, measure quality
    bench   -> per-site latency, whole-chain latency, peak memory
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
    project_logits,
    resolve_plan,
    state_metrics,
    teacher_pass,
)

SITE_EMB = 16


# ---------------------------------------------------------------------------
# graph helpers (same semantics as scripts/compensate.py)
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


def parse_sites(spec: str, n_groups: int | None = None) -> list[int]:
    """Group indices to compensate, e.g. "3,4,5,6"."""
    if spec in ("", "auto"):
        raise SystemExit("--sites must list group indices explicitly")
    sites = [int(x) for x in spec.split(",") if x.strip()]
    if not sites:
        raise SystemExit("empty --sites")
    if len(set(sites)) != len(sites):
        raise SystemExit(f"duplicate site in {sites}")
    if n_groups is not None and (min(sites) < 0 or max(sites) >= n_groups):
        raise SystemExit(f"site out of range for {n_groups} groups: {sites}")
    return sorted(sites)


# ---------------------------------------------------------------------------
# the compensator
# ---------------------------------------------------------------------------


def init_params(mx, rng, dim, n_sites, hidden, rank, per_site):
    """Gated residual cell + output head, all float32.

    Shared:  Wz [h, D+E]        Per-site: Wz [S, h, D+E]
    The gate keeps the state bounded without a full GRU; the head reads both
    the new state and the current entry, so a site can answer from `a` alone
    when the state carries nothing useful.
    """
    lead = (n_sites,) if per_site else ()
    din = dim + SITE_EMB

    def w(*shape):
        return mx.array(rng.normal(0.0, 0.02, shape).astype(np.float32))

    p = {
        "emb": w(n_sites, SITE_EMB),
        "Wz": w(*lead, hidden, din), "Uz": w(*lead, hidden, hidden),
        "bz": mx.zeros(tuple(lead) + (hidden,)),
        "Wn": w(*lead, hidden, din), "Un": w(*lead, hidden, hidden),
        "bn": mx.zeros(tuple(lead) + (hidden,)),
        "W1": w(*lead, rank, hidden + din), "b1": mx.zeros(tuple(lead) + (rank,)),
        "W2": w(*lead, dim, rank), "b2": mx.zeros(tuple(lead) + (dim,)),
    }
    return p


def count_params(p) -> int:
    return int(sum(np.asarray(v).size for v in p.values()))


def step(p, si, a, s, carry, per_site, mx, nn):
    """One site. a [B,T,D], s [B,T,h] -> (c_hat [B,T,D], s_next [B,T,h])."""
    idx = (si,) if per_site else ()
    sel = (lambda k: p[k][idx]) if per_site else (lambda k: p[k])

    b, t, d = a.shape
    e = mx.broadcast_to(p["emb"][si][None, None, :], (b, t, SITE_EMB))
    x = mx.concatenate([a.astype(mx.float32), e], axis=-1)

    s_in = s if carry else mx.zeros_like(s)
    z = mx.sigmoid(x @ sel("Wz").T + s_in @ sel("Uz").T + sel("bz"))
    n = mx.tanh(x @ sel("Wn").T + s_in @ sel("Un").T + sel("bn"))
    s_out = (1.0 - z) * s_in + z * n

    u = mx.concatenate([s_out, x], axis=-1)
    c = nn.gelu(u @ sel("W1").T + sel("b1")) @ sel("W2").T + sel("b2")
    return c, s_out


class SerialComp:
    """Numpy wrapper used at eval time; mirrors `step` exactly."""

    def __init__(self, params, meta):
        self.p = params
        self.meta = meta
        self.hidden = int(params["bz"].shape[-1])
        self.rank = int(params["b1"].shape[-1])
        self.carry = bool(meta["carry_state"])
        self.per_site = bool(meta["per_site"])
        self.n_params = count_params(params)
        self._p_mx = None

    def save(self, path):
        np.savez(path, **self.p,
                 meta=json.dumps(self.meta), kind="serial")

    def step_mx(self, si, a, s, mx):
        import mlx.nn as nn
        if self._p_mx is None:
            self._p_mx = {k: mx.array(np.asarray(v)) for k, v in self.p.items()}
        return step(self._p_mx, si, a, s, self.carry, self.per_site, mx, nn)

    def zero_state(self, b, t, mx):
        return mx.zeros((b, t, self.hidden), dtype=mx.float32)


def load_serial(path: Path) -> SerialComp:
    # np.load(path) trips the sandboxed open shim on this machine; hand it a
    # real file handle instead.
    with open(str(path), "rb") as fh:
        z = dict(np.load(fh))
    meta = json.loads(str(z.pop("meta")))
    z.pop("kind", None)
    return SerialComp(z, meta)


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def cmd_collect(args, mx, load):
    from mlx_lm.models.base import create_attention_mask
    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    n = len(model.model.layers)
    plan = resolve_plan(n, args.plan)
    sites = parse_sites(args.sites, len(plan.groups))
    pos_of = {gi: k for k, gi in enumerate(sites)}

    comp = load_serial(Path(args.compensator[0])) if args.compensator else None

    args.output.mkdir(parents=True, exist_ok=True)
    A, C, ids_out, rms_rows = [], [], [], []
    done = 0
    for line in args.input.open(encoding="utf-8"):
        if done >= args.max_samples:
            break
        sample = json.loads(line)
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states, _ = teacher_pass(model, ids, mask, mx)

        h = h0
        s = comp.zero_state(1, ids.shape[1], mx) if comp else None
        a_site, c_site, rms = [], [], []
        for gi, group in enumerate(plan.groups):
            a = h
            par = run_group(model, h, group, mask, mx)
            if gi in pos_of:
                truth = states[group[-1] + 1]
                c_star = truth.astype(mx.float32) - par.astype(mx.float32)
                mx.eval(a, c_star)
                a_site.append(np.asarray(a.astype(mx.float32)).reshape(-1, a.shape[-1]))
                c_site.append(np.asarray(c_star).reshape(-1, c_star.shape[-1]))
                rms.append(float(mx.sqrt(mx.mean(c_star * c_star)).item()))
                if comp is not None:
                    c_hat, s = comp.step_mx(pos_of[gi], a, s, mx)
                    mx.eval(c_hat, s)
                    par = (par.astype(mx.float32) + c_hat.astype(mx.float32)).astype(par.dtype)
                    mx.eval(par)
            h = par

        A.append(np.stack(a_site, axis=0))
        C.append(np.stack(c_site, axis=0))
        ids_out.append(sample["sample_id"])
        rms_rows.append(rms)
        done += 1
        print(f"[collect {done}/{args.max_samples}] {sample['sample_id']} "
              + " ".join(f"g{gi}={r:.2f}" for gi, r in zip(sites, rms)), flush=True)

    A = np.stack(A, axis=0).astype(np.float32)     # [n, S, T, D]
    C = np.stack(C, axis=0).astype(np.float32)
    # raw .npy, not .npz: fit mmaps them so a 64-window, 4-site tensor does not
    # have to sit in RSS all at once.
    np.save(args.output / "A.npy", A)
    np.save(args.output / "C.npy", C)
    manifest = {
        "stage": "collect", "plan": plan.name, "groups": plan.groups,
        "sites": sites, "site_layers": [plan.groups[i] for i in sites],
        "effective_depth": plan.effective_depth,
        "executed_blocks": plan.executed_blocks,
        "model": args.model, "model_revision": args.revision,
        "input": str(args.input), "n_samples": done,
        "shape": list(A.shape),
        "sample_ids": ids_out,
        "files": {"A": "A.npy", "C": "C.npy"},
        "compensated_during_collection": bool(comp),
        "compensator": args.compensator[0] if args.compensator else None,
        "c_rms_per_site": {f"g{gi}": float(np.mean([r[k] for r in rms_rows]))
                           for k, gi in enumerate(sites)},
        "input_is": "free-running candidate group entry (deployable)",
        "label_is": "teacher group exit minus candidate parallel group output",
        "git_commit": git_sha(),
    }
    (args.output / "collect_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"saved {A.shape} -> {args.output}")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def cmd_fit(args, mx, load):
    import mlx.nn as nn
    import mlx.optimizers as optim

    root = Path(args.pairs)
    if root.is_dir():
        A_all = np.load(root / "A.npy", mmap_mode="r")
        C_all = np.load(root / "C.npy", mmap_mode="r")
    else:                                     # legacy single .npz
        with open(str(root), "rb") as fh:
            z = dict(np.load(fh))
        A_all, C_all = z["A"], z["C"]
    n, S, T, D = A_all.shape
    want = parse_sites(args.sites)
    if S != len(want):
        raise SystemExit(f"pairs have {S} sites, --sites has {len(want)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    va, tr = perm[:n_val], perm[n_val:]

    mx.random.seed(args.seed)
    params = init_params(mx, np.random.default_rng(args.seed + 1), D, S,
                         args.hidden, args.rank, args.per_site)
    n_params = count_params(params)

    # Plain MSE weights each site by the magnitude of its correction, and those
    # magnitudes differ by ~17x across groups. The largest site then dominates
    # the gradient and the sites that matter for NLL get starved. The
    # `normalized` option divides each site by its own variance first, so every
    # site contributes equally; it is reported alongside the plain baseline,
    # never instead of it.
    scale = np.mean(np.asarray(C_all[tr]) ** 2, axis=(0, 2, 3))
    if args.site_weight == "normalized":
        w = 1.0 / np.maximum(scale, 1e-9)
        w = w * (len(w) / w.sum())
    else:
        w = np.ones(len(scale))
    w_mx = mx.array(w.astype(np.float32))

    def loss_fn(p, a, c):
        b, s_, t, d = a.shape
        s = mx.zeros((b, t, args.hidden), dtype=mx.float32)
        total = 0.0
        for si in range(s_):
            c_hat, s = step(p, si, a[:, si], s, args.carry_state,
                            args.per_site, mx, nn)
            total = total + w_mx[si] * mx.mean((c_hat - c[:, si]) ** 2)
        return total / s_

    grad_fn = mx.value_and_grad(loss_fn)
    opt = optim.Adam(learning_rate=args.lr)
    hist = []
    bs = args.batch_samples
    for ep in range(args.epochs):
        order = np.random.default_rng(args.seed + 1000 + ep).permutation(tr)
        tot, cnt = 0.0, 0
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]
            xb = mx.array(np.asarray(A_all[idx]))
            yb = mx.array(np.asarray(C_all[idx]))
            loss, grads = grad_fn(params, xb, yb)
            params = opt.apply_gradients(grads, params)
            mx.eval(params, loss)
            tot += float(loss.item()) * len(idx)
            cnt += len(idx)
        rec = {"epoch": ep, "train_mse": tot / cnt}
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            xv = mx.array(np.asarray(A_all[va]))
            yv = mx.array(np.asarray(C_all[va]))
            v = float(loss_fn(params, xv, yv).item())
            rec["val_mse"] = v
            print(f"  epoch {ep:3d} train={tot / cnt:.5f} val={v:.5f}", flush=True)
        else:
            print(f"  epoch {ep:3d} train={tot / cnt:.5f}", flush=True)
        hist.append(rec)

    # per-site diagnostics on the held-out windows
    comp = SerialComp({k: np.asarray(v) for k, v in params.items()}, {
        "carry_state": bool(args.carry_state), "per_site": bool(args.per_site),
        "hidden": args.hidden, "rank": args.rank, "sites": args.sites,
        "plan": args.plan, "pairs": str(args.pairs),
        "n_train_windows": int(len(tr)), "n_val_windows": int(len(va)),
    })
    per_site = {}
    for si in range(S):
        s = np.zeros((len(va), T, args.hidden), dtype=np.float32)
        preds = []
        for si2 in range(S):
            a2 = mx.array(np.asarray(A_all[va][:, si2]))
            sm = mx.array(s)
            c_hat, s2 = comp.step_mx(si2, a2, sm, mx)
            mx.eval(c_hat, s2)
            s = np.asarray(s2)
            preds.append(np.asarray(c_hat))
            if si2 == si:
                break
        P, Y = preds[-1], np.asarray(C_all[va][:, si])
        err = P - Y
        t_rms = float(np.sqrt((Y ** 2).mean()))
        cos = float((P * Y).sum() /
                    (np.sqrt((P ** 2).sum() * (Y ** 2).sum()) + 1e-12))
        per_site[f"site{si}"] = {
            "target_rms": t_rms,
            "pred_rms": float(np.sqrt((P ** 2).mean())),
            "rel_err": float(np.sqrt((err ** 2).mean()) / max(t_rms, 1e-12)),
            "cosine": cos}

    comp.save(args.output / f"serial_{args.tag}.npz")
    report = {
        "pairs": str(args.pairs), "tag": args.tag,
        "n_params": n_params, "hidden": args.hidden, "rank": args.rank,
        "carry_state": bool(args.carry_state), "per_site": bool(args.per_site),
        "sites": S, "n_train_windows": int(len(tr)),
        "n_val_windows": int(len(va)),
        "site_weight": args.site_weight,
        "site_weights": [float(x) for x in w],
        "site_c_rms": [float(np.sqrt(x)) for x in scale],
        "final_train_mse": hist[-1]["train_mse"],
        "final_val_mse": hist[-1].get("val_mse"),
        "per_site_val": per_site,
        "history": hist,
    }
    (args.output / f"fit_{args.tag}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     indent=1, default=str))


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
    sites = parse_sites(args.sites, len(plan.groups))
    pos_of = {gi: k for k, gi in enumerate(sites)}

    apply_sites = set(parse_sites(args.apply_sites)) if args.apply_sites else set(sites)
    if not apply_sites <= set(sites):
        raise SystemExit(f"--apply-sites must be a subset of --sites ({sites})")

    comps = {}
    for path in args.compensator:
        p = Path(path)
        comps[p.stem] = load_serial(p)

    variants = ["serial", plan.name, f"{plan.name}+oracle",
                f"{plan.name}+oracle_ex_last"]
    variants += [f"{plan.name}+{k}" for k in comps]

    rows, done = [], 0
    t_pred = {k: 0.0 for k in comps}
    for line in args.input.open(encoding="utf-8"):
        if done >= args.max_samples:
            break
        sample = json.loads(line)
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states, _ = teacher_pass(model, ids, mask, mx)
        ref = model(ids)
        mx.eval(ref)
        row = {"sample_id": sample["sample_id"], "variants": {},
               "sites": {}}

        row["variants"]["serial"] = logits_metrics(
            ref, project_logits(model, states[-1], mx), ids, mx)

        def run(mode, comp=None, sink=None, oracle_sites=(), trace=None):
            h = h0
            s = comp.zero_state(1, ids.shape[1], mx) if comp else None
            for gi, group in enumerate(plan.groups):
                a = h
                if gi in oracle_sites:
                    h = states[group[-1] + 1]
                    continue
                par = run_group(model, h, group, mask, mx)
                if trace is not None:
                    trace.append(round(float(
                        state_metrics(par, states[group[-1] + 1], mx)["rel_rms_diff"]), 5))
                if comp is not None and gi in pos_of and gi in apply_sites:
                    t0 = time.perf_counter()
                    c_hat, s = comp.step_mx(pos_of[gi], a, s, mx)
                    mx.eval(c_hat, s)
                    sink["predict_seconds"] = sink.get("predict_seconds", 0.0) + \
                        (time.perf_counter() - t0)
                    par = (par.astype(mx.float32)
                           + c_hat.astype(mx.float32)).astype(par.dtype)
                    mx.eval(par)
                h = par
            return h

        tr_base = []
        base = run("base", trace=tr_base)
        mx.eval(base)
        row["variants"][plan.name] = {
            **logits_metrics(ref, project_logits(model, base, mx), ids, mx),
            "final_state": state_metrics(base, states[-1], mx),
            "exit_rel_rms": tr_base}

        # perfect compensation at every chosen site. Because the last site is
        # the final group, this reproduces the serial model exactly; the number
        # is kept as a sanity check, not as an interesting bound.
        orc = run("oracle", oracle_sites=set(sites))
        mx.eval(orc)
        row["variants"][f"{plan.name}+oracle"] = {
            **logits_metrics(ref, project_logits(model, orc, mx), ids, mx),
            "final_state": state_metrics(orc, states[-1], mx)}

        # the same, but the last group still has to be predicted. This is the
        # bound that actually constrains a compensator: it cannot hand itself
        # the final hidden state.
        orc_x = run("oracle", oracle_sites=set(sites[:-1]))
        mx.eval(orc_x)
        row["variants"][f"{plan.name}+oracle_ex_last"] = {
            **logits_metrics(ref, project_logits(model, orc_x, mx), ids, mx),
            "final_state": state_metrics(orc_x, states[-1], mx)}

        for name, comp in comps.items():
            sink = {}
            tr_c = []
            out = run("comp", comp, sink, trace=tr_c)
            mx.eval(out)
            t_pred[name] += sink.get("predict_seconds", 0.0)
            row["variants"][f"{plan.name}+{name}"] = {
                **logits_metrics(ref, project_logits(model, out, mx), ids, mx),
                "final_state": state_metrics(out, states[-1], mx),
                "predict_seconds": sink.get("predict_seconds", 0.0),
                "exit_rel_rms": tr_c}
        rows.append(row)
        done += 1
        print(f"[eval {done}/{args.max_samples}] {sample['sample_id']} " + " ".join(
            f"{k.split('+')[-1]}={v['delta_nll']:.3f}"
            for k, v in row["variants"].items()), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
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
        traces = [r["variants"][v].get("exit_rel_rms") for r in rows
                  if v in r["variants"]]
        traces = [t for t in traces if t]
        if traces:
            agg[v]["exit_rel_rms_mean"] = list(np.mean(np.array(traces), axis=0))
    base_nll = agg[plan.name]["delta_nll_mean"]
    for v, a in agg.items():
        # gap closure: the share of the parallel gap that this variant removes.
        # Well defined for every configuration, so it is safe to compare across
        # families; `recovery_vs_oracle` is not, because each family has a
        # different oracle.
        a["gap_closure"] = (base_nll - a["delta_nll_mean"]) / base_nll
        for okey in (f"{plan.name}+oracle", f"{plan.name}+oracle_ex_last"):
            o = agg[okey]["delta_nll_mean"]
            if base_nll - o > 1e-9:
                a[f"recovery_vs_{okey.split('+')[-1]}"] = \
                    (base_nll - a["delta_nll_mean"]) / (base_nll - o)

    manifest = {
        "stage": "eval", "plan": plan.name, "groups": plan.groups,
        "sites": sites, "site_layers": [plan.groups[i] for i in sites],
        "effective_depth": plan.effective_depth,
        "executed_blocks": plan.executed_blocks,
        "model": args.model, "model_revision": args.revision,
        "input": str(args.input), "n_samples": done,
        "compensators": {k: {
            "n_params": c.n_params, "hidden": c.hidden, "rank": c.rank,
            "carry_state": c.carry, "per_site": c.per_site,
            "flops_per_token": 4 * c.n_params,
            "predict_ms_per_sample": 1000.0 * t_pred[k] / max(done, 1),
        } for k, c in comps.items()},
        "aggregate": agg,
        "git_commit": git_sha(),
        "warning": "Measured cost only. The graph is still executed "
                   "sequentially; no speedup is claimed.",
    }
    (args.output / "eval_summary.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(agg, indent=2))


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def cmd_bench(args, mx, load):
    """Per-site and whole-chain latency, plus peak memory.

    Medians, not means: the first MLX call on a shape compiles and allocates,
    and including it would report the warm-up as if it were steady state.
    """
    comp = load_serial(Path(args.compensator[0]))
    t, d = 256, 1024
    a = mx.random.normal((1, t, d)).astype(mx.float32)
    mx.eval(a)
    n_sites = len(parse_sites(comp.meta["sites"]))

    def one_site(si):
        s0 = comp.zero_state(1, t, mx)
        mx.eval(s0)
        c, s1 = comp.step_mx(si, a, s0, mx)
        mx.eval(c, s1)

    def chain():
        s0 = comp.zero_state(1, t, mx)
        mx.eval(s0)
        for si in range(n_sites):
            c, s0 = comp.step_mx(si, a, s0, mx)
            mx.eval(c, s0)

    for _ in range(args.warmup):
        for si in range(n_sites):
            one_site(si)
        chain()

    mx.reset_peak_memory()
    per_site = []
    for si in range(n_sites):
        ts = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            one_site(si)
            ts.append(time.perf_counter() - t0)
        per_site.append(float(np.median(ts)))
    peak = mx.get_peak_memory()

    ts = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        chain()
        ts.append(time.perf_counter() - t0)
    chain_ms = float(np.median(ts)) * 1000.0

    out = {
        "compensator": args.compensator[0],
        "n_params": comp.n_params, "hidden": comp.hidden, "rank": comp.rank,
        "carry_state": comp.carry, "per_site": comp.per_site,
        "n_sites": n_sites,
        "tokens": t, "repeat": args.repeat, "warmup": args.warmup,
        "per_site_ms": [1000.0 * x for x in per_site],
        "per_site_total_ms": 1000.0 * sum(per_site),
        "chain_ms": chain_ms,
        "flops_per_token": 4 * comp.n_params,
        "peak_memory_mb": peak / 2 ** 20,
        "note": "one call per site, 256 tokens per call; medians over repeats",
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PINNED_MODEL_REPO)
    ap.add_argument("--revision", default=PINNED_MODEL_REVISION)
    ap.add_argument("--stage",
                    choices=["collect", "fit", "eval", "bench"], required=True)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plan", default="irr-a:0|1|2|3-10|11-19|20-25|26-27")
    ap.add_argument("--sites", default="3,4,5,6")
    ap.add_argument("--apply-sites", default=None,
                    help="subset of --sites that actually receives a "
                         "correction; used to price each extra site")
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--compensator", action="append", default=[],
                    help="serial npz, repeatable. collect uses the first one; "
                         "eval measures all of them; bench uses the first.")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--carry-state", action="store_true",
                    help="carry s across sites; without it s is zeroed at "
                         "every site and the model is a per-site function")
    ap.add_argument("--per-site", action="store_true",
                    help="separate parameters per site instead of sharing")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.125)
    ap.add_argument("--site-weight", choices=["mse", "normalized"],
                    default="mse")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
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
    elif args.stage == "eval":
        if args.input is None:
            raise SystemExit("--input is required for eval")
        cmd_eval(args, mx, load)
    else:
        cmd_bench(args, mx, load)


if __name__ == "__main__":
    main()
