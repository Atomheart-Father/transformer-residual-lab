"""M1 -- submodule-level trace of a missing real upstream residual update.

One restricted mechanism verification on the frozen pinned Qwen3-0.6B-Base.
Nothing is trained, no weight is modified, no new model is used.

Research question
-----------------
When a Qwen3 block's input lacks a *real* upstream residual update Delta_j,
through which sub-module does the resulting discrepancy appear, and how is it
amplified or erased on the way to the logits?

Design (fixed before any measurement -- see results/mech_m1/PREREGISTRATION.md
and results/mech_m1/PREREGISTRATION_AMENDMENT.md)
-----------------------------------------------------------------------------
* 4 pre-registered edges, selected from the existing Stage 2.1 single-edge
  cost map only (results/stage2_1/search_single/search.json):
      P1 = (15,2)   global max cost      (0.06985 nats)
      P2 = (16,1)   global min cost      (0.00072 nats)
      C1 = (15,4)  cheapest edge with reader 15  (0.00472 nats)
      C2 = (16,15) dearest  edge with reader 16  (0.04119 nats)
* 7 interventions per edge, all applied to the *same* real teacher input x_i:
      real          delta = -Delta_j                     (persistent semantics)
      matched_radial same per-token norm, radial direction (sign-matched)
      matched_tan    same per-token norm, orthogonal to x_i
      matched_rand   same per-token norm, random direction
      small_radial   fixed eta (1%) relative to ||x_i||
      small_tan      fixed eta (1%), orthogonal to x_i
      small_rand     fixed eta (1%), random direction
* The perturbation is applied **per token over the whole sequence**.  The
  single-token variant is NOT run and the two settings are never mixed.
* Read-outs are taken at seven points inside the block:
      LN1 -> r_attn -> h = x + r_attn -> LN2 -> r_mlp -> out  (and Delta = out - x)

Two propagation semantics are measured and reported separately
--------------------------------------------------------------
S-PERSIST  (primary)  the removed update never comes back: the perturbed state
                        is carried through the untouched suffix blocks i+1..27.
                        This is the counterfactual "Delta_j never happened".
S-STAGE2   (contrast)  the semantics the Stage 2.1 executor actually used
                        (scripts/dag_exec.py, merge_mode="materialized"): the
                        reader's predecessor set loses Delta_j, but the *next*
                        reader's predecessor set still contains it, so Delta_j is
                        algebraically re-injected downstream.  Measured with the
                        repository's own executor, not re-implemented here.

Numerical discipline
---------------------
Gate 1  the readout path must be **bit-identical** to the installed
        TransformerBlock call, and x_{i+1} == out_i bitwise.
Gate 2  (a) the repository's own measure_batch must reproduce the recorded
            Stage 2.1 single-edge numbers on the 32-sample calibration split;
        (b) two independent implementations of the S-PERSIST intervention must
            agree bit-for-bit in delta-NLL;
        (c) under S-STAGE2 the four pre-registered edges must reproduce the
            Stage 2.1 ordering P1 > C2 > C1 > P2;
        (d) dag_forward and measure_batch must agree on every variant.
        Failure => stop and report, do not enlarge the sample.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PINNED_MODEL_REPO = "mlx-community/Qwen3-0.6B-Base-bf16"
PINNED_MODEL_REVISION = "e8e73ff38f0eb570eb57ed044ef3f95588c43bc8"

READOUTS = ("ln1", "r_attn", "h", "ln2", "r_mlp", "out", "delta_new")
CONDITIONS = ("real", "matched_radial", "matched_tan", "matched_rand",
              "small_radial", "small_tan", "small_rand")
# denominator guard: a read-out whose teacher norm is below this fraction of
# its own per-sample median is reported as n/a for the *relative* statistic.
REL_DENOM_FLOOR = 1e-3
# bf16 keeps 7 explicit mantissa bits -> one ULP is 2^-8 of the magnitude, and
# rounding the perturbed input to bf16 therefore injects an O(2^-8) relative
# perturbation that is independent of the intervention direction.  It is a
# floor under every relative read-out and is reported alongside them.
BF16_ULP_REL = 2.0 ** -8


# ---------------------------------------------------------------------------
# block readout (uses the installed submodules verbatim -- no re-implementation)
# ---------------------------------------------------------------------------


def to_np(x):
    """MLX array (possibly bf16) -> numpy float64. bf16 has no numpy dtype."""
    import mlx.core as _mx
    return np.asarray(x.astype(_mx.float32), dtype=np.float64)


def block_readout(block, x, mask):
    """Run one installed TransformerBlock and expose its internal tensors.

    Mirrors mlx_lm/models/qwen3.py:116-126 exactly, calling the *installed*
    submodules.  Gate 1 asserts bit-identity against ``block(x, mask, None)``.
    """
    u = block.input_layernorm(x)
    r_attn = block.self_attn(u, mask, None)
    h = x + r_attn
    w = block.post_attention_layernorm(h)
    r_mlp = block.mlp(w)
    out = h + r_mlp
    return {"ln1": u, "r_attn": r_attn, "h": h, "ln2": w,
            "r_mlp": r_mlp, "out": out, "delta_new": out - x}


def embed(model, ids, mx):
    return model.model.embed_tokens(ids)


def project_logits(model, h, mx):
    inner = model.model
    out = inner.norm(h)
    if model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(out)
    return model.lm_head(out)


def logits_metrics(logits_teacher, logits_candidate, ids, mx):
    """Same definition as scripts/measure.py:263 (reused verbatim)."""
    t = logits_teacher[:, :-1, :].astype(mx.float32)
    s = logits_candidate[:, :-1, :].astype(mx.float32)
    y = ids[:, 1:].astype(mx.int32)
    log_pt = t - mx.logsumexp(t, axis=-1, keepdims=True)
    log_ps = s - mx.logsumexp(s, axis=-1, keepdims=True)
    tok_t = -mx.take_along_axis(log_pt, y[..., None], axis=-1).reshape(-1)
    tok_s = -mx.take_along_axis(log_ps, y[..., None], axis=-1).reshape(-1)
    nll_t = mx.mean(tok_t)
    nll_s = mx.mean(tok_s)
    kl = mx.mean(mx.sum(mx.exp(log_pt) * (log_pt - log_ps), axis=-1))
    top1 = mx.mean((mx.argmax(t, axis=-1) == mx.argmax(s, axis=-1)).astype(mx.float32))
    mx.eval(nll_t, nll_s, kl, top1)
    return {"teacher_nll": float(nll_t.item()),
            "candidate_nll": float(nll_s.item()),
            "delta_nll": float(nll_s.item() - nll_t.item()),
            "teacher_to_candidate_kl": float(kl.item()),
            "top1_agreement": float(top1.item())}


# ---------------------------------------------------------------------------
# token-level ratio statistics
# ---------------------------------------------------------------------------


def ratio_stats(num, den, floor=1e-12):
    """Per-token ratio statistics, plus the rms-weighted ratio.

    ``num``/``den`` are (L,) arrays of per-token norms.  The rms-weighted
    ratio is rms(num) / rms(den) -- it is the view the tensor-wide RMS sees
    and can be dominated by a single high-norm token (the BOS position).
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    ok = den > floor
    r = num[ok] / den[ok]
    return {
        "mean": float(r.mean()) if r.size else None,
        "median": float(np.median(r)) if r.size else None,
        "p90": float(np.percentile(r, 90)) if r.size else None,
        "max": float(r.max()) if r.size else None,
        "rms_weighted": float(np.sqrt((num ** 2).mean()) /
                              max(np.sqrt((den ** 2).mean()), floor)),
        "n_excluded": int((~ok).sum()),
    }


# ---------------------------------------------------------------------------
# perturbations (per token, whole sequence)
# ---------------------------------------------------------------------------


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12), n


def make_perturbation(kind, x_np, real_np, eta_small, seed):
    """Build delta for one condition.

    x_np, real_np: (L, D) float64.  real_np is the true -Delta_j (per token).
    Returns (delta (L,D) float64, metadata dict).
    """
    rng = np.random.default_rng(seed)
    xh, xn = _unit(x_np)
    m = np.linalg.norm(real_np, axis=-1, keepdims=True)      # per-token norm
    radial_sign = np.sign(np.sum(real_np * xh, axis=-1, keepdims=True))
    radial_sign[radial_sign == 0] = 1.0

    # one shared random draw per token, reused by every random condition
    v = rng.standard_normal(x_np.shape)
    v_orth = v - np.sum(v * xh, axis=-1, keepdims=True) * xh
    v_orth, _ = _unit(v_orth)
    v_unit, _ = _unit(v)

    if kind == "real":
        delta = real_np
    elif kind == "matched_radial":
        delta = m * radial_sign * xh
    elif kind == "matched_tan":
        delta = m * v_orth
    elif kind == "matched_rand":
        delta = m * v_unit
    elif kind == "small_radial":
        delta = (eta_small * xn) * xh
    elif kind == "small_tan":
        delta = (eta_small * xn) * v_orth
    elif kind == "small_rand":
        delta = (eta_small * xn) * v_unit
    else:
        raise ValueError(kind)

    dn = np.linalg.norm(delta, axis=-1)
    # radial / tangential split of the perturbation itself
    rad = np.sum(delta * xh, axis=-1)
    tan2 = dn ** 2 - rad ** 2
    meta = {
        "abs_norm_mean": float(dn.mean()),
        "abs_norm_median": float(np.median(dn)),
        "rel_norm_mean": float(np.mean(dn / np.maximum(xn[:, 0], 1e-12))),
        "rel_norm_median": float(np.median(dn / np.maximum(xn[:, 0], 1e-12))),
        "rel_norm_rms_weighted": float(np.sqrt((dn ** 2).mean()) /
                                       max(np.sqrt((xn[:, 0] ** 2).mean()), 1e-12)),
        "radial_fraction_mean": float(np.mean(np.abs(rad) / np.maximum(dn, 1e-12))),
        "tangential_fraction_mean": float(np.mean(np.sqrt(np.maximum(tan2, 0)) / np.maximum(dn, 1e-12))),
        "cos_with_x_mean": float(np.mean(rad / np.maximum(dn, 1e-12))),
    }
    return delta, meta


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def readout_metrics(pert: dict, base: dict, delta_np, x_np, mx):
    """Per read-out: absolute change, relative change, normalised response gain."""
    out = {}
    dnorm = np.linalg.norm(delta_np, axis=-1)                 # (L,)
    rel_in = dnorm / np.maximum(np.linalg.norm(x_np, axis=-1), 1e-12)
    rel_in_mean = float(rel_in.mean())
    for key in READOUTS:
        p = to_np(pert[key][0])        # (L, D)
        b = to_np(base[key][0])
        d = p - b
        dn = np.linalg.norm(d, axis=-1)
        bn = np.linalg.norm(b, axis=-1)
        med = float(np.median(bn))
        ok = bn >= REL_DENOM_FLOOR * max(med, 1e-12)
        rel_vals = dn[ok] / bn[ok] if ok.any() else np.array([])
        cos = np.sum(p * b, axis=-1) / np.maximum(np.linalg.norm(p, axis=-1) * bn, 1e-12)
        out[key] = {
            "abs_change_mean": float(dn.mean()),
            "rel_change_mean": float(rel_vals.mean()) if rel_vals.size else None,
            "rel_change_median": float(np.median(rel_vals)) if rel_vals.size else None,
            "rel_change_rms_weighted": float(np.sqrt((dn ** 2).mean()) /
                                              max(np.sqrt((bn ** 2).mean()), 1e-12)),
            "rel_change_excluded_tokens": int((~ok).sum()),
            "rel_denominator_median": med,
            "response_gain_abs": float(np.mean(dn / np.maximum(dnorm, 1e-12))),
            "response_gain_rel": (float(np.mean(rel_vals) / rel_in_mean)
                                   if rel_vals.size and rel_in_mean > 0 else None),
            "cos_pert_base_mean": float(cos.mean()),
            "baseline_norm_mean": float(bn.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def build_dag(reader, contributor, n=28):
    from dag_exec import Dag
    preds = [frozenset(range(i)) for i in range(n)]
    preds[reader] = frozenset(range(reader)) - {contributor}
    return Dag(f"d{reader}_{contributor}", n, tuple(preds))


def replicate_stage2(model, mx, out_dir, n_samples=32):
    """Gate 2a: the repository's own measurement must reproduce search.json."""
    from stage21_search import (load_split, precompute_refs, measure_batch,
                                dag_from_drops)
    recorded = json.loads(Path("results/stage2_1/search_single/search.json")
                          .read_text(encoding="utf-8"))["variants"]
    samples = load_split("cal", n_samples)
    refs = precompute_refs(model, mx, samples)
    dags = {"serial": dag_from_drops("serial", []),
            "d15_2": dag_from_drops("d15_2", [(15, 2)]),
            "d16_1": dag_from_drops("d16_1", [(16, 1)])}
    rows = measure_batch(model, mx, refs, dags)
    res = {"n_samples": len(samples), "variants": {}}
    ok = True
    for name in dags:
        dn = np.array([r["variants"][name]["delta_nll"] for r in rows])
        got = float(dn.mean())
        rec = float(recorded[name]["delta_nll_mean"])
        res["variants"][name] = {"measured_mean": got, "recorded_mean": rec,
                                 "abs_diff": abs(got - rec),
                                 "per_sample": [float(x) for x in dn]}
        if abs(got - rec) > 1e-6:
            ok = False
    res["passed"] = ok
    (out_dir / "gate2_stage2_replication.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data257/s21_cal.jsonl")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--eta-small", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", default="results/mech_m1")
    ap.add_argument("--edge-selection", default="results/mech_m1/edge_selection.json")
    ap.add_argument("--replicate-stage2", action="store_true",
                    help="also run Gate 2a (32-sample Stage 2.1 replication)")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from dag_exec import dag_forward
    from stage21_search import (load_split, precompute_refs, measure_batch,
                                dag_from_drops)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- edge selection -------------------------------------------------
    sel_path = Path(args.edge_selection)
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    edges = sel["selected"]
    print("edges:", [(e["id"], e["reader"], e["contributor"]) for e in edges], flush=True)

    # ---- data -----------------------------------------------------------
    samples = []
    with open(args.split, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n_samples:
                break
    print(f"samples: {len(samples)} from {args.split}", flush=True)

    # ---- model ----------------------------------------------------------
    model, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)

    manifest = {
        "status": "RUNNING",
        "purpose": "M1 submodule-level trace of a missing real upstream residual update",
        "model": PINNED_MODEL_REPO, "model_revision": PINNED_MODEL_REVISION,
        "mlx_lm": __import__("mlx_lm").__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "git_sha": git_sha(),
        "split": args.split, "n_samples": len(samples),
        "sample_ids": [s["sample_id"] for s in samples],
        "edges": edges, "conditions": list(CONDITIONS),
        "readouts": list(READOUTS),
        "eta_small": args.eta_small, "seed": args.seed,
        "rel_denom_floor": REL_DENOM_FLOOR,
        "bf16_ulp_rel": BF16_ULP_REL,
        "perturbation_axis": "per-token, whole sequence",
        "semantics_primary": "S-PERSIST (removed update never returns)",
        "semantics_contrast": "S-STAGE2 (scripts/dag_exec.py materialized)",
        "n_layers": n_layers,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    local_f = (out_dir / "local_readouts.jsonl").open("w", encoding="utf-8")
    suffix_f = (out_dir / "suffix_propagation.jsonl").open("w", encoding="utf-8")
    stage2_f = (out_dir / "stage2_contrast.jsonl").open("w", encoding="utf-8")

    gate1 = {"checks": [], "passed": True}
    gate2 = {"b_persist_two_impls": {"rows": [], "max_abs_diff": 0.0, "passed": True},
             "c_stage2_ordering": {"rows": [], "passed": True},
             "d_dag_vs_measure_batch": {"rows": [], "max_abs_diff": 0.0, "passed": True}}
    layer_stats = []
    s2_delta_nll = {e["id"]: [] for e in edges}

    for si, sample in enumerate(samples):
        ids = mx.array(sample["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        mx.eval(h0, mask)

        # ---- teacher pass (records states, deltas, per-layer stats) ----
        # states[li]      = input of block li   (length n_layers)
        # states_after[k] = state after block k (length n_layers+1, states_after[n_layers] = final)
        states, deltas, ro_base = [], [], {}
        h = h0
        for li in range(n_layers):
            x_i = h
            ro = block_readout(layers[li], x_i, mask)
            out = ro["out"]
            mx.eval(out)
            states.append(x_i)
            deltas.append(out - x_i)
            if li in {e["reader"] for e in edges}:
                ro_base[li] = ro
            xn = float(np.linalg.norm(to_np(x_i[0]), axis=-1).mean())
            rn = float(np.linalg.norm(to_np(ro["r_attn"][0]), axis=-1).mean())
            mn = float(np.linalg.norm(to_np(ro["r_mlp"][0]), axis=-1).mean())
            dn = float(np.linalg.norm(to_np(ro["delta_new"][0]), axis=-1).mean())
            hn = float(np.linalg.norm(to_np(ro["h"][0]), axis=-1).mean())
            xh = to_np(x_i[0])
            ra = to_np(ro["r_attn"][0])
            cos_h_r = float(np.mean(np.sum(xh * ra, -1) /
                                    np.maximum(np.linalg.norm(xh, axis=-1) *
                                               np.linalg.norm(ra, axis=-1), 1e-12)))
            layer_stats.append({"layer": li, "rms_x": xn, "rms_h": hn, "rms_r_attn": rn,
                                "rms_r_mlp": mn, "rms_delta": dn,
                                "rel_update_r_attn": rn / max(hn, 1e-12),
                                "rel_update_r_mlp": mn / max(hn, 1e-12),
                                "rel_update_delta": dn / max(hn, 1e-12),
                                "cos_x_r_attn": cos_h_r})
            h = out
        final_state = h                      # out_27 -- the teacher's last state
        states_after = states + [final_state]   # states_after[k] = state after block k

        # Gate 1: readout path == installed block call, bitwise
        if si == 0:
            for li in (0, n_layers // 2, n_layers - 1):
                ref = layers[li](states[li], mask, None)
                mx.eval(ref)
                got = block_readout(layers[li], states[li], mask)["out"]
                mx.eval(got)
                md = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())
                gate1["checks"].append({"layer": li, "max_abs_diff": md})
                if md != 0.0:
                    gate1["passed"] = False
            md2 = 0.0
            for li in range(n_layers - 1):
                a = states[li + 1]
                b = layers[li](states[li], mask, None)
                mx.eval(a, b)
                md2 = max(md2, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
            gate1["checks"].append({"chain_x_next_eq_out": md2})
            if md2 != 0.0:
                gate1["passed"] = False
            print("gate1:", gate1, flush=True)

        # teacher logits from the FINAL state
        teacher_logits = project_logits(model, final_state, mx)
        mx.eval(teacher_logits)

        for e in edges:
            li, lj = e["reader"], e["contributor"]
            x_i = states[li]
            real_delta = to_np(deltas[lj][0])
            x_np = to_np(x_i[0])

            for cond in CONDITIONS:
                d_np, dmeta = make_perturbation(cond, x_np, -real_delta,
                                                args.eta_small, args.seed)
                # perturbed input is stored in bf16, exactly like the real
                # residual stream; the bf16 rounding of this sum is an
                # O(2^-8) floor that applies to every condition alike.
                pert = mx.array(d_np.astype(np.float32))[None, :, :]
                mx.eval(pert)
                x_p = (x_i.astype(mx.float32) + pert).astype(mx.bfloat16)
                mx.eval(x_p)

                ro_p = block_readout(layers[li], x_p, mask)
                mx.eval(ro_p["out"])
                lm = readout_metrics(ro_p, ro_base[li], d_np, x_np, mx)

                # ---- Gate 2b: second, independent implementation of the
                #      same S-PERSIST intervention (plain block calls only) --
                if cond == "real":
                    x_p2 = (x_i.astype(mx.float32) - deltas[lj]).astype(mx.bfloat16)
                    mx.eval(x_p2)
                    h2 = layers[li](x_p2, mask, None)
                    mx.eval(h2)
                    for k in range(li + 1, n_layers):
                        h2 = layers[k](h2, mask, None)
                        mx.eval(h2)
                    lg2 = project_logits(model, h2, mx)
                    mx.eval(lg2)
                    qm2 = logits_metrics(teacher_logits, lg2, ids, mx)

                # ---- suffix propagation (S-PERSIST) -----------------------
                # entry k = state AFTER block k, compared with the teacher's
                # state after block k (states_after[k+1]).
                h = ro_p["out"]
                suffix = []
                for k in range(li, n_layers):
                    tk = states_after[k + 1]
                    d = to_np(h[0]) - to_np(tk[0])
                    dn2 = np.linalg.norm(d, axis=-1)
                    tn = np.linalg.norm(to_np(tk[0]), axis=-1)
                    suffix.append({"layer": k,
                                   "rel_change": ratio_stats(dn2, tn),
                                   "abs_change_mean": float(dn2.mean()),
                                   "response_gain_abs": float(np.mean(dn2 / np.maximum(np.linalg.norm(d_np, axis=-1), 1e-12)))})
                    if k + 1 < n_layers:
                        h = layers[k + 1](h, mask, None)
                        mx.eval(h)
                cand_logits = project_logits(model, h, mx)
                mx.eval(cand_logits)
                qm = logits_metrics(teacher_logits, cand_logits, ids, mx)

                local_f.write(json.dumps({
                    "sample_id": sample["sample_id"], "edge": e["id"],
                    "reader": li, "contributor": lj, "condition": cond,
                    "perturbation": dmeta, "readouts": lm,
                    "final": qm,
                }, ensure_ascii=False) + "\n")
                suffix_f.write(json.dumps({
                    "sample_id": sample["sample_id"], "edge": e["id"],
                    "reader": li, "contributor": lj, "condition": cond,
                    "perturbation": dmeta, "suffix": suffix, "final": qm,
                }, ensure_ascii=False) + "\n")
                local_f.flush()
                suffix_f.flush()

                if cond == "real":
                    d1 = qm["delta_nll"]
                    d2v = qm2["delta_nll"]
                    gate2["b_persist_two_impls"]["rows"].append(
                        {"sample_id": sample["sample_id"], "edge": e["id"],
                         "readout_impl": d1, "plain_impl": d2v, "abs_diff": abs(d1 - d2v)})
                    gate2["b_persist_two_impls"]["max_abs_diff"] = max(
                        gate2["b_persist_two_impls"]["max_abs_diff"], abs(d1 - d2v))

            # ---- S-STAGE2 contrast: the repository's own executor ----------
            dag = build_dag(li, lj, n_layers)
            h_s2, recs = dag_forward(model, h0, dag, mask, mx, record=True,
                                     merge_mode="materialized")
            mx.eval(h_s2)
            lg_s2 = project_logits(model, h_s2, mx)
            mx.eval(lg_s2)
            q_s2 = logits_metrics(teacher_logits, lg_s2, ids, mx)
            s2_delta_nll[e["id"]].append(q_s2["delta_nll"])
            s2_traj = []
            for k in range(li, n_layers):
                # state AFTER block k in the DAG run: the input of block k+1
                # (for the last block: the final merged state).
                sk = recs["x"][k + 1] if k + 1 < n_layers else h_s2
                tk = states_after[k + 1]
                d = to_np(sk[0]) - to_np(tk[0])
                dn2 = np.linalg.norm(d, axis=-1)
                tn = np.linalg.norm(to_np(tk[0]), axis=-1)
                s2_traj.append({"layer": k, "rel_change": ratio_stats(dn2, tn),
                                "abs_change_mean": float(dn2.mean())})
            stage2_f.write(json.dumps({
                "sample_id": sample["sample_id"], "edge": e["id"],
                "reader": li, "contributor": lj, "final": q_s2,
                "trajectory": s2_traj,
            }, ensure_ascii=False) + "\n")
            stage2_f.flush()

        print(f"sample {si+1}/{len(samples)} done ({time.time()-t_start:.1f}s)", flush=True)

    local_f.close()
    suffix_f.close()
    stage2_f.close()

    # ---- Gate 2c/d: ordering under S-STAGE2 and cross-check ------------
    means = {k: float(np.mean(v)) for k, v in s2_delta_nll.items() if v}
    gate2["c_stage2_ordering"]["means"] = means
    gate2["c_stage2_ordering"]["recorded_stage21_means"] = {
        e["id"]: e["stage21_delta_nll"] for e in edges}
    order_ok = (means.get("P1", -1) > means.get("C2", -1) >
                means.get("C1", -1) > means.get("P2", -1))
    gate2["c_stage2_ordering"]["ordering_P1_gt_C2_gt_C1_gt_P2"] = bool(order_ok)
    gate2["c_stage2_ordering"]["passed"] = bool(order_ok)

    # independent cross-check: dag_forward vs the repository's measure_batch
    from stage21_search import load_split as _load_split, precompute_refs as _refs
    ref_list = _load_split("cal", args.n_samples)
    ref_objs = _refs(model, mx, ref_list)
    dags = {"serial": dag_from_drops("serial", [])}
    for e in edges:
        dags[e["id"]] = dag_from_drops(e["id"], [(e["reader"], e["contributor"])])
    rows = measure_batch(model, mx, ref_objs, dags)
    for name in dags:
        dn = np.array([r["variants"][name]["delta_nll"] for r in rows])
        gate2["d_dag_vs_measure_batch"]["rows"].append(
            {"variant": name, "measure_batch_mean": float(dn.mean())})
        if name in means:
            diff = abs(float(dn.mean()) - means[name])
            gate2["d_dag_vs_measure_batch"]["max_abs_diff"] = max(
                gate2["d_dag_vs_measure_batch"]["max_abs_diff"], diff)
    gate2["d_dag_vs_measure_batch"]["passed"] = bool(
        gate2["d_dag_vs_measure_batch"]["max_abs_diff"] < 1e-6)

    gate2["b_persist_two_impls"]["passed"] = bool(
        gate2["b_persist_two_impls"]["max_abs_diff"] < 1e-6)

    gate2["passed"] = bool(gate2["b_persist_two_impls"]["passed"] and
                           gate2["c_stage2_ordering"]["passed"] and
                           gate2["d_dag_vs_measure_batch"]["passed"])

    # ---- Gate 2a: optional Stage 2.1 replication ------------------------
    if args.replicate_stage2:
        rep = replicate_stage2(model, mx, out_dir)
        gate2["a_stage2_replication"] = {"passed": rep["passed"],
                                         "measured": {k: v["measured_mean"]
                                                      for k, v in rep["variants"].items()},
                                         "recorded": {k: v["recorded_mean"]
                                                      for k, v in rep["variants"].items()}}
        gate2["passed"] = bool(gate2["passed"] and rep["passed"])

    print("gate2:", json.dumps(gate2, indent=2), flush=True)

    (out_dir / "gate1_equivalence.json").write_text(json.dumps(gate1, indent=2), encoding="utf-8")
    (out_dir / "gate2_semantics.json").write_text(json.dumps(gate2, indent=2), encoding="utf-8")
    (out_dir / "layer_stats.json").write_text(json.dumps(layer_stats, indent=2), encoding="utf-8")

    manifest["status"] = "DONE" if (gate1["passed"] and gate2["passed"]) else "GATE_FAILED"
    manifest["gate1_passed"] = gate1["passed"]
    manifest["gate2_passed"] = gate2["passed"]
    manifest["elapsed_s"] = round(time.time() - t_start, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest status:", manifest["status"], flush=True)


if __name__ == "__main__":
    main()
