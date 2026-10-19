"""Stage-1 measurement: serial vs shared-anchor parallel residual graphs.

Three error types are kept strictly separate and are never mixed:

  layer   : c_i = d_i^T - d_i^P(a),  a = teacher state at the group entry
  local   : G_P(h_s^T) - h_e^T       parallel group run on the *teacher* entry,
                                     so no error from earlier groups is included
  freerun : h_hat_e - h_e^T          candidate runs on its own states

At every group boundary the free-running error is split exactly into three
additive parts (see `run_plan`):

  h_hat_e - h_e^T = (c_s - t_s)                          inherited
                  + [G_P(c_s) - G_P(t_s) - (c_s - t_s)]   amplification
                  + [G_P(t_s) - h_e^T]                    newly created

Weights are frozen; nothing is trained. The Python loop is *not* evidence of
GPU concurrency, and no speedup is claimed.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

WINDOWS = {"early": (0, 8), "mid": (10, 18), "late": (20, 28)}


@dataclass
class Plan:
    name: str
    groups: list[list[int]]
    kind: str
    executed_blocks: int
    effective_depth: int
    deleted_layers: list[int] = field(default_factory=list)
    note: str = ""


def even_keep(n: int, k: int) -> list[int]:
    """Deterministic, reproducible layer subset of size k (integer rounding)."""
    if not 1 <= k <= n:
        raise ValueError(f"keep={k} outside 1..{n}")
    if k == 1:
        return [0]
    keep = [(i * (n - 1) + (k - 1) // 2) // (k - 1) for i in range(k)]
    if len(set(keep)) != k:
        raise ValueError(f"degenerate keep set for k={k}: {keep}")
    return keep


def make_plan(n: int, spec: str) -> Plan:
    if spec == "serial":
        return Plan("serial", [[i] for i in range(n)], "serial", n, n,
                    note="reference chain")

    m = re.fullmatch(r"group(\d+)", spec)
    if m:
        g = int(m.group(1))
        if not 1 <= g <= n:
            raise ValueError(f"group size {g} outside 1..{n}")
        groups = [list(range(i, min(n, i + g))) for i in range(0, n, g)]
        return Plan(spec, groups, "parallel", n, len(groups))

    m = re.fullmatch(r"window-(early|mid|late)-g(\d+)", spec)
    if m:
        wname, g = m.group(1), int(m.group(2))
        s, e = WINDOWS[wname]
        e = min(e, n)
        if s >= n:
            raise ValueError(f"window {wname} outside model")
        inner = [list(range(i, min(e, i + g))) for i in range(s, e, g)]
        groups = [[i] for i in range(s)] + inner + [[i] for i in range(e, n)]
        return Plan(spec, groups, "window", n, len(groups),
                    note=f"{wname} window [{s},{e}) with group size {g}")

    m = re.fullmatch(r"delete-keep(\d+)", spec)
    if m:
        k = int(m.group(1))
        keep = even_keep(n, k)
        deleted = [i for i in range(n) if i not in set(keep)]
        return Plan(spec, [[i] for i in keep], "delete", k, k, deleted,
                    note=f"keep {k}/{n} evenly spaced layers")

    raise ValueError(f"unknown plan: {spec}")


def default_plans(n: int) -> list[str]:
    base = ["serial", "group1", "group2", "group4", "group8", f"group{n}"]
    base += [f"delete-keep{k}" for k in (7, 14, 21) if k < n]
    base += [f"window-{w}-g{g}" for w in ("early", "mid", "late") for g in (2, 4, 8)]
    return base


# ---------------------------------------------------------------------------
# model primitives
# ---------------------------------------------------------------------------


def scalar(x, mx) -> float:
    mx.eval(x)
    return float(x.item())


def rms(x, mx) -> float:
    x = x.astype(mx.float32)
    return scalar(mx.sqrt(mx.mean(x * x)), mx)


def state_metrics(a, b, mx) -> dict:
    """Compare state a (candidate) against b (teacher)."""
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = a32 - b32
    rms_a = mx.sqrt(mx.mean(a32 * a32))
    rms_b = mx.sqrt(mx.mean(b32 * b32))
    rms_d = mx.sqrt(mx.mean(diff * diff))
    cos = mx.sum(a32 * b32) / (mx.sqrt(mx.sum(a32 * a32) * mx.sum(b32 * b32)) + 1e-12)
    max_abs = mx.max(mx.abs(diff))
    mx.eval(rms_a, rms_b, rms_d, cos, max_abs)
    rb = float(rms_b.item())
    rd = float(rms_d.item())
    return {
        "rms_candidate": float(rms_a.item()),
        "rms_teacher": rb,
        "rms_diff": rd,
        "rel_rms_diff": rd / max(rb, 1e-12),
        "cosine": float(cos.item()),
        "max_abs_diff": float(max_abs.item()),
    }


def logits_metrics(logits_teacher, logits_candidate, ids, mx) -> dict:
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
    mx.eval(nll_t, nll_s, kl, top1, tok_s)
    tok_np = np.asarray(tok_s, dtype=np.float64)
    return {
        "teacher_nll": float(nll_t.item()),
        "candidate_nll": float(nll_s.item()),
        "delta_nll": float(nll_s.item() - nll_t.item()),
        "teacher_to_candidate_kl": float(kl.item()),
        "top1_agreement": float(top1.item()),
        "token_nll_p50": float(np.percentile(tok_np, 50)),
        "token_nll_p95": float(np.percentile(tok_np, 95)),
        "n_predicted_tokens": int(tok_np.size),
    }


def embed(model, ids, mx):
    return model.model.embed_tokens(ids)


def project_logits(model, h, mx):
    inner = model.model
    out = inner.norm(h)
    if model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(out)
    return model.lm_head(out)


def teacher_pass(model, ids, mask, mx):
    h = embed(model, ids, mx)
    states, deltas = [h], []
    for block in model.model.layers:
        out = block(h, mask, None)
        mx.eval(out)
        states.append(out)
        deltas.append(out - h)
        h = out
    return states, deltas


def combine(anchor, deltas, accum, mx):
    if accum == "bf16":
        acc = anchor
        for d in deltas:
            acc = acc + d
        return acc
    acc = anchor.astype(mx.float32)
    for d in deltas:
        acc = acc + d.astype(mx.float32)
    return acc.astype(anchor.dtype)


def decompose_error(c_s, t_s, gp_cs, gp_ts, t_e):
    """Exact additive split of the free-running group error.

    c_s   candidate entry state, t_s teacher entry state
    gp_cs G_P(c_s) (what the rollout produced), gp_ts G_P(t_s) (local parallel)
    t_e   teacher exit state
    """
    inherited = c_s - t_s
    amplified = (gp_cs - gp_ts) - inherited
    newly = gp_ts - t_e
    total = gp_cs - t_e
    return inherited, amplified, newly, total


def group_forward(model, anchor, layers, mask, accum, mx):
    """Shared-anchor parallel group: a + sum_i (B_i(a) - a)."""
    if len(layers) == 1:
        out = model.model.layers[layers[0]](anchor, mask, None)
        mx.eval(out)
        return out, []
    deltas = [model.model.layers[i](anchor, mask, None) - anchor for i in layers]
    out = combine(anchor, deltas, accum, mx)
    mx.eval(out)
    return out, deltas


def run_plan(model, ids, mask, plan: Plan, teacher_states, accum, mx,
             keep_group_records: bool = True):
    """Free-running execution of a plan; records per-group entry/exit errors."""
    h = embed(model, ids, mx)
    records = []
    for gi, group in enumerate(plan.groups):
        s, e = group[0], group[-1] + 1
        t_s = teacher_states[s]
        t_e = teacher_states[e]
        entry = state_metrics(h, t_s, mx) if keep_group_records else None
        out, _ = group_forward(model, h, group, mask, accum, mx)
        exit_ = state_metrics(out, t_e, mx) if keep_group_records else None
        rec = {
            "group_index": gi, "start": s, "end": e, "size": len(group),
            "entry_rms_diff": entry["rms_diff"] if entry else None,
            "entry_rel_rms_diff": entry["rel_rms_diff"] if entry else None,
            "exit_rms_diff": exit_["rms_diff"] if exit_ else None,
            "exit_rel_rms_diff": exit_["rel_rms_diff"] if exit_ else None,
            "teacher_state_rms": exit_["rms_teacher"] if exit_ else None,
        }
        if keep_group_records:
            # local error: same group evaluated at the *teacher* entry state
            local_out, _ = group_forward(model, t_s, group, mask, accum, mx)
            local = state_metrics(local_out, t_e, mx)
            # exact additive split, computed in fp32 so the identity holds to rounding
            inherited_vec, ampl, new_err, total_vec = decompose_error(
                h.astype(mx.float32), t_s.astype(mx.float32),
                out.astype(mx.float32), local_out.astype(mx.float32),
                t_e.astype(mx.float32))
            residual = total_vec - (inherited_vec + ampl + new_err)
            rec.update({
                "local_rms_diff": local["rms_diff"],
                "local_rel_rms_diff": local["rel_rms_diff"],
                "local_cosine": local["cosine"],
                "decomp_inherited_rms": rms(inherited_vec, mx),
                "decomp_amplified_rms": rms(ampl, mx),
                "decomp_new_rms": rms(new_err, mx),
                "decomp_residual_rms": rms(residual, mx),
                "decomp_total_rms": rms(total_vec, mx),
            })
        records.append(rec)
        h = out
    return h, records


def diagnose_plan(model, plan: Plan, teacher_states, mask, accum, mx):
    """Layer-level corrections using each group's *teacher* entry as anchor."""
    rows, flats = [], []
    for group in plan.groups:
        if len(group) == 1:
            continue
        s = group[0]
        anchor = teacher_states[s]
        for i in group:
            d_t = teacher_states[i + 1] - teacher_states[i]
            d_p = model.model.layers[i](anchor, mask, None) - anchor
            mx.eval(d_p)
            t32, p32 = d_t.astype(mx.float32), d_p.astype(mx.float32)
            tn = mx.sqrt(mx.mean(t32 * t32))
            pn = mx.sqrt(mx.mean(p32 * p32))
            c = t32 - p32
            cn = mx.sqrt(mx.mean(c * c))
            cos = mx.sum(t32 * p32) / (mx.sqrt(mx.sum(t32 * t32) * mx.sum(p32 * p32)) + 1e-12)
            mx.eval(tn, pn, cn, cos)
            rows.append({
                "plan": plan.name, "group_start": s, "layer": i,
                "anchor": "teacher-group-entry",
                "teacher_delta_rms": float(tn.item()),
                "parallel_delta_rms": float(pn.item()),
                "correction_rms": float(cn.item()),
                "correction_relative_rms": float(cn.item()) / max(float(tn.item()), 1e-12),
                "delta_cosine": float(cos.item()),
            })
            flats.append(c.reshape(-1))
    gram = None
    if flats:
        c = mx.stack(flats, axis=0)
        gram = np.asarray(c @ mx.transpose(c), dtype=np.float64)
    return rows, gram


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-Base-bf16")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/stage1"))
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--plans", default=None,
                    help="comma separated; default = serial,group1,2,4,8,28,delete-keep*,windows")
    ap.add_argument("--accum", default="fp32,bf16",
                    help="residual accumulation dtype variants")
    ap.add_argument("--diagnose-plans", default="group2,group4,group8",
                    help="plans that get layer-level diagnostics ('' to disable)")
    ap.add_argument("--eq-rms", type=float, default=1e-2)
    ap.add_argument("--eq-top1", type=float, default=0.999)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(args.model, revision=args.revision, return_config=True)
    model.eval()
    n = len(model.model.layers)
    args.output.mkdir(parents=True, exist_ok=True)

    plan_names = ([p.strip() for p in args.plans.split(",") if p.strip()]
                  if args.plans else default_plans(n))
    plans = [make_plan(n, p) for p in plan_names]
    accums = [a.strip() for a in args.accum.split(",") if a.strip()]
    diag_names = {p.strip() for p in args.diagnose_plans.split(",") if p.strip()}

    manifest = {
        "status": "RUNNING",
        "model": args.model,
        "model_revision_requested": args.revision,
        "model_config": config,
        "n_layers": n,
        "mlx_version": getattr(mx, "__version__", "UNKNOWN"),
        "mlx_lm_version": getattr(mlx_lm, "__version__", "UNKNOWN"),
        "python": sys.version,
        "platform": platform.platform(),
        "input": str(args.input),
        "max_samples": args.max_samples,
        "plans": [{"name": p.name, "kind": p.kind, "executed_blocks": p.executed_blocks,
                   "effective_depth": p.effective_depth, "groups": p.groups,
                   "deleted_layers": p.deleted_layers, "note": p.note} for p in plans],
        "accum": accums,
        "diagnose_plans": sorted(diag_names),
        "git_commit": git_sha(),
        "warning": "Graph semantics only. No real GPU concurrency and no speedup claim.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    grams = {name: None for name in diag_names}
    n_done = 0
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as sf, \
         (args.output / "groups.jsonl").open("w", encoding="utf-8") as gf, \
         (args.output / "layers.jsonl").open("w", encoding="utf-8") as lf:

        for line in args.input.open(encoding="utf-8"):
            if n_done >= args.max_samples:
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
            serial_logits = project_logits(model, states[-1], mx)
            mx.eval(serial_logits)

            eq = logits_metrics(ref_logits, serial_logits, ids, mx)
            if eq["delta_nll"] != 0.0 or eq["top1_agreement"] < args.eq_top1:
                raise RuntimeError(f"S0 failed for {sample['sample_id']}: {eq}")
            eq_state = state_metrics(serial_logits, ref_logits, mx)
            if eq_state["rms_diff"] > args.eq_rms:
                raise RuntimeError(f"S0 logits mismatch for {sample['sample_id']}: {eq_state}")

            row = {
                "sample_id": sample["sample_id"],
                "n_tokens": int(ids.shape[1]),
                "tokens_sha256": sample.get("tokens_sha256"),
                "serial_equivalence": eq_state,
                "variants": {},
            }

            for accum in accums:
                for plan in plans:
                    out, grecs = run_plan(model, ids, mask, plan, states, accum, mx)
                    cand_logits = project_logits(model, out, mx)
                    mx.eval(cand_logits)
                    key = f"{plan.name}@{accum}"
                    row["variants"][key] = {
                        **logits_metrics(ref_logits, cand_logits, ids, mx),
                        "final_state": state_metrics(out, states[-1], mx),
                        "executed_blocks": plan.executed_blocks,
                        "effective_depth": plan.effective_depth,
                        "accum": accum,
                        "kind": plan.kind,
                    }
                    for r in grecs:
                        gf.write(json.dumps({"sample_id": sample["sample_id"],
                                             "plan": plan.name, "accum": accum, **r}) + "\n")

            for plan in plans:
                if plan.name not in diag_names or plan.kind != "parallel":
                    continue
                rows, gram = diagnose_plan(model, plan, states, mask, "fp32", mx)
                for r in rows:
                    lf.write(json.dumps({"sample_id": sample["sample_id"], **r}) + "\n")
                if gram is not None:
                    grams[plan.name] = gram if grams[plan.name] is None else grams[plan.name] + gram

            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            n_done += 1
            print(f"[{n_done}/{args.max_samples}] {sample['sample_id']} " + " ".join(
                f"{k}:dNLL={v['delta_nll']:.2f}" for k, v in row["variants"].items()
                if v["accum"] == accums[0]), flush=True)

    for name, gram in grams.items():
        if gram is None:
            continue
        # raw Gram only; spectra are derived by summarize.py so the definition
        # lives in exactly one place
        np.save(args.output / f"gram_{name}.npy", gram)

    manifest["status"] = "MEASURED" if n_done else "NO_SAMPLES"
    manifest["n_measured"] = n_done
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"saved {n_done} samples -> {args.output}")


if __name__ == "__main__":
    main()
