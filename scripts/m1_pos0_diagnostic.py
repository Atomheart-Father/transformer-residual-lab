"""POST-HOC DIAGNOSTIC (not part of the pre-registered M1 experiment).

Motivation: the pre-registered data show an anomaly that the pre-registered
design cannot explain.

  * at reader 15, a random-direction perturbation whose per-token magnitude
    profile is ||Delta_2(t)|| gives |delta r_attn| / ||r_attn|| = 1.01,
    while the *same direction* with a uniform relative magnitude of 0.112
    gives only 0.41;
  * the same reader with the ||Delta_4(t)|| profile gives only 0.25,
    although ||Delta_4|| is larger than ||Delta_2|| at almost every token.

Hypothesis (H0): the residual stream of this model carries a *massive
activation* at position 0 -- ||x_15[0]|| is ~130x the median token norm --
and Delta_2 contributes essentially all of it.  Removing Delta_2 therefore
destroys the position-0 key/value, and since every position attends to
position 0, the attention distribution of the whole sequence changes.

This script tests H0 with the smallest possible additional measurement:
  (a) per-layer position-0 vs median-token norms, and the channel
      decomposition of the position-0 state;
  (b) an attribution test -- the real perturbation applied ONLY at position 0,
      vs applied at ALL positions EXCEPT position 0.

Same 8 calibration windows, same 4 pre-registered edges, same frozen weights.
This diagnostic does not select anything; it only attributes an effect that
was already observed in the pre-registered data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

from measure import embed, project_logits, PINNED_MODEL_REPO, PINNED_MODEL_REVISION
from m1_submodule_trace import block_readout, to_np, logits_metrics

N_LAYERS = 28
READOUTS = ("ln1", "r_attn", "h", "ln2", "r_mlp", "out", "delta_new")
EDGE_ORDER = ("P1", "C2", "C1", "P2")


def main():
    out_dir = Path("results/mech_m1")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    layers = model.model.layers

    samples = [json.loads(l) for l in
               Path("data257/s21_cal.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()][:8]
    sel = json.loads((out_dir / "edge_selection.json").read_text(encoding="utf-8"))
    edges = sel["selected"]

    layer_rows = []
    att_rows = []
    chan_rows = []

    for si, s in enumerate(samples):
        ids = mx.array(s["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        mx.eval(h0, mask)

        # NOTE: xs[li] must be the *input* of block li.  Do not pre-seed the
        # list with h0 and also append h inside the loop -- that shifts the
        # whole list by one (the bug this script originally had).
        xs, ds = [], []
        h = h0
        for li in range(N_LAYERS):
            ro = block_readout(layers[li], h, mask)
            mx.eval(ro["out"])
            ds.append(ro["out"] - h); xs.append(h); h = ro["out"]
        final_state = h
        states_after = xs + [final_state]

        # consistency audit: ||Delta_l|| <= ||x_{l+1}|| + ||x_l|| must hold
        for li in range(N_LAYERS - 1):
            a = np.linalg.norm(to_np(ds[li][0]), axis=-1)
            b = np.linalg.norm(to_np(states_after[li + 1][0]), axis=-1)
            c = np.linalg.norm(to_np(xs[li][0]), axis=-1)
            assert not (a > (b + c) * 1.001).any(), f"triangle violation at layer {li}"
        teacher_logits = project_logits(model, final_state, mx)
        mx.eval(teacher_logits)

        # (a) per-layer position-0 vs median norms
        for li in range(N_LAYERS):
            x = to_np(xs[li][0]); d = to_np(ds[li][0])
            xn = np.linalg.norm(x, axis=-1); dn = np.linalg.norm(d, axis=-1)
            layer_rows.append({
                "sample": si, "layer": li,
                "x_norm_pos0": float(xn[0]), "x_norm_median": float(np.median(xn)),
                "x_norm_max": float(xn.max()),
                "x_pos0_over_median": float(xn[0] / max(np.median(xn), 1e-12)),
                "x_argmax_token": int(np.argmax(xn)),
                "d_norm_pos0": float(dn[0]), "d_norm_median": float(np.median(dn)),
                "d_norm_max": float(dn.max()),
                "d_pos0_over_median": float(dn[0] / max(np.median(dn), 1e-12)),
                "d_argmax_token": int(np.argmax(dn)),
            })

        # channel decomposition of the position-0 state at layer 15 and of Delta_2
        x15 = to_np(xs[15][0]); d2 = to_np(ds[2][0])
        for tag, vec in (("x15_pos0", x15[0]), ("delta2_pos0", d2[0])):
            o = np.argsort(-np.abs(vec))
            chan_rows.append({
                "sample": si, "vec": tag,
                "total_norm": float(np.linalg.norm(vec)),
                "top1": float(abs(vec[o[0]])), "top1_ch": int(o[0]),
                "top2": float(abs(vec[o[1]])), "top2_ch": int(o[1]),
                "top3": float(abs(vec[o[2]])), "top3_ch": int(o[2]),
                "top8_share": float(np.sum(np.abs(vec[o[:8]]) ** 2) /
                                   max(np.sum(vec ** 2), 1e-12)),
            })

        # (b) attribution test
        for e in edges:
            li, lj = e["reader"], e["contributor"]
            x_i = xs[li]
            x_np = to_np(x_i[0])
            d_full = -to_np(ds[lj][0])
            base = block_readout(layers[li], x_i, mask)
            b = {k: to_np(base[k][0]) for k in READOUTS}

            for tag, mask_vec in (("pos0_only", "pos0"), ("all_but_pos0", "rest")):
                if mask_vec == "pos0":
                    d = np.zeros_like(d_full); d[0] = d_full[0]
                else:
                    d = d_full.copy(); d[0] = 0.0
                pert = mx.array(d.astype(np.float32))[None, :, :]
                xp = (x_i.astype(mx.float32) + pert).astype(mx.bfloat16)
                mx.eval(xp)
                rp = block_readout(layers[li], xp, mask)
                mx.eval(rp["out"])
                rm = {}
                for k in READOUTS:
                    p = to_np(rp[k][0])
                    dn = np.linalg.norm(p - b[k], axis=-1)
                    bn = np.linalg.norm(b[k], axis=-1)
                    rm[k] = float(np.mean(dn / np.maximum(bn, 1e-12)))
                hp = rp["out"]
                for k in range(li + 1, N_LAYERS):
                    hp = layers[k](hp, mask, None)
                    mx.eval(hp)
                lg = project_logits(model, hp, mx)
                mx.eval(lg)
                qm = logits_metrics(teacher_logits, lg, ids, mx)
                att_rows.append({
                    "sample": si, "edge": e["id"], "reader": li, "contributor": lj,
                    "variant": tag,
                    "pert_norm_mean": float(np.linalg.norm(d, axis=-1).mean()),
                    "rel_change": rm, "final": qm,
                })
            print(f"sample {si+1}/{len(samples)} done", flush=True)

    (out_dir / "pos0_diagnostic.json").write_text(json.dumps(
        {"layer_norms": layer_rows, "channel_decomposition": chan_rows,
         "attribution": att_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "pos0_attribution.jsonl").open("w", encoding="utf-8") as f:
        for r in att_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick console summary
    print("\n--- per-layer position-0 dominance (mean over samples) ---")
    print(f"{'L':>3} {'x_pos0/median':>15} {'d_pos0/median':>15} {'argmax_tok(x)':>15} {'argmax_tok(d)':>15}")
    for li in range(N_LAYERS):
        rr = [r for r in layer_rows if r["layer"] == li]
        print(f"{li:>3} {np.mean([r['x_pos0_over_median'] for r in rr]):>15.1f} "
              f"{np.mean([r['d_pos0_over_median'] for r in rr]):>15.1f} "
              f"{rr[0]['x_argmax_token']:>15} {rr[0]['d_argmax_token']:>15}")

    print("\n--- attribution: position-0 only vs everywhere else ---")
    print(f"{'edge':>4} {'variant':>14} {'|pert|':>9} {'r_attn rel':>11} {'dnew rel':>9} {'dNLL':>9}")
    for e in EDGE_ORDER:
        for tag in ("pos0_only", "all_but_pos0"):
            rr = [r for r in att_rows if r["edge"] == e and r["variant"] == tag]
            if not rr:
                continue
            print(f"{e:>4} {tag:>14} {np.mean([r['pert_norm_mean'] for r in rr]):>9.3f} "
                  f"{np.mean([r['rel_change']['r_attn'] for r in rr]):>11.4f} "
                  f"{np.mean([r['rel_change']['delta_new'] for r in rr]):>9.4f} "
                  f"{np.mean([r['final']['delta_nll'] for r in rr]):>+9.4f}")


if __name__ == "__main__":
    main()
