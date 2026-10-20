"""M1 analysis: aggregate the raw M1 read-outs into the summary tables and figures.

Reads (read-only):
    results/mech_m1/local_readouts.jsonl
    results/mech_m1/suffix_propagation.jsonl
    results/mech_m1/stage2_contrast.jsonl
    results/mech_m1/gate1_equivalence.json
    results/mech_m1/gate2_semantics.json
    results/mech_m1/layer_stats.json
Writes:
    results/mech_m1/summary.json
    results/mech_m1/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/mech_m1")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

READOUTS = ("ln1", "r_attn", "h", "ln2", "r_mlp", "out", "delta_new")
LABEL = {"ln1": "LN1", "r_attn": "r_attn", "h": "h=x+r_attn", "ln2": "LN2",
         "r_mlp": "r_mlp", "out": "out", "delta_new": "Delta_new"}
CONDITIONS = ("real", "matched_radial", "matched_tan", "matched_rand",
              "small_radial", "small_tan", "small_rand")
EDGE_ORDER = ("P1", "C2", "C1", "P2")


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


local = jl(ROOT / "local_readouts.jsonl")
suffix = jl(ROOT / "suffix_propagation.jsonl")
stage2 = jl(ROOT / "stage2_contrast.jsonl")
layer_stats = json.loads((ROOT / "layer_stats.json").read_text(encoding="utf-8"))
gate1 = json.loads((ROOT / "gate1_equivalence.json").read_text(encoding="utf-8"))
gate2 = json.loads((ROOT / "gate2_semantics.json").read_text(encoding="utf-8"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CN = {"P1": "P1 (15,2)", "C2": "C2 (16,15)", "C1": "C1 (15,4)", "P2": "P2 (16,1)"}
COND_CN = {"real": "real (-Delta_j)", "matched_radial": "matched_radial",
           "matched_tan": "matched_tan", "matched_rand": "matched_rand",
           "small_radial": "small_radial", "small_tan": "small_tan",
           "small_rand": "small_rand"}
COL = {"real": "#c0392b", "matched_radial": "#2e86c1", "matched_tan": "#27ae60",
       "matched_rand": "#8e44ad", "small_radial": "#7fb3d5",
       "small_tan": "#82e0aa", "small_rand": "#c39bd3"}

# ---------------------------------------------------------------- aggregate
def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "sd": float(np.std(vals)), "n": len(vals)}


summary = {
    "n_samples": len({r["sample_id"] for r in local}),
    "gates": {"gate1": gate1, "gate2": gate2},
    "edges": {},
}

for edge in EDGE_ORDER:
    e_rows = [r for r in local if r["edge"] == edge]
    if not e_rows:
        continue
    e_s2 = [r for r in stage2 if r["edge"] == edge]
    ent = {"conditions": {}, "stage2": {}}
    for cond in CONDITIONS:
        cr = [r for r in e_rows if r["condition"] == cond]
        if not cr:
            continue
        c = {"perturbation": {}, "readouts": {}, "final": {}}
        for k in ("rel_norm_mean", "rel_norm_median", "rel_norm_rms_weighted",
                  "radial_fraction_mean", "tangential_fraction_mean",
                  "cos_with_x_mean", "abs_norm_mean", "abs_norm_median"):
            c["perturbation"][k] = agg(cr, None) if False else agg(
                [{"v": r["perturbation"].get(k)} for r in cr], "v")
        for ro in READOUTS:
            c["readouts"][ro] = {}
            for k in ("rel_change_mean", "rel_change_rms_weighted", "abs_change_mean",
                      "response_gain_abs", "response_gain_rel", "cos_pert_base_mean"):
                c["readouts"][ro][k] = agg(
                    [{"v": r["readouts"][ro].get(k)} for r in cr], "v")
        for k in ("delta_nll", "teacher_to_candidate_kl", "top1_agreement"):
            c["final"][k] = agg([{"v": r["final"].get(k)} for r in cr], "v")
        ent["conditions"][cond] = c

    # S-STAGE2 contrast
    if e_s2:
        ent["stage2"] = {
            "delta_nll": agg([{"v": r["final"]["delta_nll"]} for r in e_s2], "v"),
            "kl": agg([{"v": r["final"]["teacher_to_candidate_kl"]} for r in e_s2], "v"),
            "top1": agg([{"v": r["final"]["top1_agreement"]} for r in e_s2], "v"),
            "trajectory_rel_rms": {
                str(e_s2[0]["trajectory"][k]["layer"]): float(np.mean(
                    [r["trajectory"][k]["rel_change"]["rms_weighted"] for r in e_s2]))
                for k in range(len(e_s2[0]["trajectory"]))},
        }
    # S-PERSIST trajectory (real condition)
    sr = [r for r in suffix if r["edge"] == edge and r["condition"] == "real"]
    if sr:
        base_layer = sr[0]["suffix"][0]["layer"]
        ent["persist_trajectory_rel_rms"] = {
            str(base_layer + i): float(np.mean(
                [s["suffix"][i]["rel_change"]["rms_weighted"] for s in sr]))
            for i in range(len(sr[0]["suffix"]))}
    summary["edges"][edge] = ent

# first amplification point: first read-out (computation order) whose
# relative response gain reaches 1.0
def first_amp(r):
    for ro in READOUTS:
        g = r["readouts"][ro].get("response_gain_rel")
        if g is not None and g >= 1.0:
            return ro
    return None

summary["first_amplification"] = {}
for edge in EDGE_ORDER:
    for cond in CONDITIONS:
        cr = [r for r in local if r["edge"] == edge and r["condition"] == cond]
        fa = [first_amp(r) for r in cr]
        vals = [READOUTS.index(x) for x in fa if x]
        summary["first_amplification"].setdefault(edge, {})[cond] = {
            "counts": {ro: int(sum(1 for x in fa if x == ro)) for ro in READOUTS},
            "median_index": float(np.median(vals)) if vals else None,
        }

(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

# --- Fig 1: submodule response profile, S-PERSIST, matched-magnitude conditions
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for ax, edge in zip(axes, EDGE_ORDER):
    ent = summary["edges"][edge]["conditions"]
    x = np.arange(len(READOUTS))
    for cond in ("real", "matched_radial", "matched_tan", "matched_rand"):
        y = [ent[cond]["readouts"][ro]["rel_change_mean"]["mean"] for ro in READOUTS]
        style = "--^" if cond == "matched_tan" else "o-"
        ax.plot(x, y, style, color=COL[cond], label=COND_CN[cond], lw=1.6, ms=5,
                mfc="none" if cond == "matched_tan" else COL[cond])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[r] for r in READOUTS], rotation=40, ha="right", fontsize=8)
    ax.set_title(f"{CN[edge]}   eta={ent['real']['perturbation']['rel_norm_mean']['mean']:.3f}"
                 f"  radial={ent['real']['perturbation']['radial_fraction_mean']['mean']:.2f}",
                 fontsize=9)
axes[0].set_ylabel("relative change of read-out (log)")
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Fig.1  Sub-module response profile to a *missing* upstream update  "
             "(S-PERSIST, matched per-token magnitude)", fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig1_submodule_profile.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 2: small-perturbation (eta=1%) profile -> radial vs tangential
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for ax, edge in zip(axes, EDGE_ORDER):
    ent = summary["edges"][edge]["conditions"]
    x = np.arange(len(READOUTS))
    for cond in ("small_radial", "small_tan", "small_rand"):
        y = [ent[cond]["readouts"][ro]["rel_change_mean"]["mean"] for ro in READOUTS]
        style = "--^" if cond == "small_tan" else "o-"
        ax.plot(x, y, style, color=COL[cond], label=COND_CN[cond], lw=1.6, ms=5,
                mfc="none" if cond == "small_tan" else COL[cond])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[r] for r in READOUTS], rotation=40, ha="right", fontsize=8)
    ax.set_title(CN[edge], fontsize=9)
axes[0].set_ylabel("relative change (log)")
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Fig.2  Local Jacobian profile at eta = 1% of ||x_i||  "
             "(RMSNorm freezes the radial direction, not the tangential one)", fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig2_eta1_profile.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 3: suffix propagation trajectory, S-PERSIST vs S-STAGE2
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for ax, edge in zip(axes, EDGE_ORDER):
    ent = summary["edges"][edge]
    s2t = ent.get("stage2", {}).get("trajectory_rel_rms", {})
    pt = ent.get("persist_trajectory_rel_rms", {})
    if s2t:
        ks = sorted(int(k) for k in s2t)
        ax.plot(ks, [s2t[str(k)] for k in ks], "s--", color="#2e86c1", ms=3.5,
                label="S-STAGE2 (Stage-2.1 semantics)")
    if pt:
        ks = sorted(int(k) for k in pt)
        ax.plot(ks, [pt[str(k)] for k in ks], "o-", color="#c0392b", ms=3.5,
                label="S-PERSIST (missing update stays missing)")
    ax.set_yscale("log")
    ax.set_xlabel("block index")
    ax.set_title(CN[edge], fontsize=9)
axes[0].set_ylabel("relative state deviation (rms)")
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Fig.3  Propagation of the missing update through the suffix: "
             "eliminated (re-injected) vs. carried", fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig3_suffix_propagation.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 4: end-to-end damage, three semantics
edges = [e for e in EDGE_ORDER if e in summary["edges"]]
labels = [CN[e] for e in edges]
recorded = [summary["edges"][e]["conditions"]["real"]["perturbation"] and
            {"P1": 0.06985, "C2": 0.04119, "C1": 0.00472, "P2": 0.00072}[e] for e in edges]
s_persist = [summary["edges"][e]["conditions"]["real"]["final"]["delta_nll"]["mean"] for e in edges]
s_stage2 = [summary["edges"][e]["stage2"]["delta_nll"]["mean"] for e in edges]
x = np.arange(len(edges))
w = 0.27
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(x - w, recorded, w, color="#95a5a6", label="Stage 2.1 recorded (32 samples)")
ax.bar(x, s_stage2, w, color="#2e86c1", label="S-STAGE2 (this run, 8 samples)")
ax.bar(x + w, s_persist, w, color="#c0392b", label="S-PERSIST (this run, 8 samples)")
for xi, (a, b, c) in enumerate(zip(recorded, s_stage2, s_persist)):
    ax.text(xi - w, a, f"{a:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.text(xi, b, f"{b:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.text(xi + w, c, f"{c:.3f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("delta NLL (nats)")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Fig.4  End-to-end damage depends on the intervention semantics,\n"
             "not only on the magnitude of the removed update", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig4_semantics_damage.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 5: local response vs end-to-end damage (decoupling)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
markers = {"real": "o", "matched_radial": "^", "matched_tan": "s", "matched_rand": "D"}
for edge in edges:
    ent = summary["edges"][edge]["conditions"]
    for cond in ("real", "matched_radial", "matched_tan", "matched_rand"):
        g = ent[cond]["readouts"]["delta_new"]["response_gain_rel"]["mean"]
        d = ent[cond]["final"]["delta_nll"]["mean"]
        if g is None:
            continue
        ax.scatter(g, d, marker=markers[cond], color=COL[cond], s=42,
                   edgecolor="k", linewidth=0.4, zorder=3)
        if cond == "real":
            ax.annotate(edge, (g, d), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("local gain at Delta_new  (relative output change / relative input change)")
ax.set_ylabel("end-to-end delta NLL (S-PERSIST)")
ax.set_title("Fig.5  Local submodule gain and end-to-end damage are\n"
             "largely decoupled across conditions", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig5_local_vs_global.png", bbox_inches="tight")
plt.close(fig)

print("summary written ->", ROOT / "summary.json")
print("figures ->", sorted(p.name for p in FIG.glob("*.png")))
