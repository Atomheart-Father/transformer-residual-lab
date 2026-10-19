"""Render Stage-1 figures from measured summaries. No synthetic values.

All figures are generated from results/<run>/summary.json, group_summary.csv and
gram_<plan>.json, which are themselves built from the raw JSONL records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

C_PAR = "#2F6FB2"
C_DEL = "#C0504D"
C_LOCAL = "#7F7F7F"
C_ACC = ["#2F6FB2", "#E08214", "#4C9A52", "#8C5AA8"]


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    q = pd.DataFrame(summary["quality"])
    g = pd.DataFrame(summary["groups"]) if summary["groups"] else pd.DataFrame()
    return q, g, summary


def fig_quality_vs_depth(q: pd.DataFrame, out: Path) -> None:
    main = q[(q.accum == "fp32") & q.plan.isin(
        ["serial", "group1", "group2", "group4", "group8", "group28"])].copy()
    dele = q[(q.accum == "fp32") & q.plan.str.startswith("delete-")].copy()
    if main.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, xcol, title in ((axes[0], "effective_depth", "sequential stages (critical path)"),
                            (axes[1], "executed_blocks", "executed decoder blocks (real compute)")):
        for _, r in main.iterrows():
            ax.errorbar(r[xcol], r.delta_nll_mean,
                        yerr=[[r.delta_nll_mean - r.delta_nll_ci95_lo],
                              [r.delta_nll_ci95_hi - r.delta_nll_mean]],
                        fmt="o-", color=C_PAR, ms=5, lw=1.4, capsize=2)
            if r.delta_nll_mean > 0.01:
                ax.annotate(r.plan.replace("group", "g"), (r[xcol], r.delta_nll_mean),
                            textcoords="offset points", xytext=(4, 4), fontsize=7,
                            color=C_PAR)
        for _, r in dele.iterrows():
            ax.errorbar(r[xcol], r.delta_nll_mean,
                        yerr=[[r.delta_nll_mean - r.delta_nll_ci95_lo],
                              [r.delta_nll_ci95_hi - r.delta_nll_mean]],
                        fmt="s", color=C_DEL, ms=5, capsize=2)
            ax.annotate(r.plan.replace("delete-keep", "keep"), (r[xcol], r.delta_nll_mean),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=C_DEL)
        ax.set_xlabel(title)
        ax.set_xlim(left=0)
    if (main.delta_nll_mean < 0.01).any():
        axes[0].annotate("serial = g1", (main.effective_depth.max(), 0),
                         textcoords="offset points", xytext=(-6, -13),
                         ha="right", fontsize=7, color=C_PAR)
    axes[0].set_ylabel("$\\Delta$NLL (nats/token, mean, 95% bootstrap CI)")
    axes[0].set_title("shared-anchor parallel (blue) vs layer deletion (red)")
    fig.tight_layout()
    fig.savefig(out / "fig_quality_vs_depth.png", bbox_inches="tight")
    plt.close(fig)


def fig_group_growth(g: pd.DataFrame, out: Path) -> None:
    if g.empty:
        return
    plans = [p for p in ["group2", "group4", "group8", "group28"] if p in set(g.plan)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, plan in enumerate(plans):
        sub = g[(g.plan == plan) & (g.accum == "fp32")].sort_values("group_index")
        x = sub.start.to_numpy()
        axes[0].plot(x, sub.exit_rms_diff, "o-", ms=3.5, lw=1.3,
                     color=C_ACC[i % len(C_ACC)], label=plan)
        axes[1].plot(x, sub.local_rms_diff, "o-", ms=3.5, lw=1.3,
                     color=C_ACC[i % len(C_ACC)], label=plan)
    axes[0].set_ylabel("||$\\hat h_e - h_e^T$|| RMS (free running)")
    axes[1].set_ylabel("||$G_P(h_s^T) - h_e^T$|| RMS (local)")
    for ax in axes:
        ax.set_xlabel("group start layer")
        ax.set_yscale("log")
    axes[0].set_title("total error at group exit")
    axes[1].set_title("error created inside the group")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_group_error_growth.png", bbox_inches="tight")
    plt.close(fig)


def fig_decomposition(g: pd.DataFrame, out: Path, plan: str = "group4") -> None:
    if g.empty or "plan" not in g.columns:
        return
    sub = g[(g.plan == plan) & (g.accum == "fp32")].sort_values("group_index")
    if sub.empty:
        return
    x = np.arange(len(sub))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(x - w, sub.decomp_inherited_rms, w, label="inherited", color="#B8B8B8")
    ax.bar(x, sub.decomp_amplified_rms, w, label="amplification of inherited", color="#E08214")
    ax.bar(x + w, sub.decomp_new_rms, w, label="newly created", color=C_PAR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}-{int(e)-1}" for s, e in zip(sub.start, sub.end)], fontsize=7)
    ax.set_xlabel("group layer range")
    ax.set_ylabel("RMS of decomposition term")
    ax.set_yscale("log")
    ax.set_title(f"free-running error split at each group boundary ({plan}, fp32)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"fig_decomposition_{plan}.png", bbox_inches="tight")
    plt.close(fig)


def fig_layer_correction(layers: pd.DataFrame, out: Path) -> None:
    if layers.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, plan in enumerate([p for p in ["group2", "group4", "group8"]
                              if p in set(layers.plan)]):
        sub = layers[layers.plan == plan].sort_values("layer")
        ax.plot(sub.layer, sub.correction_relative_rms, "o-", ms=3, lw=1.3,
                color=C_ACC[i], label=plan)
    for lo, hi, name in ((0, 8, "early"), (10, 18, "mid"), (20, 28, "late")):
        ax.axvspan(lo, hi, color="#000000", alpha=0.04)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92, name, ha="center",
                fontsize=7, color="#666666")
    ax.set_xlabel("layer index")
    ax.set_ylabel("||$c_i$|| / ||$d_i^T$||")
    ax.set_title("layer-level correction, anchor = teacher state at group entry")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_layer_correction.png", bbox_inches="tight")
    plt.close(fig)


def fig_gram(run: Path, out: Path) -> None:
    files = sorted(run.glob("gram_group*.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for i, f in enumerate(files):
        d = json.loads(f.read_text(encoding="utf-8"))
        name = f.stem.replace("gram_", "")
        for ax, key, label in ((axes[0], "raw", "raw Gram"),
                               (axes[1], "normalized_correlation", "unit-diagonal correlation")):
            spec = d[key]
            cum = np.asarray(spec["cumulative_ratio"])
            ax.plot(np.arange(1, len(cum) + 1), cum, "-", lw=1.6,
                    color=C_ACC[i % len(C_ACC)],
                    label=f"{name} (rank90={spec['effective_rank_90pct']}, "
                          f"PR={spec['participation_ratio']:.1f})")
        axes[0].set_ylabel("cumulative explained energy")
    for ax, title in zip(axes, ("raw spectrum (magnitude dominated)",
                                "normalized spectrum (direction only)")):
        ax.axhline(0.9, color="#999999", ls="--", lw=0.8)
        ax.set_xlabel("principal direction index")
        ax.set_title(title)
        ax.legend(fontsize=6.5)
    fig.suptitle("correction Gram spectrum, anchor = teacher group entry", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_gram_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def fig_accum(q: pd.DataFrame, out: Path) -> None:
    if not {"fp32", "bf16"} <= set(q.accum):
        return
    a = q[q.accum == "fp32"].set_index("plan").delta_nll_mean
    b = q[q.accum == "bf16"].set_index("plan").delta_nll_mean
    common = a.index.intersection(b.index)
    if common.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(a[common], b[common], s=22, color=C_PAR)
    lim = [min(a[common].min(), b[common].min()) - 0.5,
           max(a[common].max(), b[common].max()) + 0.5]
    ax.plot(lim, lim, "--", color="#999999", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$\\Delta$NLL, fp32 accumulation")
    ax.set_ylabel("$\\Delta$NLL, bf16 accumulation")
    ax.set_title("numeric control: rounding is not the driver")
    fig.tight_layout()
    fig.savefig(out / "fig_accum_control.png", bbox_inches="tight")
    plt.close(fig)


def fig_windows(q: pd.DataFrame, out: Path) -> None:
    sub = q[(q.accum == "fp32") & q.plan.str.startswith("window-")]
    if sub.empty:
        return
    sub = sub.copy()
    sub["window"] = sub.plan.str.extract(r"window-(early|mid|late)")[0]
    sub["size"] = sub.plan.str.extract(r"-g(\d+)")[0].astype(int)
    pivot = sub.pivot_table(index="window", columns="size", values="delta_nll_mean")
    pivot = pivot.reindex(["early", "mid", "late"])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    im = ax.imshow(pivot.values, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [f"g{c}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > np.nanmax(pivot.values) * 0.6 else "black")
    ax.set_title("$\\Delta$NLL when only one 8-layer window is parallelised")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="$\\Delta$NLL")
    fig.tight_layout()
    fig.savefig(out / "fig_window_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_prefix_scan(q: pd.DataFrame, out: Path) -> None:
    sub = q[(q.accum == "fp32") & q.plan.str.startswith("prefix")].copy()
    if sub.empty:
        return
    if "group_size" not in sub.columns:
        sub["group_size"] = sub.plan.str.extract(r"-g(\d+)$")[0].astype(float)
    sub["prefix"] = sub.plan.str.extract(r"^prefix(\d+)")[0].astype(float)
    dele = q[(q.accum == "fp32") & (q.kind == "delete")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for i, g in enumerate((2, 4, 8)):
        s = sub[sub.group_size == g].sort_values("prefix")
        axes[0].plot(s.prefix, s.delta_nll_mean, "o-", ms=4, lw=1.3,
                     color=C_ACC[i], label=f"groups of {g}")
    axes[0].set_xlabel("number of protected serial prefix layers")
    axes[0].set_ylabel("$\\Delta$NLL (nats/token)")
    axes[0].set_title("does more serial prefix always help?")
    axes[0].legend(fontsize=7)

    axes[1].scatter(sub.effective_depth, sub.delta_nll_mean, s=14, color=C_PAR,
                    alpha=0.65, label="prefix protection")
    axes[1].scatter(dele.effective_depth, dele.delta_nll_mean, s=18, marker="s",
                    color=C_DEL, label="layer deletion", zorder=3)
    axes[1].set_xlabel("effective serial depth")
    axes[1].set_ylabel("$\\Delta$NLL (nats/token)")
    axes[1].set_title("quality vs depth (each dot = one plan)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_prefix_scan.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration(q: pd.DataFrame, out: Path) -> None:
    """Grouped bars: uncalibrated sum / mean / sqrt / fitted per plan."""
    sub = q[q.accum == "fp32"].copy()
    if sub.empty:
        return
    if "label" not in sub.columns:
        parts = sub.plan.str.split("+", n=1, expand=True)
        sub["base_plan"] = parts[0]
        sub["label"] = parts[1].fillna("") if parts.shape[1] > 1 else ""
    else:
        sub["base_plan"] = sub.plan
    sub = sub[sub.base_plan.str.startswith("prefix")]
    if sub.empty:
        return
    rows = []
    for plan in sorted(sub.base_plan.unique()):
        p = sub[sub.base_plan == plan]
        base = p[p.label == ""]
        if base.empty:
            continue
        pick = lambda pref: (float(p[p.label.str.startswith(pref)].delta_nll_mean.min())
                             if p.label.str.startswith(pref).any() else np.nan)
        rows.append({
            "plan": plan,
            "depth": int(base.effective_depth.iloc[0]),
            "sum": float(base.delta_nll_mean.iloc[0]),
            "mean": pick("mean"), "sqrt": pick("sqrt"),
            "group_ls": pick("group_ls"), "layer_ls": pick("layer_ls"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("depth")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(df))
    w = 0.17
    series = [("sum", "uncalibrated sum", "#B8B8B8"),
              ("mean", "fixed mean (1/k)", "#9CC3E5"),
              ("sqrt", "fixed sqrt (1/$\\sqrt{k}$)", "#7FB3D5"),
              ("group_ls", "fitted, one per group", C_PAR),
              ("layer_ls", "fitted, one per layer", "#8C5AA8")]
    for i, (col, label, color) in enumerate(series):
        ax.bar(x + (i - 2) * w, df[col], w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nd={d}" for p, d in zip(df.plan, df.depth)], fontsize=7)
    ax.set_ylabel("$\\Delta$NLL (nats/token)")
    ax.set_title("residual fusion calibration on held-out validation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_alpha(coeffs_dir: Path, out: Path) -> None:
    files = sorted(coeffs_dir.glob("*.json")) if coeffs_dir and coeffs_dir.exists() else []
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    picked = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["scheme"] != "layer_ls" or d["lambda_rel"] != 0.01:
            continue
        picked[d["plan"]] = d
    if not picked:
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d["scheme"] == "layer_ls":
                picked.setdefault(d["plan"], d)
    for i, (plan, d) in enumerate(sorted(picked.items())):
        layers = sorted(int(k) for k in d["alpha"])
        ax.plot(layers, [d["alpha"][str(k)] for k in layers], "o-", ms=3, lw=1.3,
                color=C_ACC[i % len(C_ACC)], label=plan)
    ax.axhline(1.0, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("layer index")
    ax.set_ylabel("fitted $\\alpha_i$")
    ax.set_title("per-layer residual coefficient (layer_ls, $\\lambda$=1e-2)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_alpha.png", bbox_inches="tight")
    plt.close(fig)


def fig_b3(b3_dir: Path, scan_dir: Path, out: Path) -> None:
    """B3: magnitude vs direction, plus per-group attribution of the residual."""
    src = b3_dir / "samples.jsonl"
    if not src.exists():
        return
    acc: dict[str, dict[str, list[float]]] = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for k, v in r["variants"].items():
            plan, mode = k.split("::")
            acc.setdefault(plan, {}).setdefault(mode, []).append(v["delta_nll"])
    plans = [p for p in acc if "oracle-dir" in acc[p]]
    if not plans:
        return
    modes = ["none", "oracle-norm", "oracle-dir", "oracle-both"]
    labels = ["no intervention", "oracle: fix magnitude", "oracle: fix direction",
              "oracle: fix both"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(plans))
    w = 0.2
    for i, (m, lab) in enumerate(zip(modes, labels)):
        vals = [float(np.mean(acc[p][m])) if m in acc[p] else np.nan for p in plans]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab,
               color=["#B8B8B8", "#9CC3E5", "#8C5AA8", "#2F6FB2"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(plans, fontsize=7)
    ax.set_ylabel("$\\Delta$NLL (nats/token)")
    ax.set_title("B3: one oracle intervention at the first parallel group exit")
    ax.legend(fontsize=7)

    ax = axes[1]
    if scan_dir and scan_dir.exists():
        gs, vals = [], []
        for d in sorted(scan_dir.iterdir(), key=lambda p: p.name):
            f = d / "samples.jsonl"
            if not f.is_file():
                continue
            ds = []
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                for k, v in r["variants"].items():
                    if k.endswith("::oracle-both"):
                        ds.append(v["delta_nll"])
            if ds:
                gs.append(int(d.name.lstrip("g")))
                vals.append(float(np.mean(ds)))
        if gs:
            ax.plot(gs, vals, "o-", ms=4, lw=1.3, color=C_PAR)
            base = float(np.mean(acc["prefix3-g8"]["none"])) \
                if "prefix3-g8" in acc and "none" in acc["prefix3-g8"] else np.nan
            if not np.isnan(base):
                ax.axhline(base, color="#999999", ls="--", lw=0.8,
                           label="no intervention")
                ax.legend(fontsize=7)
            ax.set_xlabel("group index given the teacher state at its exit")
            ax.set_ylabel("$\\Delta$NLL (nats/token)")
            ax.set_title("prefix3-g8: where the residual actually comes from")
    fig.tight_layout()
    fig.savefig(out / "fig_b3.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/stage1"))
    ap.add_argument("--runs", default=None, help="comma separated run dirs to merge")
    ap.add_argument("--coeffs", type=Path, default=None)
    ap.add_argument("--figdir", type=Path, default=None)
    ap.add_argument("--b3", type=Path, default=None)
    ap.add_argument("--b3-scan", type=Path, default=None)
    args = ap.parse_args()

    runs = [Path(x.strip()) for x in args.runs.split(",")] if args.runs else [args.run]
    out = args.figdir or (runs[0] / "figures")
    out.mkdir(parents=True, exist_ok=True)

    parts, g = [], pd.DataFrame()
    for r in runs:
        qq, gg, _ = load(r)
        parts.append(qq)
        if g.empty:
            g = gg
    q = pd.concat(parts, ignore_index=True)

    layers_path = runs[0] / "layer_summary.csv"
    layers = pd.DataFrame()
    if layers_path.exists() and layers_path.stat().st_size > 2:
        try:
            layers = pd.read_csv(layers_path)
        except pd.errors.EmptyDataError:
            layers = pd.DataFrame()

    fig_quality_vs_depth(q, out)
    fig_group_growth(g, out)
    fig_decomposition(g, out, "group4")
    fig_layer_correction(layers, out)
    fig_gram(runs[0], out)
    fig_accum(q, out)
    fig_windows(q, out)
    fig_prefix_scan(q, out)
    fig_calibration(q, out)
    fig_alpha(args.coeffs, out)
    if args.b3:
        fig_b3(args.b3, args.b3_scan, out)
    print(f"figures -> {out}")
    for p in sorted(out.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
