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


def fig_stage16(root: Path, out: Path) -> None:
    """Boundary scan and compensation, from the Stage 1.6 run directories."""
    def dmap(path):
        if not path.exists():
            return {}
        s = json.loads(path.read_text(encoding="utf-8"))
        return {r["plan"]: r["delta_nll_mean"] for r in s["quality"]
                if r["accum"] == "fp32"}

    cal = {}
    for p in ("screen", "screen_d10extra", "irregular_cal"):
        cal.update(dmap(root / p / "summary.json"))
    val = {}
    for p in ("val", "irregular_val"):
        val.update(dmap(root / p / "summary.json"))

    # --- tail scan: quality vs last-group size, plus the amplification factor
    tail = [("tail-b27", 1), ("tail-b26", 2), ("tail-b25", 3),
            ("tail-b24", 4), ("tail-b23", 5)]
    sizes, cvals, vvals = [], [], []
    for name, size in tail:
        if name in cal:
            sizes.append(size)
            cvals.append(cal[name])
            vvals.append(val.get(name, np.nan))
    if sizes:
        amp = {}
        gpath = root / "screen" / "groups.jsonl"
        if gpath.exists():
            acc: dict[str, dict[int, list[float]]] = {}
            for line in gpath.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["plan"] not in {n for n, _ in tail}:
                    continue
                gi = r["group_index"]
                d = acc.setdefault(r["plan"], {}).setdefault(gi, [0.0, 0.0])
                d[0] += r["decomp_inherited_rms"]
                d[1] += r["decomp_amplified_rms"]
            for name, _ in tail:
                if name in acc:
                    last = acc[name][max(acc[name])]
                    amp[name] = last[1] / max(last[0], 1e-9)

        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        ax.plot(sizes, cvals, "o-", ms=5, lw=1.4, color=C_PAR, label="calibration")
        ax.plot(sizes, vvals, "s--", ms=5, lw=1.2, color=C_DEL, label="validation")
        ax.set_xlabel("size of the last parallel group  ([19,b) + [b,28), depth 7)")
        ax.set_ylabel("$\\Delta$NLL (nats/token)")
        ax.set_title("tail boundary: the last group must stay small")
        ax.legend(fontsize=7, loc="upper left")
        if amp:
            ax2 = ax.twinx()
            ax2.plot([size for name, size in tail if name in amp],
                     [amp[name] for name, _ in tail if name in amp],
                     "^:", ms=5, lw=1.1, color="#8C5AA8")
            ax2.set_ylabel("amplification of inherited error (x)",
                           color="#8C5AA8", fontsize=8)
            ax2.tick_params(axis="y", colors="#8C5AA8")
            ax2.grid(False)
        fig.tight_layout()
        fig.savefig(out / "fig16_tail_scan.png", bbox_inches="tight")
        plt.close(fig)

    # --- group-by-group exit error for the tail family
    gpath = root / "screen" / "groups.jsonl"
    if gpath.exists():
        sel = ["tail-b27", "tail-b26", "tail-b25", "tail-b24"]
        data: dict[str, dict[int, float]] = {}
        for line in gpath.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["plan"] not in sel:
                continue
            d = data.setdefault(r["plan"], {})
            d[r["group_index"]] = d.get(r["group_index"], 0.0) + r["exit_rms_diff"]
        if data:
            n = max(len(v) for v in data.values())
            fig, ax = plt.subplots(figsize=(7.6, 3.4))
            for i, plan in enumerate(sel):
                if plan not in data:
                    continue
                xs = sorted(data[plan])
                ys = [data[plan][g] / 64.0 for g in xs]
                ax.plot(xs, ys, "o-", ms=4, lw=1.3, color=C_ACC[i % len(C_ACC)],
                        label=plan)
            ax.set_yscale("log")
            ax.set_xlabel("group index")
            ax.set_ylabel("||$\\hat h_e - h_e^T$|| RMS at group exit")
            ax.set_title("shrinking the last group shrinks its exit error")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / "fig16_group_exit.png", bbox_inches="tight")
            plt.close(fig)

    # --- compensation comparison
    runs = [("validation", root / "comp" / "eval_val" / "eval_summary.json"),
            ("fresh test", root / "comp" / "eval_fresh" / "eval_summary.json")]
    runs = [(n, p) for n, p in runs if p.exists()]
    if runs:
        order = ["", "+oracle", "+linear_r8", "+mlp_r16", "+mlp_r32"]
        labels = ["no comp", "oracle (upper bound)", "rank-8 linear",
                  "rank-16 MLP", "rank-32 MLP"]
        fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.5),
                                 squeeze=False)
        for j, (name, path) in enumerate(runs):
            m = json.loads(path.read_text(encoding="utf-8"))
            agg = m["aggregate"]
            plan = m.get("plan", "")
            vals, labs = [], []
            for suf, lab in zip(order, labels):
                k = f"{plan}{suf}"
                if k in agg:
                    vals.append(agg[k]["delta_nll_mean"])
                    labs.append(lab)
            ax = axes[0][j]
            bars = ax.bar(range(len(vals)), vals,
                          color=["#B8B8B8", "#4C9A52", "#9CC3E5", "#8C5AA8", "#2F6FB2"][:len(vals)])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
            ax.set_ylabel("$\\Delta$NLL (nats/token)")
            ax.set_title(f"one compensator at group 5 ({name})")
        fig.tight_layout()
        fig.savefig(out / "fig16_compensation.png", bbox_inches="tight")
        plt.close(fig)


def _eval(root: Path, name: str) -> dict:
    path = root / name / "eval_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fig_stage17(root: Path, out: Path) -> None:
    """Stage 1.7: fixed-offset controls, site interference, serial vs parallel."""
    plan = "irr-a"
    # ---------------------------------------------------------------- A
    sets = [("test_v2 (same corpus)", "A_test"), ("imdb (other corpus)", "A_ood")]
    labels = ["no comp", "mean", "shuffled", "rank-32", "rank-64"]
    fig, axes = plt.subplots(1, len(sets), figsize=(9.0, 3.4), sharey=True)
    for ax, (title, key) in zip(np.atleast_1d(axes), sets):
        s = _eval(root, key)
        agg = s.get("aggregate", {})
        if not agg:
            continue
        base = agg[f"{plan}"]["delta_nll_mean"]
        vals, errs = [0.0], [0.0]
        vals.append(agg[f"{plan}+mean"]["delta_nll_mean"] - base)
        sh = [v["delta_nll_mean"] - base for k, v in agg.items()
              if "shuffled" in k]
        vals.append(float(np.mean(sh)) if sh else 0.0)
        errs.append(0.0)
        errs.append(float(np.std(sh)) if len(sh) > 1 else 0.0)
        for r in ("mlp_r32", "mlp_r64"):
            k = f"{plan}+{r}"
            vals.append(agg[k]["delta_nll_mean"] - base if k in agg else 0.0)
            errs.append(0.0)
        cols = [C_LOCAL, C_LOCAL, C_DEL, C_PAR, C_ACC[3]]
        ax.bar(range(len(vals)), vals, yerr=errs, color=cols, width=0.66,
               error_kw={"elinewidth": 0.8, "capsize": 2})
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("$\\Delta$NLL change vs no compensation")
    fig.suptitle("Constant offset explains none of the gain; a wrong pairing is worse than nothing",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig17_fixed_offset.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- table-driven part
    # 后面的图全部从 table.json / paired_stats.json 读，保证图与报告表格同源，
    # 避免"图上一套数、表里一套数"。这两个文件由
    #   scripts/make_stage17_table.py
    #   scripts/stage17_stats.py
    # 生成。
    table = {}
    tp = root / "table.json"
    if tp.exists():
        table = json.loads(tp.read_text(encoding="utf-8"))
    rows = table.get("rows", [])

    def row(method: str, split: str) -> dict | None:
        for r in rows:
            if r["method"] == method and r["split"] == split:
                return r
        return None

    # ---------------------------------------------------------------- sites
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    subs = []
    for d in sorted(root.glob("B_sub_*")):
        s = _eval(root, d.name)
        agg = s.get("aggregate", {}) if s else {}
        key = next((k for k in agg if k.startswith(f"{plan}+serial")), None)
        if not key:
            continue
        subs.append((d.name.replace("B_sub_", "").replace("2_", ""), agg[key]["gap_closure"]))
    if subs:
        subs.sort(key=lambda x: x[1])
        ax.barh([n for n, _ in subs], [v * 100 for _, v in subs],
                color=[C_DEL if v < 0 else C_PAR for _, v in subs])
        ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("gap closed vs no compensation (%)")
    ax.set_title("round-0 4-site r32 model\n(one weight file, only --apply-sites changes)", fontsize=9)

    ax = axes[1]
    price = table.get("final_site_pricing", [])
    if price:
        order = {"4": 0, "5": 1, "4,5": 2}
        price = sorted(price, key=lambda r: order.get(r["apply_sites"], 9))
        ax.bar([r["apply_sites"] for r in price],
               [r["gap_closure"] * 100 for r in price],
               color=[C_DEL if r["gap_closure"] < 0 else C_PAR for r in price], width=0.55)
        ax.axhline(0, color="k", lw=0.8)
        for i, r in enumerate(price):
            ax.annotate(f"{r['delta_nll']:.3f}", (i, r["gap_closure"] * 100),
                        ha="center", va="bottom" if r["gap_closure"] >= 0 else "top",
                        fontsize=7, xytext=(0, 3 if r["gap_closure"] >= 0 else -3),
                        textcoords="offset points")
    ax.set_xlabel("sites that receive the correction")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("final 2-site B3 r64 model\n(site 5 alone is harmful, 4+5 is best)", fontsize=9)
    fig.suptitle("Corrections are only valid on the stream they were fitted on", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig17_site_interference.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ recurrence vs recovery
    # 左：把"站点数"和"是否重采样"分开成两组柱，避免同一条折线在 x=2 处
    #     上下折返（r0 与 r2 是同一个 x）。B1 是 1 站点、40 epoch 的同预算对照。
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    groups = [
        ("1 site\n(B1, 40 ep)", ["B1 single-site r64"], ["B1 single-site r64"]),
        ("2 sites\nround 0", ["B2 2-site r64 r0"], ["B3 2-site r64 r0"]),
        ("2 sites\nround 2", ["B2 2-site r64 r2"], ["B3 2-site r64 r2"]),
        ("4 sites\nround 0", ["B2 4-site r64 (no recollect)"],
         ["B3 4-site r64 (no recollect)"]),
    ]
    w = 0.36
    for k, (lab, b2m, b3m) in enumerate(groups):
        for j, (methods, col) in enumerate(((b2m, "#2F6FB2"), (b3m, "#E08214"))):
            r = row(methods[0], "val")
            if r is None or r.get("gap_closure") is None:
                continue
            ax.bar(k + (j - 0.5) * w, r["gap_closure"] * 100, w * 0.92, color=col,
                   label=("B2 no state" if j == 0 else "B3 carried state") if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("gap closed vs no compensation (%)")
    ax.set_title("Adding sites hurts until the data is re-collected", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ps = root / "paired_stats.json"
    if ps.exists():
        st = json.loads(ps.read_text(encoding="utf-8"))["rows"]
        want = ["B: B2(2site) vs B1(1site)", "B: B3(2site,carry) vs B2(2site)",
                "B: B3(2site,carry) vs B1(1site)"]
        cols = {"val": C_ACC[0], "test": C_ACC[2], "ood": C_ACC[3]}
        w2 = 0.26
        for k, lab in enumerate(want):
            for j, split in enumerate(("val", "test", "ood")):
                r = next((x for x in st if x.get("contrast") == lab and x.get("split") == split), None)
                if not r or "mean_diff" not in r:
                    continue
                lo, hi = r["ci95"]
                ax.errorbar([k + (j - 1) * w2], [r["mean_diff"]],
                            yerr=[[r["mean_diff"] - lo], [hi - r["mean_diff"]]],
                            fmt="o", color=cols[split], ms=4, capsize=2.5, lw=1.0,
                            label=split if k == 0 else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(want)))
        ax.set_xticklabels(["B2 $-$ B1", "B3 $-$ B2", "B3 $-$ B1"], fontsize=8)
        ax.set_ylabel("paired $\\Delta$NLL (negative = right-hand side better)")
        ax.set_title("95% bootstrap CI, per-sample pairing", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.text(0.99, 0.02, "B1 baseline here is the 120-epoch cell (16.6%)",
                transform=ax.transAxes, fontsize=6, ha="right", color="#555555")
    fig.suptitle("Serial depth buys little; the state vector buys almost nothing", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig17_recurrence.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- cost
    # 所有点都来自实测：参数量取 fit/bench 记录，时延取 bench 的 median
    # ms/256-token 调用（单站点 MLP 取 stage1_6 的 benchmark.json，同为 256 token
    # 一次调用）。标注里写明轮次，避免把 r0 的容量点和 r2 的收敛点混为一谈。
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    bench = table.get("bench", {})
    s16 = {}
    b16 = Path("results/stage1_6/comp/benchmark.json")
    if b16.exists():
        s16 = json.loads(b16.read_text(encoding="utf-8"))

    pts = []  # (params, ms, gap%, label, colour, marker)
    for name, method, lab, key in (
            ("r32", "A3 mlp r32", "MLP r32 (1 site)", "rank-32 MLP"),
            ("r64", "A4 mlp r64", "MLP r64 (1 site)", "rank-64 MLP")):
        r = row(method, "val")
        b = s16.get(key, {})
        if r and b:
            pts.append((b["params"], b["ms_per_256tok"], r["gap_closure"] * 100,
                        lab, C_LOCAL, "o"))
    for lab, method, key in (
            ("B1 r64 (1 site)", "B1 single-site r64", "serial_b1_r64.npz"),
            ("B2 r64 (2 sites, r2)", "B2 2-site r64 r2", "serial_b2_r64.npz"),
            ("B3 r64 (2 sites, r2)", "B3 2-site r64 r2", "serial_b3_r64.npz"),
            ("B3 r64 (2 sites, r0)", "B3 2-site r64 r0", "serial_b3_r64.npz"),
            ("B3 r32 (2 sites, r0)", "B3 2-site r32 r0", "serial_b3_r32.npz")):
        r = row(method, "val")
        b = bench.get(key)
        if r and b:
            pts.append((r.get("n_params") or b["n_params"], b["chain_ms"],
                        r["gap_closure"] * 100, lab,
                        "#E08214" if "B3" in lab else "#2F6FB2",
                        "s" if "B3" in lab else "o"))
    for ax, xi, xl, logx in ((axes[0], 0, "parameters (log scale)", True),
                             (axes[1], 1, "measured ms per 256-token call", False)):
        for p_ in pts:
            ax.scatter([p_[xi]], [p_[2]], s=34, color=p_[4], marker=p_[5], zorder=3)
        # 右上角两个 r2 点几乎重合，按标签错开，避免压字
        off = {"B2 r64 (2 sites, r2)": (-8, -12), "B3 r64 (2 sites, r2)": (6, 6),
               "B1 r64 (1 site)": (6, -12)}
        for p_ in pts:
            dx, dy = off.get(p_[3], (5, 5))
            ax.annotate(p_[3], (p_[xi], p_[2]), fontsize=6.5, xytext=(dx, dy),
                        textcoords="offset points",
                        ha="right" if dx < 0 else "left")
        ax.set_xlabel(xl)
        ax.set_ylabel("gap closed on val (%)")
        if logx:
            ax.set_xscale("log")
        ax.margins(x=0.25, y=0.2)
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_title("Doubling parameters does not help", fontsize=9)
    axes[1].set_title("One transformer block costs 2.6 ms for scale", fontsize=9)
    fig.suptitle("Cost-quality Pareto: the plain MLP is still the best point", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig17_cost_pareto.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ exit error
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    ax = axes[0]
    s = _eval(root, "B_val_45_r2")
    if s:
        agg = s["aggregate"]
        # no-state trace 与 carried 只差 ≤0.0015，画两条等于画一条
        for k, col, lab in ((plan, "k", "no compensation"),
                            (f"{plan}+serial_b3_r64", C_ACC[1], "B3 carried state (r2)")):
            t = agg.get(k, {}).get("exit_rel_rms_mean")
            if t:
                ax.plot(range(len(t)), t, "o-", color=col, ms=4, label=lab, lw=1.6)
        ax.set_xlabel("group index")
        ax.set_ylabel("relative RMS error at group exit")
        ax.set_title("Accumulated state error, round 2", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    fits = [("B1 site5", "B_r0_5", "b1_r64e120"),
            ("B2 s4", "B_r2_45", "b2_r64"), ("B2 s5", "B_r2_45", "b2_r64"),
            ("B3 s4", "B_r2_45", "b3_r64"), ("B3 s5", "B_r2_45", "b3_r64")]
    labs, rel, cos = [], [], []
    for lab, sub, tag in fits:
        f = root / sub / f"fit_{tag}.json"
        if not f.exists():
            continue
        ps = json.loads(f.read_text(encoding="utf-8"))["per_site_val"]
        if "B1" in lab:
            v = list(ps.values())[0]
            labs.append(lab)
            rel.append(v["rel_err"])
            cos.append(v["cosine"])
        else:
            idx = 0 if lab.endswith("s4") else 1
            v = ps[f"site{idx}"]
            labs.append(lab)
            rel.append(v["rel_err"])
            cos.append(v["cosine"])
    if rel:
        x = np.arange(len(labs))
        ax.bar(x, rel, 0.55, color=C_PAR, label="relative error of $\\hat c$")
        ax.plot(x, cos, "o", color=C_ACC[1], ms=5, label="cosine($\\hat c$, $c^*$)")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_ylim(0, 1.18)
        ax.set_title("Per-site prediction error (validation windows)", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper center", ncol=2)
    fig.suptitle("The predictor is weak per site; the correction still helps", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig17_exit_error.png", bbox_inches="tight")
    plt.close(fig)


def fig_stage18(root: Path, out: Path) -> None:
    """Stage 1.8: distillation quality, inheritance effect, fusion outcomes.

    Everything is read from table.json / fit_stats.json / paired_stats.json so the
    figures and the report table cannot drift apart (the Stage 1.7 lesson).
    """
    table_p = root / "table.json"
    table = json.loads(table_p.read_text(encoding="utf-8")) if table_p.exists() else {}
    fit_stats = {}
    fs_p = root / "fit_stats.json"
    if fs_p.exists():
        fit_stats = json.loads(fs_p.read_text(encoding="utf-8"))
    paired = {}
    ps_p = root / "paired_stats.json"
    if ps_p.exists():
        paired = json.loads(ps_p.read_text(encoding="utf-8"))

    names = {"a1": "A1 serial MLP", "a2": "A2 tiny (random)",
             "a3": "A3 tiny (inherited)"}

    # -------------------------------------------------- 1. distillation quality
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    ax = axes[0]
    keys = ["a1", "a2", "a3"]
    before = [r.get("before_rel_err") for r in table.get("fit", [])]
    after = [r.get("after_rel_err") for r in table.get("fit", [])]
    bcos = [r.get("before_cosine") for r in table.get("fit", [])]
    acos = [r.get("after_cosine") for r in table.get("fit", [])]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, before, w, color=C_LOCAL, label="before training")
    ax.bar(x + w / 2, after, w, color=C_PAR, label="after training")
    ax.axhline(1.0, color=C_DEL, lw=0.9, ls="--", label="chance (rel.err 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([names[k] for k in keys], rotation=12, ha="right")
    ax.set_ylabel("rel.err on distillation target")
    ax.set_title("All three narrow units fit the 8-layer delta")
    ax.legend(loc="upper right")
    for xi, v in zip(x - w / 2, before):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, after):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

    ax = axes[1]
    hist = {}
    for tag in ("a1_short", "a2_short", "a3_short"):
        d = table.get("fit", [])
        p = root / "fit" / f"fit_{tag}.json"
        if p.exists():
            hist[tag] = json.loads(p.read_text(encoding="utf-8"))
    styles = {"a1_short": (C_LOCAL, "o-"), "a2_short": (C_ACC[1], "s-"),
              "a3_short": (C_PAR, "^-")}
    labels = {"a1_short": "A1 serial MLP (random)",
              "a2_short": "A2 tiny (random)",
              "a3_short": "A3 tiny (inherited)"}
    for tag, h in hist.items():
        col, st = styles[tag]
        eps = [r["epoch"] for r in h["history"]]
        rel = [r["val_rel_err"] for r in h["history"]]
        ax.plot(eps, rel, st, ms=5, lw=1.4, color=col, label=labels[tag])
    # full-budget endpoints, drawn as isolated markers for the same units
    end_tags = ["a1_serial", "a2_random", "a3_inherited"]
    end_cols = {"a1_serial": C_LOCAL, "a2_random": C_ACC[1], "a3_inherited": C_PAR}
    for i, r in enumerate(table.get("fit", [])):
        t = end_tags[i]
        ax.plot([r["epochs"]], [r["after_rel_err"]], "o", ms=8,
                mfc="none", mew=1.6, color=end_cols[t],
                label=f"{names[keys[i]]} @ {r['epochs']} ep")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val rel.err on distillation target")
    ax.set_title("3-epoch budget: inherited leads, random catches up by 20")
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("A narrow serial unit reproduces the 8-layer delta; "
                 "inheritance buys speed, not accuracy", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig18_distillation.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------- 2. inheritance forest plot
    res = fit_stats.get("results", {})
    pool = []
    for split, d in res.items():
        for r in d.get("paired", []):
            r = dict(r)
            r["split"] = split
            pool.append(r)
    if pool:
        want = [r for r in pool if r["metric"] == "rel_err"]
        order = ["A3(inherited) vs A2(random) short",
                 "A3(inherited) vs A2(random) full",
                 "A1(serial MLP) vs A2(random) full",
                 "A3(inherited) vs A1(serial MLP) full",
                 "A1(serial MLP) vs A3(inherited) short"]
        labels, vals, los, his, cols = [], [], [], [], []
        for tag in order:
            for r in want:
                if r["contrast"] != tag:
                    continue
                labels.append(f"{tag} [{r['split']}] n={r['n']}")
                vals.append(r["mean_diff"])
                los.append(r["ci95"][0])
                his.append(r["ci95"][1])
                sig = r["significant_95"]
                # d = rel_err(A) - rel_err(B); negative means A is the better fit
                cols.append(C_ACC[2] if (sig and r["mean_diff"] < 0)
                            else (C_DEL if (sig and r["mean_diff"] > 0) else C_LOCAL))
        if labels:
            fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(labels) + 1.6))
            y = np.arange(len(labels))[::-1]
            for yi, v, lo, hi, c in zip(y, vals, los, his, cols):
                ax.plot([lo, hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
                ax.plot(v, yi, "o", ms=6, color=c)
            ax.axvline(0, color="k", lw=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("paired bootstrap mean difference of rel.err (A − B)")
            ax.set_title("Distillation target: inheritance helps at 3 ep, reverses by 20")
            ax.set_xlim(min(los) * 1.15 - 0.002, max(his) * 1.15 + 0.002)
            fig.tight_layout()
            fig.savefig(out / "fig18_inheritance.png", bbox_inches="tight")
            plt.close(fig)

    # -------------------------------------------------- 3. fusion outcomes
    ev = table.get("eval", {})
    runs = [r for r in ev if "variants" in ev.get(r, {})]
    if runs:
        show = ["irr-a", "irr-a+oracle", "tiny_sub_a3_inherited", "fuse_f_a3z",
                "fuse_f_trunk", "fuse_f_a1", "fuse_f_a3", "fuse_f_a2"]
        fig, ax = plt.subplots(figsize=(9.6, 3.6))
        x = np.arange(len(runs))
        w = 0.8 / len(show)
        palette = {"irr-a": C_LOCAL, "irr-a+oracle": C_ACC[3]}
        for j, v in enumerate(show):
            vals, errs = [], []
            for r in runs:
                m = ev[r]["variants"].get(v)
                vals.append(m["delta_nll_mean"] if m else np.nan)
                errs.append(m["delta_nll_std"] / np.sqrt(m["n"]) if m else np.nan)
            col = palette.get(v, C_ACC[j % 3])
            ax.bar(x + (j - len(show) / 2 + 0.5) * w, vals, w,
                   color=col, label=v, yerr=errs,
                   error_kw={"elinewidth": 0.7, "capsize": 2})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}\n(n={ev[r].get('n_samples')})" for r in runs],
                           fontsize=8)
        ax.set_ylabel("ΔNLL vs the teacher")
        ax.set_title("Fusion heads plugged into the free-running parallel trunk")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "fig18_fusion.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/stage1"))
    ap.add_argument("--runs", default=None, help="comma separated run dirs to merge")
    ap.add_argument("--coeffs", type=Path, default=None)
    ap.add_argument("--figdir", type=Path, default=None)
    ap.add_argument("--b3", type=Path, default=None)
    ap.add_argument("--b3-scan", type=Path, default=None)
    ap.add_argument("--stage16", type=Path, default=None,
                    help="results/stage1_6 root; also renders boundary + compensation")
    ap.add_argument("--stage17", type=Path, default=None,
                    help="results/stage1_7 root; renders the Stage 1.7 figures")
    ap.add_argument("--stage18", type=Path, default=None,
                    help="results/stage1_8 root; renders the Stage 1.8 figures")
    args = ap.parse_args()

    runs = [Path(x.strip()) for x in args.runs.split(",")] if args.runs else [args.run]
    out = args.figdir or (runs[0] / "figures")
    out.mkdir(parents=True, exist_ok=True)

    if args.stage17:
        fig_stage17(args.stage17, out)
        if not (runs[0] / "summary.json").exists():
            print(f"figures -> {out}")
            for p in sorted(out.glob("fig17_*.png")):
                print(" ", p.name)
            return

    if args.stage18:
        fig_stage18(args.stage18, out)
        if not (runs[0] / "summary.json").exists():
            print(f"figures -> {out}")
            for p in sorted(out.glob("fig18_*.png")):
                print(" ", p.name)
            return

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
    if args.stage16:
        fig_stage16(args.stage16, out)
    print(f"figures -> {out}")
    for p in sorted(out.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
