"""Stage 2.1 figures: the single-edge cost map and the depth/quality frontier.

Reads the Stage 2.1 search results (no model needed) and writes
`results/stage2_1/figures/fig_single_edges.png` and
`fig_depth_frontier.png`.

  fig 1  measured free-running dNLL of every single dropped edge, by distance
         and by block, with the depth-reducing (distance-1) set marked - the
         local sensitivity map of the real design space.
  fig 2  window schedulable depth vs best measured dNLL (exhaustive enumeration
         + closure + regular families), i.e. the depth/quality trade-off the
         whole stage is about.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage2_1/figures"


def load(name):
    return json.loads((ROOT / f"results/stage2_1/{name}/search.json").read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single = load("search_single")["variants"]
    pts = []
    for k, v in single.items():
        m = re.fullmatch(r"d(\d+)_(\d+)", k)
        if not m:
            continue
        b, j = int(m.group(1)), int(m.group(2))
        pts.append({"block": b, "miss": j, "dist": b - j,
                    "dnll": v["delta_nll_mean"],
                    "ci": v.get("vs_serial", {}).get("ci95")})

    # ---- figure 1: the single-edge map -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for p in pts:
        depth_reducing = p["dist"] == 1
        ax.errorbar(p["dist"] + np.random.default_rng(p["block"] * 31 + p["miss"]).uniform(-0.18, 0.18),
                    p["dnll"],
                    yerr=[[p["dnll"] - p["ci"][0]], [p["ci"][1] - p["dnll"]]] if p["ci"] else None,
                    fmt="o" if not depth_reducing else "D",
                    ms=4 if not depth_reducing else 6,
                    color="#c0392b" if depth_reducing else "#2c3e50",
                    alpha=0.75 if not depth_reducing else 0.95, capsize=0, lw=0.8)
    by_d = defaultdict(list)
    for p in pts:
        by_d[p["dist"]].append(p["dnll"])
    xs = sorted(by_d)
    ax.plot(xs, [np.mean(by_d[d]) for d in xs], "-", color="#e67e22", lw=2,
            label="mean per distance")
    ax.set_xlabel("distance of dropped edge (block i - dropped predecessor j)")
    ax.set_ylabel("free-running $\\Delta$NLL (nats)")
    ax.set_title("every single dropped edge in window [10,18)\ndiamonds = depth-reducing (distance 1)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    by_b = defaultdict(list)
    for p in pts:
        by_b[p["block"]].append(p["dnll"])
    bs = sorted(by_b)
    ax.plot(bs, [np.mean(by_b[b]) for b in bs], "o-", color="#2980b9", lw=2,
            label="mean over all dropped distances")
    d1 = [np.mean([p["dnll"] for p in pts if p["block"] == b and p["dist"] == 1]) for b in bs]
    ax.plot(bs, d1, "s--", color="#c0392b", lw=2,
            label="depth-reducing drops only (distance 1)")
    ax.set_xlabel("block whose predecessor set is truncated")
    ax.set_ylabel("free-running $\\Delta$NLL (nats)")
    ax.set_title("position effect: later blocks are cheaper to perturb")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_single_edges.png", dpi=160)
    plt.close(fig)

    # ---- figure 2: depth / quality frontier --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ex = ROOT / "results/stage2_1/search_exhaustive/search.json"
    if ex.exists():
        e = json.loads(ex.read_text())
        V = e["variants"]
        by = defaultdict(list)
        for k, v in V.items():
            if k == "serial":
                continue
            by[v["window_levels"]].append((v["delta_nll_mean"],
                                           v["missed_updates_in_window"]))
        depths = sorted(by, reverse=True)
        best = [min(by[d])[0] for d in depths]
        med = [float(np.median([x for x, _ in by[d]])) for d in depths]
        worst = [max(by[d])[0] for d in depths]
        ax.fill_between(depths, best, worst, color="#2980b9", alpha=0.15,
                        label="exhaustive reachable space (754 structures)")
        ax.plot(depths, best, "o-", color="#2980b9", lw=2, ms=5,
                label="best at this depth (exhaustive)")
        ax.plot(depths, med, ":", color="#2980b9", lw=1.5, label="median")
    fam = load("search_random_depth")["variants"]
    for nm, lbl, col in (("g2", "g2 (depth 4, contiguous pairs)", "#c0392b"),
                         ("g4", "g4 (depth 2)", "#8e44ad"),
                         ("mix3", "mix3 (depth 3)", "#27ae60")):
        if nm in fam:
            v = fam[nm]
            ax.plot(v["window_levels"], v["delta_nll_mean"], "*", ms=16,
                    color=col, label=lbl)
    fam12 = load("search_families")["variants"]
    if "pair-first" in fam12:
        v = fam12["pair-first"]
        ax.plot(v["window_depth"], v["delta_nll_mean"], "P", ms=9, color="#7f8c8d",
                label="single pair (depth 7)")
    ax.plot(8, 0.0, "k_", ms=14, mew=2, label="serial (depth 8)")
    ax.set_xlabel("window schedulable depth (dependency levels inside blocks 10-17)")
    ax.set_ylabel("free-running $\\Delta$NLL vs teacher (nats)")
    ax.set_title("depth / quality trade-off over the whole reachable design space")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth_frontier.png", dpi=160)
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
