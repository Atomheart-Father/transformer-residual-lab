"""Aggregate Stage-1 raw JSONL into CSV/Markdown tables with bootstrap CIs.

Only measured numbers are written. Nothing is imputed or simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUALITY = ["delta_nll", "teacher_to_candidate_kl", "top1_agreement",
           "token_nll_p50", "token_nll_p95"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def quality_table(samples: list[dict]) -> pd.DataFrame:
    recs = []
    for row in samples:
        for key, v in row["variants"].items():
            fs = v["final_state"]
            recs.append({
                "variant": key,
                "plan": key.rsplit("@", 1)[0],
                "accum": v["accum"],
                "kind": v["kind"],
                "executed_blocks": v["executed_blocks"],
                "effective_depth": v["effective_depth"],
                "label": v.get("label", ""),
                "fusion": v.get("fusion", ""),
                "prefix": v.get("prefix", 0),
                "group_size": v.get("group_size", 0),
                "sample_id": row["sample_id"],
                "teacher_nll": v["teacher_nll"],
                "candidate_nll": v["candidate_nll"],
                "delta_nll": v["delta_nll"],
                "kl": v["teacher_to_candidate_kl"],
                "top1": v["top1_agreement"],
                "token_nll_p50": v["token_nll_p50"],
                "token_nll_p95": v["token_nll_p95"],
                "state_rms": fs["rms_diff"],
                "state_rel_rms": fs["rel_rms_diff"],
                "state_cos": fs["cosine"],
                "state_max_abs": fs["max_abs_diff"],
            })
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    rows = []
    for (variant, plan, accum, kind), g in df.groupby(
            ["variant", "plan", "accum", "kind"], sort=False):
        d = g["delta_nll"].to_numpy()
        lo, hi = boot_ci(d)
        rows.append({
            "variant": variant,
            "plan": plan,
            "accum": accum,
            "kind": kind,
            "n_samples": len(g),
            "executed_blocks": int(g["executed_blocks"].iloc[0]),
            "effective_depth": int(g["effective_depth"].iloc[0]),
            "delta_nll_mean": float(d.mean()),
            "delta_nll_std": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "delta_nll_ci95_lo": lo,
            "delta_nll_ci95_hi": hi,
            "kl_mean": float(g["kl"].mean()),
            "top1_mean": float(g["top1"].mean()),
            "ppl_ratio": float(np.exp(d.mean())),
            "teacher_nll_mean": float(g["teacher_nll"].mean()),
            "state_rel_rms_mean": float(g["state_rel_rms"].mean()),
            "state_cos_mean": float(g["state_cos"].mean()),
            "state_max_abs_mean": float(g["state_max_abs"].mean()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["accum", "effective_depth", "plan"]).reset_index(drop=True)


def group_table(groups: list[dict]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    df = pd.DataFrame(groups)
    keys = ["plan", "accum", "group_index", "start", "end", "size"]
    agg = df.groupby(keys, sort=False).agg(
        entry_rms_diff=("entry_rms_diff", "mean"),
        local_rms_diff=("local_rms_diff", "mean"),
        exit_rms_diff=("exit_rms_diff", "mean"),
        exit_rel_rms_diff=("exit_rel_rms_diff", "mean"),
        decomp_inherited_rms=("decomp_inherited_rms", "mean"),
        decomp_amplified_rms=("decomp_amplified_rms", "mean"),
        decomp_new_rms=("decomp_new_rms", "mean"),
        decomp_total_rms=("decomp_total_rms", "mean"),
        teacher_state_rms=("teacher_state_rms", "mean"),
        n=("sample_id", "count"),
    ).reset_index()
    return agg


def layer_table(layers: list[dict]) -> pd.DataFrame:
    if not layers:
        return pd.DataFrame()
    df = pd.DataFrame(layers)
    agg = df.groupby(["plan", "group_start", "layer", "anchor"], sort=False).agg(
        teacher_delta_rms=("teacher_delta_rms", "mean"),
        parallel_delta_rms=("parallel_delta_rms", "mean"),
        correction_rms=("correction_rms", "mean"),
        correction_relative_rms=("correction_relative_rms", "mean"),
        delta_cosine=("delta_cosine", "mean"),
        n=("sample_id", "count"),
    ).reset_index()
    return agg


def markdown(quality: pd.DataFrame, meta: dict,
             thresholds: pd.DataFrame | None = None,
             matched: pd.DataFrame | None = None) -> str:
    lines = ["# Stage-1 measured summary", "",
             f"- model: `{meta.get('model')}` revision `{meta.get('model_revision')}`",
             f"- samples: {meta.get('n_measured')} x {meta.get('n_predicted_tokens')} predicted tokens",
             f"- git: `{meta.get('git_commit')}`",
             "", "| variant | blocks | depth | dNLL | 95% CI | KL | top1 | PPL ratio | state rel-RMS |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in quality.iterrows():
        ci = (f"[{r.delta_nll_ci95_lo:.2f}, {r.delta_nll_ci95_hi:.2f}]"
              if np.isfinite(r.delta_nll_ci95_lo) else "n/a")
        lines.append(
            f"| `{r.variant}` | {r.executed_blocks} | {r.effective_depth} | "
            f"{r.delta_nll_mean:.3f} | {ci} | {r.kl_mean:.3f} | "
            f"{r.top1_mean:.3f} | {r.ppl_ratio:.2f}x | {r.state_rel_rms_mean:.3f} |")
    lines += ["", "`blocks` = executed decoder blocks (real compute); "
                  "`depth` = sequential stages (critical path). "
                  "Neither is a latency measurement."]
    if thresholds is not None and not thresholds.empty:
        lines += ["", "## minimum effective depth at a ΔNLL budget", "",
                  "| ΔNLL budget | variant | depth | blocks | ΔNLL |",
                  "|---:|---|---:|---:|---:|"]
        for _, r in thresholds.iterrows():
            if r.variant == "NOT REACHED":
                lines.append(f"| {r.delta_nll_threshold:g} | not reached | — | — | — |")
                continue
            lines.append(f"| {r.delta_nll_threshold:g} | `{r.variant}` | "
                         f"{int(r.effective_depth)} | {int(r.executed_blocks)} | "
                         f"{r.delta_nll_mean:.3f} |")
    if matched is not None and not matched.empty:
        lines += ["", "## depth-matched comparison against deletion", "",
                  "| depth | parallel | blocks | ΔNLL | delete | blocks | ΔNLL | advantage |",
                  "|---:|---|---:|---:|---|---:|---:|---:|"]
        for _, r in matched.iterrows():
            lines.append(f"| {int(r.effective_depth)} | `{r.variant}` | "
                         f"{int(r.blocks_parallel)} | {r.delta_nll_parallel:.3f} | "
                         f"`{r.delete_variant}` | {int(r.blocks_delete)} | "
                         f"{r.delta_nll_delete:.3f} | {r.advantage_nats:+.3f} |")
    return "\n".join(lines) + "\n"


def spectrum(mat: np.ndarray) -> dict:
    ev = np.maximum(np.linalg.eigvalsh(0.5 * (mat + mat.T))[::-1], 0.0)
    total = float(ev.sum())
    ratio = ev / total if total > 0 else ev
    cum = np.cumsum(ratio)
    return {
        "eigenvalues": ev.tolist(),
        "explained_ratio": ratio.tolist(),
        "cumulative_ratio": cum.tolist(),
        "participation_ratio": float(total ** 2 / max(float((ev ** 2).sum()), 1e-30)),
        "effective_rank_90pct": int(np.searchsorted(cum, 0.90) + 1) if total > 0 else 0,
        "trace": total,
    }


def analyze_gram(run: Path) -> None:
    """Derive Gram spectra from the raw matrices written by measure.py.

    The raw spectrum mixes magnitude with direction, because layers with large
    corrections dominate. The normalized correlation spectrum removes that and
    is the one to quote for "are the corrections concentrated in few
    directions". Neither proves a small network can predict them.
    """
    for f in sorted(run.glob("gram_*.npy")):
        G = np.load(f).astype(np.float64)
        raw = spectrum(G)
        d = np.sqrt(np.maximum(np.diag(G), 0.0))
        corr = G / np.outer(d + 1e-30, d + 1e-30)
        norm = spectrum(corr)
        trace = max(raw["trace"], 1e-30)
        out = {
            "n_layers": int(G.shape[0]),
            "raw": raw,
            "normalized_correlation": norm,
            "layer_energy_share": (np.diag(G) / trace).tolist(),
            "note": "Correction Gram G_ij = sum over tokens/features of <c_i, c_j>, "
                    "with c_i taken at the teacher state of the group that owns "
                    "layer i. The normalized correlation spectrum is scale-free. "
                    "Low rank shows direction redundancy only; it does NOT show "
                    "that a small network can predict the corrections.",
        }
        (run / f"{f.stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def pareto_tables(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frontier, threshold-minimum-depth, and depth-matched deletion comparison."""
    df = q[q.accum == "fp32"].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.sort_values(["effective_depth", "delta_nll_mean"])
    rows, best = [], np.inf
    for _, r in df.iterrows():
        if r.delta_nll_mean < best - 1e-12:
            best = r.delta_nll_mean
            rows.append(r)
    frontier = pd.DataFrame(rows)[
        ["variant", "kind", "executed_blocks", "effective_depth",
         "delta_nll_mean", "kl_mean", "top1_mean", "ppl_ratio"]].reset_index(drop=True)

    thr_rows = []
    for t in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
        ok = df[df.delta_nll_mean <= t]
        if ok.empty:
            thr_rows.append({"delta_nll_threshold": t, "variant": "NOT REACHED",
                             "effective_depth": np.nan, "executed_blocks": np.nan,
                             "delta_nll_mean": np.nan})
            continue
        b = ok.sort_values(["effective_depth", "delta_nll_mean"]).iloc[0]
        thr_rows.append({"delta_nll_threshold": t, "variant": b.variant,
                         "effective_depth": int(b.effective_depth),
                         "executed_blocks": int(b.executed_blocks),
                         "delta_nll_mean": float(b.delta_nll_mean)})
    thresholds = pd.DataFrame(thr_rows)

    dele = df[df.kind == "delete"].drop_duplicates("effective_depth")
    dele = dele.set_index("effective_depth")
    match_rows = []
    for _, r in df[df.kind == "parallel"].iterrows():
        d = int(r.effective_depth)
        if d not in dele.index:
            continue
        dr = dele.loc[d]
        match_rows.append({
            "effective_depth": d,
            "variant": r.variant,
            "blocks_parallel": int(r.executed_blocks),
            "delta_nll_parallel": float(r.delta_nll_mean),
            "delete_variant": dr.variant,
            "blocks_delete": int(dr.executed_blocks),
            "delta_nll_delete": float(dr.delta_nll_mean),
            "advantage_nats": float(dr.delta_nll_mean - r.delta_nll_mean),
        })
    matched = pd.DataFrame(match_rows).sort_values(
        ["effective_depth", "delta_nll_parallel"]) if match_rows else pd.DataFrame()
    return frontier, thresholds, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/stage1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-manifest", type=Path, default=Path("data/data_manifest.json"))
    args = ap.parse_args()
    out = args.out or args.input
    out.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.input / "samples.jsonl")
    groups = read_jsonl(args.input / "groups.jsonl")
    layers = read_jsonl(args.input / "layers.jsonl")
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = None
    dm_path = args.data_manifest
    if dm_path.exists():
        data_manifest = json.loads(dm_path.read_text(encoding="utf-8"))

    quality = quality_table(samples)
    gdf = group_table(groups)
    ldf = layer_table(layers)
    analyze_gram(out)
    frontier, thresholds, matched = pareto_tables(quality)

    quality.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    gdf.to_csv(out / "group_summary.csv", index=False, float_format="%.6g")
    ldf.to_csv(out / "layer_summary.csv", index=False, float_format="%.6g")
    frontier.to_csv(out / "pareto_frontier.csv", index=False, float_format="%.6g")
    thresholds.to_csv(out / "threshold_depth.csv", index=False, float_format="%.6g")
    matched.to_csv(out / "matched_depth.csv", index=False, float_format="%.6g")

    meta = {**manifest, "n_predicted_tokens": (data_manifest or {}).get("n_predicted_tokens")}
    if data_manifest:
        meta["data_revision"] = data_manifest.get("dataset_revision")
        meta["model_revision"] = data_manifest.get("model_revision")
    (out / "summary.md").write_text(markdown(quality, meta, thresholds, matched),
                                    encoding="utf-8")

    (out / "summary.json").write_text(json.dumps({
        "meta": meta,
        "quality": quality.to_dict(orient="records"),
        "groups": gdf.to_dict(orient="records"),
        "layers": ldf.to_dict(orient="records"),
        "frontier": frontier.to_dict(orient="records"),
        "thresholds": thresholds.to_dict(orient="records"),
        "matched_depth": matched.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}/summary.{{csv,md,json}}, group_summary.csv, layer_summary.csv")
    print(quality[["variant", "executed_blocks", "effective_depth",
                   "delta_nll_mean", "kl_mean", "top1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
