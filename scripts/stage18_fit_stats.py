"""Stage 1.8 A 系列的逐样本配对检验。

`fit` 阶段只存聚合的 rel_err / cosine，但 A3 vs A2 的差距只有 0.02 量级，均值差
判断不了（Stage 1.7 的教训：B3-B2 只有 0.01 nats，配对 CI 才说明它落在噪声内）。

本脚本加载 checkpoint，在每个窗口上单独算 rel_err / cosine，然后做逐样本配对
bootstrap。窗口是独立的文本，配对成立：同一个窗口喂给两个模型。

用法
----
    python scripts/stage18_fit_stats.py --root results/stage1_8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_serial import build_from_checkpoint  # noqa: E402


def load_split(path: Path):
    h3 = np.load(path / "h3.npy", mmap_mode="r")
    tgt = np.load(path / "h11T.npy", mmap_mode="r")
    n = h3.shape[0]
    return h3[:n], tgt[:n]


def per_sample(unit, h3, tgt, batch: int = 8) -> dict[str, list[float]]:
    """rel_err / cosine / ||pred|| / ||target|| for every window."""
    rel, cos, pr, tr = [], [], [], []
    n, T, _ = h3.shape
    for i in range(0, n, batch):
        a = mx.array(np.asarray(h3[i:i + batch], dtype=np.float32))
        t = mx.array(np.asarray(tgt[i:i + batch], dtype=np.float32)) - a
        p = unit(a, "causal")
        mx.eval(p)
        pn = np.asarray(p, dtype=np.float32)                # (b, T, D)
        tn = np.asarray(t, dtype=np.float32)
        for b in range(pn.shape[0]):
            x, y = pn[b].reshape(-1), tn[b].reshape(-1)
            num = float(np.sqrt(((x - y) ** 2).mean()))
            den = max(float(np.sqrt((y ** 2).mean())), 1e-9)
            rel.append(num / den)
            cos.append(float((x * y).sum()
                             / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
            pr.append(float(np.sqrt((x ** 2).mean())))
            tr.append(float(np.sqrt((y ** 2).mean())))
    return {"rel_err": rel, "cosine": cos, "pred_rms": pr, "target_rms": tr}


def paired(a: list[float], b: list[float], n_boot: int = 20000,
           seed: int = 0) -> dict:
    x, y = np.array(a), np.array(b)
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(lo), float(hi)],
            "frac_positive": float((d > 0).mean()),
            "significant_95": bool(lo > 0 or hi < 0)}


# (label, A, B); d = metric(A) - metric(B). rel_err's d > 0 means B is the
# better fit; cosine's d > 0 means A is better.
CONTRASTS = [
    ("A3(inherited) vs A2(random) full", "a3_inherited", "a2_random"),
    ("A3(inherited) vs A2(random) short", "a3_short", "a2_short"),
    ("A3(inherited) vs A1(serial MLP) full", "a3_inherited", "a1_serial"),
    ("A1(serial MLP) vs A2(random) full", "a1_serial", "a2_random"),
    ("A1(serial MLP) vs A3(inherited) short", "a1_short", "a3_short"),
    ("A2 short vs A2 full", "a2_short", "a2_random"),
    ("A3 short vs A3 full", "a3_short", "a3_inherited"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_8")
    ap.add_argument("--splits", nargs="*", default=["cl_val"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "fit_stats.json"

    units = {}
    for tag in ("a1_serial", "a1_short", "a2_random", "a2_short",
                "a3_inherited", "a3_short"):
        p = root / "fit" / f"{tag}.npz"
        if p.exists():
            units[tag] = build_from_checkpoint(p)
    print(f"loaded units: {sorted(units)}")

    results = {}
    for split in args.splits:
        sdir = root / split
        if not (sdir / "h3.npy").exists():
            print(f"skip {split}: cache missing")
            continue
        h3, tgt = load_split(sdir)
        print(f"{split}: {h3.shape[0]} windows, {h3.shape[1]} tokens")
        M = {tag: per_sample(u, h3, tgt) for tag, u in units.items()}
        rows = []
        for label, ta, tb in CONTRASTS:
            if ta not in M or tb not in M:
                continue
            for metric in ("rel_err", "cosine"):
                st = paired(M[ta][metric], M[tb][metric], seed=hash(
                    f"{label}{metric}") % (2 ** 31))
                st.update({"contrast": label, "a": ta, "b": tb,
                           "metric": metric, "split": split})
                rows.append(st)
                print(f"  {label:34s} [{metric:7s}] d={st['mean_diff']:+.4f} "
                      f"CI[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] "
                      f"frac+={st['frac_positive']:.2f} "
                      f"{'SIG' if st['significant_95'] else '   '}")
        # 每个模型的聚合值也留档，供 table.json 引用
        results[split] = {
            "n_windows": int(h3.shape[0]),
            "per_model": {
                tag: {
                    "rel_err_mean": round(float(np.mean(M[tag]["rel_err"])), 4),
                    "rel_err_median": round(float(np.median(M[tag]["rel_err"])), 4),
                    "cosine_mean": round(float(np.mean(M[tag]["cosine"])), 4),
                    "pred_rms_mean": round(float(np.mean(M[tag]["pred_rms"])), 4),
                    "target_rms_mean": round(float(np.mean(M[tag]["target_rms"])), 4),
                } for tag in sorted(M)
            },
            "paired": rows,
        }

    payload = {
        "stage": "1.8", "splits": args.splits,
        "definition": "d = metric(A) - metric(B)，逐窗口配对 bootstrap；"
                      "rel_err 的 d>0 表示 B 误差更小",
        "n_boot": 20000,
        "results": results,
        "note": "窗口级配对：同一个窗口喂给两个模型，衡量蒸馏目标的拟合质量，"
                "不是最终语言模型质量",
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
