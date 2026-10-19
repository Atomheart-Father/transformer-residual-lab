"""Stage 1.8 配对显著性检验。

Stage 1.8 的关键对比（fusion head 是否吃到了 tiny 特征的收益、预训练权重继承是否有
任何可测增益）差距都在 0.01 nats 量级，看两个均值差无法判断。这里对 eval 产出的
`samples.jsonl` 做逐样本配对 bootstrap：同一条文本、同一批 token，配对比均值可靠。

方法：按 sample_id 对齐，取 d_i = nll_i(A) - nll_i(B)，对样本有放回重采 20000 次
（A、B 同批重采以保持配对），报告 d 的均值、95% 区间和符号比例。不假设正态。

与 Stage 1.7 的区别：这里不硬编码对比表，而是自动发现 samples.jsonl 里的全部变体
并跑全配对，再额外跑一组点名对比（`oracle gap`、`fusion vs trunk`）。

用法
----
    python scripts/stage18_stats.py --root results/stage1_8
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

# 点名对比：d = nll(A) - nll(B)，d > 0 表示 B 更好
NAMED = [
    ("tiny-only fusion vs no compensation", "irr-a", "fuse_f_a3z"),
    ("trunk-only fusion vs no compensation", "irr-a", "fuse_f_trunk"),
    ("trunk+tiny(A3) fusion vs no compensation", "irr-a", "fuse_f_a3"),
    ("trunk+tiny(A2) fusion vs no compensation", "irr-a", "fuse_f_a2"),
    ("trunk+tiny(A1) fusion vs no compensation", "irr-a", "fuse_f_a1"),
    ("inherited raw substitution vs no compensation", "irr-a", "tiny_sub_a3_inherited"),
    ("tiny-only vs trunk-only", "fuse_f_trunk", "fuse_f_a3z"),
    ("trunk+tiny vs trunk-only", "fuse_f_trunk", "fuse_f_a3"),
    ("trunk-zero+tiny vs trunk+tiny", "fuse_f_a3", "fuse_f_a3z"),
    ("inherited-tiny fusion vs random-tiny fusion", "fuse_f_a3", "fuse_f_a2"),
    ("serial-MLP fusion vs inherited-tiny fusion", "fuse_f_a1", "fuse_f_a3"),
    ("oracle vs no compensation", "irr-a", "irr-a+oracle"),
    ("tiny-only fusion vs oracle", "fuse_f_a3z", "irr-a+oracle"),
]


def load_nll(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            for v, m in rec["variants"].items():
                out.setdefault(v, {})[sid] = m["delta_nll"]
    return out


def paired(a: dict[str, float], b: dict[str, float], n_boot: int = 20000,
           seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b))
    if not ids:
        raise KeyError("no shared sample ids")
    x = np.array([a[i] for i in ids])
    y = np.array([b[i] for i in ids])
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_paired": int(len(d)),
        "mean_diff": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "ci95": [float(lo), float(hi)],
        "frac_positive": float((d > 0).mean()),
        "n_boot": n_boot,
        "significant_95": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="eval run directories under --root; default: all with samples.jsonl")
    args = ap.parse_args()
    root = Path(args.root)
    # eval runs live in <root>/eval/<run>/samples.jsonl
    eval_root = root / "eval"
    if args.runs:
        runs = args.runs
    elif eval_root.exists():
        runs = sorted(p.name for p in eval_root.iterdir()
                      if p.is_dir() and (p / "samples.jsonl").exists())
    else:
        runs = sorted(p.name for p in root.iterdir()
                      if p.is_dir() and (p / "samples.jsonl").exists())
    out = Path(args.out) if args.out else root / "paired_stats.json"

    rows = []
    for run in runs:
        p = (root / "eval" / run / "samples.jsonl") if (root / "eval").exists() \
            else (root / run / "samples.jsonl")
        if not p.exists():
            rows.append({"run": run, "error": "samples.jsonl missing"})
            continue
        variants = load_nll(p)
        names = sorted(variants)
        print(f"\n=== {run}: {len(names)} variants, n={len(next(iter(variants.values())))} ===")

        for tag, va, vb in NAMED:
            if va not in variants or vb not in variants:
                continue
            st = paired(variants[va], variants[vb])
            st.update({"run": run, "contrast": tag,
                       "a": va, "b": vb, "kind": "named"})
            rows.append(st)
            print(f"  {st['contrast']:52s} d={st['mean_diff']:+.4f} "
                  f"CI[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] "
                  f"frac+={st['frac_positive']:.2f} "
                  f"{'SIG' if st['significant_95'] else '   '} n={st['n_paired']}")

        for va, vb in combinations(names, 2):
            if (va, vb) in NAMED:
                continue
            st = paired(variants[va], variants[vb], seed=1)
            st.update({"run": run, "contrast": f"{va} -> {vb}",
                       "a": va, "b": vb, "kind": "all-pairs"})
            rows.append(st)

    payload = {
        "stage": "1.8",
        "definition": "d = nll(A) - nll(B)，配对；d>0 表示 B 更好",
        "runs": runs,
        "n_named": len([r for r in rows if r.get("kind") == "named"]),
        "n_all_pairs": len([r for r in rows if r.get("kind") == "all-pairs"]),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}: {len(rows)} contrasts")


if __name__ == "__main__":
    main()
