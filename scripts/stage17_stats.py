"""Stage 1.7 配对显著性检验。

逐样本的 delta_nll 已经写在 samples.jsonl 里，因此同一评估集上的两种方法可以做
**配对**比较（同一条文本、同一批 token），比看两个均值的差要可靠得多，尤其是
B2 vs B3 这种只差 0.01 nats 的对比。

方法：按 sample_id 对齐，取 d_i = nll_i(A) - nll_i(B)，做 20000 次配对 bootstrap
（对样本有放回重采样，同时重采 A 和 B，保持配对），报告 d 的均值、95% 区间和
单侧符号比例。不假设正态。

用法
----
    python scripts/stage17_stats.py --root results/stage1_7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (标签, 评估集, 目录 A, 变体 A, 目录 B, 变体 B)
# d = nll(A) - nll(B)；d > 0 表示 B 更好（NLL 更低）。
CONTRASTS = [
    # ---- Experiment A：单站点 ----
    ("A: mean vs no-comp", "val", "A_val", "irr-a+mean", "A_val", "irr-a"),
    ("A: mean vs no-comp", "test", "A_test", "irr-a+mean", "A_test", "irr-a"),
    ("A: mean vs no-comp", "ood", "A_ood", "irr-a+mean", "A_ood", "irr-a"),
    ("A: shuffled(s11) vs no-comp", "val", "A_val", "irr-a+shuffled@11", "A_val", "irr-a"),
    ("A: shuffled(s11) vs no-comp", "test", "A_test", "irr-a+shuffled@11", "A_test", "irr-a"),
    ("A: shuffled(s11) vs no-comp", "ood", "A_ood", "irr-a+shuffled@11", "A_ood", "irr-a"),
    ("A: mlp_r32 vs mean", "val", "A_val", "irr-a+mlp_r32", "A_val", "irr-a+mean"),
    ("A: mlp_r32 vs mean", "test", "A_test", "irr-a+mlp_r32", "A_test", "irr-a+mean"),
    ("A: mlp_r32 vs mean", "ood", "A_ood", "irr-a+mlp_r32", "A_ood", "irr-a+mean"),
    ("A: mlp_r64 vs mlp_r32", "val", "A_val", "irr-a+mlp_r64", "A_val", "irr-a+mlp_r32"),
    ("A: mlp_r64 vs mlp_r32", "test", "A_test", "irr-a+mlp_r64", "A_test", "irr-a+mlp_r32"),
    ("A: mlp_r64 vs mlp_r32", "ood", "A_ood", "irr-a+mlp_r64", "A_ood", "irr-a+mlp_r32"),
    # ---- Experiment B：B1 / B2 / B3 ----
    ("B: B2(2site) vs B1(1site)", "val", "B_val_45_r2b", "irr-a+serial_b2_r64", "B_val_5e", "irr-a+serial_b1_r64e120"),
    ("B: B2(2site) vs B1(1site)", "test", "B_test_r2", "irr-a+serial_b2_r64", "B_test_5", "irr-a+serial_b1_r64"),
    ("B: B2(2site) vs B1(1site)", "ood", "B_ood_r2", "irr-a+serial_b2_r64", "B_ood_5", "irr-a+serial_b1_r64"),
    ("B: B3(2site,carry) vs B2(2site)", "val", "B_val_45_r2", "irr-a+serial_b3_r64", "B_val_45_r2b", "irr-a+serial_b2_r64"),
    ("B: B3(2site,carry) vs B2(2site)", "test", "B_test_r2", "irr-a+serial_b3_r64", "B_test_r2", "irr-a+serial_b2_r64"),
    ("B: B3(2site,carry) vs B2(2site)", "ood", "B_ood_r2", "irr-a+serial_b3_r64", "B_ood_r2", "irr-a+serial_b2_r64"),
    ("B: B3(2site,carry) vs B1(1site)", "val", "B_val_45_r2", "irr-a+serial_b3_r64", "B_val_5e", "irr-a+serial_b1_r64e120"),
    ("B: B3(2site,carry) vs B1(1site)", "test", "B_test_r2", "irr-a+serial_b3_r64", "B_test_5", "irr-a+serial_b1_r64"),
    ("B: B3(2site,carry) vs B1(1site)", "ood", "B_ood_r2", "irr-a+serial_b3_r64", "B_ood_5", "irr-a+serial_b1_r64"),
    # ---- 重采样的作用：同一模型结构，r0 数据 vs r2 数据 ----
    ("B: B3 r2 vs r0 (recollection)", "val", "B_val_45_r2", "irr-a+serial_b3_r64", "B_val_45", "irr-a+serial_b3_r64"),
    ("B: B2 r2 vs r0 (recollection)", "val", "B_val_45_r2b", "irr-a+serial_b2_r64", "B_val_45", "irr-a+serial_b2_r64"),
    # ---- 站点干扰：4 站点模型 vs 不补偿 ----
    ("B: 4-site B2 (apply 3,4,5) vs no-comp", "val", "B_sub_3,4,5", "irr-a+serial_b2_r32", "B_sub_3,4,5", "irr-a"),
]


def load_nll(root: Path, run: str, variant: str) -> dict[str, float]:
    out: dict[str, float] = {}
    p = root / run / "samples.jsonl"
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            v = rec["variants"].get(variant)
            if v is None:
                raise KeyError(f"{run}: variant {variant} missing for {rec['sample_id']}")
            out[rec["sample_id"]] = v["candidate_nll"]
    return out


def paired(a: dict[str, float], b: dict[str, float], n_boot: int = 20000, seed: int = 0):
    ids = sorted(set(a) & set(b))
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "paired_stats.json"

    cache: dict[tuple[str, str], dict[str, float]] = {}

    def get(run: str, variant: str) -> dict[str, float]:
        key = (run, variant)
        if key not in cache:
            cache[key] = load_nll(root, run, variant)
        return cache[key]

    rows = []
    for label, split, run_a, var_a, run_b, var_b in CONTRASTS:
        try:
            a, b = get(run_a, var_a), get(run_b, var_b)
        except (FileNotFoundError, KeyError) as e:
            rows.append({"contrast": label, "split": split, "error": str(e)})
            continue
        st = paired(a, b)
        st.update(
            {
                "contrast": label,
                "split": split,
                "a": f"{run_a}:{var_a}",
                "b": f"{run_b}:{var_b}",
                "significant_95": bool(st["ci95"][0] > 0 or st["ci95"][1] < 0),
            }
        )
        rows.append(st)
        print(
            f"{label:42s} [{split:4s}] d={st['mean_diff']:+.4f} "
            f"CI[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] "
            f"frac+={st['frac_positive']:.2f} n={st['n_paired']}"
        )

    with open(out, "w") as f:
        json.dump(
            {
                "stage": "1.7",
                "definition": "d = nll(A) - nll(B)，配对；d>0 表示 B 更好",
                "rows": rows,
            },
            f,
            indent=1,
            ensure_ascii=False,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
