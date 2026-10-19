"""Stage 1.7 统一结果表。

把分散在 results/stage1_7/<run>/eval_summary.json 的评估结果、fit_*.json 的训练
统计、bench_*.json 的实测开销合并成一张表（table.json），供 REPORT.md 直接引用。

设计原则
--------
1. 不做任何重新计算或"修正"：所有数字都来自磁盘上的记录，本脚本只负责汇总与
   计算 well-defined 的派生量（gap_closure / recovery）。
2. gap_closure 在 A 系列（单站点 group 5）由本脚本从 base 与 oracle 推出，因为
   compensate.py 的 aggregate 里没有这个字段；B 系列（serial_compensate.py）
   自带该字段，直接采用，避免两处定义漂移。
3. 缺失的键一律跳过并记入 `missing`，不静默补值。

用法
----
    python scripts/make_stage17_table.py --root results/stage1_7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (run 目录, 评估集标签, 方法标签, compensator 键, 权重文件, 权重所在目录)
# 评估集：val = data257/validation.jsonl, test = data257/test_v2.jsonl,
#         ood = data257/ood_imdb.jsonl
ROWS = [
    # ---------- Experiment A：单站点（group 5），固定偏移 vs 输入依赖 ----------
    ("A_val", "val", "A0 no compensation", None, None, None),
    ("A_val", "val", "A1 mean offset", "irr-a+mean", None, None),
    ("A_val", "val", "A2 shuffled s11", "irr-a+shuffled@11", None, None),
    ("A_val", "val", "A2 shuffled s12", "irr-a+shuffled@12", None, None),
    ("A_val", "val", "A2 shuffled s13", "irr-a+shuffled@13", None, None),
    ("A_val", "val", "A3 mlp r32", "irr-a+mlp_r32", None, None),
    ("A_val", "val", "A4 mlp r64", "irr-a+mlp_r64", None, None),
    ("A_val", "val", "oracle g5", "irr-a+oracle", None, None),
    ("A_test", "test", "A0 no compensation", None, None, None),
    ("A_test", "test", "A1 mean offset", "irr-a+mean", None, None),
    ("A_test", "test", "A2 shuffled s11", "irr-a+shuffled@11", None, None),
    ("A_test", "test", "A2 shuffled s12", "irr-a+shuffled@12", None, None),
    ("A_test", "test", "A2 shuffled s13", "irr-a+shuffled@13", None, None),
    ("A_test", "test", "A3 mlp r32", "irr-a+mlp_r32", None, None),
    ("A_test", "test", "A4 mlp r64", "irr-a+mlp_r64", None, None),
    ("A_test", "test", "oracle g5", "irr-a+oracle", None, None),
    ("A_ood", "ood", "A0 no compensation", None, None, None),
    ("A_ood", "ood", "A1 mean offset", "irr-a+mean", None, None),
    ("A_ood", "ood", "A2 shuffled s11", "irr-a+shuffled@11", None, None),
    ("A_ood", "ood", "A2 shuffled s12", "irr-a+shuffled@12", None, None),
    ("A_ood", "ood", "A2 shuffled s13", "irr-a+shuffled@13", None, None),
    ("A_ood", "ood", "A3 mlp r32", "irr-a+mlp_r32", None, None),
    ("A_ood", "ood", "A4 mlp r64", "irr-a+mlp_r64", None, None),
    ("A_ood", "ood", "oracle g5", "irr-a+oracle", None, None),
    # ---------- Experiment B：串行 / 共享 / 状态 ----------
    # 单站点 B1（同架构、1 site、r64、40 epoch；e120 为补训对照）
    ("B_val_5", "val", "B1 single-site r64", "irr-a+serial_b1_r64", "B_r0_5/serial_b1_r64.npz", "B_r0_5"),
    ("B_test_5", "test", "B1 single-site r64", "irr-a+serial_b1_r64", "B_r0_5/serial_b1_r64.npz", "B_r0_5"),
    ("B_ood_5", "ood", "B1 single-site r64", "irr-a+serial_b1_r64", "B_r0_5/serial_b1_r64.npz", "B_r0_5"),
    # 4 站点（3,4,5,6）未做重采样的 B2/B3：site interference 的证据
    ("B_val", "val", "B2 4-site r64 (no recollect)", "irr-a+serial_b2_r64", "B_r0/serial_b2_r64.npz", "B_r0"),
    ("B_val", "val", "B3 4-site r64 (no recollect)", "irr-a+serial_b3_r64", "B_r0/serial_b3_r64.npz", "B_r0"),
    # 2 站点（4,5）第 0 轮：用未补偿分布采集的数据训练
    ("B_val_45", "val", "B2 2-site r16 r0", "irr-a+serial_b2_r16", "B_r0_45/serial_b2_r16.npz", "B_r0_45"),
    ("B_val_45", "val", "B2 2-site r32 r0", "irr-a+serial_b2_r32", "B_r0_45/serial_b2_r32.npz", "B_r0_45"),
    ("B_val_45", "val", "B2 2-site r64 r0", "irr-a+serial_b2_r64", "B_r0_45/serial_b2_r64.npz", "B_r0_45"),
    ("B_val_45", "val", "B3 2-site r16 r0", "irr-a+serial_b3_r16", "B_r0_45/serial_b3_r16.npz", "B_r0_45"),
    ("B_val_45", "val", "B3 2-site r32 r0", "irr-a+serial_b3_r32", "B_r0_45/serial_b3_r32.npz", "B_r0_45"),
    ("B_val_45", "val", "B3 2-site r64 r0", "irr-a+serial_b3_r64", "B_r0_45/serial_b3_r64.npz", "B_r0_45"),
    # 第 1 轮：用"补偿已生效"的分布重采数据
    ("B_val_45_r1", "val", "B2 2-site r64 r1", "irr-a+serial_b2_r64", "B_r1_45/serial_b2_r64.npz", "B_r1_45"),
    ("B_val_45_r1", "val", "B3 2-site r64 r1", "irr-a+serial_b3_r64", "B_r1_45/serial_b3_r64.npz", "B_r1_45"),
    # 第 2 轮：收敛
    ("B_val_45_r2b", "val", "B2 2-site r64 r2", "irr-a+serial_b2_r64", "B_r2_45/serial_b2_r64.npz", "B_r2_45"),
    ("B_val_45_r2", "val", "B3 2-site r64 r2", "irr-a+serial_b3_r64", "B_r2_45/serial_b3_r64.npz", "B_r2_45"),
    # 独立测试集（test_v2 / IMDB OOD）——只用第 2 轮权重，不做任何再选择
    ("B_test_r2", "test", "B2 2-site r64 r2", "irr-a+serial_b2_r64", "B_r2_45/serial_b2_r64.npz", "B_r2_45"),
    ("B_test_r2", "test", "B3 2-site r64 r2", "irr-a+serial_b3_r64", "B_r2_45/serial_b3_r64.npz", "B_r2_45"),
    ("B_ood_r2", "ood", "B2 2-site r64 r2", "irr-a+serial_b2_r64", "B_r2_45/serial_b2_r64.npz", "B_r2_45"),
    ("B_ood_r2", "ood", "B3 2-site r64 r2", "irr-a+serial_b3_r64", "B_r2_45/serial_b3_r64.npz", "B_r2_45"),
]

# 站点干扰消融：同一个 4 站点 r32 模型（B_r0/serial_b2_r32.npz），只改变
# --apply-sites，用于给每个站点定价。
SITE_ABLATION = [
    ("B_sub_3", "3"),
    ("B_sub_4", "4"),
    ("B_sub_5", "5"),
    ("B_sub_6", "6"),
    ("B_sub_3,4", "3,4"),
    ("B_sub_4,5", "4,5"),
    ("B_sub_5,6", "5,6"),
    ("B_sub_3,4,5", "3,4,5"),
    ("B_sub2_5", "5 (per-site r32)"),
    ("B_sub2_4,5", "4,5 (per-site r32)"),
    ("B_sub2_5,6", "5,6 (per-site r32)"),
]

# 对**最终** 2 站点 B3 r64 模型做单站点定价（同一个权重，只改 --apply-sites）。
# 这一组比 SITE_ABLATION 更有意义：SITE_ABLATION 用的是第 0 轮、4 站点模型。
FINAL_SITE_PRICING = [
    ("B_site4_r2", "4"),
    ("B_site5_r2", "5"),
    ("B_val_45_r2", "4,5"),
]

BENCHES = [
    "bench_serial_b1_r64.json",
    "bench_serial_b2_r64.json",
    "bench_serial_b3_r64.json",
    "bench_serial_b3_r32.json",
]


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def gap_from_aggregate(agg: dict, name: str) -> float | None:
    """A 系列没有 gap_closure，用 base 与单站点 oracle 现场推。"""
    if name in agg and "gap_closure" in agg[name]:
        return agg[name]["gap_closure"]
    base = agg["irr-a"]["delta_nll_mean"]
    if "irr-a+oracle" not in agg:
        return None
    denom = base - agg["irr-a+oracle"]["delta_nll_mean"]
    if abs(denom) < 1e-9:
        return None
    return (base - agg[name]["delta_nll_mean"]) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "table.json"

    cache: dict[str, dict] = {}
    rows, missing = [], []

    def summary(run: str) -> dict | None:
        if run not in cache:
            p = root / run / "eval_summary.json"
            cache[run] = load(p) if p.exists() else None
        return cache[run]

    for run, split, label, comp_key, wfile, wdir in ROWS:
        s = summary(run)
        if s is None:
            missing.append({"run": run, "reason": "no eval_summary.json"})
            continue
        agg = s["aggregate"]
        key = comp_key if comp_key else "irr-a"
        if key not in agg:
            missing.append({"run": run, "reason": f"no aggregate key {key}"})
            continue
        a = agg[key]
        row = {
            "family": "A" if run.startswith("A_") else "B",
            "method": label,
            "split": split,
            "run": run,
            "n": a.get("n"),
            "sites": s.get("sites"),
            "delta_nll": a["delta_nll_mean"],
            "delta_nll_std": a.get("delta_nll_std"),
            "kl": a.get("kl_mean"),
            "top1": a.get("top1_mean"),
            "gap_closure": gap_from_aggregate(agg, key),
            "recovery_vs_oracle": a.get("recovery_vs_oracle"),
            "recovery_vs_oracle_ex_last": a.get("recovery_vs_oracle_ex_last"),
            "exit_rel_rms_mean": a.get("exit_rel_rms_mean"),
            "weights": None,
        }
        # 参数量 / 训练统计：从 fit 记录取，权重文件仅记录相对路径
        if wfile and wdir:
            row["weights"] = f"results/stage1_7/{wfile}"
            tag = Path(wfile).stem.replace("serial_", "")
            fitp = root / wdir / f"fit_{tag}.json"
            if fitp.exists():
                f = load(fitp)
                row["n_params"] = f.get("n_params")
                row["flops_per_token"] = (4 * f["n_params"]) if f.get("n_params") else None
                row["hidden"] = f.get("hidden")
                row["rank"] = f.get("rank")
                row["carry_state"] = f.get("carry_state")
                row["per_site"] = f.get("per_site")
                row["epochs"] = len(f.get("history", []))
                row["final_train_mse"] = f.get("final_train_mse")
                row["final_val_mse"] = f.get("final_val_mse")
                row["per_site_val"] = f.get("per_site_val")
                row["site_weight"] = f.get("site_weight")
            else:
                missing.append({"run": run, "reason": f"no {fitp.name}"})
        rows.append(row)

    # 站点干扰消融
    abl = []
    for run, sites in SITE_ABLATION:
        s = summary(run)
        if s is None:
            missing.append({"run": run, "reason": "no eval_summary.json"})
            continue
        agg = s["aggregate"]
        cand = [k for k in agg if k.startswith("irr-a+serial")]
        if not cand:
            missing.append({"run": run, "reason": "no serial compensator in aggregate"})
            continue
        key = cand[0]
        a = agg[key]
        abl.append(
            {
                "apply_sites": sites,
                "model": key.replace("irr-a+", ""),
                "run": run,
                "delta_nll": a["delta_nll_mean"],
                "gap_closure": a.get("gap_closure"),
                "n": a.get("n"),
            }
        )

    # 最终模型的站点定价
    pricing = []
    for run, sites in FINAL_SITE_PRICING:
        s = summary(run)
        if s is None:
            missing.append({"run": run, "reason": "no eval_summary.json"})
            continue
        agg = s["aggregate"]
        cand = [k for k in agg if k.startswith("irr-a+serial")]
        if not cand:
            missing.append({"run": run, "reason": "no serial compensator in aggregate"})
            continue
        a = agg[cand[0]]
        pricing.append(
            {
                "apply_sites": sites,
                "model": cand[0].replace("irr-a+", ""),
                "run": run,
                "delta_nll": a["delta_nll_mean"],
                "gap_closure": a.get("gap_closure"),
                "recovery_vs_oracle_ex_last": a.get("recovery_vs_oracle_ex_last"),
                "n": a.get("n"),
            }
        )

    # 参考量：每个评估集上的 no-comp / oracle，便于跨表核对
    refs = {}
    for run, split in [
        ("B_val_45_r2", "val"),
        ("B_test_r2", "test"),
        ("B_ood_r2", "ood"),
    ]:
        s = summary(run)
        if s is None:
            continue
        agg = s["aggregate"]
        refs[split] = {
            "run": run,
            "n": agg["irr-a"]["n"],
            "no_comp_delta_nll": agg["irr-a"]["delta_nll_mean"],
            "oracle_all_delta_nll": agg["irr-a+oracle"]["delta_nll_mean"],
            "oracle_ex_last_delta_nll": agg["irr-a+oracle_ex_last"]["delta_nll_mean"],
            "serial_delta_nll": agg["serial"]["delta_nll_mean"],
            "gap_all": agg["irr-a"]["delta_nll_mean"] - agg["irr-a+oracle"]["delta_nll_mean"],
            "gap_ex_last": agg["irr-a"]["delta_nll_mean"] - agg["irr-a+oracle_ex_last"]["delta_nll_mean"],
        }

    bench = {}
    for b in BENCHES:
        p = root / b
        if p.exists():
            d = load(p)
            bench[d["compensator"].split("/")[-1]] = {
                "n_params": d["n_params"],
                "n_sites": d["n_sites"],
                "carry_state": d["carry_state"],
                "per_site_ms": d["per_site_ms"],
                "per_site_total_ms": d["per_site_total_ms"],
                "chain_ms": d["chain_ms"],
                "flops_per_token": d["flops_per_token"],
                "peak_memory_mb": d["peak_memory_mb"],
                "tokens": d["tokens"],
                "repeat": d["repeat"],
            }
        else:
            missing.append({"run": b, "reason": "no benchmark"})

    table = {
        "stage": "1.7",
        "note": (
            "所有 delta_nll 均为相对串行教师模型的平均 NLL 差（nats/token），"
            "越小越好；gap_closure = (base - variant) / (base - oracle)，"
            "oracle 的站点集合随行标注；recovery_vs_oracle_ex_last 只在 4 站点族中可跨方法比较。"
        ),
        "rows": rows,
        "site_ablation": abl,
        "final_site_pricing": pricing,
        "references": refs,
        "bench": bench,
        "missing": missing,
    }
    with open(out, "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(rows)} rows, {len(abl)} ablation rows, {len(missing)} missing")
    for m in missing:
        print("  missing:", m)


if __name__ == "__main__":
    main()
