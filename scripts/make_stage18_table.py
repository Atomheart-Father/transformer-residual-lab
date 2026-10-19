"""Stage 1.8 统一表生成器。

纪律（沿用 Stage 1.7）：报告里出现的每一个数字都必须由本脚本从 JSON 汇总得到，
图和表同源，禁止手抄。脚本只汇总不重算；缺键进 `missing`，不静默补 0。

输入
----
    results/stage1_8/fit/fit_*.json        教师状态误差（蒸馏目标）
    results/stage1_8/eval/*/eval_summary.json   接入并联主干后的 ΔNLL / KL / top-1
    results/stage1_8/bench/bench_*.json    实测成本
    results/stage1_8/paired_stats.json     配对 bootstrap

输出
----
    results/stage1_8/table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 架构行。compensator 是 Stage 1.8 A 系列里的 checkpoint 名字；
# variants 是接入并联主干后报告的自由运行变体。
ARCHES = [
    {"key": "a1", "name": "A1 串行 MLP（无注意力）",
     "compensator": "a1_serial", "kind": "mlp",
     "note": "同样宽度、同样残差形状，只去掉注意力"},
    {"key": "a2", "name": "A2 Tiny Transformer（随机初始化）",
     "compensator": "a2_random", "kind": "tiny",
     "note": "128 宽 / 4 深 / 1 头 / head_dim 128 / inter 512，RoPE+RMSNorm+GQA 布局与 Qwen3 一致"},
    {"key": "a3", "name": "A3 Tiny Transformer（原模型参数继承）",
     "compensator": "a3_inherited", "kind": "tiny",
     "note": "SliceGPT 式 PCA 旋转 + LLM-Pruner 神经元/头选择，与 A2 完全同结构"},
]

# 每行报告的变体：(eval 里出现的变体名, 表里的说明, fusion checkpoint 或 None)
VARIANTS = [
    ("tiny_sub", "Tiny 原始输出直接当修正", None),
    ("fuse_f_trunk", "仅并联主干（不用 Tiny）", "f_trunk"),
    ("fuse_f_a3", "主干 + Tiny(A3) 特征", "f_a3"),
    ("fuse_f_a3z", "主干置零 + Tiny(A3) 特征", "f_a3z"),
    ("fuse_f_a2", "主干 + Tiny(A2) 特征", "f_a2"),
    ("fuse_f_a1", "主干 + Tiny(A1) 特征", "f_a1"),
]

UNLIMITED_ARCH = {"a1": ["tiny_sub", "fuse_f_a1"],
                  "a2": ["tiny_sub", "fuse_f_a2"],
                  "a3": ["tiny_sub", "fuse_f_a3", "fuse_f_a3z", "fuse_f_trunk"]}


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fit_row(root: Path, arch: dict) -> dict:
    d = load_json(root / "fit" / f"fit_{arch['compensator']}.json")
    if d is None:
        return {"missing": f"fit_{arch['compensator']}.json"}
    b, a = d["before_training"]["val"], d["after_training"]["val"]
    return {
        "name": arch["name"], "kind": d["kind"],
        "init_source": d["init_source"],
        "n_params": d["n_params"],
        "flops_per_token": d["flops_per_token"],
        "n_train_windows": d["n_train_windows"],
        "n_train_tokens": d["n_train_tokens"],
        "epochs": d["epochs"], "seconds": d["seconds"],
        "before_rel_err": round(b["rel_err"], 4),
        "before_cosine": round(b["cosine"], 4),
        "before_pred_rms": round(b["pred_rms"], 4),
        "after_rel_err": round(a["rel_err"], 4),
        "after_cosine": round(a["cosine"], 4),
        "after_pred_rms": round(a["pred_rms"], 4),
        "target_rms": round(a["target_rms"], 4),
    }


def short_budget(root: Path) -> list[dict]:
    out = []
    for tag in ("a1_short", "a2_short", "a3_short"):
        d = load_json(root / "fit" / f"fit_{tag}.json")
        if d is None:
            continue
        out.append({
            "tag": d["tag"], "init_source": d["init_source"],
            "epochs": d["epochs"],
            "final_val_rel_err": round(d["after_training"]["val"]["rel_err"], 4),
            "final_val_cosine": round(d["after_training"]["val"]["cosine"], 4),
            "note": "scratch probe: same budget for both, tests whether "
                    "inheritance at least speeds up convergence",
        })
    return out


def eval_rows(root: Path, runs: list[str]) -> dict:
    out: dict = {}
    for run in runs:
        d = load_json(root / "eval" / run / "eval_summary.json")
        if d is None:
            out[run] = {"missing": f"eval/{run}/eval_summary.json"}
            continue
        agg = d["aggregate"]
        out[run] = {
            "input": d["input"], "n_samples": d["n_samples"],
            "plan": d["plan"], "group": d["group"],
            "tiny_ms_per_sample": d["tiny_ms_per_sample"],
            "warning": d.get("warning"),
            "variants": {},
        }
        for v, m in agg.items():
            out[run]["variants"][v] = {
                "n": m["n"],
                "delta_nll_mean": round(m["delta_nll_mean"], 4),
                "delta_nll_std": round(m["delta_nll_std"], 4),
                "kl_mean": round(m["kl_mean"], 5),
                "top1_mean": round(m["top1_mean"], 4),
                "gap_closure": round(m["gap_closure"], 4),
                "ppl_ratio": round(m["ppl_ratio"], 3),
            }
    return out


def bench_rows(root: Path) -> dict:
    out = {}
    for p in sorted((root / "bench").glob("bench_*.json")):
        d = json.loads(p.read_text())
        if "units" in d or "fuse_heads" in d:
            out["measured"] = d           # bench_s18.json: nested by role
        else:
            out[p.stem.replace("bench_", "")] = d
    return out


def fuse_rows(root: Path) -> dict:
    out = {}
    for p in sorted((root / "fuse").glob("fuse_*.json")):
        out[p.stem.replace("fuse_", "")] = json.loads(p.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "table.json"

    runs = [p.name for p in sorted((root / "eval").iterdir())
            if p.is_dir() and (p / "eval_summary.json").exists()]

    table = {
        "stage": "1.8",
        "site": 3, "group": [3, 4, 5, 6, 7, 8, 9, 10],
        "plan": "irr-a:0|1|2|3-10|11-19|20-25|26-27",
        "eval_runs": runs,
        "fit": [fit_row(root, a) for a in ARCHES],
        "short_budget": short_budget(root),
        "eval": eval_rows(root, runs),
        "fuse": fuse_rows(root),
        "bench": bench_rows(root),
    }

    paired = load_json(root / "paired_stats.json")
    if paired is not None:
        table["paired"] = paired
    fit_stats = load_json(root / "fit_stats.json")
    if fit_stats is not None:
        table["fit_stats"] = fit_stats

    missing = []
    for a in table["fit"]:
        if "missing" in a:
            missing.append(a["missing"])
    for r, d in table["eval"].items():
        if "missing" in d:
            missing.append(d["missing"])
    if missing:
        table["missing"] = missing
    else:
        table["missing"] = []

    out.write_text(json.dumps(table, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    print(f"  fit rows: {len(table['fit'])}  short probes: {len(table['short_budget'])}")
    print(f"  eval runs: {len(table['eval'])}  fuse heads: {len(table['fuse'])}"
          f"  bench rows: {len(table['bench'])}")
    if missing:
        print(f"  MISSING: {missing}")


if __name__ == "__main__":
    main()
