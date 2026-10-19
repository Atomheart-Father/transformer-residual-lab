"""Stage 1.8 实测成本。

用户要求的第③类评价：参数、FLOPs/深度、实测延迟（median + p10/p90）、峰值内存、
融合成本，并区分独立执行与接入后总成本。成本必须实测，不能用 FLOPs 折算
（Stage 1.7 的教训：rank-32 的 FLOPs 只占单 block 的 0.61%，实测是 13.5%）。

除了三个 tiny 单元，还测：
  * 原始教师 block 组 [3..10]（8 个 block 串行）作为参照基线
  * 每个 fusion head 的前向
  * Stage 1.7 的串行单元（如果本地有 checkpoint）作为跨阶段对照

用法
----
    python scripts/stage18_bench.py --root results/stage1_8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

PINNED_MODEL_REPO = "mlx-community/Qwen3-0.6B-Base-bf16"
PINNED_MODEL_REVISION = "e8e73ff38f0eb570eb57ed044ef3f95588c43bc8"

T = 256          # predicted tokens per window
D = 1024         # Qwen3-0.6B hidden
REPEAT = 60
WARMUP = 8


def timeit(fn, repeat: int = REPEAT, warmup: int = WARMUP) -> dict:
    for _ in range(warmup):
        fn()
    mx.reset_peak_memory()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    peak = mx.get_peak_memory()
    return {
        "median_ms": 1000.0 * float(np.median(ts)),
        "p10_ms": 1000.0 * float(np.percentile(ts, 10)),
        "p90_ms": 1000.0 * float(np.percentile(ts, 90)),
        "repeat": repeat,
        "peak_memory_mb": peak / 2 ** 20,
    }


def bench_unit(path: Path) -> dict:
    from tiny_serial import build_from_checkpoint
    unit = build_from_checkpoint(path)
    a = mx.random.normal((1, T, D)).astype(mx.float32)
    mx.eval(a)
    out = {"checkpoint": path.name, "kind": unit.kind,
           "hidden": unit.hidden, "n_layers": unit.n_layers,
           "n_params": unit.n_params(), "tokens": T,
           "note": "standalone forward, one call, 256 tokens"}
    out.update(timeit(lambda: mx.eval(unit(a, "causal"))))
    return out


def bench_fuse_head(path: Path) -> dict:
    from tiny_serial import FuseHead, load_tree, flat_to_tree
    f, meta = load_tree(path)
    head = FuseHead(meta["dim_in"], meta["hidden"], D)
    head.update(flat_to_tree(f, head))
    x = mx.random.normal((1, T, meta["dim_in"])).astype(mx.float32)
    mx.eval(x)
    out = {"checkpoint": path.name, "mode": meta["mode"],
           "n_params": int(sum(np.asarray(v).size for _, v in f.items())),
           "tokens": T, "note": "standalone head forward, 256 tokens"}
    out.update(timeit(lambda: mx.eval(head(x))))
    return out


def bench_serial_b3(path: Path) -> dict:
    """Re-time Stage 1.7's serial units for a like-for-like cost comparison.

    Different parameterisation from the Stage 1.8 units: a GRU-style recurrent
    cell carrying a 64-d state between 2 sites, closed by an MLP. Uses
    `serial_compensate` directly so the numbers are comparable to the Stage 1.7
    bench rather than a re-implementation.
    """
    from serial_compensate import load_serial
    comp = load_serial(path)
    sites = [int(s) for s in str(comp.meta.get("sites", "")).split(",") if s != ""]
    a = mx.random.normal((1, T, D)).astype(mx.float32)
    mx.eval(a)

    def call():
        s = comp.zero_state(1, T, mx)
        for si in sites:
            c, s = comp.step_mx(si, a, s, mx)
            mx.eval(c)
        mx.eval(s)

    out = {"checkpoint": path.name, "kind": "serial",
           "sites": comp.meta.get("sites"), "hidden": comp.hidden,
           "rank": comp.rank, "n_params": comp.n_params, "tokens": T,
           "n_sites": len(sites),
           "note": "Stage 1.7 serial unit: %d site(s), %d-d state, "
                   "flops come from the Stage 1.7 bench" % (len(sites), comp.hidden)}
    out.update(timeit(call))
    return out


def bench_teacher_group() -> dict:
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from serial_compensate import run_group
    model, _, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                       return_config=True)
    model.eval()
    ids = mx.array([1] + list(range(1000, 1000 + T)), dtype=mx.int32)[None, :]
    mask = create_attention_mask(mx.zeros((1, T, D)), None)
    h = mx.random.normal((1, T, D)).astype(mx.float32)
    mx.eval(h, mask)
    group = [3, 4, 5, 6, 7, 8, 9, 10]
    out = {"blocks": group, "n_blocks": len(group), "tokens": T,
           "note": "the raw teacher group this stage replaces; same input h"}
    out.update(timeit(lambda: mx.eval(run_group(model, h, group, mask, mx))))
    n_params = 0
    for i in group:
        for _, v in _flat_iter(model.model.layers[i].parameters()):
            n_params += np.asarray(_to_np(v)).size
    out["n_params"] = int(n_params)
    return out


def _flat_iter(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat_iter(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat_iter(v, f"{prefix}.{i}")
    else:
        yield prefix, tree


def _to_np(x) -> np.ndarray:
    return np.asarray(mx.asarray(x).astype(mx.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/stage1_8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-group", action="store_true",
                    help="also time the raw teacher group (loads the 0.6B model)")
    ap.add_argument("--stage17-units", nargs="*", default=[],
                    help="optional Stage 1.7 serial-unit npz files to re-time")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "bench" / "bench_s18.json"

    rows = {"units": [], "fuse_heads": [], "references": {}}

    for tag in ("a1_serial", "a2_random", "a3_inherited"):
        p = root / "fit" / f"{tag}.npz"
        if p.exists():
            rows["units"].append(bench_unit(p))

    for p in sorted((root / "fuse").glob("f_*.npz")):
        rows["fuse_heads"].append(bench_fuse_head(p))

    if args.with_group:
        rows["references"]["teacher_group_3_10"] = bench_teacher_group()
        # 单 block 参照
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask
        from serial_compensate import run_group
        model, _, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION,
                           return_config=True)
        model.eval()
        h = mx.random.normal((1, T, D)).astype(mx.float32)
        mask = create_attention_mask(mx.zeros((1, T, D)), None)
        mx.eval(h, mask)
        single = {"blocks": [3], "n_blocks": 1, "tokens": T,
                  "note": "one raw teacher block, same input h"}
        single.update(timeit(lambda: mx.eval(run_group(model, h, [3], mask, mx))))
        n_p = 0
        for _, v in _flat_iter(model.model.layers[3].parameters()):
            n_p += _to_np(v).size
        single["n_params"] = int(n_p)
        rows["references"]["teacher_block_3"] = single

    for p in args.stage17_units:
        pp = Path(p)
        if pp.exists():
            rows["references"][f"stage17_{pp.stem}"] = bench_serial_b3(pp)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    for u in rows["units"]:
        print(f"unit {u['checkpoint']:18s} {u['kind']:5s} params={u['n_params']:>9,} "
              f"median={u['median_ms']:6.3f} ms p10={u['p10_ms']:.3f} "
              f"p90={u['p90_ms']:.3f} peak={u['peak_memory_mb']:.2f} MB")
    for f in rows["fuse_heads"]:
        print(f"fuse {f['checkpoint']:18s} {f['mode']:11s} params={f['n_params']:>9,} "
              f"median={f['median_ms']:6.3f} ms")
    for k, r in rows["references"].items():
        print(f"ref  {k:28s} params={r.get('n_params', 0):>11,} "
              f"median={r['median_ms']:6.3f} ms p10={r['p10_ms']:.3f} "
              f"p90={r['p90_ms']:.3f} peak={r['peak_memory_mb']:.2f} MB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
