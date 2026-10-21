#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trlab_kaggle.ipynb 的 CPU-only 回归测试。

用法：
    python3 research/kaggle/test_trlab.py

不下载、不联网、不需要 GPU、不需要 llama.cpp。它做两件事：
  1. 把 notebook 里所有 code cell 抽出来在临时目录里 exec 一遍（保证不会 import 就炸）；
  2. 对 v0.1 暴露出的四类缺陷逐条断言（P0-B 算术、P0-C bf16、P1-F 分组、P1-G 未知类型），
     外加指纹失效与返回码检查。

所有断言都针对 notebook 里的**真实代码**，不是复制的副本 —— 所以改了 notebook 就跑得出来。
"""

import json
import os
import struct
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "trlab_kaggle.ipynb"


def load_namespace():
    """在临时工作目录里执行全部 code cell，返回命名空间。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_test_"))
    os.chdir(tmp)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ns = {"__name__": "__trlab_test__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    return ns


# ---------------------------------------------------------------- P0-B

def test_need_bytes_arithmetic(ns):
    """v0.1 写成 `size + 1 << 30`，实际是 `(size+1)<<30`，把需求夸大约 9.5 亿倍。"""
    f = ns["_est_need_bytes"]
    GiB = 1 << 30

    assert f(8 * GiB, 1.0) == 9 * GiB, f"8GiB+1GiB → {f(8 * GiB, 1.0)}"
    assert f(16 * GiB, 2.0) == 18 * GiB
    assert f(8 * GiB, 2.0) == 10 * GiB
    assert f(0, 0.0) == 0

    # 显式记录旧写法错在哪，防止有人"简化"回去
    size = 8 * GiB
    buggy = size + 1 << 30          # (size+1) << 30
    fixed = size + (1 << 30)
    assert buggy > fixed * 1e6, "旧写法应当离谱地大"
    assert fixed == f(size, 1.0)


def test_anchor_decision_t4_skips(ns):
    """8B f16 = 15.25 GiB 在 15.0 GiB 可用的 T4 上必须 skipped。"""
    d = ns["_anchor_decision"]
    GiB = 1 << 30
    r = d(int(15.25 * GiB), 16.0, "f16", allow_approx=False)
    assert r["status"] == "skipped", r
    assert r["fits_on_gpu"] is False
    assert r["required_gb"] < 100, f"需求被夸大了：{r['required_gb']} GiB"
    assert 17.0 < r["required_gb"] < 17.5, r["required_gb"]
    assert 14.9 < r["vram_usable_gb"] < 15.1
    assert r["tier"] == "high"


def test_anchor_decision_q8_is_approx(ns):
    """Q8_0 能装下，但属二次量化：默认拒绝，显式放行后标 approx_only。"""
    d = ns["_anchor_decision"]
    GiB = 1 << 30
    b = int(8.1 * GiB)

    r1 = d(b, 16.0, "q8_0", allow_approx=False)
    assert r1["fits_on_gpu"] is True, r1
    assert r1["tier"] == "approx"
    assert r1["status"] == "skipped", "默认必须拒绝 Q8 当锚点"
    assert "有效性" in r1["reason"]

    r2 = d(b, 16.0, "q8_0", allow_approx=True)
    assert r2["status"] == "ready"
    assert r2["validity_scope"] == "approx_only"


def test_anchor_decision_a100_ready(ns):
    """同一份 15.25 GiB 锚点在 40 GB 上可以跑，且是 high_precision。"""
    d = ns["_anchor_decision"]
    GiB = 1 << 30
    r = d(int(15.25 * GiB), 40.0, "f16", allow_approx=False)
    assert r["fits_on_gpu"] is True, r
    assert r["status"] == "ready"
    assert r["validity_scope"] == "high_precision"


# ---------------------------------------------------------------- P0-C

def test_bench_summary_bf16_error(ns):
    """v0.1 在 S9 里无条件索引 tb['bf16']['tflops']，T4 上直接 KeyError。"""
    f = ns["_bench_summary"]
    tb = {"fp16": {"tflops": 50.0},
          "bf16": {"error": "RuntimeError: not implemented for BFloat16"}}
    s = f(tb)                                    # 不能抛
    assert s["fp16_tflops"] == 50.0
    assert s["bf16_available"] is False
    assert s["bf16_tflops"] is None
    assert s["ratio_bf16_over_fp16"] is None
    assert "not implemented" in s["bf16_error"]


def test_bench_summary_degenerate_inputs(ns):
    f = ns["_bench_summary"]
    for tb in ({}, None, {"error": "no torch"}, {"cuda": False},
               {"fp16": {}, "bf16": {}}, {"fp16": {"tflops": 0}}):
        s = f(tb)                                # 全部不能抛
        assert s["ratio_bf16_over_fp16"] is None or tb.get("fp16", {}).get("tflops")
    s = f({"fp16": {"tflops": 40.0}, "bf16": {"tflops": 20.0}})
    assert s["ratio_bf16_over_fp16"] == 0.5


# ---------------------------------------------------------------- P1-G

def test_gguf_unknown_type_fails_closed(ns):
    """v0.1 把未知 type 当 8 bpw 正常出表；现在必须抛。"""
    write = ns["gguf_write_synth"]
    scan = ns["gguf_scan"]
    err = ns["UnknownGGMLType"]
    tmp = Path(ns["WORK"]) / "test_unknown.gguf"

    # writer 默认也是 strict 的，构造坏文件必须显式降级
    try:
        write(tmp, [("blk.0.bogus.weight", (256, 256), 200)])
        raise AssertionError("writer 对未知 type 也不该静默通过")
    except err:
        pass

    write(tmp, [("blk.0.bogus.weight", (256, 256), 200)], strict_types=False)
    try:
        try:
            scan(tmp)
            raise AssertionError("未知 type 没有抛 UnknownGGMLType")
        except err:
            pass

        # strict=False 时允许降级，但不产出字节、也不产出 bpw
        r = scan(tmp, strict_types=False)
        assert r["unknown_types"] == ["UNKNOWN"], r["unknown_types"]
        assert r["tensors"][0]["nbytes"] is None
        assert r["bpw_effective"] is None, f"0.0 bpw 也是假数：{r['bpw_effective']}"
    finally:
        if tmp.exists():
            tmp.unlink()


def test_gguf_roundtrip_bytes(ns):
    write = ns["gguf_write_synth"]
    scan = ns["gguf_scan"]
    types = ns["GGML_TYPES"]
    tmp = Path(ns["WORK"]) / "test_roundtrip.gguf"

    specs = [
        ("token_embd.weight",     (256, 128), 1),
        ("blk.0.ffn_gate.weight", (256, 256), 17),
        ("blk.0.ffn_up.weight",   (256, 256), 23),
        ("blk.0.attn_q.weight",   (256, 256), 12),
        ("output.weight",         (128, 256), 0),
    ]
    write(tmp, specs)
    try:
        sc = scan(tmp)
        got = {t["name"]: t["nbytes"] for t in sc["tensors"]}
        exp = {}
        for name, dims, tid in specs:
            n = 1
            for d in dims:
                n *= d
            _, blk, tsz = types[tid]
            exp[name] = ((n + blk - 1) // blk) * tsz
        assert got == exp, (got, exp)

        def bpw(name):
            t = [x for x in sc["tensors"] if x["name"] == name][0]
            return t["nbytes"] * 8 / t["n_elements"]

        assert abs(bpw("blk.0.ffn_gate.weight") - 2.3125) < 1e-9
        assert abs(bpw("blk.0.ffn_up.weight") - 4.25) < 1e-9
        assert sc["header_bytes"] == sc["file_bytes"] - sc["tensor_bytes"]
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------- P1-F

def test_split_eval_exact_groups(ns):
    """v0.1 用 ceil(len/n)，41 段 8 组只会得到 7 组。"""
    split = ns["split_eval"]
    work = Path(ns["WORK"])

    cases = [(5, 8), (7, 3), (8, 8), (9, 8), (40, 8), (41, 8), (100, 16), (1, 4)]
    for n_paras, n_groups in cases:
        tp = work / f"t_{n_paras}_{n_groups}.txt"
        tp.write_text("\n\n".join(f"para{i} " + ("w " * 30) for i in range(n_paras)),
                      encoding="utf-8")
        try:
            paths, meta = split(tp, n_groups, work / f"g_{n_paras}_{n_groups}")
            expect = min(n_groups, n_paras)
            assert len(paths) == expect, f"{n_paras}段/{n_groups}组 → {len(paths)} 组（期望 {expect}）"
            assert sum(m["paras"] for m in meta) == n_paras, "段落丢失或重复"
            assert all(m["chars"] > 0 for m in meta)
            # 组间只能差一个段落（divmod 的保证），字符数因此近似相等
            ps = [m["paras"] for m in meta]
            assert max(ps) - min(ps) <= 1, f"段落数不均：{ps}"
            cs = [m["chars"] for m in meta]
            assert max(cs) <= min(cs) * (max(ps) / min(ps)) * 1.1, f"组间字符数悬殊：{cs}"
        finally:
            if tp.exists():
                tp.unlink()


# ---------------------------------------------------------------- P1-D

def test_fingerprint_invalidation(ns):
    fp = ns["_fingerprint"]
    CFG = ns["CFG"]

    base = fp("S4_QUANT")
    assert fp("S4_QUANT") == base, "同样输入必须同样指纹"

    old = CFG["BASE_KIND"]
    CFG["BASE_KIND"] = old + "_changed"
    try:
        assert fp("S4_QUANT") != base, "改了 BASE_KIND 指纹必须变"
    finally:
        CFG["BASE_KIND"] = old
    assert fp("S4_QUANT") == base, "改回来指纹应还原"

    old2 = CFG["NLL_GROUPS"]
    CFG["NLL_GROUPS"] = old2 + 1
    try:
        a = fp("S6_NLL")
    finally:
        CFG["NLL_GROUPS"] = old2
    assert fp("S6_NLL") != a, "改了 NLL_GROUPS 指纹必须变"

    # 不同阶段的指纹不应相同
    assert fp("S0_ENV") != fp("S4_QUANT")


# ---------------------------------------------------------------- P1-E

def test_run_cmd_nonzero_raises(ns):
    run_cmd = ns["run_cmd"]
    err = ns["CommandFailed"]
    logs = Path(ns["LOGS"])
    logs.mkdir(parents=True, exist_ok=True)

    try:
        run_cmd(["sh", "-c", "echo boom; exit 3"], logs / "t_rc.log", __import__("time").time() + 30)
        raise AssertionError("rc=3 没有抛 CommandFailed")
    except err as e:
        assert e.rc == 3, e.rc
        assert "boom" in (e.tail or "")

    rc, tail = run_cmd(["sh", "-c", "echo ok"], logs / "t_ok.log", __import__("time").time() + 30)
    assert rc == 0


# ---------------------------------------------------------------- 统计

def test_paired_bootstrap_edges(ns):
    f = ns["paired_bootstrap"]
    assert f([])["mean"] is None
    assert f([])["ci95"] is None
    r1 = f([0.1])
    assert r1["mean"] == 0.1 and r1["ci95"] is None

    d = [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04]
    r = f(d)
    assert abs(r["mean"] - sum(d) / len(d)) < 1e-12
    assert r["ci95"][0] < r["mean"] < r["ci95"][1]
    assert r["ci95"][0] > 0, "全正差值应给出不含 0 的 CI"


def test_ggml_type_table(ns):
    t = ns["GGML_TYPES"]
    for tid, name, bpw in [(17, "IQ2_XS", 2.3125), (23, "IQ4_XS", 4.25),
                           (16, "IQ2_XXS", 2.0625), (35, "TQ2_0", 2.0625),
                           (12, "Q4_K", 4.5), (8, "Q8_0", 8.5), (30, "BF16", 16.0)]:
        nm, blk, tsz = t[tid]
        assert nm == name, (tid, nm, name)
        assert abs(tsz * 8 / blk - bpw) < 1e-9, (name, tsz * 8 / blk, bpw)


# ---------------------------------------------------------------- 真实规模词表

def test_gguf_vocab_array_skip(ns):
    """Qwen 的词表是 15 万个字符串的数组。必须 seek 跳过，不能物化。

    这条只在"真模型头"上才会暴露：合成小 GGUF 的 KV 里没有大数组。
    """
    scan = ns["gguf_scan"]
    tmp = Path(ns["WORK"]) / "test_vocab.gguf"
    N = 150000

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 1)      # n_tensors
    buf += struct.pack("<Q", 3)      # n_kv

    def put_str(s):
        b = s.encode()
        buf.extend(struct.pack("<Q", len(b)))
        buf.extend(b)

    put_str("general.architecture"); buf += struct.pack("<I", 8); put_str("qwen3")
    put_str("general.alignment");    buf += struct.pack("<I", 4)
    buf += struct.pack("<I", 32)

    put_str("tokenizer.ggml.tokens")
    buf += struct.pack("<I", 9)      # array
    buf += struct.pack("<I", 8)      # of string
    buf += struct.pack("<Q", N)
    for i in range(N):
        put_str(f"tok{i}")

    put_str("token_embd.weight")
    buf += struct.pack("<I", 2)
    buf += struct.pack("<Q", 256)
    buf += struct.pack("<Q", 256)
    buf += struct.pack("<I", 17)     # IQ2_XS
    buf += struct.pack("<Q", 0)

    while len(buf) % 32:
        buf += b"\x00"
    data = (256 * 256 // 256) * 74    # IQ2_XS: 256 elem/block, 74 B/block
    buf += b"\x00" * data
    tmp.write_bytes(bytes(buf))

    try:
        import time
        t = time.time()
        sc = scan(tmp)
        el = time.time() - t
        assert sc["n_tensors"] == 1
        assert sc["architecture"] == "qwen3"
        assert sc["n_params"] == 256 * 256
        assert sc["tensor_bytes"] == data
        assert sc["header_bytes"] == sc["file_bytes"] - data
        assert abs(sc["bpw_effective"] - 2.3125) < 1e-9, sc["bpw_effective"]
        assert el < 20, f"解析 15 万词表用了 {el:.1f} s，可能没有 seek 跳过"
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------- 默认阶段

def test_default_stages_complete(ns):
    """用户要求：默认只开 S0/S1/S9 时，三个必须都 ok，且 S9 产出有效报告。

    这条是 v0.2 自己的回归闸门 —— 加 P1-G 时 gguf_write_synth 变成 strict，
    S1 里构造坏文件那行会把整个 S1 打挂，而 run_stage 会吞掉异常只记 status=error。
    """
    STATE = ns["STATE"]
    for stage in ("S0_ENV", "S1_SMOKE", "S9_REPORT"):
        rec = STATE.get(stage) or {}
        assert rec.get("status") == "ok", (
            f"{stage} 默认状态下没有完成：status={rec.get('status')} "
            f"err={str(rec.get('error'))[:200]}")

    s1 = STATE["S1_SMOKE"]["result"]
    assert s1["smoke_ok"] is True, s1["checks"]
    assert s1["checks"]["unknown_type_fails_closed"] is True
    assert s1["checks"]["unknown_type_degrades_to_none"] is True

    s9 = STATE["S9_REPORT"]["result"]
    rp = Path(s9["report"])
    assert rp.exists() and rp.stat().st_size > 1000, f"报告没写出来：{rp}"
    md = rp.read_text(encoding="utf-8")
    for must in ("# transformer-residual-lab · Kaggle 运行报告",
                 "## 2. Smoke test", "## 8. 结论边界",
                 "SMOKE_OK = **True**"):
        assert must in md, f"报告缺少必要内容：{must!r}"

    rj = json.loads(Path(s9["results"]).read_text(encoding="utf-8"))
    assert rj["notebook_version"] == ns["NOTEBOOK_VERSION"]
    assert rj["stages"]["S1_SMOKE"]["status"] == "ok"
    assert set(rj["stages"]) == set(ns["STAGES"]), "阶段清单不完整"
    assert rj["disclaimer"]


def test_s9_report_bf16_both_ways(ns):
    """T4 上 bf16 可能失败也可能成功：两条分支都必须渲染出有效报告（P0-C 完整路径）。"""
    stage_s9 = ns["stage_s9_report"]
    STATE = ns["STATE"]
    s1_rec = STATE["S1_SMOKE"]
    orig = json.loads(json.dumps(s1_rec["result"]["bench"]["torch"]))
    deadline = __import__("time").time() + 120

    try:
        # 分支 1：T4（CC 7.5）—— bf16 GEMM 直接报错
        s1_rec["result"]["bench"]["torch"] = {
            "fp16": {"tflops": 50.0},
            "bf16": {"error": "RuntimeError: not implemented for BFloat16"},
            "copy": {"gb_per_s_derived": 300.0},
        }
        r = stage_s9(deadline)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "bf16：**不可用**" in md, md[-1500:]
        assert "这不是缺陷" in md
        assert "TFLOP/s（相对 fp16 比值" not in md

        # 分支 2：A100 类 —— bf16 正常
        s1_rec["result"]["bench"]["torch"] = {
            "fp16": {"tflops": 50.0},
            "bf16": {"tflops": 50.0},
            "copy": {"gb_per_s_derived": 1500.0},
        }
        r = stage_s9(deadline)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "bf16：50.0 TFLOP/s" in md, md[-1500:]
        assert "比值 1.0" in md
        assert "bf16：**不可用**" not in md

        # 分支 3：连 torch 都没有（CPU-only 内核）
        s1_rec["result"]["bench"]["torch"] = {"error": "torch 不可用: no module"}
        r = stage_s9(deadline)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "GPU 微基准不可用" in md
        assert Path(r["results"]).exists()
    finally:
        s1_rec["result"]["bench"]["torch"] = orig


# ---------------------------------------------------------------- T4 mock 端到端

T4_BENCH = {
    "fp16": {"ms": 0.34, "tflops": 50.4},
    "bf16": {"error": "RuntimeError: \"addmm_impl_cpu_\" not implemented for 'BFloat16'"},
    "copy": {"gb_moved": 0.5, "ms": 1.8, "gb_per_s_derived": 289.0},
    "vram_after_gb": {"free": 14.6, "total": 15.0},
}

A100_BENCH = {
    "fp16": {"ms": 0.09, "tflops": 190.0},
    "bf16": {"ms": 0.09, "tflops": 195.0},
    "copy": {"gb_moved": 0.5, "ms": 0.4, "gb_per_s_derived": 1300.0},
}


def _run_chain_with_bench(ns, bench):
    """把 S1 的 GPU 微基准换成给定结果，然后按默认开关重跑 S1 → S9。"""
    run_stage = ns["run_stage"]
    CFG = ns["CFG"]
    orig_bench = ns["_torch_microbench"]
    orig_force = CFG["FORCE_RERUN"]
    ns["_torch_microbench"] = lambda deadline: json.loads(json.dumps(bench))
    CFG["FORCE_RERUN"] = True
    try:
        r1 = run_stage("S1_SMOKE", ns["stage_s1_smoke"])
        r9 = run_stage("S9_REPORT", ns["stage_s9_report"])
    finally:
        ns["_torch_microbench"] = orig_bench
        CFG["FORCE_RERUN"] = orig_force
    return r1, r9


def test_t4_mock_end_to_end(ns):
    """T4（CC 7.5，无原生 bf16）：默认 S0/S1/S9 全链必须跑完，报告必须有效。"""
    STATE = ns["STATE"]
    r1, r9 = _run_chain_with_bench(ns, T4_BENCH)

    assert STATE["S1_SMOKE"]["status"] == "ok", STATE["S1_SMOKE"].get("error")
    assert r1["smoke_ok"] is True, r1["checks"]
    assert STATE["S9_REPORT"]["status"] == "ok", STATE["S9_REPORT"].get("error")
    assert STATE["S0_ENV"]["status"] == "ok"

    md = Path(r9["report"]).read_text(encoding="utf-8")
    assert "bf16：**不可用**" in md
    assert "T4（CC 7.5）没有原生 bf16" in md
    assert "fp16：50.4 TFLOP/s" in md
    assert "SMOKE_OK = **True**" in md

    # bf16 不可用不等于报告不可用：results.json 仍然完整可解析
    rj = json.loads(Path(r9["results"]).read_text(encoding="utf-8"))
    assert rj["s1_smoke"]["bench"]["torch_summary"]["bf16_available"] is False
    assert rj["stages"]["S9_REPORT"]["status"] == "ok"


def test_a100_mock_end_to_end(ns):
    """反例：bf16 可用时不能还打印「不可用」，且要给出比值。"""
    STATE = ns["STATE"]
    r1, r9 = _run_chain_with_bench(ns, A100_BENCH)

    assert STATE["S1_SMOKE"]["status"] == "ok", STATE["S1_SMOKE"].get("error")
    assert STATE["S9_REPORT"]["status"] == "ok", STATE["S9_REPORT"].get("error")
    md = Path(r9["report"]).read_text(encoding="utf-8")
    assert "bf16：195.0 TFLOP/s" in md, md[-1200:]
    assert "比值 1.026" in md, md[-1200:]
    assert "bf16：**不可用**" not in md
    assert "T4（CC 7.5）没有原生 bf16" not in md


# ---------------------------------------------------------------- runner

TESTS = [
    test_need_bytes_arithmetic,
    test_anchor_decision_t4_skips,
    test_anchor_decision_q8_is_approx,
    test_anchor_decision_a100_ready,
    test_bench_summary_bf16_error,
    test_bench_summary_degenerate_inputs,
    test_gguf_unknown_type_fails_closed,
    test_gguf_roundtrip_bytes,
    test_gguf_vocab_array_skip,
    test_split_eval_exact_groups,
    test_fingerprint_invalidation,
    test_run_cmd_nonzero_raises,
    test_paired_bootstrap_edges,
    test_ggml_type_table,
    test_default_stages_complete,
    test_s9_report_bf16_both_ways,
    test_t4_mock_end_to_end,
    test_a100_mock_end_to_end,
]


def main():
    if not NB.exists():
        print(f"找不到 {NB}")
        return 2
    print(f"notebook: {NB.name}  ({NB.stat().st_size} B)\n")

    try:
        ns = load_namespace()
    except Exception:
        print("!! notebook 的 code cell 无法全部执行：")
        traceback.print_exc()
        return 1
    print(f"code cells 执行完毕，notebook v{ns['NOTEBOOK_VERSION']}\n")

    failed = []
    for t in TESTS:
        name = t.__name__
        try:
            t(ns)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    print()
    if failed:
        print(f"{len(failed)} / {len(TESTS)} 失败")
        for name, e in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(e), e, e.__traceback__)
        return 1
    print(f"全部 {len(TESTS)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
