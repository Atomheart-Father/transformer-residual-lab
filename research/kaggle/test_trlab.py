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


# ---------------------------------------------------------------- v0.2.1 · P0-1

def test_base_requant_plan(ns):
    """从已量化的 base 量化，上游会直接抛；这里必须先拦住。"""
    f = ns["_base_requant_plan"]

    r = f("f16", False)
    assert r["status"] == "ok" and r["base_is_quantized"] is False
    assert r["flags"] == [], r["flags"]
    assert r["validity_scope"] == "from_high_precision"

    r = f("BF16", False)                    # 大小写不该影响判断
    assert r["status"] == "ok" and r["flags"] == []

    r = f("q8_0", False)
    assert r["status"] == "blocked", r
    assert r["needs_allow_requantize"] is True
    assert "requantize" in r["reason"]
    assert "IMATRIX_BASE" in r["reason"], "报错必须给出可执行的替代方案"
    assert "severely reduce quality" in r["reason"], "要引用上游帮助原文"

    r = f("q8_0", True)
    assert r["status"] == "ok"
    assert r["flags"] == ["--allow-requantize"]
    assert r["validity_scope"] == "requantized_pilot_only"
    assert r["provenance_tag"] == "requantized_pilot_only"

    for k in ("q4_k_m", "IQ2_XS", "mxfp4", "tq2_0"):
        assert f(k, False)["status"] == "blocked", k
    for k in ("f32", "bf16", "f16", "fp16"):
        assert f(k, False)["status"] == "ok", k

    assert f("", False)["status"] == "blocked", "未设置 BASE_KIND 也必须拦住"


def test_quant_cmd_shape(ns):
    """--allow-requantize 只能出现在需要它的时候，且位置参数顺序不能乱。"""
    cmd_f = ns["_quant_cmd"]
    plan_f = ns["_base_requant_plan"]

    cmd = cmd_f("/b/llama-quantize", "/m/im.gguf", "/m/f16.gguf",
                "/m/out.partial", "IQ2_XS", 8, plan_f("f16", False))
    assert cmd[:2] == ["/b/llama-quantize", "--imatrix"]
    assert cmd[2] == "/m/im.gguf"
    assert cmd[3] == "/m/f16.gguf", "base 必须是第一个位置参数"
    assert cmd[4] == "/m/out.partial"
    assert cmd[5] == "IQ2_XS"
    assert cmd[6] == "8"
    assert "--allow-requantize" not in cmd

    cmd = cmd_f("/b/llama-quantize", "/m/im.gguf", "/m/q8.gguf",
                "/m/out.partial", "IQ4_XS", 8, plan_f("q8_0", True))
    assert cmd[-1] == "--allow-requantize", cmd
    assert cmd[3] == "/m/q8.gguf", "加 flag 不能挤动位置参数"


def test_quant_fp_binds_inputs(ns):
    """产物指纹必须绑住 base / imatrix / commit / 格式 / 二次量化 / 参数。"""
    f = ns["_quant_fp"]
    plan_f = ns["_base_requant_plan"]
    primary = plan_f("f16", False)
    pilot = plan_f("q8_0", True)
    base = {"size": 1, "mtime": 2, "head": "aa"}
    params = {"imatrix_ctx": 512, "imatrix_chunks": 32}

    a = f(base, "deadbeef", "cafe1234", "IQ2_XS", primary, params)
    assert len(a) == 32
    assert a == f(base, "deadbeef", "cafe1234", "IQ2_XS", primary, params), "同样输入必须同样指纹"

    variants = {
        "换 imatrix": f(base, "deadbeee", "cafe1234", "IQ2_XS", primary, params),
        "换 commit": f(base, "deadbeef", "cafe9999", "IQ2_XS", primary, params),
        "换格式": f(base, "deadbeef", "cafe1234", "IQ4_XS", primary, params),
        "换 base": f({"size": 9, "mtime": 2, "head": "aa"}, "deadbeef", "cafe1234",
                     "IQ2_XS", primary, params),
        "换二次量化标记": f(base, "deadbeef", "cafe1234", "IQ2_XS", pilot, params),
        "换量化参数": f(base, "deadbeef", "cafe1234", "IQ2_XS", primary,
                        {"imatrix_ctx": 1024, "imatrix_chunks": 32}),
    }
    for name, fp in variants.items():
        assert fp != a, f"{name} 之后指纹没有变，会静默复用旧产物"


def test_s4_blocks_quantized_base(ns):
    """BASE_KIND=q8_0 且没开 ALLOW_REQUANTIZE：必须在碰 llama-quantize 之前就报错。"""
    CFG = ns["CFG"]
    work = Path(ns["WORK"])
    snap = {k: CFG[k] for k in ("BASE_GGUF", "BASE_KIND", "IMATRIX_GGUF", "IMATRIX_BASE",
                                "IMATRIX_BASE_KIND", "ALLOW_REQUANTIZE")}
    orig_bin = ns["llama_bin"]

    base = work / "s4_q8.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 8)])
    imat = work / "s4_q8_im.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])

    CFG.update(BASE_GGUF=str(base), BASE_KIND="q8_0", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=False)
    ns["llama_bin"] = lambda name: (_ for _ in ()).throw(
        AssertionError(f"llama_bin({name}) 不该被调用：应该在它之前就拦住"))

    try:
        ns["stage_s4_quant"](__import__("time").time() + 60)
        raise AssertionError("q8_0 base 没有报错")
    except ns["GuardError"] as e:
        msg = str(e)
        assert "requantize" in msg
        assert "f16/bf16" in msg
        assert "IMATRIX_BASE" in msg
    finally:
        ns["llama_bin"] = orig_bin
        CFG.update(snap)
        for p in (base, imat):
            if p.exists():
                p.unlink()


def test_s4_rebuilds_when_imatrix_changes(ns):
    """P0-2：imatrix 指纹变了，两个 GGUF 必须实质重建，不能复用。"""
    CFG = ns["CFG"]
    MDLS = Path(ns["MDLS"])
    work = Path(ns["WORK"])
    snap = {k: CFG[k] for k in ("BASE_GGUF", "BASE_KIND", "IMATRIX_GGUF", "IMATRIX_BASE",
                                "IMATRIX_BASE_KIND", "ALLOW_REQUANTIZE", "FORCE_RERUN",
                                "HASH_GGUF")}
    orig_bin, orig_run = ns["llama_bin"], ns["run_cmd"]

    base = work / "s4_base_f16.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 1),
                                  ("blk.0.ffn_gate.weight", (256, 256), 1)])
    imat = work / "s4_imatrix.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])

    CFG.update(BASE_GGUF=str(base), BASE_KIND="f16", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=False,
               FORCE_RERUN=False, HASH_GGUF=True)

    calls = []

    def fake_bin(name):
        return Path("/nonexistent") / name

    def fake_run(cmd, log_path, deadline, cwd=None, env=None):
        fmt = [c for c in cmd if c in ("IQ2_XS", "IQ4_XS")][0]
        out = [c for c in cmd if str(c).endswith(".partial")][0]
        tid = 17 if fmt == "IQ2_XS" else 23
        ns["gguf_write_synth"](out, [("blk.0.ffn_gate.weight", (256, 256), tid)])
        calls.append({"fmt": fmt, "cmd": [str(c) for c in cmd]})
        return 0, ""

    ns["llama_bin"], ns["run_cmd"] = fake_bin, fake_run
    deadline = __import__("time").time() + 300
    made = []
    try:
        # 第一次：两个都要造出来
        r1 = ns["stage_s4_quant"](deadline)
        assert len(calls) == 2, f"第一次应当量化两次，实际 {len(calls)}"
        assert r1["IQ2_XS"]["reused"] is False and r1["IQ4_XS"]["reused"] is False
        for fmt in ("IQ2_XS", "IQ4_XS"):
            dst = MDLS / f"qwen3-8b-{fmt}.gguf"
            prov = MDLS / f"qwen3-8b-{fmt}.gguf.provenance.json"
            assert dst.exists() and prov.exists(), f"{fmt} 产物或 provenance 缺失"
            rec = json.loads(prov.read_text(encoding="utf-8"))
            assert rec["quant_fp"] == r1[fmt]["quant_fp"]
            assert rec["requant"] == "primary"
            assert rec["base_kind"] == "f16"
            made += [dst, prov]

        # 第二次：指纹一致 → 复用，不再调用外部命令
        n_before = len(calls)
        r2 = ns["stage_s4_quant"](deadline)
        assert len(calls) == n_before, "指纹一致时不该重新量化"
        assert r2["IQ2_XS"]["reused"] is True and r2["IQ4_XS"]["reused"] is True
        assert r2["IQ2_XS"]["quant_fp"] == r1["IQ2_XS"]["quant_fp"]

        # 第三次：换 imatrix → 必须重建两个，且旧文件被移走
        ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0),
                                      ("blk.0.ffn_up.weight", (256, 256), 0)])
        r3 = ns["stage_s4_quant"](deadline)
        assert len(calls) == n_before + 2, "换 imatrix 后必须实质重建两个 GGUF"
        assert r3["IQ2_XS"]["reused"] is False and r3["IQ4_XS"]["reused"] is False
        assert r3["IQ2_XS"]["quant_fp"] != r1["IQ2_XS"]["quant_fp"]
        for fmt in ("IQ2_XS", "IQ4_XS"):
            stale = list(MDLS.glob(f"qwen3-8b-{fmt}.gguf.stale-*"))
            assert len(stale) == 1, f"{fmt} 的旧产物没有被移走：{stale}"
            rec = json.loads((MDLS / f"qwen3-8b-{fmt}.gguf.provenance.json")
                             .read_text(encoding="utf-8"))
            assert rec["quant_fp"] == r3[fmt]["quant_fp"], "provenance 没有更新"
            made += stale

        # 第四次：换成量化 base 但显式放行 → 命令必须带 --allow-requantize，且标 pilot
        CFG["BASE_KIND"] = "q8_0"
        CFG["ALLOW_REQUANTIZE"] = True
        r4 = ns["stage_s4_quant"](deadline)
        assert len(calls) == n_before + 4
        for c in calls[-2:]:
            assert "--allow-requantize" in c["cmd"], c["cmd"]
        assert r4["_fairness"]["base_scope"] == "requantized_pilot_only"
        assert r4["_fairness"]["requant"] == "requantized_pilot_only"
        assert r4["IQ2_XS"]["reused"] is False
        made += list(MDLS.glob("qwen3-8b-IQ*_XS.gguf.provenance.json"))
        made += list(MDLS.glob("qwen3-8b-IQ*_XS.gguf"))
    finally:
        ns["llama_bin"], ns["run_cmd"] = orig_bin, orig_run
        CFG.update(snap)
        for p in made + list(MDLS.glob("qwen3-8b-IQ*_XS.gguf*")) + [base, imat]:
            if p.exists():
                p.unlink()


def test_t4_split_base_path(ns):
    """T4 推荐路径：imatrix 用 Q8_0，量化用 f16。两者结构一致即可，且不得要求 requantize。"""
    CFG = ns["CFG"]
    MDLS = Path(ns["MDLS"])
    work = Path(ns["WORK"])
    keys = ("BASE_GGUF", "BASE_KIND", "IMATRIX_BASE", "IMATRIX_BASE_KIND",
            "IMATRIX_GGUF", "ALLOW_REQUANTIZE", "FORCE_RERUN", "HASH_GGUF",
            "IMATRIX_CTX", "IMATRIX_CHUNKS", "CALIB_TXT")
    snap = {k: CFG[k] for k in keys}
    orig_bin, orig_run = ns["llama_bin"], ns["run_cmd"]

    shapes = [("blk.0.attn_q.weight", (256, 256)),
              ("blk.0.ffn_gate.weight", (256, 256))]
    base_f16 = work / "split_base_f16.gguf"
    base_q8 = work / "split_base_q8.gguf"
    ns["gguf_write_synth"](base_f16, [(n, d, 1) for n, d in shapes])
    ns["gguf_write_synth"](base_q8, [(n, d, 8) for n, d in shapes])
    calib = work / "split_calib.txt"
    calib.write_text("hello world " * 20000, encoding="utf-8")
    imat = work / "split_imatrix.gguf"

    CFG.update(BASE_GGUF=str(base_f16), BASE_KIND="f16",
               IMATRIX_BASE=str(base_q8), IMATRIX_BASE_KIND="q8_0",
               IMATRIX_GGUF=str(imat), ALLOW_REQUANTIZE=False,
               FORCE_RERUN=False, HASH_GGUF=True, CALIB_TXT=str(calib))

    calls = []

    def fake_run(cmd, log_path, deadline, cwd=None, env=None):
        calls.append([str(c) for c in cmd])
        if "--imatrix" in cmd:                       # S4
            fmt = [c for c in cmd if c in ("IQ2_XS", "IQ4_XS")][0]
            out = [c for c in cmd if str(c).endswith(".partial")][0]
            ns["gguf_write_synth"](out, [("blk.0.ffn_gate.weight", (256, 256),
                                          17 if fmt == "IQ2_XS" else 23)])
        else:                                        # S3
            out = cmd[cmd.index("-o") + 1]
            ns["gguf_write_synth"](out, [("blk.0.attn_q.weight", (256, 256), 0)])
        return 0, ""

    ns["llama_bin"] = lambda name: Path("/nonexistent") / name
    ns["run_cmd"] = fake_run
    made = []
    s3_old = ns["STATE"].get("S3_IMATRIX")
    try:
        s3 = ns["stage_s3_imatrix"](__import__("time").time() + 120)
        assert s3["base_kind"] == "q8_0", s3["base_kind"]
        assert s3["imatrix_base_from_approx"] is True
        assert s3["base_gguf"] == str(base_q8)
        made.append(imat)
        # S4 从 STATE 读 S3 记录，这里直接注入（没走 run_stage）
        ns["STATE"]["S3_IMATRIX"] = {"status": "ok", "result": s3}

        s4 = ns["stage_s4_quant"](__import__("time").time() + 300)
        fair = s4["_fairness"]
        assert fair["imatrix_base_same_as_quant_base"] is False, "两个 base 本就不是同一个文件"
        assert fair["imatrix_base_consistent"]["ok"] is True, fair["imatrix_base_consistent"]
        assert fair["imatrix_base_from_approx"] is True, "imatrix 来自 Q8_0，必须如实记录"
        assert fair["base_scope"] == "from_high_precision"
        assert fair["requant"] == "primary"
        for c in calls:
            assert "--allow-requantize" not in c, f"量化 base 是 f16，不该要这个 flag: {c}"

        for fmt in ("IQ2_XS", "IQ4_XS"):
            rec = json.loads((MDLS / f"qwen3-8b-{fmt}.gguf.provenance.json")
                             .read_text(encoding="utf-8"))
            assert rec["imatrix_base_kind"] == "q8_0"
            assert rec["base_kind"] == "f16"
            assert rec["imatrix_base_consistent"]["ok"] is True
        made += list(MDLS.glob("qwen3-8b-IQ*_XS.gguf*"))
    finally:
        ns["llama_bin"], ns["run_cmd"] = orig_bin, orig_run
        CFG.update(snap)
        if s3_old is None:
            ns["STATE"].pop("S3_IMATRIX", None)
        else:
            ns["STATE"]["S3_IMATRIX"] = s3_old
        for p in made + [base_f16, base_q8, calib]:
            if p.exists():
                p.unlink()


def test_base_consistency(ns):
    """imatrix base 与量化 base 必须同源：架构、张量名、形状、参数量。"""
    f = ns["_base_consistency"]
    a = {"architecture": "qwen3", "n_params": 100,
         "tensors": [{"name": "blk.0.attn_q.weight", "dims": [4096, 4096]},
                     {"name": "blk.0.ffn_gate.weight", "dims": [12288, 4096]}]}
    b = {"architecture": "qwen3", "n_params": 100,
         "tensors": [{"name": "blk.0.attn_q.weight", "dims": [4096, 4096]},
                     {"name": "blk.0.ffn_gate.weight", "dims": [12288, 4096]}]}
    r = f(a, b)
    assert r["ok"] is True and r["n_tensors"] == 2

    assert f(a, {**b, "architecture": "llama"})["ok"] is False
    assert "architecture" in f(a, {**b, "architecture": "llama"})["why"]

    c = {**b, "tensors": b["tensors"][:1]}
    r = f(a, c)
    assert r["ok"] is False and "张量名" in r["why"]

    d = {"architecture": "qwen3", "n_params": 100,
         "tensors": [{"name": "blk.0.attn_q.weight", "dims": [4096, 4096]},
                     {"name": "blk.0.ffn_gate.weight", "dims": [11008, 4096]}]}
    r = f(a, d)
    assert r["ok"] is False and "形状" in r["why"], r

    e = {**b, "n_params": 999}
    assert f(a, e)["ok"] is False

    assert f(None, b)["ok"] is None, "没有 scan 结果时必须是 None（无法验证），不是 True"


# ---------------------------------------------------------------- v0.2.1 · rho

def test_rho_sign(ns):
    """rho 的分子分母都要按「后减前」，d4 必须是正数。"""
    f = ns["_rho_from_nll"]

    r = f(anchor_nll=2.0, q4_nll=2.5, q2_nll=3.0)
    assert abs(r["d4"] - 0.5) < 1e-12, r
    assert abs(r["d2"] - 1.0) < 1e-12
    assert r["d4"] > 0, "锚点→Q4 的退化必须为正"
    assert r["d2"] > r["d4"], "Q2 应当比 Q4 退化更多"
    assert abs(r["rho"] - 1.0) < 1e-12, r          # (1.0-0.5)/0.5

    # 与旧写法对比：比值相同，但 d4 的符号相反 —— 这正是被指出的问题
    old_d4 = 2.0 - 2.5
    assert old_d4 < 0 < r["d4"]
    assert abs((r["d2"] - r["d4"]) / r["d4"] - (r["d2"] - r["d4"]) / r["d4"]) < 1e-12

    # 锚点没有比 Q4 更好 → rho 无定义，而不是给出一个数
    r = f(anchor_nll=3.0, q4_nll=2.5, q2_nll=3.0)
    assert r["rho"] is None and "无定义" in r["reason"], r

    assert f(None, 1.0, 2.0)["rho"] is None


def test_s9_rho_rendering(ns):
    """S9 里 dNLL(Q4,anchor) 的符号必须与代数定义一致（正数）。"""
    STATE = ns["STATE"]
    s6_old, s8_old = STATE.get("S6_NLL"), STATE.get("S8_BF16_ANCHOR")
    try:
        STATE["S6_NLL"] = {"status": "ok", "result": {
            "aggregate": {
                "IQ2_XS": {"mean_nll": 3.0, "mean_ppl": 20.1, "nll_sd": 0.02},
                "IQ4_XS": {"mean_nll": 2.5, "mean_ppl": 12.2, "nll_sd": 0.01}},
            "paired_delta_nll_iq2_minus_iq4": {"mean": 0.5, "ci95": [0.4, 0.6], "n": 8},
            "groups": 8, "fairness_verified": True,
            "paired_delta": {"char_weighted_mean": 0.5},
            "isolation": {"ok": True, "overlap_lines": 0},
        }}
        STATE["S8_BF16_ANCHOR"] = {"status": "ok", "result": {
            "status": "ok", "validity_scope": "high_precision",
            "mean_nll": 2.0, "base_kind": "f16", "tier": "high",
            "required_gb": 17.25, "vram_usable_gb": 39.0,
            "anchor_dtype_note": "test",
        }}
        r = ns["stage_s9_report"](__import__("time").time() + 60)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "dNLL(Q4,anchor)=+0.5000" in md, md[-1800:]
        assert "dNLL(Q2,anchor)=+1.0000" in md
        assert "rho = dNLL(Q2,Q4)/dNLL(Q4,anchor) = 1.0000" in md
        assert "dNLL(Q4,anchor)=-0.5000" not in md, "符号反了"
    finally:
        for k, v in (("S6_NLL", s6_old), ("S8_BF16_ANCHOR", s8_old)):
            if v is None:
                STATE.pop(k, None)
            else:
                STATE[k] = v


def test_s9_marks_requantized_base(ns):
    """二次量化的 base 必须在报告顶部就有醒目声明。"""
    STATE = ns["STATE"]
    old = STATE.get("S4_QUANT")
    try:
        STATE["S4_QUANT"] = {"status": "ok", "result": {
            "_fairness": {"base_scope": "requantized_pilot_only",
                          "requant": "requantized_pilot_only"}}}
        r = ns["stage_s9_report"](__import__("time").time() + 60)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "base 本身是二次量化" in md, md[-1500:]
        assert "ALLOW_REQUANTIZE" in md
        assert "不能**作为『相对原始 checkpoint BF16』" in md

        STATE["S4_QUANT"] = {"status": "ok", "result": {
            "_fairness": {"base_scope": "from_high_precision", "requant": "primary"}}}
        r = ns["stage_s9_report"](__import__("time").time() + 60)
        md = Path(r["report"]).read_text(encoding="utf-8")
        assert "base 本身是二次量化" not in md
    finally:
        if old is None:
            STATE.pop("S4_QUANT", None)
        else:
            STATE["S4_QUANT"] = old


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
    test_base_requant_plan,
    test_quant_cmd_shape,
    test_quant_fp_binds_inputs,
    test_s4_blocks_quantized_base,
    test_s4_rebuilds_when_imatrix_changes,
    test_t4_split_base_path,
    test_base_consistency,
    test_rho_sign,
    test_s9_rho_rendering,
    test_s9_marks_requantized_base,
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
