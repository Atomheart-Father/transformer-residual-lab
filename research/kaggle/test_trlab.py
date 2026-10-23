#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trlab 包的 CPU-only 回归测试。

用法：
    python3 research/kaggle/test_trlab.py

不下载、不联网、不需要 GPU、不需要 llama.cpp。它直接 **import trlab 包**
（不解析 notebook —— notebook 只是薄入口），然后对历史缺陷逐条断言。

v1.0 起测试对象是 research/kaggle/trlab/*.py；notebook 只由
build_notebook.py 生成，本身不含逻辑。
"""

import copy
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 有些容器/沙箱里 /tmp 可写但**不可删**，tempfile 的可用性探测会直接失败，
# 于是连 import 都过不去。显式钉一个项目内的临时根，避免探测。
_TMP_ROOT = HERE / ".trlab_tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP_ROOT)


def _trlab_modules():
    """包 + 所有子模块，按依赖顺序（config 在前）。"""
    import pkgutil
    import importlib
    import trlab
    mods = [trlab]
    for m in pkgutil.iter_modules(trlab.__path__):
        mods.append(importlib.import_module("trlab." + m.name))
    order = ["config", "errors", "paths", "gguf", "core", "fetch", "imatrix",
             "deps", "stages_env", "stages_build", "stages_imatrix",
             "stages_quant", "stages_eval", "report", "run"]
    def key(m):
        n = m.__name__.split(".")[-1]
        return order.index(n) if n in order else 99
    return sorted(mods, key=key)


class NS:
    """跨模块命名空间门面。

    `ns["CFG"]` 返回包内任意模块持有的那个对象（同一个 dict，原地修改全局可见）；
    `ns["X"] = v` 写在**所有**持有 X 的模块上 —— 这样测试替换一个函数后，
    包内任何模块的调用都能看到，等价于旧版"改 exec 命名空间"。
    """

    def __init__(self, mods):
        self._mods = mods

    def _find(self, name):
        for m in self._mods:
            if hasattr(m, name):
                return m
        return None

    def __getitem__(self, name):
        m = self._find(name)
        if m is None:
            raise KeyError(name)
        return getattr(m, name)

    def __setitem__(self, name, value):
        hits = [m for m in self._mods if hasattr(m, name)]
        if not hits:
            hits = [self._mods[0]]      # 兜底挂到 trlab 包自己身上
        for m in hits:
            setattr(m, name, value)

    def __contains__(self, name):
        return self._find(name) is not None

    def get(self, name, default=None):
        m = self._find(name)
        return getattr(m, name) if m is not None else default


def load_namespace():
    """在临时工作目录里 import trlab，返回跨模块命名空间。

    显式调 ensure_dirs()：v1.0 起 import 包不再有副作用（旧 notebook 是在
    cell 里 mkdir 的），测试要自己把 working 目录树建起来。
    """
    tmp = Path(tempfile.mkdtemp(prefix="trlab_test_"))
    os.chdir(tmp)
    ns = NS(_trlab_modules())
    ns["ensure_dirs"]()
    # 默认开关下 notebook 只会执行这三个阶段；补跑，让 STATE 与旧版一致
    for _name, _fn in (("S0_ENV", ns["stage_s0_env"]),
                       ("S1_SMOKE", ns["stage_s1_smoke"]),
                       ("S9_REPORT", ns["stage_s9_report"])):
        ns["run_stage"](_name, _fn)
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
    """选项必须全部在位置参数之前。

    quantize.cpp 的循环条件是 `strncmp(argv[arg_idx], "--", 2) == 0`，
    遇到第一个位置参数就退出 —— 之后出现的 --allow-requantize 会被当成
    多余位置参数丢掉，requantize 仍然被拒。v0.2.1 就错在这里。
    """
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
    assert cmd[1] == "--allow-requantize", f"选项必须在最前面：{cmd}"
    i = cmd.index("/m/q8.gguf")
    assert cmd[i:] == ["/m/q8.gguf", "/m/out.partial", "IQ4_XS", "8"], cmd
    assert all(not str(c).startswith("--") for c in cmd[i:]), \
        f"位置参数之后不能再出现任何选项：{cmd[i:]}"
    assert cmd.index("--allow-requantize") < cmd.index("--imatrix")


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
                                "IMATRIX_BASE_KIND", "ALLOW_REQUANTIZE",
                                "IMATRIX_ALLOW_UNPINNED")}
    orig_bin = ns["llama_bin"]

    base = work / "s4_q8.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 8)])
    imat = work / "s4_q8_im.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])

    CFG.update(BASE_GGUF=str(base), BASE_KIND="q8_0", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=False,
               # 合成 imatrix 不是 IMATRIX_PINNED 那个真实产物，pin 门必须显式放行，
               # 否则被测的就不是"二次量化拦截"而是"imatrix pin 拦截"了
               IMATRIX_ALLOW_UNPINNED=True)
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
                                "HASH_GGUF", "IMATRIX_ALLOW_UNPINNED")}
    orig_bin, orig_run = ns["llama_bin"], ns["run_cmd"]

    base = work / "s4_base_f16.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 1),
                                  ("blk.0.ffn_gate.weight", (256, 256), 1)])
    imat = work / "s4_imatrix.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])

    CFG.update(BASE_GGUF=str(base), BASE_KIND="f16", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=False,
               FORCE_RERUN=False, HASH_GGUF=True,
               # 合成 imatrix 过不了 IMATRIX_PINNED 的 sha256 门；这里测的是指纹复用，
               # 不是 pin 校验（pin 校验由 v0.5 的 _verify_imatrix 专项测试覆盖）
               IMATRIX_ALLOW_UNPINNED=True)

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

        # 第四次：v0.5 起本路线**禁止**二次量化 —— 即使显式设 ALLOW_REQUANTIZE=True
        # 也必须被拦死，且命令里不许出现 --allow-requantize（需求 1）
        CFG["BASE_KIND"] = "q8_0"
        CFG["ALLOW_REQUANTIZE"] = True
        try:
            ns["stage_s4_quant"](deadline)
            raise AssertionError("v0.5 起 BASE_KIND=q8_0 必须被直接拒绝")
        except ns["GuardError"] as e:
            assert "禁止二次量化" in str(e), e
            assert "ALLOW_REQUANTIZE 必须保持 False" in str(e), e
        assert len(calls) == n_before + 2, "被拒之后不该再调用 llama-quantize"
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
            "IMATRIX_CTX", "IMATRIX_CHUNKS", "CALIB_TXT", "IMATRIX_ALLOW_UNPINNED")
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
               FORCE_RERUN=False, HASH_GGUF=True, CALIB_TXT=str(calib),
               # S3 会写出带真实 sha256 的 provenance sidecar，所以这里本来就能过
               # provenance_verified；显式放行只是让结果不依赖测试执行顺序
               IMATRIX_ALLOW_UNPINNED=True)

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
        # v0.5 语义变更：两个 base 文件不同（Q8_0 vs f16）是设计如此，
        # 判据从"路径相同"改成"同源性核验到了哪一级"。
        assert fair["imatrix_base_same_as_quant_base"] is None, \
            "v0.5 起这个字段不再是路径比较，必须是 None（判据是 consistency level）"
        assert fair["imatrix_base_consistency_level"] == "file", fair
        assert fair["imatrix_base_consistent"]["ok"] is True, fair["imatrix_base_consistent"]
        assert fair["imatrix_base_kind"] == "q8_0", fair["imatrix_base_kind"]
        assert fair["verified_level"] == "file", fair["verified_level"]
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
                 "## 2. 工具链（S2_BUILD）", "## 3. Smoke test",
                 "## 4. imatrix（S3_IMATRIX）",
                 "## 10. 结论边界",
                 "SMOKE_OK = **True**"):
        assert must in md, f"报告缺少必要内容：{must!r}"

    rj = json.loads(Path(s9["results"]).read_text(encoding="utf-8"))
    assert rj["notebook_version"] == ns["NOTEBOOK_VERSION"]
    assert rj["stages"]["S1_SMOKE"]["status"] == "ok"
    # 自一致性：S9 写盘时自己还没登记进 STATE，不能留 null
    assert rj["stages"]["S9_REPORT"]["status"] == "ok", rj["stages"]["S9_REPORT"]
    assert rj["stages"]["S9_REPORT"]["self_reported"] is True
    assert s9["stage_status"]["S9_REPORT"] == "ok"
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


# ---------------------------------------------------------------- v0.2.2 · S2

def test_cuda_preflight(ns):
    """BUILD_CUDA=True 但 nvcc 不可用时必须在 cmake 之前拦住。"""
    f = ns["_cuda_preflight"]

    assert f({"nvcc": "Cuda compilation tools, release 12.8"}, True)["ok"] is True
    assert f({"nvcc": "Cuda compilation tools, release 12.8"}, False)["ok"] is True

    for bad in ({"nvcc": "不可用: FileNotFoundError"}, {"nvcc": "rc=127"}, {}, {"nvcc": ""}):
        r = f(bad, True)
        assert r["ok"] is False, bad
        assert "BUILD_CUDA" in r["note"] and "CPU" in r["note"], r["note"]

    # 关掉 CUDA 时，nvcc 缺失不该拦
    assert f({"nvcc": "不可用: FileNotFoundError"}, False)["ok"] is True


def test_toolchain_report(ns):
    """本机没有 nvcc，也不能抛；返回值要覆盖四个键。"""
    tc = ns["_toolchain_report"]()
    assert set(tc) == {"cmake", "nvcc", "gxx", "gcc"}, tc
    assert tc["cmake"].startswith("cmake version"), tc["cmake"]
    assert isinstance(tc["nvcc"], str) and tc["nvcc"]


def test_s2_cmake_flags(ns):
    """S2 必须：关掉 UI 预构建下载、CUDA 只编一个架构、且不碰 /kaggle/input 写。

    v0.3.1 追加：默认传 `-DGGML_CUDA_NO_VMM=ON`（Kaggle T4 上 CUDA::cuda_driver
    target 找不到），`CUDA_NO_VMM=False` 或 `BUILD_CUDA=False` 时不传。
    """
    CFG = ns["CFG"]
    BUILD = Path(ns["BUILD"])
    work = Path(ns["WORK"])
    src = work / "fake_llama_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "CMakeLists.txt").write_text("# fake\n", encoding="utf-8")

    keys = ("LLAMA_SRC", "LLAMA_BIN", "LLAMA_COMMIT", "BUILD_CUDA", "CUDA_ARCH",
            "CUDA_NO_VMM", "BUILD_JOBS", "FORCE_RERUN")
    snap = {k: CFG[k] for k in keys}
    orig_run, orig_tc = ns["run_cmd"], ns["_toolchain_report"]
    calls = []

    def fake_run(cmd, log_path, deadline, cwd=None, env=None):
        calls.append([str(c) for c in cmd])
        if "--build" in cmd:
            for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
                p = BUILD / "bin" / t
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                os.chmod(p, 0o755)
        return 0, ""

    def clear_bins():
        for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
            for c in (BUILD / "bin" / t, BUILD / t):
                if c.exists():
                    c.unlink()

    ns["run_cmd"] = fake_run
    ns["_toolchain_report"] = lambda: {"cmake": "cmake version 3.31.0",
                                       "nvcc": "Cuda compilation tools, release 12.8",
                                       "gxx": "11", "gcc": "11"}
    try:
        CFG.update(LLAMA_SRC=str(src), LLAMA_BIN="", LLAMA_COMMIT="",
                   BUILD_CUDA=True, CUDA_ARCH="75", CUDA_NO_VMM=True,
                   BUILD_JOBS=4, FORCE_RERUN=True)
        clear_bins()
        r = ns["stage_s2_build"](__import__("time").time() + 120)

        cfg_call = calls[0]
        assert cfg_call[0] == "cmake" and "-S" in cfg_call
        assert str(src) in cfg_call and str(BUILD) in cfg_call
        assert "-DLLAMA_USE_PREBUILT_UI=OFF" in cfg_call, \
            "否则构建期会 file(DOWNLOAD) 去 HF 取 UI，Internet Off 时白等"
        assert "-DCMAKE_CUDA_ARCHITECTURES=75" in cfg_call, \
            "不指定时上游默认九套架构，4 核上编译量翻几倍"
        assert "-DGGML_CUDA=ON" in cfg_call
        assert "-DLLAMA_BUILD_TESTS=OFF" in cfg_call
        assert "-DLLAMA_BUILD_EXAMPLES=OFF" in cfg_call
        assert str(src / "build") not in " ".join(cfg_call), "build 不能落在源码树（只读挂载）"
        # v0.3.1：Kaggle 真实失败 —— CUDA::cuda_driver target 找不到
        assert "-DGGML_CUDA_NO_VMM=ON" in cfg_call, \
            "Kaggle T4 上 find_package(CUDAToolkit) 解析不出 CUDA::cuda_driver"
        assert r["cuda_no_vmm"] is True, r

        build_call = calls[1]
        assert "--build" in build_call and "--target" in build_call
        for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
            assert t in build_call, t
        assert r["reused"] is False and r["cuda"] is True
        assert r["cuda_arch"] == "75"
        assert r["toolchain"]["nvcc"].startswith("Cuda")

        # BUILD_CUDA=False：不传架构，GGML_CUDA=OFF，也不传 NO_VMM
        calls.clear()
        clear_bins()
        CFG.update(BUILD_CUDA=False)
        r2 = ns["stage_s2_build"](__import__("time").time() + 120)
        cfg2 = calls[0]
        assert "-DGGML_CUDA=OFF" in cfg2
        assert not any(a.startswith("-DCMAKE_CUDA_ARCHITECTURES") for a in cfg2), cfg2
        assert "-DGGML_CUDA_NO_VMM=ON" not in cfg2, "CPU 版不需要 NO_VMM"
        assert r2["cuda"] is False and r2["cuda_no_vmm"] is False

        # CUDA_NO_VMM=False：显式关掉时也不传
        calls.clear()
        clear_bins()
        CFG.update(BUILD_CUDA=True, CUDA_NO_VMM=False)
        r2b = ns["stage_s2_build"](__import__("time").time() + 120)
        assert "-DGGML_CUDA_NO_VMM=ON" not in calls[0], calls[0]
        assert r2b["cuda_no_vmm"] is False

        # nvcc 不可用 + BUILD_CUDA=True → GuardError，且一次 cmake 都没跑
        calls.clear()
        clear_bins()
        ns["_toolchain_report"] = lambda: {"cmake": "cmake version 3.31.0",
                                           "nvcc": "不可用: FileNotFoundError",
                                           "gxx": "11", "gcc": "11"}
        CFG.update(BUILD_CUDA=True)
        try:
            ns["stage_s2_build"](__import__("time").time() + 120)
            raise AssertionError("nvcc 缺失时没有拦住")
        except ns["GuardError"] as e:
            assert "BUILD_CUDA" in str(e)
        assert calls == [], f"拦住之后不该再调用外部命令：{calls}"
    finally:
        ns["run_cmd"], ns["_toolchain_report"] = orig_run, orig_tc
        CFG.update(snap)
        clear_bins()


def test_s2_commit_fallback(ns):
    """源码不是 git 仓库（tarball 解出来的）时，commit 要记成"声明值"而不是 None。

    从 GitHub 的 archive/<sha>.tar.gz 解出来的目录没有 .git，`git rev-parse` 拿不到
    任何东西。如果就这样记 None，S4 的 provenance 里 llama_commit 会空，事后无法
    区分"没记录"和"记录了但没校验"。这里钉住三条：
      1. 有预注册 commit → 记成它，且 commit_source="declared"；
      2. 没有预注册 → commit=None，commit_source=None（不编造）；
      3. 有 git 且与预注册不符 → 必须抛 GuardError。
    """
    CFG = ns["CFG"]
    BUILD = Path(ns["BUILD"])
    work = Path(ns["WORK"])
    src = work / "fake_llama_src_nogit"
    src.mkdir(parents=True, exist_ok=True)
    (src / "CMakeLists.txt").write_text("# fake\n", encoding="utf-8")

    keys = ("LLAMA_SRC", "LLAMA_BIN", "LLAMA_COMMIT", "BUILD_CUDA", "CUDA_ARCH",
            "BUILD_JOBS", "FORCE_RERUN")
    snap = {k: CFG[k] for k in keys}
    orig_run, orig_tc, orig_git = ns["run_cmd"], ns["_toolchain_report"], ns["_git_commit"]
    SHA = "d454a204ef2b63af357744809c05af897611a72e"

    def fake_run(cmd, log_path, deadline, cwd=None, env=None):
        return 0, ""

    def make_bins():
        for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
            p = BUILD / "bin" / t
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            os.chmod(p, 0o755)

    def clear_bins():
        for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
            for c in (BUILD / "bin" / t, BUILD / t):
                if c.exists():
                    c.unlink()

    ns["run_cmd"] = fake_run
    ns["_toolchain_report"] = lambda: {"cmake": "cmake version 3.31.0",
                                       "nvcc": "Cuda compilation tools, release 12.8",
                                       "gxx": "11", "gcc": "11"}
    try:
        CFG.update(LLAMA_SRC=str(src), LLAMA_BIN="", BUILD_CUDA=True,
                   CUDA_ARCH="75", BUILD_JOBS=4, FORCE_RERUN=False)
        make_bins()

        # 1) 无 .git + 有预注册值 → declared
        ns["_git_commit"] = lambda p: None
        CFG.update(LLAMA_COMMIT=SHA)
        r = ns["stage_s2_build"](__import__("time").time() + 120)
        assert r["commit"] == SHA, r["commit"]
        assert r["commit_source"] == "declared", r["commit_source"]

        # 2) 无 .git + 无预注册值 → None，不能编造
        CFG.update(LLAMA_COMMIT="")
        r2 = ns["stage_s2_build"](__import__("time").time() + 120)
        assert r2["commit"] is None, r2["commit"]
        assert r2["commit_source"] is None, r2["commit_source"]

        # 3) 有 git 且不符 → GuardError
        ns["_git_commit"] = lambda p: "0" * 40
        CFG.update(LLAMA_COMMIT=SHA)
        try:
            ns["stage_s2_build"](__import__("time").time() + 120)
            raise AssertionError("commit 不符时没有拦住")
        except ns["GuardError"] as e:
            assert "不符" in str(e), e

        # 4) 有 git 且相符 → commit_source="git"
        ns["_git_commit"] = lambda p: SHA
        CFG.update(LLAMA_COMMIT=SHA[:12])
        r3 = ns["stage_s2_build"](__import__("time").time() + 120)
        assert r3["commit_source"] == "git", r3["commit_source"]
        assert r3["commit"] == SHA
    finally:
        ns["run_cmd"], ns["_toolchain_report"], ns["_git_commit"] = \
            orig_run, orig_tc, orig_git
        CFG.update(snap)
        clear_bins()


# ---------------------------------------------------------------- v0.3 S2 一键取源码

def _make_source_tarball(tmp, dirname=None, marker=b"# fake llama.cpp\n"):
    """造一个"合法"的 llama.cpp 源码 tarball，返回 (tar_path, dirname)。

    sha256 / bytes / 顶层目录名全部按真实文件算，测试里再把 ns["S2_PINNED"]
    指过去 —— 这样走的是真代码路径，只是把"从 GitHub 下载"换成"本地复制"。
    """
    import tarfile as _tf
    import hashlib as _hl

    dirname = dirname or ("llama.cpp-" + "a" * 40)
    tar_path = Path(tmp) / f"{dirname}.tar.gz"
    stage = Path(tmp) / f".stage-{dirname}"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    (stage / dirname / "tools" / "quantize").mkdir(parents=True, exist_ok=True)
    (stage / dirname / "CMakeLists.txt").write_bytes(marker)
    (stage / dirname / "tools" / "quantize" / "quantize.cpp").write_bytes(b"// fake\n")
    with _tf.open(tar_path, "w:gz") as tf:
        for p in sorted((stage / dirname).rglob("*")):
            tf.add(p, arcname=str(p.relative_to(stage)), recursive=False)
    shutil.rmtree(stage, ignore_errors=True)
    return tar_path, dirname


def _pin_to(ns, tar_path, dirname, commit="a" * 40):
    """把 ns["S2_PINNED"] 临时指向本地 tarball，返回 (pin, restore)。"""
    import hashlib as _hl

    sha = _hl.sha256(tar_path.read_bytes()).hexdigest()
    pin = {"commit": commit,
           "url": "https://example.invalid/llama.cpp.tar.gz",
           "sha256": sha, "bytes": tar_path.stat().st_size, "dir": dirname}
    orig = ns["S2_PINNED"]
    ns["S2_PINNED"] = pin
    return pin, (lambda: ns.__setitem__("S2_PINNED", orig))


def _fake_downloader(src_file):
    """假装成 _http_download：把本地文件复制到 dst，返回与真函数同形的字典。"""
    import hashlib as _hl

    state = {"calls": 0}

    def fake(url, dst, deadline, expect_bytes=None, timeout=60, chunk=1 << 20):
        state["calls"] += 1
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = Path(src_file).read_bytes()
        dst.write_bytes(data)
        return {"bytes": len(data), "sha256": _hl.sha256(data).hexdigest(),
                "path": str(dst), "seconds": 0.0}

    fake.state = state
    return fake


def test_s2_default_off_no_network(ns):
    """默认 RUN_S2=False / RUN_S3=False / RUN_S4=False：一个字段都不改，也不许联网。

    Kaggle 上"Run All"不应该因为导入 notebook 就悄悄下载。想下载必须显式把
    开关改成 True —— 这一点由 _apply_run_switches 的早退保证。
    """
    assert ns["CFG"]["RUN_S2"] is False, "默认值必须是 False"
    assert ns["CFG"]["RUN_S3"] is False, "默认值必须是 False"
    assert ns["CFG"]["RUN_S4"] is False, "默认值必须是 False（不许默认下 15 GiB BF16）"
    before_stages = dict(ns["STAGES"])
    before_cfg = {k: ns["CFG"][k] for k in ("S2_AUTO_FETCH_SOURCE", "LLAMA_COMMIT",
                                           "CUDA_ARCH", "BUILD_JOBS", "BUILD_CUDA",
                                           "S3_AUTO_FETCH_MODEL", "S3_AUTO_FETCH_CALIB",
                                           "S4_AUTO_FETCH_BASE", "S4_RECLAIM_Q8_CACHE",
                                           "S3_SKIP_IF_IMATRIX_OK")}
    out = ns["_apply_run_switches"]()
    assert out == {"run_s2": False, "run_s3": False, "run_s4": False,
                   "run_eval": False}, out
    assert ns["STAGES"] == before_stages, "三个开关都 False 时不许改 STAGES"
    for k, v in before_cfg.items():
        assert ns["CFG"][k] == v, f"三个开关都 False 时不许改 CFG['{k}']"

    # 没有手动源码 + 没开自动获取 → GuardError，且消息里要提到 RUN_S2
    CFG = ns["CFG"]
    snap = dict(CFG)
    orig_fetch = ns["_fetch_pinned_source"]

    def boom(deadline):
        raise AssertionError("RUN_S2=False 时不允许触发下载")

    ns["_fetch_pinned_source"] = boom
    try:
        CFG.update(LLAMA_SRC="", S2_AUTO_FETCH_SOURCE=False)
        try:
            ns["_find_llama_src"](__import__("time").time() + 30)
            raise AssertionError("没有拦住")
        except ns["GuardError"] as e:
            assert "RUN_S2" in str(e), e
            assert "Internet" in str(e), e
    finally:
        ns["_fetch_pinned_source"] = orig_fetch
        CFG.update(snap)


def test_s2_run_switch_only_turns_on_s2(ns):
    """RUN_S2=True 只开 S2；S3–S8 必须保持 False（本任务不授权自动跑实验）。"""
    STAGES, CFG = ns["STAGES"], ns["CFG"]
    snap_s = dict(STAGES)
    snap_c = {k: CFG[k] for k in ("S2_AUTO_FETCH_SOURCE", "LLAMA_COMMIT",
                                  "CUDA_ARCH", "BUILD_JOBS", "BUILD_CUDA")}
    orig = ns["CFG"]["RUN_S2"]
    try:
        ns["CFG"]["RUN_S2"] = True
        out = ns["_apply_run_switches"]()
        assert out["run_s2"] is True
        assert STAGES["S2_BUILD"] is True
        for k in ("S3_IMATRIX", "S4_QUANT", "S5_BYTES", "S6_NLL",
                  "S7_QUALITY", "S8_BF16_ANCHOR", "SX_EXPLORE"):
            assert STAGES[k] is False, f"{k} 被 RUN_S2 顺手打开了"
        assert CFG["S2_AUTO_FETCH_SOURCE"] is True
        assert CFG["LLAMA_COMMIT"] == ns["S2_PINNED"]["commit"]
        assert CFG["CUDA_ARCH"] == "75"
        assert CFG["BUILD_JOBS"] == 4
        assert CFG["BUILD_CUDA"] is True
        assert out["commit"] == ns["S2_PINNED"]["commit"]
        assert out["stages"] == [k for k, v in STAGES.items() if v]
    finally:
        ns["CFG"]["RUN_S2"] = orig
        ns["_apply_run_switches"]()
        STAGES.clear(); STAGES.update(snap_s)
        CFG.update(snap_c)


def test_s2_download_checksum_mismatch(ns):
    """digest 与预登记值不符 → 拒绝解压、删掉半成品、不留源码目录。"""
    import tempfile as _tf

    SRC_CACHE, DL_DIR = Path(ns["SRC_CACHE"]), Path(ns["DL_DIR"])
    tmp = _tf.mkdtemp(prefix="trlab_s2_ck_")
    tar_path, dirname = _make_source_tarball(tmp)
    pin, restore = _pin_to(ns, tar_path, dirname)
    pin["sha256"] = "0" * 64                      # 故意对不上

    orig_dl = ns["_http_download"]
    dl = _fake_downloader(tar_path)
    ns["_http_download"] = dl
    orig_cfg = dict(ns["CFG"])
    try:
        ns["CFG"]["S2_AUTO_FETCH_SOURCE"] = True
        try:
            ns["_fetch_pinned_source"](__import__("time").time() + 60)
            raise AssertionError("sha256 不符却没有拦住")
        except ns["GuardError"] as e:
            assert "sha256" in str(e), e
        assert dl.state["calls"] == 1
        # 下载到的包必须被删掉，不能留着下次误用
        expect_tar = DL_DIR / f"llama.cpp-{pin['commit'][:12]}.tar.gz"
        assert not expect_tar.exists(), "digest 不符的包必须删除"
        assert not (SRC_CACHE / pin["dir"]).exists(), "不许解压"
    finally:
        ns["_http_download"] = orig_dl
        ns["CFG"].update(orig_cfg)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s2_extract_rejects_traversal(ns):
    """归档里有 `..` / 绝对路径 / 链接 → 拒绝解压，不"尽量解压"。"""
    import tarfile as _tf
    import tempfile as _tf2

    tmp = _tf2.mkdtemp(prefix="trlab_s2_trav_")
    bad = Path(tmp) / "bad.tar.gz"
    with _tf.open(bad, "w:gz") as tf:
        info = _tf.TarInfo("../../evil.txt")
        data = b"pwned\n"
        info.size = len(data)
        tf.addfile(info, __import__("io").BytesIO(data))
    try:
        ns["_safe_extract"](bad, Path(tmp) / "out")
        raise AssertionError("没有拦截 ../ 越界条目")
    except ns["GuardError"] as e:
        assert "evil.txt" in str(e), e
    shutil.rmtree(tmp, ignore_errors=True)


def test_s2_repeat_run_uses_cache(ns):
    """重复 Run All：第二次完全离线复用；包在只解压；包被改坏就重下。"""
    import tempfile as _tf

    SRC_CACHE, DL_DIR = Path(ns["SRC_CACHE"]), Path(ns["DL_DIR"])
    tmp = _tf.mkdtemp(prefix="trlab_s2_cache_")
    tar_path, dirname = _make_source_tarball(tmp)
    pin, restore = _pin_to(ns, tar_path, dirname)

    orig_dl = ns["_http_download"]
    dl = _fake_downloader(tar_path)
    ns["_http_download"] = dl
    orig_cfg = dict(ns["CFG"])
    deadline = __import__("time").time() + 120
    try:
        ns["CFG"]["S2_AUTO_FETCH_SOURCE"] = True

        src1, m1 = ns["_fetch_pinned_source"](deadline)
        assert m1["origin"] == "downloaded" and m1["sha256_ok"] is True
        assert Path(src1) == SRC_CACHE / pin["dir"], f"源码必须落在固定目录：{src1}"
        assert (Path(src1) / "CMakeLists.txt").exists()
        assert dl.state["calls"] == 1

        # 2) 源码目录 + stamp 都在 → 离线复用，零下载
        src2, m2 = ns["_fetch_pinned_source"](deadline)
        assert m2["origin"] == "cache" and m2["archive_cached"] is True
        assert Path(src2) == Path(src1)
        assert dl.state["calls"] == 1, "缓存命中时不该再下载"

        # 3) 删掉源码目录、保留 tarball → 只解压，不重下
        shutil.rmtree(SRC_CACHE / pin["dir"], ignore_errors=True)
        src3, m3 = ns["_fetch_pinned_source"](deadline)
        assert m3["origin"] == "cached_archive", m3
        assert dl.state["calls"] == 1, "tarball digest 一致时不该重下"
        assert (Path(src3) / "CMakeLists.txt").exists()

        # 4) tarball 被改坏（同尺寸）→ digest 不符，删掉重下
        expect_tar = DL_DIR / f"llama.cpp-{pin['commit'][:12]}.tar.gz"
        raw = bytearray(expect_tar.read_bytes())
        raw[-1] ^= 0xFF
        expect_tar.write_bytes(bytes(raw))
        shutil.rmtree(SRC_CACHE / pin["dir"], ignore_errors=True)
        src4, m4 = ns["_fetch_pinned_source"](deadline)
        assert m4["origin"] == "downloaded", m4
        assert dl.state["calls"] == 2, "包坏了必须重下"
        assert (Path(src4) / "CMakeLists.txt").exists()
    finally:
        ns["_http_download"] = orig_dl
        ns["CFG"].update(orig_cfg)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s2_dir_name_mismatch_fails(ns):
    """digest 对得上但顶层目录名与预登记不符 → 拒绝，说明 pin 信息本身有问题。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s2_dir_")
    tar_path, real_dir = _make_source_tarball(tmp)
    pin, restore = _pin_to(ns, tar_path, real_dir)
    pin["dir"] = "llama.cpp-" + "b" * 40          # 声明的目录名与归档不符

    orig_dl = ns["_http_download"]
    ns["_http_download"] = _fake_downloader(tar_path)
    orig_cfg = dict(ns["CFG"])
    try:
        ns["CFG"]["S2_AUTO_FETCH_SOURCE"] = True
        try:
            ns["_fetch_pinned_source"](__import__("time").time() + 60)
            raise AssertionError("目录名不符却没有拦住")
        except ns["GuardError"] as e:
            assert "顶层目录" in str(e), e
    finally:
        ns["_http_download"] = orig_dl
        ns["CFG"].update(orig_cfg)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s2_commit_from_pinned_tarball(ns):
    """tarball 来源：没有 .git，但归档 digest 已校验 → pinned_tarball（不是 declared）。"""
    import tempfile as _tf

    SRC_CACHE = Path(ns["SRC_CACHE"])
    tmp = _tf.mkdtemp(prefix="trlab_s2_commit_")
    tar_path, dirname = _make_source_tarball(tmp)
    pin, restore = _pin_to(ns, tar_path, dirname)

    orig_dl, orig_git = ns["_http_download"], ns["_git_commit"]
    ns["_http_download"] = _fake_downloader(tar_path)
    orig_cfg = dict(ns["CFG"])
    orig_run, orig_tc = ns["run_cmd"], ns["_toolchain_report"]
    BUILD = Path(ns["BUILD"])

    def fake_run(cmd, log_path, deadline, cwd=None, env=None):
        if "--build" in cmd:
            for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
                p = BUILD / "bin" / t
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                os.chmod(p, 0o755)
        return 0, ""

    ns["run_cmd"] = fake_run
    ns["_toolchain_report"] = lambda: {"cmake": "cmake version 3.31.0",
                                       "nvcc": "Cuda compilation tools, release 12.8",
                                       "gxx": "11", "gcc": "11"}
    try:
        ns["CFG"].update(LLAMA_SRC="", LLAMA_BIN="", S2_AUTO_FETCH_SOURCE=True,
                        LLAMA_COMMIT="", BUILD_CUDA=False, CUDA_ARCH="",
                        BUILD_JOBS=4, FORCE_RERUN=True)
        ns["_git_commit"] = lambda p: None
        r = ns["stage_s2_build"](__import__("time").time() + 120)
        assert r["commit"] == pin["commit"], r["commit"]
        assert r["commit_source"] == "pinned_tarball", r["commit_source"]
        assert r["commit_verified"] is True, r
        assert r["source"]["origin"] == "downloaded"
        assert r["source"]["archive_sha256"] == pin["sha256"]
        assert r["bins"]["llama-quantize"].endswith("llama-quantize")
        assert r["disk_free_gb"] > 0

        # 有 git 且与 pin 一致 → 走 git 来源
        ns["_git_commit"] = lambda p: pin["commit"]
        ns["CFG"]["LLAMA_COMMIT"] = pin["commit"]
        r2 = ns["stage_s2_build"](__import__("time").time() + 120)
        assert r2["commit_source"] == "git" and r2["commit_verified"] is True

        # 有 git 但与 pin 不符 → 必须拒绝
        ns["_git_commit"] = lambda p: "c" * 40
        try:
            ns["stage_s2_build"](__import__("time").time() + 120)
            raise AssertionError("commit 不符却没有拦住")
        except ns["GuardError"] as e:
            assert "不符" in str(e), e
    finally:
        ns["run_cmd"], ns["_toolchain_report"] = orig_run, orig_tc
        ns["_git_commit"] = orig_git
        ns["_http_download"] = orig_dl
        ns["CFG"].update(orig_cfg)
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s2_failure_message_actionable(ns):
    """编译失败要给"症状 → 处置"，不能把几十行 cmake 输出原样丢给人。"""
    f = ns["_explain_s2_failure"]
    CF = ns["CommandFailed"]

    def mk(tail):
        return CF(["cmake", "--build", "x"], 2, tail, "/tmp/s2_build.log")

    r = f(mk("error: unsupported gnu version! gcc versions later than 13 are not supported"))
    assert "BUILD_CUDA" in r and "False" in r, r
    assert "/tmp/s2_build.log" in r
    r = f(mk("nvcc fatal   : Could not find nvcc"))
    assert "BUILD_CUDA" in r, r
    r = f(mk("cmake error at ggml/CMakeLists.txt:100"))
    assert "CMakeLists.txt" in r or "配置阶段" in r, r
    r = f(mk("No space left on device"))
    assert "磁盘" in r, r
    r = f(mk("c++: internal compiler error: Killed (program cc1plus)"))
    assert "BUILD_JOBS" in r, r
    r = f(mk("some totally unknown failure"))
    assert "日志" in r, r


def test_s2_failure_classifier_cuda_driver(ns):
    """Kaggle 真实失败：CUDA::cuda_driver target 找不到，不能被误判成"找不到 nvcc"。

    用户 2026-10-21 的真实日志尾部同时含 "nvcc"（工具链检测行）与
    "but the target was not found"，v0.3.0 的全文包含判断把它归类成
    "找不到 nvcc" 并劝用户关 GPU —— 而 nvcc 12.8.93 明明检测通过了。
    """
    f = ns["_explain_s2_failure"]
    CF = ns["CommandFailed"]
    real_tail = (
        "-- The CUDA compiler identification is CUDA 12.8.93\n"
        "-- Check for working compiler: /usr/local/cuda/bin/nvcc\n"
        "-- Detecting CUDA compiler ABI info - done\n"
        "-- Including CUDA backend\n"
        "CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:181 (target_link_libraries):\n"
        "  Target \"ggml-cuda\" links to: CUDA::cuda_driver\n"
        "  but the target was not found.\n"
        "CMake Generate step failed.\n"
        "  Error running CMake.\n")
    r = f(CF(["cmake", "-S", "x", "-B", "y"], 1, real_tail, "/tmp/s2_cmake.log"))
    assert "CUDA::cuda_driver" in r, r
    assert "GGML_CUDA_NO_VMM" in r, r
    assert "找不到 nvcc" not in r, f"误判成 nvcc 缺失：{r}"
    assert "BUILD_CUDA']=False" not in r, "这条错误不该劝用户关 GPU"

    # 真正的 nvcc 缺失仍然要判成 nvcc（同行匹配，别把 driver 分支吃成 nvcc）
    r2 = f(CF(["cmake"], 1,
              "CMake Error at CMakeLists.txt:20 (enable_language):\n"
              "  No CMAKE_CUDA_COMPILER could be found.\n", "/tmp/x.log"))
    assert "找不到 nvcc" in r2 or "BUILD_CUDA" in r2, r2

    r3 = f(CF(["cmake"], 1,
              "-- Could NOT find nvcc (missing: CUDA_NVCC_EXECUTABLE)\n", "/tmp/x.log"))
    assert "找不到 nvcc" in r3, r3


def test_s2_fingerprint_binds_cuda_no_vmm(ns):
    """CUDA_NO_VMM 必须进 S2 指纹：改了开关不能误 RESUME 上一轮 build。"""
    CFG = ns["CFG"]
    snap = CFG["CUDA_NO_VMM"]
    try:
        CFG["CUDA_NO_VMM"] = True
        a = ns["_fingerprint"]("S2_BUILD")
        CFG["CUDA_NO_VMM"] = False
        b = ns["_fingerprint"]("S2_BUILD")
        assert a != b, "CUDA_NO_VMM 不在 S2 指纹里，改开关会复用旧 build"
        CFG["CUDA_NO_VMM"] = True
        assert ns["_fingerprint"]("S2_BUILD") == a, "改回去必须还原"
    finally:
        CFG["CUDA_NO_VMM"] = snap


def test_s2_probe_version_reads_stderr(ns):
    """`llama-cli --version` 把 build info 写到 stderr（build-info.h:13 默认 FILE*=stderr），
    只取 stdout 会得到空串 —— v0.3.0 真实 Kaggle 运行就是这样。"""
    probe = ns["_probe_version"]
    orig_run = ns["subprocess"].run

    class R:
        def __init__(self, out, err, rc=0):
            self.stdout, self.stderr, self.returncode = out, err, rc

    try:
        ns["subprocess"].run = lambda *a, **k: R("", "build: 1a2b3c (1234)\n")
        v = probe("/fake/llama-cli")
        assert "1a2b3c" in v, f"stderr 没被接上：{v!r}"

        ns["subprocess"].run = lambda *a, **k: R("version 0.3.2\n", "")
        assert "0.3.2" in probe("/fake/llama-cli")

        ns["subprocess"].run = lambda *a, **k: R("", "")
        assert "rc=" in probe("/fake/llama-cli"), "两个流都空时要给出 rc 而不是空串"

        def boom(*a, **k):
            raise OSError("no such file")
        ns["subprocess"].run = boom
        assert "不可用" in probe("/nonexistent/llama-cli")
    finally:
        ns["subprocess"].run = orig_run


def test_s2_network_failure_message(ns):
    """真的连不上时给一句人话，且不留下 .partial。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s2_net_")
    dst = Path(tmp) / "llama.tar.gz"
    try:
        try:
            # 127.0.0.1:9 必然拒绝连接；不碰外网
            ns["_http_download"]("http://127.0.0.1:9/llama.cpp.tar.gz", dst,
                                 __import__("time").time() + 30)
            raise AssertionError("连不上却没有报错")
        except ns["GuardError"] as e:
            assert "Internet" in str(e), e
        assert not dst.exists(), "不许留下半成品"
        assert not dst.with_name(dst.name + ".partial").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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




# ---------------------------------------------------------------- S3 一键化（v0.4）

def _make_fake_bins(dirpath, runnable=True):
    """在 dirpath 下造四个"能跑/不能跑"的假 llama 二进制。返回路径列表。"""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"):
        p = d / name
        # 刻意写到 stderr：真实 llama-cli --version 就是把 build info 写 stderr
        p.write_text("#!/bin/sh\necho 'version: fake build' 1>&2\nexit 0\n",
                     encoding="utf-8")
        p.chmod(0o755 if runnable else 0o644)
        out.append(p)
    return out


def _pin_blob(ns, src_file, sha=None, bytes_=None, url="https://example.invalid/blob"):
    """把 ns['S3_PINNED'] / ns['CALIB_PINNED'] 指向本地文件，返回 (pin, restore)。"""
    import hashlib as _hl

    data = Path(src_file).read_bytes()
    pin = {"repo": "fake/repo", "revision": "b" * 40, "filename": Path(src_file).name,
           "url": url, "sha256": sha or _hl.sha256(data).hexdigest(),
           "bytes": bytes_ if bytes_ is not None else len(data),
           "kind": "q8_0", "etag_header": "x-linked-etag"}
    orig = ns["S3_PINNED"]
    ns["S3_PINNED"] = pin
    return pin, (lambda: ns.__setitem__("S3_PINNED", orig))


def _fake_stream(src_file):
    """假装成 _stream_blob：把本地文件从 offset 处追加写进 path。

    记录每次调用的 offset，用来验证断点续传真的发了 Range。
    """
    state = {"calls": [], "offsets": []}

    def fake(url, path, offset, hasher, deadline, tag, expect_bytes, headers=None):
        state["calls"].append({"url": url, "offset": offset,
                               "path": str(path), "tag": tag})
        state["offsets"].append(offset)
        data = Path(src_file).read_bytes()
        with open(path, "ab") as f:
            f.write(data[offset:])
            hasher.update(data[offset:])
        return 206 if offset else 200, len(data) - offset, len(data), None

    fake.state = state
    return fake


def test_s3_run_switch_scope(ns):
    """RUN_S3=True 只开 S2+S3；S4–S8 必须全部保持 False。"""
    STAGES, CFG = ns["STAGES"], ns["CFG"]
    snap_s, snap_c = dict(STAGES), dict(CFG)
    orig = ns["CFG"]["RUN_S3"]
    try:
        ns["CFG"]["RUN_S3"] = True
        out = ns["_apply_run_switches"]()
        assert out["run_s3"] is True
        assert STAGES["S2_BUILD"] is True, "RUN_S3 必须隐含打开 S2（imatrix 要二进制）"
        assert STAGES["S3_IMATRIX"] is True
        for k in ("S4_QUANT", "S5_BYTES", "S6_NLL", "S7_QUALITY",
                  "S8_BF16_ANCHOR", "SX_EXPLORE"):
            assert STAGES[k] is False, f"{k} 被 RUN_S3 顺手打开了"
        assert CFG["S3_AUTO_FETCH_MODEL"] is True
        assert CFG["S3_AUTO_FETCH_CALIB"] is True
        assert CFG["NGL_MARGIN_GB"] == 2.0, "v0.4 起 -ngl 余量至少 2 GiB"
        assert CFG["S3_DELETE_Q8_AFTER"] is True
        assert CFG["BASE_GGUF"] == "", "RUN_S3 不许设 BASE_GGUF（S4 是关闭的）"
        assert out["imatrix_base"] == ns["S3_PINNED"]["filename"]
    finally:
        ns["CFG"]["RUN_S3"] = orig
        ns["_apply_run_switches"]()
        STAGES.clear(); STAGES.update(snap_s)
        CFG.clear(); CFG.update(snap_c)


def test_s3_pins_are_registered(ns):
    """Q8_0 与 wikitext 的预登记值必须与 HF 上核对过的一致（防手滑改坏）。"""
    q8, cal = ns["S3_PINNED"], ns["CALIB_PINNED"]
    assert q8["bytes"] == 8709519168, q8["bytes"]
    assert q8["sha256"] == ("0cfbf745760f07a76ddeb358dd025a27"
                           "f2e11d1ca9c9a4169a373d52990fe86e")
    assert q8["revision"] == "a6adef130ffb23ddaf1a62fec9dced968c9bc482"
    assert q8["url"].endswith(f"/resolve/{q8['revision']}/{q8['filename']}")
    assert cal["bytes"] == 6357543, cal["bytes"]
    assert cal["sha256"] == ("e83889baabc497075506f91975be5fac"
                            "0d45c5290b6b20582c8cd1e853d0c9f7")
    assert cal["split"] == "train", "校准必须来自 train split（与 S6 的 test 隔离）"


def test_s3_fresh_session_no_binaries(ns):
    """全新 Session、四个二进制都不在 → 必须走到"下载源码 + 编译"那条路。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_fresh_")
    orig_build, orig_in = ns["BUILD"], ns["IN"]
    ns["BUILD"] = Path(tmp) / "build-llama"
    ns["IN"] = Path(tmp) / "input"
    try:
        bins, origin, why = ns["_find_llama_bins"]()
        assert bins is None, "没有二进制时不该返回可复用的构建"
        assert origin is None
        assert why and len(why) > 10, "必须说清为什么没找到"

        # 探活没结果 → stage_s2_build 应该继续走 _find_llama_src（下载源码）
        sentinel = "REACHED_SOURCE_DOWNLOAD_PATH"

        def boom(deadline):
            raise RuntimeError(sentinel)

        orig_src = ns["_find_llama_src"]
        ns["_find_llama_src"] = boom
        try:
            try:
                ns["stage_s2_build"](time.time() + 60)
                raise AssertionError("应当继续走下载源码的分支")
            except RuntimeError as e:
                assert sentinel in str(e), e
        finally:
            ns["_find_llama_src"] = orig_src
    finally:
        ns["BUILD"], ns["IN"] = orig_build, orig_in
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_existing_build_reused(ns):
    """WORK/build-llama 里有四个二进制且跑得通 → 零编译复用，不下载源码。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_exist_")
    orig_build, orig_in = ns["BUILD"], ns["IN"]
    ns["BUILD"] = Path(tmp) / "build-llama"
    ns["IN"] = Path(tmp) / "input"
    try:
        _make_fake_bins(ns["BUILD"] / "bin", runnable=True)
        called = {"src": False}

        def boom(deadline):
            called["src"] = True
            raise AssertionError("复用二进制时不该下载源码")

        orig_src = ns["_find_llama_src"]
        ns["_find_llama_src"] = boom
        try:
            r = ns["stage_s2_build"](time.time() + 120)
        finally:
            ns["_find_llama_src"] = orig_src
        assert called["src"] is False
        assert r["reused"] is True
        assert r["build_reuse"] == "existing_build", r["build_reuse"]
        assert len(r["bins"]) == 4
        # 没有源码 → commit 只能记为声明值，不许谎称已验证
        assert r["commit_source"] == "declared", r["commit_source"]
        assert r["commit_verified"] is False
        assert r["version"], "复用也必须探到版本串"
    finally:
        ns["BUILD"], ns["IN"] = orig_build, orig_in
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_attached_input_reused(ns):
    """/kaggle/input 下挂着上次的 Output → 自动扫到并用起来，不用填路径。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_attach_")
    orig_build, orig_in = ns["BUILD"], ns["IN"]
    ns["BUILD"] = Path(tmp) / "build-llama"          # 空的：WORK 里没有
    ns["IN"] = Path(tmp) / "input"
    try:
        # Kaggle 会把挂载的东西放在 /kaggle/input/<slug>/...，路径深度不确定
        deep = ns["IN"] / "my-notebook-output" / "trlab" / "build-llama" / "bin"
        _make_fake_bins(deep, runnable=True)

        bins, origin, detail = ns["_find_llama_bins"]()
        assert bins is not None, f"应当自动扫到挂载的二进制：{detail}"
        assert origin == "attached_input", origin
        assert bins["llama-cli"] == deep / "llama-cli"

        r = ns["stage_s2_build"](time.time() + 120)
        assert r["build_reuse"] == "attached_input"
        assert r["reused"] is True
        # 复用挂载二进制时源码不在本 Session 内 → 必须标 declared，不能假装 git 校验
        assert r["commit_source"] == "declared"
        assert r["commit_verified"] is False
        assert "源码不在本 Session 内" in (r["source"].get("note") or "")
    finally:
        ns["BUILD"], ns["IN"] = orig_build, orig_in
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_probe_rejects_unrunnable(ns):
    """二进制存在但跑不起来（chmod 644）→ 不算可复用，必须继续往下找。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_dead_")
    orig_build, orig_in = ns["BUILD"], ns["IN"]
    ns["BUILD"] = Path(tmp) / "build-llama"
    ns["IN"] = Path(tmp) / "input"
    try:
        _make_fake_bins(ns["BUILD"] / "bin", runnable=False)
        ok, detail = ns["_probe_bin_runnable"](ns["BUILD"] / "bin" / "llama-cli")
        assert ok is False, f"不可执行的文件必须判为跑不起来：{detail}"
        bins, origin, why = ns["_find_llama_bins"]()
        assert bins is None, "跑不起来的二进制不能被复用"
        assert "跑不起来" in why, why
    finally:
        ns["BUILD"], ns["IN"] = orig_build, orig_in
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_s2_failure_blocks_s3(ns):
    """S2 失败 → S3 必须 SKIP，不许启动后报一句"找不到 llama-imatrix"。"""
    STAGES, STATE = ns["STAGES"], ns["STATE"]
    snap_s = dict(STAGES)
    snap_state = dict(STATE)
    orig_req = ns["STAGE_REQUIRES_OK"]
    try:
        STAGES["S2_BUILD"] = True
        STAGES["S3_IMATRIX"] = True
        STATE["S2_BUILD"] = {"status": "error",
                             "error": "RuntimeError: CUDA::cuda_driver target not found"}
        called = {"s3": False}

        def fn(deadline):
            called["s3"] = True
            return {}

        out = ns["run_stage"]("S3_IMATRIX", fn)
        assert called["s3"] is False, "S2 失败了 S3 不该执行"
        assert out is None
        assert STATE.get("S3_IMATRIX") is None, "SKIP 不该写进 STATE"
    finally:
        STAGES.clear(); STAGES.update(snap_s)
        STATE.clear(); STATE.update(snap_state)
        ns["STAGE_REQUIRES_OK"] = orig_req


def test_s3_restart_after_disconnect(ns):
    """掉线后重启：STATE 清空（新 Session）但磁盘上的二进制还在 → 直接复用。

    Kaggle 交互会话空闲 20 分钟就断。断掉之后内存里的 STATE 没了，
    但 /kaggle/working 里的文件可能还在。本测试钉住"文件在就能复用、
    且不依赖任何旧内存状态"这条。
    """
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_restart_")
    orig_build, orig_in = ns["BUILD"], ns["IN"]
    ns["BUILD"] = Path(tmp) / "build-llama"
    ns["IN"] = Path(tmp) / "input"
    try:
        _make_fake_bins(ns["BUILD"] / "bin", runnable=True)
        # 模拟"新 Session"：STATE 完全为空
        ns["STATE"].clear()
        assert ns["STATE"] == {}

        r = ns["stage_s2_build"](time.time() + 120)
        assert r["reused"] is True
        assert r["build_reuse"] == "existing_build"
        # 复用时 _find_llama_src 一次都不该被调用（不下载 36.6 MiB 源码）
        assert r["source"]["origin"] == "existing_build"
    finally:
        ns["BUILD"], ns["IN"] = orig_build, orig_in
        shutil.rmtree(tmp, ignore_errors=True)


def _prep_model_dl(ns, src_file):
    """把 MODEL_DL 指到临时目录并返回 (tmp, restore)。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_dl_")
    orig = ns["MODEL_DL"]
    ns["MODEL_DL"] = Path(tmp) / "models"
    return Path(tmp), (lambda: ns.__setitem__("MODEL_DL", orig))


def test_s3_disk_precheck_insufficient(ns):
    """磁盘不够 → 下载之前就拒，一个字节都不下。"""
    tmp, restore = _prep_model_dl(ns, None)
    try:
        orig_disk = ns["disk_gb"]
        ns["disk_gb"] = lambda p: {"total": 20.0, "used": 19.5, "free": 0.5}
        called = {"stream": False}

        def boom(*a, **k):
            called["stream"] = True
            raise AssertionError("磁盘不够时不该开始下载")

        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = boom
        try:
            try:
                ns["_disk_precheck"](ns["S3_PINNED"]["bytes"], "S3 下载")
                raise AssertionError("磁盘不足时必须抛 GuardError")
            except ns["GuardError"] as e:
                assert "磁盘不够" in str(e), e
            assert called["stream"] is False

            # 完整路径：_ensure_q8 也必须被同一道预检拦下
            try:
                ns["_ensure_q8"](time.time() + 60)
                raise AssertionError("_ensure_q8 必须被磁盘预检拦下")
            except ns["GuardError"] as e:
                assert "磁盘不够" in str(e), e
            assert called["stream"] is False
        finally:
            ns["_stream_blob"] = orig_stream
            ns["disk_gb"] = orig_disk
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_checksum_mismatch(ns):
    """下载完 sha256 对不上 → 删文件、删 sidecar、报错，不留半成品。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(4096))
    pin, restore_pin = _pin_blob(ns, src)
    pin["sha256"] = "1" * 64                      # 故意对不上
    try:
        fake = _fake_stream(src)
        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = fake
        try:
            dst = ns["_s3_base_path"]()
            try:
                ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
                raise AssertionError("sha256 不符必须抛 GuardError")
            except ns["GuardError"] as e:
                assert "sha256 不匹配" in str(e), e
            assert not dst.exists(), "校验不过不许留下成品"
            assert not dst.with_name(dst.name + ".partial").exists(), "半成品也必须删"
            assert not dst.with_name(dst.name + ".partial.json").exists()
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_etag_mismatch_refuses(ns):
    """响应头 x-linked-etag 与预登记值不一致 → 拒收（上游换过文件）。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(2048))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        def fake(url, path, offset, hasher, deadline, tag, expect_bytes, headers=None):
            data = Path(src).read_bytes()
            with open(path, "ab") as f:
                f.write(data)
                hasher.update(data)
            return 200, len(data), len(data), "f" * 64   # 错的 etag

        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = fake
        try:
            dst = ns["_s3_base_path"]()
            try:
                ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
                raise AssertionError("etag 不符必须抛 GuardError")
            except ns["GuardError"] as e:
                assert "x-linked-etag" in str(e), e
            assert not dst.exists()
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_no_duplicate_full_copy(ns):
    """下载全程磁盘上只有一份完整副本（不允许缓存一份 + working 一份）。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(8192))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        seen = {"max_big": 0, "files_at_end": None}
        real = _fake_stream(src)

        def spy(url, path, offset, hasher, deadline, tag, expect_bytes, headers=None):
            r = real(url, path, offset, hasher, deadline, tag, expect_bytes)
            big = [p.name for p in Path(path).parent.iterdir()
                   if p.is_file() and p.stat().st_size >= expect_bytes * 0.5]
            seen["max_big"] = max(seen["max_big"], len(big))
            return r

        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = spy
        try:
            dst = ns["_s3_base_path"]()
            meta = ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
            assert meta["sha256"] == pin["sha256"]
            seen["files_at_end"] = sorted(p.name for p in dst.parent.iterdir())
            assert seen["max_big"] <= 1, f"同时存在多个大文件：{seen}"
            assert seen["files_at_end"] == [dst.name], seen["files_at_end"]
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_resume_from_partial(ns):
    """中断后重跑：从 .partial 的偏移处续传，不把 8 GiB 重下一遍。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(16384))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        dst = ns["_s3_base_path"]()
        part = dst.with_name(dst.name + ".partial")
        side = dst.with_name(dst.name + ".partial.json")
        part.parent.mkdir(parents=True, exist_ok=True)
        cut = 6000
        part.write_bytes(src.read_bytes()[:cut])
        import hashlib as _hl
        ns["_write_stamp"](side, url=pin["url"], bytes_done=cut,
                           sha256_prefix=_hl.sha256(src.read_bytes()[:cut]).hexdigest(),
                           expect_bytes=pin["bytes"], expect_sha256=pin["sha256"])

        fake = _fake_stream(src)
        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = fake
        try:
            meta = ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
            assert fake.state["offsets"] == [cut], fake.state["offsets"]
            assert meta["sha256"] == pin["sha256"]
            assert dst.read_bytes() == src.read_bytes()
            assert not part.exists() and not side.exists()
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_bad_partial_restarts(ns):
    """.partial 与 sidecar 对不上（前缀哈希不符）→ 不信它，从头下。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(8192))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        dst = ns["_s3_base_path"]()
        part = dst.with_name(dst.name + ".partial")
        side = dst.with_name(dst.name + ".partial.json")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"garbage that does not match the sidecar")
        ns["_write_stamp"](side, url=pin["url"], bytes_done=42,
                           sha256_prefix="0" * 64,
                           expect_bytes=pin["bytes"], expect_sha256=pin["sha256"])
        fake = _fake_stream(src)
        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = fake
        try:
            meta = ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
            assert fake.state["offsets"] == [0], fake.state["offsets"]
            assert meta["sha256"] == pin["sha256"]
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_download_network_failure_keeps_partial(ns):
    """网络失败 → 保留 .partial 与 sidecar，消息说清"重跑会续传"。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(4096))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        dst = ns["_s3_base_path"]()
        part = dst.with_name(dst.name + ".partial")
        side = dst.with_name(dst.name + ".partial.json")
        part.parent.mkdir(parents=True, exist_ok=True)

        def boom(url, path, offset, hasher, deadline, tag, expect_bytes, headers=None):
            with open(path, "ab") as f:
                f.write(b"half")
                hasher.update(b"half")
            raise OSError("connection reset by peer")

        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = boom
        try:
            try:
                ns["_fetch_pinned_blob"](pin, dst, time.time() + 60, "S3/test")
                raise AssertionError("网络失败必须抛 GuardError")
            except ns["GuardError"] as e:
                assert "续传" in str(e), e
                assert "Internet" in str(e), e
            assert part.exists(), "必须保留断点"
            assert side.exists(), "必须保留 sidecar"
            assert side.read_text(encoding="utf-8").find("bytes_done") > 0
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_repeat_run_uses_cache(ns):
    """已经下载并校验过 → 第二次 _ensure_q8 一次都不下载。"""
    tmp, restore = _prep_model_dl(ns, None)
    src = tmp / "payload.bin"
    src.write_bytes(os.urandom(4096))
    pin, restore_pin = _pin_blob(ns, src)
    try:
        fake = _fake_stream(src)
        orig_stream = ns["_stream_blob"]
        ns["_stream_blob"] = fake
        try:
            p1, m1 = ns["_ensure_q8"](time.time() + 60)
            assert m1["origin"] == "downloaded"
            assert fake.state["offsets"] == [0]
            p2, m2 = ns["_ensure_q8"](time.time() + 60)
            assert p1 == p2
            assert m2["origin"] == "cache"
            assert fake.state["offsets"] == [0], "第二次不该再下载"
        finally:
            ns["_stream_blob"] = orig_stream
    finally:
        restore_pin()
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_ngl_ladder(ns):
    """-ngl 阶梯：99 折算成真实层数再对折；0 只有一级。"""
    f = ns["_ngl_ladder"]
    assert f(99, 36) == [99, 18, 0], f(99, 36)
    assert f(0, 36) == [0], f(0, 36)
    assert f(24, 36) == [24, 12, 0], f(24, 36)
    assert f(1, 36) == [1, 0], f(1, 36)
    assert f(99, 1) == [99, 1, 0], f(99, 1)   # 1 层模型："一半"仍是 1 层
    assert f(99, 2) == [99, 1, 0], f(99, 2)
    # 阶梯必须严格递减且以 0 结尾（纯 CPU 兜底）
    for fitted in (0, 1, 5, 18, 36, 99):
        lad = f(fitted, 36)
        assert lad == sorted(set(lad), reverse=True), lad
        assert lad[-1] == 0, lad


def test_s3_ngl_fallback_on_oom(ns):
    """第一次 OOM → 自动降 -ngl 重试，并把两次尝试都记进 provenance。"""
    calls = []

    def fake_run_cmd(cmd, log_path, deadline, cwd=None, env=None):
        calls.append(list(cmd))
        ngl = cmd[cmd.index("-ngl") + 1]
        if str(ngl) == "99":                 # 第一次：显存不够
            Path(log_path).write_text(
                "llama_model_load: ...\n"
                "ggml_cuda_reserve: failed to allocate GPU memory\n"
                "error: out of memory\n", encoding="utf-8")
            raise ns["CommandFailed"](cmd, 2, "out of memory", log_path)
        Path(log_path).write_text("done\n", encoding="utf-8")
        return 0, "done"

    orig = ns["run_cmd"]
    ns["run_cmd"] = fake_run_cmd
    tmp = Path(ns["MDLS"]) / "_ngl_test_out.gguf"
    try:
        ngl, attempts = ns["_run_imatrix_with_fallback"](
            "m.gguf", "c.txt", tmp, time.time() + 120, 99)
        assert ngl == 18, ngl                 # 36 层的一半
        assert len(calls) == 2, calls
        assert [a["ok"] for a in attempts] == [False, True]
        assert attempts[0]["ngl"] == 99 and attempts[0]["oom"] is True
        assert attempts[1]["ngl"] == 18
        assert attempts[0]["tail"], "失败尝试必须留日志尾部"
    finally:
        ns["run_cmd"] = orig


def test_s3_imatrix_failure_explains(ns):
    """imatrix 失败要给人话，而不是把日志丢给用户。"""
    f = ns["_explain_imatrix_failure"]

    class E:
        rc = 1
        tail = "llama_model_load: failed to open /x/y.gguf"

    assert "打不开" in f(E())

    class E2:
        rc = 1
        tail = ("error: you need at least 16384 tokens for a context of 512 tokens\n"
                "the data file you provided tokenizes to only 100 tokens")

    assert "token 数不够" in f(E2())

    class E3:
        rc = 1
        tail = "ggml_gallocr_reserve_n: failed to allocate 8192 MiB"

    assert "显存不足" in f(E3())


def _stub_s3(ns, tmp, *, imatrix_ok=True, q8_bytes=12345678):
    """把 stage_s3_imatrix 的外部依赖全部换成桩，只留真逻辑。

    返回 (patches, restore)。imatrix_ok=False 时模拟 llama-imatrix 失败。
    """
    patches = []

    def patch(name, value):
        patches.append((name, ns[name]))
        ns[name] = value

    q8 = Path(tmp) / "dl" / "models" / ns["S3_PINNED"]["filename"]
    q8.parent.mkdir(parents=True, exist_ok=True)
    q8.write_bytes(b"x" * q8_bytes)
    patch("_s3_base_path", lambda: q8)
    patch("_ensure_q8", lambda deadline: (q8, {"origin": "downloaded",
                                               "sha256": "a" * 64,
                                               "bytes": q8_bytes}))
    calib = Path(tmp) / "calib.txt"
    calib.write_text("some calibration text\n" * 200, encoding="utf-8")
    patch("_ensure_calib_txt", lambda deadline: (calib, {"origin": "downloaded",
                                                         "nonempty_rows": 200}))
    patch("_calib_txt_path", lambda: calib)
    patch("vram_gb", lambda: {"total": 15.0, "free": 14.0, "used": 1.0})
    patch("_fit_ngl", lambda n, u, m=2.0: 99 if n + m * 2**30 < u * 2**30 else 0)
    # 让 _imatrix_base_kind() 有值：S3 自动下载路径固定是 q8_0
    patch("CFG", {**ns["CFG"], "IMATRIX_BASE_KIND": "q8_0"})

    def fake_cmd(cmd, log_path, deadline, cwd=None, env=None):
        Path(log_path).write_text("ok\n", encoding="utf-8")
        if imatrix_ok:
            out = cmd[cmd.index("-o") + 1]
            # 必须 > 100_000 B，否则 stage_s3_imatrix 的产物体检会拒收
            Path(out).write_bytes(b"IMATRIX" + b"\0" * (200 * 1024))
        return 0, "ok"

    patch("run_cmd", fake_cmd)
    patch("gguf_scan", lambda p: {"n_tensors": 361})
    patch("sha256_of", lambda p, deadline=None: "b" * 64)
    patch("_path_fp", lambda p: "fp" + "c" * 29)

    def restore():
        for name, value in reversed(patches):
            ns[name] = value

    return q8, calib, restore


def test_s3_q8_deleted_only_after_record(ns):
    """Q8_0 必须在"结果记录写盘之后"才删 —— 删除那一刻记录文件必须已存在。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_reclaim_")
    orig_out = ns["OUT"]
    ns["OUT"] = Path(tmp) / "out"
    ns["OUT"].mkdir(parents=True, exist_ok=True)
    snap_cfg = dict(ns["CFG"])
    try:
        q8, calib, restore = _stub_s3(ns, tmp, imatrix_ok=True)
        # 记录"删除那一刻"报告文件是否已在盘上
        seen = {"record_at_delete": None}
        # 用包装版替换：先观察"删除那一刻记录是否已在盘上"，再真的删
        real_reclaim = ns["_reclaim_q8"]

        def spy(path, stamp, rec):
            seen["record_at_delete"] = (ns["OUT"] / "S3_IMATRIX.json").exists()
            return real_reclaim(path, stamp, rec)

        ns["_reclaim_q8"] = spy
        ns["CFG"]["S3_AUTO_FETCH_MODEL"] = True
        ns["CFG"]["S3_AUTO_FETCH_CALIB"] = True
        ns["CFG"]["IMATRIX_BASE"] = ""
        ns["CFG"]["CALIB_TXT"] = ""
        ns["CFG"]["IMATRIX_GGUF"] = ""
        try:
            rec = ns["stage_s3_imatrix"](time.time() + 120)
        finally:
            restore()
            ns["_reclaim_q8"] = real_reclaim

        assert seen["record_at_delete"] is True, \
            "删除 Q8_0 时结果记录还没写盘 —— 会把可恢复状态毁掉"
        assert not q8.exists(), "成功后 Q8_0 应被回收"
        assert rec["q8_reclaimed"] is True
        assert rec["q8_reclaimed_bytes"] == 12345678
        assert rec["ngl"] == 99 and rec["on_gpu"] is True
        assert rec["n_tensors"] == 361
        assert (ns["OUT"] / "S3_IMATRIX.json").exists()
    finally:
        ns["CFG"].clear(); ns["CFG"].update(snap_cfg)
        ns["OUT"] = orig_out
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_q8_kept_when_imatrix_fails(ns):
    """imatrix 失败 → Q8_0 必须留着，让用户修完直接重跑。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_keep_")
    orig_out = ns["OUT"]
    ns["OUT"] = Path(tmp) / "out"
    ns["OUT"].mkdir(parents=True, exist_ok=True)
    snap_cfg = dict(ns["CFG"])
    # ns 在所有用例间共享，先清掉别的用例留下的 imatrix，否则测的不是自己
    stale = ns["MDLS"] / "imatrix.gguf"
    for p in (stale, stale.with_name(stale.name + ".partial")):
        p.unlink(missing_ok=True)
    try:
        q8, calib, restore = _stub_s3(ns, tmp, imatrix_ok=False)
        ns["CFG"]["S3_AUTO_FETCH_MODEL"] = True
        ns["CFG"]["S3_AUTO_FETCH_CALIB"] = True
        ns["CFG"]["IMATRIX_BASE"] = ""
        ns["CFG"]["CALIB_TXT"] = ""
        ns["CFG"]["IMATRIX_GGUF"] = ""
        try:
            try:
                ns["stage_s3_imatrix"](time.time() + 120)
                raise AssertionError("imatrix 失败必须抛异常")
            except Exception:
                pass
        finally:
            restore()
        assert q8.exists(), "失败时 Q8_0 必须保留"
        assert q8.stat().st_size == 12345678
        # 半成品既不能当结果、也不能续跑，必须清掉，别占 working 的 20 GB
        assert not stale.exists(), "失败不该留下 imatrix.gguf"
        assert not stale.with_name(stale.name + ".partial").exists(), \
            "失败必须清掉 imatrix.gguf.partial（不可续跑）"
        assert not (ns["OUT"] / "S3_IMATRIX.json").exists(), \
            "失败不该写成功记录"
    finally:
        ns["CFG"].clear(); ns["CFG"].update(snap_cfg)
        ns["OUT"] = orig_out
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_q8_not_deleted_when_configured_off(ns):
    """CFG['S3_DELETE_Q8_AFTER']=False → 保留 Q8_0，并记录原因。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_keepcfg_")
    orig_out = ns["OUT"]
    ns["OUT"] = Path(tmp) / "out"
    ns["OUT"].mkdir(parents=True, exist_ok=True)
    snap_cfg = dict(ns["CFG"])
    try:
        q8, calib, restore = _stub_s3(ns, tmp, imatrix_ok=True)
        ns["CFG"]["S3_DELETE_Q8_AFTER"] = False
        ns["CFG"]["S3_AUTO_FETCH_MODEL"] = True
        ns["CFG"]["S3_AUTO_FETCH_CALIB"] = True
        ns["CFG"]["IMATRIX_BASE"] = ""
        ns["CFG"]["CALIB_TXT"] = ""
        ns["CFG"]["IMATRIX_GGUF"] = ""
        try:
            rec = ns["stage_s3_imatrix"](time.time() + 120)
        finally:
            restore()
        assert q8.exists(), "关掉回收开关时必须保留"
        assert rec["q8_reclaimed"] is False
        assert "S3_DELETE_Q8_AFTER" in rec["q8_reclaim_reason"]
    finally:
        ns["CFG"].clear(); ns["CFG"].update(snap_cfg)
        ns["OUT"] = orig_out
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_manual_base_not_touched(ns):
    """用户手动指定 IMATRIX_BASE 时，S3 不许下载也不许删那个文件。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_s3_manual_")
    orig_out = ns["OUT"]
    ns["OUT"] = Path(tmp) / "out"
    ns["OUT"].mkdir(parents=True, exist_ok=True)
    snap_cfg = dict(ns["CFG"])
    try:
        q8, calib, restore = _stub_s3(ns, tmp, imatrix_ok=True)
        ns["CFG"]["S3_AUTO_FETCH_MODEL"] = False
        ns["CFG"]["S3_AUTO_FETCH_CALIB"] = False
        ns["CFG"]["IMATRIX_BASE"] = str(q8)          # 手动指过去
        ns["CFG"]["IMATRIX_BASE_KIND"] = "q8_0"
        ns["CFG"]["CALIB_TXT"] = str(calib)
        ns["CFG"]["IMATRIX_GGUF"] = ""
        called = {"q8": False, "calib": False}
        real_q8, real_calib = ns["_ensure_q8"], ns["_ensure_calib_txt"]

        def spy_q8(d):
            called["q8"] = True
            return real_q8(d)

        def spy_calib(d):
            called["calib"] = True
            return real_calib(d)

        ns["_ensure_q8"] = spy_q8
        ns["_ensure_calib_txt"] = spy_calib
        try:
            rec = ns["stage_s3_imatrix"](time.time() + 120)
        finally:
            restore()
        assert called["q8"] is False and called["calib"] is False
        assert rec["base_origin"] == "manual"
        assert q8.exists(), "手动指定的 base 一律不许删"
        # v0.5：即使不删，也要把"为什么不删"写进记录（原来这个键直接不存在）
        assert rec["q8_reclaimed"] is False
        assert "manual" in rec["q8_reclaim_reason"]
    finally:
        ns["CFG"].clear(); ns["CFG"].update(snap_cfg)
        ns["OUT"] = orig_out
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_report_lines(ns):
    """S9 的 imatrix 小节要覆盖 provenance 的关键字段。"""
    r = {
        "imatrix": "/x/imatrix.gguf", "bytes": 300 * 1024, "n_tensors": 361,
        "sha256": "a" * 64, "base_gguf": "/dl/models/Qwen3-8B-Q8_0.gguf",
        "base_kind": "q8_0", "base_bytes": 8709519168, "base_origin": "auto_download",
        "base_download": {"sha256": "a" * 64, "origin": "downloaded"},
        "calib_txt": "/w/calib.txt", "calib_bytes": 11 * 1024 * 1024,
        "calib_origin": "auto_download", "calib_source": "wikitext-2-raw/train",
        "ctx": 512, "chunks": 32, "need_tokens": 16384,
        "ngl": 18, "ngl_initial": 99, "on_gpu": True,
        "vram_total_gb": 15.0, "vram_margin_gb": 2.0,
        "ngl_attempts": [{"ngl": 99, "ok": False, "oom": True},
                         {"ngl": 18, "ok": True}],
        "q8_reclaimed": True, "q8_reclaimed_bytes": 8709519168,
        "q8_reclaim_reason": "imatrix 已验证且结果记录已写盘",
        "imatrix_base_from_approx": True,
    }
    L = ns["_s3_report_lines"](r)
    md = "\n".join(L)
    for must in ("Qwen3-8B-Q8_0.gguf", "wikitext-2-raw/train", "16384 token",
                 "初判 99 → 实际 **18**", "已回收", "361 个张量",
                 "是对 f16 的近似", "尝试记录", '"oom": true'):
        assert must in md, f"报告缺少 {must!r}"


# ---------------------------------------------------------------- S4 一键量化（v0.5）
# 覆盖需求 1-7：开关作用域、imatrix pin 三态、provenance 级同源核验、
# base 四级取源、暂存盘探测、遗留 Q8_0 回收、S3 重跑来源判定、禁止二次量化。


class _Patch:
    """极小的 patch 器：记录原值，undo() 一律还原（测试失败也不留污染）。"""

    def __init__(self, ns):
        self.ns = ns
        self.saved = []

    def __call__(self, name, value):
        self.saved.append((name, self.ns[name]))
        self.ns[name] = value
        return value

    def undo(self):
        for name, value in reversed(self.saved):
            self.ns[name] = value
        self.saved = []


class _FakeUsage:
    def __init__(self, free, total=None):
        self.free = free
        self.total = total if total is not None else free * 2
        # disk_gb() 会读 used（core.py:85），缺了它 run_stage→guard 直接 AttributeError
        self.used = self.total - self.free


class _FakeShutil:
    """只实现 _resolve_s4_base / disk_gb 用到的那一点点。"""

    def __init__(self, free, total=None):
        self._free = free
        self._total = total

    def disk_usage(self, p):
        return _FakeUsage(self._free, self._total)

    def which(self, name):
        return None


class _FakeOS:
    """包住真 os，只按路径覆盖 stat().st_dev，用来造"另一个文件系统"。"""

    def __init__(self, real, devmap):
        self._real = real
        self._devmap = {str(k): v for k, v in devmap.items()}

    def stat(self, p):
        s = str(p)
        if s in self._devmap:
            import types as _t
            return _t.SimpleNamespace(st_dev=self._devmap[s])
        return self._real.stat(p)

    def __getattr__(self, k):
        return getattr(self._real, k)


def _scan_dispatch(mapping):
    """假 gguf_scan：按路径返回预定 scan 字典。"""

    def fake(path, deadline=None, **kw):
        key = str(path)
        if key not in mapping:
            raise KeyError("假 gguf_scan 没有 %s 的记录" % key)
        return mapping[key]

    return fake


def _fake_base_scan(layers=(0, 1, 2), arch="qwen3", n_params=1000):
    """bf16 base 的 scan：ffn_down 的输入通道是 12288，其余 4096
    （与真实 Qwen3-8B 的 imatrix 实测一致）。"""
    tensors = []
    for i in layers:
        tensors.append({"name": "blk.%d.attn_q.weight" % i, "dims": [4096, 4096]})
        tensors.append({"name": "blk.%d.ffn_down.weight" % i, "dims": [12288, 4096]})
    return {"architecture": arch, "n_params": n_params, "tensors": tensors}


def _fake_imatrix_scan(layers=(0, 1, 2), extra=()):
    """imatrix 的 scan：每个模型张量 4 个统计量条目，只有 .in_sum 记 n_in。"""
    tensors = []
    for i in layers:
        for suf in (".in_sum", ".in_sum2", ".counts", ".stats"):
            tensors.append({"name": "blk.%d.attn_q.weight%s" % (i, suf),
                            "n_elements": 4096})
            tensors.append({"name": "blk.%d.ffn_down.weight%s" % (i, suf),
                            "n_elements": 12288})
    for name, n_in in extra:
        tensors.append({"name": name, "n_elements": n_in})
    return {"n_tensors": len(tensors), "tensors": tensors}


# ---- 需求 6：默认不下载；RUN_S4 的作用域 ----

def test_s4_run_switch_scope(ns):
    """RUN_S4=True 只开 S2/S3/S4/S5；S6-S8 与 SX 必须全部保持 False。"""
    STAGES, CFG = ns["STAGES"], ns["CFG"]
    snap_s, snap_c = dict(STAGES), dict(CFG)
    orig = ns["CFG"]["RUN_S4"]
    try:
        ns["CFG"]["RUN_S4"] = True
        out = ns["_apply_run_switches"]()
        assert out["run_s4"] is True
        assert STAGES["S2_BUILD"] is True, "RUN_S4 必须隐含打开 S2（量化要二进制）"
        assert STAGES["S3_IMATRIX"] is True, "RUN_S4 必须隐含打开 S3（imatrix 是 IQ2_XS 的前提）"
        assert STAGES["S4_QUANT"] is True
        assert STAGES["S5_BYTES"] is True
        for k in ("S6_NLL", "S7_QUALITY", "S8_BF16_ANCHOR", "SX_EXPLORE"):
            assert STAGES[k] is False, "%s 被 RUN_S4 顺手打开了" % k
        assert CFG["BASE_KIND"] == ns["S4_PINNED"]["kind"] == "bf16"
        assert CFG["S4_AUTO_FETCH_BASE"] is True
        assert CFG["S4_RECLAIM_Q8_CACHE"] is True
        assert CFG["S3_SKIP_IF_IMATRIX_OK"] is True
        assert CFG["S3_DELETE_Q8_AFTER"] is True
        assert CFG["ALLOW_REQUANTIZE"] is False, "本路线禁止二次量化，逃生门必须关着"
        assert CFG["BASE_GGUF"] == "", "RUN_S4 不许预设 BASE_GGUF（base 由 S4 自己取）"
        assert out["quant_base"] == ns["S4_PINNED"]["filename"]
        assert out["formats"] == ns["FORMATS"] == ["IQ2_XS", "IQ4_XS"]
    finally:
        ns["CFG"]["RUN_S4"] = orig
        ns["_apply_run_switches"]()
        STAGES.clear(); STAGES.update(snap_s)
        CFG.clear(); CFG.update(snap_c)


def test_s4_pins_are_registered(ns):
    """BF16 与 imatrix 的预登记值必须与 HF / 真实运行核对过的一致。"""
    s4 = ns["S4_PINNED"]
    assert s4["sha256"] == ("5e416a2020fe63e76ea13c8979be35fc"
                            "6070aaf3578f7876400c55c2f5c3eb30"), s4["sha256"]
    assert s4["bytes"] == 16388044384, s4["bytes"]
    assert s4["kind"] == "bf16" and s4["filename"] == "Qwen3-8B-BF16.gguf"
    assert s4["url"].endswith("/resolve/%s/%s" % (s4["revision"], s4["filename"]))
    # 承重墙：imatrix 的 base 与量化 base 必须同仓库同 revision，
    # provenance 级同源核验的第 ② 项全靠这一条
    assert s4["repo"] == ns["S3_PINNED"]["repo"], "量化 base 与 imatrix base 不是同一个仓库"
    assert s4["revision"] == ns["S3_PINNED"]["revision"], "不是同一次转换"

    im = ns["IMATRIX_PINNED"]
    assert im["sha256"] == ("34062ab6d5dfaaef56ee659461c19f4e"
                            "94254ec14481b07d5531459fbb16ac37"), im["sha256"]
    assert im["bytes"] == 10701952 and im["n_tensors"] == 1008, im


def test_s4_default_off_downloads_nothing(ns):
    """RUN_S4=False 时 S4_AUTO_FETCH_BASE 必须保持 False（不许默认下 15 GiB）。"""
    CFG = ns["CFG"]
    snap = dict(CFG)
    try:
        ns["_apply_run_switches"]()
        assert CFG["S4_AUTO_FETCH_BASE"] is False
        assert CFG["S4_RECLAIM_Q8_CACHE"] is False
        assert CFG["BASE_KIND"] == snap["BASE_KIND"]
    finally:
        CFG.clear(); CFG.update(snap)


# ---- 需求 3：imatrix 的 pin 三态与发现路径 ----

def test_verify_imatrix_three_states(ns):
    """imatrix 可信度三态：pinned / provenance / unverified，外加坏文件。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_verify_")
    p = Path(tmp) / "imatrix.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 4096)
    CFG = ns["CFG"]
    snap_h = CFG["HASH_GGUF"]
    patch = _Patch(ns)
    try:
        patch("gguf_scan", _scan_dispatch({str(p): _fake_imatrix_scan()}))
        CFG["HASH_GGUF"] = True

        # ① 与 pin 一致 → pinned_verified（最强：真实 Kaggle 运行产物）
        patch("sha256_of", lambda x, deadline=None: ns["IMATRIX_PINNED"]["sha256"])
        patch("_imatrix_provenance_records", lambda: [])
        v = ns["_verify_imatrix"](p)
        assert v["ok"] is True and v["trust"] == "pinned_verified", v
        assert "IMATRIX_PINNED" in v["why"]

        # ② 与某条 provenance 记录一致 → provenance_verified
        patch("sha256_of", lambda x, deadline=None: "c" * 64)
        patch("_imatrix_provenance_records",
              lambda: [{"sha256": "c" * 64, "_source": "/x/imatrix.provenance.json"}])
        v = ns["_verify_imatrix"](p)
        assert v["trust"] == "provenance_verified", v
        assert v["provenance"]["_source"] == "/x/imatrix.provenance.json"

        # ③ 谁都对不上 → unverified（默认不放行）
        patch("sha256_of", lambda x, deadline=None: "e" * 64)
        patch("_imatrix_provenance_records", lambda: [])
        v = ns["_verify_imatrix"](p)
        assert v["ok"] is True and v["trust"] == "unverified", v

        # ④ 关掉 HASH_GGUF → 没有 sha256 可对，只能是 unverified
        CFG["HASH_GGUF"] = False
        v = ns["_verify_imatrix"](p)
        assert v["sha256"] is None and v["trust"] == "unverified", v
        CFG["HASH_GGUF"] = True

        # ⑤ 字节数 / 张量数与 pin 不符 → 进 warnings，但不改变 trust
        patch("sha256_of", lambda x, deadline=None: ns["IMATRIX_PINNED"]["sha256"])
        v = ns["_verify_imatrix"](p)
        assert v["trust"] == "pinned_verified"
        assert any("字节数" in w for w in v["warnings"]), v["warnings"]

        # ⑥ 文件不存在 / 解析失败 / 空文件
        v = ns["_verify_imatrix"](Path(tmp) / "nope.gguf")
        assert v["ok"] is False and v["trust"] == "missing", v

        def boom(path, deadline=None, **kw):
            raise RuntimeError("bad magic")

        patch("gguf_scan", boom)
        v = ns["_verify_imatrix"](p)
        assert v["ok"] is False and v["trust"] == "unreadable", v
        assert "GGUF 解析失败" in v["why"]

        patch("gguf_scan", _scan_dispatch({str(p): {"n_tensors": 0, "tensors": []}}))
        v = ns["_verify_imatrix"](p)
        assert v["ok"] is False and v["trust"] == "empty", v
    finally:
        patch.undo()
        CFG["HASH_GGUF"] = snap_h


def test_imatrix_found_in_input_and_candidates(ns):
    """/kaggle/input 里挂着的 imatrix 必须被发现，且候选顺序符合优先级。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_cand_")
    in_dir = Path(tmp) / "input" / "some-slug"
    in_dir.mkdir(parents=True)
    (in_dir / "imatrix.gguf").write_bytes(b"GGUF" + b"\0" * 1024)
    (in_dir / "other.gguf").write_bytes(b"GGUF" + b"\0" * 1024)
    mdls = Path(tmp) / "models"
    mdls.mkdir()
    (mdls / "imatrix.gguf").write_bytes(b"GGUF" + b"\0" * 2048)

    patch = _Patch(ns)
    snap = {k: ns["CFG"][k] for k in ("IMATRIX_GGUF",)}
    try:
        patch("IN", Path(tmp) / "input")
        patch("MDLS", mdls)
        ns["CFG"]["IMATRIX_GGUF"] = ""

        hits = ns["_scan_input_gguf"](max_hits=32, time_budget_sec=10.0)
        assert len(hits) == 2 and all(h.suffix == ".gguf" for h in hits), hits

        cands = [str(c) for c in ns["_imatrix_candidates"]()]
        assert cands[0] == str(mdls / "imatrix.gguf"), cands
        assert str(in_dir / "imatrix.gguf") in cands, "Input 里的 imatrix 必须进候选"

        # resolve_imatrix 取第一个存在的
        assert ns["resolve_imatrix"]() == mdls / "imatrix.gguf"
        (mdls / "imatrix.gguf").unlink()
        assert ns["resolve_imatrix"]() == in_dir / "imatrix.gguf"
        (in_dir / "imatrix.gguf").unlink()
        assert ns["resolve_imatrix"]() is None

        # 显式指定的 CFG['IMATRIX_GGUF'] 排最前
        ns["CFG"]["IMATRIX_GGUF"] = str(in_dir / "other.gguf")
        assert ns["resolve_imatrix"]() == in_dir / "other.gguf"
    finally:
        patch.undo()
        ns["CFG"].update(snap)


def test_imatrix_provenance_records_sidecar_first(ns):
    """provenance 记录按可信度排序：挨着 imatrix 的 sidecar 优先于 OUT/。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_prov_")
    mdls, out = Path(tmp) / "models", Path(tmp) / "out"
    mdls.mkdir(); out.mkdir()
    patch = _Patch(ns)
    try:
        patch("MDLS", mdls)
        patch("OUT", out)
        assert ns["_imatrix_provenance_records"]() == []

        ns["write_json"](out / "S3_IMATRIX.json",
                         {"status": "ok", "result": {"imatrix": "/m/imatrix.gguf",
                                                     "sha256": "b" * 64}})
        recs = ns["_imatrix_provenance_records"]()
        assert len(recs) == 1 and recs[0]["_source"] == str(out / "S3_IMATRIX.json")

        ns["write_json"](mdls / "imatrix.provenance.json",
                         {"schema": 2, "imatrix": "/m/imatrix.gguf",
                          "sha256": "c" * 64, "base_repo": "unsloth/Qwen3-8B-GGUF"})
        recs = ns["_imatrix_provenance_records"]()
        assert len(recs) == 2
        assert recs[0]["_source"] == str(mdls / "imatrix.provenance.json")
        assert recs[0]["sha256"] == "c" * 64, "sidecar 必须排第一（挨着产物，最可能还在）"
        assert recs[1]["_source"] == str(out / "S3_IMATRIX.json")
    finally:
        patch.undo()


# ---- 需求 4：Q8_0 已回收时的同源性核验 ----

def _consistency_env(ns, tmp, *, base_layers=(0, 1, 2), imat_layers=(0, 1, 2),
                     imat_extra=(), q8_present=False, sha="c" * 64):
    """搭一套 _imatrix_base_consistency 的桩环境，返回 (patch, base, imat, rec, undo)。"""
    base = Path(tmp) / "Qwen3-8B-BF16.gguf"
    imat = Path(tmp) / "imatrix.gguf"
    q8 = Path(tmp) / "Qwen3-8B-Q8_0.gguf"
    base.write_bytes(b"x" * 2048)
    imat.write_bytes(b"x" * 2048)
    if q8_present:
        q8.write_bytes(b"x" * 2048)
    mapping = {str(base): _fake_base_scan(layers=base_layers),
               str(imat): _fake_imatrix_scan(layers=imat_layers, extra=imat_extra)}
    if q8_present:
        mapping[str(q8)] = _fake_base_scan(layers=base_layers)
    patch = _Patch(ns)
    patch("gguf_scan", _scan_dispatch(mapping))
    patch("sha256_of", lambda p, deadline=None: sha)
    rec = {"sha256": sha, "base_gguf": str(q8) if q8_present else "/gone/Q8_0.gguf",
           "base_kind": "q8_0", "base_repo": ns["S3_PINNED"]["repo"],
           "base_revision": ns["S3_PINNED"]["revision"],
           "base_bytes": ns["S3_PINNED"]["bytes"]}
    return patch, base, imat, rec


def test_consistency_provenance_level_ok(ns):
    """Q8_0 已回收：结构 + sha256 + 来源三项都过 → provenance 级通过。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_cons_")
    patch, base, imat, rec = _consistency_env(ns, tmp)
    try:
        r = ns["_imatrix_base_consistency"](imat, base, rec, None)
        assert r["ok"] is True, r
        assert r["level"] == "provenance", r
        assert r["n_tensors"] == 6 and r["layers"] == 3, r
        assert r["n_in_orientation"] == "dims[0]", r
        assert "sha256" in r["verified_by"] and "来源" in r["verified_by"]
        assert "已被回收" in r["note"], r["note"]
    finally:
        patch.undo()


def test_consistency_structure_failures(ns):
    """结构三项（张量名 / 层数 / 输入通道）任一不过都必须拒。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_consf_")
    f = ns["_imatrix_base_consistency"]

    # ① imatrix 里有 base 中不存在的张量
    patch, base, imat, rec = _consistency_env(
        ns, tmp, imat_extra=[("blk.9.attn_q.weight.in_sum", 4096)])
    try:
        r = f(imat, base, rec, None)
        assert r["ok"] is False and r["level"] == "structure", r
        assert "不存在" in r["why"]
    finally:
        patch.undo()

    # ② 层数不同（base 3 层 vs imatrix 2 层，名字仍全部存在）
    patch, base, imat, rec = _consistency_env(ns, tmp, imat_layers=(0, 1))
    try:
        r = f(imat, base, rec, None)
        assert r["ok"] is False and r["level"] == "structure", r
        assert "层数不同" in r["why"]
    finally:
        patch.undo()

    # ③ 输入通道数两个方向都对不上
    patch, base, imat, rec = _consistency_env(
        ns, tmp, imat_extra=[("blk.0.attn_q.weight.in_sum", 999)])
    try:
        r = f(imat, base, rec, None)
        assert r["ok"] is False and r["level"] == "structure", r
        assert "输入通道数对不上" in r["why"] and "dims[0]" in r["why"]
    finally:
        patch.undo()

    # ④ 方向反过来也认（dims[1] = n_in），不算失败
    patch = _Patch(ns)
    base2 = Path(tmp) / "rev.gguf"
    imat2 = Path(tmp) / "rev_im.gguf"
    base2.write_bytes(b"x" * 16); imat2.write_bytes(b"x" * 16)
    patch("gguf_scan", _scan_dispatch({
        str(base2): {"architecture": "qwen3", "n_params": 9, "tensors": [
            {"name": "blk.0.attn_q.weight", "dims": [4096, 4096]}]},
        str(imat2): {"n_tensors": 4, "tensors": [
            {"name": "blk.0.attn_q.weight.in_sum", "n_elements": 4096},
            {"name": "blk.0.attn_q.weight.in_sum2", "n_elements": 4096},
            {"name": "blk.0.attn_q.weight.counts", "n_elements": 4096},
            {"name": "blk.0.attn_q.weight.stats", "n_elements": 4096}]}}))
    patch("sha256_of", lambda p, deadline=None: "c" * 64)
    try:
        r = f(imat2, base2, {"sha256": "c" * 64,
                             "base_repo": ns["S3_PINNED"]["repo"],
                             "base_revision": ns["S3_PINNED"]["revision"],
                             "base_bytes": ns["S3_PINNED"]["bytes"]}, None)
        assert r["ok"] is True, r
        assert r["n_in_orientation"] in ("dims[0]", "dims[1]")
    finally:
        patch.undo()


def test_consistency_provenance_failures(ns):
    """provenance 级的四个失败分支：无记录 / sha 不符 / 来源不符 / 字节数不符。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_consp_")
    f = ns["_imatrix_base_consistency"]

    # ① 没有任何记录
    patch, base, imat, _ = _consistency_env(ns, tmp)
    try:
        r = f(imat, base, None, None)
        assert r["ok"] is False and r["level"] == "provenance", r
        assert "没有" in r["why"]
    finally:
        patch.undo()

    # ② imatrix sha256 与记录不符（换了文件）
    patch, base, imat, rec = _consistency_env(ns, tmp, sha="d" * 64)
    try:
        r = f(imat, base, dict(rec, sha256="c" * 64), None)
        assert r["ok"] is False and "sha256 与记录不符" in r["why"], r
    finally:
        patch.undo()

    # ③ 来源不是同一个仓库 / revision
    patch, base, imat, rec = _consistency_env(ns, tmp)
    try:
        bad = dict(rec, base_repo="someone/other-GGUF")
        r = f(imat, base, bad, None)
        assert r["ok"] is False and "来源不是 S4_PINNED" in r["why"], r
        bad = dict(rec, base_revision="0" * 40)
        r = f(imat, base, bad, None)
        assert r["ok"] is False and "来源不是 S4_PINNED" in r["why"], r
        # 缺 base_repo/revision 时回落到 S3_PINNED（同仓库同 revision）→ 仍然通过
        ok = {k: v for k, v in rec.items() if k not in ("base_repo", "base_revision")}
        r = f(imat, base, ok, None)
        assert r["ok"] is True, r
    finally:
        patch.undo()

    # ④ 记录的 base 字节数与 pin 不符
    patch, base, imat, rec = _consistency_env(ns, tmp)
    try:
        r = f(imat, base, dict(rec, base_bytes=123), None)
        assert r["ok"] is False and "字节数" in r["why"], r
    finally:
        patch.undo()


def test_consistency_file_level_when_q8_present(ns):
    """Q8_0 还在时走文件级：逐张量比形状，强于 provenance 级。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_consfile_")
    patch, base, imat, rec = _consistency_env(ns, tmp, q8_present=True)
    try:
        r = ns["_imatrix_base_consistency"](imat, base, rec, None)
        assert r["ok"] is True and r["level"] == "file", r
        assert r["n_tensors"] == 6
        assert r["structure"]["imatrix_has_architecture"] is False, \
            "imatrix GGUF 实测没有 general.architecture，别假设它有"
    finally:
        patch.undo()


def test_imatrix_base_does_not_throw_when_reclaimed(ns):
    """需求 4：Q8_0 被回收后 _imatrix_base() 不许抛异常（v0.4 就是在这里炸的）。"""
    CFG = ns["CFG"]
    snap = {k: CFG[k] for k in ("IMATRIX_BASE", "IMATRIX_BASE_KIND",
                                "BASE_GGUF", "S3_AUTO_FETCH_MODEL",
                                "S3_AUTO_FETCH_CALIB")}
    patch = _Patch(ns)
    try:
        patch("_s3_base_path", lambda: Path("/gone/Qwen3-8B-Q8_0.gguf"))
        CFG.update(IMATRIX_BASE="/gone/Qwen3-8B-Q8_0.gguf", BASE_GGUF="")
        got = ns["_imatrix_base"]()
        assert got == Path("/gone/Qwen3-8B-Q8_0.gguf")
        assert not got.exists(), "桩路径本来就不存在 —— 调用方自己判 .exists()"
        CFG.update(IMATRIX_BASE="", S3_AUTO_FETCH_MODEL=True)
        assert ns["_imatrix_base"]() == Path("/gone/Qwen3-8B-Q8_0.gguf")
        CFG.update(S3_AUTO_FETCH_MODEL=False, S3_AUTO_FETCH_CALIB=False,
                   BASE_GGUF="/also/gone.gguf")
        assert ns["_imatrix_base"]() == Path("/also/gone.gguf")
    finally:
        patch.undo()
        CFG.update(snap)


# ---- 需求 2：量化 base 四级取源 ----

def _s4_base_env(ns, tmp):
    """_resolve_s4_base 的公共桩：Input 扫描、暂存盘、下载、磁盘预检。"""
    state = {"fetch": [], "precheck": [], "input": [], "scratch": []}

    def fake_fetch(pin, dst, deadline, tag=""):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"BF16" * 16)
        meta = {"origin": "downloaded", "bytes": pin["bytes"],
                "sha256": pin["sha256"], "path": str(dst)}
        state["fetch"].append({"pin": dict(pin), "dst": str(dst), "tag": tag})
        return meta

    def fake_precheck(extra_bytes, tag, reserve_gb=None, root=None):
        state["precheck"].append({"extra": extra_bytes, "tag": tag,
                                  "reserve_gb": reserve_gb, "root": root})

    patch = _Patch(ns)
    patch("_fetch_pinned_blob", fake_fetch)
    patch("_disk_precheck", fake_precheck)
    patch("_scan_input_gguf",
          lambda max_hits=8, time_budget_sec=20.0: list(state["input"]))
    patch("_scratch_dirs", lambda need, margin_gb=2.0: list(state["scratch"]))
    return patch, state


def test_resolve_s4_base_manual(ns):
    """第一优先：CFG['BASE_GGUF'] 手动指定，存在就用，不碰网络。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_base_")
    base = Path(tmp) / "hand-picked.gguf"
    base.write_bytes(b"x" * 4096)
    patch, state = _s4_base_env(ns, tmp)
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF", "S4_AUTO_FETCH_BASE")}
    try:
        ns["CFG"].update(BASE_GGUF=str(base), S4_AUTO_FETCH_BASE=True)
        p, origin, meta = ns["_resolve_s4_base"](None)
        assert origin == "manual" and p == base, (origin, p)
        assert meta == {"bytes": 4096}
        assert state["fetch"] == [] and state["precheck"] == []
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_resolve_s4_base_attached_input(ns):
    """第二优先：/kaggle/input 已挂载 → 零下载；字节数比对 + 可选 sha256 强校验。"""
    import tempfile as _tf
    import types as _t

    class _FakeHit:
        """假Input 文件：只有名字和字节数是真的（不能真写 15 GiB）。"""

        def __init__(self, name, size):
            self.name = name
            self._size = size

        def stat(self):
            return _t.SimpleNamespace(st_size=self._size)

        def __str__(self):
            return "/kaggle/input/fake-slug/" + self.name

    tmp = _tf.mkdtemp(prefix="trlab_v5_input_")
    pin = ns["S4_PINNED"]
    hit = _FakeHit(pin["filename"], pin["bytes"])
    patch, state = _s4_base_env(ns, tmp)
    state["input"] = [hit]
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF", "S4_AUTO_FETCH_BASE", "HASH_GGUF")}
    try:
        ns["CFG"].update(BASE_GGUF="", S4_AUTO_FETCH_BASE=True, HASH_GGUF=False)
        p, origin, meta = ns["_resolve_s4_base"](None)
        assert origin == "attached_input" and p == hit, (origin, p)
        assert meta["sha256"] is None and meta["sha256_checked"] is False
        assert state["fetch"] == [], "Input 里有了就不该再下载"

        # 字节数不对的候选要被跳过；没有别的来源时就该失败而不是将就
        other = _FakeHit("Qwen3-8B-Q8_0.gguf", pin["bytes"] - 1)
        state["input"] = [other]
        patch("shutil", _FakeShutil(free=2 * 2**30))
        try:
            ns["_resolve_s4_base"](None)
            raise AssertionError("字节数不符的候选不该被采用")
        except ns["GuardError"]:
            pass
        patch("shutil", _FakeShutil(free=500 * 2**30))

        # 开 HASH_GGUF：sha256 不符必须拒绝（不是 pinned revision 的那个文件）
        state["input"] = [hit]
        ns["CFG"]["HASH_GGUF"] = True
        patch("sha256_of", lambda x, deadline=None: "f" * 64)
        try:
            ns["_resolve_s4_base"](None)
            raise AssertionError("sha256 不符时必须拒绝")
        except ns["GuardError"] as e:
            assert "sha256 不符" in str(e), e
        patch("sha256_of", lambda x, deadline=None: pin["sha256"])
        p, origin, meta = ns["_resolve_s4_base"](None)
        assert origin == "attached_input" and meta["sha256_checked"] is True
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_resolve_s4_base_refuses_without_auto_fetch(ns):
    """没挂 Input、又没开自动下载 → 明确报错并指向 RUN_S4，不瞎试。"""
    patch, state = _s4_base_env(ns, Path("/tmp"))
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF", "S4_AUTO_FETCH_BASE")}
    try:
        ns["CFG"].update(BASE_GGUF="", S4_AUTO_FETCH_BASE=False)
        try:
            ns["_resolve_s4_base"](None)
            raise AssertionError("没有拦住")
        except ns["GuardError"] as e:
            assert "S4_AUTO_FETCH_BASE" in str(e) and "RUN_S4" in str(e), e
        assert state["fetch"] == []
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_resolve_s4_base_scratch_download(ns):
    """第三优先：working 之外的暂存盘 —— 下载不占 20 GB 配额，预检按暂存盘算。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_scratch_")
    scratch = Path(tmp) / "kaggle-tmp"
    scratch.mkdir()
    patch, state = _s4_base_env(ns, tmp)
    state["scratch"] = [{"path": str(scratch), "free_gb": 500.0,
                         "total_gb": 500.0, "st_dev": 424242}]
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF", "S4_AUTO_FETCH_BASE",
                                      "S4_RAM_MARGIN_GB")}
    try:
        ns["CFG"].update(BASE_GGUF="", S4_AUTO_FETCH_BASE=True,
                         S4_RAM_MARGIN_GB=4.0)
        p, origin, meta = ns["_resolve_s4_base"](None)
        assert origin == "scratch_download", origin
        assert str(p).startswith(str(scratch)), p
        assert p.name == ns["S4_PINNED"]["filename"]
        assert meta["sha256"] == ns["S4_PINNED"]["sha256"]
        assert len(state["precheck"]) == 1
        assert str(state["precheck"][0]["root"]) == str(scratch), \
            "暂存盘的磁盘预检必须按暂存盘算，不能按 working"
        assert state["precheck"][0]["reserve_gb"] == 4.0
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_resolve_s4_base_working_and_guard(ns):
    """第四优先：working 装得下才下；装不下就给一次性操作说明，不瞎试。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_work_")
    patch, state = _s4_base_env(ns, tmp)
    patch("MODEL_DL", Path(tmp) / "dl" / "models")
    patch("shutil", _FakeShutil(free=500 * 2**30))
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF", "S4_AUTO_FETCH_BASE",
                                      "MIN_FREE_DISK_GB")}
    try:
        ns["CFG"].update(BASE_GGUF="", S4_AUTO_FETCH_BASE=True,
                         MIN_FREE_DISK_GB=5.0)
        p, origin, meta = ns["_resolve_s4_base"](None)
        assert origin == "working_download", origin
        assert str(p).startswith(str(Path(tmp) / "dl")), p
        assert len(state["precheck"]) == 1 and state["precheck"][0]["root"] is None

        # 磁盘账不过 → GuardError，且消息里带最少点击方案
        patch("shutil", _FakeShutil(free=2 * 2**30))
        try:
            ns["_resolve_s4_base"](None)
            raise AssertionError("磁盘不够时必须拦住")
        except ns["GuardError"] as e:
            msg = str(e)
            assert "trlab_fetch_bf16.ipynb" in msg, "必须给出一次性操作说明"
            assert "Add Input" in msg and "Save Version" in msg, msg
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_scratch_dirs_same_device_excluded(ns):
    """暂存盘判定：只认 st_dev 与 working 不同的候选；同设备不算额外空间。"""
    import os as _os
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_dev_")
    same = Path(tmp) / "same-dev"
    other = Path(tmp) / "other-dev"
    same.mkdir(); other.mkdir()
    wdev = _os.stat(ns["WORK"]).st_dev
    GiB = 2 ** 30

    patch = _Patch(ns)
    patch("os", _FakeOS(_os, {str(same): wdev, str(other): wdev + 1}))
    try:
        f = ns["_scratch_dirs"]
        # ① 只有同设备候选 → 一律不算，哪怕空闲 500 GiB
        patch("_scratch_candidates", lambda: [same])
        patch("shutil", _FakeShutil(free=500 * GiB))
        assert f(100 * GiB) == [], "同设备的'另一个目录'不提供额外空间"

        # ② 加上另一个文件系统 → 只认它，且报出空闲/总量
        patch("_scratch_candidates", lambda: [same, other])
        got = f(100 * GiB, margin_gb=4.0)
        assert len(got) == 1 and got[0]["path"] == str(other), got
        assert got[0]["free_gb"] == 500.0 and got[0]["st_dev"] == wdev + 1

        # ③ 另一个文件系统但不够大 → 不认（余量要算进去：12 < 10 + 4）
        patch("shutil", _FakeShutil(free=50 * GiB))
        assert f(100 * GiB, margin_gb=4.0) == []
        assert len(f(10 * GiB, margin_gb=4.0)) == 1, "50 GiB 装 10+4 该认"
        patch("shutil", _FakeShutil(free=12 * GiB))
        assert f(10 * GiB, margin_gb=4.0) == [], "余量要算进去"
    finally:
        patch.undo()


def test_s4_output_bytes_estimate(ns):
    """产物体积估算：两种格式都要估，未知格式按 6.5 GiB 兜底并告警。"""
    f = ns["_s4_output_bytes"]
    GiB = 2 ** 30
    total = f()
    # 8B 模型：IQ2_XS ≈ 2.4 GiB + IQ4_XS ≈ 4.4 GiB，再加 15% 余量
    assert 6.0 * GiB < total < 9.0 * GiB, total / GiB

    patch = _Patch(ns)
    try:
        patch("FORMATS", ["NOT_A_REAL_TYPE"])
        assert f() == int(6.5 * GiB), "未知格式必须按 6.5 GiB 兜底"
    finally:
        patch.undo()


# ---- 需求 6：遗留 Q8_0 回收 / S3 重跑来源判定 ----

def test_reclaim_stale_q8_cache(ns):
    """imatrix 已验证过 → 上一轮遗留的 Q8_0 只是占盘，必须回收。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_stale_")
    q8 = Path(tmp) / "Qwen3-8B-Q8_0.gguf"
    q8.write_bytes(b"x" * 4096)
    stamp = Path(tmp) / "q8.stamp.json"
    stamp.write_text("{}", encoding="utf-8")
    part = q8.with_name(q8.name + ".partial")
    part.write_bytes(b"x" * 16)
    imat = Path(tmp) / "imatrix.gguf"
    imat.write_bytes(b"x" * 16)

    patch = _Patch(ns)
    patch("_s3_base_path", lambda: q8)
    patch("_q8_stamp_path", lambda: stamp)
    patch("resolve_imatrix", lambda: imat)
    snap = dict(ns["CFG"])
    try:
        ns["CFG"]["S4_RECLAIM_Q8_CACHE"] = True
        # ① 已验证 → 删文件、删 stamp、删 .partial，并报告释放了多少
        patch("_verify_imatrix",
              lambda p, deadline=None: {"ok": True, "trust": "pinned_verified"})
        before = q8.stat().st_size
        r = ns["_reclaim_stale_q8_cache"](None)
        assert r is not None and r["freed_bytes"] == before, r
        assert not q8.exists() and not stamp.exists() and not part.exists()
        assert "已验证" in r["reason"]

        # ② imatrix 不可信 → 不删（它还是可恢复状态）
        q8.write_bytes(b"x" * 32)
        patch("_verify_imatrix",
              lambda p, deadline=None: {"ok": True, "trust": "unverified"})
        assert ns["_reclaim_stale_q8_cache"](None) is None
        assert q8.exists()

        # ③ 找不到 imatrix → 不删
        patch("resolve_imatrix", lambda: None)
        assert ns["_reclaim_stale_q8_cache"](None) is None
        assert q8.exists()
        patch("resolve_imatrix", lambda: imat)

        # ④ 开关关掉 → 不删
        ns["CFG"]["S4_RECLAIM_Q8_CACHE"] = False
        assert ns["_reclaim_stale_q8_cache"](None) is None
        assert q8.exists()
    finally:
        patch.undo(); ns["CFG"].clear(); ns["CFG"].update(snap)


def test_s3_base_origin_regression(ns):
    """v0.4 真 bug 回归：下载路径被写回 CFG['IMATRIX_BASE'] 后，重跑必须仍判
    auto_download（否则 8.11 GiB 的 Q8_0 永远不回收）。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_origin_")
    q8 = Path(tmp) / "Qwen3-8B-Q8_0.gguf"
    q8.write_bytes(b"x" * 32)
    manual = Path(tmp) / "my-own.gguf"
    manual.write_bytes(b"x" * 32)
    stamp = Path(tmp) / "q8.stamp.json"
    ns["write_json"](stamp, {"sha256": ns["S3_PINNED"]["sha256"],
                             "bytes": ns["S3_PINNED"]["bytes"],
                             "written_utc": "2026-10-22T00:00:00+00:00"})

    patch = _Patch(ns)
    patch("_s3_base_path", lambda: q8)
    patch("_q8_stamp_path", lambda: stamp)
    snap = {k: ns["CFG"][k] for k in ("IMATRIX_BASE", "S3_AUTO_FETCH_MODEL",
                                      "S3_AUTO_FETCH_CALIB")}
    try:
        f = ns["_s3_base_origin"]
        # bug 现场：CFG 被上一轮污染，指向自动下载的落点 + stamp 对得上
        ns["CFG"].update(IMATRIX_BASE=str(q8), S3_AUTO_FETCH_MODEL=True,
                         S3_AUTO_FETCH_CALIB=True)
        origin, meta = f(q8)
        assert origin == "auto_download", "污染后必须仍判 auto_download"
        assert meta["origin"] == "cache_or_download" and meta["bytes"]

        # 没有 stamp → 仍然是自动下载（落点本身就是证据）
        stamp.unlink()
        origin, meta = f(q8)
        assert origin == "auto_download" and meta["origin"] == "downloaded_expected"

        # 真的手动文件 → manual，且没有 meta（不许删别人的文件）
        assert f(manual) == ("manual", None)

        # 没开自动取数 → 一律 manual（保守）
        ns["CFG"].update(S3_AUTO_FETCH_MODEL=False, S3_AUTO_FETCH_CALIB=False)
        assert f(q8) == ("manual", None)
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_s3_rerun_reclaims_q8(ns):
    """端到端复现 v0.4 bug：CFG 被污染的重跑，Q8_0 也必须被回收。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_rerun_")
    orig_out, orig_mdls = ns["OUT"], ns["MDLS"]
    ns["OUT"] = Path(tmp) / "out"; ns["OUT"].mkdir(parents=True)
    ns["MDLS"] = Path(tmp) / "models"; ns["MDLS"].mkdir(parents=True)
    snap_cfg = dict(ns["CFG"])
    orig_bin = ns["llama_bin"]
    try:
        q8, calib, restore = _stub_s3(ns, tmp, imatrix_ok=True)
        ns["llama_bin"] = lambda name: Path("/fake") / name
        ns["CFG"].update(S3_AUTO_FETCH_MODEL=True, S3_AUTO_FETCH_CALIB=True,
                         IMATRIX_BASE=str(q8),      # ← v0.4 bug 现场
                         IMATRIX_BASE_KIND="q8_0",
                         CALIB_TXT="", IMATRIX_GGUF="",
                         S3_SKIP_IF_IMATRIX_OK=False,   # 强制走完整 S3
                         S3_DELETE_Q8_AFTER=True)
        try:
            rec = ns["stage_s3_imatrix"](time.time() + 120)
        finally:
            restore(); ns["llama_bin"] = orig_bin

        assert rec["base_origin"] == "auto_download", rec["base_origin"]
        assert rec["base_download"] is not None
        assert rec["q8_reclaimed"] is True and rec["q8_reclaimed_bytes"] > 0
        assert not q8.exists(), "重跑后 Q8_0 必须被回收"
        assert rec["base_file_present_now"] is False
        # provenance sidecar 是"新 Session 恢复"的关键，必须落盘
        side = ns["MDLS"] / "imatrix.provenance.json"
        assert side.exists(), "imatrix.provenance.json 没写"
        sc = json.loads(side.read_text(encoding="utf-8"))
        assert sc["schema"] == 2 and sc["base_repo"] == ns["S3_PINNED"]["repo"]
        assert sc["base_revision"] == ns["S3_PINNED"]["revision"]
        assert sc["base_expected_sha256"] == ns["S3_PINNED"]["sha256"]
    finally:
        ns["CFG"].clear(); ns["CFG"].update(snap_cfg)
        ns["OUT"], ns["MDLS"] = orig_out, orig_mdls
        shutil.rmtree(tmp, ignore_errors=True)


def test_s3_skips_when_imatrix_verified(ns):
    """需求 3：已有可信 imatrix → 整段跳过，不重下 Q8_0、不重算。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_skip_")
    imat = Path(tmp) / "imatrix.gguf"
    imat.write_bytes(b"x" * 16)
    patch = _Patch(ns)
    patch("resolve_imatrix", lambda: imat)
    patch("_verify_imatrix", lambda p, deadline=None: {
        "ok": True, "trust": "pinned_verified", "sha256": ns["IMATRIX_PINNED"]["sha256"],
        "bytes": 10701952, "n_tensors": 1008, "why": "sha256 与 pin 一致",
        "warnings": []})
    patch("llama_bin", lambda name: Path("/fake") / name)

    def boom(deadline):
        raise AssertionError("已有 imatrix 时不许再下载 Q8_0")

    patch("_ensure_q8", boom)
    snap = dict(ns["CFG"])
    try:
        ns["CFG"]["S3_SKIP_IF_IMATRIX_OK"] = True
        rec = ns["stage_s3_imatrix"](time.time() + 60)
        assert rec["skipped"] is True and rec["reason"] == "imatrix_reuse"
        assert rec["trust"] == "pinned_verified"
        assert "provenance" in rec["note"]
        md = "\n".join(ns["_s3_report_lines"](rec))
        assert "S3 未运行，复用已有 imatrix" in md
        assert "pinned_verified" in md
    finally:
        patch.undo(); ns["CFG"].clear(); ns["CFG"].update(snap)


def test_s3_untrusted_imatrix_reruns(ns):
    """imatrix 存在但不可信 → 不能跳过，必须重新生成。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_untrust_")
    imat = Path(tmp) / "imatrix.gguf"
    imat.write_bytes(b"x" * 16)
    patch = _Patch(ns)
    patch("resolve_imatrix", lambda: imat)
    patch("_verify_imatrix", lambda p, deadline=None: {
        "ok": True, "trust": "unverified", "sha256": "e" * 64, "bytes": 16,
        "n_tensors": 3, "why": "对不上", "warnings": []})
    patch("llama_bin", lambda name: Path("/fake") / name)
    called = {"q8": False}

    def fake_q8(deadline):
        called["q8"] = True
        raise RuntimeError("REACHED_REGENERATE_PATH")

    patch("_ensure_q8", fake_q8)
    snap = dict(ns["CFG"])
    try:
        ns["CFG"]["S3_SKIP_IF_IMATRIX_OK"] = True
        ns["CFG"]["S3_AUTO_FETCH_MODEL"] = True
        ns["CFG"]["IMATRIX_BASE"] = ""
        try:
            ns["stage_s3_imatrix"](time.time() + 60)
            raise AssertionError("不可信 imatrix 不该跳过")
        except RuntimeError as e:
            assert "REACHED_REGENERATE_PATH" in str(e)
        assert called["q8"] is True
    finally:
        patch.undo(); ns["CFG"].clear(); ns["CFG"].update(snap)


# ---- 需求 1：S4 禁止二次量化；S4 的两道门 ----

def test_s4_forbids_requantize_even_when_opted_in(ns):
    """即使显式设 ALLOW_REQUANTIZE=True，RUN_S4 路线也必须拒绝（不给活路）。"""
    CFG = ns["CFG"]
    work = Path(ns["WORK"])
    snap = {k: CFG[k] for k in ("BASE_GGUF", "BASE_KIND", "IMATRIX_GGUF", "IMATRIX_BASE",
                                "IMATRIX_BASE_KIND", "ALLOW_REQUANTIZE",
                                "IMATRIX_ALLOW_UNPINNED")}
    orig_bin = ns["llama_bin"]
    patch = _Patch(ns)
    patch("_reclaim_stale_q8_cache", lambda deadline: None)

    base = work / "s4_requant_base.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 8)])
    imat = work / "s4_requant_im.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])
    CFG.update(BASE_GGUF=str(base), BASE_KIND="q8_0", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=True,
               IMATRIX_ALLOW_UNPINNED=True)
    ns["llama_bin"] = lambda name: (_ for _ in ()).throw(
        AssertionError("llama_bin 不该被调用：应该在它之前就拦住"))
    try:
        ns["stage_s4_quant"](time.time() + 60)
        raise AssertionError("ALLOW_REQUANTIZE=True 也必须被拒绝")
    except ns["GuardError"] as e:
        assert "禁止二次量化" in str(e), e
        assert "ALLOW_REQUANTIZE 必须保持 False" in str(e), e
        assert "Qwen3-8B-BF16.gguf" in str(e), e
    finally:
        ns["llama_bin"] = orig_bin
        patch.undo(); CFG.update(snap)
        for p in (base, imat):
            if p.exists():
                p.unlink()


def test_s4_requires_imatrix(ns):
    """没有 imatrix 直接拒：IQ2_XS 没有 imatrix 会被上游 bail out，不做降级。"""
    patch = _Patch(ns)
    patch("resolve_imatrix", lambda: None)
    try:
        try:
            ns["stage_s4_quant"](time.time() + 30)
            raise AssertionError("没有拦住")
        except ns["GuardError"] as e:
            assert "找不到 imatrix" in str(e), e
            assert "S3" in str(e)
    finally:
        patch.undo()


def test_s4_rejects_unverified_imatrix(ns):
    """sha256 对不上任何记录的 imatrix：默认拒，显式放行后才继续。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_gate_")
    imat = Path(tmp) / "imatrix.gguf"
    imat.write_bytes(b"x" * 16)
    patch = _Patch(ns)
    patch("resolve_imatrix", lambda: imat)
    patch("_verify_imatrix", lambda p, deadline=None: {
        "ok": True, "trust": "unverified", "sha256": "e" * 64, "bytes": 16,
        "n_tensors": 3, "why": "没有找到能与 sha256 对上的 provenance 记录",
        "warnings": []})
    patch("_reclaim_stale_q8_cache", lambda deadline: None)
    sentinel = "REACHED_BASE_RESOLUTION"

    def fake_base(deadline):
        raise RuntimeError(sentinel)

    patch("_resolve_s4_base", fake_base)
    snap = dict(ns["CFG"])
    try:
        ns["CFG"]["IMATRIX_ALLOW_UNPINNED"] = False
        try:
            ns["stage_s4_quant"](time.time() + 30)
            raise AssertionError("不可信 imatrix 没有拦住")
        except ns["GuardError"] as e:
            assert "IMATRIX_ALLOW_UNPINNED" in str(e), e
            assert ns["IMATRIX_PINNED"]["sha256"] in str(e), "要把 pin 值告诉用户"

        ns["CFG"]["IMATRIX_ALLOW_UNPINNED"] = True
        try:
            ns["stage_s4_quant"](time.time() + 30)
            raise AssertionError("没走到 base 解析")
        except RuntimeError as e:
            assert sentinel in str(e), "放行后必须继续往下走"
    finally:
        patch.undo(); ns["CFG"].clear(); ns["CFG"].update(snap)


# ---- 需求 7：S5 用 S4 实际取的 base ----

def test_s4_base_for_s5(ns):
    """S5 优先分析 S4 实际用的 base（可能在暂存盘或 Input），不是只看 CFG。"""
    import tempfile as _tf

    tmp = _tf.mkdtemp(prefix="trlab_v5_s5_")
    used = Path(tmp) / "used.gguf"
    used.write_bytes(b"x" * 16)
    cfg_base = Path(tmp) / "cfg.gguf"
    cfg_base.write_bytes(b"x" * 16)
    patch = _Patch(ns)
    snap = {k: ns["CFG"][k] for k in ("BASE_GGUF",)}
    try:
        patch("_get", lambda name: {"_fairness": {"quant_base": str(used)}}
              if name == "S4_QUANT" else None)
        p, src = ns["_s4_base_for_s5"]()
        assert p == used and src == "from_S4", (p, src)

        patch("_get", lambda name: None)
        ns["CFG"]["BASE_GGUF"] = str(cfg_base)
        p, src = ns["_s4_base_for_s5"]()
        assert p == cfg_base and src == "from_CFG", (p, src)

        ns["CFG"]["BASE_GGUF"] = ""
        p, src = ns["_s4_base_for_s5"]()
        assert p is None and src is None, (p, src)
    finally:
        patch.undo(); ns["CFG"].update(snap)


def test_ram_precheck(ns):
    """RAM 预检：base + 余量与实测量对比，不够时只警告不拦。"""
    GiB = 2 ** 30
    r1 = ns["_ram_precheck"](0, "S4 量化")
    assert r1 is not None and r1["ram_gb"] > 0
    assert r1["need_gb"] == round(ns["CFG"]["S4_RAM_MARGIN_GB"], 2), r1
    r2 = ns["_ram_precheck"](100 * GiB, "S4 量化")
    assert r2["need_gb"] == round((100 * GiB + ns["CFG"]["S4_RAM_MARGIN_GB"] * GiB)
                                  / GiB, 2), r2
    assert r2["need_gb"] > r2["ram_gb"], "100 GiB base 在任何开发机上都该告警"


def test_s4_helper_hint(ns):
    """一次性操作说明必须写清文件名与步骤，且不能多于 6 次点击。"""
    hint = ns["_s4_helper_notebook_hint"]()
    for must in ("trlab_fetch_bf16.ipynb", "Internet = On", "Run All",
                 "Save Version", "Add Input", "BASE_GGUF"):
        assert must in hint, must
    assert hint.count("\n") <= 8, "说明要短，超过一屏就等于没有"


def test_s4_report_lines_render(ns):
    """S9 第 5 节：provenance 级必须显式声明，file 级不声明；表格两行。"""
    f = ns["_s4_report_lines"]
    common_fair = {
        "quant_base": "/kaggle/input/x/Qwen3-8B-BF16.gguf",
        "quant_base_kind": "bf16", "quant_base_origin": "attached_input",
        "quant_base_download": {"sha256": "a" * 64, "sha256_checked": False},
        "quant_base_repo": ns["S4_PINNED"]["repo"],
        "quant_base_revision": ns["S4_PINNED"]["revision"],
        "imatrix": "/kaggle/working/trlab/models/imatrix.gguf",
        "imatrix_sha256": ns["IMATRIX_PINNED"]["sha256"],
        "imatrix_trust": "pinned_verified",
        "base_scope": "from_high_precision", "verified": True,
    }
    r = {"IQ2_XS": {"file_gb": 2.4, "bpw_effective": 2.3125,
                    "dominant_type": "IQ2_XS", "reused": False, "sha256": "a" * 64},
         "IQ4_XS": {"file_gb": 4.4, "bpw_effective": 4.25,
                    "dominant_type": "IQ4_XS", "reused": True, "sha256": "b" * 64},
         "_fairness": dict(common_fair,
                           imatrix_base_consistency_level="provenance",
                           imatrix_base_consistent={
                               "ok": True, "verified_by": "sha256 + 来源 + 结构三项"})}
    md = "\n".join(f(r))
    for must in ("Qwen3-8B-BF16.gguf", "attached_input", "pinned_verified",
                 "provenance 级", "弱于文件级", "from_high_precision",
                 "| IQ2_XS | 2.400 |", "| IQ4_XS | 4.400 |"):
        assert must in md, must

    # file 级：不该出现"Q8_0 已回收"的声明
    r2 = {"IQ2_XS": r["IQ2_XS"], "IQ4_XS": r["IQ4_XS"],
          "_fairness": dict(common_fair,
                            imatrix_base_consistency_level="file",
                            imatrix_base_consistent={"ok": True, "verified_by": "x"})}
    md2 = "\n".join(f(r2))
    assert "文件级" in md2 and "已回收" not in md2, md2


# ---------------------------------------------------------------- v1.0 重构回归

def _have_git():
    return shutil.which("git") is not None


def _make_local_src_repo(tmp):
    """在 tmp/remote 建一个只含 research/kaggle/trlab 的 git 仓（模拟公开仓库）。

    返回 (repo_path, commit_sha)。fetch 用本地路径，所以整组测试不联网。
    """
    remote = tmp / "remote"
    pkg = remote / "research" / "kaggle" / "trlab"
    pkg.parent.mkdir(parents=True)
    shutil.copytree(HERE / "trlab", pkg,
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    for args in (("init", "-q", "."),
                 ("-c", "commit.gpgsign=false", "add", "-A"),
                 ("-c", "commit.gpgsign=false", "commit", "-q", "-m", "src")):
        r = subprocess.run(["git", *args], cwd=str(remote), env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"git {' '.join(args)} 失败：{r.stderr[-400:]}"
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(remote), env=env,
                       capture_output=True, text=True, timeout=60)
    sha = r.stdout.strip()
    assert len(sha) == 40, sha
    return remote, sha


def _gen_notebook(tmp, commit, repo_url, embed=False):
    """调 build_notebook 生成 notebook；repo_url 可指向本地路径（离线测 fetch）。"""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import build_notebook as bn
    old_url, old_argv = bn.REPO_URL, sys.argv
    try:
        bn.REPO_URL = repo_url
        sys.argv = ["build_notebook.py", "--commit", commit,
                    "--out", str(tmp / "nb.ipynb")] + (["--embed"] if embed else [])
        bn.main()
    finally:
        bn.REPO_URL, sys.argv = old_url, old_argv
    out = tmp / "nb.ipynb"
    assert out.exists(), "notebook 没生成"
    return out


def _run_bootstrap_cell(nb_path, work):
    """干净子进程里执行 notebook 的两个代码格（到 import trlab 为止，不跑 pipeline）。"""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 2, f"notebook 应有 2 个代码格，实际 {len(code)}"
    src = code[0] + "\n\n" + code[1].split("trlab.run(")[0]
    script = work / "_nb_cell.py"
    script.write_text(src, encoding="utf-8")
    return subprocess.run([sys.executable, str(script)], cwd=str(work),
                          capture_output=True, text=True, timeout=300)


def test_import_has_no_side_effects(ns):
    """v1.0：import trlab 必须零副作用 —— 不建目录、不打印、不联网。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_imp_"))
    try:
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import trlab; "
                f"print('IMPORTED', trlab.__version__)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(tmp),
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-800:]
        assert "IMPORTED" in r.stdout, r.stdout
        assert list(tmp.iterdir()) == [], f"import 竟创建了文件：{list(tmp.iterdir())}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_each_module_imports_alone(ns):
    """无循环 import：每个子模块在干净子进程里单独 import 都必须成功。"""
    subs = [m.__name__ for m in ns._mods if "." in m.__name__]
    assert len(subs) >= 15, f"子模块太少：{subs}"
    env = dict(os.environ, PYTHONPATH=str(HERE))
    for full in subs:
        r = subprocess.run([sys.executable, "-c", f"import {full}"], env=env,
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, f"{full} 单独 import 失败：\n{r.stderr[-500:]}"


def test_switches_off_change_nothing(ns):
    """三个开关默认 False：_apply_run_switches() 一个字段都不该改（误点 Run All 安全）。"""
    cfg, stages = ns["CFG"], ns["STAGES"]
    saved = (copy.deepcopy(cfg), copy.deepcopy(stages))
    try:
        for k in ("RUN_S2", "RUN_S3", "RUN_S4"):
            cfg[k] = False
        plan = ns["_apply_run_switches"]()
        assert plan == {"run_s2": False, "run_s3": False, "run_s4": False,
                        "run_eval": False}, plan
        assert cfg == saved[0], "全 False 时 CFG 被改了"
        assert stages == saved[1], "全 False 时 STAGES 被改了"
    finally:
        cfg.clear(); cfg.update(saved[0])
        stages.clear(); stages.update(saved[1])


def test_switch_s4_expands_plan(ns):
    """RUN_S4=True：S2+S3+S4+S5 开，S6/S7/S8/SX 关，BASE_GGUF 仍留空（由 S4 自己找）。"""
    cfg, stages = ns["CFG"], ns["STAGES"]
    saved = (copy.deepcopy(cfg), copy.deepcopy(stages))
    try:
        cfg["RUN_S4"] = True
        plan = ns["_apply_run_switches"]()
        assert plan["run_s2"] and plan["run_s3"] and plan["run_s4"], plan
        on = set(plan["stages"])
        for need in ("S2_BUILD", "S3_IMATRIX", "S4_QUANT", "S5_BYTES", "S4_PREFLIGHT"):
            assert need in on, f"{need} 没开：{sorted(on)}"
        for off in ("S6_NLL", "S7_QUALITY", "S8_BF16_ANCHOR", "SX_EXPLORE"):
            assert off not in on, f"{off} 不该开：{sorted(on)}"
        assert cfg["BASE_GGUF"] == "", cfg["BASE_GGUF"]
        assert plan["quant_base"] == ns["S4_PINNED"]["filename"], plan
        assert plan["imatrix_base"] == ns["S3_PINNED"]["filename"], plan
        assert cfg["ALLOW_REQUANTIZE"] is False
    finally:
        cfg.clear(); cfg.update(saved[0])
        stages.clear(); stages.update(saved[1])


def test_run_returns_plan_and_stages(ns):
    """入口 run() 返回 plan + 各阶段状态；默认只跑 S0/S1/S9。"""
    cfg = ns["CFG"]
    saved = {k: cfg[k] for k in ("RUN_S2", "RUN_S3", "RUN_S4")}
    try:
        for k in saved:
            cfg[k] = False
        out = ns["run"]()
        assert isinstance(out, dict) and "plan" in out and "stages" in out, out
        assert out["plan"]["run_s4"] is False, out["plan"]
        for k in ("S0_ENV", "S1_SMOKE", "S9_REPORT"):
            assert out["stages"][k] == "ok", (k, out["stages"][k])
        for k in ("S2_BUILD", "S3_IMATRIX", "S4_QUANT", "S5_BYTES"):
            assert out["stages"][k] is None, (k, out["stages"][k])
    finally:
        cfg.update(saved)


def test_overrides_route_stages_vs_cfg(ns):
    """v1.0 notebook 只有三个开关格；S6–S8/SX 靠 run(..., S6_NLL=True) 打开。"""
    cfg, stages = ns["CFG"], ns["STAGES"]
    saved = (copy.deepcopy(cfg), copy.deepcopy(stages))
    try:
        ns["_apply_overrides"]({"S6_NLL": True, "SX_EXPLORE": False, "NLL_CTX": 512})
        assert stages["S6_NLL"] is True
        assert stages["SX_EXPLORE"] is False
        assert stages["S2_BUILD"] is False
        assert cfg["NLL_CTX"] == 512
        assert "S6_NLL" not in cfg and "NLL_CTX" not in stages
        ns["_apply_overrides"]({"S6_NLL": 0})
        assert stages["S6_NLL"] is False
    finally:
        cfg.clear(); cfg.update(saved[0])
        stages.clear(); stages.update(saved[1])


def test_generated_notebook_is_thin(ns):
    """notebook 必须薄：≤5 cells、<20 KB、含钉死 commit、默认不内嵌执行包。"""
    nb_path = HERE / "trlab_kaggle.ipynb"
    if not nb_path.exists():
        if not _have_git():
            print("    （跳过：没有 notebook 也没有 git）")
            return
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _, sha = _make_local_src_repo(tmp)
            nb_path = _gen_notebook(tmp, sha, "https://example.invalid/x")
    raw = nb_path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert len(nb["cells"]) <= 5, f"notebook 有 {len(nb['cells'])} cells"
    assert nb_path.stat().st_size < 20_000, nb_path.stat().st_size
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert len(src) < 12_000, f"notebook 源码 {len(src)} B"
    for must in ("RUN_EVAL", "RUN_S2", "RUN_S3", "RUN_S4", "trlab.run(", "rev-parse"):
        assert must in src, must
    # v1.1：RUN_EVAL 必须真的被传给 run()，否则开关格里的那一行等于装饰
    assert re.search(r"trlab\.run\([^)]*\brun_eval=RUN_EVAL", src), \
        "notebook 的 trlab.run(...) 没把 RUN_EVAL 传进去"
    assert "_BUNDLE_B64" not in src, "默认 notebook 不该内嵌执行包"
    m = re.search(r'SRC_COMMIT\s*=\s*"([0-9a-f]{40})"', src)
    assert m, "找不到钉死的 SRC_COMMIT"
    if _have_git():
        r = subprocess.run(["git", "merge-base", "--is-ancestor", m.group(1), "HEAD"],
                           cwd=str(HERE), capture_output=True, text=True)
        assert r.returncode == 0, (f"notebook 钉的 {m.group(1)[:12]} 不在当前历史里，"
                                   "请重跑 build_notebook.py")


def test_bootstrap_fetch_verify_reuse(ns):
    """冷启动：本地仓 sparse fetch + rev-parse 校验 + 同 Session 二次执行复用。"""
    if not _have_git():
        print("    （跳过：没有 git）")
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        remote, sha = _make_local_src_repo(tmp)
        nb = _gen_notebook(tmp, sha, str(remote))
        work = tmp / "work"
        work.mkdir()
        r = _run_bootstrap_cell(nb, work)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        assert "执行包就绪" in r.stdout, r.stdout[-800:]
        pkg = work / "_trlab_src" / "research" / "kaggle" / "trlab"
        assert (pkg / "__init__.py").exists()
        stamp = work / "_trlab_src" / ".trlab_commit"
        assert stamp.read_text().strip() == sha
        for f in sorted((HERE / "trlab").glob("*.py")):
            assert (pkg / f.name).read_bytes() == f.read_bytes(), f.name
        r2 = _run_bootstrap_cell(nb, work)
        assert r2.returncode == 0, r2.stderr[-800:]
        assert "已在盘上" in r2.stdout, r2.stdout[-800:]


def test_bootstrap_embedded_fallback(ns):
    """仓库不可读时回退内嵌包（sha256 校验后解压），不静默失败。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        remote, sha = _make_local_src_repo(tmp)
        nb = _gen_notebook(tmp, sha, str(tmp / "no_such_repo"), embed=True)
        work = tmp / "work"
        work.mkdir()
        r = _run_bootstrap_cell(nb, work)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        assert "从仓库取包失败" in r.stdout, r.stdout[-800:]
        assert "改用 notebook 内嵌的执行包" in r.stdout, r.stdout[-800:]
        pkg = work / "_trlab_src" / "research" / "kaggle" / "trlab"
        assert (pkg / "__init__.py").exists()
        assert (work / "_trlab_src" / ".trlab_commit").read_text().strip() == sha


def test_bootstrap_embedded_sha_guard(ns):
    """内嵌包 sha 标注被篡改 → fail closed，绝不静默解压。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        remote, sha = _make_local_src_repo(tmp)
        nb_path = _gen_notebook(tmp, sha, str(tmp / "no_such_repo"), embed=True)
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        tampered = False
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            txt = "".join(c["source"])
            m = re.search(r'_BUNDLE_SHA = "([0-9a-f]{64})"', txt)
            if m:
                c["source"] = txt.replace(m.group(1), "0" * 64).splitlines(keepends=True)
                tampered = True
                break
        assert tampered, "内嵌 notebook 里找不到 _BUNDLE_SHA"
        bad = tmp / "nb_bad.ipynb"
        bad.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        work = tmp / "work"
        work.mkdir()
        r = _run_bootstrap_cell(bad, work)
        assert r.returncode != 0, "sha 不符居然跑通了"
        assert "校验失败" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-800:]
        assert not (work / "_trlab_src" / "research" / "kaggle" / "trlab"
                    / "__init__.py").exists()


# ---------------------------------------------------------------- v1.0.1 S4 预检

class _PathShutil:
    """按路径给不同空闲容量的假 shutil（预检测 working 与暂存盘两条账）。"""

    def __init__(self, mapping, default):
        self._m = {str(k): v for k, v in mapping.items()}
        self._d = default

    def disk_usage(self, p):
        return _FakeUsage(self._m.get(str(p), self._d))

    def which(self, name):
        return None


def _pf_env(ns, *, work_free_gb=200.0, scratch=(), input_hits=(), imat=None, devs=None):
    """预检测试的公共打桩：容量 / 暂存盘候选 / Input 扫描 / imatrix / st_dev。"""
    patch = _Patch(ns)
    GiB = 2 ** 30
    patch("shutil", _FakeShutil(free=int(work_free_gb * GiB)))
    patch("_preflight_scratch_candidates", lambda: list(scratch))
    patch("_scan_input_gguf", lambda **k: list(input_hits))
    patch("resolve_imatrix", lambda: imat)
    if devs:
        import os as _os
        patch("os", _FakeOS(_os, devs))
    return patch


def test_mount_info_parser(ns):
    """/proc/mounts 解析：最长前缀匹配 + fstype/device；读不到返回 None。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_mnt_"))
    try:
        mf = tmp / "mounts"
        mf.write_text(
            "overlay / overlay rw,relatime 0 0\n"
            "/dev/nvme0n1p1 /kaggle/working ext4 rw,relatime 0 0\n"
            "/dev/nvme1n1 /kaggle/tmp xfs rw,noatime,quota 0 0\n", encoding="utf-8")
        mi = ns["_mount_info"](Path("/kaggle/tmp/x"), mounts_path=str(mf))
        assert mi and mi["fstype"] == "xfs" and mi["device"] == "/dev/nvme1n1", mi
        assert "quota" in mi["opts"], mi
        mi2 = ns["_mount_info"](Path("/kaggle/working/trlab"), mounts_path=str(mf))
        assert mi2["mount"] == "/kaggle/working", mi2
        mi3 = ns["_mount_info"](Path("/usr/local/bin"), mounts_path=str(mf))
        assert mi3["mount"] == "/", mi3
        assert ns["_mount_info"](Path("/x"), mounts_path=str(tmp / "nope")) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_probe_readonly(ns):
    """写入探针必须真写：只读目录直接判否（os.access 会说谎，这个不会）。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_wp_"))
    ro = tmp / "ro"
    ro.mkdir()
    try:
        ok, why = ns["_write_probe"](ro)
        assert ok is True, why
        os.chmod(ro, 0o555)
        ok2, why2 = ns["_write_probe"](ro)
        assert ok2 is False and why2, "只读目录竟然写成功了"
    finally:
        os.chmod(ro, 0o755)
        shutil.rmtree(tmp, ignore_errors=True)


def test_s4_preflight_scratch_ok(ns):
    """暂存盘是另一个文件系统且够大 → BF16 下到那里，预检通过。"""
    import os as _os
    tmp = Path(tempfile.mkdtemp(prefix="trlab_pf_ok_"))
    scratch = tmp / "kaggle-tmp"
    scratch.mkdir()
    wdev = _os.stat(ns["WORK"]).st_dev
    patch = _pf_env(ns, work_free_gb=200.0, scratch=[scratch],
                    devs={str(scratch): wdev + 7})
    try:
        r = ns["stage_s4_preflight"](time.time() + 60)
        assert r["ok"] is True, r
        assert r["bf16"]["source"] == "scratch_download", r["bf16"]
        assert r["scratch"][0]["usable"] is True, r["scratch"]
        assert r["s3_will_run"] is True, r["imatrix"]     # imatrix 缺失 → S3 会重算
        assert "8.11" in r["imatrix"]["note"], r["imatrix"]
        assert r["work_peak_gb"] > 0 and r["work_free_gb"] == 200.0
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s4_preflight_input_bf16(ns):
    """/kaggle/input 已有字节数吻合的 BF16 → 一个暂存盘都不需要。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_pf_in_"))
    ind = tmp / "qwen3"
    ind.mkdir(parents=True)
    bf = ind / "Qwen3-8B-BF16.gguf"
    bf.write_bytes(b"")
    os.truncate(str(bf), ns["S4_PINNED"]["bytes"])   # 稀疏文件：不真占 15 GiB
    patch = _pf_env(ns, work_free_gb=200.0, scratch=[], input_hits=[bf])
    try:
        r = ns["stage_s4_preflight"](time.time() + 60)
        assert r["bf16"]["source"] == "attached_input", r["bf16"]
        assert r["ok"] is True, r
        assert r["bf16"]["sha256_checked"] is False, "预检不算 sha256（S4 才核）"
        assert "S4 核对" in r["bf16"]["note"], r["bf16"]
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s4_preflight_fail_closed_no_place(ns):
    """没有 Input、没有暂存盘、working 只有 19.07 GiB → 下载前 fail-closed。"""
    patch = _pf_env(ns, work_free_gb=19.07, scratch=[], input_hits=[])
    patch("_fetch_pinned_blob", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("预检失败时不该下载任何东西")))
    try:
        ns["stage_s4_preflight"](time.time() + 60)
        raise AssertionError("应当 fail-closed")
    except ns["GuardError"] as e:
        msg = str(e)
        assert "没有可用的落点" in msg, msg
        assert "trlab_fetch_bf16.ipynb" in msg, "必须给唯一的最简下一步"
        assert msg.count("唯一的最简下一步") == 1, msg
    finally:
        patch.undo()


def test_s4_preflight_working_peak_too_small(ns):
    """BF16 有 Input，但 working 放不下 build+产物 → 提示清 working，而不是去搞 Input。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_pf_pk_"))
    bf = tmp / "Qwen3-8B-BF16.gguf"
    bf.write_bytes(b"")
    os.truncate(str(bf), ns["S4_PINNED"]["bytes"])
    patch = _pf_env(ns, work_free_gb=6.0, scratch=[], input_hits=[bf])
    try:
        ns["stage_s4_preflight"](time.time() + 60)
        raise AssertionError("应当 fail-closed")
    except ns["GuardError"] as e:
        msg = str(e)
        assert "峰值账不够" in msg, msg
        assert "清掉 working" in msg, msg
        assert "trlab_fetch_bf16" not in msg, "这不是 Input 的问题，别把用户带错方向"
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s4_preflight_scratch_rejections(ns):
    """同设备 / tmpfs / 只读 / 容量不足 / 同设备名 —— 五种候选各有具体理由。"""
    import os as _os
    tmp = Path(tempfile.mkdtemp(prefix="trlab_pf_rej_"))
    same, ro, tiny, ram, mnt = (tmp / n for n in ("same", "ro", "tiny", "ram", "mnt"))
    for d in (same, ro, tiny, ram, mnt):
        d.mkdir()
    wdev = _os.stat(ns["WORK"]).st_dev
    GiB = 2 ** 30
    patch = _Patch(ns)
    patch("_preflight_scratch_candidates", lambda: [same, ro, tiny, ram, mnt])
    patch("os", _FakeOS(_os, {str(same): wdev, str(ro): wdev + 1,
                              str(tiny): wdev + 2, str(ram): wdev + 3,
                              str(mnt): wdev + 4}))
    patch("shutil", _PathShutil({tiny: 1 * GiB}, 200 * GiB))
    mounts = {
        str(ns["WORK"]): {"device": "/dev/w", "mount": str(ns["WORK"]),
                          "fstype": "ext4", "opts": []},
        str(ram): {"device": "/dev/ram0", "mount": str(ram),
                   "fstype": "tmpfs", "opts": []},
        str(mnt): {"device": "/dev/w", "mount": str(mnt),
                   "fstype": "ext4", "opts": []},
    }
    patch("_mount_info", lambda p, mounts_path="/proc/mounts": mounts.get(str(Path(p))))
    os.chmod(ro, 0o555)
    try:
        ev = ns["_preflight_scratch_evidence"](ns["S4_PINNED"]["bytes"])
        by = {e["path"]: e for e in ev}
        assert by[str(same)]["usable"] is False and "同一个文件系统" in by[str(same)]["why"]
        assert by[str(ro)]["usable"] is False and "写不进去" in by[str(ro)]["why"]
        assert by[str(tiny)]["usable"] is False and "空闲" in by[str(tiny)]["why"]
        assert by[str(ram)]["usable"] is False and "内存盘" in by[str(ram)]["why"]
        assert by[str(mnt)]["usable"] is False and "同一设备" in by[str(mnt)]["why"]
    finally:
        os.chmod(ro, 0o755)
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s4_preflight_config_and_budget(ns):
    """预检必须每次重跑；S4 预算从 1800 s 提到 5400 s 并加单格式帽。"""
    assert "S4_PREFLIGHT" in ns["ALWAYS_RERUN"], "磁盘状态每次都会变，不能缓存 ok"
    b = ns["CFG"]["STAGE_BUDGET_SEC"]
    assert b["S4_QUANT"] >= 5400, f"S4 总预算还是 {b['S4_QUANT']} s，误杀风险没解决"
    assert b["S4_PREFLIGHT"] <= 180, b["S4_PREFLIGHT"]
    assert ns["CFG"]["S4_PER_FORMAT_BUDGET_SEC"] >= 1800
    g = ns["STAGE_REQUIRES_OK"]
    assert g["S2_BUILD"] == ["S4_PREFLIGHT"], g["S2_BUILD"]
    assert "S4_PREFLIGHT" in g["S3_IMATRIX"] and "S4_PREFLIGHT" in g["S4_QUANT"]
    assert "S4_PREFLIGHT" in g["S5_BYTES"]


def test_preflight_only_runs_only_preflight(ns):
    """preflight_only=True：只跑 S4_PREFLIGHT，S0/S1/S9 都不跑。"""
    cfg, stages, STATE = ns["CFG"], ns["STAGES"], ns["STATE"]
    saved = (copy.deepcopy(cfg), copy.deepcopy(stages), dict(STATE))
    patch = _pf_env(ns, work_free_gb=19.07, scratch=[], input_hits=[])
    try:
        # STATE 是所有测试共享的全局量，前面几项已经把 S0_ENV 跑成 ok 了；
        # 不先清空，"S0 没跑"这件事根本断言不出来。
        STATE.clear()
        out = ns["run"](preflight_only=True)
        assert out["preflight"] == "failed", out
        assert out["stages"]["S4_PREFLIGHT"] == "error"
        for k in ("S0_ENV", "S1_SMOKE", "S2_BUILD", "S3_IMATRIX",
                  "S4_QUANT", "S5_BYTES", "S9_REPORT"):
            assert out["stages"][k] is None, (k, out["stages"][k])
        assert stages["S4_PREFLIGHT"] is True and stages["S9_REPORT"] is False
    finally:
        patch.undo()
        cfg.clear(); cfg.update(saved[0])
        stages.clear(); stages.update(saved[1])
        STATE.clear(); STATE.update(saved[2])


def test_s4_quant_budget_is_protective(ns):
    """单格式预算被杀时：日志路径 / 已完成格式 / 重跑复用 / "保护上限" 都要说清。"""
    CFG = ns["CFG"]
    work = Path(ns["WORK"])
    snap = {k: CFG[k] for k in ("BASE_GGUF", "BASE_KIND", "IMATRIX_GGUF", "IMATRIX_BASE",
                                "IMATRIX_BASE_KIND", "ALLOW_REQUANTIZE",
                                "IMATRIX_ALLOW_UNPINNED", "S4_PER_FORMAT_BUDGET_SEC")}
    patch = _Patch(ns)
    base = work / "s4_budget_base.gguf"
    ns["gguf_write_synth"](base, [("blk.0.attn_q.weight", (256, 256), 8)])
    imat = work / "s4_budget_im.gguf"
    ns["gguf_write_synth"](imat, [("blk.0.attn_q.weight", (256, 256), 0)])
    CFG.update(BASE_GGUF=str(base), BASE_KIND="bf16", IMATRIX_GGUF=str(imat),
               IMATRIX_BASE="", IMATRIX_BASE_KIND="", ALLOW_REQUANTIZE=False,
               IMATRIX_ALLOW_UNPINNED=True, S4_PER_FORMAT_BUDGET_SEC=1)
    patch("_reclaim_stale_q8_cache", lambda deadline: None)
    patch("_resolve_s4_base",
          lambda deadline: (base, "manual", {"bytes": base.stat().st_size}))
    patch("_imatrix_base_consistency", lambda *a, **k: {
        "ok": True, "level": "provenance", "n_tensors": 1,
        "architecture": "qwen3", "verified_by": "test"})
    patch("llama_bin", lambda name: "/bin/true")
    patch("_s4_output_bytes", lambda: 0)
    patch("_ram_precheck", lambda *a, **k: None)

    def fake_run_cmd(cmd, log_path, deadline, cwd=None, env=None):
        raise ns["BudgetExceeded"]("命令超预算被杀")

    patch("run_cmd", fake_run_cmd)
    try:
        ns["stage_s4_quant"](time.time() + 600)
        raise AssertionError("应当抛 BudgetExceeded")
    except ns["BudgetExceeded"] as e:
        msg = str(e)
        for must in ("超出单格式预算", "日志保留在", "已完成的格式",
                     "不会重量", "保护上限"):
            assert must in msg, (must, msg)
    finally:
        patch.undo(); CFG.update(snap)
        for p in (base, imat):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------- v1.0.2 S3 bytes 回归

class _FakeParquetModule:
    """假 pyarrow.parquet：只为让**真的** `_parquet_to_txt` 跑完。

    不测 pyarrow 本身，测的是"转换元数据里有哪些 key"——那正是 v0.4 / v1.0.1
    真实 Kaggle 失败的地方（key 叫 bytes → 与 stamp 的 bytes= 撞车 → TypeError）。
    """

    __version__ = "0.0-fake"

    class _Schema:
        names = ["text"]

        def field(self, c):
            return types.SimpleNamespace(type="string")

    class _Col:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return self._rows

    class _Table:
        schema = None

        def __init__(self):
            self.schema = _FakeParquetModule._Schema()

        def column(self, c):
            return _FakeParquetModule._Col(["first paragraph", "", "second paragraph"])

    @staticmethod
    def read_table(p):
        return _FakeParquetModule._Table()


def _with_fake_parquet(fn):
    """临时把假 pyarrow 塞进 sys.modules，跑完还原（含"本来没装"的情况）。"""
    saved = {k: sys.modules.get(k) for k in ("pyarrow", "pyarrow.parquet")}
    mod = types.ModuleType("pyarrow")
    pq = types.ModuleType("pyarrow.parquet")
    pq.read_table = _FakeParquetModule.read_table
    pq.__version__ = _FakeParquetModule.__version__
    mod.parquet = pq
    sys.modules["pyarrow"] = mod
    sys.modules["pyarrow.parquet"] = pq
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_calib_txt_full_flow_no_duplicate_bytes(ns):
    """完整走 `_ensure_calib_txt`：下载(桩) → 真转换 → 写 stamp → 二次命中缓存。

    钉住 v0.4 的真实 Kaggle 失败：`_parquet_to_txt` 返回 `bytes` 时，
    `_write_stamp(..., bytes=pin["bytes"], **conv)` 直接
    `TypeError: got multiple values for keyword argument 'bytes'`。
    这里不 stub `_ensure_calib_txt`（旧测试就是这么漏掉的），而是走完整链路。
    """
    tmp = Path(tempfile.mkdtemp(prefix="trlab_calib_flow_"))
    parquet = tmp / "dl" / "train.parquet"
    txt = tmp / "calib.txt"
    stamp = tmp / "calib.stamp.json"
    patch = _Patch(ns)
    patch("_calib_parquet_path", lambda: parquet)
    patch("_calib_txt_path", lambda: txt)
    patch("_calib_stamp_path", lambda: stamp)

    def fake_fetch(pin, dst, deadline, tag):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"PARQUET-FAKE" * 10)   # 内容无所谓，转换被假 pyarrow 接管
        return {"origin": "downloaded", "bytes": pin["bytes"],
                "sha256": pin["sha256"], "seconds": 0.0}

    patch("_fetch_pinned_blob", fake_fetch)
    try:
        p, meta = _with_fake_parquet(lambda: ns["_ensure_calib_txt"](time.time() + 60))
        assert p == txt and txt.exists()
        # 1) 不炸（这是回归本体）
        assert meta["origin"] == "downloaded"
        # 2) parquet 的 bytes 必须原样保留，不能被 txt 大小覆盖
        assert meta["bytes"] == ns["CALIB_PINNED"]["bytes"], meta
        assert meta["sha256"] == ns["CALIB_PINNED"]["sha256"], meta
        # 3) 转换产物的字节数要单独有一个不会撞车的名字
        assert meta["converted_bytes"] == txt.stat().st_size, meta

        st = json.loads(stamp.read_text())
        assert st["bytes"] == ns["CALIB_PINNED"]["bytes"], st
        assert st["txt_bytes"] == txt.stat().st_size, st
        assert st["converted_bytes"] == txt.stat().st_size, st
        assert st["nonempty_rows"] == 2, st
        assert st["split"] == "train" and st["sha256"] == ns["CALIB_PINNED"]["sha256"]

        # 4) 二次调用命中缓存，不再下载（_fetch 被换成会炸的桩）
        patch("_fetch_pinned_blob", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("缓存命中还去下载了")))
        p2, meta2 = ns["_ensure_calib_txt"](time.time() + 60)
        assert meta2["origin"] == "cache" and p2 == txt
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_splat_into_write_stamp(ns):
    """静态不变量：`_write_stamp(...)` 不许**同时**出现显式 `bytes=` 与 `**` 摊字典。

    v0.4 / v1.0.1 的 bug 就是 `**conv` 里的 bytes 撞上显式 `bytes=`。用 AST 查
    （注释和 docstring 里出现的同名字样不算），比字符串搜索可靠。
    """
    import ast

    offenders = []
    for f in sorted((HERE / "trlab").glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute) else
                    (fn.id if isinstance(fn, ast.Name) else None))
            if name != "_write_stamp":
                continue
            has_bytes = any(k.arg == "bytes" for k in node.keywords)
            has_splat = any(k.arg is None for k in node.keywords)
            if has_bytes and has_splat:
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, (
        "`_write_stamp` 不能同时用 bytes= 和 ** 摊字典（v0.4 的 TypeError 根因）："
        + ", ".join(offenders))


# ---------------------------------------------------------------- v1.1 HF 输入 / S6-S7 评测

def _hf_env(ns, *, token="hf_TESTTOKEN_do_not_log", source="env:HF_TOKEN"):
    """把 token 查找钉住，避免读到真机环境变量/缓存。"""
    patch = _Patch(ns)
    patch("hf_token", lambda env=None, allow_cache=True: {"token": token,
                                                          "source": source if token else None})
    patch("_kaggle_secret", lambda *a, **k: None)
    patch("_hf_cache_token", lambda: None)
    return patch


def test_hf_token_chain(ns):
    """凭据查找优先级：env → kaggle secret → HF 缓存文件；都没有就明确返回 None。"""
    f = ns["hf_token"]
    assert f(env={"HF_TOKEN": "hf_a"}, allow_cache=False) == {
        "token": "hf_a", "source": "env:HF_TOKEN"}
    assert f(env={"HUGGING_FACE_HUB_TOKEN": "hf_b"}, allow_cache=False)["source"] == \
        "env:HUGGING_FACE_HUB_TOKEN"
    # HF_TOKEN 优先于其它两个环境变量
    r = f(env={"HF_TOKEN": "first", "HUGGINGFACEHUB_API_TOKEN": "second"},
          allow_cache=False)
    assert r["token"] == "first", r

    patch = _Patch(ns)
    try:
        patch("_kaggle_secret", lambda *a, **k: "hf_secret")
        patch("_hf_cache_token", lambda: "hf_cache")
        assert f(env={})["source"] == "kaggle_secret"
        patch("_kaggle_secret", lambda *a, **k: None)
        assert f(env={})["source"] == "hf_cache_file"
        patch("_hf_cache_token", lambda: None)
        assert f(env={}) == {"token": None, "source": None}
    finally:
        patch.undo()


def test_hf_token_never_logged(ns):
    """token 只能以"来源名字"出现：异常消息、status() 的 JSON 里都不许有值。"""
    secret = "hf_THIS_MUST_NEVER_APPEAR_9f8e7d"
    patch = _hf_env(ns, token=secret)
    try:
        patch("_http_get", lambda *a, **k: (401, {}, b""))
        out = ns["status"]("someone/private-repo")
        blob = json.dumps(out, ensure_ascii=False, default=str)
        assert secret not in blob, "status() 把 token 打出来了"
        assert out["token_present"] is True and out["token_source"] == "env:HF_TOKEN"
        assert out["ok"] is False and "不要把 token 写进" in out["error"]

        # 底层库异常里带 token 时，redact 必须抹掉
        assert secret not in ns["redact"](f"boom {secret} end", secret)
        assert "***REDACTED***" in ns["redact"](f"boom {secret}", secret)
    finally:
        patch.undo()


def test_hf_tree_parse(ns):
    """tree 解析：size 取整、lfs.oid 当 sha256、非 LFS 不冒充 sha256、目录跳过。"""
    page2 = [{"type": "file", "path": "models/qwen3-8b-IQ4_XS.gguf",
              "size": 4560000000, "lfs": {"oid": "b" * 64}}]
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None, follow=True, read_body=True):
        calls["n"] += 1
        if calls["n"] == 1:
            body = [{"type": "file", "path": "models/qwen3-8b-IQ2_XS.gguf",
                     "size": 2900000000, "lfs": {"oid": "a" * 64}},
                    {"type": "file", "path": "models/README.md", "size": 123},
                    {"type": "directory", "path": "models/sub"}]
            return 200, {"Link": '<https://hf/api/next>; rel="next"'}, \
                json.dumps(body).encode()
        return 200, {}, json.dumps(page2).encode()

    patch = _Patch(ns)
    patch("_http_get", fake_get)
    try:
        files = ns["repo_files"]("r/x")
        by = {f["path"]: f for f in files}
        assert len(files) == 3 and calls["n"] == 2, (len(files), calls)
        g = by["models/qwen3-8b-IQ2_XS.gguf"]
        assert g["size"] == 2900000000 and g["sha256"] == "a" * 64 and g["is_lfs"] is True
        md = by["models/README.md"]
        assert md["sha256"] is None and md["is_lfs"] is False, \
            "非 LFS 文件 HF 只给 git blob sha1，不能冒充内容 sha256"
        assert by["models/qwen3-8b-IQ4_XS.gguf"]["sha256"] == "b" * 64
    finally:
        patch.undo()


def test_hf_api_errors_are_actionable(ns):
    """401/404 都要给"怎么给 token"，而且私有仓库的 404 必须解释清楚。"""
    patch = _Patch(ns)
    try:
        patch("_http_get", lambda *a, **k: (401, {}, b""))
        try:
            ns["repo_files"]("r/x")
            raise AssertionError("401 必须抛")
        except ns["GuardError"] as e:
            assert "不要把 token 写进" in str(e) and "HF_TOKEN" in str(e)

        patch("_http_get", lambda *a, **k: (404, {}, b""))
        try:
            ns["repo_files"]("r/x")
            raise AssertionError("404 必须抛")
        except ns["GuardError"] as e:
            msg = str(e)
            assert "404" in msg and "没有 token" in msg, msg
    finally:
        patch.undo()


def test_hf_fetch_file_reuse_and_etag_guard(ns):
    """下载：tree 的 size/oid 是期望值；命中本地就复用；etag 与 oid 不符要拒收。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_hf_fetch_"))
    dst = tmp / "models" / "qwen3-8b-IQ2_XS.gguf"
    state = {"fetch": 0, "etag": "a" * 64}

    def fake_fetch(pin, dstp, deadline, tag, fetch_url=None, auth_headers=None):
        state["fetch"] += 1
        Path(dstp).write_bytes(b"z" * 16)
        return {"origin": "downloaded", "bytes": 16,
                "sha256": pin.get("sha256") or "z" * 64, "path": str(dstp)}

    patch = _hf_env(ns)
    patch("_fetch_pinned_blob", fake_fetch)
    patch("resolve_signed", lambda url, token=None: ("https://cdn/x", state["etag"]))
    expect = {"bytes": 16, "sha256": "a" * 64}
    try:
        r1 = ns["fetch_file"]("r/x", "models/qwen3-8b-IQ2_XS.gguf", dst,
                              time.time() + 60, token="hf_t", expect=expect)
        assert r1["origin"] == "downloaded" and r1["sha256_verified"] is True
        r2 = ns["fetch_file"]("r/x", "models/qwen3-8b-IQ2_XS.gguf", dst,
                              time.time() + 60, token="hf_t", expect=expect)
        assert r2["origin"] == "reused" and state["fetch"] == 1, \
            "本地已有且与 tree 记录一致时必须复用，不许重下 2.7 GiB"

        state["etag"] = "b" * 64          # 上游换过文件
        try:
            ns["fetch_file"]("r/x", "models/qwen3-8b-IQ2_XS.gguf", dst,
                             time.time() + 60, token="hf_t", expect=expect,
                             force=True)
            raise AssertionError("etag 与 tree oid 不符必须拒收")
        except ns["GuardError"] as e:
            assert "x-linked-etag" in str(e), str(e)
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_hf_verify_remote_marks_what_was_checked(ns):
    """上传后复核：size 必比；LFS 再比内容 sha256；非 LFS 明确说没比内容。"""
    entries = [
        {"path": "models/a.gguf", "size": 10, "sha256": "a" * 64, "is_lfs": True},
        {"path": "results/report.md", "size": 5, "sha256": None, "is_lfs": False},
    ]
    patch = _Patch(ns)
    patch("repo_files", lambda *a, **k: entries)
    try:
        ok = ns["verify_remote"]("r/x", [
            {"path": "models/a.gguf", "bytes": 10, "sha256": "a" * 64},
            {"path": "results/report.md", "bytes": 5, "sha256": "c" * 64},
        ])
        assert ok["ok"] is True, ok
        by = {f["path"]: f for f in ok["files"]}
        assert by["models/a.gguf"]["content_sha256_verified"] is True
        assert by["results/report.md"]["content_sha256_verified"] is False, \
            "非 LFS 文件没有可比的内容哈希，不能假装比过"

        bad = ns["verify_remote"]("r/x", [
            {"path": "models/a.gguf", "bytes": 11, "sha256": "a" * 64}])
        assert bad["ok"] is False and bad["files"][0]["size_match"] is False

        wrong_sha = ns["verify_remote"]("r/x", [
            {"path": "models/a.gguf", "bytes": 10, "sha256": "d" * 64}])
        assert wrong_sha["ok"] is False, "LFS sha256 不符必须判否"
    finally:
        patch.undo()


def test_hf_upload_guards_and_optin(ns):
    """上传：没 token 就明说；缺库时禁止 `pip -U`；默认 opt-in 关闭时一个字节都不发。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_hf_up_"))
    f = tmp / "report.md"
    f.write_text("x", encoding="utf-8")
    patch = _hf_env(ns, token=None)
    try:
        try:
            ns["upload_files"]("r/x", [(f, "results/report.md")], token=None)
            raise AssertionError("没有 token 必须拒绝")
        except ns["GuardError"] as e:
            assert "不要把 token 写进" in str(e), str(e)

        # 默认不 opt-in：upload_after_run 必须直接跳过
        CFG = ns["CFG"]
        old = CFG["HF_UPLOAD_AFTER_RUN"]
        CFG["HF_UPLOAD_AFTER_RUN"] = False
        try:
            rec = ns["upload_after_run"]([(f, "results/report.md")])
            assert rec.get("skipped") is True, rec
            assert "False" in rec["why"]
        finally:
            CFG["HF_UPLOAD_AFTER_RUN"] = old

        # dry-run 不联网
        rec = ns["upload_files"]("r/x", [(f, "results/report.md")],
                                 token="hf_t", dry_run=True)
        assert rec["dry_run"] is True and rec["would_upload"][0]["bytes"] == 1
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_hf_missing_hub_refuses_pip_u(ns):
    """缺 huggingface_hub 时给"别乱 -U"的指引，而不是自动装。"""
    patch = _hf_env(ns, token="hf_t")

    def boom():
        raise ns["GuardError"](
            "上传需要 `huggingface_hub`，当前解释器没有这个包。\n"
            "  → **不要** `pip install -U huggingface_hub`：Kaggle 镜像里 "
            "tokenizers 0.23.x 对 hub 有 <2.0 的约束")
    patch("_hf_hub", boom)
    try:
        try:
            ns["upload_files"]("r/x", [(Path("/nope"), "a")], token="hf_t")
            raise AssertionError("必须抛")
        except ns["GuardError"] as e:
            assert "pip install -U" in str(e) and "tokenizers" in str(e), str(e)
    finally:
        patch.undo()


def test_eval_models_local_first(ns):
    """本地有 S4 产物就用本地的 —— 一次 HF 请求都不该发。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_local_models_"))
    patch = _Patch(ns)
    patch("_model_path", lambda fmt: tmp / f"qwen3-8b-{fmt}.gguf")
    patch("repo_files", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("本地已有文件却去列 HF")))
    try:
        for fmt in ("IQ2_XS", "IQ4_XS"):
            (tmp / f"qwen3-8b-{fmt}.gguf").write_bytes(b"q" * 8)
        models = ns["resolve_eval_models"](time.time() + 60)
        assert set(models) == {"IQ2_XS", "IQ4_XS"}
        assert all(m["source"] == "local_s4" for m in models.values()), models
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_models_probe_only_never_downloads(ns):
    """预检只探测：拿到 size/oid 就够，绝不能为了"看一眼"去拉 7 GiB。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_probe_"))
    entries = [{"path": f"models/qwen3-8b-{f}.gguf", "size": 1234 * i,
                "sha256": "e" * 64, "is_lfs": True}
               for i, f in enumerate(("IQ2_XS", "IQ4_XS"), start=1)]
    patch = _hf_env(ns)
    patch("_model_path", lambda fmt: tmp / f"qwen3-8b-{fmt}.gguf")
    patch("repo_files", lambda *a, **k: entries)
    patch("_fetch_pinned_blob", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("probe_only 不许下载")))
    old = ns["CFG"]["HF_FETCH_ARTIFACTS"]
    try:
        ns["CFG"]["HF_FETCH_ARTIFACTS"] = True
        models = ns["resolve_eval_models"](time.time() + 60, probe_only=True)
        assert all(m["source"] == "hf_pending" for m in models.values()), models
        assert models["IQ2_XS"]["bytes"] == 1234
        assert models["IQ4_XS"]["bytes"] == 2468
        assert not (tmp / "qwen3-8b-IQ2_XS.gguf").exists()
    finally:
        ns["CFG"]["HF_FETCH_ARTIFACTS"] = old
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_fairness_from_sidecars_is_declared_not_verified(ns):
    """sidecar 一致性可以核；但"没重算"就必须写 verified=False / 级别=hf_sidecar_declared。"""
    base = {"base_repo": "u/Q", "base_revision": "r1",
            "base_expected_sha256": "f" * 64, "imatrix_sha256": "i" * 64,
            "imatrix_base_kind": "q8_0", "validity_scope": "from_high_precision",
            "params": {"ctx": 512}}
    models = {"IQ2_XS": {"sidecar": dict(base), "sha256_verified": True},
              "IQ4_XS": {"sidecar": dict(base), "sha256_verified": True}}
    fair = ns["fairness_from_sidecars"](models)
    assert fair["verified"] is False, "没重算就不能叫 verified"
    assert fair["verified_level"] == "hf_sidecar_declared"
    assert fair["same_imatrix"] is True and fair["same_base"] is True
    assert fair["imatrix_base_from_approx"] is True, "imatrix base 是 Q8_0，必须标近似"
    assert fair["imatrix_base_consistent"]["ok"] is True
    assert fair["sha256_verified"] is True

    bad = {"IQ2_XS": {"sidecar": dict(base), "sha256_verified": True},
           "IQ4_XS": {"sidecar": {**base, "imatrix_sha256": "j" * 64},
                      "sha256_verified": True}}
    fair2 = ns["fairness_from_sidecars"](bad)
    assert fair2["same_imatrix"] is False
    assert fair2["imatrix_base_consistent"]["ok"] is False

    only_one = {"IQ2_XS": {"sidecar": dict(base)}}
    fair3 = ns["fairness_from_sidecars"](only_one)
    assert fair3["same_imatrix"] is None and fair3["verified"] is False


def test_eval_cost_estimate_blocks_cpu_infeasible(ns):
    """旧的 8×4×2048 在纯 CPU 上估算是 14 小时级 —— 必须在开跑前挡住。"""
    est = ns["eval_cost_estimate"]
    default = est(2, ns["CFG"]["NLL_GROUPS"], ns["CFG"]["NLL_CHUNKS_PER_GROUP"],
                  ns["CFG"]["NLL_CTX"])
    assert default["fits_budget"] is True, default
    assert default["assumed_tok_per_sec"] == 2.5
    assert "不是实测" in default["assumed_rate_note"]

    old = est(2, 8, 4, 2048)
    assert old["fits_budget"] is False, old
    assert old["wall_est_hours"] > 10, old

    gpu = est(2, 8, 4, 2048, allow_gpu=True)
    assert gpu["gpu_allowed"] is True and gpu["wall_est_hours"] < old["wall_est_hours"]


def test_eval_preflight_fails_closed_without_binaries(ns):
    """没有 llama-perplexity 就停，并只给一步（把 RUN_S2 一起打开）。"""
    patch = _Patch(ns)

    def no_bin(name):
        raise ns["GuardError"](f"找不到 {name}")
    patch("llama_bin", no_bin)
    try:
        try:
            ns["eval_preflight"](time.time() + 60)
            raise AssertionError("必须 fail-closed")
        except ns["GuardError"] as e:
            msg = str(e)
            assert "RUN_S2" in msg and "不下载模型" in msg, msg
            assert msg.count("唯一的最简下一步") == 1, msg
    finally:
        patch.undo()


def test_eval_preflight_cpu_only_and_resources(ns):
    """体检要报 RAM/显存，并且默认 gpu_allowed=False（不擅自消耗额度）。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_pf_eval_"))
    eval_txt = tmp / "eval.txt"
    eval_txt.write_text(("a paragraph long enough to be counted as one line " * 2 + "\n\n") * 40,
                        encoding="utf-8")
    calib = tmp / "calib.txt"
    calib.write_text(("another paragraph, definitely different from eval " * 2 + "\n\n") * 40,
                     encoding="utf-8")
    patch = _Patch(ns)
    patch("llama_bin", lambda name: Path("/bin/echo"))
    patch("_model_path", lambda fmt: tmp / f"qwen3-8b-{fmt}.gguf")
    patch("_ensure_eval_txt", lambda d: (eval_txt, {"origin": "stub"}))
    patch("ensure_calib_for_isolation", lambda d: calib)
    patch("vram_gb", lambda: {"total": 15.0, "free": 14.0, "used": 1.0})
    patch("_ram_total_gb", lambda: 24.0)
    try:
        for fmt in ("IQ2_XS", "IQ4_XS"):
            (tmp / f"qwen3-8b-{fmt}.gguf").write_bytes(b"q" * 8)
        rep = ns["eval_preflight"](time.time() + 60)
        assert rep["binaries"] == "ok"
        assert rep["resources"]["gpu_allowed"] is False, "默认不许上 GPU"
        assert rep["resources"]["ram_total_gb"] == 24.0
        assert rep["resources"]["vram_total_gb"] == 15.0
        assert rep["cost"]["gpu_allowed"] is False
        assert rep["calib_corpus"]["available"] is True
    finally:
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_eval_switch_scope(ns):
    """RUN_EVAL=True：只开 S6/S7，绝不带 S2/S3/S4，也不许上 GPU。"""
    CFG, STAGES = ns["CFG"], ns["STAGES"]
    snap = (dict(CFG), dict(STAGES))
    try:
        for k in ("RUN_S2", "RUN_S3", "RUN_S4"):
            CFG[k] = False
        CFG["RUN_EVAL"] = True
        for k in STAGES:
            STAGES[k] = False
        plan = ns["_apply_run_switches"]()
        assert plan["run_eval"] is True, plan
        assert plan["run_s2"] is False and plan["run_s3"] is False
        assert STAGES["S6_NLL"] is True and STAGES["S7_QUALITY"] is True
        assert STAGES["S2_BUILD"] is False and STAGES["S3_IMATRIX"] is False
        assert STAGES["S4_QUANT"] is False, "评测模式绝不能顺带重量化"
        assert STAGES["S4_PREFLIGHT"] is False, "不编译不下载，预检也不必跑"
        assert CFG["HF_FETCH_ARTIFACTS"] is True
        assert CFG["EVAL_ALLOW_GPU"] is False
        assert "hf:" in plan["eval_models_from"]
    finally:
        CFG.clear(); CFG.update(snap[0])
        STAGES.clear(); STAGES.update(snap[1])


def test_run_ppl_cpu_by_default(ns):
    """有 15 GiB 显存也必须 -ngl 0；只有 EVAL_ALLOW_GPU=True 才上 GPU。"""
    CFG = ns["CFG"]
    tmp = Path(tempfile.mkdtemp(prefix="trlab_ppl_"))
    mp = tmp / "m.gguf"
    mp.write_bytes(b"q" * (3 << 20))
    txt = tmp / "t.txt"
    txt.write_text("hello world", encoding="utf-8")
    seen = {}

    def fake_cmd(cmd, log_path, deadline, cwd=None, env=None):
        seen["cmd"] = [str(c) for c in cmd]
        Path(log_path).write_text("final estimate: PPL = 10.0000 +/- 0.1\n",
                                  encoding="utf-8")
        return 0, "ok"

    patch = _Patch(ns)
    patch("llama_bin", lambda name: Path("/bin/echo"))
    patch("run_cmd", fake_cmd)
    patch("vram_gb", lambda: {"total": 15.0, "free": 14.0, "used": 1.0})
    old = CFG["EVAL_ALLOW_GPU"]
    try:
        CFG["EVAL_ALLOW_GPU"] = False
        r = ns["run_ppl"](mp, txt, time.time() + 60, "cpu")
        cmd = seen["cmd"]
        assert cmd[cmd.index("-ngl") + 1] == "0", cmd
        assert r["ngl"] == 0
        assert abs(r["nll"] - 2.302585) < 1e-3

        CFG["EVAL_ALLOW_GPU"] = True
        ns["run_ppl"](mp, txt, time.time() + 60, "gpu")
        cmd = seen["cmd"]
        assert cmd[cmd.index("-ngl") + 1] == "99", cmd
    finally:
        CFG["EVAL_ALLOW_GPU"] = old
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_s6_end_to_end_from_hf(ns):
    """S6 走 HF 输入：记录来源、声明级 fairness、纯 CPU，且不给"已核验"的假象。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_s6_"))
    eval_txt = tmp / "eval.txt"
    eval_txt.write_text(("paragraph about the sea and the wind and the rain " * 2 + "\n\n") * 24,
                        encoding="utf-8")
    calib = tmp / "calib.txt"
    calib.write_text(("totally different calibration sentence number " * 2 + "\n\n") * 24,
                     encoding="utf-8")
    sc = {"base_repo": "u/Q", "base_revision": "r1", "base_expected_sha256": "f" * 64,
          "imatrix_sha256": "i" * 64, "imatrix_base_kind": "q8_0",
          "validity_scope": "from_high_precision", "params": {"ctx": 512}}
    models = {f: {"path": str(tmp / f"qwen3-8b-{f}.gguf"), "bytes": 3 << 20,
                  "sha256": "a" * 64, "source": "hf", "sha256_verified": True,
                  "sidecar_path": None, "sidecar": dict(sc)}
              for f in ("IQ2_XS", "IQ4_XS")}

    patch = _Patch(ns)
    patch("eval_preflight", lambda d, do_download=False: {
        "binaries": "ok", "resources": {"gpu_allowed": False, "ram_total_gb": 24.0},
        "cost": {"fits_budget": True, "wall_est_hours": 0.5}})
    patch("resolve_eval_models", lambda d, formats=None, probe_only=False: models)
    patch("run_ppl", lambda mp, txt, d, tag: {
        "ppl": 10.0 + (0.5 if "IQ2" in str(mp) else 0.0), "ppl_err": 0.1,
        "nll": math.log(10.0 + (0.5 if "IQ2" in str(mp) else 0.0)),
        "ngl": 0, "chunks": 1, "ctx": 512})
    patch("_get", lambda name: {})            # 没有 S4 运行时记录
    patch("resolve_imatrix", lambda: None)
    patch("vram_gb", lambda: {"total": 15.0})
    old = (ns["CFG"]["EVAL_TXT"], ns["CFG"]["CALIB_TXT"], ns["CFG"]["EVAL_SOURCE"],
           ns["CFG"]["NLL_GROUPS"])
    try:
        ns["CFG"]["EVAL_TXT"] = str(eval_txt)
        ns["CFG"]["CALIB_TXT"] = str(calib)
        ns["CFG"]["EVAL_SOURCE"] = "wikitext-2-raw-v1/test"
        ns["CFG"]["NLL_GROUPS"] = 4
        out = ns["stage_s6_nll"](time.time() + 300)
        assert out["fairness_verified"] is False
        assert out["fairness_level"] == "hf_sidecar_declared"
        assert out["isolation"]["ok"] is True, out["isolation"]
        assert out["provenance"]["inputs"]["IQ2_XS"]["source"] == "hf"
        assert out["provenance"]["gpu"] is False
        assert out["provenance"]["self_quantized"] is False, \
            "输入是 HF 上已存的产物，不是本次量化"
        assert out["measured_throughput"]["tok_per_sec"] > 0
        assert out["paired_delta_nll_iq2_minus_iq4"]["n"] == 4
        assert "IQ2" in out["verdict"] or "CI" in out["verdict"]
    finally:
        (ns["CFG"]["EVAL_TXT"], ns["CFG"]["CALIB_TXT"], ns["CFG"]["EVAL_SOURCE"],
         ns["CFG"]["NLL_GROUPS"]) = old
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_corpus_is_test_split(ns):
    """评测语料必须是 test split，且与 calib(train) 是两个不同的文件。"""
    ev, ca = ns["EVAL_PINNED"], ns["CALIB_PINNED"]
    assert ev["split"] == "test" and ca["split"] == "train"
    assert ev["sha256"] != ca["sha256"], "同一个文件不能同时当校准集和评测集"
    assert ev["config"] == ca["config"] == "wikitext-2-raw-v1"
    assert ev["bytes"] == 732610 and ev["revision"] == ca["revision"]


def test_eval_txt_full_flow(ns):
    """`_ensure_eval_txt` 全流程：下载(桩) → 真转换 → stamp → 命中缓存。"""
    tmp = Path(tempfile.mkdtemp(prefix="trlab_eval_flow_"))
    parquet = tmp / "dl" / "test.parquet"
    txt = tmp / "eval.txt"
    stamp = tmp / "eval.stamp.json"
    patch = _Patch(ns)
    patch("_eval_parquet_path", lambda: parquet)
    patch("_eval_txt_path", lambda: txt)
    patch("_eval_stamp_path", lambda: stamp)

    def fake_fetch(pin, dst, deadline, tag):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"PARQUET" * 5)
        return {"origin": "downloaded", "bytes": pin["bytes"], "sha256": pin["sha256"]}

    patch("_fetch_pinned_blob", fake_fetch)
    old = (ns["CFG"]["EVAL_TXT"], ns["CFG"]["EVAL_SOURCE"])
    try:
        p, meta = _with_fake_parquet(lambda: ns["_ensure_eval_txt"](time.time() + 60))
        assert p == txt and txt.exists()
        assert ns["CFG"]["EVAL_TXT"] == str(txt)
        assert ns["CFG"]["EVAL_SOURCE"] == "wikitext-2-raw-v1/test"
        assert meta["bytes"] == ns["EVAL_PINNED"]["bytes"]
        assert meta["converted_bytes"] == txt.stat().st_size
        st = json.loads(stamp.read_text())
        assert st["split"] == "test" and st["converted_bytes"] == txt.stat().st_size

        patch("_fetch_pinned_blob", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("缓存命中还去下载了")))
        _p2, meta2 = ns["_ensure_eval_txt"](time.time() + 60)
        assert meta2["origin"] == "cache"
    finally:
        ns["CFG"]["EVAL_TXT"], ns["CFG"]["EVAL_SOURCE"] = old
        patch.undo()
        shutil.rmtree(tmp, ignore_errors=True)


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
    test_cuda_preflight,
    test_toolchain_report,
    test_s2_cmake_flags,
    test_s2_commit_fallback,
    test_s2_default_off_no_network,
    test_s2_run_switch_only_turns_on_s2,
    test_s2_download_checksum_mismatch,
    test_s2_extract_rejects_traversal,
    test_s2_repeat_run_uses_cache,
    test_s2_dir_name_mismatch_fails,
    test_s2_commit_from_pinned_tarball,
    test_s2_failure_message_actionable,
    test_s2_failure_classifier_cuda_driver,
    test_s2_fingerprint_binds_cuda_no_vmm,
    test_s2_probe_version_reads_stderr,
    test_s2_network_failure_message,
    test_paired_bootstrap_edges,
    test_ggml_type_table,
    test_default_stages_complete,
    test_s9_report_bf16_both_ways,
    test_t4_mock_end_to_end,
    test_a100_mock_end_to_end,
    test_s3_run_switch_scope,
    test_s3_pins_are_registered,
    test_s3_fresh_session_no_binaries,
    test_s3_existing_build_reused,
    test_s3_attached_input_reused,
    test_s3_probe_rejects_unrunnable,
    test_s3_s2_failure_blocks_s3,
    test_s3_restart_after_disconnect,
    test_s3_disk_precheck_insufficient,
    test_s3_download_checksum_mismatch,
    test_s3_download_etag_mismatch_refuses,
    test_s3_download_no_duplicate_full_copy,
    test_s3_download_resume_from_partial,
    test_s3_download_bad_partial_restarts,
    test_s3_download_network_failure_keeps_partial,
    test_s3_repeat_run_uses_cache,
    test_s3_ngl_ladder,
    test_s3_ngl_fallback_on_oom,
    test_s3_imatrix_failure_explains,
    test_s3_q8_deleted_only_after_record,
    test_s3_q8_kept_when_imatrix_fails,
    test_s3_q8_not_deleted_when_configured_off,
    test_s3_manual_base_not_touched,
    test_s3_report_lines,
    test_s4_run_switch_scope,
    test_s4_pins_are_registered,
    test_s4_default_off_downloads_nothing,
    test_verify_imatrix_three_states,
    test_imatrix_found_in_input_and_candidates,
    test_imatrix_provenance_records_sidecar_first,
    test_consistency_provenance_level_ok,
    test_consistency_structure_failures,
    test_consistency_provenance_failures,
    test_consistency_file_level_when_q8_present,
    test_imatrix_base_does_not_throw_when_reclaimed,
    test_resolve_s4_base_manual,
    test_resolve_s4_base_attached_input,
    test_resolve_s4_base_refuses_without_auto_fetch,
    test_resolve_s4_base_scratch_download,
    test_resolve_s4_base_working_and_guard,
    test_scratch_dirs_same_device_excluded,
    test_s4_output_bytes_estimate,
    test_reclaim_stale_q8_cache,
    test_s3_base_origin_regression,
    test_s3_rerun_reclaims_q8,
    test_s3_skips_when_imatrix_verified,
    test_s3_untrusted_imatrix_reruns,
    test_s4_forbids_requantize_even_when_opted_in,
    test_s4_requires_imatrix,
    test_s4_rejects_unverified_imatrix,
    test_s4_base_for_s5,
    test_ram_precheck,
    test_s4_helper_hint,
    test_s4_report_lines_render,
    # v1.0 重构回归
    test_import_has_no_side_effects,
    test_each_module_imports_alone,
    test_switches_off_change_nothing,
    test_switch_s4_expands_plan,
    test_run_returns_plan_and_stages,
    test_overrides_route_stages_vs_cfg,
    test_generated_notebook_is_thin,
    test_bootstrap_fetch_verify_reuse,
    test_bootstrap_embedded_fallback,
    test_bootstrap_embedded_sha_guard,
    # v1.0.1 S4 预检
    test_mount_info_parser,
    test_write_probe_readonly,
    test_s4_preflight_scratch_ok,
    test_s4_preflight_input_bf16,
    test_s4_preflight_fail_closed_no_place,
    test_s4_preflight_working_peak_too_small,
    test_s4_preflight_scratch_rejections,
    test_s4_preflight_config_and_budget,
    test_preflight_only_runs_only_preflight,
    test_s4_quant_budget_is_protective,
    # v1.0.2 S3 bytes 回归
    test_calib_txt_full_flow_no_duplicate_bytes,
    test_no_splat_into_write_stamp,
    # v1.1 HF 输入 / S6-S7 评测
    test_hf_token_chain,
    test_hf_token_never_logged,
    test_hf_tree_parse,
    test_hf_api_errors_are_actionable,
    test_hf_fetch_file_reuse_and_etag_guard,
    test_hf_verify_remote_marks_what_was_checked,
    test_hf_upload_guards_and_optin,
    test_hf_missing_hub_refuses_pip_u,
    test_eval_models_local_first,
    test_eval_models_probe_only_never_downloads,
    test_fairness_from_sidecars_is_declared_not_verified,
    test_eval_cost_estimate_blocks_cpu_infeasible,
    test_eval_preflight_fails_closed_without_binaries,
    test_eval_preflight_cpu_only_and_resources,
    test_run_eval_switch_scope,
    test_run_ppl_cpu_by_default,
    test_s6_end_to_end_from_hf,
    test_eval_corpus_is_test_split,
    test_eval_txt_full_flow,
]


def main():
    print(f"包目录: {HERE / 'trlab'}  "
          f"({sum(f.stat().st_size for f in (HERE / 'trlab').glob('*.py'))} B)\n")

    try:
        ns = load_namespace()
    except Exception:
        print("!! trlab 包无法 import：")
        traceback.print_exc()
        return 1
    print(f"import 完毕，trlab v{ns['NOTEBOOK_VERSION']}\n")

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
