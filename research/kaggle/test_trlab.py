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

import json
import os
import shutil
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


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
    assert out == {"run_s2": False, "run_s3": False, "run_s4": False}, out
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

    def fake(url, path, offset, hasher, deadline, tag, expect_bytes):
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
        def fake(url, path, offset, hasher, deadline, tag, expect_bytes):
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

        def spy(url, path, offset, hasher, deadline, tag, expect_bytes):
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

        def boom(url, path, offset, hasher, deadline, tag, expect_bytes):
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
