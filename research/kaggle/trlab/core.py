"""跨阶段共享的基础设施：异常、日志、state、守卫、命令执行、指纹、统计。

stage 函数不该自己 print 结论 —— 交给 run_stage() 统一记录状态与 JSON。
"""

from .config import ALWAYS_RERUN, CFG, CKPT, LOGS, MDLS, NOTEBOOK_VERSION, OUT, STAGES, STAGE_REQUIRES_OK, WORK
from .errors import BudgetExceeded, CommandFailed, GuardError
from .paths import _git_commit, llama_bin

import hashlib, json, math, os, re, shutil, subprocess, time, traceback
from datetime import datetime, timezone
from pathlib import Path

def ensure_dirs():
    """建好 working 目录树。import 包不许有副作用，由 run() 显式调用。"""
    for _d in (WORK, OUT, LOGS, CKPT, MDLS):
        _d.mkdir(parents=True, exist_ok=True)



T0 = time.time()



STATE_PATH = CKPT / "state.json"



def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s] {msg}", flush=True)



def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    return path



def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            log("state.json 损坏，重新开始")
    return {}



STATE = load_state()



def save_state():
    write_json(STATE_PATH, STATE)



def _get(stage):
    """取某阶段的 result，只有 status=='ok' 才算数。"""
    r = STATE.get(stage) or {}
    return r.get("result") if r.get("status") == "ok" else None



def vram_gb():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"free": free / 2**30, "total": total / 2**30, "used": (total - free) / 2**30}
    except Exception:
        return None



def disk_gb(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(p)
    return {"total": u.total / 2**30, "used": u.used / 2**30, "free": u.free / 2**30}



def guard(tag):
    """阶段**入口**检查。

    注意语义边界：这里只能拦住"进入时已经不满足"的情况。
    llama.cpp 是外部进程，它在运行中自己分配显存，Python 侧无法阻止它 OOM。
    运行中的保护靠 STAGE_BUDGET_SEC 的墙钟超时 + run_cmd 的 kill。
    """
    el = time.time() - T0
    if el > CFG["MAX_WALL_SEC"]:
        raise BudgetExceeded(
            f"{tag}: 累计墙钟 {el/3600:.2f} h 超过上限 {CFG['MAX_WALL_SEC']/3600:.2f} h")
    v = vram_gb()
    if v and v["used"] > CFG["MAX_VRAM_GB"]:
        raise BudgetExceeded(
            f"{tag}: 进入时显存已占 {v['used']:.2f} GB，超过上限 {CFG['MAX_VRAM_GB']} GB")
    d = disk_gb(WORK)
    if d["free"] < CFG["MIN_FREE_DISK_GB"]:
        raise BudgetExceeded(
            f"{tag}: 磁盘剩余 {d['free']:.2f} GB 低于下限 {CFG['MIN_FREE_DISK_GB']} GB")



def check_deadline(deadline, tag):
    if time.time() > deadline:
        raise BudgetExceeded(f"{tag}: 阶段预算已用尽")



def _est_need_bytes(size_bytes, extra_gb=2.0):
    """权重字节 + extra_gb（KV/context/workspace）→ 总需求字节。

    P0-B：括号必须显式。`size + 1 << 30` 会被解析成 `(size+1) << 30`，
    在 8 GiB 输入上把需求夸大到 8.59e9 GiB。
    """
    return int(size_bytes) + int(round(float(extra_gb) * (1 << 30)))



def _anchor_decision(anchor_bytes, vram_total_gb, base_kind, allow_approx,
                     extra_gb=2.0, margin_gb=1.0):
    """纯函数：给定锚点大小与显存，判断能不能跑、跑出来的结论算什么效力。"""
    usable = max(0.0, float(vram_total_gb) - float(margin_gb))
    required = _est_need_bytes(anchor_bytes, extra_gb) / 2**30
    tier = ANCHOR_TIERS.get(str(base_kind).lower(), "unknown")

    d = {
        "anchor_bytes": int(anchor_bytes),
        "anchor_gb": round(anchor_bytes / 2**30, 3),
        "required_gb": round(required, 3),
        "vram_total_gb": round(float(vram_total_gb), 2),
        "vram_usable_gb": round(usable, 2),
        "fits_on_gpu": required < usable,
        "tier": tier,
        "base_kind": str(base_kind),
    }
    if not d["fits_on_gpu"]:
        d.update(status="skipped",
                 reason=(f"需要 {required:.2f} GiB（权重 {anchor_bytes/2**30:.2f} + "
                         f"余量 {extra_gb} GiB），可用 {usable:.2f} GiB。"
                         "单张 T4 16GB 装不下 8B 高精度锚点。"))
    elif tier != "high" and not allow_approx:
        d.update(status="skipped",
                 reason=(f"锚点精度 = {base_kind}（tier={tier}），属二次量化近似，"
                         "不能用于『相对原始 checkpoint BF16』的有效性结论。"
                         "要跑近似参照请显式设 CFG['ANCHOR_ALLOW_APPROX']=True。"))
    elif tier != "high":
        d.update(status="ready", validity_scope="approx_only",
                 caveat="近似参照：可用来估算 rho 的量级，不可作为有效性证据")
    else:
        d.update(status="ready", validity_scope="high_precision")
    return d



# bf16/f16 才算高精度锚点；Q8 及以下都是二次量化
ANCHOR_TIERS = {
    "bf16": "high", "f16": "high", "fp16": "high",
    "q8_0": "approx", "q6_k": "approx", "q5_k": "approx",
    "q4_k_m": "approx", "q4_k": "approx", "q4_0": "approx",
}



def _bench_summary(tb):
    """把 S1 的 torch 微基准压成 S9 能安全渲染的字段。

    P0-C：绝不假设 bf16 一定拿到了 TFLOP/s。T4（CC 7.5）上 bf16 GEMM 可能直接报错。
    """
    tb = tb or {}
    fp = tb.get("fp16") or {}
    bf = tb.get("bf16") or {}
    cp = tb.get("copy") or {}
    out = {
        "torch_error": tb.get("error"),
        "cuda": tb.get("cuda"),
        "fp16_tflops": fp.get("tflops"),
        "bf16_tflops": bf.get("tflops"),
        "bf16_error": bf.get("error"),
        "bf16_available": bf.get("tflops") is not None,
        "fp16_available": fp.get("tflops") is not None,
        "ratio_bf16_over_fp16": None,
        "copy_gb_per_s": cp.get("gb_per_s_derived"),
        "copy_error": cp.get("error"),
    }
    if out["fp16_tflops"] and out["bf16_tflops"]:
        out["ratio_bf16_over_fp16"] = round(out["bf16_tflops"] / out["fp16_tflops"], 3)
    return out



# 已量化类型：从这些 base 再量化需要 --allow-requantize
QUANTIZED_KINDS = {
    "q8_0", "q8_1", "q6_k", "q5_k", "q5_0", "q5_1", "q4_k", "q4_k_m",
    "q4_0", "q4_1", "q3_k", "q2_k", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs",
    "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs", "tq1_0", "tq2_0", "mxfp4",
}



def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None



def _retire_product(dst, tag=None):
    """指纹不符的旧产物：改名成 .stale-<tag>，并删掉更早的同类 stale。

    不直接覆盖 —— 新量化失败时旧文件还能对照；也不无限堆积，
    每种格式最多留一份 stale（两个 GiB 级文件不会把 working 撑爆）。
    """
    dst = Path(dst)
    if not dst.exists():
        return None
    if dst.stat().st_size == 0:
        dst.unlink()
        return None
    for old in sorted(dst.parent.glob(dst.name + ".stale-*")):
        old.unlink()
    stale = dst.with_name(f"{dst.name}.stale-{(tag or 'old')[:8]}")
    dst.rename(stale)
    return stale



def _rho_from_nll(anchor_nll, q4_nll, q2_nll):
    """rho = dNLL(Q2,Q4) / dNLL(Q4,anchor)，全部按「后减前」的正向定义。

    v0.2 写成 `anchor - Q4`，符号是反的：比值因为分子分母同号碰巧正确，
    但打印出来的 `dNLL(Q4,anchor)` 与代数定义相反。这里统一成 Q4 - anchor。
    """
    d = {"anchor_nll": anchor_nll, "q4_nll": q4_nll, "q2_nll": q2_nll}
    if anchor_nll is None or q4_nll is None or q2_nll is None:
        d.update(d4=None, d2=None, rho=None, reason="有 NLL 缺失")
        return d
    d4 = q4_nll - anchor_nll          # 锚点 → Q4 的退化（应为正）
    d2 = q2_nll - anchor_nll          # 锚点 → Q2 的退化（应比 d4 更大）
    d.update(d4=d4, d2=d2)
    if d4 <= 1e-9:
        d.update(rho=None,
                 reason=(f"dNLL(Q4,anchor)={d4:+.4f} ≤ 0：锚点没有比 Q4 更好，"
                         "rho 无定义（锚点无效，或 Q4 与锚点不可区分）"))
        return d
    d.update(rho=(d2 - d4) / d4, reason=None)
    return d



def need(key):
    v = CFG.get(key, "")
    if not v:
        raise GuardError(f"CFG['{key}'] 未设置")
    p = Path(v)
    if not p.exists():
        raise GuardError(f"CFG['{key}'] 指向的路径不存在: {p}")
    return p



def _path_fp(p):
    """文件：size + mtime + 首尾各 1 MiB 哈希。目录：CMakeLists + git commit。"""
    if p is None:
        return None
    p = Path(p)
    if not p.exists():
        return None
    if p.is_dir():
        d = {"kind": "dir"}
        cml = p / "CMakeLists.txt"
        if cml.exists():
            st = cml.stat()
            d.update(cmake_size=st.st_size, cmake_mtime=st.st_mtime_ns)
        d["git"] = _git_commit(p)
        return d
    st = p.stat()
    rec = {"kind": "file", "size": st.st_size, "mtime": st.st_mtime_ns}
    try:
        with open(p, "rb") as f:
            rec["head"] = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]
            if st.st_size > (2 << 20):
                f.seek(-(1 << 20), os.SEEK_END)
                rec["tail"] = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]
    except OSError as e:
        rec["read_error"] = repr(e)
    return rec



def _fingerprint(name):
    """阶段输入指纹。任何依赖变化都会改变它，从而让旧结果失效。"""
    # STAGE_DEPS 的解析器要调用 paths/imatrix/stages_build，那些模块又依赖
    # core；在函数体内延迟导入打破循环（顶层导入会 ImportError）。
    from .deps import STAGE_DEPS
    cfg_keys, file_res = STAGE_DEPS.get(name, ([], []))
    payload = {
        "nb": NOTEBOOK_VERSION,
        "stage": name,
        "cfg": {k: CFG.get(k) for k in cfg_keys},
        "files": {},
    }
    for i, r in enumerate(file_res):
        try:
            p = r()
        except Exception:
            p = None
        payload["files"][f"f{i}"] = _path_fp(p)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]



def run_stage(name, fn):
    """开关 + 前置依赖 + 指纹断点 + 预算 + JSON 落盘的统一入口。"""
    if not STAGES.get(name, False):
        log(f"[{name}] SKIP —— STAGES['{name}'] 为 False")
        return None

    # v0.4 / D-6：前置阶段真的失败过，就不再启动本阶段。
    # 只有"前置被运行过且失败"才拦；前置根本没开（False）时不拦 ——
    # 那种情况是用户显式手动配置（比如自带 LLAMA_BIN），应当放行让它自己报缺什么。
    for dep in STAGE_REQUIRES_OK.get(name, []):
        d = STATE.get(dep) or {}
        if d.get("status") == "error":
            log(f"[{name}] SKIP —— 前置阶段 {dep} 失败：{str(d.get('error'))[:220]}")
            log(f"          先解决 {dep}（日志见 {LOGS}/ 与 {OUT}/{dep}.json），再重跑本阶段。")
            return None

    fp = _fingerprint(name)
    prev = STATE.get(name) or {}
    always = name in ALWAYS_RERUN

    if always:
        log(f"[{name}] 强制重跑（在 ALWAYS_RERUN 里，汇总不能停留在旧结果）")
    elif prev.get("status") == "ok" and prev.get("fingerprint") == fp and not CFG["FORCE_RERUN"]:
        log(f"[{name}] RESUME —— 指纹一致 {fp}，产物 {prev.get('result_path', '')}")
        return prev.get("result")
    elif prev.get("status") == "ok" and not CFG["FORCE_RERUN"]:
        log(f"[{name}] INVALIDATE —— 输入指纹变化 "
            f"{prev.get('fingerprint')} → {fp}，重跑该阶段")

    budget = CFG["STAGE_BUDGET_SEC"].get(name, 600)
    log(f"[{name}] START（预算 {budget} s）")
    t = time.time()
    rec = {"stage": name,
           "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fingerprint": fp, "notebook_version": NOTEBOOK_VERSION}
    try:
        guard(name)
        result = fn(t + budget)
        rec.update(status="ok", elapsed_sec=round(time.time() - t, 2),
                   result=result, result_path=str(OUT / f"{name}.json"))
        write_json(OUT / f"{name}.json", {"stage": name, "meta": rec, "result": result})
        log(f"[{name}] OK  {rec['elapsed_sec']} s  →  {rec['result_path']}")
    except Exception as e:
        rec.update(status="error", elapsed_sec=round(time.time() - t, 2),
                   error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-2000:])
        write_json(OUT / f"{name}.json", {"stage": name, "meta": rec, "result": None})
        log(f"[{name}] FAIL  {rec['error']}")

    STATE[name] = rec
    save_state()
    return rec.get("result")



def run_cmd(cmd, log_path, deadline, cwd=None, env=None):
    """跑外部命令。rc != 0 一律抛 CommandFailed（P1-E：不再静默忽略返回码）。"""
    cmd = [str(c) for c in cmd]
    log(f"  $ {' '.join(cmd)}")
    t = time.time()
    with open(log_path, "wb") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        while p.poll() is None:
            if time.time() > deadline:
                p.kill()
                raise BudgetExceeded(f"命令超预算被杀: {cmd[0]}")
            time.sleep(0.5)
    rc = p.returncode
    tail = Path(log_path).read_text(errors="replace")[-1800:]
    log(f"  rc={rc}  {time.time() - t:.1f} s  →  {log_path}")
    if rc != 0:
        raise CommandFailed(cmd, rc, tail, log_path)
    return rc, tail



def atomic_replace(src, dst):
    """先写 .partial，验完再原子改名。避免把截断/旧文件当成功产物。"""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise GuardError(f"atomic_replace: 源文件不存在 {src}")
    os.replace(src, dst)
    return dst



def paired_bootstrap(diffs, n=None, seed=None):
    """配对差值的 bootstrap 均值 CI。n<2 时不给 CI，而不是崩。"""
    import random
    n = n or CFG["BOOTSTRAP_N"]
    seed = CFG["SEED"] if seed is None else seed
    m = len(diffs)
    if m == 0:
        return {"n": 0, "mean": None, "ci95": None}
    mean = sum(diffs) / m
    if m < 2:
        return {"n": m, "mean": mean, "ci95": None, "note": "只有 1 对，无法给 CI"}
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            s += diffs[rng.randrange(m)]
        means.append(s / m)
    means.sort()
    return {"n": m, "mean": mean,
            "ci95": [means[int(0.025 * n)], means[int(0.975 * n)]],
            "boot": n}



PPL_RE = re.compile(r"final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", re.I)



def _fit_ngl(nbytes, vram_usable_gb, margin_gb=2.0):
    """权重能不能进显存。返回 llama.cpp 的 -ngl 值（99 = 全卸载，0 = 纯 CPU）。

    v0.4 起默认余量 2.0 GiB：imatrix 运行时要放 KV cache、logits 和
    compute buffer，1 GiB 在 8B 上不够稳。调用方一般显式传
    CFG['NGL_MARGIN_GB']，这里只是兜底默认值。
    """
    if not vram_usable_gb:
        return 0
    need = nbytes / 2**30 + margin_gb
    return 99 if need < vram_usable_gb else 0



def split_eval(path, n_groups, out_dir):
    """把评测文本切成**恰好** n_groups 份不相交的连续子文件。

    P1-F：旧版用 ceil(len/n)，41 段 8 组时只会得到 7 组。
    现在整除 + 余数分配，组数严格等于 min(n_groups, 段数)，且总段数不变。
    """
    txt = Path(path).read_text(errors="replace")
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    if len(paras) < n_groups:
        paras = [p for p in txt.split("\n") if p.strip()]
    if not paras:
        raise ValueError(f"评测语料为空: {path}")

    n = max(1, min(int(n_groups), len(paras)))
    base, rem = divmod(len(paras), n)
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    assert sum(sizes) == len(paras) and len(sizes) == n

    groups, i = [], 0
    for s in sizes:
        groups.append(paras[i:i + s])
        i += s

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, meta = [], []
    for gi, g in enumerate(groups):
        fp = out_dir / f"eval_g{gi:03d}.txt"
        body = "\n\n".join(g)
        fp.write_text(body, encoding="utf-8")
        paths.append(fp)
        meta.append({"group": gi, "paras": len(g), "chars": len(body)})
    return paths, meta



def run_ppl(model_path, text_path, deadline, tag):
    """跑 llama-perplexity，解析 Final estimate。返回 NLL = ln(PPL)。

    run_cmd 已经在 rc != 0 时抛 CommandFailed，所以能走到解析这一步
    说明进程正常退出（P1-E）。
    """
    v = vram_gb()
    usable = (v["total"] - 1.0) if v else 0.0
    ngl = _fit_ngl(Path(model_path).stat().st_size, usable)
    cmd = [llama_bin("llama-perplexity"),
           "-m", model_path, "-f", text_path,
           "-c", CFG["NLL_CTX"],
           "--chunks", CFG["NLL_CHUNKS_PER_GROUP"],
           "-ngl", ngl, "-t", os.cpu_count() or 4,
           "-b", "512", "--seed", CFG["SEED"]]
    logf = LOGS / f"ppl_{tag}.log"
    run_cmd(cmd, logf, deadline)

    full = logf.read_text(errors="replace")
    m = PPL_RE.search(full)
    if not m:
        raise RuntimeError(f"{tag}: rc=0 但解析不到 PPL。日志尾部:\n{full[-1200:]}")
    ppl = float(m.group(1))
    return {"ppl": ppl, "ppl_err": float(m.group(2)), "nll": math.log(ppl),
            "ngl": ngl, "chunks": CFG["NLL_CHUNKS_PER_GROUP"], "ctx": CFG["NLL_CTX"]}
