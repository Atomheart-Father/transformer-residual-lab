"""S0 环境检查 + S1 短时 smoke test。"""

from .config import CFG, IN, NOTEBOOK_VERSION, ON_KAGGLE, WORK
from .errors import UnknownGGMLType
from .gguf import ggml_type_info, gguf_scan, gguf_write_synth
from .core import _bench_summary, _est_need_bytes, check_deadline, disk_gb, split_eval

import json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

def stage_s0_env(deadline):
    info = {}
    info["notebook_version"] = NOTEBOOK_VERSION
    info["where"] = "kaggle" if ON_KAGGLE else "local"
    info["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["cpu_count"] = os.cpu_count()

    # ---- RAM ----
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = {"total_gb": round(vm.total / 2**30, 2), "available_gb": round(vm.available / 2**30, 2)}
    except Exception:
        try:
            ram = {"total_gb": round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 2),
                   "available_gb": None}
        except Exception as e:
            ram = {"error": repr(e)}
    info["ram"] = ram

    # ---- GPU ----
    gpus = []
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "idx": i, "name": p.name, "cc": f"{p.major}.{p.minor}",
                "total_gb": round(total / 2**30, 2),
                "free_gb": round(free / 2**30, 2),
                "sm_count": p.multi_processor_count,
                "native_bf16": p.major >= 8,   # CC<8.0 没有原生 bf16（T4 = 7.5）
            })
    except Exception as e:
        info["torch_error"] = repr(e)
    info["gpus"] = gpus
    info["vram_usable_gb_est"] = (round(max(g["total_gb"] for g in gpus) - 1.0, 2)
                                  if gpus else 0.0)

    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"(unavailable: {e})"

    # ---- 磁盘 ----
    disks = {}
    for p in ["/kaggle/working", "/kaggle/temp", "/kaggle/input", str(WORK)]:
        try:
            if Path(p).exists():
                disks[p] = {k: round(v, 2) for k, v in disk_gb(p).items()}
        except Exception as e:
            disks[p] = {"error": repr(e)}
    info["disks_gb"] = disks

    # ---- 工具链 ----
    tools = {}
    for b in ["cmake", "g++", "gcc", "nvcc", "git", "make",
              "llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli"]:
        tools[b] = shutil.which(b)
    info["tools"] = tools

    # ---- Python 包 ----
    pkgs = {}
    for name in ["torch", "numpy", "psutil", "gguf", "transformers", "datasets"]:
        try:
            import importlib.metadata as md
            pkgs[name] = md.version(name)
        except Exception:
            pkgs[name] = None
    info["packages"] = pkgs

    # ---- /kaggle/input 清单（这是你填 CFG 路径的依据）----
    inv = []
    if IN.is_dir():
        for d in sorted(IN.iterdir()):
            if not d.is_dir():
                continue
            n_files, tot, ggufs = 0, 0, []
            try:
                for j, p in enumerate(d.rglob("*")):
                    if j % 200 == 0:
                        check_deadline(deadline, "S0_ENV 扫描 /kaggle/input")
                    if p.is_file():
                        n_files += 1
                        try:
                            sz = p.stat().st_size
                        except OSError:
                            continue
                        tot += sz
                        if p.suffix.lower() == ".gguf":
                            ggufs.append({"path": str(p), "gb": round(sz / 2**30, 3)})
            except Exception as e:
                inv.append({"dataset": d.name, "error": repr(e)})
                continue
            inv.append({"dataset": d.name, "files": n_files,
                        "gb": round(tot / 2**30, 2), "gguf": ggufs[:20],
                        "gguf_count": len(ggufs)})
    info["input_datasets"] = inv

    # ---- 结论性提示 ----
    warn = []
    if not gpus:
        warn.append("没有可用 GPU。S6/S7 的 llama.cpp 会退到 CPU，速度慢几十倍。")
    elif any(not g["native_bf16"] for g in gpus):
        warn.append("GPU 不支持原生 bf16（CC<8.0）。高精度锚点只能用 f16；"
                    "bf16 矩阵乘可能直接报错 —— S1 会实测，S9 不会假设它可用。")
    if info["vram_usable_gb_est"] and info["vram_usable_gb_est"] < 17.3:
        warn.append(f"可用显存约 {info['vram_usable_gb_est']:.1f} GiB，"
                    "装不下 8B f16（15.25 GiB 权重 + KV）。S8 会拒绝执行。")
    if not any(x.get("gguf_count") for x in inv if isinstance(x, dict)):
        # v1.1.3：RUN_EVAL 的输入来自 HF 私有仓库，这条老提示会自相矛盾。
        if CFG.get("RUN_EVAL"):
            warn.append("/kaggle/input 下没有 .gguf —— **本次不需要**：`RUN_EVAL=True` "
                        "的 IQ2_XS / IQ4_XS 从 HF 私有仓库取（凭据门会先验一次）。")
        else:
            warn.append("/kaggle/input 下没有找到 .gguf。S3-S4 需要你手动挂载 base GGUF"
                        "（S5-S7 的输入来自 S4 产物或 HF）。")
    if tools["cmake"] is None:
        warn.append("没有 cmake，S2_BUILD 无法执行。")
    info["warnings"] = warn

    # ---- 打印 ----
    print("=" * 78)
    print(f"Python {info['python']} | CPU {info['cpu_count']} | RAM {ram}")
    for g in gpus:
        print(f"GPU{g['idx']}: {g['name']}  CC{g['cc']}  {g['total_gb']} GiB "
              f"(free {g['free_gb']})  SM={g['sm_count']}  native_bf16={g['native_bf16']}")
    print(f"可用显存估计: {info['vram_usable_gb_est']} GiB")
    print("-" * 78)
    for k, v in disks.items():
        print(f"disk {k:22s} total={v.get('total')} GiB  free={v.get('free')} GiB")
    print("-" * 78)
    print("tools:", {k: (v.split("/")[-1] if v else None) for k, v in tools.items()})
    print("pkgs :", pkgs)
    print("-" * 78)
    if inv:
        print("/kaggle/input:")
        for d in inv:
            if "error" in d:
                print(f"  {d['dataset']}: ERROR {d['error']}")
            else:
                print(f"  {d['dataset']}: {d['files']} files, {d['gb']} GiB, {d['gguf_count']} gguf")
                for g in d["gguf"]:
                    print(f"      {g['gb']:8.2f} GiB  {g['path']}")
    else:
        print("/kaggle/input 为空")
    print("=" * 78)
    for w in warn:
        print("!! " + w)

    return info



def _torch_microbench(deadline):
    """fp16 vs bf16 矩阵乘 + 拷贝带宽。

    T4（CC 7.5）上 bf16 GEMM 可能直接报错，这里只如实记录，
    不做任何"应该能拿到 TFLOP/s"的假设。
    """
    out = {}
    try:
        import torch
    except Exception as e:
        return {"error": f"torch 不可用: {e}"}
    if not torch.cuda.is_available():
        return {"cuda": False, "note": "无 GPU，跳过"}

    N = 2048
    flops = 2.0 * N**3
    for dt, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        try:
            check_deadline(deadline, "S1 矩阵乘")
            a = torch.randn(N, N, device="cuda", dtype=dt)
            b = torch.randn(N, N, device="cuda", dtype=dt)
            for _ in range(2):
                c = a @ b
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                c = a @ b
            torch.cuda.synchronize()
            el = (time.time() - t) / 5
            out[label] = {"ms": round(el * 1000, 3), "tflops": round(flops / el / 1e12, 2)}
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            out[label] = {"error": f"{type(e).__name__}: {e}"}
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    try:
        nbytes = 256 << 20
        x = torch.empty(nbytes // 4, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        for _ in range(2):
            y.copy_(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            y.copy_(x)
        torch.cuda.synchronize()
        el = (time.time() - t) / 5
        out["copy"] = {"gb_moved": round(2 * nbytes / 2**30, 3),
                       "ms": round(el * 1000, 2),
                       "gb_per_s_derived": round(2 * nbytes / el / 1e9, 1),
                       "note": "read+write 的派生值，不是 DRAM 计数器实测"}
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        out["copy"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        free, total = torch.cuda.mem_get_info()
        out["vram_after_gb"] = {"free": round(free / 2**30, 2), "total": round(total / 2**30, 2)}
    except Exception:
        pass
    return out



def _disk_bench(deadline, mb=256):
    p = WORK / "smoke_disk.bin"
    chunk = b"\x00" * (1 << 20)
    t = time.time()
    try:
        with open(p, "wb") as f:
            for _ in range(mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        el = time.time() - t
        return {"mb": mb, "sec": round(el, 3), "mb_per_s": round(mb / el, 1)}
    finally:
        if p.exists():
            p.unlink()



def stage_s1_smoke(deadline):
    res = {"checks": {}, "bench": {}}

    # ---- 1. 合成 GGUF 往返 ----
    specs = [
        ("token_embd.weight",     (256, 128), 1),   # F16
        ("blk.0.ffn_gate.weight", (256, 256), 17),  # IQ2_XS
        ("blk.0.ffn_up.weight",   (256, 256), 23),  # IQ4_XS
        ("blk.0.attn_q.weight",   (256, 256), 12),  # Q4_K
        ("output.weight",         (128, 256), 0),   # F32
    ]
    sp = WORK / "smoke_synth.gguf"
    gguf_write_synth(sp, specs)
    scan = gguf_scan(sp)

    expected = {}
    for name, dims, tid in specs:
        n = 1
        for d in dims:
            n *= d
        _, blk, tsz = ggml_type_info(tid)
        expected[name] = ((n + blk - 1) // blk) * tsz

    got = {t["name"]: t["nbytes"] for t in scan["tensors"]}
    res["checks"]["gguf_roundtrip_bytes_match"] = (got == expected)
    res["checks"]["gguf_expected"] = expected
    res["checks"]["gguf_got"] = got

    def bpw_of(name):
        t = [x for x in scan["tensors"] if x["name"] == name][0]
        return t["nbytes"] * 8 / t["n_elements"]

    res["checks"]["bpw_iq2_xs"] = round(bpw_of("blk.0.ffn_gate.weight"), 6)
    res["checks"]["bpw_iq4_xs"] = round(bpw_of("blk.0.ffn_up.weight"), 6)
    res["checks"]["bpw_iq2_xs_ok"] = abs(res["checks"]["bpw_iq2_xs"] - 2.3125) < 1e-6
    res["checks"]["bpw_iq4_xs_ok"] = abs(res["checks"]["bpw_iq4_xs"] - 4.25) < 1e-6

    # ---- 2. P1-G：未知 type 必须 fail closed ----
    bad = WORK / "smoke_unknown.gguf"
    gguf_write_synth(bad, [("blk.0.bogus.weight", (256, 256), 200)], strict_types=False)
    try:
        try:
            gguf_scan(bad)
            res["checks"]["unknown_type_fails_closed"] = False
            res["checks"]["unknown_type_error"] = "(没有抛异常，这是缺陷)"
        except UnknownGGMLType as e:
            res["checks"]["unknown_type_fails_closed"] = True
            res["checks"]["unknown_type_error"] = str(e)[:80]
        except Exception as e:
            res["checks"]["unknown_type_fails_closed"] = False
            res["checks"]["unknown_type_error"] = f"{type(e).__name__}: {e}"

        # 显式降级时：不产出字节，也不产出 bpw（0.0 bpw 同样是假数）
        try:
            sc_bad = gguf_scan(bad, strict_types=False)
            res["checks"]["unknown_type_degrades_to_none"] = (
                sc_bad["unknown_types"] == ["UNKNOWN"]
                and sc_bad["tensors"][0]["nbytes"] is None
                and sc_bad["bpw_effective"] is None)
            res["checks"]["unknown_type_degrades_detail"] = {
                "unknown_types": sc_bad["unknown_types"],
                "nbytes": sc_bad["tensors"][0]["nbytes"],
                "bpw_effective": sc_bad["bpw_effective"]}
        except Exception as e:
            res["checks"]["unknown_type_degrades_to_none"] = False
            res["checks"]["unknown_type_degrades_detail"] = f"{type(e).__name__}: {e}"
    finally:
        if bad.exists():
            bad.unlink()

    # ---- 3. P0-B：需求字节算术自检（防止优先级回归）----
    arith = {
        "8GiB+1GiB": _est_need_bytes(8 << 30, 1.0) == (9 << 30),
        "16GiB+2GiB": _est_need_bytes(16 << 30, 2.0) == (18 << 30),
        "8GiB+2GiB": _est_need_bytes(8 << 30, 2.0) == (10 << 30),
    }
    res["checks"]["need_bytes_arithmetic_ok"] = all(arith.values())
    res["checks"]["need_bytes_detail"] = arith
    res["checks"]["need_bytes_8gib_plus_2"] = _est_need_bytes(8 << 30, 2.0) / 2**30

    # ---- 4. P1-F：分组组数自检 ----
    grp_ok = True
    grp_detail = {}
    for n_paras, n_groups in [(5, 8), (7, 3), (8, 8), (9, 8), (40, 8), (41, 8), (100, 16)]:
        tmp = WORK / f"smoke_split_{n_paras}_{n_groups}"
        txt = "\n\n".join(f"para{i} " + ("w " * 40) for i in range(n_paras))
        tp = WORK / "smoke_split.txt"
        tp.write_text(txt)
        try:
            paths, meta = split_eval(tp, n_groups, tmp)
            exp = min(n_groups, n_paras)
            body = "".join(p.read_text() for p in paths)
            total = sum(m["paras"] for m in meta)
            ok = (len(paths) == exp and total == n_paras)
            grp_detail[f"{n_paras}p/{n_groups}g"] = f"{len(paths)}（期望 {exp}）"
            grp_ok = grp_ok and ok
        finally:
            if tp.exists():
                tp.unlink()
    res["checks"]["split_eval_exact_groups"] = grp_ok
    res["checks"]["split_eval_detail"] = grp_detail

    # ---- 5. P0-C：bf16 缺失时 _bench_summary 不能崩 ----
    try:
        s_ok = _bench_summary({"fp16": {"tflops": 50.0},
                               "bf16": {"error": "RuntimeError: not implemented for BFloat16"}})
        res["checks"]["bench_summary_bf16_error_ok"] = (
            s_ok["bf16_available"] is False
            and s_ok["ratio_bf16_over_fp16"] is None
            and "not implemented" in (s_ok["bf16_error"] or ""))
        s_no = _bench_summary({})
        res["checks"]["bench_summary_empty_ok"] = s_no["fp16_tflops"] is None
    except Exception as e:
        res["checks"]["bench_summary_bf16_error_ok"] = False
        res["checks"]["bench_summary_error"] = f"{type(e).__name__}: {e}"

    # ---- 6. GPU 微基准 ----
    tb = _torch_microbench(deadline)
    res["bench"]["torch"] = tb
    res["bench"]["torch_summary"] = _bench_summary(tb)
    res["bench"]["disk"] = _disk_bench(deadline)

    # ---- 7. 预算换算 ----
    res["budget_math"] = {
        "weights_iq2_xs_gib_lower_bound": round(CFG["PARAM_COUNT"] * 2.3125 / 8 / 2**30, 3),
        "weights_iq4_xs_gib_lower_bound": round(CFG["PARAM_COUNT"] * 4.25 / 8 / 2**30, 3),
        "weights_f16_gib": round(CFG["PARAM_COUNT"] * 2 / 2**30, 3),
        "note": "GiB 下界，不含 embed/head 的高精度保留；真实字节由 S5 从文件读",
        "nll_tokens_planned": CFG["NLL_GROUPS"] * CFG["NLL_CHUNKS_PER_GROUP"] * CFG["NLL_CTX"],
    }

    ok = all(v for k, v in res["checks"].items() if isinstance(v, bool))
    res["smoke_ok"] = ok

    print("=" * 78)
    print("smoke checks:")
    for k, v in res["checks"].items():
        print(f"  {k:34s} {v}")
    print("bench:", json.dumps(res["bench"], indent=1, ensure_ascii=False, default=str))
    print("budget:", json.dumps(res["budget_math"], indent=1, ensure_ascii=False))
    print(f"SMOKE_OK = {ok}")
    print("=" * 78)

    if not ok:
        raise RuntimeError("smoke 自检失败，先修解析器/算术再用大文件")
    return res
