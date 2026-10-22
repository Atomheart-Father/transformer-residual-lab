"""S4：量化（BF16 → IQ2_XS / IQ4_XS）+ S5：真实字节统计。"""

from .config import BUILD, CFG, FORMATS, IMATRIX_PINNED, LOGS, MDLS, MODEL_DL, NOTEBOOK_VERSION, S3_PINNED, S4_PINNED, WORK
from .errors import BudgetExceeded, CommandFailed, GuardError, UnknownGGMLType
from .paths import _git_commit, _llama_src_path, _model_path, _q8_stamp_path, _s3_base_path, llama_bin
from .gguf import _ggml_type_id, ggml_type_info, gguf_scan
from .core import QUANTIZED_KINDS, _get, _path_fp, _read_json, _retire_product, atomic_replace, check_deadline, log, run_cmd, write_json
from .fetch import _disk_precheck, _fetch_pinned_blob, partial_path, sha256_of
from .imatrix import _imatrix_base, _imatrix_base_consistency, _imatrix_base_kind, _imatrix_provenance_records, _scan_input_gguf, _verify_imatrix, resolve_imatrix

import hashlib, json, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path

def _base_requant_plan(base_kind, allow_requantize):
    """从给定 base 量化到 IQ2_XS/IQ4_XS 需要什么，结果算什么效力。

    上游事实（src/llama-quant.cpp 的 `llama_model_quantize_impl` 主循环）：
        if (ggml_is_quantized(tensor->type) && !params->allow_requantize)
            throw std::runtime_error("requantizing from type Q8_0 is disabled");
    `allow_requantize` 默认 false。tools/quantize/quantize.cpp 的帮助文本原文：
        "--allow-requantize   allow requantizing tensors that have already been quantized
         WARNING: this can severely reduce quality compared to quantizing from 16bit or 32bit!"
    """
    k = str(base_kind or "").strip().lower()
    is_q = k in QUANTIZED_KINDS
    d = {"base_kind": k, "base_is_quantized": is_q,
         "needs_allow_requantize": is_q, "allow_requantize": bool(allow_requantize),
         "flags": [], "validity_scope": None, "provenance_tag": None}

    if not k:
        d.update(status="blocked",
                 reason="CFG['BASE_KIND'] 未设置。必须写实际类型（f16 / bf16 / q8_0 …），"
                        "否则无法判断要不要 --allow-requantize。")
        return d
    if not is_q:
        d.update(status="ok", validity_scope="from_high_precision",
                 provenance_tag="primary")
        return d
    if not allow_requantize:
        d.update(status="blocked", reason=(
            f"BASE_KIND={k} 已经是量化类型。llama-quantize 遇到已量化的张量会直接抛 "
            f"`requantizing from type ... is disabled`，除非显式传 --allow-requantize；"
            f"而官方帮助对那个开关的原文是 "
            f"\"WARNING: this can severely reduce quality compared to quantizing from "
            f"16bit or 32bit!\"。正确做法：BASE_GGUF 指向同一 checkpoint 的 f16/bf16"
            f"（llama-quantize 是 CPU 路径，不吃显存），用 IMATRIX_BASE 指向这个 {k} 去算 imatrix。"
            f"只做 smoke 时，显式设 CFG['ALLOW_REQUANTIZE']=True，"
            f"产物会被标 requantized_pilot_only。"))
        return d
    d.update(status="ok", flags=["--allow-requantize"],
             validity_scope="requantized_pilot_only",
             provenance_tag="requantized_pilot_only",
             caveat=("二次量化：IQ2/IQ4 之间仍然同源可比，但不能作为"
                     "『相对原始 checkpoint BF16』的高保真低比特结论。"))
    return d



def _quant_cmd(qbin, imatrix, base, out_path, fmt, threads, plan):
    """llama-quantize 的 argv。**选项必须全部在位置参数之前。**

    上游 tools/quantize/quantize.cpp 的参数循环是
        for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++)
    遇到第一个不以 `--` 开头的 token 就退出，之后一律当位置参数读
    （model / output / ftype / nthreads）。所以把 --allow-requantize 追加在末尾
    等于没传：它会被当成多余的位置参数丢掉，requantize 仍然被拒绝。
    """
    flags = [str(f) for f in ((plan or {}).get("flags") or [])]
    return ([str(qbin)] + flags
            + ["--imatrix", str(imatrix),
               str(base), str(out_path), str(fmt), str(threads)])



def _quant_fp(base_fp, imatrix_sha256, llama_commit, fmt, plan, params):
    """量化产物指纹：决定字节的东西全在里，notebook 版本只记录不参与哈希。

    只有指纹完全一致时才允许复用旧 GGUF —— 否则「换了 base 或 imatrix，
    S4 重跑了，但文件还是旧的」这种误报会一路传到 S6 的配对结论里。
    """
    payload = {
        "schema": 1,
        "base": base_fp,
        "imatrix_sha256": imatrix_sha256,
        "llama_commit": llama_commit,
        "fmt": str(fmt),
        "requant": (plan or {}).get("provenance_tag"),
        "params": params,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]



def _s4_output_bytes():
    """S4 两个产物预计占用的字节（用于磁盘账）。

    只按"权重块大小 × 参数量"估，再加 15% 覆盖 GGUF 头、对齐填充和保留高精度
    的 embedding / output / norm 张量。**估高不估低** —— 磁盘账保守一点，
    真到 S4 跑完 S5 会给出精确值。
    """
    total, unknown = 0, []
    for fmt in FORMATS:
        tid = _ggml_type_id(fmt)
        info = ggml_type_info(tid) if tid is not None else None
        if not info or info[1] is None:
            unknown.append(fmt)
            continue
        _name, blk, tsz = info
        total += int(CFG["PARAM_COUNT"] / blk * tsz * 1.15)
    if unknown:
        log(f"!! 不知道 {unknown} 的块尺寸，产物体积按 6.5 GiB 兜底估算")
    return total or int(6.5 * 2**30)



def _scratch_candidates():
    """working 之外可能存在的暂存目录（Kaggle 官方只说有、没给容量）。"""
    return [Path("/kaggle/tmp"), Path("/tmp"), Path("/kaggle"), Path("/")]



def _scratch_dirs(need_bytes, margin_gb=2.0):
    """找 working 之外、且**真的是另一个文件系统**的暂存盘。

    只认 st_dev 与 working 不同的候选 —— 同一个设备上的"另一个目录"
    不提供额外空间，算进来会导致虚假的磁盘账。容量未知也如实记下来。
    """
    out = []
    try:
        wdev = os.stat(WORK).st_dev
    except OSError:
        wdev = None
    for c in _scratch_candidates():
        try:
            if not c.is_dir() or not os.access(c, os.W_OK):
                continue
            st = os.stat(c)
            if wdev is not None and st.st_dev == wdev:
                continue
            du = shutil.disk_usage(c)
            if du.free >= need_bytes + margin_gb * 2**30:
                out.append({"path": str(c), "free_gb": round(du.free / 2**30, 2),
                            "total_gb": round(du.total / 2**30, 2),
                            "st_dev": st.st_dev})
        except OSError:
            continue
    return out



def _provenance_path(dst):
    """产物旁的 provenance 文件。文件名带 .gguf，所以不能用 with_suffix 替换。"""
    dst = Path(dst)
    return dst.with_name(dst.name + ".provenance.json")



def _s4_helper_notebook_hint():
    return (
        "  → 一次性操作（之后每个 Session 都零点击）：\n"
        "    1. 导入随本 notebook 一起交付的 `trlab_fetch_bf16.ipynb`（3 个 cell，只做下载）\n"
        "    2. 右侧 Settings → Internet = On → Run All（约 15–30 分钟，取决于 Kaggle 出口带宽）\n"
        "    3. Save Version（Quick Save 即可）\n"
        "    4. 回到本 notebook → 右侧 Add Input → 选那个 notebook 的输出\n"
        "    5. 重跑本 notebook。BF16 变成只读 Input，之后永久复用、不再下载。\n"
        "  → 或者：把 CFG['BASE_GGUF'] 手动指到一个已挂载的 BF16 路径上。")



def _resolve_s4_base(deadline):
    """解析量化 base。返回 (path, origin, meta)。

    origin ∈ manual / attached_input / scratch_download / working_download
    """
    # 1) 手动指定
    if CFG["BASE_GGUF"]:
        p = Path(CFG["BASE_GGUF"])
        if not p.exists():
            raise GuardError(f"CFG['BASE_GGUF'] 指向的路径不存在: {p}")
        log(f"量化 base：手动指定 {p}（{p.stat().st_size / 2**30:.2f} GiB）")
        return p, "manual", {"bytes": p.stat().st_size}

    pin = S4_PINNED
    outputs = _s4_output_bytes()

    # 2) /kaggle/input 已挂载 —— 零点击，优先
    if not CFG.get("S4_AUTO_FETCH_BASE"):
        raise GuardError(
            "CFG['BASE_GGUF'] 未设置，且 S4_AUTO_FETCH_BASE=False（不自动下载 BF16）。\n"
            "  → 要么把 CFG['BASE_GGUF'] 指到已挂载的 BF16；\n"
            "  → 要么把 RUN_S4 改成 True（会自动挑最省事的取数路径）。")
    for hit in _scan_input_gguf(max_hits=32, time_budget_sec=25.0):
        try:
            if hit.stat().st_size != pin["bytes"]:
                continue
        except OSError:
            continue
        log(f"量化 base：在 /kaggle/input 找到 {hit.name}"
            f"（{pin['bytes'] / 2**30:.2f} GiB，字节数与 pin 一致）")
        meta = {"bytes": pin["bytes"], "path": str(hit),
                "note": "只读 Input；字节数与 S4_PINNED 一致"}
        if CFG.get("HASH_GGUF"):
            d = sha256_of(hit, deadline)
            if d != pin["sha256"]:
                raise GuardError(
                    f"/kaggle/input 里的 {hit.name} 字节数对但 sha256 不符：\n"
                    f"  实际 {d}\n  预期 {pin['sha256']}\n"
                    "这不是 pinned revision 的那个文件，拒绝使用。")
            meta["sha256"] = d
            meta["sha256_checked"] = True
        else:
            meta["sha256"] = None
            meta["sha256_checked"] = False
        return hit, "attached_input", meta

    # 3) working 之外的暂存盘（运行时探测；官方没说容量，所以只认"另一个文件系统"）
    scratch = _scratch_dirs(pin["bytes"], margin_gb=CFG.get("S4_RAM_MARGIN_GB", 4.0))
    if scratch:
        s = scratch[0]
        dst = Path(s["path"]) / pin["filename"]
        log(f"量化 base：working 之外有暂存盘 {s['path']}"
            f"（空闲 {s['free_gb']} GiB / 共 {s['total_gb']} GiB，另一个文件系统）")
        log("  → 下到暂存盘：不占 working 的 20 GB，一次 Run All 就能跑完 S4+S5")
        log("  → 代价：暂存盘不跨 Session 保存，下次开会话要重下")
        _disk_precheck(pin["bytes"], f"S4 下载 {pin['filename']}",
                       reserve_gb=CFG.get("S4_RAM_MARGIN_GB", 4.0), root=Path(s["path"]))
        meta = _fetch_pinned_blob(pin, dst, deadline, tag=f"S4/{pin['filename']}")
        return dst, "scratch_download", meta

    # 4) working 自己 —— 只有磁盘账真的过得去才下
    free = shutil.disk_usage(WORK).free
    need = pin["bytes"] + outputs
    if free >= need + CFG["MIN_FREE_DISK_GB"] * 2**30:
        log(f"量化 base：working 装得下（空闲 {free / 2**30:.2f} GiB ≥ "
            f"BF16 {pin['bytes'] / 2**30:.2f} + 产物 {outputs / 2**30:.2f} GiB）→ 下载")
        _disk_precheck(pin["bytes"], f"S4 下载 {pin['filename']}")
        dst = MODEL_DL / pin["filename"]
        meta = _fetch_pinned_blob(pin, dst, deadline, tag=f"S4/{pin['filename']}")
        return dst, "working_download", meta

    raise GuardError(
        f"BF16（{pin['bytes'] / 2**30:.2f} GiB）放不进 /kaggle/working"
        f"（空闲 {free / 2**30:.2f} GiB，需要 BF16 + 两个产物 ≈ {need / 2**30:.2f} GiB），"
        "也没找到 working 之外够大的暂存盘。\n"
        f"{_s4_helper_notebook_hint()}")



def _reclaim_stale_q8_cache(deadline):
    """S4 开跑前，回收"已验证过、但 S3 已经用完了"的 Q8_0 缓存。

    v0.4 真实运行里 Q8_0 因为 origin 误记成 manual 而没被回收，8.11 GiB 一直
    占着 working。S4 需要空间下 BF16，所以这里补一道：只要 imatrix 已经拿到手
    且验证通过，这个 Q8_0 就不再是"可恢复状态"，而是纯粹的占盘。
    """
    if not CFG.get("S4_RECLAIM_Q8_CACHE"):
        return None
    q8 = _s3_base_path()
    if not q8.exists():
        return None
    imat = resolve_imatrix()
    if imat is None:
        log("!! working 里还有 Q8_0，但找不到 imatrix —— 不删（它还是可恢复状态）")
        return None
    v = _verify_imatrix(imat, deadline)
    if not (v.get("ok") and v.get("trust") in ("pinned_verified", "provenance_verified")):
        log("!! working 里还有 Q8_0，但 imatrix 未验证 —— 不删（它还是可恢复状态）")
        return None
    freed = q8.stat().st_size
    q8.unlink()
    _q8_stamp = _q8_stamp_path()
    Path(_q8_stamp).unlink(missing_ok=True)
    for extra in (q8.with_name(q8.name + ".partial"),
                  q8.with_name(q8.name + ".partial.json")):
        extra.unlink(missing_ok=True)
    log(f"已回收上一轮遗留的 Q8_0：释放 {freed / 2**30:.2f} GiB"
        f"（imatrix 已验证，不再需要它）")
    return {"freed_bytes": freed, "path": str(q8), "reason": "imatrix 已验证，S4 需要空间"}



def _ram_precheck(base_bytes, tag):
    """量化前的 RAM 提示。llama-quantize 是 CPU 路径，输入走 mmap（页缓存可回收），
    但 `--max-buffer-size` 默认 8192 MiB 是实打实要占的。不够会 exit 137。"""
    try:
        import psutil  # noqa
        have = psutil.virtual_memory().total
    except Exception:
        try:
            have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            have = None
    if not have:
        log(f"{tag}: 读不到 RAM 总量，跳过预检")
        return None
    need = base_bytes + CFG.get("S4_RAM_MARGIN_GB", 4.0) * 2**30
    if have < need:
        log(f"!! {tag}: RAM {have / 2**30:.1f} GiB 可能不够"
            f"（base {base_bytes / 2**30:.2f} GiB + 余量 "
            f"{CFG.get('S4_RAM_MARGIN_GB', 4.0)} GiB ≈ {need / 2**30:.1f} GiB）")
        log("  → 若 llama-quantize 报 exit 137，把 CFG['S4_RAM_MARGIN_GB'] 调小，"
            "或在 runbook 6.3 里用 --max-buffer-size 降压")
    else:
        log(f"{tag}: RAM {have / 2**30:.1f} GiB ≥ 需要 {need / 2**30:.1f} GiB")
    return {"ram_gb": round(have / 2**30, 2), "need_gb": round(need / 2**30, 2)}



# ---------------------------------------------------------------- S4 预检（v1.0.1）

def _mount_info(path, mounts_path="/proc/mounts"):
    """从挂载表找 path 的归属挂载点（Kaggle/Linux 有；本地 macOS 没有 → None）。

    为什么不能只信 st_dev：st_dev 不同是"另一个文件系统"的**必要线索**，
    不是充分证明 —— 只读挂载、tmpfs（吃 RAM）、配额都要另看。这里补上
    设备名 / fstype / 挂载选项三项，让"工作区外存得下"有据可查。
    `mounts_path` 可注入，方便离线测试。
    """
    p = Path(path)
    try:
        p = p.resolve()
    except OSError:
        pass
    best = None
    try:
        with open(mounts_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        dev, mp, fstype, opts = parts[0], parts[1], parts[2], parts[3]
        mp_path = Path(mp.replace("\\040", " "))
        try:
            under = (p == mp_path) or (mp_path in p.parents)
        except OSError:
            continue
        if under and (best is None or len(mp) > len(best["mount"])):
            best = {"device": dev, "mount": mp, "fstype": fstype,
                    "opts": [o for o in opts.split(",") if o]}
    return best



def _write_probe(d):
    """1 字节写入探针：建得起来、写得进去、删得掉，立刻删。

    `os.access(W_OK)` 会说谎（只读挂载、SELinux、quota 满都可能过不了真写），
    所以用真写。只写 1 字节，不留痕迹。
    """
    try:
        probe = Path(d) / ".trlab_write_probe"
        probe.write_bytes(b"x")
        probe.unlink()
        return True, None
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"



def _preflight_scratch_candidates():
    """预检要看的暂存盘候选。单独一个函数是为了测试能注入假候选。"""
    cands, seen = [], set()
    for c in (Path("/kaggle/tmp"), Path("/tmp"), Path("/var/tmp")):
        if str(c) not in seen:
            seen.add(str(c)); cands.append(c)
    try:   # 挂载表里 /kaggle 下（除 working）的其它挂载点也补进来
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                mp = Path(parts[1].replace("\\040", " "))
                if str(mp).startswith("/kaggle") and str(mp) != str(WORK) \
                        and str(mp) not in seen:
                    seen.add(str(mp)); cands.append(mp)
    except OSError:
        pass
    return cands



def _preflight_scratch_evidence(need_bytes):
    """枚举 working 之外的暂存盘候选，逐个记录"能不能用"的证据。

    判据与 `_scratch_dirs()`（下载时真用的那个函数）保持一致：
    存在 / st_dev 与 WORK 不同 / 可写 / 空闲 ≥ need + 余量。
    另外多做两件 `_scratch_dirs` 不做的事：1 字节写入探针（比 os.access 诚实）、
    读挂载表补设备名/fstype（tmpfs 直接排除 —— 那是 RAM，15 GiB 放进去必爆）。
    """
    out = []
    try:
        wdev = os.stat(WORK).st_dev
    except OSError:
        wdev = None
    wmount = _mount_info(WORK)
    margin = float(CFG.get("S4_RAM_MARGIN_GB", 4.0))

    for c in _preflight_scratch_candidates():
        rec = {"path": str(c)}
        try:
            if not c.is_dir():
                rec.update(exists=False, usable=False, why="不存在")
                out.append(rec); continue
            st = os.stat(c)
        except OSError as e:
            rec.update(exists=False, usable=False, why=f"stat 失败: {e}")
            out.append(rec); continue
        rec["exists"] = True
        rec["st_dev"] = st.st_dev
        rec["work_st_dev"] = wdev
        mi = _mount_info(c)
        rec["mount"] = mi
        if wmount is None:
            rec["mount_note"] = "挂载表不可读（非 Linux？）——st_dev 判定仍有效，fstype/配额未核"
        if wdev is not None and st.st_dev == wdev:
            rec.update(usable=False, same_device=True,
                       why="与 working 同一个文件系统（st_dev 相同），不提供额外空间")
            out.append(rec); continue
        rec["same_device"] = False
        if mi and mi.get("fstype") in ("tmpfs", "ramfs", "devtmpfs"):
            rec.update(usable=False,
                       why=f"{mi.get('fstype')} 是内存盘，15 GiB 放进去会挤占 RAM")
            out.append(rec); continue
        if mi and wmount and mi.get("device") == wmount.get("device"):
            rec.update(usable=False,
                       why=f"挂载表显示与 working 同一设备 {mi.get('device')}")
            out.append(rec); continue
        ok_w, why_w = _write_probe(c)
        rec["writable"] = ok_w
        if not ok_w:
            rec.update(usable=False, why=f"写不进去: {why_w}")
            out.append(rec); continue
        try:
            du = shutil.disk_usage(c)
        except OSError as e:
            rec.update(usable=False, why=f"读不到容量: {e}")
            out.append(rec); continue
        rec["free_gb"] = round(du.free / 2**30, 2)
        rec["total_gb"] = round(du.total / 2**30, 2)
        rec["need_gb"] = round((need_bytes + margin * 2**30) / 2**30, 2)
        if du.free < need_bytes + margin * 2**30:
            rec.update(usable=False,
                       why=f"空闲 {rec['free_gb']} GiB < 需要 {rec['need_gb']} GiB")
            out.append(rec); continue
        rec["usable"] = True
        out.append(rec)
    return out



def _s4_preflight_imatrix_state():
    """imatrix 现在能不能直接用；不能的话 S3 会重下 8.11 GiB + 重算。"""
    imat = resolve_imatrix()
    if imat is None:
        return {"state": "missing", "path": None, "trust": None,
                "note": "working 里没有 imatrix —— S3 会照常运行："
                        "下 8.11 GiB Q8_0 + 校准语料，重算后校验（约 10 分钟，估算）"}
    v = _verify_imatrix(imat)
    trust = v.get("trust")
    if v.get("ok") and trust in ("pinned_verified", "provenance_verified") \
            and CFG.get("S3_SKIP_IF_IMATRIX_OK"):
        return {"state": "reusable", "path": str(imat), "trust": trust,
                "sha256": v.get("sha256"), "bytes": v.get("bytes"),
                "note": "sha256 与 pin/provenance 对得上 —— S3 整段跳过，"
                        "不重下 8.11 GiB、不重算"}
    if v.get("ok"):
        return {"state": "unverified", "path": str(imat), "trust": trust,
                "sha256": v.get("sha256"),
                "note": "有文件但对不上任何已知记录 —— S3 会重算（或显式放行，"
                        "见 S4 的 IMATRIX_ALLOW_UNPINNED）"}
    return {"state": "unusable", "path": str(imat), "trust": trust,
            "why": v.get("why"),
            "note": "文件在但读不通/不可信 —— S3 会重算"}



def stage_s4_preflight(deadline):
    """S4 前的只读预检（v1.0.1）。在任何编译/下载/量化**之前**跑，几秒钟。

    要证明两件事，缺一件就 fail-closed，并且只给一个最简下一步：
      1. BF16（15.26 GiB）在工作区外存得下，或者已经挂在 /kaggle/input 里；
      2. 两个产物（≈6.7 GiB）在 /kaggle/working 存得下。
    同时如实记录：working/暂存盘的挂载标识、容量、写权限、imatrix 能否复用
    （不能复用就注明 S3 会重下 8.11 GiB + 重算）。

    这一层存在的理由：不要等 28 分钟编译 + 8 GiB 下载跑完，才在下载 BF16 时
    发现 21.96 GiB 放不进 19.07 GiB。
    """
    pin = S4_PINNED
    out_bytes = _s4_output_bytes()
    out_gb = out_bytes / 2**30
    bf16_gb = pin["bytes"] / 2**30
    reserve_gb = float(CFG["MIN_FREE_DISK_GB"])
    build_gb = float(CFG.get("S2_BUILD_DIR_EST_GB", 3.0))

    # ---- working 本身 ----
    ok_w, why_w = _write_probe(WORK)
    wdu = shutil.disk_usage(WORK)
    work = {"path": str(WORK), "writable": ok_w,
            "free_gb": round(wdu.free / 2**30, 2),
            "total_gb": round(wdu.total / 2**30, 2),
            "mount": _mount_info(WORK)}
    if not ok_w:
        raise GuardError(f"working 写不进去（{why_w}）—— 什么都跑不了。")
    log(f"working：空闲 {work['free_gb']} GiB / 共 {work['total_gb']} GiB，可写")

    # ---- 暂存盘（工作区外）----
    scratch = _preflight_scratch_evidence(pin["bytes"])
    usable_scratch = [s for s in scratch if s.get("usable")]
    for s in scratch:
        if s.get("usable"):
            log(f"暂存盘 {s['path']}：可用（空闲 {s['free_gb']} GiB，"
                f"另一个文件系统 st_dev={s['st_dev']}，可写）")
        elif s.get("exists"):
            log(f"暂存盘 {s['path']}：不可用 —— {s.get('why')}")

    # ---- imatrix ----
    imat = _s4_preflight_imatrix_state()
    s3_will_run = imat["state"] != "reusable"
    log(f"imatrix：{imat['state']} —— {imat['note']}")

    # ---- BF16 从哪来 ----
    bf16 = {"bytes": pin["bytes"], "gb": round(bf16_gb, 2),
            "sha256_checked": False, "path": None, "source": None}
    manual = CFG.get("BASE_GGUF")
    if manual:
        p = Path(manual)
        if not p.exists():
            raise GuardError(f"CFG['BASE_GGUF'] 指向的路径不存在: {p}")
        bf16.update(source="manual", path=str(p),
                    note=f"手动指定；{p.stat().st_size / 2**30:.2f} GiB")
        if p.stat().st_size != pin["bytes"]:
            log(f"!! 手动 base 字节数与 S4_PINNED 不符"
                f"（{p.stat().st_size} vs {pin['bytes']}）—— S4 会按文件实际内容判类型")
    else:
        hit = None
        for c in _scan_input_gguf(max_hits=32, time_budget_sec=20.0):
            try:
                if c.stat().st_size == pin["bytes"]:
                    hit = c
                    break
            except OSError:
                continue
        if hit is not None:
            bf16.update(source="attached_input", path=str(hit),
                        note="已在 /kaggle/input（只读）；字节数与 pin 一致，"
                             "sha256 由 S4 核对")
        elif usable_scratch:
            s = usable_scratch[0]
            bf16.update(source="scratch_download", path=str(Path(s["path"]) / pin["filename"]),
                        note=f"下到工作区外暂存盘 {s['path']}（空闲 {s['free_gb']} GiB）；"
                             "代价：不跨 Session，下次要重下")
        else:
            need_all = pin["bytes"] + out_bytes + reserve_gb * 2**30
            if wdu.free >= need_all:
                bf16.update(source="working_download",
                            path=str(MODEL_DL / pin["filename"]),
                            note="working 装得下 BF16 + 两个产物")
            else:
                bf16.update(source=None,
                            note=f"既不在 /kaggle/input，也没有可用的工作区外暂存盘，"
                                 f"working 也装不下（空闲 {work['free_gb']} GiB < "
                                 f"BF16 {bf16_gb:.2f} + 产物 {out_gb:.2f} + 保底 "
                                 f"{reserve_gb:.0f} GiB）")
    log(f"BF16 来源：{bf16['source'] or '无'} —— {bf16.get('note', '')}")

    # ---- working 峰值的磁盘账 ----
    # S2 的 build 目录一直在（S3/S4 还要用）；Q8_0 与产物不同时存在
    # （imatrix 到手后 Q8_0 即回收），所以峰值取二者较大值。
    q8_gb = 0.0 if not s3_will_run else (S3_PINNED["bytes"] / 2**30)
    peak_gb = build_gb + max(q8_gb, out_gb) + reserve_gb
    if bf16["source"] == "working_download":
        peak_gb += bf16_gb
    fits_work = wdu.free / 2**30 >= peak_gb
    outputs_fit = wdu.free / 2**30 >= build_gb + out_gb + reserve_gb
    log(f"working 峰值账：build≈{build_gb:.1f} + max(Q8_0 {q8_gb:.2f}, "
        f"产物 {out_gb:.2f}) + 保底 {reserve_gb:.0f}"
        + (f" + BF16 {bf16_gb:.2f}" if bf16["source"] == "working_download" else "")
        + f" = {peak_gb:.2f} GiB vs 空闲 {work['free_gb']} GiB → "
        + ("够" if fits_work else "不够"))

    # ---- RAM ----
    ram = _ram_precheck(pin["bytes"], "S4 预检")

    # ---- 判定 ----
    if bf16["source"] is None:
        raise GuardError(
            f"S4 预检不过：BF16（{bf16_gb:.2f} GiB）没有可用的落点。\n"
            f"  · /kaggle/input 里没找到字节数为 {pin['bytes']} 的 .gguf；\n"
            f"  · working 之外没有「另一个文件系统且空闲 ≥ {bf16_gb:.1f} GiB」的暂存盘；\n"
            f"  · working 自己也装不下（空闲 {work['free_gb']} GiB）。\n"
            f"  → 唯一的最简下一步：{_s4_helper_notebook_hint()}")
    if not fits_work:
        raise GuardError(
            f"S4 预检不过：BF16 有地方放（{bf16['source']}），但 working 的峰值账不够。\n"
            f"  空闲 {work['free_gb']} GiB < 峰值 {peak_gb:.2f} GiB"
            f"（build≈{build_gb:.1f} + max(Q8_0 {q8_gb:.2f}, 产物 {out_gb:.2f}）"
            f" + 保底 {reserve_gb:.0f}）。\n"
            f"  → 唯一的最简下一步：清掉 working 里的旧产物"
            f"（{MDLS} 下的 .stale-* / 上一轮 GGUF）后重跑；"
            f"确知自己在做什么，再考虑调小 CFG['MIN_FREE_DISK_GB']。")

    return {"ok": True,
            "work": work,
            "scratch": scratch,
            "bf16": bf16,
            "outputs": {"bytes": out_bytes, "gb": round(out_gb, 2),
                        "fits_in_work": outputs_fit},
            "work_peak_gb": round(peak_gb, 2),
            "work_free_gb": work["free_gb"],
            "imatrix": imat,
            "ram": ram,
            "s3_will_run": s3_will_run,
            "pinned": {"repo": pin["repo"], "revision": pin["revision"],
                       "filename": pin["filename"], "sha256": pin["sha256"]}}



def stage_s4_quant(deadline):
    """S4：从**原始 F16/BF16** 量化出 IQ2_XS 与 IQ4_XS，共用同一份 imatrix。

    v0.5 起这条路线不再要求用户在网页里准备任何东西：
      · 量化 base 由 `_resolve_s4_base()` 按优先级自动取（Input > 暂存盘 > working）
      · imatrix 由 S3 产出或被复用（sha256 与 pin 校验），Q8_0 不在也能核验同源性
      · 禁止二次量化：`ALLOW_REQUANTIZE` 保持 False，base 必须是 f16/bf16
    """
    # ---- 1. imatrix：必须存在且可信 ----
    imat = resolve_imatrix()
    if imat is None:
        raise GuardError(
            "找不到 imatrix。IQ2_XS 没有 imatrix 会被 llama-quantize 直接 bail out，"
            "所以这里不做降级：先跑 S3，或把 CFG['IMATRIX_GGUF'] 指向已有文件。")
    imat_v = _verify_imatrix(imat, deadline)
    if not imat_v.get("ok"):
        raise GuardError(f"imatrix 不可用: {imat_v.get('why')}")
    if imat_v.get("trust") == "unverified" and not CFG.get("IMATRIX_ALLOW_UNPINNED"):
        raise GuardError(
            f"imatrix 的 sha256 与任何已知记录都对不上（{imat_v.get('why')}）。\n"
            f"  pin sha256 = {IMATRIX_PINNED['sha256']}\n"
            f"  实际 sha256 = {imat_v.get('sha256')}\n"
            "  换 imatrix 会让 S4 的结论指向另一套统计量，必须显式承认：\n"
            "  → 确认这是你有意为之，就设 CFG['IMATRIX_ALLOW_UNPINNED']=True；\n"
            "  → 或者把 IMATRIX_GGUF 指回那份已验证的文件。")
    for w in imat_v.get("warnings") or []:
        log(f"!! imatrix {w}")
    log(f"imatrix：{imat}（{imat_v.get('trust')}，{imat_v.get('bytes')} B，"
        f"{imat_v.get('n_tensors')} 个张量，sha256 {str(imat_v.get('sha256'))[:16]}…）")

    # ---- 2. 量化 base：按优先级自动取（v0.5）----
    _reclaim_stale_q8_cache(deadline)
    base, base_origin, base_meta = _resolve_s4_base(deadline)
    if CFG["BASE_KIND"] and str(CFG["BASE_KIND"]).lower() != str(S4_PINNED["kind"]).lower():
        log(f"!! CFG['BASE_KIND']={CFG['BASE_KIND']} 与 S4_PINNED['kind']="
            f"{S4_PINNED['kind']} 不一致；以 CFG 为准，请确认文件真实类型")

    # ---- P0-1：已量化的 base 必须显式 opt-in（否则上游直接抛）----
    # v0.5：本路线**禁止**二次量化。base 是从 BF16 来的，一旦 BASE_KIND 是量化
    # 类型就说明取错了文件，直接拦死，不给 --allow-requantize 的活路。
    plan = _base_requant_plan(CFG["BASE_KIND"], CFG["ALLOW_REQUANTIZE"])
    if plan.get("status") == "blocked":
        raise GuardError(plan["reason"])
    if plan.get("flags"):
        raise GuardError(
            f"BASE_KIND={CFG['BASE_KIND']} 是量化类型，而本路线禁止二次量化"
            f"（ALLOW_REQUANTIZE 必须保持 False）。量化 base 必须是 S4_PINNED 的 "
            f"{S4_PINNED['filename']}（{S4_PINNED['kind']}）。")

    # ---- 3. imatrix 与量化 base 是否同源（v0.5：不要求 Q8_0 还在）----
    s3r = _get("S3_IMATRIX") or {}
    if not isinstance(s3r, dict):
        s3r = {}
    prov_recs = _imatrix_provenance_records()
    s3_prov = None
    for r in prov_recs:
        if r.get("sha256") and r["sha256"] == imat_v.get("sha256"):
            s3_prov = r
            break
    if s3_prov is None and s3r:
        s3_prov = dict(s3r)
        s3_prov["_source"] = "OUT/S3_IMATRIX.json"

    imat_base = _imatrix_base()
    if imat_base and Path(imat_base).exists() and str(Path(imat_base)) == str(base):
        consistency = {"ok": True, "level": "file", "why": "同一个文件",
                       "n_tensors": imat_v.get("n_tensors"),
                       "architecture": gguf_scan(base, deadline).get("architecture")}
    else:
        consistency = _imatrix_base_consistency(imat, base, s3_prov, deadline)
        if consistency.get("ok") is False:
            raise GuardError(
                f"imatrix 与量化 base 无法证明同源（{consistency.get('level')} 级）："
                f"{consistency.get('why')}。imatrix 是逐张量、逐输入通道的统计量，"
                "对不上就没有意义，硬量化只会得到一个说不清来源的产物。")
        log(f"imatrix 与量化 base 同源核验通过（{consistency.get('level')} 级）："
            f"{consistency.get('n_tensors')} 个张量，arch="
            f"{consistency.get('architecture')}，"
            f"{consistency.get('verified_by') or consistency.get('why') or ''}")

    # ---- 4. 磁盘 / RAM 预检 ----
    out_bytes = _s4_output_bytes()
    free = shutil.disk_usage(MDLS).free
    if free < out_bytes + CFG["MIN_FREE_DISK_GB"] * 2**30:
        raise GuardError(
            f"working 放不下两个产物：空闲 {free / 2**30:.2f} GiB，"
            f"需要约 {out_bytes / 2**30:.2f} GiB + 保底 "
            f"{CFG['MIN_FREE_DISK_GB']} GiB。\n"
            "  → 先清 working（例如上一轮遗留的 Q8_0 / .stale-* 产物）；\n"
            "  → 或把 CFG['MIN_FREE_DISK_GB'] 调小（不推荐）。")
    _ram_precheck(base.stat().st_size, "S4 量化")

    qbin = llama_bin("llama-quantize")
    threads = os.cpu_count() or 4
    base_fp = _path_fp(base)
    im_sha = imat_v.get("sha256")
    llm_commit = (_git_commit(_llama_src_path()) or _git_commit(BUILD)
                  or (CFG["LLAMA_COMMIT"] or None))

    # 决定字节的参数都进指纹（线程数不影响结果，不放）
    params = {"imatrix_ctx": CFG["IMATRIX_CTX"], "imatrix_chunks": CFG["IMATRIX_CHUNKS"],
              "imatrix_sha256": im_sha}

    # v1.0.1：单格式预算帽。总预算从 1800 s 提到 5400 s（30 分钟连做两次 8B
    # CPU 量化，误杀风险太高），再给单个格式加帽，避免第一个吃光第二个的时间。
    # **保护上限，不是耗时承诺** —— 真实耗时未实测。
    per_fmt = float(CFG.get("S4_PER_FORMAT_BUDGET_SEC", 3600))
    total_budget = float(CFG["STAGE_BUDGET_SEC"].get("S4_QUANT", 5400))

    out = {}
    for fmt in FORMATS:
        check_deadline(deadline, f"S4_QUANT {fmt}")
        fmt_deadline = min(deadline, time.time() + per_fmt)
        dst = _model_path(fmt)
        prov_path = _provenance_path(dst)
        fp = _quant_fp(base_fp, im_sha, llm_commit, fmt, plan, params)
        prov = _read_json(prov_path)

        # P0-2：只有 provenance 完全匹配才复用。文件存在 ≠ 文件是对的。
        reuse = (dst.exists() and dst.stat().st_size > 0
                 and isinstance(prov, dict) and prov.get("quant_fp") == fp
                 and not CFG["FORCE_RERUN"])

        if reuse:
            log(f"{fmt} 复用：产物指纹 {fp} 与当前 base/imatrix/commit 一致")
        else:
            if dst.exists():
                why = ("CFG['FORCE_RERUN']=True" if CFG["FORCE_RERUN"] else
                       ("没有 provenance 记录，无法证明它来自当前输入"
                        if not isinstance(prov, dict) else
                        f"指纹变化 {(prov or {}).get('quant_fp')} → {fp}"))
                retired = _retire_product(dst, tag=(prov or {}).get("quant_fp") or fp)
                log(f"{fmt} 不复用（{why}）；旧产物 → {retired or '已删除（0 字节）'}")

            part = partial_path(dst)
            if part.exists():
                part.unlink()

            cmd = _quant_cmd(qbin, imat, base, part, fmt, threads, plan)
            try:
                run_cmd(cmd, LOGS / f"s4_quant_{fmt}.log", fmt_deadline)
            except BudgetExceeded as e:
                done = [f for f in FORMATS
                        if _model_path(f).exists() and _model_path(f).stat().st_size > 0]
                raise BudgetExceeded(
                    f"{fmt} 量化超出单格式预算 {per_fmt:.0f} s"
                    f"（阶段总预算 {total_budget:.0f} s）。\n"
                    f"  日志保留在 {LOGS / f's4_quant_{fmt}.log'}，可从尾部看进度。\n"
                    f"  已完成的格式：{done or '无'} —— 重跑会按 provenance 指纹自动复用，不会重量。\n"
                    f"  这是保护上限，不是耗时承诺：真实 CPU 量化耗时未实测。要放宽就调 "
                    f"CFG['S4_PER_FORMAT_BUDGET_SEC'] 或 STAGE_BUDGET_SEC['S4_QUANT']。"
                ) from e
            except CommandFailed as e:
                hint = ""
                low = (e.tail or "").lower()
                if "requantizing" in low:
                    hint = ("  ← 上游拒绝二次量化。BASE_KIND 是量化类型时必须显式设 "
                            "CFG['ALLOW_REQUANTIZE']=True，或改用 f16/bf16 的 BASE_GGUF。")
                elif "imatrix" in low or "garbage" in low or "importance matrix" in low:
                    hint = "  ← 这是 imatrix 缺失/无效触发的 bail out"
                raise RuntimeError(f"{fmt} 量化失败（rc={e.rc}）。{hint}") from e

            if not part.exists() or part.stat().st_size == 0:
                raise RuntimeError(f"{fmt} 量化后没有产物: {part}")
            atomic_replace(part, dst)

            write_json(prov_path, {
                "schema": 2,
                "fmt": fmt,
                "quant_fp": fp,
                "notebook": NOTEBOOK_VERSION,
                "base": str(base), "base_kind": CFG["BASE_KIND"], "base_fp": base_fp,
                "base_origin": base_origin, "base_download": base_meta,
                "base_repo": S4_PINNED["repo"], "base_revision": S4_PINNED["revision"],
                "base_expected_sha256": S4_PINNED["sha256"],
                "imatrix": str(imat), "imatrix_sha256": im_sha,
                "imatrix_trust": imat_v.get("trust"),
                "imatrix_base": str(imat_base), "imatrix_base_kind": _imatrix_base_kind(),
                "imatrix_base_present": bool(imat_base and Path(imat_base).exists()),
                "imatrix_base_consistent": consistency,
                "llama_commit": llm_commit,
                "requant": plan.get("provenance_tag"),
                "validity_scope": plan.get("validity_scope"),
                "params": params,
                "cmd": [str(c) for c in cmd],
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })

        # 无论复用还是新产出，都重新验一遍
        sc = gguf_scan(dst, deadline)
        if sc["n_tensors"] == 0:
            raise RuntimeError(f"{fmt} 产物没有任何张量: {dst}")
        dom = max(sc["bytes_per_type"].items(), key=lambda x: x[1])[0] if sc["bytes_per_type"] else None
        if dom != fmt:
            log(f"!! {fmt} 的主导类型是 {dom}，不是 {fmt} —— 检查量化是否真的生效")

        out[fmt] = {
            "path": str(dst),
            "file_bytes": sc["file_bytes"],
            "file_gb": sc["file_gb"],
            "n_params": sc["n_params"],
            "bpw_effective": sc["bpw_effective"],
            "dominant_type": dom,
            "reused": bool(reuse),
            "quant_fp": fp,
            "provenance": str(prov_path),
            "sha256": sha256_of(dst, deadline) if CFG["HASH_GGUF"] else None,
        }
        log(f"{fmt}: {sc['file_gb']:.2f} GiB, {sc['bpw_effective']} bpw, "
            f"主导类型 {dom}, {'复用' if reuse else '新产出'}")

    # ---- 公平性：imatrix 与量化 base 的同源性，按实际核验等级说话 ----
    if s3_prov:
        same_imatrix = (s3_prov.get("sha256") == im_sha) if im_sha else None
        imatrix_base_kind = s3_prov.get("base_kind")
        origin = "S3" if s3_prov.get("imatrix") == str(imat) else "user_supplied"
    else:
        same_imatrix, imatrix_base_kind, origin = None, None, "user_supplied"
    imatrix_base_kind = imatrix_base_kind or _imatrix_base_kind()

    # v0.5：`imatrix_base_same_as_quant_base` 的语义改了。base 文件不同
    # （Q8_0 vs BF16）是**设计如此**，所以"路径相同"不再是判据；
    # 真正的判据是 consistency 的 level：file = 逐张量比过形状，
    # provenance = 靠 sha256 + 来源 + 结构三项核验。
    level = consistency.get("level")
    out["_fairness"] = {
        "quant_base": str(base),
        "quant_base_kind": CFG["BASE_KIND"],
        "quant_base_fp": base_fp,
        "quant_base_origin": base_origin,
        "quant_base_download": base_meta,
        "quant_base_repo": S4_PINNED["repo"],
        "quant_base_revision": S4_PINNED["revision"],
        "imatrix": str(imat),
        "imatrix_sha256": im_sha,
        "imatrix_trust": imat_v.get("trust"),
        "imatrix_base": str(imat_base),
        "imatrix_base_kind": imatrix_base_kind,
        "imatrix_base_present": bool(imat_base and Path(imat_base).exists()),
        # 语义变更：不再是比较路径字符串，而是"同源性核验到了哪一级"
        "imatrix_base_same_as_quant_base": None,
        "imatrix_base_consistency_level": level,
        "imatrix_base_consistent": consistency,
        "imatrix_base_from_approx": str(imatrix_base_kind).lower() in QUANTIZED_KINDS,
        "same_imatrix": same_imatrix,
        "imatrix_origin": origin,
        "base_scope": plan.get("validity_scope"),
        "requant": plan.get("provenance_tag"),
        "llama_commit": llm_commit,
        "verified": (bool(same_imatrix) and consistency.get("ok") is True),
        "verified_level": ("file" if level == "file" else
                           ("provenance" if consistency.get("ok") else None)),
        "note": ("verified 需要：imatrix 与 provenance 记录的 sha256 一致，且同源性核验通过。"
                 "None 表示无法验证（没有 provenance 记录 / 未开 HASH_GGUF）。"
                 "imatrix_base_consistency_level=file 表示 imatrix base 原文件还在，"
                 "做过逐张量形状比对；=provenance 表示 Q8_0 已按设计回收，"
                 "改用 sha256 + 来源 repo/revision + 张量名/层数/输入通道三项核验"
                 "（弱于 file 级，但三项互相独立，且来源核对比'形状一致'强）。"
                 "base_scope=from_high_precision 表示从 BF16 量化而来，"
                 "是相对原始 checkpoint 的高保真低比特结论；"
                 "requantized_pilot_only 表示 base 本身是量化文件，不可。"),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out



def _s4_base_for_s5():
    """S5 要分析的量化 base：优先 S4 实际用的那个（可能在暂存盘或 Input），
    而不是只看 CFG['BASE_GGUF'] —— v0.5 起 S4 会自己取 base，那条路径不会回填 CFG。"""
    s4r = _get("S4_QUANT") or {}
    fair = s4r.get("_fairness") or {}
    for key in ("quant_base",):
        p = fair.get(key)
        if p and Path(p).exists():
            return Path(p), "from_S4"
    if CFG["BASE_GGUF"] and Path(CFG["BASE_GGUF"]).exists():
        return Path(CFG["BASE_GGUF"]), "from_CFG"
    return None, None



def stage_s5_bytes(deadline):
    targets = {}
    base_p, base_src = _s4_base_for_s5()
    if base_p is not None:
        targets[f"base_{CFG['BASE_KIND'] or 'unknown'}"] = base_p
    for fmt in FORMATS:
        p = _model_path(fmt)
        if p.exists():
            targets[fmt] = p
    if not targets:
        raise GuardError("没有可分析的 GGUF。先跑 S4，或设置 CFG['BASE_GGUF']。")
    if base_p is not None:
        log(f"S5 分析量化 base：{base_p}（来源 {base_src}，"
            f"{base_p.stat().st_size / 2**30:.2f} GiB）")

    rows, detail, unavailable = [], {}, []
    for name, p in targets.items():
        check_deadline(deadline, f"S5 {name}")
        try:
            sc = gguf_scan(p, deadline, strict_types=True)
        except UnknownGGMLType as e:
            # P1-G：拒绝把未知类型按 8 bpw 写进有效结果
            log(f"!! {name}: {e}")
            unavailable.append({"name": name, "path": str(p), "reason": str(e)})
            detail[name] = {"byte_breakdown_available": False, "path": str(p),
                            "file_bytes": p.stat().st_size,
                            "file_gb": round(p.stat().st_size / 2**30, 3),
                            "error": str(e)}
            continue

        detail[name] = {
            "byte_breakdown_available": True,
            "file_bytes": sc["file_bytes"],
            "file_gb": sc["file_gb"],
            "n_params": sc["n_params"],
            "bpw_effective": sc["bpw_effective"],
            "bpw_filesize": sc["bpw_filesize"],
            "bytes_per_type": sc["bytes_per_type"],
            "bytes_per_class": sc["bytes_per_class"],
            "header_bytes": sc["header_bytes"],
            "n_tensors": sc["n_tensors"],
        }
        rows.append((name, sc["file_gb"], sc["bpw_effective"],
                     sc["bytes_per_class"].get("embed/head", 0), sc["n_params"]))

    print("=" * 78)
    print(f"{'name':22s} {'file GiB':>9s} {'bpw':>8s} {'embed+head GiB':>15s}")
    print("-" * 78)
    for name, gb, bpw, eh, _n in rows:
        bpw_s = f"{bpw:8.4f}" if bpw is not None else f"{'n/a':>8s}"
        print(f"{name:22s} {gb:9.3f} {bpw_s} {eh/2**30:15.3f}")
    print("-" * 78)
    for name, d in detail.items():
        if not d.get("byte_breakdown_available", False):
            print(f"{name}: 字节拆分不可用（未知 ggml type），不出 bpw")
            continue
        print(f"{name} 按类型:", {k: round(v / 2**30, 3) for k, v in d["bytes_per_type"].items()})
        if d["n_params"] == 0:
            print(f"  !! {name} 没有张量（只有词表/元数据），bpw 无法计算")
    print("=" * 78)

    ratios = {}
    d2, d4 = detail.get("IQ2_XS", {}), detail.get("IQ4_XS", {})
    if d2.get("byte_breakdown_available") and d4.get("byte_breakdown_available"):
        b2, b4 = d2["file_bytes"], d4["file_bytes"]
        t2 = b2 - d2["header_bytes"]
        t4 = b4 - d4["header_bytes"]
        ratios = {
            "iq4_over_iq2": round(b4 / b2, 4),
            "bytes_saved_pct": round((1 - b2 / b4) * 100, 2),
            "tensor_only_iq4_over_iq2": round(t4 / t2, 4),
            "note": "文件字节比，不是端到端加速比",
        }
        print("IQ4_XS / IQ2_XS 字节比:", ratios)

    return {"per_model": detail, "ratios": ratios,
            "byte_breakdown_unavailable": unavailable}
