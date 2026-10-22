"""imatrix 的发现、验证、来源记录与同源性核验。

三态可信度：pinned_verified / provenance_verified / unverified。
"""

from .config import CFG, IMATRIX_PINNED, IN, MDLS, OUT, S3_PINNED, S4_PINNED
from .paths import _calib_txt_path, _s3_base_path
from .gguf import _base_consistency, gguf_scan
from .core import _read_json, need
from .fetch import sha256_of

import os
from pathlib import Path

def _imatrix_base():
    """算 imatrix 用的 base。**文件不在也不抛** —— v0.5 起 S3 跑完会回收 Q8_0，
    S4 靠 provenance 而不是靠重新读那个文件来核验同源性。

    返回的 Path 可能是"记录里有、磁盘上没有"的路径；调用方用 `.exists()`
    自己决定走文件级还是 provenance 级核验。
    """
    if CFG["IMATRIX_BASE"]:
        return Path(CFG["IMATRIX_BASE"])
    if CFG.get("S3_AUTO_FETCH_MODEL") or CFG.get("S3_AUTO_FETCH_CALIB"):
        return _s3_base_path()
    return Path(CFG["BASE_GGUF"] or "")



def _imatrix_base_kind():
    return CFG["IMATRIX_BASE_KIND"] or CFG["BASE_KIND"]



IMATRIX_STAT_SUFFIXES = (".in_sum", ".in_sum2", ".counts", ".stats")



def _imatrix_provenance_path():
    """imatrix 旁边的 provenance（**与 imatrix 同在 MDLS**，随产物一起被保存）。

    放这里而不是只放 OUT/，是因为 OUT/ 属于"本次运行的输出目录"，
    新 Session 恢复时 models/ 更可能还在；而且 S4 关心的是"这个 imatrix
    是谁产的"，天然该挨着文件本身。
    """
    return MDLS / "imatrix.provenance.json"



def _imatrix_candidates():
    """按优先级列出可能的 imatrix 路径（都不保证存在）。"""
    out = []
    if CFG["IMATRIX_GGUF"]:
        out.append(Path(CFG["IMATRIX_GGUF"]))
    out.append(MDLS / "imatrix.gguf")
    if IN.is_dir():
        for hit in _scan_input_gguf(max_hits=8):
            if hit.name == "imatrix.gguf":
                out.append(hit)
    return out



def _scan_input_gguf(max_hits=8, time_budget_sec=20.0):
    """在 /kaggle/input 下找 .gguf 文件（限时、限数量，不遍历到天荒地老）。"""
    import time as _t

    found, t0, dirs = [], _t.time(), 0
    if not IN.is_dir():
        return found
    for root, dnames, fnames in os.walk(IN):
        dirs += 1
        if dirs % 64 == 0 and _t.time() - t0 > time_budget_sec:
            break
        for fn in fnames:
            if fn.endswith(".gguf") and not fn.endswith(".partial"):
                found.append(Path(root) / fn)
                if len(found) >= max_hits:
                    return found
    return found



def _imatrix_model_tensors(scan):
    """从 imatrix 的 scan 结果反推它描述的模型张量名 → 输入通道数。

    imatrix.gguf 里每个模型张量对应 4 个统计量条目：
        blk.0.attn_q.weight.{in_sum,in_sum2,counts,stats}
    `.in_sum` 的元素个数就是该模型张量的**输入通道数 n_in**
    （实测：Qwen3-8B 上 ffn_down.weight → 12288 = ffn 宽度，其余 → 4096 = hidden）。
    这是不读原 base 文件也能做结构核验的关键信号。
    """
    out = {}
    for t in scan.get("tensors", []):
        name = t["name"]
        for suf in IMATRIX_STAT_SUFFIXES:
            if name.endswith(suf):
                base_name = name[: -len(suf)]
                if suf == ".in_sum":
                    out[base_name] = t.get("n_elements")
                break
    return out



def _imatrix_provenance_records():
    """所有可能描述"当前这份 imatrix 怎么来的"的记录，按可信度排序。"""
    recs = []
    p = _imatrix_provenance_path()
    r = _read_json(p)
    if isinstance(r, dict):
        r = dict(r)
        r["_source"] = str(p)
        recs.append(r)
    r2 = _read_json(OUT / "S3_IMATRIX.json")
    if isinstance(r2, dict):
        res = r2.get("result") if isinstance(r2.get("result"), dict) else r2
        if isinstance(res, dict) and res.get("imatrix"):
            res = dict(res)
            res["_source"] = str(OUT / "S3_IMATRIX.json")
            recs.append(res)
    return recs



def _verify_imatrix(path, deadline=None):
    """校验一份 imatrix：算 sha256、读结构、比对 pin 与 provenance。

    返回 dict，`trust` ∈
      pinned_verified    sha256 与 IMATRIX_PINNED 一致（最强：真实运行产物）
      provenance_verified  sha256 与某条 provenance 记录一致（换过配置，可解释）
      unverified         有文件但没有能对上的记录
    调用方必须自己决定 `unverified` 是否放行（默认不放行）。
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"ok": False, "path": str(p), "trust": "missing",
                "why": f"文件不存在或为空: {p}"}
    size = p.stat().st_size
    digest = sha256_of(p, deadline) if CFG.get("HASH_GGUF", True) else None
    try:
        sc = gguf_scan(p, deadline)
    except Exception as e:
        return {"ok": False, "path": str(p), "sha256": digest, "bytes": size,
                "trust": "unreadable", "why": f"GGUF 解析失败: {e}"}
    if sc["n_tensors"] == 0:
        return {"ok": False, "path": str(p), "sha256": digest, "bytes": size,
                "trust": "empty", "why": "合法 GGUF 但没有任何张量"}

    pin = IMATRIX_PINNED
    trust, why = "unverified", "没有找到能与 sha256 对上的 provenance 记录"
    if digest and digest == pin["sha256"]:
        trust, why = "pinned_verified", "sha256 与 IMATRIX_PINNED 一致（真实 Kaggle 运行产物）"
    else:
        for rec in _imatrix_provenance_records():
            if rec.get("sha256") and rec.get("sha256") == digest:
                trust = "provenance_verified"
                why = (f"sha256 与 {rec['_source']} 的记录一致"
                       f"（与 pin 不同：{digest[:12]}… vs {pin['sha256'][:12]}…）")
                break

    warn = []
    if pin.get("bytes") and size != pin["bytes"]:
        warn.append(f"字节数 {size} ≠ pin 的 {pin['bytes']}")
    if pin.get("n_tensors") and sc["n_tensors"] != pin["n_tensors"]:
        warn.append(f"张量数 {sc['n_tensors']} ≠ pin 的 {pin['n_tensors']}")

    return {"ok": True, "path": str(p), "sha256": digest, "bytes": size,
            "n_tensors": sc["n_tensors"], "n_params": sc.get("n_params"),
            "trust": trust, "why": why, "warnings": warn,
            "model_tensors": _imatrix_model_tensors(sc),
            "provenance": (rec if trust == "provenance_verified" else None)}



def _imatrix_base_consistency(imat, base, s3_rec, deadline=None):
    """imatrix 与量化 base 是否同源。**不要求 imatrix base 原文件存在。**

    两条路：
      file         imatrix base 文件还在 → 逐张量比形状（最强，v0.4 的行为）
      provenance   Q8_0 已被回收 → 用 provenance + GGUF 元数据 + 来源核验

    provenance 路的四项检查互相独立：
      ① sha256：imatrix 的 sha256 与记录一致（防"换了文件"）
      ② 来源：记录的 base repo/revision 与 S4_PINNED 相同（同仓库同 revision
         ⇒ 同一次转换 ⇒ 同一份权重；这比"张量形状一致"强得多）
      ③ 结构：imatrix 的张量名 ⊆ base 的张量名，且层数一致
      ④ 尺寸：每个 imatrix 张量的 n_in 与 base 对应张量的输入通道一致
    """
    base = Path(base)
    base_sc = gguf_scan(base, deadline)
    imat_sc = gguf_scan(imat, deadline)
    imat_tensors = _imatrix_model_tensors(imat_sc)
    base_tensors = {t["name"]: t for t in base_sc.get("tensors", [])}
    layers_i = sorted({int(n.split(".")[1]) for n in imat_tensors
                       if n.startswith("blk.") and "." in n[4:]})
    layers_b = sorted({int(n.split(".")[1]) for n in base_tensors
                       if n.startswith("blk.") and "." in n[4:]})

    # ---- 结构检查（两条路都要做）----
    missing = sorted(set(imat_tensors) - set(base_tensors))
    if missing:
        return {"ok": False, "level": "structure",
                "why": (f"imatrix 里有 {len(missing)} 个张量在量化 base 中不存在，"
                        f"例如 {missing[:3]} —— 这不是同一个模型")}
    if layers_i and layers_b and layers_i != layers_b:
        return {"ok": False, "level": "structure",
                "why": f"层数不同：imatrix {len(layers_i)} 层 vs base {len(layers_b)} 层"}

    # n_in 方向自动判定：GGUF/ggml 的 dims[0] 约定可能与写入端相反，
    # 所以要求"所有张量用同一个方向 100% 匹配"，不允许一半对一半。
    orient, bad = None, []
    for cand in (0, 1):
        mism = []
        for name, n_in in imat_tensors.items():
            dims = base_tensors[name].get("dims") or []
            if len(dims) <= cand or dims[cand] != n_in:
                mism.append((name, n_in, dims))
        if not mism:
            orient, bad = cand, []
            break
        if bad is None or not bad:
            bad = mism
    if orient is None:
        n, n_in, dims = bad[0]
        return {"ok": False, "level": "structure",
                "why": (f"输入通道数对不上：{len(bad)} 个张量不一致，例如 {n} "
                        f"imatrix n_in={n_in} vs base dims={dims}"
                        f"（dims[0] 与 dims[1] 两个方向都不匹配）")}

    structure = {"ok": True, "n_tensors": len(imat_tensors),
                 "architecture": base_sc.get("architecture"),
                 "layers": len(layers_i) or None,
                 "n_in_orientation": f"dims[{orient}]",
                 "imatrix_has_architecture": bool(imat_sc.get("architecture"))}

    # ---- 文件级：imatrix base 还在，直接比形状 ----
    rec_base = (s3_rec or {}).get("base_gguf")
    if rec_base and Path(rec_base).exists():
        other = gguf_scan(rec_base, deadline)
        c = _base_consistency(other, base_sc)
        c.update(level="file", structure=structure)
        return c

    # ---- provenance 级 ----
    if not isinstance(s3_rec, dict) or not s3_rec:
        return {"ok": False, "level": "provenance",
                "why": ("没有 imatrix base 文件、也没有 S3 的 provenance 记录，"
                        "无法证明 imatrix 与量化 base 同源")}
    problems = []
    if not s3_rec.get("sha256"):
        problems.append("记录里没有 imatrix sha256")
    im_sha = sha256_of(imat, deadline) if CFG.get("HASH_GGUF", True) else None
    if im_sha and s3_rec.get("sha256") and im_sha != s3_rec["sha256"]:
        problems.append(f"imatrix sha256 与记录不符（{im_sha[:12]}… vs "
                        f"{s3_rec['sha256'][:12]}…）")
    pin_repo = s3_rec.get("base_repo") or S3_PINNED["repo"]
    pin_rev = s3_rec.get("base_revision") or S3_PINNED["revision"]
    if pin_repo != S4_PINNED["repo"] or pin_rev != S4_PINNED["revision"]:
        problems.append(
            f"imatrix base 的来源不是 S4_PINNED 的那个仓库/revision："
            f"{pin_repo}@{pin_rev} ≠ {S4_PINNED['repo']}@{S4_PINNED['revision']}")
    if s3_rec.get("base_bytes") and s3_rec["base_bytes"] != S3_PINNED["bytes"]:
        problems.append(f"记录的 base 字节数 {s3_rec['base_bytes']} ≠ "
                        f"S3_PINNED 的 {S3_PINNED['bytes']}")
    if problems:
        return {"ok": False, "level": "provenance",
                "why": "；".join(problems)}
    return {"ok": True, "level": "provenance",
            "n_tensors": structure["n_tensors"],
            "architecture": structure["architecture"],
            "layers": structure["layers"],
            "n_in_orientation": structure["n_in_orientation"],
            "verified_by": ("imatrix sha256 + 来源 repo/revision 与 S4_PINNED 一致"
                            " + 张量名/层数/输入通道结构一致"),
            "note": ("imatrix base 原文件已被回收（设计如此），"
                     "本条是 provenance 级核验，弱于文件级逐张量比形状"),
            "structure": structure}



def resolve_imatrix():
    """imatrix 的解析路径；不存在返回 None（不抛）。"""
    for p in _imatrix_candidates():
        if p.exists() and p.stat().st_size > 0:
            return p
    return None



def _imatrix_base_for_fp():
    """指纹用的 imatrix base 解析：**不抛异常**，自动下载路径也算数。

    必须和 `stage_s3_imatrix` 实际用的文件一致，否则"换了 Q8_0 但指纹没变"
    会让 S3 拿着旧 imatrix 冒充新结果。手动 > 自动下载 > 量化 base。
    """
    if CFG["IMATRIX_BASE"]:
        p = Path(CFG["IMATRIX_BASE"])
        return p if p.exists() else None
    if CFG.get("S3_AUTO_FETCH_MODEL"):
        p = _s3_base_path()
        return p if p.exists() else None
    try:
        return need("BASE_GGUF")
    except Exception:
        return None



def _calib_txt_for_fp():
    """指纹用的校准语料解析：不抛异常，自动下载/转换的路径也算数。"""
    if CFG["CALIB_TXT"]:
        p = Path(CFG["CALIB_TXT"])
        return p if p.exists() else None
    if CFG.get("S3_AUTO_FETCH_CALIB"):
        p = _calib_txt_path()
        return p if p.exists() else None
    return None
