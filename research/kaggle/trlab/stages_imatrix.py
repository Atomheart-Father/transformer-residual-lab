"""S3：Q8_0 / 校准语料获取 + llama-imatrix 生成 + 回收。"""

from .config import CALIB_PINNED, CFG, LOGS, MDLS, NOTEBOOK_VERSION, OUT, S3_PINNED
from .errors import CommandFailed, GuardError
from .paths import _calib_parquet_path, _calib_stamp_path, _calib_txt_path, _q8_stamp_path, _s3_base_path, llama_bin
from .gguf import gguf_scan
from .core import QUANTIZED_KINDS, _fit_ngl, _path_fp, atomic_replace, check_deadline, log, need, run_cmd, vram_gb, write_json
from .fetch import _disk_precheck, _fetch_pinned_blob, _read_stamp, _write_stamp, partial_path, sha256_of
from .imatrix import _imatrix_base_kind, _imatrix_provenance_path, _verify_imatrix, resolve_imatrix

import json, os, time
from pathlib import Path

def _ngl_ladder(fitted, n_layers):
    """-ngl 的降级阶梯（v0.4 / D-7）。

    llama.cpp 的 `-ngl 99` 表示"全部卸载"。这里把 99 折算成真实层数再对折，
    得到 [全量, 全量/2, 0]；fitted=0（纯 CPU）时只有 [0]。
    OOM 时逐级下调，每一级都记录，不静默。
    """
    fitted = int(fitted)
    if fitted <= 0:
        return [0]
    full = n_layers if fitted >= int(n_layers) else fitted
    ladder = [fitted, max(1, full // 2), 0]
    return sorted(set(ladder), reverse=True)



def _ensure_q8(deadline):
    """确保 pinned Q8_0 在本地且校验通过。返回 (path, meta)。

    优先级：已验证的缓存 > 下载。**重复 Run All 不会重下 8 GiB** ——
    这一步是"掉线后重启"还能跑通的关键之一。
    """
    pin = S3_PINNED
    dst = _s3_base_path()
    stamp = _q8_stamp_path()

    st = _read_stamp(stamp)
    if (dst.exists() and st.get("filename") == pin["filename"]
            and st.get("sha256") == pin["sha256"]
            and st.get("bytes") == pin["bytes"]
            and dst.stat().st_size == pin["bytes"]):
        log(f"imatrix base 复用：{dst.name} {pin['bytes'] / 2**30:.2f} GiB，"
            f"sha256 已验证过，不重下")
        return dst, {"origin": "cache", "bytes": pin["bytes"],
                     "sha256": pin["sha256"], "path": str(dst),
                     "etag_checked": False,
                     "downloaded_utc": st.get("written_utc")}

    # 下载前的磁盘预检（D-4）。Q8_0 边下边写 .partial，任何时刻只有一份，
    # 所以按"新增一份完整大小"预留就够。
    _disk_precheck(pin["bytes"], f"S3 下载 {pin['filename']}")
    meta = _fetch_pinned_blob(pin, dst, deadline, tag=f"S3/{pin['filename']}")
    _write_stamp(stamp, filename=pin["filename"], sha256=pin["sha256"],
                 bytes=pin["bytes"], repo=pin["repo"],
                 revision=pin["revision"], url=pin["url"])
    return dst, meta



def _parquet_to_txt(parquet_path, out_txt, label="校准语料"):
    """parquet → 纯文本（`llama-imatrix -f` 只吃文本；上游 `arg.cpp:1808`
    是 `params.prompt = read_file(value)`）。

    wikitext 的 parquet 只有一列 `text`（每行一个段落）。列名探测优先
    `text`，否则取第一个字符串列 —— 不硬编码，换语料不用改代码。

    **只标准库不够**：parquet 是 Thrift + Snappy 的列式格式，纯 Python 读不了。
    这里按 pyarrow → pandas 的顺序探测，两个都没有就 fail fast 并说清要装什么，
    而不是留一个空的 calib.txt 让 imatrix 拿到 0 token 再报一句难懂的错。
    """
    import importlib

    pq = None
    try:
        pq = importlib.import_module("pyarrow.parquet")
    except Exception:
        try:
            pq = importlib.import_module("pandas")
        except Exception:
            pq = None
    if pq is None:
        raise GuardError(
            "读 parquet 需要 pyarrow 或 pandas，当前解释器两个都没有。\n"
            "  → Kaggle 镜像默认带这两个库；若你的环境没有，请 "
            "`pip install pyarrow`（或 pandas）后重跑。\n"
            "  → 或者绕开：自己把 wikitext-2-raw train 转成 .txt，"
            "设 CFG['CALIB_TXT'] 指过去，并把 CFG['S3_AUTO_FETCH_CALIB'] 设 False。")

    out_txt = Path(out_txt)
    tmp = out_txt.with_name(out_txt.name + ".partial")
    if hasattr(pq, "read_table"):
        tbl = pq.read_table(str(parquet_path))
        cols = [c for c in tbl.schema.names]
        col = "text" if "text" in cols else next(
            (c for c in cols if str(tbl.schema.field(c).type).startswith(("string", "large_string"))),
            cols[0] if cols else None)
        if col is None:
            raise GuardError(f"parquet 里没有可用列：{cols}")
        rows = tbl.column(col).to_pylist()
        engine = f"pyarrow {getattr(pq, '__version__', '?')}"
    else:
        df = pq.read_parquet(str(parquet_path))
        cols = list(df.columns)
        col = "text" if "text" in cols else next(
            (c for c in cols if df[c].dtype == object), cols[0] if cols else None)
        if col is None:
            raise GuardError(f"parquet 里没有可用列：{cols}")
        rows = [str(x) for x in df[col].tolist()]
        engine = f"pandas {getattr(pq, '__version__', '?')}"

    n_nonempty = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            s = (r or "").strip()
            if not s:
                continue
            f.write(s)
            f.write("\n\n")
            n_nonempty += 1
    if n_nonempty == 0:
        tmp.unlink(missing_ok=True)
        raise GuardError(
            f"parquet 列 `{col}` 里没有任何非空文本（{len(rows)} 行全空）")
    os.replace(tmp, out_txt)
    log(f"{label}就绪：{out_txt.name} {out_txt.stat().st_size / 1024:.0f} KB，"
        f"{n_nonempty} 段（{engine}，列 `{col}`）")
    # v1.0.2：这里**不能**叫 `bytes`。调用方 `_ensure_calib_txt()` 会把本字典
    # `**conv` 摊进 `_write_stamp(..., bytes=pin["bytes"], **conv)`，同关键字重复
    # 直接 `TypeError: got multiple values for keyword argument 'bytes'` ——
    # 这是 v0.4 的真实 Kaggle 失败，v1.0.1 拆包时被原样抄了回来。
    # 改名后语义也更准确：这是**转换出来的 txt** 的字节数，不是 parquet 的。
    return {"engine": engine, "column": col, "rows": len(rows),
            "nonempty_rows": n_nonempty,
            "converted_bytes": out_txt.stat().st_size}



def _ensure_calib_txt(deadline):
    """确保校准语料在本地。返回 (path, meta)。缓存命中不重下。"""
    pin = CALIB_PINNED
    parquet = _calib_parquet_path()
    txt = _calib_txt_path()
    stamp = _calib_stamp_path()

    if txt.exists() and txt.stat().st_size > 0:
        st = _read_stamp(stamp)
        if (st.get("sha256") == pin["sha256"]
                and st.get("txt_bytes") == txt.stat().st_size):
            log(f"校准语料复用：{txt}（{txt.stat().st_size / 1024:.0f} KB，不重下）")
            return txt, {"origin": "cache", "txt_bytes": txt.stat().st_size,
                         "parquet_sha256": pin["sha256"]}
        log("校准语料的 stamp 与当前 pin 不符 → 重新取")

    _disk_precheck(pin["bytes"], f"S3 下载 {pin['path'].split('/')[-1]}")
    meta = _fetch_pinned_blob(pin, parquet, deadline,
                              tag=f"S3/{pin['path'].split('/')[-1]}")
    conv = _parquet_to_txt(parquet, txt)
    # v1.0.2：显式合并 + 重名检查，而不是 `_write_stamp(..., bytes=..., **conv)`。
    # 那种写法一旦 conv 里出现 bytes（v0.4/v1.0.1 就是这样）就直接 TypeError，
    # 而且报错信息完全不指向真正的原因。
    stamp_rec = {"sha256": pin["sha256"], "bytes": pin["bytes"],
                 "txt_bytes": txt.stat().st_size, "repo": pin["repo"],
                 "revision": pin["revision"], "config": pin["config"],
                 "split": pin["split"]}
    clash = sorted(set(stamp_rec) & set(conv))
    if clash:
        raise GuardError(
            f"转换元数据与 stamp 字段重名 {clash}：继续写会把 parquet 的 "
            f"bytes / sha256 覆盖成转换产物的大小，provenance 就错了。")
    stamp_rec.update(conv)
    _write_stamp(stamp, **stamp_rec)
    meta.update(conv)          # parquet 的 bytes/sha256 必须原样保留
    return txt, meta



def _calib_manifest(calib, need_tokens):
    """记录"这次 imatrix 大概消费了校准语料的哪一段"（v1.2.0）。

    没有观测值这件事必须写清楚：`llama-imatrix` 顺序读文件、按 `-c` 切块，
    但**不写** manifest、也不支持断点，所以只能给近似窗口。
    """
    from .isolation import prefix_window, sha256_text
    text = Path(calib).read_text(errors="replace")
    win = prefix_window(text, need_tokens)
    return {
        "calib_path": str(calib),
        "calib_sha256": sha256_text(text),
        "calib_bytes": len(text.encode("utf-8")),
        "chunks": CFG["IMATRIX_CHUNKS"],
        "ctx": CFG["IMATRIX_CTX"],
        "need_tokens": need_tokens,
        "prefix_window": win,
        "level": "derived_prefix_window",
        "caveat": ("llama-imatrix 不写 manifest：这里是**近似**窗口"
                   "（文件前缀 × need_tokens × 4 chars/token），"
                   "不是观测到的 token 序列。不能据此声称『imatrix 只看过这一段』。"),
    }


def _is_oom_tail(tail):
    """判断 llama-imatrix 的失败是不是显存不够（同行/邻近匹配，不全文猜）。

    `CommandFailed.tail` 只有日志最后 1800 字符，所以判据要短而准。
    宁可漏判（当成普通失败重报）也不要把普通失败说成 OOM 让人白等。
    """
    t = (tail or "").lower()
    keys = ("out of memory", "cudaerrormemoryallocation", "cudaoom",
            "failed to allocate", "bad_alloc", "std::bad_alloc",
            "llama_context: ... failed", "ggml_gallocr_reserve_n: failed to allocate",
            "cuda error", "error: out of memory")
    return any(k in t for k in keys)



def _run_imatrix_with_fallback(base, calib, part, deadline, ngl0):
    """跑 llama-imatrix；显存不够就按阶梯降 -ngl，每次尝试都记录（D-7）。

    返回 (final_ngl, attempts)。attempts 是每次尝试的字典列表，
    原样进 provenance —— "OOM 过、降到几"必须可查，不能只写最终值。
    """
    v = vram_gb()
    ladder = _ngl_ladder(ngl0, CFG["N_LAYERS"])
    if len(ladder) > 1:
        log(f"-ngl 阶梯：{ladder}（显存不足时逐级降；"
            f"总量 {v['total'] if v else '?'} GiB）")
    attempts = []
    last = None
    for i, ngl in enumerate(ladder):
        check_deadline(deadline, f"S3 imatrix 尝试 {i + 1}/{len(ladder)}（-ngl {ngl}）")
        if part.exists():
            part.unlink()
        cmd = [llama_bin("llama-imatrix"),
               "-m", base, "-f", calib, "-o", part,
               "-c", CFG["IMATRIX_CTX"], "--chunks", CFG["IMATRIX_CHUNKS"],
               "-ngl", ngl, "-t", os.cpu_count() or 4]
        t0 = time.time()
        try:
            run_cmd(cmd, LOGS / "s3_imatrix.log", deadline)
            attempts.append({"ngl": ngl, "ok": True,
                             "seconds": round(time.time() - t0, 1)})
            return ngl, attempts
        except CommandFailed as e:
            oom = _is_oom_tail(e.tail)
            attempts.append({"ngl": ngl, "ok": False, "seconds": round(time.time() - t0, 1),
                             "rc": e.rc, "oom": oom,
                             "tail": (e.tail or "")[-400:]})
            last = e
            remaining = [x for x in ladder[i + 1:]]
            if oom and remaining:
                nxt = remaining[0]
                log(f"!! -ngl {ngl} 显存不足（日志尾部含 OOM 特征）→ 降到 -ngl {nxt} 重试")
                continue
            break
    raise RuntimeError(
        f"llama-imatrix 失败（已试 -ngl {ladder}）。\n"
        + _explain_imatrix_failure(last)
        + f"\n完整日志：{LOGS / 's3_imatrix.log'}")



def _explain_imatrix_failure(exc):
    """把 llama-imatrix 的失败压成"症状 → 处置"。"""
    if exc is None:
        return "没有可用日志。"
    tail = (exc.tail or "").lower()
    lines = [l for l in tail.splitlines() if l.strip()]

    def _has(n):
        return n in tail

    def _same_line(a, b):
        return any(a in l and b in l for l in lines)

    if _is_oom_tail(tail):
        return ("显存不足（已自动逐级降低 -ngl 到 0 仍失败）。\n"
                "  → 处置：确认没有别的进程占显存；或把 CFG['NGL_MARGIN_GB'] 调大；"
                "或设 CFG['IMATRIX_BASE'] 指向更小的 base。")
    if _same_line("failed to open", "model") or _has("failed to load model"):
        return ("模型文件打不开。多半是下载被截断或 base 指向了不存在的路径。\n"
                "  → 处置：删掉 `trlab/dl/models/` 下对应文件与 `.stamp.json` 重跑，"
                "让 S3 重新下载并校验 sha256。")
    if _same_line("you need at least", "tokens") or _has("tokenizes to only"):
        return ("校准语料 token 数不够 `--chunks × -c`。\n"
                f"  → 当前需要 {CFG['IMATRIX_CHUNKS']} × {CFG['IMATRIX_CTX']} = "
                f"{CFG['IMATRIX_CHUNKS'] * CFG['IMATRIX_CTX']} token。\n"
                "  → 处置：换更大的语料，或把 CFG['IMATRIX_CHUNKS'] / "
                "CFG['IMATRIX_CTX'] 调小。")
    if _has("imatrix") and (_has("garbage") or _has("bailing out")):
        return ("imatrix 相关错误。\n  → 处置：确认 `-o` 指向的目录可写，且没有残留 "
                "`.partial`。")
    return (f"命令失败 rc={exc.rc}。\n  → 处置：看上面这段日志尾部；"
            f"最常见的是语料/模型路径与显存两类，都已覆盖。")



def _reclaim_q8(path, stamp, rec):
    """imatrix 验证通过**且结果记录已写盘**之后，才回收下载来的 Q8_0（D-5）。

    只在"这个文件确实是本阶段自动下载的"时才删。用户手动配的
    `CFG['IMATRIX_BASE']` 一概不碰。删之前把释放的字节数记进 rec，
    让报告能对上磁盘账。
    """
    if not CFG["S3_DELETE_Q8_AFTER"]:
        log(f"CFG['S3_DELETE_Q8_AFTER']=False → 保留 {path}")
        rec["q8_reclaimed"] = False
        rec["q8_reclaim_reason"] = "CFG['S3_DELETE_Q8_AFTER']=False"
        return False
    p = Path(path)
    if not p.exists():
        rec["q8_reclaimed"] = False
        rec["q8_reclaim_reason"] = "文件已不在（上一轮已回收？）"
        return False
    freed = p.stat().st_size
    p.unlink()
    Path(stamp).unlink(missing_ok=True)
    part = p.with_name(p.name + ".partial")
    if part.exists():
        part.unlink()
    rec["q8_reclaimed"] = True
    rec["q8_reclaimed_bytes"] = freed
    rec["q8_reclaim_reason"] = "imatrix 已验证且结果记录已写盘"
    log(f"已回收 Q8_0：释放 {freed / 2**30:.2f} GiB（{p.name}）")
    return True



def _s3_base_origin(path):
    """判定一份 imatrix base 是"我们自己下载的"还是"用户手动给的"。

    **不能**再看 `CFG['IMATRIX_BASE']` 是否非空 —— v0.4 的真实 bug：
    `_ensure_q8()` 会把下载到的路径写回 `CFG['IMATRIX_BASE']`（为了让指纹和
    下游看到同一份），于是重跑时"手动指定"分支抢先命中，`base_origin` 被记成
    `manual`，`_reclaim_q8()` 的 `origin == "auto_download"` 条件不成立，
    8.11 GiB 的 Q8_0 就永远留在 working 里。真实运行的 S3_IMATRIX.json 里
    `base_origin: "manual"` + `base_download: null` 就是这个 bug 的现场。

    判定改成看**路径本身**：落在 `_s3_base_path()`（自动下载的唯一落点）上，
    或者 stamp 里记了这个文件，就算我们自己的。
    """
    p = Path(path)
    if not CFG.get("S3_AUTO_FETCH_MODEL") and not CFG.get("S3_AUTO_FETCH_CALIB"):
        # 用户没开自动取数 → 一律按手动处理（保守：不删别人的文件）
        return ("manual", None)
    try:
        same = p.resolve() == _s3_base_path().resolve()
    except OSError:
        same = False
    if same:
        st = _read_stamp(_q8_stamp_path())
        if st.get("sha256") == S3_PINNED["sha256"]:
            return ("auto_download", {"origin": "cache_or_download",
                                      "bytes": st.get("bytes"),
                                      "sha256": st.get("sha256"),
                                      "downloaded_utc": st.get("written_utc"),
                                      "path": str(p)})
        return ("auto_download", {"origin": "downloaded_expected",
                                  "bytes": S3_PINNED["bytes"],
                                  "sha256": S3_PINNED["sha256"],
                                  "path": str(p)})
    if CFG["IMATRIX_BASE"]:
        return ("manual", None)
    return ("manual", None)



def stage_s3_imatrix(deadline):
    """S3：取 imatrix base + 校准语料 → 生成 imatrix → 验证 → 记录 → 回收 Q8_0。

    四条顺序上的硬约束（都是 v0.4/v0.5 踩过或预演过的坑）：
      · 先 `llama_bin()` 找二进制（S2 失败时 `run_stage` 已经拦过一次，
        这里是双保险：直接跑会报"找不到 llama-imatrix"，说清楚缺什么）。
      · 已有**已验证**的 imatrix 时整段跳过 —— 不重下 8.11 GiB 的 Q8_0、
        不重算 58 秒的 imatrix。这是"新 Session 也能恢复"的关键。
      · imatrix 产物**验证通过**才 `os.replace` 成正式文件名。
      · Q8_0 在**结果记录写盘之后**才删。任一步失败都不删，
        重跑能从缓存/断点继续，不必把 8 GiB 重下一遍。
    """
    bin_imatrix = llama_bin("llama-imatrix")

    # ---- 0. 已有 imatrix 就复用（v0.5 / 需求 3）----
    if CFG.get("S3_SKIP_IF_IMATRIX_OK"):
        have = resolve_imatrix()
        if have is not None:
            v = _verify_imatrix(have, deadline)
            if v.get("ok") and v.get("trust") in ("pinned_verified",
                                                  "provenance_verified"):
                log(f"imatrix 已存在且已验证（{v['trust']}）：{have}")
                log(f"  sha256 {str(v.get('sha256'))[:16]}…  {v.get('bytes')} B  "
                    f"{v.get('n_tensors')} 个张量")
                log("  → 跳过 S3：不重下 Q8_0、不重算 imatrix")
                return {"skipped": True, "reason": "imatrix_reuse",
                        "imatrix": str(have), "sha256": v.get("sha256"),
                        "bytes": v.get("bytes"), "n_tensors": v.get("n_tensors"),
                        "trust": v.get("trust"), "why": v.get("why"),
                        "warnings": v.get("warnings") or [],
                        "note": ("S3 未运行；imatrix 来自上一次运行并经 sha256 验证。"
                                 "S4 会用它做 provenance 级同源核验。")}
            log(f"!! 找到 imatrix 但不可信（{v.get('trust')}）：{v.get('why')}"
                f" → 重新生成")

    # ---- 1. imatrix base ----
    base_meta = None
    if CFG["IMATRIX_BASE"]:
        base = Path(CFG["IMATRIX_BASE"])
        if not base.exists():
            raise GuardError(f"CFG['IMATRIX_BASE'] 指向的路径不存在: {base}")
        # v0.5：来源判定不看 CFG 是否被上一轮污染，看路径与 stamp
        base_origin, base_meta = _s3_base_origin(base)
        log(f"imatrix base：{'自动下载' if base_origin == 'auto_download' else '手动指定'} {base}")
    elif CFG["S3_AUTO_FETCH_MODEL"]:
        base, base_meta = _ensure_q8(deadline)
        CFG["IMATRIX_BASE"] = str(base)     # 让指纹与下游阶段看到同一份
        base_origin = "auto_download"
    else:
        base = need("BASE_GGUF")
        base_origin = "cfg_base_gguf"
        log(f"imatrix base：用 CFG['BASE_GGUF'] = {base}")
    if base_origin == "auto_download" and not base_meta:
        base_meta = {"origin": "downloaded", "bytes": S3_PINNED["bytes"],
                     "sha256": S3_PINNED["sha256"], "path": str(base)}
    base_kind = _imatrix_base_kind()
    if not base_kind:
        raise GuardError(
            "CFG['BASE_KIND'] 与 CFG['IMATRIX_BASE_KIND'] 都没设置"
            "（必须写 f16 / bf16 / q8_0 ...）")
    if str(base_kind).lower() not in QUANTIZED_KINDS and base_kind.lower() not in (
            "f16", "bf16", "fp16", "f32"):
        log(f"!! 不认识的 base_kind={base_kind}，按'未量化'处理")

    # ---- 2. 校准语料 ----
    calib_meta = None
    if CFG["CALIB_TXT"]:
        calib = Path(CFG["CALIB_TXT"])
        if not calib.exists():
            raise GuardError(f"CFG['CALIB_TXT'] 指向的路径不存在: {calib}")
        calib_origin = "manual"
    elif CFG["S3_AUTO_FETCH_CALIB"]:
        calib, calib_meta = _ensure_calib_txt(deadline)
        CFG["CALIB_TXT"] = str(calib)
        calib_origin = "auto_download"
    else:
        calib = need("CALIB_TXT")
        calib_origin = "cfg_calib_txt"

    need_tokens = CFG["IMATRIX_CHUNKS"] * CFG["IMATRIX_CTX"]
    if calib.stat().st_size < 200_000:
        log(f"!! 校准语料只有 {calib.stat().st_size / 1024:.0f} KB，"
            f"可能凑不满 {need_tokens} token")

    # ---- 2.5 校准 manifest（v1.2.0）----
    # llama-imatrix 不写 manifest，也不支持断点，所以"它到底看过哪些字"没有
    # 观测值。能做的是把**近似窗口**（文件前缀 × need_tokens × 4 chars/token）
    # 钉在产物旁边：以后 S6 判 `actual_imatrix_calib_overlap` 时至少有个可核的
    # 依据，而不是只能写 unknown。等级永远是 derived_prefix_window，不是 observed。
    calib_manifest = _calib_manifest(calib, need_tokens)
    log(f"校准 manifest：calib sha256 {calib_manifest['calib_sha256'][:16]}…，"
        f"近似窗口前 {calib_manifest['prefix_window']['chars']} 字符"
        f"（{need_tokens} token × {calib_manifest['prefix_window']['chars_per_token']}）")


    # ---- 3. -ngl 拟合（余量默认 2 GiB，v0.4 / D-7）----
    v = vram_gb()
    usable = (v["total"] - CFG["NGL_MARGIN_GB"]) if v else 0.0
    ngl0 = _fit_ngl(base.stat().st_size, usable, CFG["NGL_MARGIN_GB"])
    log(f"base={base.name} {base.stat().st_size / 2**30:.2f} GiB，"
        f"显存总量 {v['total']:.2f} GiB、预留 {CFG['NGL_MARGIN_GB']} GiB"
        f" → 可用 {usable:.2f} GiB → -ngl {ngl0}"
        if v else f"base={base.name}，无 GPU → -ngl 0")

    out_path = Path(CFG["IMATRIX_GGUF"]) if CFG["IMATRIX_GGUF"] else (MDLS / "imatrix.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(out_path.parent, os.W_OK):
        raise GuardError(
            f"{out_path.parent} 不可写。CFG['IMATRIX_GGUF'] 不能指向 "
            "/kaggle/input（只读）；留空则写到 working。")

    part = partial_path(out_path)
    try:
        ngl, attempts = _run_imatrix_with_fallback(base, calib, part, deadline, ngl0)

        # ---- 4. 验证后才原子改名（P1-E）----
        if not part.exists() or part.stat().st_size < 100_000:
            raise RuntimeError(
                f"imatrix 产物缺失或过小: {part} "
                f"({part.stat().st_size if part.exists() else 'missing'} B)")
        sc = gguf_scan(part)
        if sc["n_tensors"] == 0:
            raise RuntimeError(f"imatrix 产物是合法 GGUF 但没有任何张量: {part}")
        atomic_replace(part, out_path)
    except BaseException:
        # 半成品 imatrix 不能续跑（llama-imatrix 没有断点），留着只是白占
        # /kaggle/working 的 20 GB。Q8_0 / 校准语料 / 下载断点都保留 ——
        # 那些才是"修完直接重跑"需要的可恢复状态。
        if part.exists():
            part.unlink(missing_ok=True)
            log(f"已删除半成品 imatrix {part.name}（不可续跑，不占 working 空间）")
        raise

    rec = {
        "imatrix": str(out_path),
        "bytes": out_path.stat().st_size,
        "n_tensors": sc["n_tensors"],
        "sha256": sha256_of(out_path, deadline) if CFG["HASH_GGUF"] else None,
        "base_gguf": str(base),
        "base_kind": base_kind,
        "base_bytes": base.stat().st_size,
        "base_fp": _path_fp(base),
        "base_is_quantized": str(base_kind).lower() in QUANTIZED_KINDS,
        "imatrix_base_from_approx": str(base_kind).lower() in QUANTIZED_KINDS,
        "base_origin": base_origin,
        "base_download": base_meta,
        # v0.5：S4 在 Q8_0 被回收后要靠这几项做 provenance 级同源核验，
        # 所以**即使文件已经不在了也必须写全**。它们来自 S3_PINNED（notebook 内
        # 预登记、与 HF 双源核对过），不是从文件现读的。
        "base_repo": S3_PINNED["repo"],
        "base_revision": S3_PINNED["revision"],
        "base_url": S3_PINNED["url"],
        "base_expected_sha256": S3_PINNED["sha256"],
        "base_expected_bytes": S3_PINNED["bytes"],
        "base_file_present_now": base.exists(),
        "calib_txt": str(calib),
        "calib_bytes": calib.stat().st_size,
        "calib_source": CFG["CALIB_SOURCE"],
        "calib_origin": calib_origin,
        "calib_download": calib_meta,
        "calib_manifest": calib_manifest,
        "ctx": CFG["IMATRIX_CTX"],
        "chunks": CFG["IMATRIX_CHUNKS"],
        "need_tokens": need_tokens,
        "ngl": ngl,
        "ngl_initial": ngl0,
        "ngl_attempts": attempts,
        "on_gpu": ngl > 0,
        "vram_total_gb": (v["total"] if v else None),
        "vram_margin_gb": CFG["NGL_MARGIN_GB"],
        "llama_bin": str(bin_imatrix),
    }
    if rec["imatrix_base_from_approx"]:
        log(f"!! imatrix 由 {base_kind} 算的：激活是对 f16 的近似。"
            "这比『从 Q8_0 二次量化』好得多，但要在报告里声明。")

    # ---- 5. 结果记录写盘（D-5：必须在回收 Q8_0 之前）----
    # 两份都写：
    #   OUT/S3_IMATRIX.json            本次运行的阶段记录（run_stage 会覆盖成带指纹版）
    #   MDLS/imatrix.provenance.json   **挨着 imatrix 的** provenance（v0.5）
    # 后者才是"新 Session 恢复"的关键：OUT/ 可能随会话丢，models/ 里的产物
    # 和它的 provenance 是一起被保存/挂载的。S4 只认第二份。
    pre_path = OUT / "S3_IMATRIX.json"
    write_json(pre_path, {
        "stage": "S3_IMATRIX",
        "meta": {"status": "ok", "fingerprint": None, "prewritten": True,
                 "note": ("由 stage_s3_imatrix 在回收 Q8_0 之前预写；"
                          "run_stage 随后会用带计时与指纹的版本覆盖同一路径")},
        "result": rec})
    log(f"imatrix 结果已写盘：{pre_path}")

    sidecar = _imatrix_provenance_path()
    try:
        write_json(sidecar, dict(rec, schema=2, written_by="stage_s3_imatrix",
                                 notebook=NOTEBOOK_VERSION))
        log(f"imatrix provenance 已写盘：{sidecar}")
    except OSError as e:
        log(f"!! imatrix provenance 写失败（{e}）—— 新 Session 将无法做 provenance 级核验")

    # ---- 6. 现在才允许回收 Q8_0 ----
    # auto_download 与"缓存命中"都算我们自己的：缓存是上一轮自己下的，
    # 不删就会让 working 长期少 8.11 GiB（v0.4 真实运行就撞在这一条上）。
    if base_origin == "auto_download":
        _reclaim_q8(base, _q8_stamp_path(), rec)
        # 回收后把记录再写一次，让磁盘账与 q8_reclaimed 一致
        rec["base_file_present_now"] = base.exists()   # 刚删掉，必须是 False
        write_json(pre_path, {
            "stage": "S3_IMATRIX",
            "meta": {"status": "ok", "fingerprint": None, "prewritten": True,
                     "note": "同上；此版本含 Q8_0 回收结果"},
            "result": rec})
        try:
            write_json(sidecar, dict(rec, schema=2, written_by="stage_s3_imatrix",
                                     notebook=NOTEBOOK_VERSION))
        except OSError:
            pass
    else:
        rec.setdefault("q8_reclaimed", False)
        rec.setdefault("q8_reclaim_reason",
                       f"base_origin={base_origin}，不是本阶段自动下载的文件，不删")

    print(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    return rec
