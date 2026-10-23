"""评测输入解析（v1.1）：本地 S4 产物 **或** HF 上已存的量化 GGUF。

存在理由：量化已经在真实 Kaggle 上跑完并把产物存到 Hugging Face 了
（`Atomheart-Father/trlab-qwen3-8b-quant-pilot`，commit `097e692`）。
再跑一次 BF16 → IQ2/IQ4 是纯粹的浪费：15.26 GiB 下载 + 两次 8B CPU 量化，
而 S6/S7 只需要那两个 .gguf 和它们的 provenance sidecar。

这里只做四件事：
  1. 找到（或按 tree 的 size/lfs.oid 校验着下载）IQ2_XS / IQ4_XS；
  2. 把 sidecar 里的自述整理成**声明级** fairness 记录（不冒充已核验）；
  3. 评测语料用 pinned wikitext test 自动取，并与 train 做重叠断言；
  4. 算清楚"这次评测大概要跑多久"，不可行就 fail-closed 给一步。

**不重量化、不下载 BF16、不重做 S3。**
"""

from .config import CFG, EVAL_PINNED, FORMATS, MDLS, WORK
from .errors import GuardError
from .core import _read_json, log, vram_gb, write_json
from .fetch import _disk_precheck, _fetch_pinned_blob, _read_stamp, _write_stamp
from .paths import (_eval_parquet_path, _eval_stamp_path, _eval_txt_path,
                    _model_path, llama_bin)
from .stages_imatrix import _parquet_to_txt
from .stages_quant import _ram_precheck
from . import hf

import os
from pathlib import Path


# 成本模型里的速率是**假设值**，不是实测。真实速率必须由 S6 记录后回填。
ASSUMED_TOK_S_CPU = 2.5
ASSUMED_TOK_S_GPU = 30.0
PER_INVOCATION_OVERHEAD_SEC = 6.0


# ---------------------------------------------------------------- 评测语料

def _ensure_eval_txt(deadline):
    """确保 wikitext-2-raw **test** 在本地。缓存命中不重下。

    与 `_ensure_calib_txt()` 同一套 stamp/校验流程，只是 pin 换成 test split，
    并且**必须**是另一个文件 —— 拿 train 当评测集会把 NLL 直接说低。
    """
    pin = EVAL_PINNED
    parquet = _eval_parquet_path()
    txt = _eval_txt_path()
    stamp = _eval_stamp_path()

    if txt.exists() and txt.stat().st_size > 0:
        st = _read_stamp(stamp)
        if (st.get("sha256") == pin["sha256"]
                and st.get("txt_bytes") == txt.stat().st_size):
            CFG["EVAL_TXT"] = str(txt)
            CFG["EVAL_SOURCE"] = f"{pin['config']}/{pin['split']}"
            log(f"评测语料复用：{txt}（{txt.stat().st_size / 1024:.0f} KB，不重下）")
            return txt, {"origin": "cache", "txt_bytes": txt.stat().st_size,
                         "parquet_sha256": pin["sha256"],
                         "source": CFG["EVAL_SOURCE"]}

    _disk_precheck(pin["bytes"], f"评测语料 {pin['path'].split('/')[-1]}")
    meta = _fetch_pinned_blob(pin, parquet, deadline,
                              tag=f"eval/{pin['path'].split('/')[-1]}")
    conv = _parquet_to_txt(parquet, txt, label="评测语料")
    stamp_rec = {"sha256": pin["sha256"], "bytes": pin["bytes"],
                 "txt_bytes": txt.stat().st_size, "repo": pin["repo"],
                 "revision": pin["revision"], "config": pin["config"],
                 "split": pin["split"]}
    clash = sorted(set(stamp_rec) & set(conv))
    if clash:
        raise GuardError(
            f"转换元数据与 stamp 字段重名 {clash} —— 会把 parquet 的 bytes "
            f"覆盖成 txt 大小，provenance 就错了。")
    stamp_rec.update(conv)
    _write_stamp(stamp, **stamp_rec)
    meta.update(conv)
    CFG["EVAL_TXT"] = str(txt)
    CFG["EVAL_SOURCE"] = f"{pin['config']}/{pin['split']}"
    return txt, meta


def ensure_calib_for_isolation(deadline):
    """S6 要做"校准集 vs 评测集"重叠断言，所以 calib 也得在本地。

    S6 单独跑（RUN_EVAL）时 S3 不会跑，calib.txt 可能根本不存在 ——
    那就主动取一次（只有 6.35 MB）。取不到不算致命：返回 None，S6 会
    在结论里标注"无法做隔离断言"。
    """
    p = Path(CFG["CALIB_TXT"]) if CFG.get("CALIB_TXT") else None
    if p and p.exists():
        return p
    try:
        from .stages_imatrix import _ensure_calib_txt
        txt, _meta = _ensure_calib_txt(deadline)
        CFG["CALIB_TXT"] = str(txt)
        CFG.setdefault("CALIB_SOURCE", "wikitext-2-raw/train")
        return txt
    except Exception as e:      # 网络/依赖问题都不该让 S6 直接死
        log(f"!! 取校准语料失败（{type(e).__name__}: {e}）—— 隔离断言将跳过")
        return None


# ---------------------------------------------------------------- 模型输入

def _sidecar_path(gguf):
    gguf = Path(gguf)
    return gguf.with_name(gguf.name + ".provenance.json")


def _hf_stamp(gguf):
    """HF 下载器落在产物旁边的 `.hf.json`（有就说明**这个文件是从 HF 下的**）。"""
    gguf = Path(gguf)
    return _read_json(gguf.with_name(gguf.name + ".hf.json"))


def _local_models(formats):
    """本地 `models/` 里的 GGUF。**来源按 stamp 判定，不按"文件在这儿"猜。**

    v1.1.3 的真实 bug：两个 GGUF 明明是 HF 下载的，`source` 却被一律写成
    `local_s4` —— 于是报告里"本次自己量化"的错觉，而且**完全不读**
    `.hf.json` 里的 size/sha256，也不补 sidecar。
    """
    out = {}
    for f in formats:
        p = _model_path(f)
        if not p.exists():
            continue
        size = p.stat().st_size
        rec = {"path": str(p), "bytes": size, "source": "local_s4",
               "sha256": None, "sha256_verified": False,
               "origin_detail": "local_no_hf_stamp", "hf_stamp": None,
               "sidecar_path": str(_sidecar_path(p))
               if _sidecar_path(p).exists() else None}
        st = _hf_stamp(p)
        if isinstance(st, dict) and st.get("repo") == CFG["HF_MODEL_REPO"]:
            if int(st.get("bytes") or -1) != size:
                raise GuardError(
                    f"{p.name}: 本地大小 {size} B 与下载 stamp 记录的 "
                    f"{st.get('bytes')} B 不符。\n"
                    f"  → 文件被改过或没下完；删掉 {p}（和它的 .hf.json）让 S6 重新下载。")
            rec.update(
                source="hf_cache", sha256=st.get("sha256"),
                sha256_verified=bool(st.get("sha256_verified")),
                origin_detail="hf_download_stamp",
                hf_stamp={k: st.get(k) for k in
                          ("repo", "revision", "bytes", "sha256",
                           "sha256_verified", "fetched_utc", "fetch_auth")})
        out[f] = rec
    return out


def _pick_remote(entries, fmt):
    """从 tree 里挑出这个格式的 .gguf；有歧义就报错，不猜。"""
    want = f"qwen3-8b-{fmt}.gguf".lower()
    cands = [e for e in entries
             if e["path"].lower().endswith(".gguf") and fmt.lower() in e["path"].lower()]
    exact = [e for e in cands if Path(e["path"]).name.lower() == want]
    if len(exact) == 1:
        return exact[0]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    raise GuardError(
        f"HF 仓库里匹配 {fmt} 的 .gguf 不唯一：{[c['path'] for c in cands]}。\n"
        f"  → 把 CFG['HF_MODEL_REVISION'] 钉到一个确定的 commit，或手动把文件放到 "
        f"{MDLS} 下（本地文件优先，不会被覆盖）。")


def _remote_models(formats, token):
    """只探测，不下载：列 tree 并挑出每个格式的文件。"""
    repo = CFG["HF_MODEL_REPO"]
    rev = CFG["HF_MODEL_REVISION"]
    entries = hf.repo_files(repo, rev, token)
    picked = {}
    for f in formats:
        e = _pick_remote(entries, f)
        if e:
            picked[f] = e
    return repo, rev, entries, picked


def resolve_eval_models(deadline, formats=None, fetch=None, probe_only=False):
    """返回 {fmt: {path, bytes, sha256, source, sidecar_path, sidecar}}。

    本地优先（S4 刚跑完的产物，或上一轮 HF 下载后留下的文件），其次 HF。

    v1.2.0 修了三个"本地就有文件"时才会发作的问题：
      1. 以前两个模型都在本地就**直接 early return**，不读 `.hf.json`、
         不补 sidecar、不核远端 oid —— 于是 `source` 被写成 `local_s4`，
         fairness 悄悄降级成"没有 sidecar"。现在：stamp 一定读，sidecar
         一定尝试补齐（本地优先，其次 HF），`EVAL_VERIFY_HASH` 时一定核内容。
      2. `source` 只有三种：`local_s4`（无 HF stamp）/ `hf_cache`（有 stamp，
         大小已核）/ `hf`（本次下载）。"从 HF 下的" 不再冒充"本次量化"。
      3. `sha256_verified` 只代表**内容 sha256 与远端 oid 比对过**，
         不是"文件存在"。

    `probe_only=True` 时**只探测不下载**：本地没有的格式从 tree 拿到 size/oid，
    `source="hf_pending"`、`downloaded=False`。预检用它 —— "先算清楚要下多少、
    要跑多久"，不要为了看一眼就拉 7 GiB。
    """
    formats = tuple(formats or FORMATS)
    fetch = CFG["HF_FETCH_ARTIFACTS"] if fetch is None else fetch
    out = _local_models(formats)
    missing = [f for f in formats if f not in out]
    repo, rev = CFG["HF_MODEL_REPO"], CFG["HF_MODEL_REVISION"]
    token = hf.hf_token()["token"] if fetch else None

    if not missing and not fetch:
        # 纯本地模式：不发任何网络请求，sidecar 缺就明说缺
        for f, rec in out.items():
            rec["sidecar"] = _read_json(rec["sidecar_path"]) if rec["sidecar_path"] else None
            if rec["sidecar"] is None:
                rec["sidecar_missing"] = True
                log(f"!! {f}: 本地没有 provenance sidecar —— "
                    "fairness 只能记『无法判断』，不能当已核验")
        return out

    if not fetch:
        raise GuardError(
            f"本地没有 {missing} 的 GGUF（{MDLS}），而 CFG['HF_FETCH_ARTIFACTS']=False。\n"
            "  → 想从 Hugging Face 上已存的产物直接评测：把 RUN_EVAL 改成 True；\n"
            "  → 或者把 CFG['HF_FETCH_ARTIFACTS'] 设 True；\n"
            "  → 或者把两个 .gguf 手动放到上面那个目录（本地文件优先）。")

    entries = hf.repo_files(repo, rev, token)
    remote_by_name = {Path(e["path"]).name: e for e in entries}

    # ---- 本地文件也要和 tree 交叉核对（大小必核；内容按开关）----
    for f, rec in out.items():
        e = remote_by_name.get(Path(rec["path"]).name)
        if not e:
            rec["tree_match"] = False
            log(f"!! {f}: 本地文件 {Path(rec['path']).name} 不在 "
                f"{repo}@{rev} 的 tree 里 —— 来源无法交叉核对")
            continue
        rec["tree_match"] = True
        rec["remote_path"] = e["path"]
        if int(e["size"]) != int(rec["bytes"]):
            raise GuardError(
                f"{f}: 本地 {rec['bytes']} B ≠ HF tree 的 {e['size']} B。\n"
                f"  → 文件被改过或没下完。删掉 {rec['path']} 让 S6 重新下载。")
        if e.get("sha256") and CFG.get("EVAL_VERIFY_HASH"):
            from .fetch import sha256_of
            got = sha256_of(rec["path"])
            if got != e["sha256"]:
                raise GuardError(
                    f"{f}: 本地 sha256 与 HF tree 的 lfs.oid 不一致。\n"
                    f"  tree {e['sha256']}\n  本地 {got}\n"
                    "  → 文件被改过或没下完。删掉它让 S6 重新下载。")
            rec["sha256"] = got
            rec["sha256_verified"] = True

    if probe_only:
        for f in missing:
            e = _pick_remote(entries, f)
            if e is None:
                raise GuardError(
                    f"{repo}@{rev} 里找不到 {f} 的 .gguf。\n"
                    f"  现有文件：{[x['path'] for x in entries][:20]}")
            out[f] = {"path": str(_model_path(f)), "bytes": e["size"],
                      "sha256": e.get("sha256"), "source": "hf_pending",
                      "downloaded": False, "sha256_verified": False,
                      "remote_path": e["path"], "repo": repo, "revision": rev,
                      "sidecar_path": None, "sidecar": None}
        return out

    for f in missing:
        e = _pick_remote(entries, f)
        if e is None:
            raise GuardError(
                f"{repo}@{rev} 里找不到 {f} 的 .gguf。\n"
                f"  现有文件：{[x['path'] for x in entries][:20]}\n"
                f"  → 用 `python -m trlab --check-hf` 看清单。")
        dst = _model_path(f)
        log(f"HF 输入：{e['path']}（{e['size'] / 2**30:.2f} GiB，"
            f"lfs.oid {str(e.get('sha256'))[:16]}…）→ {dst}")
        meta = hf.fetch_file(repo, e["path"], dst, deadline, revision=rev,
                             token=token, expect=e, tag=f"HF/{f}")
        out[f] = {"path": str(dst), "bytes": meta["bytes"],
                  "sha256": meta["sha256"], "source": "hf",
                  "sha256_verified": meta.get("sha256_verified"),
                  "fetch_auth": meta.get("fetch_auth"),
                  "origin_detail": "downloaded_this_run",
                  "sidecar_path": None}

    # ---- sidecar：本地优先（可能是 S4 刚写的），否则从 HF 取同名小文件 ----
    # 这一步**不能**因为"两个模型都在本地"就跳过（v1.1.3 的 bug）。
    for f in formats:
        rec = out[f]
        sc_local = _sidecar_path(rec["path"])
        if sc_local.exists() and sc_local.stat().st_size > 0:
            rec["sidecar_path"] = str(sc_local)
            rec["sidecar_source"] = "local"
        else:
            rel = Path(rec["path"]).name + ".provenance.json"
            cand = [p for p in remote_by_name if Path(p).name == rel]
            if not cand:
                rec["sidecar_missing"] = True
                log(f"!! {f}: HF 上没有 {rel} —— fairness 只能是『无法判断』")
            else:
                e = remote_by_name[cand[0]]
                log(f"HF sidecar：{cand[0]}（{e['size']} B，非 LFS 也走同源 Bearer）")
                meta = hf.fetch_file(repo, cand[0], sc_local, deadline,
                                     revision=rev, token=token, expect=e,
                                     tag=f"HF/{f}.provenance")
                rec["sidecar_path"] = str(sc_local)
                rec["sidecar_source"] = "hf"
                rec["sidecar_fetch_auth"] = meta.get("fetch_auth")
        rec["sidecar"] = _read_json(rec["sidecar_path"]) if rec["sidecar_path"] else None
    return out


def _verify_local_hashes(models, entries):
    """本地文件也核一遍 sha256 与 tree 的 lfs.oid（4.5 GiB 约 2–4 s）。"""
    from .fetch import sha256_of
    remote = {Path(e["path"]).name: e for e in entries}
    for f, rec in models.items():
        e = remote.get(Path(rec["path"]).name)
        if not e or not e.get("sha256"):
            continue
        got = sha256_of(rec["path"])
        if got != e["sha256"]:
            raise GuardError(
                f"{f}: 本地文件 {rec['path']} 的 sha256 与 HF tree 的 lfs.oid 不一致。\n"
                f"  tree {e['sha256']}\n  本地 {got}\n"
                "  → 文件被改过或没下完。删掉它让 S6 重新下（断点/校验都会重来）。")
        rec["sha256"] = got
        rec["sha256_verified"] = True


# ---------------------------------------------------------------- fairness（声明级）

def fairness_from_sidecars(models):
    """把两个 sidecar 的自述整理成 fairness 记录。

    **这是"声明级"证据，不是"已核验"。** 能真核的只有两件：
      · 两个 sidecar 之间是否一致（同 imatrix / 同 base revision / 同参数）；
      · GGUF 内容是否与 HF tree 的 lfs.oid 一致（`sha256_verified`）。
    sidecar 本身是否诚实，只能靠"它由 S4 在真实运行里写出"这一条来源。
    """
    sc = {f: (m.get("sidecar") or {}) for f, m in models.items()}
    have = {f: s for f, s in sc.items() if s}
    if len(have) < 2:
        return {"verified": False, "verified_level": None, "same_imatrix": None,
                "imatrix_base_consistent": {"ok": None},
                "note": ("缺 provenance sidecar，无法判断两个量化是否同源。"
                         "结论只能当「两个文件各自的 NLL」用，不能当配对证据。")}

    im_shas = {f: s.get("imatrix_sha256") for f, s in have.items()}
    same_imatrix = len({v for v in im_shas.values() if v}) == 1 and all(im_shas.values())
    bases = {(s.get("base_repo"), s.get("base_revision"),
              s.get("base_expected_sha256")) for s in have.values()}
    same_base = len(bases) == 1
    params = {f: s.get("params") for f, s in have.items()}
    same_params = len({str(v) for v in params.values()}) == 1
    kinds = {s.get("imatrix_base_kind") for s in have.values()}
    imatrix_base_kind = kinds.pop() if len(kinds) == 1 else None
    scopes = {s.get("validity_scope") for s in have.values()}
    base_scope = scopes.pop() if len(scopes) == 1 else None

    consistent = bool(same_imatrix and same_base and same_params)
    rec = {
        "verified": False,          # 没有重算，不能叫 verified
        "verified_level": "hf_sidecar_declared",
        "same_imatrix": same_imatrix,
        "same_base": same_base,
        "same_params": same_params,
        "imatrix_sha256": sorted({v for v in im_shas.values() if v}),
        "imatrix_base_kind": imatrix_base_kind,
        "imatrix_base_from_approx": str(imatrix_base_kind).lower() in
                                    ("q8_0", "q4_k_m", "q5_k_m", "q6_k"),
        "base_scope": base_scope,
        "base_repo": (have[next(iter(have))].get("base_repo")),
        "base_revision": (have[next(iter(have))].get("base_revision")),
        "imatrix_base_consistent": {
            "ok": consistent,
            "level": "sidecar",
            "why": ("两个 sidecar 自述一致（同 imatrix sha256 / 同 base repo+revision+"
                    "预期 sha256 / 同量化参数）" if consistent else
                    "两个 sidecar 的自述不一致，配对比较的前提不成立"),
        },
        "sha256_verified": all(m.get("sha256_verified") for m in models.values()),
        "note": ("verified_level=hf_sidecar_declared：数字来自 S4 在真实 Kaggle 运行时"
                 "写下的 sidecar，本次**没有重算**。可核的部分只有 sidecar 之间的一致性"
                 "与 GGUF 内容对 HF tree 的 lfs.oid。"),
    }
    if not consistent:
        log("!! 两个 sidecar 的自述不一致 —— S6 的配对差不能当同源证据")
    return rec


# ---------------------------------------------------------------- 成本 / 资源

def _ram_total_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
        except Exception:
            return None


def eval_device(vram_total_gb=None, model_bytes=None):
    """这次评测到底用 GPU 还是 CPU —— **有 GPU 就用 GPU**，装不下才退并说明理由。

    v1.1.1 起默认 `EVAL_ALLOW_GPU=True`：Kaggle 上开的是 GPU Session，
    让 8B 在 4 核 CPU 上慢慢跑是浪费算力。但"允许"不等于"假装有"：
    没有 CUDA 显存、或单个模型装不下，就退 CPU，并把 `why` 写进结果 ——
    不允许静默降级（静默降级会让"GPU 上跑了 3 小时"变成"其实一直在 CPU 上"）。
    """
    if not CFG.get("EVAL_ALLOW_GPU"):
        return {"device": "cpu", "ngl": 0,
                "why": "EVAL_ALLOW_GPU=False（显式强制纯 CPU）"}
    if not vram_total_gb:
        return {"device": "cpu", "ngl": 0,
                "why": "没有检测到 CUDA 显存（不是 GPU Session，或 torch 不可用）"}
    usable = vram_total_gb - 1.0
    if model_bytes is None:
        return {"device": "gpu", "ngl": 99,
                "why": f"显存 {vram_total_gb:.2f} GiB，可用 {usable:.2f} GiB"}
    need = model_bytes / 2**30 + float(CFG.get("NGL_MARGIN_GB", 2.0))
    if need < usable:
        return {"device": "gpu", "ngl": 99,
                "why": f"权重+余量 {need:.2f} GiB < 可用 {usable:.2f} GiB"}
    return {"device": "cpu", "ngl": 0,
            "why": (f"装不下：权重+余量 {need:.2f} GiB > 可用 {usable:.2f} GiB —— "
                    "这一档会退到 CPU，慢一个量级，结论里必须写清")}


def eval_cost_estimate(n_models, groups, chunks, ctx, allow_gpu=False,
                       vram_total_gb=None, biggest_model_bytes=None):
    """估算评测墙钟。**速率是假设值，不是实测** —— 记录里必须这么写。

    真实耗时只有跑过才知道；这个函数只用来在开跑前挡住"CPU 上必然超预算"
    的配置，而不是承诺任何时间。设备判定与 `eval_device()` 一致：
    有 GPU 且装得下 → 用 GPU 速率，否则 CPU 速率。
    """
    per_model_tokens = int(groups) * int(chunks) * int(ctx)
    total_tokens = per_model_tokens * int(n_models)
    dev = eval_device(vram_total_gb if allow_gpu else None, biggest_model_bytes)
    rate = ASSUMED_TOK_S_GPU if dev["device"] == "gpu" else ASSUMED_TOK_S_CPU
    overhead = PER_INVOCATION_OVERHEAD_SEC * int(groups) * int(n_models)
    wall = total_tokens / max(1e-6, rate) + overhead
    budget = float(CFG.get("EVAL_CPU_MAX_WALL_SEC", 5400))
    return {
        "models": int(n_models), "groups": int(groups), "chunks": int(chunks),
        "ctx": int(ctx), "tokens_per_model": per_model_tokens,
        "tokens_total": total_tokens,
        "device": dev["device"], "device_why": dev["why"],
        "assumed_tok_per_sec": rate,
        "assumed_rate_note": ("假设值（CPU 2.5 / GPU 30 tok/s），**不是实测**。"
                              "llama-perplexity 是批处理（-b 512），实际通常显著快于"
                              "逐 token 生成；真实速率要看 S6 记录里的 elapsed_sec。"),
        "wall_est_sec": round(wall, 1),
        "wall_est_hours": round(wall / 3600, 2),
        "budget_sec": budget,
        # 上限只对 CPU 有意义：GPU 速率与 CPU 差一个量级，拿同一个上限拦会误杀
        "budget_applies": dev["device"] == "cpu",
        "fits_budget": (wall <= budget) or (dev["device"] == "gpu"),
        "gpu_allowed": bool(allow_gpu),
        "vram_total_gb": vram_total_gb,
    }


def eval_preflight(deadline, formats=None, do_download=False):
    """S6/S7 开跑前的体检。`do_download=False` 时**只探测不下载**。

    检查：二进制、输入可得性（本地 or tree 上的 size/oid）、磁盘、RAM/显存、
    语料、成本。任何一项不成立就 fail-closed，并只给一步。

    顺序刻意是"先算账后下载"：7 GiB 的 GGUF 不该在还没确认跑得完之前就拉下来。
    """
    formats = tuple(formats or FORMATS)
    report = {}

    # ---- 0. HF 凭据 / 仓库（fail fast：token 不对就别浪费任何时间）----
    tk = hf.hf_token()
    report["hf"] = {"repo": CFG.get("HF_MODEL_REPO"),
                    "revision": CFG.get("HF_MODEL_REVISION"),
                    "token_present": bool(tk["token"]),
                    "token_source": tk["source"]}

    # ---- 1. 二进制 ----
    missing = []
    for name in ("llama-perplexity", "llama-cli"):
        try:
            llama_bin(name)
        except GuardError:
            missing.append(name)
    if missing:
        raise GuardError(
            f"找不到 {missing} —— 评测要跑 llama.cpp 的二进制。\n"
            "  → v1.1.2 起 `RUN_EVAL=True` 会**自动只编 S2**（pinned 源码 36.6 MiB，"
            "不下载模型、不重量化，约 28 分钟），**不需要**你再开第二个开关；\n"
            "  → 或者设 CFG['LLAMA_BIN'] 指到已编译的 bin 目录；\n"
            "  → 本地开发机：`python -m trlab --preflight-only` 只能做静态检查，"
            "真评测需要可执行的 llama-perplexity。")
    report["binaries"] = "ok"

    # ---- 2. 输入 ----
    models = resolve_eval_models(deadline, formats, probe_only=not do_download)
    need_bytes = sum(m["bytes"] for m in models.values())
    report["models"] = {f: {"path": m["path"], "bytes": m["bytes"],
                            "source": m["source"],
                            "sha256_verified": m.get("sha256_verified")}
                        for f, m in models.items()}

    # ---- 3. 磁盘（只算还需要下的那部分）----
    to_fetch = sum(m["bytes"] for m in models.values()
                   if m["source"] in ("hf", "hf_pending"))
    if to_fetch:
        _disk_precheck(to_fetch, "S6 取量化 GGUF")
    report["disk"] = {"need_fetch_gb": round(to_fetch / 2**30, 2),
                      "models_gb": round(need_bytes / 2**30, 2)}

    # ---- 4. RAM / 显存 / 执行设备 ----
    ram_gb = _ram_total_gb()
    v = vram_gb() or {}
    biggest = max(m["bytes"] for m in models.values()) / 2**30
    dev = eval_device(v.get("total"), int(biggest * 2**30))
    report["resources"] = {
        "ram_total_gb": ram_gb,
        "vram_total_gb": v.get("total"),
        "biggest_model_gb": round(biggest, 2),
        "gpu_allowed": bool(CFG.get("EVAL_ALLOW_GPU")),
        "device": dev["device"],
        "device_why": dev["why"],
    }
    if dev["device"] == "cpu" and CFG.get("EVAL_ALLOW_GPU") and v.get("total"):
        # 有 GPU 但装不下：不是错误，但必须**大声**说，否则会变成"以为在用 GPU"
        log(f"!! 有显存但装不下这个模型 → 退 CPU：{dev['why']}")
    log(f"执行设备：{dev['device']}（{dev['why']}）")
    if ram_gb and biggest + 2.0 > ram_gb:
        raise GuardError(
            f"RAM 不够：最大模型 {biggest:.2f} GiB + 余量 2 GiB > 总 RAM {ram_gb:.2f} GiB。\n"
            "  → 唯一的最简下一步：换内存更大的机器，或只评测较小的那个格式。")

    # ---- 5. 语料（0.73 MB，可以真取）----
    eval_txt, emeta = _ensure_eval_txt(deadline)
    report["eval_corpus"] = {"path": str(eval_txt),
                             "source": CFG.get("EVAL_SOURCE"),
                             "bytes": eval_txt.stat().st_size}
    calib = ensure_calib_for_isolation(deadline)
    report["calib_corpus"] = {"path": str(calib) if calib else None,
                              "available": bool(calib)}

    # ---- 6. 成本 ----
    cost = eval_cost_estimate(
        n_models=len(models), groups=CFG["NLL_GROUPS"],
        chunks=CFG["NLL_CHUNKS_PER_GROUP"], ctx=CFG["NLL_CTX"],
        allow_gpu=CFG["EVAL_ALLOW_GPU"], vram_total_gb=v.get("total"),
        biggest_model_bytes=int(biggest * 2**30))
    report["cost"] = cost
    if not cost["fits_budget"]:
        raise GuardError(
            f"按当前规模，CPU 评测估算要 {cost['wall_est_hours']} 小时"
            f"（{cost['tokens_total']} token / {cost['assumed_tok_per_sec']} tok/s，"
            f"**假设速率不是实测**），超过 {cost['budget_sec'] / 3600:.1f} 小时上限。\n"
            "  → 三选一，别硬跑：\n"
            "    1. 缩小 pilot：CFG['NLL_GROUPS'] / CFG['NLL_CTX'] / "
            "CFG['NLL_CHUNKS_PER_GROUP'] 调小"
            "（配对比较只要两个模型同设置，不需要大 ctx）；\n"
            "    2. 开 GPU Session（Kaggle 上 Settings → Accelerator），"
            "代码默认就会用 GPU；或显式 CFG['EVAL_ALLOW_GPU']=True；\n"
            "    3. 换更快的机器，或只跑 S7（质量抽查，token 量小两个数量级）。")
    log(f"评测成本估算：{cost['tokens_total']} token，"
        f"{cost['wall_est_hours']} h（设备 {cost['device']}，"
        f"假设 {cost['assumed_tok_per_sec']} tok/s，不是实测）")
    return report


# ---------------------------------------------------------------- 轻量预检

def lightweight_eval_preflight(deadline, formats=None):
    """`--preflight-eval`：**不编译、不下 7 GiB、不需要 GPU** 的评测链路体检。

    真实 Kaggle 上连续三次失败都不是"模型/GPU"的问题，而是下载链路与语料口径：
      · resolve URL 多了 `/models/` → 404
      · `expect` 字段名 `size`/`bytes` 不一致 → KeyError
      · 私仓 2 KB 的 provenance.json 没带 Bearer → 401
      · 校准/评测集 1 行重叠 → 整轮 S6 归零
    这四件事**都不需要**下载模型或编译就能发现。所以这里把它们做成 1 分钟级
    的体检：Range 0-0 探大文件、真下 2 KB 小文件、真跑 holdout 构建。
    """
    formats = tuple(formats or FORMATS)
    repo = CFG["HF_MODEL_REPO"]
    rev = CFG["HF_MODEL_REVISION"]
    rep = {"stage": "PREFLIGHT_EVAL", "repo": repo, "revision": rev,
           "checks": {}, "ok": False, "warnings": []}

    # ---- 1. 凭据 / 仓库 ----
    auth = hf.check_access(repo, rev, use_cache=False)
    rep["checks"]["hf_access"] = {k: auth.get(k) for k in
                                  ("ok", "status", "http_status", "n_files",
                                   "token_source", "error")}
    if not auth.get("ok"):
        rep["why"] = f"HF 授权检查失败（{auth.get('status')}）"
        _write_preflight(rep)
        return rep

    # ---- 2. tree + 大文件 Range 探测 ----
    token = hf.hf_token()["token"]
    entries = hf.repo_files(repo, rev, token)
    by_name = {Path(e["path"]).name: e for e in entries}
    probes = {}
    for f in formats:
        e = _pick_remote(entries, f)
        if e is None:
            probes[f] = {"ok": False, "why": f"tree 里找不到 {f} 的 .gguf"}
            continue
        probes[f] = hf.probe_remote_file(repo, e["path"], deadline,
                                        revision=rev, token=token, expect=e)
    rep["checks"]["gguf_probe"] = probes

    # ---- 3. sidecar：真下 2 KB 小文件（这才是 401 的现场）----
    side_dir = WORK / "preflight" / "sidecars"
    side = {}
    for f in formats:
        rec = probes.get(f) or {}
        rel = Path(rec.get("path") or f"qwen3-8b-{f}.gguf").name + ".provenance.json"
        e = by_name.get(rel)
        if not e:
            side[f] = {"ok": False, "why": f"tree 里没有 {rel}"}
            continue
        try:
            meta = hf.fetch_file(repo, e["path"], side_dir / rel, deadline,
                                 revision=rev, token=token, expect=e,
                                 tag=f"preflight/{rel}", force=True)
            side[f] = {"ok": True, "bytes": meta["bytes"],
                       "fetch_auth": meta.get("fetch_auth"),
                       "sha256_verified": meta.get("sha256_verified"),
                       "path": meta["path"]}
        except Exception as e2:          # noqa: BLE001 - 预检要把原因原样带回
            side[f] = {"ok": False, "why": f"{type(e2).__name__}: {str(e2)[:300]}"}
    rep["checks"]["sidecar_download"] = side

    # ---- 4. 语料 + clean holdout（7 MB，真实）----
    try:
        eval_txt, emeta = _ensure_eval_txt(deadline)
        calib = ensure_calib_for_isolation(deadline)
        if calib is None:
            rep["checks"]["corpus"] = {"ok": False, "why": "取不到校准语料"}
        else:
            from . import isolation
            iso = isolation.build_holdout(calib, eval_txt,
                                         WORK / "preflight" / "isolation",
                                         cfg=CFG,
                                         eval_label=CFG.get("EVAL_SOURCE"))
            rep["checks"]["corpus"] = {
                "ok": True,
                "eval": iso["eval_raw"],
                "calib": iso["calib"],
                "overlap_raw_paras": iso["overlap_raw"]["paras"],
                "overlap_details": iso["overlap_raw"]["details"][:5],
                "holdout": iso["holdout"],
                "actual_imatrix_calib_overlap":
                    iso["actual_imatrix_calib_overlap"]["value"],
            }
    except Exception as e:              # noqa: BLE001
        rep["checks"]["corpus"] = {"ok": False,
                                   "why": f"{type(e).__name__}: {str(e)[:400]}"}

    # ---- 5. 二进制（缺了会由 run() 自动只编 S2，这里只提示）----
    miss = []
    for name in ("llama-perplexity", "llama-cli"):
        try:
            llama_bin(name)
        except GuardError:
            miss.append(name)
    rep["checks"]["binaries"] = {"ok": not miss, "missing": miss}
    if miss:
        rep["warnings"].append(f"缺 {miss}：正常 run_eval 会自动只编 S2（约 28 分钟）")

    rep["ok"] = (auth.get("ok")
                 and all(v.get("ok") for v in probes.values())
                 and all(v.get("ok") for v in side.values())
                 and (rep["checks"].get("corpus") or {}).get("ok"))
    _write_preflight(rep)
    return rep


def _write_preflight(rep):
    try:
        write_json(OUT / "PREFLIGHT_EVAL.json", rep)
    except Exception:
        pass
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str)[:4000])

