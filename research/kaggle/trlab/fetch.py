"""下载缓存：断点续传、sha256/etag 强校验、原子落地、磁盘预检。

所有联网只经过这里；不用 huggingface_hub（免 ~/.cache 第二份完整副本）。
"""

from .config import BUILD, CFG, DL_DIR, S2_PINNED, S3_PINNED, SRC_CACHE, WORK
from .errors import BudgetExceeded, GuardError
from .core import _read_json, disk_gb, log, write_json

import hashlib, os, shutil, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

def _disk_precheck(extra_bytes, tag, reserve_gb=None, root=None):
    """下载**之前**的磁盘预检（v0.4 / D-4）。

    extra_bytes 是这次操作将要新占用的字节。判据是
    `free - extra >= reserve`，reserve 默认取 CFG['MIN_FREE_DISK_GB']。
    注意 Q8_0 边下边写 `.partial`、验完 `os.replace`，所以**同时只有一份**，
    不需要按"两份"预留。

    v0.5：`root` 允许指定别的文件系统（例如 working 之外的暂存盘）。
    不传就按 working 算 —— working 是唯一有 20 GB 硬配额的地方。
    """
    reserve = CFG["MIN_FREE_DISK_GB"] if reserve_gb is None else float(reserve_gb)
    root = Path(root) if root else WORK
    d = disk_gb(root)
    need = float(extra_bytes) / 2**30
    if d["free"] - need < reserve:
        where = "working" if str(root) == str(WORK) else str(root)
        raise GuardError(
            f"{tag}: 磁盘不够（{where}）。当前剩余 {d['free']:.2f} GiB，这次要新增 "
            f"{need:.2f} GiB，还需要保留 {reserve:.2f} GiB 余量。\n"
            f"  → 预期需要 {d['free']:.2f} − {need:.2f} ≥ {reserve:.2f}，"
            f"实际 {d['free'] - need:.2f}。\n"
            f"  → 处置：删掉 {DL_DIR} 下不用的文件；或把 CFG['S3_DELETE_Q8_AFTER'] "
            f"保持 True（默认）让上一轮的 Q8_0 已被回收；或手动清 "
            f"{BUILD} 里的中间产物。")



def sha256_of(path, deadline=None, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            if deadline and time.time() > deadline:
                return "aborted:deadline"
    return h.hexdigest()



def partial_path(dst):
    dst = Path(dst)
    return dst.with_name(dst.name + ".partial")



def _http_download(url, dst, deadline, expect_bytes=None, timeout=60, chunk=1 << 20):
    """把 url 下到 dst（先写 .partial），边下边算 sha256。

    只用标准库 urllib —— 不依赖 curl/wget，Kaggle 镜像里一定可用。
    任何网络异常都转成 GuardError，消息里直接写"检查 Settings → Internet = On"，
    不把 URLError 的原始栈丢给用户。
    """
    import socket
    import urllib.error
    import urllib.request

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    if part.exists():
        part.unlink()

    h = hashlib.sha256()
    got = 0
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trlab-kaggle/0.3"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = int(r.headers.get("Content-Length") or 0)
            if expect_bytes and total and total != expect_bytes:
                raise GuardError(
                    f"服务器报的 Content-Length={total}，与预登记的 {expect_bytes} 不符。"
                    "上游文件可能被替换，拒绝继续。")
            with open(part, "wb") as f:
                while True:
                    if time.time() > deadline:
                        raise BudgetExceeded(f"下载超预算：已下 {got / 2**20:.1f} MiB")
                    b = r.read(chunk)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    got += len(b)
                    if got % (8 << 20) < chunk:
                        rate = got / 2**20 / max(1e-6, time.time() - t0)
                        log(f"    已下载 {got / 2**20:.1f} MiB"
                            + (f" / {total / 2**20:.1f}" if total else "")
                            + f"  {rate:.1f} MiB/s")
    except (GuardError, BudgetExceeded):
        if part.exists():
            part.unlink()
        raise
    except (urllib.error.HTTPError, urllib.error.URLError,
            socket.timeout, OSError) as e:
        if part.exists():
            part.unlink()
        raise GuardError(
            f"下载源码失败：{type(e).__name__}: {e}\n"
            "  → 检查 Notebook 右侧 Settings 里 Internet 是否为 On；\n"
            "  → 或改用离线方式：把源码挂到 /kaggle/input，设 CFG['LLAMA_SRC'] 到含 "
            "CMakeLists.txt 的那一层。")
    os.replace(part, dst)
    return {"bytes": got, "sha256": h.hexdigest(), "path": str(dst),
            "seconds": round(time.time() - t0, 1)}



def _safe_extract(tar_path, dest):
    """解压 tarball 到 dest。拒绝绝对路径、`..`、链接与非普通文件。

    llama.cpp 的归档里全是普通文件；出现别的就是有人动过包，直接拒绝而不是
    "尽量解压"。
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise GuardError(f"归档里有可疑路径 {name!r}，拒绝解压")
            if m.issym() or m.islnk():
                raise GuardError(f"归档里有链接 {name!r}，拒绝解压")
            if not (m.isfile() or m.isdir()):
                raise GuardError(f"归档里有非普通文件 {name!r}，拒绝解压")
            tgt = (dest / name).resolve()
            if tgt != dest and dest not in tgt.parents:
                raise GuardError(f"归档条目越界 {name!r}，拒绝解压")
        try:
            tf.extractall(dest, filter="data")      # Python ≥ 3.12
        except TypeError:
            tf.extractall(dest)
    return dest



def _fetch_pinned_source(deadline):
    """下载 → 校验 sha256 → 安全解压 → 校验顶层目录名。返回 (src_dir, meta)。

    缓存优先级（重复 Run All 不会重下）：
      1. 已解压 + stamp 里的 sha256 与当前 pin 一致 → 完全离线复用；
      2. 已下载的 tarball digest 一致 → 只解压，不重下；
      3. 否则下载。
    """
    pin = S2_PINNED
    src_dir = SRC_CACHE / pin["dir"]
    stamp = SRC_CACHE / "pinned_source.json"
    tar_path = DL_DIR / f"llama.cpp-{pin['commit'][:12]}.tar.gz"

    st = _read_json(stamp)
    if (isinstance(st, dict) and st.get("sha256") == pin["sha256"]
            and (src_dir / "CMakeLists.txt").exists()):
        log(f"源码缓存命中：{src_dir}（不联网）")
        return src_dir, {"origin": "cache", "sha256_ok": True,
                         "url": pin["url"], "dir": pin["dir"],
                         "archive_sha256": st.get("sha256"),
                         "archive_bytes": st.get("bytes"),
                         "archive_cached": True}

    meta = None
    if tar_path.exists() and tar_path.stat().st_size == pin["bytes"]:
        d = sha256_of(tar_path, deadline)
        if d == pin["sha256"]:
            log(f"复用已下载的源码包 {tar_path.name}（digest 一致，不重下）")
            meta = {"origin": "cached_archive", "archive_sha256": d,
                    "archive_bytes": pin["bytes"]}
        else:
            log(f"!! 已下载的源码包 digest 不符（{d[:12]}…），删掉重下")
            tar_path.unlink()

    if meta is None:
        if not CFG["S2_AUTO_FETCH_SOURCE"]:
            raise GuardError(
                "需要下载源码，但 CFG['S2_AUTO_FETCH_SOURCE'] 是 False。"
                "把顶部 RUN_S2 改成 True 即可。")
        log(f"下载固定 commit 源码：{pin['url']}")
        log(f"  预期 {pin['bytes'] / 2**20:.1f} MiB，sha256 {pin['sha256'][:16]}…")
        d = _http_download(pin["url"], tar_path, deadline,
                           expect_bytes=pin["bytes"])
        if d["sha256"] != pin["sha256"]:
            tar_path.unlink()
            raise GuardError(
                "源码包 sha256 不匹配，拒绝解压：\n"
                f"  预期 {pin['sha256']}\n"
                f"  实际 {d['sha256']}\n"
                "  → 上游文件可能被替换，或网络返回了错误页面（比如登录页）。"
                "不要继续。")
        log(f"  sha256 校验通过（{d['bytes'] / 2**20:.1f} MiB，{d['seconds']} s）")
        meta = {"origin": "downloaded", "archive_sha256": d["sha256"],
                "archive_bytes": d["bytes"]}

    SRC_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = SRC_CACHE / f".extract-{os.getpid()}-{int(time.time())}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    _safe_extract(tar_path, tmp)
    inner = tmp / pin["dir"]
    if not (inner / "CMakeLists.txt").exists():
        got = sorted(p.name for p in tmp.iterdir())[:5]
        shutil.rmtree(tmp, ignore_errors=True)
        raise GuardError(
            f"归档顶层目录不是预期的 {pin['dir']}（实际 {got}）。"
            "sha256 对得上但目录名不符，说明预登记信息本身有问题，不要继续。")
    if src_dir.exists():
        shutil.rmtree(src_dir, ignore_errors=True)
    os.replace(inner, src_dir)
    shutil.rmtree(tmp, ignore_errors=True)

    write_json(stamp, {"sha256": pin["sha256"], "bytes": pin["bytes"],
                       "commit": pin["commit"], "dir": pin["dir"],
                       "url": pin["url"],
                       "extracted_at": datetime.now(timezone.utc).isoformat(
                           timespec="seconds")})
    log(f"源码就位：{src_dir}")
    return src_dir, {"origin": meta["origin"], "sha256_ok": True,
                     "url": pin["url"], "dir": pin["dir"],
                     "archive_sha256": pin["sha256"],
                     "archive_bytes": pin["bytes"],
                     "archive_cached": meta["origin"] == "cached_archive"}



def _read_stamp(p):
    st = _read_json(p)
    return st if isinstance(st, dict) else {}



def _write_stamp(p, **kw):
    kw["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return write_json(p, kw)



def _stream_blob(url, path, offset, hasher, deadline, tag, expect_bytes,
                 headers=None):
    """从 offset 处继续把 url 流式追加写进 path。

    返回 (http_status, written_now, total, etag)。任何网络异常都原样抛出，
    由调用方决定是保留 `.partial` 还是重来 —— 这里不擅自删文件。

    `headers` 用于私有仓库的 Authorization（v1.0.2）。注意 HF 的 resolve URL
    会 302 到带签名的 CDN 地址，那个地址**不需要**也不该带 Authorization；
    调用方应先用 `hf.resolve_signed()` 换到签名地址再进来（见 trlab/hf.py）。
    """
    import urllib.request

    hdrs = {"User-Agent": "trlab-kaggle/0.4"}
    if headers:
        hdrs.update({k: v for k, v in headers.items() if v})
    if offset:
        hdrs["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=60) as r:
        status = getattr(r, "status", None) or r.getcode()
        total = int(r.headers.get("Content-Length") or 0)
        etag = (r.headers.get("x-linked-etag") or "").strip().strip('"')
        written, t0, last = 0, time.time(), 0.0
        with open(path, "ab") as f:
            while True:
                if time.time() > deadline:
                    raise BudgetExceeded(
                        f"{tag}: 下载超预算，已写 {(offset + written) / 2**30:.2f} GiB"
                        f" / {expect_bytes / 2**30:.2f} GiB")
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                hasher.update(b)
                written += len(b)
                if written - last >= (64 << 20):
                    last = written
                    f.flush()
                    rate = written / 2**20 / max(1e-6, time.time() - t0)
                    log(f"    {tag}: {(offset + written) / 2**30:.2f}"
                        f" / {expect_bytes / 2**30:.2f} GiB  {rate:.0f} MiB/s")
        return status, written, total, etag



def _fetch_pinned_blob(pin, dst, deadline, tag, fetch_url=None, auth_headers=None):
    """下载 pin 的单个文件到 dst：断点续传 + 完整性校验（v0.4 / D-4、D-5）。

    四条硬规则：

    1. **磁盘上同时只有一份完整副本。** 边写 `<name>.partial` 边算 sha256，
       验完 `os.replace` 成正式文件。**不写 HF 缓存目录**（不用
       `huggingface_hub`），所以不存在"缓存一份 + working 一份"。
    2. **断点续传。** `<name>.partial.json` 记 `bytes_done` 与已写内容的前缀
       哈希；重跑时发 `Range: bytes=<done>-`。服务器回 200（忽略 Range）就
       从头下，回 416 也从头下 —— 都在日志里说明，不静默。
    3. **双源 digest。** HF 的 resolve URL 会回 `x-linked-etag`（= LFS oid =
       文件内容的 sha256）。它与 `pin['sha256']` 不一致就拒收：上游换过文件。
       拿不到这个头时不拦，只靠预登记 sha256，并在日志里说明。
    4. **完成后 size 与 sha256 都必须等于预登记值**，否则删掉并报错，
       绝不留一个"看着像 downloads 完"的文件。
    """
    import urllib.error
    import socket

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    side = dst.with_name(dst.name + ".partial.json")
    expect_bytes, expect_sha = int(pin["bytes"]), (pin.get("sha256") or None)
    t0 = time.time()

    def _save_sidecar(done, digest):
        _write_stamp(side, url=pin["url"], bytes_done=done,
                     sha256_prefix=digest, expect_bytes=expect_bytes,
                     expect_sha256=expect_sha)

    # ---- 续传状态：sidecar 与 pin 一致、且 .partial 的前缀哈希自洽才敢接着写 ----
    hasher = hashlib.sha256()
    offset = 0
    st = _read_stamp(side)
    if (part.exists() and st.get("url") == pin["url"]
            and st.get("expect_bytes") == expect_bytes
            and 0 < int(st.get("bytes_done") or 0) < expect_bytes):
        cand = int(st["bytes_done"])
        try:
            with open(part, "rb") as f:
                while True:
                    b = f.read(8 << 20)
                    if not b:
                        break
                    hasher.update(b)
            if part.stat().st_size != cand:
                raise ValueError(
                    f".partial 实际 {part.stat().st_size} B，sidecar 记 {cand} B")
            if st.get("sha256_prefix") and hasher.hexdigest() != st["sha256_prefix"]:
                raise ValueError("sidecar 记录的前缀哈希与 .partial 内容不符")
            offset = cand
            log(f"{tag}: 断点续传，从 {offset / 2**30:.2f} / "
                f"{expect_bytes / 2**30:.2f} GiB 继续")
        except Exception as e:
            log(f"{tag}: .partial 不可信（{e}）→ 从头下载")
            part.unlink(missing_ok=True)      # 必须删：否则会把垃圾拼到新内容后面
            offset, hasher = 0, hashlib.sha256()
    elif part.exists():
        log(f"{tag}: 有 .partial 但没有可用的 sidecar → 从头下载")
        part.unlink()

    try:
        status, written, total, etag = _stream_blob(
            fetch_url or pin["url"], part, offset, hasher, deadline, tag,
            expect_bytes, headers=auth_headers)
        if offset and status != 206:
            # 服务器不支持/忽略了 Range：只能从头下，不能把 200 的全文接到半份后面
            log(f"{tag}: 对 Range 返回 {status} 而不是 206 → 从头下载")
            part.unlink()
            offset, hasher = 0, hashlib.sha256()
            status, written, total, etag = _stream_blob(
                fetch_url or pin["url"], part, 0, hasher, deadline, tag,
                expect_bytes, headers=auth_headers)
    except (GuardError, BudgetExceeded):
        _save_sidecar(_partial_size(part), hasher.hexdigest())
        raise
    except (urllib.error.HTTPError, urllib.error.URLError,
            socket.timeout, OSError) as e:
        # 保留 .partial + sidecar：重跑会从这里续，不必把 8 GiB 重下一遍
        _save_sidecar(_partial_size(part), hasher.hexdigest())
        if str(tag).startswith("S3"):
            tail = (f"  → 若反复失败：检查 Settings → Internet 是否为 On；"
                    f"或把 {S3_PINNED['repo']} 的 Q8_0 手动挂成 Input 并设 "
                    f"CFG['IMATRIX_BASE'] 指向它。")
        else:
            tail = ("  → 若反复失败：检查网络 / Settings → Internet；"
                    "私有 HF 仓库还要确认 token 仍然有效（重跑会从这里续传）。")
        raise GuardError(
            f"{tag}: 下载失败（{type(e).__name__}: {e}）。\n"
            f"  已保留断点 {part}（{_partial_size(part) / 2**30:.2f} GiB），"
            f"重跑会从这里续传，不用从零开始。\n" + tail)

    got = part.stat().st_size
    digest = hasher.hexdigest()

    if total and total != expect_bytes:
        part.unlink()
        side.unlink(missing_ok=True)
        raise GuardError(
            f"{tag}: 服务器报 Content-Length={total}，与预登记 {expect_bytes} 不符。"
            "上游文件可能被替换，拒绝继续。")

    if got != expect_bytes:
        _save_sidecar(got, digest)
        raise GuardError(
            f"{tag}: 下载得到 {got} B，预登记 {expect_bytes} B，差 "
            f"{expect_bytes - got} B。已保留断点，重跑会续传。")

    if etag and expect_sha and etag != expect_sha:
        part.unlink()
        side.unlink(missing_ok=True)
        raise GuardError(
            f"{tag}: 响应头 x-linked-etag 与预登记 sha256 不一致。\n"
            f"  预期 {expect_sha}\n  实际 {etag}\n"
            "上游文件被替换过，拒绝继续。")

    if expect_sha and digest != expect_sha:
        part.unlink()
        side.unlink(missing_ok=True)
        raise GuardError(
            f"{tag}: sha256 不匹配，已删除不完整的文件。\n"
            f"  预期 {expect_sha}\n  实际 {digest}\n"
            "上游文件被替换、或网络返回了错误页面。不要继续。")

    # 校验全部通过才落地。os.replace 在同一文件系统上是原子的，
    # 且不产生第二份完整副本。
    os.replace(part, dst)
    side.unlink(missing_ok=True)
    log(f"{tag}: 下载并校验通过 {got / 2**30:.2f} GiB，"
        f"sha256 {digest[:16]}…，{time.time() - t0:.0f} s")
    return {"origin": "downloaded", "bytes": got, "sha256": digest,
            "seconds": round(time.time() - t0, 1), "path": str(dst),
            "etag_checked": bool(etag)}



def _partial_size(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0
