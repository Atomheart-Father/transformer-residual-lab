"""Hugging Face 私有仓库：凭据查找、tree 校验、断点续传下载、显式 opt-in 上传。

四条硬约束（来自 Issue #5 的授权，不是建议）：

1. **token 只从环境变量 / 平台 secret / HF 官方缓存文件里取，绝不打印。**
   日志里只有"来源名字"（`HF_TOKEN` / `kaggle_secret` / `hf_cache_file`），
   没有值、没有前缀、没有长度。也不要求用户把 token 写进 notebook / git / issue。
2. **下载前先要证据。** 先取 tree API 的 `size` 与 `lfs.oid`（= 内容 sha256）
   当期望值，再走 `_fetch_pinned_blob` 的断点续传 + 双源校验。拿不到 oid 的
   非 LFS 小文件只按 size 校验，并在记录里标 `sha256_verified=False`。
3. **上传是单独 opt-in，默认关。** 只传调用方显式列出的文件；传完复核远端
   size 与 LFS sha256。不传 token，不传 working 里的其它东西。
4. **不 `pip install -U huggingface_hub`。** 上传只在已安装时用；Kaggle 镜像里
   `tokenizers` 与新版 hub 有约束冲突（用户真实撞过），所以缺库就明确报错，
   让用户自己决定装哪个版本。

私有仓库的 302 会跳到**带签名的 CDN 地址**，那个地址不需要 Authorization
（带上反而可能 403），所以这里先 `resolve_signed()` 换地址，再交给下载器。
"""

from .config import CFG
from .errors import GuardError, HFHttpError
from .core import _read_json, log, write_json
from .fetch import _fetch_pinned_blob, sha256_of

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

HF_BASE = "https://huggingface.co"
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

# tree API 的分页上限。**不能和 expand=true 同时用**（Hub 会 400，实测见 repo_files）。
_TREE_PAGE_LIMIT = 1000
# HTTP 错误正文最多读这么多字节，够看清 "Invalid limit ..." 这类原因，又不至于把
# 一大段 HTML 写进日志。
_ERROR_BODY_MAX = 2048

# check_access 的进程内缓存：key = (repo, revision, 有没有 token, token 来源)。
# 只缓存"能不能读"，不缓存任何文件内容，也不缓存 token 本身。
_ACCESS_CACHE = {}
_TOKEN_FIELDS = ("token", "value", "secret")

# HTTP 状态码 → check_access 的 status 词。**唯一分类来源**，不靠字符串匹配。
_STATUS_BY_HTTP = {400: "bad_request", 401: "unauthorized", 403: "unauthorized",
                   404: "not_found", 429: "rate_limited",
                   500: "rate_limited", 502: "rate_limited",
                   503: "rate_limited", 504: "rate_limited"}


# ---------------------------------------------------------------- 凭据

def _kaggle_secret(names=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
    """Kaggle 的 Add-ons → Secrets。没装/没配就返回 None，不吵。"""
    try:
        from kaggle_secrets import UserSecretsClient      # 只在 Kaggle 镜像里有
    except Exception:
        return None
    try:
        cli = UserSecretsClient()
    except Exception:
        return None
    for n in names:
        try:
            v = cli.get_secret(n)
        except Exception:
            continue
        if v:
            return v.strip()
    return None


def _hf_cache_token():
    """HF 官方 CLI 的登录态（`huggingface-cli login` 写的那个文件）。"""
    p = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "token"
    try:
        v = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return v or None


def hf_token(env=None, allow_cache=True):
    """按优先级找 token。返回 {"token": str|None, "source": str|None}。

    **调用方不许把 token 写进日志/JSON/异常消息**；要用就只用 source。
    """
    env = os.environ if env is None else env
    for name in TOKEN_ENV_VARS:
        v = (env.get(name) or "").strip()
        if v:
            return {"token": v, "source": f"env:{name}"}
    v = _kaggle_secret()
    if v:
        return {"token": v, "source": "kaggle_secret"}
    if allow_cache:
        v = _hf_cache_token()
        if v:
            return {"token": v, "source": "hf_cache_file"}
    return {"token": None, "source": None}


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def redact(text, token):
    """兜底：万一底层库把 token 带进异常消息，落盘/打印前抹掉。"""
    s = str(text)
    if token:
        s = s.replace(token, "***REDACTED***")
    return s


def _no_token_hint(repo):
    return (
        f"访问 {repo} 需要 Hugging Face token（仓库是 Private）。\n"
        "  → 三选一，**不要把 token 写进 notebook / git / issue**：\n"
        "    1. 环境变量：`export HF_TOKEN=hf_...`（Kaggle 上就是 Add-ons → Secrets，\n"
        "       名字用 HF_TOKEN，勾选 attach 到本 notebook）；\n"
        "    2. 本机 `huggingface-cli login` 一次，之后读官方缓存；\n"
        "    3. 或者由你把仓库改成 Public —— 那样匿名就能下，代码不用改。\n"
        "       （可见性是仓库所有者的决定；本代码**不会**替你改。）\n"
        "  → 权限只需要 read（下载）/ write（上传）这一项，细粒度 token 就够。")


# ---------------------------------------------------------------- HTTP 基座

class _NoRedirect:
    """urllib 的 no-redirect opener 工厂（resolve 时要拿到 302 的 Location）。"""

    @staticmethod
    def build():
        import urllib.request

        class _H(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None       # 让 urllib 抛 HTTPError，我们从 headers 里取 Location

        return urllib.request.build_opener(_H)


def _h(headers, name):
    """大小写不敏感地取 header（urllib 给的 dict 键大小写随服务器）。"""
    if not headers:
        return ""
    low = name.lower()
    for k, v in headers.items():
        if str(k).lower() == low:
            return v
    return ""


def _http_get(url, headers=None, timeout=None, follow=True, read_body=True):
    """返回 (status, headers_dict, body_bytes)。HTTP 错误码原样返回，不抛。

    网络层异常（DNS/超时/断网）转成 GuardError —— 消息里直接说"检查网络/
    Internet 开关"，不把 URLError 的栈丢给用户。

    `read_body=False` 用于 resolve（只要 302 的 Location）：否则一个 200 的
    4.5 GiB 响应会被整个读进内存。

    **HTTP 错误也会带回正文**（v1.1.3）：只读前 2 KiB，且先抹掉 token。
    否则 400 只能报一句"HTTP 400"，无法判断是参数错还是权限错。
    """
    import socket
    import urllib.error
    import urllib.request

    hdrs = {"User-Agent": "trlab-kaggle/1.0"}
    if headers:
        hdrs.update({k: v for k, v in headers.items() if v})
    req = urllib.request.Request(url, headers=hdrs)
    opener = urllib.request.build_opener() if follow else _NoRedirect.build()
    try:
        with opener.open(req, timeout=timeout or CFG.get("HF_TIMEOUT_SEC", 60)) as r:
            return r.status, dict(r.headers), (r.read() if read_body else b"")
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()[:_ERROR_BODY_MAX]
        except Exception:
            raw = b""
        return e.code, dict(e.headers or {}), _sanitize_body(raw, headers)
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        raise GuardError(
            f"连不上 Hugging Face（{type(e).__name__}: {e}）。\n"
            "  → Kaggle 上先确认 Settings → Internet = On；\n"
            "  → 本地确认网络/代理；\n"
            "  → 若只是不想联网，可以把 GGUF 手动放到 CFG['BASE_GGUF'] 同级的\n"
            "     `models/` 目录里，S6/S7 会优先用本地文件。")


def _sanitize_body(raw, headers=None):
    """错误正文落盘前：抹 token、抹签名 URL 的查询串、压成一行。"""
    try:
        s = raw.decode("utf-8", "replace")
    except Exception:
        return b""
    tok = ""
    for k, v in (headers or {}).items():
        if str(k).lower() == "authorization":
            tok = str(v).split()[-1]
    s = redact(s, tok)
    s = re.sub(r"(https?://[^\s\"']+?)[?&](?:[^\s\"']*)", r"\1?<redacted-query>", s)
    s = " ".join(s.split())
    return s[:_ERROR_BODY_MAX].encode("utf-8")


def _api_error(status, repo, what, body=b"", token=None):
    """把 HTTP 状态码翻成人话。分类必须来自状态码，不靠猜。"""
    detail = ""
    if body:
        try:
            detail = redact(body.decode("utf-8", "replace"), token).strip()
        except Exception:
            detail = ""
    detail = " ".join(detail.split())[:300]
    tail = f"\n  · 服务器原话：`{detail}`" if detail else ""
    if status == 400:
        return HFHttpError(
            status,
            f"{what} 返回 **HTTP 400** —— 请求参数被 Hub API 拒绝，"
            "这**不是** token / 权限问题，也不是你的 Secrets 配错了。"
            f"{tail}\n"
            "  → 这是本工具的问题（v1.1.2 的 tree URL 用了 `expand=true` + "
            "`limit=1000`，Hub 只允许 expand 时 ≤50）。v1.1.3 已改成"
            "`recursive=1&limit=1000`（不带 expand）。\n"
            "  → 如果你看到这一条，说明跑的 notebook 还是旧版：换最新 pinned "
            "notebook 重跑即可，**不要**重建 token。",
            detail=detail)
    if status in (401, 403):
        return HFHttpError(
            status,
            f"{what} 被拒绝（HTTP {status}）。{tail}\n" + _no_token_hint(repo),
            detail=detail)
    if status == 404:
        return HFHttpError(
            status,
            f"{what} 返回 404：仓库或 revision 不存在。{tail}\n"
            f"  · 私有仓库在**没有 token** 时通常返回 401（不是 404），所以先确认 token；\n"
            f"  · 确认 repo id 拼写：{repo}；\n"
            f"  · 确认 revision/文件名在 tree 里存在（`python -m trlab --check-hf` 会列出来）。",
            detail=detail)
    if status in (429, 500, 502, 503, 504):
        return HFHttpError(
            status,
            f"{what} 暂时失败（HTTP {status}）：Hub 侧限流或故障。{tail}\n"
            "  → 等几十秒重跑；这不是配置问题，不用改 token。",
            detail=detail)
    return HFHttpError(status, f"{what} 失败（HTTP {status}）。{tail}", detail=detail)


# ---------------------------------------------------------------- tree / 元数据

def repo_files(repo, revision="main", token=None, dataset=False, max_pages=20):
    """列仓库文件：[{path, size, sha256, is_lfs}]。

    `sha256` 只有 LFS 文件才有（= `lfs.oid` = 内容 sha256）；非 LFS 小文件
    HF 只给 git blob sha1，这里**不**冒充 sha256，标 is_lfs=False 让调用方
    只按 size 校验。

    **URL 参数是实测钉住的（v1.1.3）**：`expand=true` 与 `limit=1000` 同时出现
    会被 Hub 以 HTTP 400 拒绝（正文 `Invalid limit for index tree pagination`）。
    2026-10-23 对 `openai-community/gpt2` 与 `hf-internal-testing/tiny-random-gpt2`
    做过 A/B：
      · `recursive=1&expand=true&limit=1000` → **400**
      · `recursive=1&limit=1000`            → 200，且**仍然带 `lfs.oid`/`size`/`pointerSize`**
      · `recursive=1&expand=true&limit=50`  → 200（但正文 28 KB vs 5 KB，expand 不需要）
    所以这里**不传 expand**：既拿到内容 sha256，又不触发 400。
    """
    kind = "datasets/" if dataset else "models/"
    url = (f"{HF_BASE}/api/{kind}{repo}/tree/{revision}"
           f"?recursive=1&limit={_TREE_PAGE_LIMIT}")
    out, pages = [], 0
    while url and pages < max_pages:
        pages += 1
        st, hdrs, body = _http_get(url, headers=_auth_headers(token))
        if st != 200:
            raise _api_error(st, repo, f"列 {repo}@{revision} 的文件", body, token)
        try:
            entries = json.loads(body.decode("utf-8") or "[]")
        except ValueError as e:
            raise GuardError(f"tree API 返回的不是 JSON（{e}）—— 上游可能改了接口")
        if not isinstance(entries, list):
            raise GuardError(
                f"tree API 返回的不是数组（{type(entries).__name__}）—— 上游可能改了接口")
        for e in entries:
            if not isinstance(e, dict) or e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            out.append({"path": e["path"], "size": int(e.get("size") or 0),
                        "sha256": lfs.get("oid"), "is_lfs": bool(lfs)})
        link = _h(hdrs, "link")
        nxt = None
        if link and 'rel="next"' in link:
            seg = [s for s in link.split(",") if 'rel="next"' in s][0]
            nxt = seg[seg.find("<") + 1:seg.find(">")]
        url = nxt
    return out


def repo_sha(repo, revision="main", token=None, dataset=False):
    """revision 指向的 commit sha —— 下载记录里存它，比 "main" 可复现。"""
    kind = "datasets/" if dataset else "models/"
    st, _hdrs, body = _http_get(f"{HF_BASE}/api/{kind}{repo}",
                               headers=_auth_headers(token))
    if st != 200:
        raise _api_error(st, repo, f"读 {repo} 的元数据", body, token)
    try:
        return json.loads(body.decode("utf-8")).get("sha")
    except ValueError:
        return None


def resolve_url(repo, revision, path, dataset=False):
    kind = "datasets/" if dataset else "models/"
    return f"{HF_BASE}/{kind}{repo}/resolve/{revision}/{path}"


def resolve_signed(url, token=None):
    """把 resolve URL 换成最终的（带签名的）CDN 地址，顺带取 x-linked-etag。

    私有仓库的 302 目标自带签名；对那个地址**不要**再带 Authorization。
    """
    st, hdrs, _body = _http_get(url, headers=_auth_headers(token),
                                follow=False, read_body=False)
    etag = (_h(hdrs, "x-linked-etag") or "").strip().strip('"')
    if st in (301, 302, 303, 307, 308):
        loc = _h(hdrs, "location")
        if not loc:
            raise GuardError(f"302 但没有 Location 头：{url}")
        return loc, etag
    if st in (200, 206):
        return url, etag            # 公开仓库可以直连，没有跳转
    if st in (401, 403, 404):
        raise GuardError(f"取 {url} 失败（HTTP {st}）。\n" + _no_token_hint("该仓库"))
    raise GuardError(f"取 {url} 失败（HTTP {st}）。")


# ---------------------------------------------------------------- 下载

def _stamp_path(dst):
    dst = Path(dst)
    return dst.with_name(dst.name + ".hf.json")


def _cached(dst, expect, repo, revision):
    """本地已有文件能否直接复用（不重算 4.5 GiB 的 sha256）。"""
    dst = Path(dst)
    if not dst.exists():
        return None
    if expect.get("bytes") and dst.stat().st_size != int(expect["bytes"]):
        return None
    st = _read_json(_stamp_path(dst))
    if not isinstance(st, dict):
        return None
    if st.get("repo") != repo or st.get("revision") != revision:
        return None
    if expect.get("bytes") and st.get("bytes") != int(expect["bytes"]):
        return None
    if expect.get("sha256") and st.get("sha256") != expect["sha256"]:
        return None
    return {"path": str(dst), "bytes": dst.stat().st_size,
            "sha256": st.get("sha256"), "origin": "reused",
            "sha256_verified": bool(st.get("sha256_verified")),
            "repo": repo, "revision": revision}


def fetch_file(repo, path, dst, deadline, revision="main", token=None,
               expect=None, dataset=False, tag=None, force=False):
    """按 tree 的 size/lfs.oid 校验着把 path 下到 dst（断点续传 + 复用）。

    返回 {path, bytes, sha256, origin, sha256_verified, repo, revision, url}。
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tag = tag or f"HF/{Path(path).name}"

    if expect is None:
        entries = {e["path"]: e for e in repo_files(repo, revision, token, dataset)}
        if path not in entries:
            raise GuardError(
                f"{repo}@{revision} 里没有 `{path}`。\n"
                f"  现有文件（前 20 个）：{sorted(entries)[:20]}")
        expect = entries[path]

    if not force:
        hit = _cached(dst, expect, repo, revision)
        if hit:
            log(f"{tag}: 本地已有且与 tree 记录一致（{hit['bytes'] / 2**30:.2f} GiB），不重下")
            return hit

    url = resolve_url(repo, revision, path, dataset)
    signed, etag = resolve_signed(url, token)
    if expect.get("sha256") and etag and etag != expect["sha256"]:
        raise GuardError(
            f"{tag}: resolve 响应头 x-linked-etag 与 tree 的 lfs.oid 不一致。\n"
            f"  tree oid  {expect['sha256']}\n"
            f"  etag      {etag}\n"
            "上游在两次请求之间换过文件，拒绝继续。")

    pin = {"url": url, "bytes": int(expect["bytes"]),
           "sha256": expect.get("sha256") or None}
    meta = _fetch_pinned_blob(pin, dst, deadline, tag, fetch_url=signed)
    local_sha = meta["sha256"]
    write_json(_stamp_path(dst), {
        "repo": repo, "revision": revision, "path": path,
        "bytes": meta["bytes"], "sha256": local_sha,
        "sha256_verified": bool(expect.get("sha256")),
        "tree_oid": expect.get("sha256"), "url": url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return {"path": str(dst), "bytes": meta["bytes"], "sha256": local_sha,
            "origin": "downloaded",
            "sha256_verified": bool(expect.get("sha256")),
            "repo": repo, "revision": revision, "url": url}


def fetch_many(repo, items, deadline, revision="main", token=None,
               dataset=False, root=None):
    """items: [(path_in_repo, local_path)]。返回 {path_in_repo: meta}。"""
    out = {}
    entries = {e["path"]: e for e in repo_files(repo, revision, token, dataset)}
    for path, dst in items:
        if path not in entries:
            raise GuardError(
                f"{repo}@{revision} 里没有 `{path}`；现有：{sorted(entries)[:20]}")
        out[path] = fetch_file(repo, path, Path(root) / dst if root else dst,
                               deadline, revision=revision, token=token,
                               expect=entries[path], dataset=dataset)
    return out


# ---------------------------------------------------------------- 上传（opt-in）

def _hf_hub():
    """按需 import；缺库时给"别乱 -U"的明确指引，绝不自动安装。"""
    try:
        import huggingface_hub
        return huggingface_hub
    except Exception:
        raise GuardError(
            "上传需要 `huggingface_hub`，当前解释器没有这个包。\n"
            "  → **不要** `pip install -U huggingface_hub`：Kaggle 镜像里 "
            "tokenizers 0.23.x 对 hub 有 <2.0 的约束，-U 会把环境搞成冲突状态"
            "（用户真实撞过）。\n"
            "  → 用镜像自带的版本；确实要装就显式钉一个与镜像兼容的版本，"
            "并在报告里记下版本号。")


def local_sha256(path, deadline=None):
    return sha256_of(path, deadline)


def upload_files(repo, files, token=None, revision="main", message=None,
                 verify=True, dry_run=False):
    """显式 opt-in 上传：files = [(local_path, path_in_repo)]。

    返回 {commit, uploaded:[...], verify:{...}}。默认**不做任何事**，
    只有调用方（CLI `--upload` / CFG['HF_UPLOAD_AFTER_RUN']=True）才会进来。
    """
    files = [(Path(a), str(b)) for a, b in files]
    if not files:
        return {"uploaded": [], "note": "没有要上传的文件"}
    if dry_run:
        return {"dry_run": True, "would_upload": [
            {"local": str(a), "path": b, "bytes": a.stat().st_size if a.exists() else None}
            for a, b in files]}

    tok = token or hf_token()["token"]
    if not tok:
        raise GuardError("上传需要 token。\n" + _no_token_hint(repo))

    hub = _hf_hub()
    api = hub.HfApi(token=tok)
    msg = message or f"trlab: upload {len(files)} file(s)"
    uploaded, commit = [], None
    for local, rel in files:
        if not local.exists():
            raise GuardError(f"要上传的文件不存在：{local}")
        try:
            info = api.upload_file(path_or_fileobj=str(local), path_in_repo=rel,
                                   repo_id=repo, repo_type="model",
                                   revision=revision, commit_message=msg)
        except Exception as e:                       # 库的异常消息里可能带 token
            raise GuardError(f"上传 {rel} 失败：{redact(e, tok)}")
        commit = getattr(info, "commit_id", None) or commit
        uploaded.append({"path": rel, "bytes": local.stat().st_size,
                         "sha256": local_sha256(local)})
        log(f"HF 上传：{rel}（{local.stat().st_size / 2**30:.2f} GiB）")

    out = {"repo": repo, "revision": revision, "commit": commit,
           "uploaded": uploaded, "verify": None}
    if verify:
        out["verify"] = verify_remote(repo, uploaded, revision=revision, token=tok)
    return out


def verify_remote(repo, local_files, revision="main", token=None):
    """上传后复核：远端 size 必须相等；LFS 文件再比 lfs.oid 与本地 sha256。

    **这是 size + LFS sha256 两件事**。非 LFS 文件 HF 只给 git blob sha1，
    这里不假装比过内容 —— 明确写 `content_sha256_verified=False`。
    """
    remote = {e["path"]: e for e in repo_files(repo, revision, token)}
    checks = []
    ok = True
    for rec in local_files:
        rel = rec["path"]
        r = remote.get(rel)
        item = {"path": rel, "local_bytes": rec["bytes"],
                "remote_bytes": (r or {}).get("size"),
                "size_match": bool(r) and r["size"] == rec["bytes"],
                "content_sha256_verified": False,
                "remote_sha256": (r or {}).get("sha256")}
        if r and r.get("sha256") and rec.get("sha256"):
            item["content_sha256_verified"] = r["sha256"] == rec["sha256"]
            item["sha256_match"] = item["content_sha256_verified"]
        item["ok"] = bool(item["size_match"]) and (
            item["content_sha256_verified"] or not (r or {}).get("sha256"))
        ok = ok and item["ok"]
        checks.append(item)
    return {"ok": ok, "checked": len(checks), "files": checks,
            "note": ("size 相等不等于内容逐字节相同；LFS 文件额外比了 lfs.oid"
                     "（= 内容 sha256）。非 LFS 文件 HF 只给 git blob sha1，"
                     "没有做内容哈希比对。")}


def check_access(repo=None, revision="main", token=None, use_cache=True):
    """**开跑前的凭据/授权快速检查**（v1.1.2）。

    存在理由：真实 Kaggle 上第一次 RUN_EVAL 时，没有 attach `HF_TOKEN`，
    而预检排在"28 分钟编译"之后 —— 用户白等半小时才看到 401。
    这个函数只做一件事：能不能**读**这个仓库的 tree（下载的前提）。
    一次 HTTP 请求，1 秒级，结果按 (repo, rev, 有没有 token) 缓存。

    不检查写权限（上传才需要）；不打印 token；不会替用户改仓库可见性。
    返回 `status`（**按 HTTP 状态码分类，不靠字符串猜**）：
      · `ok`            —— 能读，继续
      · `no_token`      —— 本 Session 根本没凭据（**不用发请求就能判定**）
      · `bad_request`   —— 400：**请求参数问题（本工具的 bug）**，与 token 无关
      · `unauthorized`  —— 401/403：token 无效、过期或对该 repo 没权限
      · `not_found`     —— 404：repo/revision 不存在
      · `rate_limited`  —— 429/5xx：Hub 侧限流或故障，等一会重跑
      · `network`       —— 连不上（DNS/超时/断网/Internet=Off）
      · `error`         —— 其它
    """
    repo = repo or CFG.get("HF_MODEL_REPO") or ""
    tk = hf_token() if token is None else {"token": token, "source": "arg"}
    key = (repo, revision, bool(tk["token"]), tk["source"])
    if use_cache and key in _ACCESS_CACHE:
        return _ACCESS_CACHE[key]

    out = {"repo": repo, "revision": revision,
           "token_present": bool(tk["token"]), "token_source": tk["source"]}
    if not repo:
        out.update(ok=False, status="error", error="CFG['HF_MODEL_REPO'] 为空")
    elif not tk["token"]:
        out.update(ok=False, status="no_token", hint=_no_token_hint(repo))
    else:
        try:
            files = repo_files(repo, revision, tk["token"])
            out.update(ok=True, status="ok", n_files=len(files))
        except HFHttpError as e:
            out.update(ok=False, status=_STATUS_BY_HTTP.get(e.status, "error"),
                       http_status=e.status,
                       error=redact(e, tk["token"]),
                       hint=(_no_token_hint(repo)
                             if e.status in (401, 403) else None))
        except GuardError as e:
            msg = redact(e, tk["token"])
            net = "连不上 Hugging Face" in msg
            out.update(ok=False, status=("network" if net else "error"), error=msg)
    if use_cache:
        _ACCESS_CACHE[key] = out
    return out


def clear_access_cache():
    """测试/重试用：清掉缓存，让下一次 check_access 真的发请求。"""
    _ACCESS_CACHE.clear()


def status(repo=None, revision="main", token=None):
    """CLI `--check-hf` 用：只报"有没有凭据 / 能不能列文件"，不打印 token。"""
    repo = repo or CFG.get("HF_MODEL_REPO") or ""
    tk = hf_token() if token is None else {"token": token, "source": "arg"}
    out = {"repo": repo, "revision": revision,
           "token_present": bool(tk["token"]), "token_source": tk["source"]}
    if not repo:
        out["error"] = "CFG['HF_MODEL_REPO'] 为空"
        return out
    try:
        files = repo_files(repo, revision, tk["token"])
        out["ok"] = True
        out["sha"] = repo_sha(repo, revision, tk["token"])
        out["files"] = [{"path": f["path"], "bytes": f["size"],
                         "lfs_sha256": bool(f["sha256"])} for f in files]
    except HFHttpError as e:
        out["ok"] = False
        out["http_status"] = e.status
        out["status"] = _STATUS_BY_HTTP.get(e.status, "error")
        out["error"] = redact(e, tk["token"])
    except GuardError as e:
        out["ok"] = False
        out["status"] = "network" if "连不上 Hugging Face" in str(e) else "error"
        out["error"] = redact(e, tk["token"])
    return out


def upload_after_run(files, deadline=None, repo=None, token=None):
    """执行完成后的自动上传入口 —— 只有 CFG['HF_UPLOAD_AFTER_RUN'] 才调。

    默认 False：不显式 opt-in 时，一个字节都不往外发。
    """
    if not CFG.get("HF_UPLOAD_AFTER_RUN"):
        log("HF 自动上传未开启（CFG['HF_UPLOAD_AFTER_RUN']=False）—— 结果只落本地")
        return {"skipped": True,
                "why": "CFG['HF_UPLOAD_AFTER_RUN']=False（默认不泄露数据）"}
    repo = repo or CFG.get("HF_UPLOAD_REPO") or CFG.get("HF_MODEL_REPO")
    if not repo:
        raise GuardError("要自动上传但 CFG['HF_UPLOAD_REPO'] 为空")
    return upload_files(repo, files, token=token)
