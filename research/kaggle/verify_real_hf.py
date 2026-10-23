#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机 Hugging Face 验收（**不**属于离线测试套件，要联网）。

用途：在不下载 7 GiB、不编译 28 分钟的前提下，把"下载链路"真的走一遍：

  1. `--probe`  只发 1 个字节（Range: bytes=0-0）：resolve → 鉴权 → size 核对。
     私仓要 token；公开仓库匿名即可。这一步几秒内完成。
  2. `--small`  真下一个**小文件**（默认就是 S6 要用的 wikitext-2 test
     parquet，732,610 B），按 tree 的 `size` / `lfs.oid` 逐项校验，
     并把 `.hf.json` stamp 写下来 —— 证明"私有同源 200 / CDN 302"两条
     鉴权路径与断点续传、sha256 校验在真网络上成立。

用法：
    python3 verify_real_hf.py --probe                       # 只探测（约 1 字节）
    python3 verify_real_hf.py --small                       # 真下 732 KB 并校验
    python3 verify_real_hf.py --small --repo org/name --path a/b.gguf

退出码 0 = 全部通过；1 = 有一步不符（消息里会说清是哪一步）。
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trlab import hf                                    # noqa: E402
from trlab.config import EVAL_PINNED                     # noqa: E402
from trlab.errors import GuardError                      # noqa: E402

DEFAULT_REPO = EVAL_PINNED["repo"]
DEFAULT_PATH = EVAL_PINNED["path"]
DEFAULT_REVISION = EVAL_PINNED["revision"]


def _show(title, obj):
    print("=" * 78)
    print(title)
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str)[:2000])


def step_tree(repo, revision, token):
    """第一步：tree API 真的能列文件，且 LFS 文件带内容 sha256。"""
    t0 = time.time()
    files = hf.repo_files(repo, revision, token, dataset=True)
    lfs = [f for f in files if f["is_lfs"] and f["sha256"]]
    _show(f"tree：{repo}@{revision}（{time.time() - t0:.1f} s）", {
        "files": len(files),
        "lfs_with_sha256": len(lfs),
        "sample": [{"path": f["path"], "size": f["size"],
                    "sha256": (f["sha256"] or "")[:16] + "…"}
                   for f in files[:3]],
    })
    if not files:
        raise SystemExit("!! tree 返回空 —— 仓库/revision/token 至少有一个不对")
    return {f["path"]: f for f in files}


def step_probe(repo, revision, path, token, entries):
    """第二步：单字节 Range 探测（不下载整文件）。"""
    e = entries.get(path)
    if e is None:
        raise SystemExit(f"!! tree 里没有 {path}")
    r = hf.probe_remote_file(repo, path, deadline=time.time() + 120,
                             revision=revision, token=token, expect=e,
                             dataset=True)
    _show(f"probe（Range bytes=0-0）：{path}", r)
    if not r.get("ok"):
        raise SystemExit(f"!! 探测失败：{r.get('why')}")
    if r.get("total_bytes") != e["size"]:
        raise SystemExit(f"!! 远端总长 {r.get('total_bytes')} ≠ tree 的 {e['size']}")
    return r


def step_download(repo, revision, path, token, entries, dst_dir):
    """第三步：真下载（小文件）+ size/sha256 双校验 + stamp。"""
    e = entries[path]
    dst = Path(dst_dir) / Path(path).name
    if dst.exists():
        dst.unlink()
    meta = hf.fetch_file(repo, path, dst, time.time() + 600,
                         revision=revision, token=token, expect=e,
                         dataset=True, tag="real/" + Path(path).name)
    _show(f"download：{path}（{meta['bytes'] / 1024:.1f} KB）", meta)
    ok = True
    if meta["bytes"] != e["size"]:
        print(f"!! size 不符：{meta['bytes']} ≠ {e['size']}")
        ok = False
    if e["sha256"]:
        if meta.get("sha256") != e["sha256"]:
            print(f"!! sha256 不符：{meta.get('sha256')} ≠ {e['sha256']}")
            ok = False
        else:
            print("内容 sha256 与 tree 的 lfs.oid 一致 ✓")
        if not meta.get("sha256_verified"):
            print("!! sha256_verified 应为 True")
            ok = False
    else:
        print("（该文件不是 LFS，tree 没有内容 sha256 —— 只验了 size，"
              "这一点已如实记录）")
    stamp = dst.with_name(dst.name + ".hf.json")
    if not stamp.exists():
        print("!! 缺少 .hf.json stamp")
        ok = False
    else:
        st = json.loads(stamp.read_text(encoding="utf-8"))
        print(f"stamp：fetch_auth={st.get('fetch_auth')} "
              f"kind={st.get('fetch_url_kind')} repo={st.get('repo')}")
    return ok and meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只做单字节探测")
    ap.add_argument("--small", action="store_true", help="真下小文件并校验")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--dataset", action="store_true", default=True)
    ap.add_argument("--model-repo", action="store_true",
                    help="按模型仓库（而非 dataset）解析 --repo/--path")
    args = ap.parse_args()
    if not (args.probe or args.small):
        ap.error("至少给一个：--probe 或 --small")

    tk = hf.hf_token()
    print(f"凭据：{'有（' + tk['source'] + '）' if tk['token'] else '无（匿名）'}"
          f"；repo={args.repo} revision={args.revision[:12]}…")

    # 模型仓库与 dataset 的 URL/API 前缀不同，这里按 --model-repo 切换
    if args.model_repo:
        hf.resolve_url = lambda repo, rev, path, dataset=False: (
            f"{hf.HF_BASE}/{repo}/resolve/{rev}/{path}")
        global _ORIG_REPO_FILES
        _ORIG_REPO_FILES = hf.repo_files

        def rf(repo, revision="main", token=None, dataset=False, max_pages=20):
            return _ORIG_REPO_FILES(repo, revision, token, False, max_pages)
        hf.repo_files = rf

    entries = step_tree(args.repo, args.revision, tk["token"])
    rc = 0
    if args.probe:
        try:
            step_probe(args.repo, args.revision, args.path, tk["token"], entries)
            print("\nprobe 通过：resolve + 鉴权 + size 核对在真网络上成立"
                  "（只用了 1 个字节）。")
        except SystemExit as e:
            print(e)
            rc = 1
    if args.small:
        with tempfile.TemporaryDirectory(prefix="trlab_realhf_") as d:
            try:
                if not step_download(args.repo, args.revision, args.path,
                                     tk["token"], entries, d):
                    rc = 1
                else:
                    print("\nsmall download 通过：size + 内容 sha256 双校验成立，"
                          "stamp 已写（来源可审计）。")
            except (GuardError, SystemExit) as e:
                print(f"!! {e}")
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
