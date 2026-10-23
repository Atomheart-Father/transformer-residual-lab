#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 trlab/ 包生成 Kaggle 用的**短** notebook。

设计要点（对应 Issue #5 comment-6346361186 的改造要求）：

* notebook 不再是 250 KB 巨物，只有 5 个 cell：标题 / 开关 / 开关格 / 执行 / 引导+run。
* 执行包不从 notebook 里搬进去，而是**钉 commit 从公开仓库 sparse fetch**
  （实测：197 KiB、1–3 秒、`git rev-parse HEAD` 强校验）。私有仓库不可匿名读，
  所以这条路要求仓库是 public 的；`--embed` 可生成不依赖仓库的自包含版本。
* 产物确定性：同样的 trlab/ 源码 → 同样的 notebook（zip 用固定时间戳 + 排序）。
* v1.1.0：开关格多了 `RUN_EVAL`（只评测：从 HF 已存 GGUF 取输入）。
* v1.1.1：评测默认用 GPU（有就整模型卸载）、一次只装一个模型、每阶段小结果出站。
* v1.1.2：`RUN_EVAL` 在新 Session 里**自动只编 S2**（缺二进制时）、HF 授权检查
  排在编译之前（没 token 就 3 秒内停）、失败写 `FAILURE.json` 且最早根因不被覆写。

用法：
    python3 build_notebook.py                 # 钉当前 HEAD（要求工作区干净）
    python3 build_notebook.py --commit <sha>  # 显式钉某个 commit
    python3 build_notebook.py --embed         # 额外内嵌 base64 包（不依赖仓库）
"""
import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "trlab"
NB_OUT = HERE / "trlab_kaggle.ipynb"
REPO_URL = "https://github.com/Atomheart-Father/transformer-residual-lab"
PKG_RELPATH = "research/kaggle/trlab"

BOOTSTRAP_TMPL = '''# =====================================================================
# 自动获取执行包并校验，然后跑。你不需要改这一格的任何东西。
# =====================================================================
import subprocess as _sp, sys as _sys, shutil as _shutil
from pathlib import Path as _P

SRC_URL     = "{src_url}"
SRC_COMMIT  = "{src_commit}"      # 钉死：GitHub 上 commit 不可变，可强校验
PKG_RELPATH = "{pkg_relpath}"
_KAGGLE     = _P("/kaggle/working").is_dir()
DEST        = _P("/kaggle/working/_trlab_src") if _KAGGLE else _P.cwd() / "_trlab_src"
STAMP       = DEST / ".trlab_commit"

def _git(*args, **kw):
    return _sp.run(["git", *args], cwd=str(DEST), capture_output=True,
                   text=True, timeout={git_timeout}, **kw)

def _already_here():
    """同一 commit 的包已经在盘上就直接复用（同一 Session 里重复 Run All 不重下）。"""
    if not (DEST / PKG_RELPATH / "__init__.py").exists():
        return False
    try:
        return STAMP.read_text().strip() == SRC_COMMIT
    except OSError:
        return False

def _fetch_from_github():
    """钉 commit + sparse checkout：只取 research/kaggle/trlab 这一个子目录。"""
    if not _shutil.which("git"):
        raise RuntimeError("环境里没有 git，无法获取执行包")
    DEST.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2, 3):
        r = _git("init", "-q")
        r = _git("config", "core.sparseCheckout", "true")
        r = _git("sparse-checkout", "set", PKG_RELPATH)
        r = _git("remote", "add", "origin", SRC_URL)
        r = _git("fetch", "--depth", "1", "--filter=blob:none",
                 "origin", SRC_COMMIT)
        if r.returncode == 0:
            break
        if attempt == 3:
            raise RuntimeError(
                "取执行包失败（试了 3 次）。最后一条 git 输出：\\n" + (r.stderr or "")[-600:]
                + "\\n常见原因：仓库不是 public、commit 不存在、或网络不通。")
        _git("remote", "remove", "origin")
    r = _git("checkout", "-q", "FETCH_HEAD")
    if r.returncode != 0:
        raise RuntimeError("checkout 失败：" + (r.stderr or "")[-400:])
    got = _git("rev-parse", "HEAD").stdout.strip()
    if got != SRC_COMMIT:
        raise RuntimeError(f"校验失败：拿到 {{got}}，期望 {{SRC_COMMIT}}")
    STAMP.write_text(SRC_COMMIT + "\\n")
    return DEST / PKG_RELPATH
{embed_block}
def _bootstrap():
    if _already_here():
        print(f"执行包已在盘上（commit {{SRC_COMMIT[:12]}}），直接复用")
        return DEST / PKG_RELPATH
    try:
        pkg = _fetch_from_github()
    except Exception as e:
        # 仓库取不到（不是 public / commit 被改 / 网络不通）时，
        # 退回 notebook 内嵌的同一份执行包（sha256 校验后解压）。
        if "_fetch_from_embedded" not in globals():
            raise
        print(f"从仓库取包失败：{{e}}")
        print("改用 notebook 内嵌的执行包（sha256 校验后解压）")
        pkg = _fetch_from_embedded()
    print(f"执行包就绪：{{SRC_COMMIT[:12]}}  {{pkg}}")
    return pkg

_pkg_parent = str(_bootstrap().parent)
if _pkg_parent not in _sys.path:
    _sys.path.insert(0, _pkg_parent)

import trlab
print(f"trlab v{{trlab.__version__}}  运行位置：{{'Kaggle' if _KAGGLE else '本地/其他'}}")

trlab.run(run_s2=RUN_S2, run_s3=RUN_S3, run_s4=RUN_S4, run_eval=RUN_EVAL)
'''

EMBED_BLOCK = '''
_BUNDLE_B64 = """{b64}"""
_BUNDLE_SHA = "{sha}"

def _fetch_from_embedded():
    """仓库不可匿名读时的退路：notebook 里内嵌的同一个包（sha256 校验后解压）。"""
    import base64 as _b64, hashlib as _hl, zipfile as _zf, io as _io
    raw = _b64.b64decode(_BUNDLE_B64)
    if _hl.sha256(raw).hexdigest() != _BUNDLE_SHA:
        raise RuntimeError("内嵌执行包校验失败（sha256 不符）—— notebook 可能被改坏")
    # zip 里的条目是包内文件名（config.py …），解到 DEST/research/kaggle/trlab，
    # 和 sparse fetch 的落点保持一致，_already_here()/sys.path 才认。
    target = DEST / PKG_RELPATH
    for attempt in (1, 2):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with _zf.ZipFile(_io.BytesIO(raw)) as z:
                z.extractall(target)
            STAMP.write_text(SRC_COMMIT + "\\n")
            return target
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"内嵌执行包解压失败：{{e}}")
    raise RuntimeError("unreachable")

'''

MD_TITLE = """# transformer-residual-lab · Kaggle 低位宽量化验证台（v{version}）

**只需要改一个开关**（下面第一个代码格），然后 `Run All`。

| 开关 | 会做什么 | 预估 |
|---|---|---|
| `RUN_EVAL = True` | **只评测**：从 HF 已存的 GGUF 取输入，跑 S6 配对 NLL + S7 质量检查。不下载 BF16、不重做 S3、不重新量化；**缺 `llama-perplexity`/`llama-cli` 时自动只编 S2**（不下载模型）；**有 GPU 就用 GPU**，一次只装一个模型 | GPU 上分钟级（假设速率，非实测）；需补编时 +≈28 分钟 |
| `RUN_S2 = True` | 编译 llama.cpp（下载 36.6 MiB 源码 → cmake → 4 个二进制） | ≈28 分钟 |
| `RUN_S3 = True` | 自动带上 S2，再生成 `imatrix.gguf`（下 8.11 GiB Q8_0 + wikitext，算完回收 Q8_0） | +≈10 分钟 |
| `RUN_S4 = True` | 自动带上 S2+S3，再从**原始 BF16** 量化 IQ2_XS / IQ4_XS，并统计真实字节 | +≈30–60 分钟 |

全部 `False` 时只跑 S0/S1/S9（环境检查 + 冒烟 + 报告），**不联网、不下载任何东西**。
打开任一开关前请确认右侧 `Settings → Internet = On`。

`RUN_EVAL = True` 需要 HF 私有仓库的 token（`HF_TOKEN` 环境变量 / Kaggle secret /
本机 `huggingface-cli login` 的缓存，三者任一）。**不要把 token 写进 notebook。**
Kaggle 上要在 `Add-ons → Secrets` 新建 `HF_TOKEN` 并**勾选 attach 到本 notebook**；
配一次就够。没有 token 时它会在**编译之前**停下（3 秒内），`out/FAILURE.json` 会同时
列出"没 token"和"缺二进制"两条根因，不用去翻日志。
它会把每个阶段的小结果（5 个 JSON/MD，不含模型、不含日志）出站到 HF 并远端复核；
想彻底不外发就 `--set HF_UPLOAD_AFTER_RUN=false`。

**新 Session 不需要你额外开 S2**：`RUN_EVAL = True` 会先探活
`llama-perplexity` / `llama-cli`，缺了就**自己只编 S2**（pinned 源码 36.6 MiB，
不下载 Q8_0/BF16、不做 S3/S4、不重量化）；都在就直接进评测。

**Kaggle 是可丢弃的算力机**：跑失败就直接关掉 Session，修代码 → commit → 新 Session
重跑，输入不用重下。失败现场写在 `out/FAILURE.json`（最早根因不会被后面的失败覆盖），
不需要 Save Version 或抢救工作区。**S6/S7 没出数据时报告顶部会写"本次运行失败"**，
不会用一份看起来正常的报告掩盖过去。

`RUN_S4 = True` 会在**任何编译、任何下载之前**先跑一次几秒的 `S4_PREFLIGHT`：
只读核查 working / 工作区外暂存盘的挂载、容量、写权限，以及 BF16 有没有落点、
imatrix 能不能复用。不通过就**就地停下**并只给一步要做的 —— 不会先花 28 分钟编译、
再下 8 GiB / 15 GiB 之后才发现磁盘不够。

**代码不在这份 notebook 里。** 第二个代码格会从公开仓库按**钉死的 commit**
sparse fetch `research/kaggle/trlab/`（约 200 KB，1–3 秒），校验 `git rev-parse HEAD`
与钉的值一致后才 import。源码在
[{repo_url}]({repo_url}/tree/main/research/kaggle/trlab)，
单元测试直接 import 它，不解析 notebook。

S8 / SX 本版不启动；产物与 provenance 落在 `/kaggle/working/trlab/out/`。
"""

MD_SWITCHES = """## 开关

默认全部 `False`。想跑哪一级就改哪一个：`RUN_EVAL` 只评测（不动模型，
缺二进制会自动只编 S2），`RUN_S4` 会自动把 S2/S3 需要的东西都带上。"""

MD_RUN = """## 执行

这一格负责：取执行包 → 校验 → `trlab.run(...)`。失败会停在这一格并说清原因，
不会带着残缺状态往下跑。

只想做那个几秒的磁盘检查、不编译不下载：把最后一行改成
`trlab.run(preflight_only=True)`。`RUN_S4 = True` 的 `Run All` 已经**自动**先跑它，
不用自己先单独点一轮。

预检不通过时这一格会直接停下，并只给一步要做的（例如先做 `trlab_fetch_bf16.ipynb`
把 BF16 变成只读 Input，或清掉 working 里的旧产物）；明细落在
`/kaggle/working/trlab/out/S4_PREFLIGHT.json`。

**只评测（`RUN_EVAL = True`）**：输入来自 HF 私有仓库里已存的三个 GGUF，
不下载 15 GiB BF16、不重做 S3、不重新量化；**本 Session 缺 `llama-perplexity` /
`llama-cli` 时它会自动只编 S2**（pinned 源码 36.6 MiB，不下载模型），所以你不需要
再开第二个开关。缺 token 时会在**编译之前**停下，`out/FAILURE.json` 同时给
"没 token"与"缺二进制"两条根因，`next_step` 只写一件事：去 `Add-ons → Secrets`
加 `HF_TOKEN` 并 attach，再 `Run all` —— **不要把 token 写进 notebook**。
**有 GPU 就用 GPU**（`-ngl 99`，装不下才退 CPU 并把原因写进 `device_by_fmt`）；
一次只装一个模型，跑完释放再装下一个。
每个阶段的小结果会自动出站到 HF（5 个 JSON/MD）并做远端复核；失败时这一格
不会假装成功，现场在 `out/FAILURE.json`，报告顶部会写"本次运行失败"。

S8 / SX 默认关闭。要跑就把最后一行改成
`trlab.run(run_eval=True, S8_BF16_ANCHOR=True)` —— 阶段名直接当关键字参数传。"""

SWITCH_CELL = """# 只需要改这一格。全部 False = 不联网、不下载（只跑 S0/S1/S9）。
RUN_EVAL = False    # ← 只评测（从 HF 已存 GGUF 取输入；缺二进制会自动只编 S2；有 GPU 就用 GPU）
RUN_S2 = False
RUN_S3 = False
RUN_S4 = False      # ← 想量化 IQ2_XS / IQ4_XS 就改成 True
"""


def git(*args):
    r = subprocess.run(["git", *args], cwd=str(HERE), capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def repo_head():
    rc, out, err = git("rev-parse", "HEAD")
    if rc != 0:
        sys.exit(f"!! git rev-parse HEAD 失败：{err}")
    rc, dirty, _ = git("status", "--porcelain", "--", ".")
    if dirty:
        sys.exit("!! 工作区有未提交改动。notebook 钉的 commit 必须真包含这些源码，\n"
                 "   请先 commit trlab/ 再跑本脚本。\n"
                 f"   未提交文件：{dirty[:400]}")
    return out


def build_zip():
    """确定性 zip：固定时间戳 + 文件名排序 + 固定权限位。"""
    files = sorted(PKG_DIR.rglob("*.py"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(PKG_DIR).as_posix(),
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, f.read_bytes())
    return buf.getvalue()


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=None, help="钉哪个 commit（默认 HEAD）")
    ap.add_argument("--embed", action="store_true",
                    help="额外内嵌 base64 执行包，仓库不可匿名读时也能跑")
    ap.add_argument("--out", default=str(NB_OUT))
    args = ap.parse_args()

    commit = args.commit or repo_head()
    version = "0.6.0"
    vfile = PKG_DIR / "config.py"
    for ln in vfile.read_text(encoding="utf-8").splitlines():
        if ln.startswith("NOTEBOOK_VERSION"):
            version = ln.split("=")[1].strip().strip('"')
            break

    embed_block = ""
    if args.embed:
        blob = build_zip()
        b64 = base64.b64encode(blob).decode()
        b64 = "\\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))
        embed_block = EMBED_BLOCK.format(b64=b64,
                                         sha=hashlib.sha256(blob).hexdigest())
        print(f"内嵌执行包：{len(blob)} B → base64 {len(b64)} B")

    boot = BOOTSTRAP_TMPL.format(src_url=REPO_URL, src_commit=commit,
                                 pkg_relpath=PKG_RELPATH, git_timeout=120,
                                 embed_block=embed_block)
    nb = {
        "cells": [
            md_cell(MD_TITLE.format(version=version, repo_url=REPO_URL)),
            md_cell(MD_SWITCHES),
            code_cell(SWITCH_CELL),
            md_cell(MD_RUN),
            code_cell(boot),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": "trlab kaggle",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(args.out)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"写出 {out}  ({out.stat().st_size} B, {len(nb['cells'])} cells)")
    print(f"钉 commit {commit}")
    print("下一步：git add trlab_kaggle.ipynb && git commit")


if __name__ == "__main__":
    main()
