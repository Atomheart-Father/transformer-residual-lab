"""S2：llama.cpp 源码获取与编译（CUDA / NO_VMM）。"""

from .config import BUILD, CFG, IN, LOGS, S2_PINNED, SRC_CACHE, WORK
from .errors import CommandFailed, GuardError
from .paths import _git_commit, _llama_src_path, _manual_llama_src
from .core import _read_json, disk_gb, log, run_cmd
from .fetch import _fetch_pinned_source

import json, os, subprocess, time
from pathlib import Path

def _toolchain_report():
    """cmake / nvcc / g++ 的版本。不抛异常，拿不到就写"不可用"。"""
    out = {}
    for name, cmd in (("cmake", ["cmake", "--version"]),
                      ("nvcc", ["nvcc", "--version"]),
                      ("gxx", ["g++", "-dumpversion"]),
                      ("gcc", ["gcc", "-dumpversion"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            txt = (r.stdout or r.stderr or "").strip()
            out[name] = (txt.split("\n")[0][:120] if r.returncode == 0
                         else f"rc={r.returncode}")
        except Exception as e:
            out[name] = f"不可用: {type(e).__name__}"
    return out



def _cuda_preflight(toolchain, build_cuda):
    """BUILD_CUDA=True 但 nvcc 不可用时提前报错，而不是让 cmake 在 FATAL_ERROR 里淹掉。"""
    nvcc = str((toolchain or {}).get("nvcc") or "")
    if not build_cuda:
        return {"ok": True,
                "note": "BUILD_CUDA=False：只构建 CPU 版；三个工具都有 CPU 路径"}
    if (not nvcc) or nvcc.startswith("不可用") or nvcc.startswith("rc="):
        return {"ok": False,
                "note": ("BUILD_CUDA=True 但 nvcc 不可用。改 CFG['BUILD_CUDA']=False —— "
                         "llama-imatrix / llama-quantize / llama-perplexity 都能走 CPU，"
                         "只是慢；或换一个带 CUDA 的镜像。")}
    return {"ok": True, "nvcc": nvcc}



def _find_llama_src(deadline):
    """返回 {"src", "origin", "sha256_ok", ...}。

    优先级：手动（CFG["LLAMA_SRC"] / /kaggle/input 约定路径）> 自动获取。
    v0.3 删掉了旧分支"ALLOW_NETWORK=True 就 git clone --depth 1 master" ——
    那个分支拿的是 HEAD，与固定 LLAMA_COMMIT 直接冲突，等于让"固定版本"这句话
    失效，属于设计错误而不是缺少功能。
    """
    p = _manual_llama_src()
    if p:
        return {"src": p, "origin": "manual", "sha256_ok": False,
                "url": None, "dir": None,
                "archive_sha256": None, "archive_bytes": None}
    if CFG["S2_AUTO_FETCH_SOURCE"]:
        src_dir, meta = _fetch_pinned_source(deadline)
        out = {"src": src_dir}
        out.update(meta)
        return out
    raise GuardError(
        "找不到 llama.cpp 源码，且没有打开自动获取。\n"
        "  → 一键：把 notebook 顶部 RUN_S2 改成 True（需要 Settings 里 "
        "Internet = On，只下载 36.6 MiB 源码，不下载模型）。\n"
        "  → 离线：把源码挂到 /kaggle/input，并把 CFG['LLAMA_SRC'] 指到含 "
        "CMakeLists.txt 的那一层。")



def _probe_version(bin_path):
    """问二进制要版本串。**必须同时接 stderr。**

    上游 `common/build-info.h:13` 的签名是
    `void llama_print_build_info(const char *, FILE * = stderr);` —— 默认流是
    stderr；`common/arg.cpp:1457` 的 `--version` 分支调用它时没有传流，所以
    `llama-cli --version` 把 build info 写到 **stderr**，stdout 是空的。
    v0.3.0 只取 stdout，于是真实 Kaggle 运行里 `version` 是空串（非阻断，但
    一份看不出二进制版本的结果本身就是缺陷）。
    """
    try:
        r = subprocess.run([str(bin_path), "--version"],
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        txt = out or err
        return (txt[:300] if txt else f"(rc={r.returncode}, 无输出)")
    except Exception as e:
        return f"不可用: {type(e).__name__}"



# v0.4 / D-1：四个必须一起出现的二进制。少任何一个都不能算"构建可用"。
LLAMA_TARGETS = ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-cli")



def _probe_bin_runnable(path, timeout=90):
    """**真的执行一次** `--version`。文件存在不等于能跑。

    反例（都真实存在）：架构不匹配的二进制、上次被杀在一半留下的文件、
    缺 libcudart 的动态链接失败。这些情况下 `Path.exists()` 全是 True，
    但一跑就 rc≠0 或直接抛 OSError。只有跑通才敢复用。
    """
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True,
                           text=True, timeout=timeout)
        txt = ((r.stdout or "") + (r.stderr or "")).strip()
        return (r.returncode == 0 and bool(txt)), txt[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"



def _scan_input_binaries(time_budget_sec=20.0, max_dirs=20000):
    """在 `/kaggle/input` 下自动找"含四个二进制"的目录（v0.4 / D-3）。

    Kaggle 官方文档明确支持把自己的 Notebook Output 通过 `Add Input` 挂成数据源
    （"search for any Notebook with output files"）。用户不需要告诉我们路径 ——
    扫到就用。这里**只读**，绝不写入；`/kaggle/input` 本来就是只读的。

    有时间和目录数双重上限：挂载的数据集可能很大，不能把启动时间拖垮。
    """
    out = []
    if not IN.is_dir():
        return out
    t0, n = time.time(), 0
    for root, dirs, files in os.walk(IN):
        n += 1
        if n > max_dirs or (time.time() - t0) > time_budget_sec:
            break
        fs = set(files)
        if all(t in fs for t in LLAMA_TARGETS):
            out.append(Path(root))
            dirs[:] = []          # 这一层已经齐了，不必再往下钻
    return out



def _find_llama_bins():
    """按优先级找四个二进制。返回 `(bins, origin, detail)`，找不到时 bins=None。

    顺序（v0.4 / D-1）：
      1. `existing_build`   —— `WORK/build-llama/bin`，同一次 Session 编过
      2. `user_bin`         —— `CFG['LLAMA_BIN']`，用户显式指定
      3. `attached_input`   —— `/kaggle/input` 自动扫描，挂载的历史 Output

    每个候选都要**真的跑一次 `--version`**；跑不起来就记下来继续找，
    最后才落到"下载源码 + 完整编译"。
    """
    cands = [("existing_build", BUILD / "bin"), ("existing_build", BUILD)]
    if CFG["LLAMA_BIN"]:
        cands.insert(1, ("user_bin", Path(CFG["LLAMA_BIN"])))
    for d in _scan_input_binaries():
        cands.append(("attached_input", d))

    problems = []
    for origin, d in cands:
        bins = {t: (d / t) for t in LLAMA_TARGETS}
        if not all(p.exists() for p in bins.values()):
            continue
        ok, detail = _probe_bin_runnable(bins["llama-cli"])
        if ok:
            return bins, origin, detail
        problems.append(f"{d}（存在但跑不起来：{detail[:140]}）")
    why = ("；".join(problems) if problems
           else "WORK/build-llama、CFG['LLAMA_BIN']、/kaggle/input 都没有四个二进制")
    return None, None, why



def _resolve_commit_record(src):
    """commit 的三档校验。返回 (commit, source, verified)。

      git            —— 目录里有 .git，`rev-parse HEAD` 成功（最强）
      pinned_tarball —— 目录里没有 .git，但归档 digest 与预登记值一致
      declared       —— 只是 CFG/预登记里写的值，**没有校验过**（最弱）

    复用挂载来的二进制时必然落到 declared 档：我们没有那份源码，
    就不能声称"已验证是这个 commit"。这一点会原样写进 S9 报告。
    即使没有 `CFG['LLAMA_COMMIT']`，也按 `S2_PINNED` 记录一个值并标 declared ——
    记 null 会让报告看不出"我们认为它是哪个版本"，比记一个未校验的值更糟。
    """
    git_commit = _git_commit(src) if src else None
    if git_commit:
        return git_commit, "git", True
    if src is not None and _archive_digest_ok(src):
        commit = CFG["LLAMA_COMMIT"] or S2_PINNED["commit"]
        return commit, "pinned_tarball", True
    if CFG["LLAMA_COMMIT"]:
        commit = CFG["LLAMA_COMMIT"]
        log(f"!! commit 记为声明值 {commit}（未校验：没有 git 仓库，也没有归档 digest）")
        return commit, "declared", False
    if src is None:
        commit = S2_PINNED["commit"]
        log(f"!! 复用二进制、本 Session 内没有源码 → commit 记为声明值 {commit}"
            "（按 S2_PINNED，**未校验**）")
        return commit, "declared", False
    log("!! 源码不是 git 仓库、也没有归档 digest，且没有预注册 commit")
    return None, None, False



def _archive_digest_ok(src):
    """src 是不是我们自动下载并校验过的那份源码（用 stamp 判断）。"""
    st = _read_json(SRC_CACHE / "pinned_source.json")
    return (isinstance(st, dict) and st.get("sha256") == S2_PINNED["sha256"]
            and str(src) == str(SRC_CACHE / S2_PINNED["dir"]))



def stage_s2_build(deadline):
    # ---- v0.4 / D-1：先找"现成就能跑"的二进制，找不到才下载源码 + 编译 ----
    # Kaggle 交互会话空闲 20 分钟就断（官方文档原文），上次编好的二进制可能
    # 已经没了；也可能用户把上一次运行的 Output 挂成了 Input。两种都自动
    # 识别，不需要任何人工配置，也不需要"先跑一遍 S2"。
    bins, bin_origin, bin_detail = _find_llama_bins()
    if bins is not None:
        log(f"复用已有二进制（来源 {bin_origin}，已实际执行 --version 通过）")
        src = _llama_src_path()
        commit, commit_source, commit_verified = _resolve_commit_record(src)
        ver = _probe_version(bins["llama-cli"])
        return {"src": str(src) if src else None,
                "src_writable": (os.access(src, os.W_OK) if src else None),
                "build": str(BUILD), "reused": True, "build_reuse": bin_origin,
                "reuse_detail": bin_detail,
                "commit": commit, "commit_source": commit_source,
                "commit_verified": commit_verified,
                "version": ver, "toolchain": _toolchain_report(),
                "source": {"origin": bin_origin, "url": None, "dir": None,
                           "archive_sha256": None, "archive_bytes": None,
                           "archive_cached": False,
                           "note": ("二进制复用自挂载的 Input；源码不在本 Session 内，"
                                    "commit 只能记为声明值") if bin_origin == "attached_input"
                           else None},
                "cuda": CFG["BUILD_CUDA"], "cuda_arch": CFG["CUDA_ARCH"],
                "cuda_no_vmm": bool(CFG.get("CUDA_NO_VMM") and CFG["BUILD_CUDA"]),
                "disk_free_gb": round(disk_gb(WORK)["free"], 2),
                "logs": {"cmake": str(LOGS / "s2_cmake.log"),
                         "build": str(LOGS / "s2_build.log")},
                "bins": {k: str(v) for k, v in bins.items()}}

    log(f"没有可复用的二进制（{bin_detail}）→ 走完整流程：下载 pinned 源码 + 编译")
    got = _find_llama_src(deadline)
    src = got["src"]
    commit, commit_source, commit_verified = _resolve_commit_record(src)

    if CFG["LLAMA_COMMIT"] and commit and not commit.startswith(CFG["LLAMA_COMMIT"][:12]):
        raise GuardError(
            f"源码 commit {commit} 与预注册的 {CFG['LLAMA_COMMIT']} 不符。"
            "同一份报告不能混用两个 llama.cpp 版本。")

    # P0-A：build 必须落在可写目录。/kaggle/input 是只读的。
    BUILD.mkdir(parents=True, exist_ok=True)
    src_writable = os.access(src, os.W_OK)
    log(f"源码: {src}（来源={got.get('origin')}，可写={src_writable}）  build: {BUILD}")

    # 工具链预检：nvcc 缺失时在这里报错，别让 cmake 的 FATAL_ERROR 淹掉
    tc = _toolchain_report()
    log(f"工具链: {json.dumps(tc, ensure_ascii=False)}")
    pf = _cuda_preflight(tc, CFG["BUILD_CUDA"])
    if not pf["ok"]:
        raise GuardError(pf["note"])

    targets = list(LLAMA_TARGETS)

    def _locate(n):
        for c in (BUILD / "bin" / n, BUILD / n):
            if c.exists():
                return c
        return None

    source = {"origin": got.get("origin"), "url": got.get("url"),
              "dir": got.get("dir"),
              "archive_sha256": got.get("archive_sha256"),
              "archive_bytes": got.get("archive_bytes"),
              "archive_cached": got.get("archive_cached", False)}

    # 走到这里说明探活阶段没找到能跑的二进制，但 BUILD 里可能有**跑不起来**的
    # 残留（上次被杀一半）。cmake --build 会把它们补完，所以不删目录，
    # 只在日志里说清楚即将发生的是"继续/重新编译"。
    have = {n: _locate(n) for n in targets}
    partial = [k for k, v in have.items() if v is not None]
    if partial:
        log(f"BUILD 里有 {len(partial)}/{len(targets)} 个二进制但探活未通过"
            f"（{bin_detail}）→ 让 cmake 增量补全")

    cmake_args = ["cmake", "-S", str(src), "-B", str(BUILD),
                  "-DCMAKE_BUILD_TYPE=Release",
                  "-DGGML_NATIVE=ON",
                  f"-DGGML_CUDA={'ON' if CFG['BUILD_CUDA'] else 'OFF'}",
                  # 当前 master 已把 LLAMA_CURL 标为 deprecated 并忽略；保留只为兼容旧 pin
                  "-DLLAMA_CURL=OFF",
                  "-DLLAMA_BUILD_TESTS=OFF",
                  "-DLLAMA_BUILD_EXAMPLES=OFF",
                  # 否则 tools/ui 会在构建期 file(DOWNLOAD) 去 HF bucket 取 UI，
                  # Internet Off 时会白等几十秒到几分钟（不会失败，但纯浪费）
                  "-DLLAMA_USE_PREBUILT_UI=OFF"]
    if CFG["BUILD_CUDA"] and CFG["CUDA_ARCH"]:
        # 不指定时上游默认 50-v/61-v/70-v/75-v/80-v/86-r/89-r/90-v/120a-r 九套架构，
        # 4 核上编译量翻几倍。T4 只需要 sm_75。
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={CFG['CUDA_ARCH']}")
    if CFG["BUILD_CUDA"] and CFG.get("CUDA_NO_VMM"):
        # v0.3.1：Kaggle T4 真实失败 —— find_package(CUDAToolkit) 解析不出
        # CUDA::cuda_driver IMPORTED target，ggml-cuda 在
        # ggml/src/ggml-cuda/CMakeLists.txt:181 的 target_link_libraries 直接
        # CMake Error（Generate step failed）。上游 178-182 行有
        # if (GGML_CUDA_NO_VMM) 空分支 / else 链 cuda_driver，option 在
        # ggml/CMakeLists.txt:205 默认 OFF。打开它只跳过 libcuda 直链与 VMM
        # 显存池（回退 ggml_cuda_pool_leg，即 cudaMalloc 池），算力内核不变。
        cmake_args.append("-DGGML_CUDA_NO_VMM=ON")

    try:
        run_cmd(cmake_args, LOGS / "s2_cmake.log", deadline)
        run_cmd(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j", str(CFG["BUILD_JOBS"]), "--target", *targets],
                LOGS / "s2_build.log", deadline)
    except CommandFailed as e:
        raise RuntimeError(_explain_s2_failure(e)) from e

    bins = {n: _locate(n) for n in targets}
    missing = [k for k, v in bins.items() if v is None]
    if missing:
        raise RuntimeError(
            f"编译结束但缺少可执行文件 {missing}。"
            f"最可能的原因：有人关掉了 LLAMA_BUILD_SERVER（cli 在它的 if 里）。"
            f"看 {LOGS}/s2_build.log 最后 30 行。")

    ver = _probe_version(bins["llama-cli"])
    return {"src": str(src), "src_writable": src_writable, "build": str(BUILD),
            "reused": False, "build_reuse": "rebuilt",
            "reuse_detail": f"完整编译（{len(LLAMA_TARGETS)} 个目标）",
            "commit": commit, "commit_source": commit_source,
            "commit_verified": commit_verified, "version": ver,
            "cuda": CFG["BUILD_CUDA"], "cuda_arch": CFG["CUDA_ARCH"],
            "cuda_no_vmm": bool(CFG.get("CUDA_NO_VMM") and CFG["BUILD_CUDA"]),
            "cmake_args": [str(a) for a in cmake_args], "toolchain": tc,
            "source": source,
            "disk_free_gb": round(disk_gb(WORK)["free"], 2),
            "logs": {"cmake": str(LOGS / "s2_cmake.log"),
                     "build": str(LOGS / "s2_build.log")},
            "bins": {k: str(v) for k, v in bins.items()}}



def _explain_s2_failure(exc):
    """把 cmake/nvcc 的失败压成"症状 → 处置"一句话，而不是把几十行日志丢给人。

    分类必须**同行匹配**：v0.3.0 用 `"nvcc" in tail and "not found" in tail`
    这种全文包含判断，结果 Kaggle 上"CUDA::cuda_driver target was not found"
    被误判成"找不到 nvcc"，还劝用户关 GPU —— 而 nvcc 明明检测通过了。
    """
    tail = (exc.tail or "").lower()
    lines = [l for l in tail.splitlines() if l.strip()]

    def _has(needle):
        return needle in tail

    def _same_line(a, b):
        return any(a in l and b in l for l in lines)

    if _has("unsupported gnu version") or _has("unsupported host compiler"):
        why = ("nvcc 不支持镜像里的 host g++ 版本。"
               "处置：CFG['BUILD_CUDA']=False（三个工具都有 CPU 路径，只是慢）。")
    elif _has("cuda::cuda_driver") or (_has("cuda_driver") and _has("target_link_libraries")):
        # 必须排在 nvcc 分支之前：这条错误里往往同时出现 nvcc 与 "not found"
        why = ("CMake 找不到 CUDA::cuda_driver IMPORTED target —— 镜像里 nvcc / "
               "CUDA runtime 都在，但 find_package(CUDAToolkit) 的 driver 组件没解析出来。"
               "v0.3.1 起默认传 -DGGML_CUDA_NO_VMM=ON 绕开（只跳过 libcuda 直链与 VMM "
               "显存池，算力内核不变）；如果你把 CFG['CUDA_NO_VMM'] 关成了 False，"
               "改回 True 再跑。")
    elif (_has("could not find cmake_cuda_compiler")
          or _has("no cmake_cuda_compiler")
          or _same_line("nvcc", "not found")
          or _same_line("nvcc", "could not find")
          or _same_line("nvcc", "no nvcc")):
        why = "找不到 nvcc。处置：CFG['BUILD_CUDA']=False，或换带 CUDA 的镜像。"
    elif _has("no space left"):
        why = ("磁盘满。处置：先删 trlab/dl 里的源码包和 trlab/build-llama 里没用的对象；"
               "别把 base 放进 /kaggle/working。")
    elif _has("killed") or _has("out of memory"):
        why = "编译时内存不足。处置：把 CFG['BUILD_JOBS'] 降到 2 再跑。"
    elif _has("cmake error"):
        why = ("cmake 配置阶段失败。处置：看日志最后一屏的 CMake Error 那几行；"
               "最常见的是源码目录指错层（要含 CMakeLists.txt 的那一层）。")
    else:
        why = "编译失败。处置：看日志最后 30 行。"
    return (f"S2 编译失败（rc={exc.rc}）。{why}\n"
            f"日志：{exc.log_path}\n"
            f"--- 日志尾部 ---\n{exc.tail}")
