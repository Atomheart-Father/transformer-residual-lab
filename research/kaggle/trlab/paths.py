"""派生路径与"东西放在哪"的助手。

路径常量在 config；这里放按路径取文件的函数，便于测试替换。
"""

from .config import BUILD, CFG, MDLS, MODEL_DL, S2_PINNED, S3_PINNED, SRC_CACHE, WORK
from .errors import GuardError

import shutil, subprocess
from pathlib import Path

def _s3_base_path():
    """RUN_S3 自动下载的 Q8_0 落点（唯一一份）。"""
    return MODEL_DL / S3_PINNED["filename"]



def _calib_parquet_path():
    return MODEL_DL / "wikitext-2-raw-v1-train.parquet"



def _calib_txt_path():
    return WORK / "calib.txt"



def _s3_stamp_path():
    """下载缓存的 stamp：记 sha256 / bytes / 完成时间，用于"不重下"与二次校验。"""
    return MODEL_DL / "pinned_inputs.json"



def _manual_llama_src():
    """只找"用户手动提供"的源码：CFG['LLAMA_SRC'] 或 /kaggle/input 下的约定路径。

    与自动下载分开，是为了让 S2 能如实标注来源（manual / cache / downloaded），
    而不是把"上次自己下载的"误当成"用户挂载的"。
    """
    if CFG["LLAMA_SRC"]:
        p = Path(CFG["LLAMA_SRC"])
        return p if (p / "CMakeLists.txt").exists() else None
    for cand in [Path("/kaggle/input/llama.cpp"),
                 Path("/kaggle/input/llama-cpp/llama.cpp"),
                 Path("/kaggle/input/llamacpp/llama.cpp")]:
        if (cand / "CMakeLists.txt").exists():
            return cand
    return None



def _llama_src_path():
    """非抛版本，供指纹使用。手动路径优先，其次自动下载的缓存目录。"""
    p = _manual_llama_src()
    if p:
        return p
    q = SRC_CACHE / S2_PINNED["dir"]
    return q if (q / "CMakeLists.txt").exists() else None



def _model_path(fmt):
    """S4 产物的约定路径。"""
    return MDLS / f"qwen3-8b-{fmt}.gguf"



def llama_bin(name):
    """按 CFG -> WORK/build-llama -> 源码树 -> PATH 的顺序找可执行文件。

    P0-A：build 目录固定在 WORK 下，因为 /kaggle/input 是只读的。
    """
    if CFG["LLAMA_BIN"]:
        p = Path(CFG["LLAMA_BIN"]) / name
        if p.exists():
            return p
    for cand in (BUILD / "bin" / name, BUILD / name):
        if cand.exists():
            return cand
    src = _llama_src_path()
    if src:
        for cand in (src / "build" / "bin" / name, src / "build" / name):
            if cand.exists():
                return cand
    w = shutil.which(name)
    if w:
        return Path(w)
    raise GuardError(
        f"找不到 {name}。先跑 S2_BUILD（build 在 {BUILD}），或设置 CFG['LLAMA_BIN']。")



def _git_commit(p):
    try:
        r = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None



def _q8_stamp_path():
    return MODEL_DL / "q8_0.stamp.json"



def _calib_stamp_path():
    return MODEL_DL / "calib.stamp.json"
