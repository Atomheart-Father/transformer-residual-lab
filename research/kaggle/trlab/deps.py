"""每阶段依赖表：CFG 键 + 文件解析器。

单独一个模块是因为解析器要调用 paths/imatrix/stages_build，而那些模块
本身依赖 core —— 放 core 里会成环。core 的 _fingerprint() 延迟导入它。
"""

from .paths import _llama_src_path, _model_path
from .core import need
from .imatrix import _calib_txt_for_fp, _imatrix_base_for_fp, resolve_imatrix


# 每阶段依赖：CFG 键 + 文件解析器（失败即 None）
STAGE_DEPS = {
    "S0_ENV":         ([], []),
    "S1_SMOKE":       (["PARAM_COUNT"], []),
    "S2_BUILD":       (["LLAMA_SRC", "LLAMA_BIN", "LLAMA_COMMIT", "BUILD_CUDA",
                        "CUDA_ARCH", "CUDA_NO_VMM", "BUILD_JOBS"],
                       [lambda: _llama_src_path()]),
    "S3_IMATRIX":     (["BASE_GGUF", "BASE_KIND", "IMATRIX_BASE", "IMATRIX_BASE_KIND",
                        "CALIB_TXT", "CALIB_SOURCE", "IMATRIX_CTX", "IMATRIX_CHUNKS",
                        "IMATRIX_GGUF", "NGL_MARGIN_GB", "S3_DELETE_Q8_AFTER",
                        "S3_AUTO_FETCH_MODEL", "S3_AUTO_FETCH_CALIB"],
                       [lambda: _imatrix_base_for_fp(), lambda: _calib_txt_for_fp()]),
    "S4_QUANT":       (["BASE_GGUF", "BASE_KIND", "IMATRIX_BASE", "IMATRIX_BASE_KIND",
                        "IMATRIX_GGUF", "ALLOW_REQUANTIZE"],
                       [lambda: need("BASE_GGUF"), resolve_imatrix]),
    "S5_BYTES":       (["BASE_GGUF", "BASE_KIND"],
                       [lambda: need("BASE_GGUF"),
                        lambda: _model_path("IQ2_XS"), lambda: _model_path("IQ4_XS")]),
    # v1.1：S6/S7 的输入可能来自 HF（本地没有时），所以指纹里既有本地路径，
    # 也有"从哪取"（repo/revision）。本地文件存在时 _path_fp 会把它的
    # 大小/mtime 带进来，换了 GGUF 就会让旧结果失效。
    "S6_NLL":         (["EVAL_TXT", "EVAL_SOURCE", "NLL_CTX", "NLL_CHUNKS_PER_GROUP",
                        "NLL_GROUPS", "SEED", "EVAL_ALLOW_GPU",
                        "HF_FETCH_ARTIFACTS", "HF_MODEL_REPO", "HF_MODEL_REVISION"],
                       [lambda: need("EVAL_TXT"),
                        lambda: _model_path("IQ2_XS"), lambda: _model_path("IQ4_XS")]),
    "S7_QUALITY":     (["QUALITY_PROMPTS", "QUALITY_N_PREDICT", "SEED", "EVAL_ALLOW_GPU",
                        "HF_FETCH_ARTIFACTS", "HF_MODEL_REPO", "HF_MODEL_REVISION"],
                       [lambda: _model_path("IQ2_XS"), lambda: _model_path("IQ4_XS")]),
    "S8_BF16_ANCHOR": (["BASE_GGUF", "BASE_KIND", "ANCHOR_EXTRA_GB", "ANCHOR_ALLOW_APPROX"],
                       [lambda: need("BASE_GGUF")]),
    "S9_REPORT":      ([], []),
    "SX_EXPLORE":     (["EXPLORE_GGUF"], [lambda: need("EXPLORE_GGUF")]),
    # 预检本身在 ALWAYS_RERUN 里（磁盘状态每次都重查），这里的指纹只作记录。
    "S4_PREFLIGHT":   (["BASE_GGUF", "S4_AUTO_FETCH_BASE", "S4_RECLAIM_Q8_CACHE",
                        "S3_SKIP_IF_IMATRIX_OK", "IMATRIX_GGUF", "MIN_FREE_DISK_GB",
                        "S4_RAM_MARGIN_GB", "S2_BUILD_DIR_EST_GB"],
                       [lambda: need("BASE_GGUF"), resolve_imatrix]),
}
