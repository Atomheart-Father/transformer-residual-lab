"""trlab —— transformer-residual-lab 的 Kaggle 低位宽量化验证台。

源码就是这里的 .py 文件；notebook 只是薄入口（一个开关 + 引导 + run()）。
单元测试直接 import 本包，不解析 notebook。

    from trlab import run
    run(run_s4=True)             # S2+S3+S4+S5
    run()                        # 只跑 S0/S1/S9，不联网、不下载
"""
from .config import (CFG, STAGES, NOTEBOOK_VERSION as __version__,
                     S2_PINNED, S3_PINNED, S4_PINNED, IMATRIX_PINNED,
                     CALIB_PINNED, FORMATS)
from .run import run, main

__all__ = ["run", "main", "CFG", "STAGES", "__version__",
           "S2_PINNED", "S3_PINNED", "S4_PINNED", "IMATRIX_PINNED",
           "CALIB_PINNED", "FORMATS"]
