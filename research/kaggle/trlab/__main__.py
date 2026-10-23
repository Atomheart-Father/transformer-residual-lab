"""`python -m trlab` 的入口：直接转给 run.main()。

保留 `python -m trlab.run` 也照样能用（历史命令不变）。
"""

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
