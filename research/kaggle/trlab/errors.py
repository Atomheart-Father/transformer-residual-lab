"""异常类型。单独一个模块，好让 paths / core / fetch 都能 import 而不成环。"""


class BudgetExceeded(RuntimeError):
    """墙钟 / 显存 / 磁盘超限。故意抛异常，而不是等 OOM killer。"""



class GuardError(RuntimeError):
    """前置条件不满足（路径缺失、二进制缺失、开关未打开）。"""



class CommandFailed(RuntimeError):
    def __init__(self, cmd, rc, tail, log_path):
        self.cmd, self.rc, self.tail, self.log_path = list(cmd), rc, tail, str(log_path)
        super().__init__(f"命令失败 rc={rc}: {' '.join(map(str, cmd))}\n--- tail ---\n{tail}")



class UnknownGGMLType(ValueError):
    """遇到不认识的 ggml type。拒绝按默认 8 bpw 假算。"""
