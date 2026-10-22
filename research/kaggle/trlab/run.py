"""编排入口：run(**开关) 顺序执行各阶段并打印摘要。

CLI: python -m trlab.run [--run-s2] [--run-s3] [--run-s4] [--set K=V]
"""

from .config import CFG, LOGS, NOTEBOOK_VERSION, OUT, STAGES, _apply_run_switches, print_banner
from .paths import _s3_base_path
from .core import STATE, ensure_dirs, log, run_stage
from .stages_env import stage_s0_env, stage_s1_smoke
from .stages_build import stage_s2_build
from .stages_imatrix import stage_s3_imatrix
from .stages_quant import stage_s4_quant, stage_s5_bytes
from .stages_eval import stage_s6_nll, stage_s7_quality, stage_s8_bf16_anchor, stage_sx_explore
from .report import _s2_report_lines, _s3_report_lines, _s4_report_lines, stage_s9_report

import argparse, json, sys

def _apply_overrides(overrides):
    """把 run(**overrides) 的键分流：**阶段名进 STAGES，其余进 CFG**。

    v1.0 的 notebook 只有 RUN_S2/S3/S4 三个开关格；S6–S8 与 SX 通过
    `trlab.run(run_s4=True, S6_NLL=True)` 打开（CLI: `--set S6_NLL=true`）。
    没有这一路，薄 notebook 就等于把后四个阶段锁死了。
    """
    for k, v in overrides.items():
        if k in STAGES:
            STAGES[k] = bool(v)
        else:
            CFG[k] = v


def run(run_s2=None, run_s3=None, run_s4=None, **overrides):
    """按开关顺序执行各阶段。

    三个开关都 None 时读 CFG 里的值（默认全 False → 只跑 S0/S1/S9，
    不联网、不下载）。overrides 可直接改任意 CFG 键，例如
    run(run_s4=True, FORCE_RERUN=True)；阶段名（S6_NLL 等）写 STAGES。
    """
    for _k, _v in (("RUN_S2", run_s2), ("RUN_S3", run_s3), ("RUN_S4", run_s4)):
        if _v is not None:
            CFG[_k] = bool(_v)
    _apply_overrides(overrides)
    plan = _apply_run_switches()
    ensure_dirs()
    print_banner()
    log(f"trlab v{NOTEBOOK_VERSION}  plan={plan}")

    s0 = run_stage("S0_ENV", stage_s0_env)

    s1 = run_stage("S1_SMOKE", stage_s1_smoke)

    s2 = run_stage("S2_BUILD", stage_s2_build)

    if s2:
        print("-" * 78)
        print("S2 摘要")
        for _line in _s2_report_lines(s2):
            print(_line)
        print("-" * 78)
    elif (STATE.get("S2_BUILD") or {}).get("status") == "error":
        print("-" * 78)
        print("S2 失败。上面那行 [S2_BUILD] FAIL 就是结论，细节在：")
        print(f"  {OUT / 'S2_BUILD.json'}")
        print(f"  {LOGS / 's2_cmake.log'}  /  {LOGS / 's2_build.log'}")
        print("S9 仍会照常生成报告，其余阶段因为开关是 False 不会启动。")
        print("-" * 78)

    s3 = run_stage("S3_IMATRIX", stage_s3_imatrix)

    if s3:
        print("-" * 78)
        print("S3 摘要")
        for _line in _s3_report_lines(s3):
            print(_line)
        print("-" * 78)
    elif (STATE.get("S3_IMATRIX") or {}).get("status") == "error":
        print("-" * 78)
        print("S3 失败。上面那行 [S3_IMATRIX] FAIL 就是结论，细节在：")
        print(f"  {OUT / 'S3_IMATRIX.json'}")
        print(f"  {LOGS / 's3_imatrix.log'}")
        _q8 = _s3_base_path()
        if _q8.exists():
            print(f"Q8_0 保留在 {_q8}（{_q8.stat().st_size / 2**30:.2f} GiB）—— "
                  "修复后重跑会直接复用它，不会重下。")
        elif _q8.with_name(_q8.name + '.partial').exists():
            _p = _q8.with_name(_q8.name + '.partial')
            print(f"下载断点保留在 {_p}（{_p.stat().st_size / 2**30:.2f} GiB）—— "
                  "重跑会从那里续传。")
        print("S9 仍会照常生成报告。")
        print("-" * 78)

    s4 = run_stage("S4_QUANT", stage_s4_quant)

    if s4:
        print("-" * 78)
        print("S4 摘要")
        for _line in _s4_report_lines(s4):
            print(_line)
        print("-" * 78)
    elif (STATE.get("S4_QUANT") or {}).get("status") == "error":
        print("-" * 78)
        print("S4 失败。上面那行 [S4_QUANT] FAIL 就是结论，细节在：")
        print(f"  {OUT / 'S4_QUANT.json'}")
        print(f"  {LOGS}/s4_quant_*.log")
        print("S4 失败不删任何输入：imatrix 与量化 base 都可直接复用。")
        print("S9 仍会照常生成报告。")
        print("-" * 78)

    s5 = run_stage("S5_BYTES", stage_s5_bytes)

    if s5:
        print("-" * 78)
        print("S5 摘要（真实文件字节，逐类型拆分）")
        for name, d in s5["per_model"].items():
            if d.get("byte_breakdown_available"):
                print(f"  {name:16s} {d['file_gb']:7.3f} GiB  "
                      f"bpw={d['bpw_effective']}  "
                      f"embed+head={d['bytes_per_class'].get('embed/head', 0) / 2**30:.3f} GiB")
            else:
                print(f"  {name:16s} {d['file_gb']:7.3f} GiB  字节拆分不可用")
        if s5.get("ratios"):
            print(f"  IQ4/IQ2 字节比 {s5['ratios']['iq4_over_iq2']}，"
                  f"IQ2 少 {s5['ratios']['bytes_saved_pct']}%")
        print("-" * 78)

    s6 = run_stage("S6_NLL", stage_s6_nll)

    s7 = run_stage("S7_QUALITY", stage_s7_quality)

    s8 = run_stage("S8_BF16_ANCHOR", stage_s8_bf16_anchor)

    sx = run_stage("SX_EXPLORE", stage_sx_explore)

    s9 = run_stage("S9_REPORT", stage_s9_report)

    return {"plan": plan,
            "stages": {k: (STATE.get(k) or {}).get("status") for k in STAGES}}


def main(argv=None):
    """CLI 入口：python -m trlab.run --run-s4 [--set KEY=JSON]"""
    ap = argparse.ArgumentParser(prog="trlab",
                                 description="transformer-residual-lab Kaggle 验证台")
    ap.add_argument("--run-s2", action="store_true", help="编译 llama.cpp（S2）")
    ap.add_argument("--run-s3", action="store_true", help="S2+S3：生成 imatrix")
    ap.add_argument("--run-s4", action="store_true", help="S2+S3+S4+S5：量化 IQ2_XS/IQ4_XS")
    ap.add_argument("--force-rerun", action="store_true", help="忽略断点强制重跑")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="覆盖任意 CFG 键，例如 --set IMATRIX_CTX=256")
    args = ap.parse_args(argv)

    over = {"FORCE_RERUN": True} if args.force_rerun else {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            over[k] = json.loads(v)
        except Exception:
            over[k] = v
    run(run_s2=args.run_s2 or None, run_s3=args.run_s3 or None,
        run_s4=args.run_s4 or None, **over)
    print_artifacts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
