"""S9 汇总报告与产物清单。报告数字只能来自 results.json。"""

from .config import CFG, FORMATS, LOGS, MDLS, NOTEBOOK_VERSION, OUT, S4_PINNED, STAGES, THIRD_PARTY_IN_COMPARISON
from .core import STATE, STATE_PATH, _bench_summary, _fingerprint, _get, _rho_from_nll, write_json

import json
from datetime import datetime, timezone
from pathlib import Path

def _s2_report_lines(r):
    """S9 里复用的 S2 摘要（结果字典 → markdown 行）。"""
    L = []
    src = r.get("source") or {}
    L.append(f"- 源码：`{r.get('src')}`（来源 `{src.get('origin')}`）")
    if src.get("archive_sha256"):
        cached = "，复用已下载的包" if src.get("archive_cached") else ""
        L.append(f"- 归档 sha256：`{src['archive_sha256'][:16]}…` 与预登记值**一致**"
                 f"（{src.get('archive_bytes', 0) / 2**20:.1f} MiB{cached}）")
    L.append(f"- commit：`{r.get('commit')}`，来源 `{r.get('commit_source')}`，"
             f"校验 = **{'通过' if r.get('commit_verified') else '未通过/未校验'}**")
    reuse = r.get("build_reuse") or ("rebuilt" if not r.get("reused") else "existing_build")
    L.append(f"- 编译产物：**{reuse}**"
             + (f"（{r['reuse_detail']}）" if r.get("reuse_detail") else "")
             + ("；**零编译复用**" if r.get("reused") else "；本次完整编译"))
    L.append(f"- CUDA：{r.get('cuda')}"
             + (f"（arch {r.get('cuda_arch')}）" if r.get("cuda_arch") else "")
             + f"；NO_VMM：{r.get('cuda_no_vmm')}"
             + f"；磁盘剩余 {r.get('disk_free_gb')} GiB")
    tc = r.get("toolchain") or {}
    L.append(f"- 工具链：cmake {tc.get('cmake')}；nvcc {tc.get('nvcc')}；"
             f"g++ {tc.get('gxx')}")
    if r.get("version"):
        L.append(f"- 二进制版本：`{str(r['version'])[:160]}`")
    L.append("- 四个二进制：")
    for k, v in (r.get("bins") or {}).items():
        L.append(f"  - `{k}` → `{v}`")
    lg = r.get("logs") or {}
    L.append(f"- 日志：`{lg.get('cmake')}`、`{lg.get('build')}`")
    return L



def _s3_report_lines(r):
    """S9 里复用的 S3 摘要（结果字典 → markdown 行）。"""
    L = []
    if r.get("skipped"):
        L.append(f"- **S3 未运行，复用已有 imatrix**：`{r.get('imatrix')}`，"
                 f"{r.get('bytes', 0) / 1024:.0f} KB，{r.get('n_tensors')} 个张量")
        L.append(f"- sha256：`{r.get('sha256')}`")
        L.append(f"- 可信度：**{r.get('trust')}** —— {r.get('why')}")
        L.append(f"- {r.get('note')}")
        for w in r.get("warnings") or []:
            L.append(f"- !! {w}")
        return L
    L.append(f"- imatrix：`{r.get('imatrix')}`，{r.get('bytes', 0) / 1024:.0f} KB，"
             f"{r.get('n_tensors')} 个张量")
    if r.get("sha256"):
        L.append(f"- sha256：`{r['sha256'][:16]}…`")
    L.append(f"- base：`{Path(str(r.get('base_gguf'))).name}`"
             f"（{r.get('base_kind')}，{r.get('base_bytes', 0) / 2**30:.2f} GiB，"
             f"来源 `{r.get('base_origin')}`）")
    if (r.get("base_download") or {}).get("sha256"):
        L.append(f"- base 校验：sha256 `{r['base_download']['sha256'][:16]}…`"
                 f"（来源 `{r['base_download'].get('origin')}`）")
    if r.get("base_repo"):
        L.append(f"- base 来源：`{r['base_repo']}@{str(r.get('base_revision'))[:12]}`"
                 f"（期望 sha256 `{str(r.get('base_expected_sha256'))[:16]}…`，"
                 f"期望 {r.get('base_expected_bytes', 0) / 2**30:.2f} GiB）"
                 f"；原文件当前{'在' if r.get('base_file_present_now') else '**不在**（已回收）'}")
    L.append(f"- 校准语料：`{Path(str(r.get('calib_txt'))).name}`，"
             f"{r.get('calib_bytes', 0) / 1024:.0f} KB，来源 `{r.get('calib_origin')}`"
             + (f"，声明 `{r.get('calib_source')}`" if r.get("calib_source") else ""))
    L.append(f"- 规模：ctx={r.get('ctx')} × chunks={r.get('chunks')} = "
             f"{r.get('need_tokens')} token（pilot）")
    L.append(f"- -ngl：初判 {r.get('ngl_initial')} → 实际 **{r.get('ngl')}**"
             f"（{'GPU' if r.get('on_gpu') else '纯 CPU'}；显存总量 "
             f"{r.get('vram_total_gb')} GiB，预留 {r.get('vram_margin_gb')} GiB）")
    att = r.get("ngl_attempts") or []
    if len(att) > 1 or any(not a.get("ok") for a in att):
        L.append(f"- -ngl 尝试记录：{json.dumps(att, ensure_ascii=False)}")
    if r.get("q8_reclaimed"):
        L.append(f"- Q8_0 已回收：释放 {r.get('q8_reclaimed_bytes', 0) / 2**30:.2f} GiB"
                 f"（{r.get('q8_reclaim_reason')}）")
    elif "q8_reclaimed" in r:
        L.append(f"- Q8_0 未回收：{r.get('q8_reclaim_reason')}")
    if r.get("imatrix_base_from_approx"):
        L.append(f"- **声明**：imatrix 由 `{r.get('base_kind')}` 算出，激活统计量"
                 f"是对 f16 的近似；这比『从 Q8_0 二次量化』好，但不等于 f16 口径。")
    return L



def _s4_report_lines(r):
    """S9 第 5 节里复用的 S4 摘要：来源、同源性核验等级、两个产物。"""
    L = []
    f = r.get("_fairness") or {}
    L.append("### 量化来源与同源性（S4_QUANT）\n")
    L.append(f"- 量化 base：`{Path(str(f.get('quant_base'))).name}`，"
             f"kind `{f.get('quant_base_kind')}`，来源 `{f.get('quant_base_origin')}`")
    if f.get("quant_base_download"):
        d = f["quant_base_download"]
        if d.get("sha256"):
            L.append(f"- base 校验：sha256 `{d['sha256'][:16]}…`"
                     f"（{'已核对' if d.get('sha256_checked') else '未核对'}）")
    L.append(f"- base 来源：`{f.get('quant_base_repo')}@"
             f"{str(f.get('quant_base_revision'))[:12]}`"
             f"（期望 sha256 `{str(S4_PINNED['sha256'])[:16]}…`）")
    L.append(f"- imatrix：`{Path(str(f.get('imatrix'))).name}`，"
             f"sha256 `{str(f.get('imatrix_sha256'))[:16]}…`，"
             f"可信度 **{f.get('imatrix_trust')}**")
    lvl = f.get("imatrix_base_consistency_level")
    lvl_txt = {"file": "文件级（逐张量比形状，最强）",
               "provenance": "provenance 级（sha256 + 来源 + 结构三项核验）"}.get(lvl, str(lvl))
    L.append(f"- 同源性核验等级：**{lvl_txt}**")
    if f.get("imatrix_base_consistent", {}).get("verified_by"):
        L.append(f"- 核验依据：{f['imatrix_base_consistent']['verified_by']}")
    if lvl == "provenance":
        L.append("- **声明**：imatrix 的 base（Q8_0）已按设计回收，本条为 provenance 级"
                 "核验 —— 弱于文件级逐张量比形状，但 sha256、来源 repo/revision、"
                 "张量名/层数/输入通道三项互相独立，且来源核对比『形状一致』强。")
    L.append(f"- 效力范围：`{f.get('base_scope')}`；`verified = {f.get('verified')}`")
    L.append("")
    L.append("| 格式 | 文件 GiB | 有效 bpw | 主导类型 | 复用 | sha256 |")
    L.append("|---|---|---|---|---|---|")
    for fmt in FORMATS:
        d = r.get(fmt)
        if not d:
            continue
        L.append(f"| {fmt} | {d['file_gb']:.3f} | {d['bpw_effective']:.4f} | "
                 f"{d['dominant_type']} | {'是' if d['reused'] else '否'} | "
                 f"`{str(d.get('sha256'))[:16]}…` |")
    L.append("")
    return L



def _s4_preflight_lines(r):
    """S4 预检摘要（run() 里打印，S9 第 4.5 节复用）。"""
    L = []
    b = r.get("bf16") or {}
    L.append("### S4 预检（编译/下载前的磁盘与暂存盘核查）\n")
    L.append(f"- working：空闲 {r.get('work_free_gb')} GiB"
             f"（峰值需要 {r.get('work_peak_gb')} GiB）")
    L.append(f"- BF16 落点：**{b.get('source') or '无'}**"
             + (f" → `{b.get('path')}`" if b.get("path") else "")
             + (f"（{b['note']}）" if b.get("note") else ""))
    im = r.get("imatrix") or {}
    L.append(f"- imatrix：{im.get('state')} —— {im.get('note', '')}")
    sc = [s for s in (r.get("scratch") or []) if s.get("usable")]
    if sc:
        L.append(f"- 工作区外暂存盘：{len(sc)} 个可用"
                 f"（{', '.join(s['path'] + ' ' + str(s.get('free_gb')) + ' GiB' for s in sc)}）")
    else:
        L.append("- 工作区外暂存盘：无（本机/macOS 上这是正常结果）")
    if r.get("s3_will_run"):
        L.append("- 预计额外耗时：S3 要下 8.11 GiB Q8_0 + 重算 imatrix（约 10 分钟，**估算**）")
    L.append("")
    return L



def stage_s9_report(deadline):
    assert THIRD_PARTY_IN_COMPARISON is False, "第三方量化禁止进入公平比较表"

    env = _get("S0_ENV")
    smoke = _get("S1_SMOKE")
    s3r, s4r = _get("S3_IMATRIX"), _get("S4_QUANT")
    s5r, s6r = _get("S5_BYTES"), _get("S6_NLL")
    s7r, s8r, sxr = _get("S7_QUALITY"), _get("S8_BF16_ANCHOR"), _get("SX_EXPLORE")

    L = []
    L.append("# transformer-residual-lab · Kaggle 运行报告\n")
    L.append(f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append(f"- Notebook 版本：v{NOTEBOOK_VERSION}")
    L.append(f"- 模型：`{CFG['MODEL_ID']}`（{CFG['PARAM_COUNT']/1e9:.2f} B 参数）\n")

    # ---- 0 运行状态（v1.1.2：失败不许被"报告生成了"掩盖）----
    _failed = [(k, v) for k, v in STATE.items()
               if isinstance(v, dict) and v.get("status") == "error"]
    _fail_doc = None
    _fp = OUT / "FAILURE.json"
    if _fp.exists():
        try:
            _fail_doc = json.loads(_fp.read_text(encoding="utf-8"))
        except Exception:
            _fail_doc = None
    if _failed or _fail_doc:
        L.append("## ⚠ 本次运行**失败**\n")
        L.append("> 这份报告是**失败现场**，不是结果。下面所有\"未运行\"的段落都因为"
                 "前面卡住了，**不要**把它们读成\"跑过但没有差异\"。\n")
        if _fail_doc:
            L.append(f"- 最早根因：`{_fail_doc.get('stage')}` / "
                     f"`{_fail_doc.get('cause') or '未分类'}`")
            _causes = _fail_doc.get("causes") or []
            if _causes:
                L.append(f"- 本次运行的根因清单（去重）：{', '.join(f'`{c}`' for c in _causes)}")
            for _it in (_fail_doc.get("all") or []):
                L.append(f"  - `{_it.get('stage')}`（`{_it.get('cause')}`）："
                         f"{str(_it.get('error'))[:180]}")
            if "binaries_missing" in _causes:
                L.append("  - `binaries_missing` **不需要**单独开 S2：凭据通过后，"
                         "`RUN_EVAL=True` 会自动只编 S2（pinned 源码 36.6 MiB，"
                         "不下载模型、不做 S3/S4）。")
            L.append(f"- 下一步：{_fail_doc.get('next_step')}")
            L.append(f"- 明细：`{_fp}`（含每条失败与 traceback 尾）")
        for _k, _v in _failed:
            L.append(f"- `{_k}` = error：{str(_v.get('error'))[:200]}")
        L.append("")
    elif _get("S6_NLL") is None and _get("S7_QUALITY") is None and CFG.get("RUN_EVAL"):
        L.append("## ⚠ 本次运行没有产出评测数据\n")
        L.append("> `RUN_EVAL=True` 但 S6/S7 都没有结果。先看 "
                 f"`{OUT / 'FAILURE.json'}`。\n")

    # ---- 1 环境 ----
    L.append("## 1. 环境\n")
    if env:
        L.append(f"- 位置：{env['where']}，Python {env['python']}")
        L.append(f"- CPU {env['cpu_count']} 核，RAM {env['ram']}")
        for g in env.get("gpus", []):
            L.append(f"- GPU{g['idx']}：{g['name']}，CC{g['cc']}，{g['total_gb']} GiB，"
                     f"native_bf16={g['native_bf16']}")
        L.append(f"- 可用显存估计：{env['vram_usable_gb_est']} GiB")
        for k, v2 in env.get("disks_gb", {}).items():
            L.append(f"- 磁盘 `{k}`：free {v2.get('free')} GiB")
        for w in env.get("warnings", []):
            L.append(f"- **注意**：{w}")
    else:
        L.append("_S0_ENV 未运行_")
    L.append("")

    # ---- 2 工具链（S2） ----
    L.append("## 2. 工具链（S2_BUILD）\n")
    _s2rec = STATE.get("S2_BUILD") or {}
    if _s2rec.get("result"):
        L.extend(_s2_report_lines(_s2rec["result"]))
    elif _s2rec.get("status") == "error":
        L.append(f"- **S2 失败**：`{str(_s2rec.get('error'))[:400]}`")
        L.append(f"- 结果文件：`{OUT}/S2_BUILD.json`")
        L.append(f"- 日志：`{LOGS}/s2_cmake.log`、`{LOGS}/s2_build.log`")
        if CFG["RUN_EVAL"]:
            L.append("- 工具链没建起来，S6/S7 **被主动跳过**（不是开关没开）；"
                     "报告里评测几节为空是**这次失败的后果**，不是「没跑评测」。")
        else:
            L.append("- 其余阶段没有启动（`STAGES` 里是 False），所以报告里下面几节是空的。")
    else:
        _auth_blocked = bool(_fail_doc) and any(
            c in ("hf_no_token", "hf_unauthorized")
            for c in (_fail_doc.get("causes") or []))
        if _auth_blocked:
            L.append("_本次在**编译之前**就被凭据门拦下了（见上方运行状态），"
                     "所以没有编译。这不是工具链问题。_")
        elif CFG["RUN_S2"]:
            L.append("_S2_BUILD 已启用但本次没有产出结果 —— 这属于异常，请检查 S2 日志。_")
        elif CFG["RUN_EVAL"]:
            L.append("_`RUN_EVAL = True` 且 `llama-perplexity` / `llama-cli` 本 Session "
                     "**已经就绪**，不需要编译（缺了才会自动只编 S2）。这一节为空不代表失败。_")
        else:
            L.append("_S2_BUILD 未运行（`RUN_S2 = False`）。这一节为空不代表失败。_")
    L.append("")

    # ---- 3 smoke ----
    L.append("## 3. Smoke test\n")
    if smoke:
        L.append(f"- SMOKE_OK = **{smoke.get('smoke_ok')}**")
        c = smoke.get("checks", {})
        L.append(f"- IQ2_XS bpw 自检：{c.get('bpw_iq2_xs')}（应为 2.3125）")
        L.append(f"- IQ4_XS bpw 自检：{c.get('bpw_iq4_xs')}（应为 4.25）")
        L.append(f"- 未知 type fail-closed：{c.get('unknown_type_fails_closed')}")
        L.append(f"- 需求字节算术（P0-B 回归）：{c.get('need_bytes_arithmetic_ok')}，"
                 f"8 GiB+2 GiB = {c.get('need_bytes_8gib_plus_2')} GiB")
        L.append(f"- 分组组数（P1-F 回归）：{c.get('split_eval_exact_groups')}")

        # P0-C：绝不假设 bf16 一定可用
        tb = (smoke.get("bench") or {}).get("torch") or {}
        bs = _bench_summary(tb)
        if bs.get("torch_error"):
            L.append(f"- GPU 微基准不可用：{bs['torch_error']}")
        elif bs.get("cuda") is False:
            L.append("- GPU 微基准：无 GPU，跳过")
        else:
            L.append(f"- fp16：{bs['fp16_tflops']} TFLOP/s")
            if bs["bf16_available"]:
                L.append(f"- bf16：{bs['bf16_tflops']} TFLOP/s"
                         f"（相对 fp16 比值 {bs['ratio_bf16_over_fp16']}）")
            else:
                L.append(f"- bf16：**不可用** —— {bs['bf16_error'] or '未测出'}")
                L.append("  → 这不是缺陷。T4（CC 7.5）没有原生 bf16，"
                         "所以锚点只能用 f16，且 bf16 相关结论一律不可引用。")
            if bs.get("copy_gb_per_s"):
                L.append(f"- 拷贝带宽（派生值，非 DRAM 计数器）：{bs['copy_gb_per_s']} GB/s")
        dsk = (smoke.get("bench") or {}).get("disk") or {}
        if dsk:
            L.append(f"- 磁盘写：{dsk.get('mb_per_s')} MB/s")
    else:
        L.append("_S1_SMOKE 未运行_")
    L.append("")

    # ---- 4 imatrix ----
    L.append("## 4. imatrix（S3_IMATRIX）\n")
    if s3r:
        L.extend(_s3_report_lines(s3r))
    elif (STATE.get("S3_IMATRIX") or {}).get("status") == "error":
        _e = str((STATE.get("S3_IMATRIX") or {}).get("error"))[:400]
        L.append(f"- **S3 失败**：`{_e}`")
        L.append(f"- 结果文件：`{OUT}/S3_IMATRIX.json`")
        L.append(f"- 日志：`{LOGS}/s3_imatrix.log`")
        L.append("- Q8_0 与下载断点**未被删除**，修复后重跑会直接复用/续传。")
    else:
        L.append("_S3_IMATRIX 未运行（`RUN_S3 = False`）。这一节为空不代表失败。_")
    L.append("")

    # ---- 5 字节 ----
    L.append("## 5. 实际字节\n")
    scope = ((s4r or {}).get("_fairness") or {}).get("base_scope")
    if scope == "requantized_pilot_only":
        L.append("> **base 本身是二次量化**（`BASE_KIND` 是量化类型，已显式打开 "
                 "`ALLOW_REQUANTIZE`）。IQ2/IQ4 之间的配对差仍然同源可比，"
                 "但**不能**作为『相对原始 checkpoint BF16』的高保真低比特结论。\n")
    if s4r:
        L.extend(_s4_report_lines(s4r))
    elif (STATE.get("S4_QUANT") or {}).get("status") == "error":
        _e = str((STATE.get("S4_QUANT") or {}).get("error"))[:400]
        L.append(f"- **S4 失败**：`{_e}`")
        L.append(f"- 结果文件：`{OUT}/S4_QUANT.json`")
        L.append("- S4 失败**不会**删任何输入文件；imatrix 与 base 都可直接复用。")
    else:
        L.append("_S4_QUANT 未运行（`RUN_S4 = False`）。_")
    L.append("")
    if s5r:
        L.append("| 文件 | 文件 GiB | 有效 bpw | embed+head GiB | 拆分可用 |")
        L.append("|---|---|---|---|---|")
        for name, d in s5r["per_model"].items():
            ok = d.get("byte_breakdown_available", False)
            if ok:
                eh = d["bytes_per_class"].get("embed/head", 0) / 2**30
                bpw = f"{d['bpw_effective']:.4f}" if d["bpw_effective"] is not None else "n/a"
                L.append(f"| {name} | {d['file_gb']:.3f} | {bpw} | {eh:.3f} | 是 |")
            else:
                L.append(f"| {name} | {d['file_gb']:.3f} | — | — | **否**（未知 ggml type） |")
        if s5r.get("ratios"):
            L.append("")
            L.append(f"字节比（IQ4/IQ2）：{s5r['ratios'].get('iq4_over_iq2')}，"
                     f"节省 {s5r['ratios'].get('bytes_saved_pct')}%")
        if s5r.get("byte_breakdown_unavailable"):
            L.append("")
            L.append("**字节拆分不可用的文件**（不产出 bpw，避免假数）：")
            for u in s5r["byte_breakdown_unavailable"]:
                L.append(f"- `{u['name']}`：{u['reason'][:90]}")
    else:
        L.append("_S5_BYTES 未运行_")
    L.append("")

    # ---- 6 NLL ----
    L.append("## 6. 配对 NLL\n")
    if s6r:
        iso = s6r.get("isolation", {})
        prov = s6r.get("provenance") or {}
        L.append(f"- 评测语料来源：{s6r.get('eval_source') or '未声明'}")
        L.append(f"- 校准/评测隔离：{iso.get('ok')}（重叠 {iso.get('overlap_lines')} 行）")
        L.append(f"- **公平性已核验**：{s6r.get('fairness_verified')}"
                 f"（级别 `{s6r.get('fairness_level') or '未声明'}`）")
        if s6r.get("fairness_level") == "hf_sidecar_declared":
            L.append("  - 这是**声明级**：数字来自 HF 上的 provenance sidecar，"
                     "本次没有重算核验。可核的只有 sidecar 之间的一致性与 GGUF 对 lfs.oid。")
        ins = prov.get("inputs") or {}
        if ins:
            L.append("- 输入：" + "；".join(
                f"{f} ← {v.get('source')}（{v.get('bytes', 0) / 2**30:.2f} GiB，"
                f"sha256 {'已核' if v.get('sha256_verified') else '未核'}）"
                for f, v in ins.items()))
        L.append(f"- 分组数：{s6r['groups']}，每组 {s6r.get('eval_chunks_per_group')} chunk × "
                 f"{s6r.get('eval_ctx')} ctx（两个模型同设置）")
        if s6r.get("measured_throughput"):
            mt = s6r["measured_throughput"]
            L.append(f"- 实测吞吐：{mt['tok_per_sec']} tok/s（{mt['elapsed_sec']} s，"
                     f"{mt['tokens']} token；只对同机器同设置可比）")
        dev = s6r.get("device_by_fmt") or {}
        if dev:
            L.append("- 执行设备（逐格式判定，**一次只装一个模型**）：" + "；".join(
                f"{f} = {d.get('device')}（`-ngl {d.get('ngl')}`，{d.get('why')}）"
                for f, d in dev.items()))
        elif prov.get("gpu") is False:
            L.append("- 执行设备：**纯 CPU**（`EVAL_ALLOW_GPU=False`，未消耗 GPU 额度）")
        if s6r.get("upload"):
            up = s6r["upload"]
            if up.get("skipped"):
                L.append(f"- 结果出站：**未上传**（{up.get('why')}）—— 结果只在本 Session 里")
            else:
                L.append(f"- 结果出站：HF 远端复核 `ok={up.get('remote_verified')}`"
                         f"（commit {up.get('commit')}）")
        L.append("")
        L.append("| 格式 | 平均 NLL | 平均 PPL | NLL 标准差 |")
        L.append("|---|---|---|---|")
        for fmt, a in s6r["aggregate"].items():
            L.append(f"| {fmt} | {a['mean_nll']:.4f} | {a['mean_ppl']:.4f} | {a['nll_sd']:.4f} |")
        st = s6r["paired_delta_nll_iq2_minus_iq4"]
        L.append("")
        L.append(f"配对差（IQ2 − IQ4）：均值 **{st['mean']:+.4f} nats**，"
                 f"字符加权 {st.get('char_weighted_mean', float('nan')):+.4f}，"
                 f"95% CI {st['ci95']}，n={st['n']}")
        L.append(f"\n判读：{s6r.get('verdict')}")
        if s6r.get("scope_note"):
            L.append(f"\n**结论效力**：{s6r['scope_note']}")
        L.append(f"\n统计口径限制：{s6r.get('statistics_note')}")
    else:
        _s6rec = STATE.get("S6_NLL") or {}
        if _s6rec.get("status") == "error":
            L.append(f"_S6_NLL **失败**：`{str(_s6rec.get('error'))[:300]}` —— "
                     "没有 NLL 数据，任何能力结论都不成立。_")
        elif CFG["RUN_EVAL"]:
            L.append("_S6_NLL 未运行 —— 本次在评测开始前就停下了（见上方运行状态），"
                     "**S6/S7 not yet evaluated**。_")
        else:
            L.append("_S6_NLL 未运行 —— **S6/S7 not yet evaluated**（这两个量化对模型能力"
                     "的影响目前没有任何实测结论）_")
    L.append("")

    # ---- 7 质量 ----
    L.append("## 7. 有限配对质量检查\n")
    if s7r:
        L.append(f"- 贪心续写完全一致：{s7r['identical_outputs']}/{s7r['n_prompts']}"
                 f"（{s7r['identical_rate']}）")
        L.append(f"- 平均 repeat4（越低越好）：{s7r['degenerate']}")
        setts = s7r.get("settings") or {}
        if setts:
            L.append(f"- 设置：temp={setts.get('temperature')} seed={setts.get('seed')} "
                     f"n_predict={setts.get('n_predict')} -ngl={setts.get('ngl')}")
        for lim in s7r["limits"]:
            L.append(f"- 限制：{lim}")
    else:
        _s7rec = STATE.get("S7_QUALITY") or {}
        if _s7rec.get("status") == "error":
            L.append(f"_S7_QUALITY **失败**：`{str(_s7rec.get('error'))[:300]}`_")
        elif CFG["RUN_EVAL"]:
            L.append("_S7_QUALITY 未运行 —— 本次在评测开始前就停下了（见上方运行状态）。_")
        else:
            L.append("_S7_QUALITY 未运行 —— **S6/S7 not yet evaluated**_")
    L.append("")

    # ---- 8 锚点 ----
    L.append("## 8. 高精度锚点\n")
    if s8r:
        L.append(f"- 状态：**{s8r.get('status')}**")
        L.append(f"- 锚点精度：{s8r.get('base_kind')}（tier={s8r.get('tier')}）")
        L.append(f"- 需要 {s8r.get('required_gb')} GiB / 可用 {s8r.get('vram_usable_gb')} GiB")
        if s8r.get("status") != "ok":
            L.append(f"- 跳过原因：{s8r.get('reason')}")
        else:
            L.append(f"- 锚点 NLL：{s8r.get('mean_nll'):.4f}（{s8r.get('validity_scope')}）")
            if s6r and s8r.get("validity_scope") == "high_precision":
                rho = _rho_from_nll(
                    anchor_nll=s8r["mean_nll"],
                    q4_nll=s6r["aggregate"]["IQ4_XS"]["mean_nll"],
                    q2_nll=s6r["aggregate"]["IQ2_XS"]["mean_nll"])
                if rho["rho"] is not None:
                    L.append(f"- rho = dNLL(Q2,Q4)/dNLL(Q4,anchor) = {rho['rho']:.4f}")
                    L.append(f"  （dNLL(Q4,anchor)={rho['d4']:+.4f}，"
                             f"dNLL(Q2,anchor)={rho['d2']:+.4f}；均为「后减前」）")
                else:
                    L.append(f"- **不计算 rho**：{rho['reason']}")
            elif s6r:
                L.append("- **不计算 rho**：锚点是近似参照，不构成有效性证据。")
        L.append(f"- dtype 偏差声明：{s8r.get('anchor_dtype_note')}")
    else:
        L.append("_S8_BF16_ANCHOR 未运行_")
    L.append("")

    # ---- 9 第三方 ----
    L.append("## 9. 第三方量化（仅探索）\n")
    if sxr:
        L.append(f"- 文件：`{sxr['path']}`，{sxr['file_gb']} GiB，"
                 f"bpw={sxr.get('bpw_effective')}")
        L.append(f"- **不参与公平比较**：{sxr['provenance']['why_not']}")
    else:
        L.append("_SX_EXPLORE 未运行_")
    L.append("")

    # ---- 10 结论边界 ----
    L.append("## 10. 结论边界\n")
    L.append("1. 所有字节数字来自**文件读取**；未知 ggml type 的文件不出 bpw。")
    L.append("2. 配对 CI 基于**分组** bootstrap；**CI 跨 0 表示未测出差异，不是等价**。")
    L.append("3. `NLL_GROUPS` 只用于 pilot。组间 token 数不严格相等，正式 CI 需按独立文档重设计。")
    L.append("4. 单张 T4、8B、单一评测语料，**不能外推到 27B 或 M4 Pro**。")
    L.append("5. T4 无原生 bf16；锚点用 f16 时与论文口径有偏差，已在第 8 节声明。")
    L.append("6. Q8_0 之类属二次量化，**不能**作为『相对原始 checkpoint BF16』的有效性证据。"
             "S4 默认拒绝从量化 base 量化；显式 `ALLOW_REQUANTIZE=True` 时产物标 "
             "`requantized_pilot_only`。")
    L.append("7. 第三方量化不进比较表；进了就是缺陷（本 cell 开头有断言）。")
    L.append("8. `guard()` 只在阶段入口检查显存，**不能阻止外部进程在运行中 OOM**。")
    L.append("9. S4 只在产物 provenance 指纹（base + imatrix + commit + 参数）完全一致时复用旧 "
             "GGUF；指纹不符的旧文件被改名成 `.stale-*`，不会被静默当成新结果。")
    L.append("10. **二进制复用不等于版本已校验。** `build_reuse=attached_input` 时源码不在本 "
             "Session 内，commit 只能记为 `declared`（未校验）；第 2 节会写清是哪一档。")
    L.append("11. **imatrix base 是 Q8_0，不是 f16。** 激活统计量是对 f16 的近似；这比『从 "
             "Q8_0 二次量化』好，但不等于 f16 口径，不能当作『相对原始 checkpoint 高精度』"
             "的证据。")
    L.append("12. Q8_0 在 imatrix 验证通过且结果记录写盘之后被删除；因此报告里的 `q8_reclaimed`"
             "为真时，磁盘上已无该文件，**重跑会重新下载**（有断点续传，不会从零开始）。")
    L.append("")

    md = "\n".join(L)
    (OUT / "report.md").write_text(md, encoding="utf-8")

    # 自一致性：写 results.json 时本阶段还没被 run_stage 登记进 STATE，
    # 直接取会是 null，与日志里的 "S9_REPORT OK" 矛盾。显式补上并标明是自报。
    stage_snapshot = {}
    for k in STAGES:
        rec = STATE.get(k) or {}
        stage_snapshot[k] = {"status": rec.get("status"),
                             "elapsed_sec": rec.get("elapsed_sec"),
                             "fingerprint": rec.get("fingerprint"),
                             "error": rec.get("error")}
    stage_snapshot["S9_REPORT"] = {
        "status": "ok",
        "elapsed_sec": None,
        "fingerprint": _fingerprint("S9_REPORT"),
        "error": None,
        "self_reported": True,
        "note": "本阶段在写盘时尚未登记进 STATE；status 由这里显式置为 ok",
    }

    results = {
        "notebook_version": NOTEBOOK_VERSION,
        "model": CFG["MODEL_ID"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": stage_snapshot,
        "config_snapshot": {k: v2 for k, v2 in CFG.items() if k != "STAGE_BUDGET_SEC"},
        "s0_env": env, "s1_smoke": smoke,
        "s3_imatrix": s3r, "s4_quant": s4r, "s5_bytes": s5r,
        "s6_nll": s6r, "s7_quality": s7r, "s8_anchor": s8r, "sx_explore": sxr,
        "disclaimer": ("自建量化与第三方量化的结果不可混用；"
                       "所有 NLL 均为该文件在该语料上的观测，不构成能力结论。"),
    }
    write_json(OUT / "results.json", results)

    print(md)
    print("=" * 78)
    print("产物:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.stat().st_size:>9d} B  {p.name}")
    return {"report": str(OUT / "report.md"), "results": str(OUT / "results.json"),
            "stage_status": {k: v2["status"] for k, v2 in stage_snapshot.items()}}



def print_artifacts():
    """收尾：产物清单、残留 .partial、旧产物、断点状态。"""
    # 最终产物清单 —— 从这里下载，或点 Kaggle 右上角 "Save Version"
    print("=" * 78)

    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size:>10d} B  {p.name}")

    print("-" * 78)

    for p in sorted(MDLS.glob("*.gguf")):
        print(f"{p.stat().st_size/2**30:>10.3f} GiB  {p.name}")

    leftover = list(MDLS.glob("*.partial")) + list(OUT.glob("*.partial"))

    if leftover:
        print("!! 残留 .partial（说明上次有阶段中途失败）:")
        for p in leftover:
            print(f"   {p}")

    stale = list(MDLS.glob("*.stale-*"))

    if stale:
        print("旧产物（指纹不符被移走，可删）:")
        for p in stale:
            print(f"   {p.stat().st_size/2**30:>8.3f} GiB  {p.name}")

    provs = sorted(MDLS.glob("*.provenance.json"))

    if provs:
        print("产物 provenance:")
        for p in provs:
            rec = json.loads(p.read_text(encoding="utf-8"))
            print(f"   {p.name}  fp={rec.get('quant_fp')}  "
                  f"base_kind={rec.get('base_kind')}  requant={rec.get('requant')}")

    print("-" * 78)

    print(f"断点文件: {STATE_PATH}")

    print("阶段状态:")

    for k in STAGES:
        r = STATE.get(k) or {}
        print(f"  {k:16s} switch={str(STAGES[k]):5s} status={r.get('status', '-'):5s} "
              f"fp={r.get('fingerprint', '-')}"
              f"{('  err: ' + str(r.get('error'))[:60]) if r.get('error') else ''}")

    print("=" * 78)

    print("提醒：/kaggle/working 只有约 20 GB。")

    print("      两个 GGUF 约 2.2 + 4.0 GiB，imatrix 只有几十 MB；不要在 working 里再放 base。")
