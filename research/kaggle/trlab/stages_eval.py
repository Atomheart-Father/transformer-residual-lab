"""S6 配对 NLL + S7 质量抽查 + S8 高精度锚点 + SX 探索。

v1.1：S6/S7 的输入不再要求"本 Session 刚量化过"。本地 `models/` 有就用，
没有就从 Hugging Face 上已存的量化 GGUF 取（`trlab/artifacts.py`），
**不重量化、不下载 BF16、不重做 S3**。评测默认纯 CPU（`EVAL_ALLOW_GPU=False`）。
"""

from .config import CFG, EVAL_PINNED, FORMATS, LOGS, WORK
from .errors import GuardError, UnknownGGMLType
from .paths import _model_path, llama_bin
from .gguf import gguf_scan
from .core import _anchor_decision, _fit_ngl, _get, check_deadline, log, need, paired_bootstrap, run_cmd, run_ppl, split_eval, vram_gb
from .imatrix import resolve_imatrix
from .artifacts import (eval_preflight, fairness_from_sidecars,
                        resolve_eval_models)

import json, os, subprocess, time
from pathlib import Path


def _eval_ngl(nbytes, vram_total_gb):
    """评测要不要上 GPU —— **默认不**。

    `EVAL_ALLOW_GPU=False` 时一律 `-ngl 0`（纯 CPU）：GPU 会消耗额度，
    必须由用户显式打开。这条不能靠"显存够就自动用"来猜。
    """
    if not CFG.get("EVAL_ALLOW_GPU"):
        return 0
    usable = (vram_total_gb - 1.0) if vram_total_gb else 0.0
    return _fit_ngl(nbytes, usable, CFG["NGL_MARGIN_GB"])


def _fairness_for_s6(models):
    """S4 的运行时记录优先（那是真核验过的），否则退到 HF sidecar 的声明级。"""
    s4r = _get("S4_QUANT") or {}
    fair = s4r.get("_fairness") or {}
    if fair:
        return fair, "s4_runtime"
    return fairness_from_sidecars(models), "hf_sidecar_declared"


def stage_s6_nll(deadline):
    # ---- 0. 预检：先算账，再决定要不要拉 7 GiB ----
    pf = eval_preflight(deadline, do_download=False)
    models = resolve_eval_models(deadline)      # 现在才真下载 + 校验
    eval_txt = Path(CFG["EVAL_TXT"]) if CFG.get("EVAL_TXT") else need("EVAL_TXT")
    if not CFG["EVAL_SOURCE"]:
        log("!! CFG['EVAL_SOURCE'] 未设置，provenance 里会缺这一项")

    # ---- 公平性：S4 运行时记录优先，否则 sidecar 声明级 ----
    fair, fair_level = _fairness_for_s6(models)
    if fair.get("same_imatrix") is False:
        raise RuntimeError(
            "两个量化用的 imatrix 不是同一份（sha256 不一致），配对比较无效。")
    cons = fair.get("imatrix_base_consistent") or {}
    if cons.get("ok") is False:
        raise RuntimeError(
            f"imatrix 的 base 与量化 base 不是同一个 checkpoint（{cons.get('why')}），"
            "配对比较无效。")
    fairness_verified = bool(fair.get("verified"))
    base_scope = fair.get("base_scope")
    scope_note = None
    if base_scope == "requantized_pilot_only":
        scope_note = ("base 本身是量化文件（已显式 ALLOW_REQUANTIZE）：IQ2 与 IQ4 都从它再量化，"
                      "两者之间的配对差仍然同源可比，但**不能**作为"
                      "『相对原始 checkpoint BF16』的高保真低比特结论。")
        log("!! " + scope_note)
    elif fair.get("imatrix_base_from_approx"):
        scope_note = ("量化 base 是高精度，但 imatrix 由量化过的 base 算出："
                      "激活是对高精度的近似，影响远小于二次量化，仍需在结论里声明。")
    if fair_level == "hf_sidecar_declared":
        log("!! 公平性来自 HF 上的 provenance sidecar（声明级，本次未重算）—— "
            "结论里必须标注 verified_level=hf_sidecar_declared")

    # ---- 校准/评测隔离断言（内容级）----
    isolation = {"ok": None}
    if CFG["CALIB_TXT"] and Path(CFG["CALIB_TXT"]).exists():
        c = Path(CFG["CALIB_TXT"]).read_text(errors="replace")
        e = eval_txt.read_text(errors="replace")
        cset = {ln.strip()[:200] for ln in c.split("\n") if len(ln.strip()) > 40}
        eset = {ln.strip()[:200] for ln in e.split("\n") if len(ln.strip()) > 40}
        ov = cset & eset
        isolation.update(
            calib=str(CFG["CALIB_TXT"]), calib_source=CFG["CALIB_SOURCE"],
            eval=str(eval_txt), eval_source=CFG["EVAL_SOURCE"],
            calib_lines=len(cset), eval_lines=len(eset),
            overlap_lines=len(ov), ok=(len(ov) == 0))
        if ov:
            raise RuntimeError(
                f"校准集与评测集有 {len(ov)} 行重叠（前 200 字符比对）。"
                "必须换 split 后再跑，否则 NLL 会被 imatrix 污染。")
        log(f"隔离检查通过：calib {len(cset)} 行 / eval {len(eset)} 行 / 重叠 0")
    else:
        log("!! 没有校准语料路径，无法做隔离断言 —— 结论里必须标注这一点")

    if len(models) < 2:
        raise GuardError(
            f"需要两个格式的文件，现有 {list(models)}。\n"
            "  → 本地没有就去 HF 取：RUN_EVAL=True（或 CFG['HF_FETCH_ARTIFACTS']=True）。")

    groups, meta = split_eval(eval_txt, CFG["NLL_GROUPS"], WORK / "eval_groups")
    log(f"评测集切成 {len(groups)} 组：" +
        ", ".join(f"g{m['group']}({m['paras']}段/{m['chars']}字符)" for m in meta))

    per_model = {}
    t_start = time.time()
    for fmt, rec in models.items():
        mp = Path(rec["path"])
        per_model[fmt] = []
        for i, g in enumerate(groups):
            check_deadline(deadline, f"S6 {fmt} g{i}")
            r = run_ppl(mp, g, deadline, f"{fmt}_g{i}")
            r["group"] = i
            r["chars"] = meta[i]["chars"]
            r["model_source"] = rec.get("source")
            per_model[fmt].append(r)
            log(f"  {fmt} g{i}: PPL={r['ppl']:.4f} NLL={r['nll']:.4f}")
    elapsed = time.time() - t_start

    d = [per_model[FORMATS[0]][i]["nll"] - per_model[FORMATS[1]][i]["nll"]
         for i in range(len(groups))]
    stats = paired_bootstrap(d)

    # 组间字符数不等时，均值口径会有偏；给一个字符加权版本作对照
    w = [m["chars"] for m in meta]
    tot_w = sum(w) or 1
    stats["char_weighted_mean"] = sum(di * wi for di, wi in zip(d, w)) / tot_w

    agg = {}
    for fmt, vals in per_model.items():
        mu = sum(x["nll"] for x in vals) / len(vals)
        agg[fmt] = {
            "mean_nll": mu,
            "mean_ppl": sum(x["ppl"] for x in vals) / len(vals),
            "nll_sd": (sum((x["nll"] - mu) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5,
        }

    # 实测速率回填：把"假设值"换成这一次的真实值，供下一轮估成本
    tokens_done = len(groups) * CFG["NLL_CHUNKS_PER_GROUP"] * CFG["NLL_CTX"] * len(models)
    measured = {"elapsed_sec": round(elapsed, 1), "tokens": tokens_done,
                "tok_per_sec": round(tokens_done / max(1e-6, elapsed), 2),
                "note": "这是**这一次**的实测吞吐，只对同机器同设置可比"}

    out = {
        "isolation": isolation,
        "fairness_verified": fairness_verified,
        "fairness_level": fair_level,
        "fairness_record": fair,
        "base_scope": base_scope,
        "scope_note": scope_note,
        "eval_source": CFG["EVAL_SOURCE"],
        "eval_ctx": CFG["NLL_CTX"],
        "eval_chunks_per_group": CFG["NLL_CHUNKS_PER_GROUP"],
        "groups": len(groups),
        "group_meta": meta,
        "per_group": per_model,
        "aggregate": agg,
        "paired_delta_nll_iq2_minus_iq4": stats,
        "deltas": [round(x, 6) for x in d],
        "resources": pf.get("resources"),
        "cost_estimate": pf.get("cost"),
        "measured_throughput": measured,
        "statistics_note": ("NLL_GROUPS 只用于 pilot 估计。bootstrap 的 n 是**组数**不是 token 数，"
                            "组间 token 数也不严格相等；正式 CI 应按独立文档重新设计。"),
        "provenance": {
            "inputs": {f: {"path": r["path"], "bytes": r["bytes"],
                           "source": r.get("source"),
                           "sha256_verified": r.get("sha256_verified")}
                       for f, r in models.items()},
            "base_gguf": CFG["BASE_GGUF"], "base_kind": CFG["BASE_KIND"],
            "imatrix": str(resolve_imatrix()),
            "imatrix_base": (fair or {}).get("imatrix_base"),
            "base_scope": base_scope,
            # v1.1：输入可能来自 HF 上已存的产物 —— 那不是本次量化，
            # 所以 self_quantized 只能是"来源自述"，不能写成 True。
            "self_quantized": fair_level == "s4_runtime",
            "fair_comparison": fairness_verified,
            "fairness_level": fair_level,
            "gpu": bool(CFG["EVAL_ALLOW_GPU"]),
            "note": ("输入来自 " + ", ".join(sorted({r.get("source") or "?" for r in models.values()}))
                     + f"；公平性级别 {fair_level}。"
                     "同 base、同 imatrix、同 token 窗口；fair_comparison=False 表示"
                     "本次没有重算核验（sidecar 声明级）。"),
        },
    }

    print("=" * 78)
    for fmt, a in agg.items():
        print(f"{fmt:10s} mean NLL={a['mean_nll']:.4f}  mean PPL={a['mean_ppl']:.4f}  sd={a['nll_sd']:.4f}")
    print(f"配对差 (IQ2 - IQ4): mean={stats['mean']:+.4f}  "
          f"char_weighted={stats['char_weighted_mean']:+.4f}  "
          f"CI95={stats['ci95']}  n={stats['n']}")
    print(f"公平性已核验: {fairness_verified}（级别 {fair_level}）")
    print(f"实测吞吐: {measured['tok_per_sec']} tok/s（{elapsed:.0f} s，{tokens_done} token）")
    print("=" * 78)

    ci = stats.get("ci95")
    if ci and ci[0] > 0:
        verdict = "IQ2_XS 显著差于 IQ4_XS（CI 全在 0 以上）"
    elif ci and ci[1] < 0:
        verdict = "IQ2_XS 显著好于 IQ4_XS（异常，先查口径）"
    else:
        verdict = "CI 跨 0 —— 未测出差异，不等于等价"
    out["verdict"] = verdict
    print("判读:", verdict)
    return out



QUALITY_PROMPTS_DEFAULT = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, the first humans landed on",
    "The chemical formula for table salt is",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
    "The three primary colors of light are",
    "Translate 'good morning' into Chinese:",
]



def probe_help(binary):
    try:
        p = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=90)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""



def _cli_flags():
    """llama-cli 的 flag 名跨版本变过，这里探测而不是硬编码。"""
    h = probe_help(llama_bin("llama-cli"))
    return {
        "no_conv": ("--no-conversation" if "--no-conversation" in h
                    else ("-no-cnv" if "-no-cnv" in h else None)),
        "no_display_prompt": "--no-display-prompt" if "--no-display-prompt" in h else None,
        "temp": "--temp" if "--temp" in h else "-temp",
        "seed": "--seed" if "--seed" in h else "-s",
        "simple_io": "--simple-io" if "--simple-io" in h else None,
    }



def repeat_ratio(text, n=4):
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)



def stage_s7_quality(deadline):
    flags = _cli_flags()
    log(f"llama-cli flags: {flags}")
    cli = llama_bin("llama-cli")

    # v1.1：输入同样走"本地优先 / 否则 HF"，且不重量化。
    models = resolve_eval_models(deadline)
    if len(models) < 2:
        raise GuardError(
            f"需要两个格式的文件，现有 {list(models)}。\n"
            "  → 本地没有就去 HF 取：RUN_EVAL=True（或 CFG['HF_FETCH_ARTIFACTS']=True）。")

    prompts = QUALITY_PROMPTS_DEFAULT[:CFG["QUALITY_PROMPTS"]]
    v = vram_gb()
    # 默认纯 CPU：GPU 会消耗额度，必须 EVAL_ALLOW_GPU=True 才用。
    ngl_by_fmt = {fmt: _eval_ngl(rec["bytes"], (v or {}).get("total"))
                  for fmt, rec in models.items()}
    log(f"S7 -ngl: {ngl_by_fmt}（EVAL_ALLOW_GPU={CFG['EVAL_ALLOW_GPU']}）")

    per_model = {}
    for fmt, rec in models.items():
        mp = Path(rec["path"])
        ngl = ngl_by_fmt[fmt]
        per_model[fmt] = []
        for i, pr in enumerate(prompts):
            check_deadline(deadline, f"S7 {fmt} p{i}")
            cmd = [cli, "-m", mp, "-p", pr, "-n", CFG["QUALITY_N_PREDICT"],
                   flags["temp"], "0", flags["seed"], CFG["SEED"],
                   "-ngl", ngl, "-t", os.cpu_count() or 4]
            for key in ("no_conv", "no_display_prompt", "simple_io"):
                if flags.get(key):
                    cmd.append(flags[key])
            logf = LOGS / f"s7_{fmt}_p{i}.log"
            run_cmd(cmd, logf, deadline)          # rc != 0 直接抛，不再当成功

            txt = logf.read_text(errors="replace")
            body = txt.split(pr, 1)[1] if pr in txt else txt
            body = body.replace("llama_perf", "\nllama_perf").split("\nllama_perf")[0].strip()
            per_model[fmt].append({"prompt": pr, "text": body[:400],
                                   "repeat4": round(repeat_ratio(body), 4),
                                   "chars": len(body)})

    pairs = []
    fa, fb = FORMATS[0], FORMATS[1]
    for i, pr in enumerate(prompts):
        a = per_model[fa][i]["text"]
        b = per_model[fb][i]["text"]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        pairs.append({"prompt": pr, "identical": a == b, "first_divergence_char": k,
                      "len_iq2": len(a), "len_iq4": len(b)})

    n_id = sum(1 for p in pairs if p["identical"])
    out = {
        "n_prompts": len(prompts),
        "identical_outputs": n_id,
        "identical_rate": round(n_id / len(prompts), 4) if prompts else None,
        "pairs": pairs,
        "per_model": per_model,
        "degenerate": {fmt: round(sum(x["repeat4"] for x in v2) / len(v2), 4)
                       for fmt, v2 in per_model.items()},
        "settings": {"temperature": 0, "seed": CFG["SEED"],
                     "n_predict": CFG["QUALITY_N_PREDICT"],
                     "ngl": ngl_by_fmt, "gpu_allowed": bool(CFG["EVAL_ALLOW_GPU"])},
        "provenance": {
            "inputs": {f: {"path": r["path"], "source": r.get("source"),
                           "sha256_verified": r.get("sha256_verified")}
                       for f, r in models.items()},
            "note": "输入来自 HF 上已存的量化产物或本地 S4 产物，本次未重量化",
        },
        "limits": [
            "贪心解码 + 32 token，样本量很小，只能发现整体崩塌，不能排序细微质量",
            "不是标准任务评测（GSM8K / HumanEval 之类），绝对分数不构成能力证据",
            "repeat4 只是退化信号，不是质量分数",
        ],
    }
    print("=" * 78)
    print(f"完全一致的输出: {n_id}/{len(prompts)}  ({out['identical_rate']})")
    print(f"平均 repeat4: {out['degenerate']}")
    for p in pairs:
        print(f"  div@{p['first_divergence_char']:4d}  {p['prompt'][:44]}")
    print("=" * 78)
    return out



def stage_s8_bf16_anchor(deadline):
    """高精度锚点。默认在 T4 上会被显存判定或精度分档拒绝。"""
    anchor = need("BASE_GGUF")
    v = vram_gb()
    total = v["total"] if v else 0.0

    d = _anchor_decision(
        anchor_bytes=anchor.stat().st_size,
        vram_total_gb=total,
        base_kind=CFG["BASE_KIND"],
        allow_approx=CFG["ANCHOR_ALLOW_APPROX"],
        extra_gb=CFG["ANCHOR_EXTRA_GB"],
    )
    d["anchor_path"] = str(anchor)
    d["anchor_dtype_note"] = (
        "T4 是 CC 7.5，没有原生 bf16。锚点请用 f16；用 bf16 文件会走转换路径，"
        "与论文口径不同，必须声明。"
        if str(CFG["BASE_KIND"]).lower() == "bf16" else
        f"锚点 dtype = {CFG['BASE_KIND']}；与论文常用的 BF16 锚点不完全一致，跨平台比较时必须声明。")

    if d["status"] != "ready":
        print(json.dumps(d, indent=1, ensure_ascii=False, default=str))
        return d

    eval_txt = need("EVAL_TXT")
    groups, meta = split_eval(eval_txt, CFG["NLL_GROUPS"], WORK / "eval_groups")
    vals = []
    for i, g in enumerate(groups):
        check_deadline(deadline, f"S8 g{i}")
        r = run_ppl(anchor, g, deadline, f"anchor_g{i}")
        r["group"] = i
        r["chars"] = meta[i]["chars"]
        vals.append(r)
        log(f"  anchor g{i}: PPL={r['ppl']:.4f}")

    d.update(
        status="ok",
        per_group=vals,
        mean_nll=sum(x["nll"] for x in vals) / len(vals),
        mean_ppl=sum(x["ppl"] for x in vals) / len(vals),
        note=("锚点 NLL。validity_scope=high_precision 时才能算 "
              "rho = dNLL(Q2,Q4)/dNLL(Q4,anchor) 并当有效性证据；"
              "approx_only 时只能当量级参考。"),
    )
    print(json.dumps({k: v2 for k, v2 in d.items() if k != "per_group"},
                     indent=1, ensure_ascii=False, default=str))
    return d



def stage_sx_explore(deadline):
    """第三方现成量化：只看字节和 NLL，**永远不进入公平比较表**。"""
    p = need("EXPLORE_GGUF")
    try:
        sc = gguf_scan(p, deadline)
        breakdown_ok = True
        info = {"file_gb": sc["file_gb"], "bpw_effective": sc["bpw_effective"],
                "bytes_per_type": sc["bytes_per_type"]}
    except UnknownGGMLType as e:
        breakdown_ok = False
        info = {"file_gb": round(p.stat().st_size / 2**30, 3),
                "bpw_effective": None, "error": str(e)}

    rec = {
        "path": str(p),
        "byte_breakdown_available": breakdown_ok,
        **info,
        "provenance": {
            "self_quantized": False,
            "fair_comparison": False,
            "why_not": ("第三方 GGUF 的校准集、imatrix、量化脚本、llama.cpp 版本全部未知。"
                        "它的 NLL 只能说明这个文件本身如何，不能与自建 IQ2_XS/IQ4_XS 做公平比较。"),
            "allowed_use": "探索性测试：确认格式可加载、量级是否合理、有没有明显崩塌",
        },
    }

    if CFG["EVAL_TXT"] and Path(CFG["EVAL_TXT"]).exists():
        groups, meta = split_eval(Path(CFG["EVAL_TXT"]), CFG["NLL_GROUPS"], WORK / "eval_groups")
        vals = []
        for i, g in enumerate(groups):
            check_deadline(deadline, f"SX g{i}")
            r = run_ppl(p, g, deadline, f"explore_g{i}")
            r["group"] = i
            vals.append(r)
        rec["mean_nll"] = sum(x["nll"] for x in vals) / len(vals)
        rec["mean_ppl"] = sum(x["ppl"] for x in vals) / len(vals)
        rec["per_group"] = vals
    else:
        rec["mean_nll"] = None
        rec["note"] = "没有评测语料，只做了字节分析"

    print(json.dumps({k: v2 for k, v2 in rec.items() if k != "per_group"},
                     indent=1, ensure_ascii=False, default=str))
    return rec
