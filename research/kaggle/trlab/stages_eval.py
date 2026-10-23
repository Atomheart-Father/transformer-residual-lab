"""S6 配对 NLL + S7 质量抽查 + S8 高精度锚点 + SX 探索。

v1.1：S6/S7 的输入不再要求"本 Session 刚量化过"。本地 `models/` 有就用，
没有就从 Hugging Face 上已存的量化 GGUF 取（`trlab/artifacts.py`），
**不重量化、不下载 BF16、不重做 S3**。

v1.1.1：**默认有 GPU 就用 GPU**（Kaggle 上开的就是 GPU Session，不该在 4 核
CPU 上磨 8B）。装不下才退 CPU，并把原因写进结果。**一次只装一个模型**：
每个格式独立一轮，跑完就释放，绝不把两个量化同时压在显存/内存里。
"""

from .config import CFG, EVAL_PINNED, FORMATS, LOGS, WORK
from .errors import GuardError, UnknownGGMLType
from .paths import _model_path, llama_bin
from .gguf import gguf_scan
from .core import (_anchor_decision, _fit_ngl, _get, check_deadline, log, need,
                   paired_bootstrap, run_cmd, run_ppl, split_eval, vram_gb)
from .imatrix import resolve_imatrix
from .isolation import build_holdout
from .artifacts import (eval_device, eval_preflight, fairness_from_sidecars,
                        resolve_eval_models)

import json, os, re, subprocess, time
from pathlib import Path


def _calib_manifest_from_imatrix():
    """从本次 S3 的结果 / imatrix provenance 里取"校准实际用了什么"的 manifest。

    有它，`actual_imatrix_calib_overlap` 才能从 `unknown` 变成
    `derived_prefix_window`（近似）。没有就老实写 unknown —— 不许假装已知。
    """
    cands = []
    s3 = _get("S3_IMATRIX") or {}
    if isinstance(s3, dict) and s3.get("calib_manifest"):
        cands.append(s3["calib_manifest"])
    try:
        p = resolve_imatrix()
        if p is not None:
            side = Path(str(p)).with_name(Path(str(p)).name + ".provenance.json")
            if side.exists():
                d = json.loads(side.read_text(encoding="utf-8"))
                if isinstance(d.get("calib_manifest"), dict):
                    cands.append(d["calib_manifest"])
    except Exception:
        pass
    return cands[0] if cands else None



def _eval_ngl(nbytes, vram_total_gb):
    """这个模型该用多少 `-ngl`，以及为什么 —— 委托给 `artifacts.eval_device`。

    v1.1.1：默认 `EVAL_ALLOW_GPU=True`，**有 GPU 就整模型卸载**（-ngl 99）。
    只有三种情况会得到 0：显式强制 CPU、没有 CUDA 显存、单个模型装不下。
    第三种会写进结果（`why`），不允许静默降级成"以为在 GPU 上跑"。
    """
    return eval_device(vram_total_gb, nbytes)


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

    # ---- 校准/评测隔离：取证 + 确定性 clean holdout（v1.2.0）----
    # 旧版在这里 `raise RuntimeError("有 1 行重叠")`，整轮 S6 归零。
    # 但"train 和 test 里都有一句相同的话"并不等于"imatrix 实际看过评测数据"：
    # imatrix 只消费 chunks×ctx ≈ 16k token，而 train 有 10.6 MB。
    # 现在改成：取证 → 剔除 → 用同一份 holdout 切组 → 如实标注 unknown。
    iso_manifest = None
    holdout_path = eval_txt
    calib_path = (Path(CFG["CALIB_TXT"])
                  if CFG.get("CALIB_TXT") and Path(CFG["CALIB_TXT"]).exists()
                  else None)
    if calib_path is None:
        isolation = {"ok": None, "overlap_lines": None,
                     "note": ("没有校准语料路径，无法做隔离断言 —— "
                              "结论里必须标注这一点"),
                     "actual_imatrix_calib_overlap": {
                         "value": "unknown", "level": "unknown",
                         "why": "没有校准语料，无法比较"}}
        log("!! 没有校准语料路径，无法做隔离断言 —— 结论里必须标注这一点")
    else:
        iso_manifest = build_holdout(
            calib_path, eval_txt, WORK / "isolation",
            imatrix_manifest=_calib_manifest_from_imatrix(),
            cfg=CFG, eval_label=CFG.get("EVAL_SOURCE"))
        holdout_path = Path(iso_manifest["holdout"]["path"])
        isolation = {
            "ok": True,
            "rule": iso_manifest["rule"],
            # 兼容旧报告的字段名：这里现在是**原始**交集条数，不是"错误数"
            "overlap_lines": iso_manifest["overlap_raw"]["paras"],
            "overlap_raw": iso_manifest["overlap_raw"],
            "cleaning": iso_manifest["cleaning"],
            "holdout": iso_manifest["holdout"],
            "actual_imatrix_calib_overlap":
                iso_manifest["actual_imatrix_calib_overlap"],
            "calib": str(calib_path), "calib_source": CFG["CALIB_SOURCE"],
            "calib_lines": iso_manifest["calib"]["paras"],
            "eval": str(eval_txt), "eval_source": CFG["EVAL_SOURCE"],
            "eval_lines": iso_manifest["eval_raw"]["paras"],
            "manifest": str(WORK / "isolation" / "isolation.json"),
            "note": iso_manifest["note"],
        }
        log(f"隔离：原始交集 {isolation['overlap_lines']} 段 → 剔除后 "
            f"{isolation['cleaning']['kept_paras']} 段保留；"
            f"actual_imatrix_calib_overlap="
            f"{isolation['actual_imatrix_calib_overlap']['value']}")

    if len(models) < 2:
        raise GuardError(
            f"需要两个格式的文件，现有 {list(models)}。\n"
            "  → 本地没有就去 HF 取：RUN_EVAL=True（或 CFG['HF_FETCH_ARTIFACTS']=True）。")

    groups, meta = split_eval(holdout_path, CFG["NLL_GROUPS"], WORK / "eval_groups")
    log(f"评测集（clean holdout）切成 {len(groups)} 组：" +
        ", ".join(f"g{m['group']}({m['paras']}段/{m['chars']}字符)" for m in meta))


    per_model = {}
    # 一次只装一个模型：每个格式算自己的 -ngl，跑完就释放（llama-perplexity
    # 是独立进程，进程退出即释放，所以"两个模型同时在显存里"这件事不会发生）。
    _v = vram_gb() or {}
    dev_by_fmt = {fmt: _eval_ngl(rec["bytes"], _v.get("total"))
                  for fmt, rec in models.items()}
    for fmt, d in dev_by_fmt.items():
        log(f"设备 {fmt}: {d['device']} -ngl {d['ngl']}（{d['why']}）")
    t_start = time.time()
    for fmt, rec in models.items():
        mp = Path(rec["path"])
        per_model[fmt] = []
        _d = dev_by_fmt[fmt]
        log(f"装入 {fmt}（{_d['device']} -ngl {_d['ngl']}，"
            f"{rec['bytes'] / 2**30:.2f} GiB）；跑完释放再装下一个")
        for i, g in enumerate(groups):
            check_deadline(deadline, f"S6 {fmt} g{i}")
            r = run_ppl(mp, g, deadline, f"{fmt}_g{i}")
            r["group"] = i
            r["chars"] = meta[i]["chars"]
            r["model_source"] = rec.get("source")
            r["device"] = _d["device"]
            per_model[fmt].append(r)
            log(f"  {fmt} g{i}: PPL={r['ppl']:.4f} NLL={r['nll']:.4f} "
                f"(ngl={r['ngl']})")
        log(f"释放 {fmt}")
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

    # 实测速率回填：把"假设值"换成这一次的真实值，供下一轮估成本。
    # v1.2.0：区分 nominal（我们请求的 chunks×ctx）与 observed（日志自报的
    # tokenized N tokens）。语料不够长时 llama.cpp 会少算，吞吐不能拿 nominal 冒充。
    per_group_nominal = int(CFG["NLL_CHUNKS_PER_GROUP"]) * int(CFG["NLL_CTX"])
    tokens_nominal = per_group_nominal * len(groups) * len(models)
    _obs = [r.get("tokens_observed") for vals in per_model.values() for r in vals
            if r.get("tokens_observed")]
    tokens_observed = sum(_obs) if _obs else None
    used_tokens = tokens_observed or tokens_nominal
    measured = {
        "elapsed_sec": round(elapsed, 1),
        "tokens_nominal": tokens_nominal,
        "tokens_observed": tokens_observed,
        "tokens_used_for_rate": used_tokens,
        "token_count_source": ("llama-perplexity 日志的 `tokenized N tokens`"
                               if tokens_observed else
                               "日志没有自报 token 数 → 退回 chunks×ctx（上界）"),
        "tok_per_sec": round(used_tokens / max(1e-6, elapsed), 2),
        "note": ("这是**这一次**的实测吞吐，只对同机器同设置可比；"
                 "token 数优先用日志自报值，缺失时才用 chunks×ctx。"),
    }
    if tokens_observed and tokens_observed != tokens_nominal:
        log(f"!! 日志自报 {tokens_observed} token ≠ 请求的 {tokens_nominal} —— "
            "语料可能不够长，或 llama.cpp 调整了 chunk 数；已按观测值算吞吐")


    out = {
        "isolation": isolation,
        "fairness_verified": fairness_verified,
        "fairness_level": fair_level,
        "fairness_record": fair,
        "base_scope": base_scope,
        "scope_note": scope_note,
        "eval_source": CFG["EVAL_SOURCE"],
        "eval_txt": str(eval_txt),
        "eval_used_txt": str(holdout_path),
        "eval_holdout_sha256": (iso_manifest or {}).get("holdout", {}).get("sha256"),
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
        "device_by_fmt": {f: {"device": d["device"], "ngl": d["ngl"], "why": d["why"]}
                          for f, d in dev_by_fmt.items()},
        "settings": {"ctx": CFG["NLL_CTX"],
                     "chunks_per_group": CFG["NLL_CHUNKS_PER_GROUP"],
                     "groups": CFG["NLL_GROUPS"], "seed": CFG["SEED"],
                     "batch": 512, "one_model_at_a_time": True},
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
            # v1.1.1：设备是**逐格式**判定的（装不下会退 CPU），所以不能再写一个 bool
            "device": {f: d["device"] for f, d in dev_by_fmt.items()},
            "ngl": {f: d["ngl"] for f, d in dev_by_fmt.items()},
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
    if isolation.get("ok") is not None:
        _ov = isolation["actual_imatrix_calib_overlap"]
        print(f"隔离：原始交集 {isolation['overlap_lines']} 段 → "
              f"clean holdout 剔除 {isolation['cleaning']['removed_paras']} 段、"
              f"保留 {isolation['cleaning']['kept_paras']} 段"
              f"（≈{isolation['cleaning']['kept_tokens_est']} token）")
        print(f"actual_imatrix_calib_overlap = {_ov['value']}（level {_ov['level']}）"
              + ("；近似窗口检查见 isolation.json" if _ov.get("prefix_window_check") else ""))
    print("执行设备: " + ", ".join(f"{f}={d['device']}(-ngl {d['ngl']})"
                                  for f, d in dev_by_fmt.items()))
    print(f"实测吞吐: {measured['tok_per_sec']} tok/s（{elapsed:.0f} s，"
          f"{measured['tokens_used_for_rate']} token，来源：{measured['token_count_source']}）")

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

# **smoke** 用的期望片段（小写、子串匹配）。这不是能力评测：
# 它的唯一用途是区分"模型在正常续写"与"输出是 banner / 乱码 / 空"。
# 命中率低**不能**推出"IQ2 比 IQ4 差"；命中率高也**不能**推出"两个量化都好"。
QUALITY_SMOKE_EXPECT = {
    "The capital of France is": ["paris"],
    "def fibonacci(n):": ["return", "fib"],
    "In 1969, the first humans landed on": ["moon"],
    "The chemical formula for table salt is": ["nacl"],
    "Q: What is 17 * 23?\nA:": ["391"],
    "The three primary colors of light are": ["red", "green", "blue"],
    "Translate 'good morning' into Chinese:": ["早上好", "早安"],
}


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# llama.cpp 的元信息行（banner / 计时 / 采样参数 / 模型加载）。
# 只用于**剥掉**它们，绝不把整行内容当生成文本。
_META_LINE_RE = re.compile(
    r"^(?:"
    r"llama_perf|llama_print_timings|llama_model_loader|llama_context|llama_kv_cache|"
    r"llama_model_quantize|llama_new_context|ggml_|load_backend|load_tensors|"
    r"register_backend|system_info|sampling|generate|main|build|model params|"
    r"print_info|using device|CUDA|ggml_cuda|"
    r"load time|prompt eval time|eval time|total time|sampling time|"
    r"[IWED]\s+\d{2}:\d{2}:\d{2}"
    r")\b",
    re.I,
)


def _is_meta_line(line):
    s = line.strip()
    if not s:
        return False
    return bool(_META_LINE_RE.match(s))


def extract_generation(log_text, prompt=None):
    """从 `llama-cli` 的混合日志里只取**生成文本**（v1.2.0）。

    旧版是 `txt.split(prompt, 1)[1]`，再按 `llama_perf` 切一刀。两个问题：
      · `split(..., 1)` 取**第一处** prompt —— banner 里若回显 prompt（或 prompt
        是常见词），会把 banner 之后的所有东西都当成生成；
      · 只认 `llama_perf`，别的版本打的是 `llama_print_timings`，计时行会混进
        "生成文本"，于是两个模型的 `identical` 比较被噪声决定。

    这里按结构切：ANSI → 去掉尾部计时 → 从最后一个 `generate:` / `sampling:`
    块之后开始 → 去掉 prompt 回显 → 去掉 `[end of text]` → 去掉残余元信息行。
    返回的每个字段都能进 JSON，方便事后复核"到底比的是什么"。
    """
    raw = log_text or ""
    clean = _ANSI_RE.sub("", raw)
    lines = clean.split("\n")

    # 1) 尾部：第一条计时/加载行之后全部丢弃
    cut = len(lines)
    for i, ln in enumerate(lines):
        if re.match(r"^(llama_perf|llama_print_timings|llama_model_loader)",
                    ln.strip(), re.I):
            cut = i
            break
    body_lines = lines[:cut]

    # 2) 头部：最后一个 generate:/sampling: 块之后才是正文
    start = 0
    for i, ln in enumerate(body_lines):
        s = ln.strip()
        if re.match(r"^(generate:|sampling:)", s, re.I):
            start = i + 1
    # 跳过紧跟其后的参数续行（以 tab / 空格开头）与空行
    while start < len(body_lines) and (
            not body_lines[start].strip() or body_lines[start][:1] in ("\t", " ")):
        start += 1
    # 3) 再跳过残余的 banner 行（build:/main:/system_info:/model params ...）
    while start < len(body_lines) and _is_meta_line(body_lines[start]):
        start += 1
    body = "\n".join(body_lines[start:])

    prompt_echoed = False
    if prompt:
        if prompt in body:
            prompt_echoed = True
            body = body.rsplit(prompt, 1)[1]      # 取**最后**一处，不是第一处
        elif prompt.strip() and prompt.strip() in body:
            prompt_echoed = True
            body = body.rsplit(prompt.strip(), 1)[1]

    # 4) 交互式提示符与结束标记
    body = body.strip()
    while body.startswith(">"):
        body = body[1:].lstrip()
    body = body.replace("[end of text]", "").strip()

    # 5) 兜底：清掉夹在正文里的元信息行（只删整行匹配的）
    kept = [ln for ln in body.split("\n") if not _is_meta_line(ln)]
    dropped = len(body.split("\n")) - len(kept)
    body = "\n".join(kept).strip()

    return {
        "text": body,
        "chars": len(body),
        "empty": not body,
        "prompt_echoed": prompt_echoed,
        "dropped_meta_lines": dropped,
        "raw_chars": len(raw),
        "sha256": __import__("hashlib").sha256(body.encode("utf-8")).hexdigest(),
    }


def smoke_match(prompt, text):
    """期望片段命中（**smoke**，不是评分）。"""
    exp = QUALITY_SMOKE_EXPECT.get(prompt) or []
    low = (text or "").lower()
    hit = next((e for e in exp if e in low), None) if exp else None
    return {"expected_any": exp, "matched": bool(hit) if exp else None, "hit": hit}


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
    # v1.1.1：有 GPU 就用 GPU（与 S6 同一套判定），装不下才退 CPU 并说明原因。
    dev_by_fmt = {fmt: _eval_ngl(rec["bytes"], (v or {}).get("total"))
                  for fmt, rec in models.items()}
    log(f"S7 设备: " + ", ".join(f"{f}={d['device']}(-ngl {d['ngl']}, {d['why']})"
                                for f, d in dev_by_fmt.items()))

    per_model = {}
    for fmt, rec in models.items():
        mp = Path(rec["path"])
        ngl = dev_by_fmt[fmt]["ngl"]
        per_model[fmt] = []
        log(f"装入 {fmt}（{dev_by_fmt[fmt]['device']} -ngl {ngl}）；跑完释放再装下一个")
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
            g = extract_generation(txt, pr)
            row = {"prompt": pr, "text": g["text"][:400],
                   "text_sha256": g["sha256"], "chars": g["chars"],
                   "empty": g["empty"],
                   "prompt_echoed": g["prompt_echoed"],
                   "dropped_meta_lines": g["dropped_meta_lines"],
                   "raw_chars": g["raw_chars"],
                   "repeat4": round(repeat_ratio(g["text"]), 4),
                   "smoke": smoke_match(pr, g["text"]),
                   "log": str(logf)}
            per_model[fmt].append(row)
            if g["empty"]:
                log(f"!! {fmt} p{i} 的生成文本为空（rc=0）—— 记为空输出，不算有效样本")
            elif g["dropped_meta_lines"]:
                log(f"  {fmt} p{i}: 已剥掉 {g['dropped_meta_lines']} 行元信息/计时")
        log(f"释放 {fmt}")

    pairs = []
    fa, fb = FORMATS[0], FORMATS[1]
    for i, pr in enumerate(prompts):
        a = per_model[fa][i]["text"]
        b = per_model[fb][i]["text"]
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        pairs.append({"prompt": pr, "identical": a == b, "first_divergence_char": k,
                      "len_iq2": len(a), "len_iq4": len(b),
                      "both_empty": (not a and not b)})

    n_id = sum(1 for p in pairs if p["identical"])
    empty = {fmt: sum(1 for x in v2 if x["empty"]) for fmt, v2 in per_model.items()}
    sm = {}
    for fmt, v2 in per_model.items():
        got = [x["smoke"]["matched"] for x in v2 if x["smoke"]["matched"] is not None]
        sm[fmt] = {"matched": sum(1 for x in got if x),
                   "of": len(got),
                   "rate": round(sum(1 for x in got if x) / len(got), 4) if got else None}
    out = {
        "n_prompts": len(prompts),
        "identical_outputs": n_id,
        "identical_rate": round(n_id / len(prompts), 4) if prompts else None,
        "pairs": pairs,
        "per_model": per_model,
        "empty_outputs": empty,
        "smoke_match": sm,
        "degenerate": {fmt: round(sum(x["repeat4"] for x in v2) / len(v2), 4)
                       for fmt, v2 in per_model.items()},
        "settings": {"temperature": 0, "seed": CFG["SEED"],
                     "n_predict": CFG["QUALITY_N_PREDICT"],
                     "ngl": {f: d["ngl"] for f, d in dev_by_fmt.items()},
                     "device": {f: d["device"] for f, d in dev_by_fmt.items()},
                     "one_model_at_a_time": True,
                     "gpu_allowed": bool(CFG["EVAL_ALLOW_GPU"]),
                     "parser": "extract_generation/v1.2.0"},
        "evidence": {
            "identical_outputs": "smoke_only",
            "repeat4": "degenerate_signal_only",
            "smoke_match": "smoke_only_substring_not_benchmark",
            "can_rank_quality": False,
            "why": ("8 题 × 32 token 的贪心续写只能发现整体崩塌（空输出、乱码、"
                    "复读），**不能**排序 IQ2 与 IQ4 的细微质量差异。"
                    "0/8 完全一致不等于 IQ2 更差；8/8 一致也不等于两者等价。"),
        },
        "provenance": {
            "inputs": {f: {"path": r["path"], "source": r.get("source"),
                           "sha256_verified": r.get("sha256_verified"),
                           "origin_detail": r.get("origin_detail")}
                       for f, r in models.items()},
            "note": ("输入来自 HF 上已存的量化产物或本地 S4 产物，本次未重量化；"
                     "source=hf_cache 表示本地文件带 HF 下载 stamp（不是本次量化）。"),
        },
        "limits": [
            "贪心解码 + 32 token，样本量很小，只能发现整体崩塌，不能排序细微质量",
            "不是标准任务评测（GSM8K / HumanEval 之类），绝对分数不构成能力证据",
            "repeat4 只是退化信号，不是质量分数",
            "smoke_match 是子串匹配，用来排除 banner/空输出，不是准确率",
        ],
    }
    print("=" * 78)
    print(f"完全一致的输出: {n_id}/{len(prompts)}  ({out['identical_rate']}) "
          "—— smoke，不是质量结论")
    print(f"空输出: {empty}；期望片段命中（smoke）: {sm}")
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

    # v1.2.0：锚点必须与 S6 用**同一份** clean holdout，否则 rho 的分子分母
    # 不是同一个评测集，比出来的东西没有意义。
    _s6 = _get("S6_NLL") or {}
    if _s6.get("eval_used_txt"):
        eval_txt = Path(_s6["eval_used_txt"])
    else:
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
        # 与 S6 同口径：有 clean holdout 就用它
        _s6 = _get("S6_NLL") or {}
        evp = Path(_s6["eval_used_txt"]) if _s6.get("eval_used_txt") else Path(CFG["EVAL_TXT"])
        groups, meta = split_eval(evp, CFG["NLL_GROUPS"], WORK / "eval_groups")

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
