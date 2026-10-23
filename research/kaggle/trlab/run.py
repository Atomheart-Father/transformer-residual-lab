"""编排入口：run(**开关) 顺序执行各阶段并打印摘要。

CLI: python -m trlab.run [--run-s2] [--run-s3] [--run-s4] [--run-eval] [--set K=V]

v1.1.1 的执行哲学（用户 2026-10-23 确认）：**Kaggle 是可丢弃的 GPU worker**。
所以这里只做三件事，不做跨 Session 状态管理、不做 Save Version 备援：
  1. 每个阶段跑完就把它的小结果**出站**（HF，白名单 5 个文件）+ 远端复核；
  2. 失败就 fail fast，写一份 `FAILURE.json`（阶段 / 报错 / traceback 尾 / 下一步）；
  3. 评测结束写一份几百字节的 `EVAL_SUMMARY.json`，方便直接贴到聊天里。
"""

from .config import (CFG, LOGS, NOTEBOOK_VERSION, OUT, STAGES, _apply_run_switches,
                     _s2_only, print_banner)
from .errors import GuardError
from .paths import _s3_base_path, llama_bin
from .core import STATE, _get, ensure_dirs, log, run_stage, write_json
from .stages_env import stage_s0_env, stage_s1_smoke
from .stages_build import stage_s2_build
from .stages_imatrix import stage_s3_imatrix
from .stages_quant import stage_s4_quant, stage_s5_bytes, stage_s4_preflight
from .stages_eval import stage_s6_nll, stage_s7_quality, stage_s8_bf16_anchor, stage_sx_explore
from .report import _s2_report_lines, _s3_report_lines, _s4_report_lines, _s4_preflight_lines, stage_s9_report

import argparse, json, sys
from datetime import datetime, timezone


# ---------------------------------------------------------------- 出站 / 失败记录

# 只允许这 5 个小文件出站。模型、日志、secrets 一律不传。
_UPLOAD_WHITELIST = {
    "S6_NLL":     ("S6_NLL.json", "EVAL_SUMMARY.json"),
    "S7_QUALITY": ("S7_QUALITY.json",),
    "S9_REPORT":  ("report.md", "results.json", "S9_REPORT.json"),
}

_RUN_ID = None


def _run_id():
    """一次运行的标签，用来在 HF 上留历史（`results/v{ver}/{run_id}/`）。"""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _RUN_ID


# check_access 的 status → FAILURE.json 的 cause 代码。
# 分开的理由：400 是**我们自己的请求参数**错了，401/403 才是凭据问题。
# v1.1.2 真实首跑把 400 归成 hf_unauthorized，让用户去重建 token —— 错误分类。
_CAUSE_BY_AUTH = {
    "no_token": "hf_no_token",
    "unauthorized": "hf_unauthorized",
    "bad_request": "hf_bad_request",
    "not_found": "hf_not_found",
    "rate_limited": "hf_rate_limited",
    "network": "hf_network",
}

_SECRETS_STEP = ("在 Kaggle 里做**一次**配置：`Add-ons → Secrets` 新建 `HF_TOKEN`"
                 "（Fine-grained token，对该 HF repo 勾 Read + Write），"
                 "然后在本 notebook 上 **attach** 它，再 `Run all`。"
                 "不要把 token 写进 notebook / git / issue。")


def _next_step_for_auth(status):
    """按 check_access 的 status 给**一条**下一步 —— 400 不许再怪用户 Secrets。"""
    if status in ("no_token", "unauthorized"):
        return _SECRETS_STEP
    if status == "bad_request":
        return ("**这是本工具请求参数的问题，不是你的 Secrets**（HTTP 400 来自 tree API）。"
                "换最新 pinned notebook 重跑即可，**不要**重建 token；"
                "若新版仍报 400，把 `FAILURE.json` 贴到 Issue #5。")
    if status == "not_found":
        return ("HF 说仓库/revision 不存在（404）：核对 `CFG['HF_MODEL_REPO']` 与 revision；"
                "私仓要先确认这个 token 真的能看到该仓库。")
    if status == "rate_limited":
        return "Hub 侧限流或临时故障（429/5xx）：等 30–60 秒重跑，不用改任何配置。"
    if status == "network":
        return ("连不上 Hugging Face：先确认 `Settings → Internet = On`，"
                "再确认不是 Kaggle 出口被拦；这不涉及 token。")
    return "看 `FAILURE.json` 里的服务器原话与 traceback 尾。"


def _classify(stage, err):
    """错误 → (cause 代码, 一条下一步)。cause 用来在摘要里去重合并。

    顺序很重要：400 的消息里也出现 "token" 字样（"这不是 token 问题"），
    所以**必须**先判 400，否则又会误分类成凭据问题。
    """
    e = str(err or "")
    low = e.lower()
    if "HTTP 400" in e:
        return "hf_bad_request", _next_step_for_auth("bad_request")
    if "HTTP 404" in e:
        return "hf_not_found", _next_step_for_auth("not_found")
    if "连不上 Hugging Face" in e:
        return "hf_network", _next_step_for_auth("network")
    if "HTTP 429" in e or "HTTP 50" in e:
        return "hf_rate_limited", _next_step_for_auth("rate_limited")
    if "token" in low or "401" in e or "403" in e:
        code = "hf_unauthorized" if ("401" in e or "403" in e) else "hf_no_token"
        return code, _SECRETS_STEP
    if "llama-perplexity" in e or "llama-cli" in e:
        return "binaries_missing", (
            "缺 llama.cpp 二进制：v1.1.2 起 `RUN_EVAL=True` 会**自动只编 S2**"
            "（pinned 源码 36.6 MiB，不下载模型、不重量化）。若仍失败，看 S2_BUILD 的日志。")
    if stage == "S2_BUILD":
        return "s2_build_failed", "S2 编译失败：看 `logs/s2_build.log` 尾部那句话；修好后重跑。"
    if "超过" in e and "上限" in e:
        return "budget", ("估算超上限：调小 NLL_GROUPS / NLL_CTX，或开 GPU Session"
                          "（代码默认就会用 GPU）。")
    if "sha256" in low or "lfs.oid" in low:
        return "hash_mismatch", "本地 GGUF 与 HF 上的哈希不一致：删掉那个文件重跑（会重新下载并校验）。"
    if "重叠" in e:
        return "corpus_overlap", "校准集与评测集重叠：这是语料配置问题，换 split，不要绕过断言。"
    return "other", (f"看 {OUT / (stage + '.json')} 里的 traceback；修好后在新 Session 重跑即可"
                     "（HF 上的输入不用重下）。")


def _record_failure(stage, error=None, traceback_tail=None, plan=None,
                    upload_state=None, cause=None, next_step=None,
                    extra_causes=None):
    """写 `FAILURE.json`：**最早根因不被后续失败覆盖**（v1.1.2）。

    结构：顶层 `stage/error/next_step` 永远是最早那一条；`all` 按时间顺序累积；
    `causes` 是去重后的根因代码（例如 `["hf_no_token", "binaries_missing"]`），
    这样"两条根因"在一屏里能看全，不用翻多个日志。
    """
    rec = STATE.get(stage) or {}
    if error is None and rec.get("status") != "error":
        return None
    err = str(error if error is not None else rec.get("error"))
    if cause is None or next_step is None:
        c, s = _classify(stage, err)
        cause, next_step = (cause or c), (next_step or s)

    item = {"stage": stage, "cause": cause, "error": err,
            "traceback_tail": (traceback_tail if traceback_tail is not None
                               else (rec.get("traceback") or "")[-1500:]),
            "next_step": next_step,
            "started": rec.get("started"), "elapsed_sec": rec.get("elapsed_sec")}

    path = OUT / "FAILURE.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    items = list(prev.get("all") or [])
    if not items and prev.get("stage"):        # 兼容 v1.1.1 的单条格式
        items = [{"stage": prev["stage"], "cause": prev.get("cause"),
                  "error": prev.get("error"), "next_step": prev.get("next_step")}]
    if not any(i.get("stage") == stage for i in items):
        items.append(item)

    causes = []
    for c in [i.get("cause") for i in items] + list(extra_causes or []):
        if c and c not in causes:
            causes.append(c)
    first = items[0]
    doc = {
        "stage": first.get("stage"), "cause": first.get("cause"),
        "error": first.get("error"), "next_step": first.get("next_step"),
        "traceback_tail": first.get("traceback_tail"),
        "all": items, "causes": causes,
        "run_id": _run_id(), "notebook_version": NOTEBOOK_VERSION,
        "plan": plan, "upload": upload_state,
        "note": ("Kaggle 只是可丢弃的算力机：这份 JSON 和阶段 JSON 都在本地产物目录里。"
                 "修代码 → commit → 新 Session 重跑，输入不用重下（HF 是长期源）。"),
    }
    write_json(path, doc)

    print("=" * 78)
    print(f"运行失败 —— 最早根因：{first.get('stage')} / {first.get('cause')}")
    for i in items:
        print(f"  · [{i.get('cause')}] {i.get('stage')}: {str(i.get('error'))[:160]}")
    print("  下一步（按顺序做）：")
    for i in items:
        print(f"    - {i.get('next_step')}")
    print(f"  现场：{path}（含 traceback 尾）；阶段 JSON 在 {OUT}/")
    print("  你可以直接关掉这个 Session；修好后新 Session 重跑，HF 上的输入不用重下。")
    print("=" * 78)
    return doc


def _write_failure(stage, plan=None, upload_state=None):
    """兼容 v1.1.1 的调用点：阶段失败时记录（合并进 `all`）。"""
    return _record_failure(stage, plan=plan, upload_state=upload_state)


# ---------------------------------------------------------------- 评测前的两道门

def _missing_eval_binaries():
    """本地探活（不发网络请求）：评测真正需要的两个二进制缺哪个。"""
    miss = []
    for name in ("llama-perplexity", "llama-cli"):
        try:
            llama_bin(name)
        except GuardError:
            miss.append(name)
    return miss


def _auth_gate(plan=None):
    """开跑前的 HF 授权检查（1 次只读请求；无 token 时连请求都不发）。

    排在编译之前是刻意的：真实 Kaggle 上第一次 RUN_EVAL 就是因为没有 attach
    `HF_TOKEN`，白等了 28 分钟编译才在下载时看到 401。
    """
    from .hf import check_access
    a = check_access()
    if a.get("ok"):
        log(f"HF 授权检查通过：{a['repo']}@{a['revision']}"
            f"（{a.get('n_files')} 个文件，凭据来源 {a.get('token_source')}）")
    else:
        _h = f"，HTTP {a['http_status']}" if a.get("http_status") else ""
        log(f"!! HF 授权检查失败（{a.get('status')}{_h}）：{str(a.get('error'))[:300]}")
    return a


def _autobuild_binaries(plan):
    """`RUN_EVAL=True` 但本 Session 没有二进制 → **自动只编 S2**。

    用户仍然只有一个开关。这里只下载 36.6 MiB 的 pinned 源码并编译，
    **不下载 Q8_0/BF16、不做 S3/S4、不重量化**。
    """
    miss = _missing_eval_binaries()
    if not miss:
        log("二进制就绪：llama-perplexity / llama-cli 都在（不编译）")
        return plan
    log(f"评测需要 {miss}，本 Session 没有 → 自动只编 S2"
        "（pinned 源码 36.6 MiB；不下载模型、不做 S3/S4、不重量化）")
    _s2_only()
    plan = dict(plan)
    plan.update(autobuild="S2_BUILD", autobuild_why=f"缺 {miss}", run_s2=True,
                stages=[k for k, v in STAGES.items() if v])
    return plan


def _abort_eval(plan, auth, upload_state=None):
    """凭据门不通就地停：写 FAILURE.json（含"缺二进制"这条并存的根因）+ 出报告。

    **分类按 `auth['status']`（来自 HTTP 状态码），不按字符串猜。**
    v1.1.2 把 HTTP 400（tree API 参数错）归成 `hf_unauthorized`，等于让用户
    去重建一个本来就正确的 token —— 这条是错的，v1.1.3 分开。
    """
    status = str(auth.get("status") or "error")
    cause = _CAUSE_BY_AUTH.get(status, "hf_other")
    extra = []
    miss = _missing_eval_binaries()
    if miss:
        extra.append("binaries_missing")
    step = _next_step_for_auth(status)
    http = auth.get("http_status")
    doc = _record_failure("HF_AUTH",
                          error=(f"HF 授权检查失败（{status}"
                                 + (f"，HTTP {http}" if http else "") + "）"
                                 + (f"：{str(auth.get('error'))[:600]}"
                                    if auth.get("error") else "")),
                          plan=plan, upload_state=upload_state,
                          cause=cause, next_step=step, extra_causes=extra)
    if extra:
        print("  另外：本 Session 也缺 llama.cpp 二进制 —— 你**不需要**单独开 S2，"
              "凭据通了之后重跑 RUN_EVAL 会自动只编 S2。")
    if cause == "hf_bad_request":
        print("  ！！注意：HTTP 400 是**本工具的请求参数问题**，不是你的 Secrets ——"
              " 不要重建 token。")
    # 仍然出一份报告（0.03 s），但报告顶部会写明"本次运行失败"
    try:
        run_stage("S9_REPORT", stage_s9_report)
    except Exception as e:
        log(f"!! 失败后生成报告也失败了：{e}")
    return {"ok": False, "failed": ["HF_AUTH"], "causes": doc.get("causes"),
            "plan": plan,
            "stages": {k: (STATE.get(k) or {}).get("status") for k in STAGES}}


def _upload_stage(stage, force=False):
    """把一个阶段的小结果传到 HF 并**远端复核**。

    只有 `CFG['HF_UPLOAD_AFTER_RUN']`（RUN_EVAL 会打开）或 `upload=True` 才动手。
    失败**不改变**阶段结论，但必须明确说"哪些结果尚未持久化" —— 不许虚报。
    """
    names = _UPLOAD_WHITELIST.get(stage, ())
    files = [(OUT / n, f"results/v{NOTEBOOK_VERSION}/{_run_id()}/{n}")
             for n in names if (OUT / n).exists()]
    if not files:
        return {"stage": stage, "skipped": True, "why": "没有可上传的文件"}
    if not (force or CFG.get("HF_UPLOAD_AFTER_RUN")):
        return {"stage": stage, "skipped": True,
                "why": "CFG['HF_UPLOAD_AFTER_RUN']=False（结果只落本地，Session 结束即丢）",
                "not_persisted": [b for _, b in files]}

    from .hf import upload_after_run
    try:
        rec = upload_after_run(files)
        if rec.get("skipped"):
            rec["not_persisted"] = [b for _, b in files]
        else:
            v = rec.get("verify") or {}
            rec["remote_verified"] = bool(v.get("ok"))
            rec["not_persisted"] = ([] if v.get("ok")
                                    else [b for _, b in files])
        print(f"HF 出站（{stage}）：{json.dumps(rec, ensure_ascii=False)[:400]}")
        return rec
    except Exception as e:
        rec = {"stage": stage, "ok": False, "error": str(e),
               "not_persisted": [b for _, b in files]}
        print(f"!! HF 出站失败（{stage}）—— 这些结果**尚未持久化**："
              f"{rec['not_persisted']}\n   原因：{e}")
        return rec


def _write_eval_summary(plan=None, upload_state=None):
    """几百字节的摘要，专为"贴到聊天里"设计。数字全部来自 S6/S7 的 JSON。"""
    s6 = _get("S6_NLL") or {}
    s7 = _get("S7_QUALITY") or {}
    if not (s6 or s7):
        return None
    st = s6.get("paired_delta_nll_iq2_minus_iq4") or {}
    agg = s6.get("aggregate") or {}
    inputs = ((s6.get("provenance") or {}).get("inputs") or {})
    sm = {
        "run_id": _run_id(),
        "version": NOTEBOOK_VERSION,
        "device": (s6.get("device_by_fmt") or {}),
        "fairness_level": s6.get("fairness_level"),
        "fairness_verified": s6.get("fairness_verified"),
        "inputs": {f: {"source": r.get("source"),
                       "sha256_verified": r.get("sha256_verified")}
                   for f, r in inputs.items()},
        "eval": {"corpus": s6.get("eval_source"), "ctx": s6.get("eval_ctx"),
                 "groups": s6.get("groups"),
                 "chunks_per_group": s6.get("eval_chunks_per_group")},
        "nll": {f: round(a.get("mean_nll", float("nan")), 5) for f, a in agg.items()},
        "delta_nll_iq2_minus_iq4": {
            "mean": (round(st["mean"], 5) if st.get("mean") is not None else None),
            "ci95": st.get("ci95"), "n": st.get("n"),
            "char_weighted": (round(st["char_weighted_mean"], 5)
                              if st.get("char_weighted_mean") is not None else None),
        },
        "verdict": s6.get("verdict"),
        "throughput_tok_s": (s6.get("measured_throughput") or {}).get("tok_per_sec"),
        "s7": ({"n_prompts": s7.get("n_prompts"),
                "identical_rate": s7.get("identical_rate"),
                "degenerate_repeat4": s7.get("degenerate")} if s7 else
               "S7 not run"),
        "upload": upload_state,
        "limits": ["pilot：512 ctx / 8 组；bootstrap 的 n 是组数不是 token 数",
                   "S6/S7 不是标准任务评测，不能当能力排名",
                   "fairness_level=hf_sidecar_declared 表示没有重算核验"],
    }
    write_json(OUT / "EVAL_SUMMARY.json", sm)
    print("=" * 78)
    print("评测摘要（可直接复制粘贴，也写在 EVAL_SUMMARY.json）：")
    print(json.dumps(sm, ensure_ascii=False, indent=1))
    print("=" * 78)
    return sm

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


def run(run_s2=None, run_s3=None, run_s4=None, run_eval=None,
        preflight_only=False, upload=False, **overrides):
    """按开关顺序执行各阶段。

    三个量化开关都 None 时读 CFG 里的值（默认全 False → 只跑 S0/S1/S9，
    不联网、不下载）。overrides 可直接改任意 CFG 键，例如
    run(run_s4=True, FORCE_RERUN=True)；阶段名（S6_NLL 等）写 STAGES。

    preflight_only=True：只跑 S4_PREFLIGHT（几秒的只读磁盘/暂存盘/imatrix
    核查），不编译、不下载。RUN_S4=True 的正常一键运行会**自动**先跑它，
    不过就地在下载前停下。

    run_eval=True（v1.1）：只评测。从 HF 上已存的量化 GGUF 取输入，跑 S6/S7/S9，
    **不下载 15 GiB BF16、不重做 S3、不重新量化**；有 GPU 就用 GPU。

    v1.1.2 起，`run_eval=True` 在**新 Session** 上会自己补齐前置条件，用户仍然只有
    一个开关：
      1. 探活 `llama-perplexity` / `llama-cli`；缺了就**自动只编 S2**
         （pinned 源码 36.6 MiB，不下载模型、不做 S3/S4）；
      2. 编译之前先做一次 **HF 授权检查**（1 次只读请求；没 token 连请求都不发），
         不通就地停并写 `FAILURE.json` —— 不让用户等 28 分钟才看到 401。

    upload=True：显式 opt-in，把本轮小结果（JSON/report）传到 HF。默认不传。
    """
    for _k, _v in (("RUN_S2", run_s2), ("RUN_S3", run_s3),
                   ("RUN_S4", run_s4), ("RUN_EVAL", run_eval)):
        if _v is not None:
            CFG[_k] = bool(_v)
    _apply_overrides(overrides)
    if preflight_only:
        # 只留预检：其余阶段全关，四个开关也归位（_apply_run_switches 就不再改任何字段）
        for _k in STAGES:
            STAGES[_k] = False
        STAGES["S4_PREFLIGHT"] = True
        CFG["RUN_S2"] = CFG["RUN_S3"] = CFG["RUN_S4"] = CFG["RUN_EVAL"] = False
    plan = _apply_run_switches()
    ensure_dirs()
    print_banner()
    log(f"trlab v{NOTEBOOK_VERSION}  plan={plan}")

    # ---- S4 预检：任何编译/下载之前。不过就地停（fail-closed）----
    if STAGES.get("S4_PREFLIGHT"):
        pf = run_stage("S4_PREFLIGHT", stage_s4_preflight)
        pf_status = (STATE.get("S4_PREFLIGHT") or {}).get("status")
        if pf_status != "ok":
            print("-" * 78)
            print("S4 预检未通过 —— 在 28 分钟编译、8 GiB / 15 GiB 下载开始之前就停下了。")
            print("上面 [S4_PREFLIGHT] FAIL 那行给出了唯一需要做的一步；细节在：")
            print(f"  {OUT / 'S4_PREFLIGHT.json'}")
            print("做好那一步再重跑即可，已下过的东西都有断点/指纹保护。")
            print("-" * 78)
            return {"plan": plan, "preflight": "failed",
                    "stages": {k: (STATE.get(k) or {}).get("status") for k in STAGES}}
        print("-" * 78)
        print("S4 预检摘要")
        for _line in _s4_preflight_lines(pf or {}):
            print(_line)
        print("-" * 78)
        if preflight_only:
            return {"plan": plan, "preflight": "ok",
                    "result": pf,
                    "stages": {k: (STATE.get(k) or {}).get("status") for k in STAGES}}

    s0 = run_stage("S0_ENV", stage_s0_env)

    s1 = run_stage("S1_SMOKE", stage_s1_smoke)

    # ---- v1.1.2 评测前的两道门：凭据（1 次只读请求）→ 二进制（本地探活）----
    # 顺序刻意如此：凭据不通就别花 28 分钟编译；但"缺二进制"是免费信息，
    # 所以先探活、一起报给用户（两条根因一屏看全）。
    if CFG["RUN_EVAL"]:
        _miss = _missing_eval_binaries()
        if _miss:
            log(f"二进制探活：缺 {_miss}（稍后会自动只编 S2）")
        auth = _auth_gate(plan)
        if not auth.get("ok"):
            return _abort_eval(plan, auth)
        plan = _autobuild_binaries(plan)

    s2 = run_stage("S2_BUILD", stage_s2_build)

    s2_failed = (STATE.get("S2_BUILD") or {}).get("status") == "error"
    if s2:
        print("-" * 78)
        print("S2 摘要")
        for _line in _s2_report_lines(s2):
            print(_line)
        print("-" * 78)
    elif s2_failed:
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
        for name, d in (s5.get("per_model") or {}).items():
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

    # ---- S6/S7：S2（自动补编）没成功就不启动，否则只会再报一次"找不到二进制" ----
    skip_eval = bool(CFG["RUN_EVAL"] and s2_failed)
    if skip_eval:
        _record_failure("S2_BUILD", plan=plan)
        print("-" * 78)
        print("S2 没成功 → S6/S7 不启动（否则只会再报一次『找不到 llama-perplexity』）。")
        print("修好 S2 后重跑即可：模型不用重下，S3/S4 也不会被顺带触发。")
        print("-" * 78)

    if skip_eval:
        s6 = s7 = None
    else:
        s6 = run_stage("S6_NLL", stage_s6_nll)
        if (STATE.get("S6_NLL") or {}).get("status") == "error":
            _write_failure("S6_NLL", plan)
        elif s6:
            # 阶段一完成就出站（小 JSON + 摘要），不等全部跑完 —— Session 随时可丢
            _write_eval_summary(plan)
            _up6 = _upload_stage("S6_NLL")
            if s6:
                s6["upload"] = _up6

        s7 = run_stage("S7_QUALITY", stage_s7_quality)
        if (STATE.get("S7_QUALITY") or {}).get("status") == "error":
            _write_failure("S7_QUALITY", plan)
        elif s7:
            _up7 = _upload_stage("S7_QUALITY")
            s7["upload"] = _up7
            _write_eval_summary(plan, upload_state=_up7)

    s8 = run_stage("S8_BF16_ANCHOR", stage_s8_bf16_anchor)

    sx = run_stage("SX_EXPLORE", stage_sx_explore)

    s9 = run_stage("S9_REPORT", stage_s9_report)

    if s9 or upload:
        _upload_stage("S9_REPORT", force=bool(upload))

    # ---- 收尾：失败不许被"报告生成了"掩盖（v1.1.2）----
    stages = {k: (STATE.get(k) or {}).get("status") for k in STAGES}
    failed = [k for k, v in stages.items() if v == "error"]
    causes = []
    fp = OUT / "FAILURE.json"
    if fp.exists():
        try:
            causes = list(json.loads(fp.read_text(encoding="utf-8")).get("causes") or [])
        except Exception:
            causes = []
    ok = not failed
    print("=" * 78)
    if ok:
        print("本次运行完成：所有已开启的阶段都成功。")
        if CFG["RUN_EVAL"] and not _get("S6_NLL") and not _get("S7_QUALITY"):
            print("!! 但 S6/S7 没有产出数据 —— 这不叫成功，检查开关与日志。")
            ok = False
    else:
        print(f"本次运行**失败**：{failed}；根因 {causes or ['未分类']}")
        print(f"  现场：{fp}；报告（含失败说明）：{OUT / 'report.md'}")
        print("  修好后新 Session 重跑即可，HF 上的输入不用重下。")
    print("=" * 78)
    return {"ok": ok, "failed": failed, "causes": causes,
            "plan": plan, "stages": stages}


def main(argv=None):
    """CLI 入口：python -m trlab [--run-s4 | --run-eval] [--set KEY=JSON]"""
    ap = argparse.ArgumentParser(prog="trlab",
                                 description="transformer-residual-lab Kaggle 验证台")
    ap.add_argument("--run-s2", action="store_true", help="编译 llama.cpp（S2）")
    ap.add_argument("--run-s3", action="store_true", help="S2+S3：生成 imatrix")
    ap.add_argument("--run-s4", action="store_true", help="S2+S3+S4+S5：量化 IQ2_XS/IQ4_XS")
    ap.add_argument("--run-eval", "--eval-only", dest="run_eval", action="store_true",
                    help="只评测 S6/S7/S9：从 HF 已存的 GGUF 取输入，不编译、不重量化")
    ap.add_argument("--preflight-only", action="store_true",
                    help="只跑 S4_PREFLIGHT：几秒内核查磁盘/暂存盘/imatrix，不编译不下载")
    ap.add_argument("--check-hf", action="store_true",
                    help="只检查 HF 凭据与仓库可读性，不打印 token")
    ap.add_argument("--upload", action="store_true",
                    help="显式 opt-in：把本轮小结果传到 HF（默认不传）")
    ap.add_argument("--force-rerun", action="store_true", help="忽略断点强制重跑")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="覆盖任意 CFG 键，例如 --set NLL_CTX=256")
    args = ap.parse_args(argv)

    if args.check_hf:
        from .hf import status
        print(json.dumps(status(), indent=1, ensure_ascii=False))
        return 0

    over = {"FORCE_RERUN": True} if args.force_rerun else {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            over[k] = json.loads(v)
        except Exception:
            over[k] = v
    run(run_s2=args.run_s2 or None, run_s3=args.run_s3 or None,
        run_s4=args.run_s4 or None, run_eval=args.run_eval or None,
        preflight_only=args.preflight_only, upload=args.upload, **over)
    print_artifacts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
