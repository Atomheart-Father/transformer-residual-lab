"""校准集 / 评测集的**可审计隔离**（v1.2.0）。

v1.1.x 的做法是：把 `calib.txt`（wikitext-2-raw **train** 全文，10.6 MB）和
`eval.txt`（**test** 全文）按行 `strip()[:200]` 求交，出现任意 1 行就
`raise RuntimeError("校准集与评测集有 1 行重叠")`，整轮 S6 归零。

这个断言有两个问题，都是**科研上的**，不是工程上的：

1. **它证明不了污染。** `stages_imatrix` 把整份 train 交给 `llama-imatrix -f`，
   真正被消费的只有 `IMATRIX_CHUNKS × IMATRIX_CTX`（32 × 512 ≈ 16.4k）个 token。
   10.6 MB 里绝大多数段落**从未进入** imatrix。一条 train/test 都有的句子
   （维基百科的标题、列表项、常见句式）与"实际校准样本泄漏到评测集"是两件事。
2. **它也证明不了没有污染。** 反过来，如果交集为空，同样不能推出 imatrix
   没看过评测数据（采样位置未知）。

所以这里不再"见 1 行就全局拒绝"，而是做三件可审计的事：

  A. **取证**：每条交集都记下完整文本 sha256、train/test 段索引、是否只是
     前 200 字符相同、是不是 wiki 标题、以及它落在 train 文本的哪个字符偏移。
  B. **确定性 clean holdout**：把与校准集规范化后前 200 字符相同的 **评测段**
     剔除，得到 `eval_holdout.txt`；记录规则、剔除数量、被剔除段的全量哈希、
     保留段数/字符数/估算 token 数、holdout 文件自身的 sha256。
     两个量化必须用**完全同一份** holdout（S6 只切一次组）。
  C. **诚实标注**：`actual_imatrix_calib_overlap` 只有在**本次运行真的产出了
     imatrix 且写下了消费窗口**时才有值，否则一律 `unknown`。
     即便有值，等级也只是 `derived_prefix_window`（llama-imatrix 顺序读文件，
     窗口按 `need_tokens × 4 chars/token` 近似），**永远不是 observed**。

不做的事：不改已存在的 imatrix / S3 / S4 产物；不把断言删掉；不"换个 split"
糊过去；不在 holdout 之外偷偷用原始 eval。
"""

from .config import CFG
from .core import log, write_json
from .errors import GuardError

import hashlib
import re
import unicodedata
from pathlib import Path


# 规则版本写进 manifest：换规则必须换这个字符串，旧 manifest 才不会假装同一条规则。
HOLDOUT_RULE = "drop_eval_paras_sharing_nfkc_casefold_prefix200_with_calib_v1"

PREFIX_CHARS = 200        # 只比较规范化后的前 200 字符（与旧断言同一尺度）
MIN_PARA_CHARS = 40       # 太短的段落（空行、单字）不参与，否则误杀一片
CHARS_PER_TOKEN = 4.0     # 仅用于"窗口/ token 数"的近似，**不是**分词器


# ---------------------------------------------------------------- 规范化

def normalize(text):
    """NFKC + 折叠空白 + casefold。

    刻意只做这三件：再激进（去标点、去数字）会把不同的段落误判成重复，
    clean holdout 就会开始删真实评测数据；保守一点，宁可少删。
    """
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paragraphs(text):
    """与 `core.split_eval` 同一套分段：空行分段，退化时按行。"""
    ps = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(ps) < 2:
        ps = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return ps


def paragraphs_with_offsets(text):
    """[(char_offset, para)] —— 偏移用于判断"这条交集落在 imatrix 窗口内吗"。"""
    out, pos = [], 0
    for chunk in re.split(r"(\n\s*\n)", text):
        if chunk and not re.fullmatch(r"\n\s*\n", chunk):
            lead = len(chunk) - len(chunk.lstrip())
            s = chunk.strip()
            if s:
                out.append((pos + lead, s))
        pos += len(chunk)
    return out


def _is_heading(raw):
    return bool(re.fullmatch(r"=+\s*[^=\n]+?\s*=+", raw.strip()))


def para_record(index, raw, offset=None):
    """一个段落的取证记录（不落盘原始全文，只留哈希与预览）。"""
    n = normalize(raw)
    return {
        "index": index,
        "offset": offset,
        "norm_len": len(n),
        "full_sha256": sha256_text(n),
        "key200": (sha256_text(n[:PREFIX_CHARS])
                   if len(n) > MIN_PARA_CHARS else None),
        "is_heading": _is_heading(raw),
        "preview": re.sub(r"\s+", " ", raw)[:120],
        "raw_len": len(raw),
    }


# ---------------------------------------------------------------- 取证

def find_overlaps(calib_paras, eval_paras, max_details=None):
    """返回 [{eval_index, calib_indices, full_equal, ...}]，按 eval 顺序。

    `calib_paras` / `eval_paras` 可以是 `paragraphs()` 的字符串列表，
    也可以是 `paragraphs_with_offsets()` 的 (offset, str) 列表。
    """
    def _norm_in(seq):
        out = []
        for i, item in enumerate(seq):
            if isinstance(item, (tuple, list)):
                off, raw = item
            else:
                off, raw = None, item
            out.append(para_record(i, raw, off))
        return out

    c = _norm_in(calib_paras)
    e = _norm_in(eval_paras)
    cidx = {}
    for r in c:
        if r["key200"]:
            cidx.setdefault(r["key200"], []).append(r)

    details = []
    for er in e:
        if not er["key200"]:
            continue
        cands = cidx.get(er["key200"])
        if not cands:
            continue
        full_equal = any(cr["full_sha256"] == er["full_sha256"] for cr in cands)
        details.append({
            "eval_index": er["index"],
            "eval_offset": er["offset"],
            "eval_full_sha256": er["full_sha256"],
            "eval_norm_len": er["norm_len"],
            "eval_raw_len": er["raw_len"],
            "eval_is_heading": er["is_heading"],
            "eval_preview": er["preview"],
            "calib_indices": [cr["index"] for cr in cands][:5],
            "calib_offsets": [cr["offset"] for cr in cands][:5],
            "calib_full_sha256": [cr["full_sha256"] for cr in cands][:3],
            "calib_norm_lens": [cr["norm_len"] for cr in cands][:3],
            "full_equal": full_equal,
            # 只有前 200 字符相同、整段不同 —— 这正是"不能据此断言污染"的情形
            "same_prefix200_only": not full_equal,
        })
    if max_details is not None:
        details = details[:max_details]
    return details


# ---------------------------------------------------------------- imatrix 窗口

def prefix_window(text, need_tokens, chars_per_token=CHARS_PER_TOKEN):
    """llama-imatrix 顺序读文件 → 前 `need_tokens` 个 token 的**近似**字符窗口。

    `llama-imatrix` 没有断点、也不写 manifest；它从文件开头按 `-c` 切块消费
    `--chunks` 块。所以"实际看过哪些字"最接近的近似是文件前缀。
    但 token ≠ 字符，这里用 4 chars/token 的英文经验值，**只能叫近似**。
    """
    need_tokens = int(need_tokens or 0)
    end = min(len(text), int(need_tokens * float(chars_per_token)))
    body = text[:end]
    return {
        "level": "approx_derived_prefix_window",
        "start_char": 0,
        "end_char": end,
        "chars": end,
        "text_sha256": sha256_text(body),
        "need_tokens": need_tokens,
        "chars_per_token": float(chars_per_token),
        "method": (f"prefix of {need_tokens} tokens × {chars_per_token} chars/token；"
                   "llama-imatrix 顺序读文件，但本仓库没有分词器，所以是近似"),
        "caveat": "近似值。**不能**当作『imatrix 只看过这一段』的证明。",
    }


def imatrix_overlap(calib_text, overlap_details, manifest=None, need_tokens=None):
    """`actual_imatrix_calib_overlap` —— 默认 `unknown`，近似结果单独放子字段。

    `manifest` 是本次 S3 写下的校准 manifest（含 `prefix_window`）。
    只有它存在时 value 才不是 unknown；等级仍只是 `derived_prefix_window`。
    """
    need_tokens = need_tokens or (CFG.get("IMATRIX_CHUNKS", 32)
                                  * CFG.get("IMATRIX_CTX", 512))
    win = (manifest or {}).get("prefix_window") or prefix_window(calib_text, need_tokens)
    inside = []
    for d in overlap_details:
        for off in (d.get("calib_offsets") or []):
            if off is None:
                continue
            if off < win["end_char"]:
                inside.append({"eval_index": d["eval_index"],
                               "calib_offset": off,
                               "eval_preview": d["eval_preview"]})
                break
    check = {
        "window": {k: win.get(k) for k in
                   ("level", "start_char", "end_char", "chars", "need_tokens",
                    "chars_per_token", "text_sha256", "method", "caveat")},
        "overlap_paras_inside_window": len(inside),
        "inside": inside[:10],
        "meaning": ("`overlap_paras_inside_window=0` 只能说明：按近似窗口，"
                    "这些 train/test 共同段落**大概没被 imatrix 读过**。"
                    "它**不是**『已证明无污染』。"),
    }
    if manifest:
        return {
            "value": ("present" if inside else "none"),
            "level": "derived_prefix_window",
            "why": ("本次运行产出了 imatrix，并写下了校准窗口（近似）。"
                    "窗口内的交集条数见 prefix_window_check。"),
            "prefix_window_check": check,
        }
    return {
        "value": "unknown",
        "level": "unknown",
        "why": ("本次没有运行 S3：imatrix 来自 HF 上已存的产物，"
                "实际消费的 token 序列无法从产物恢复（llama-imatrix 不写 manifest）。"
                "所以**不能**声称已严格证明不重叠。"),
        "prefix_window_check": check,
    }


# ---------------------------------------------------------------- clean holdout

def build_holdout(calib_path, eval_path, out_dir, imatrix_manifest=None,
                  cfg=None, eval_label=None):
    """生成 `eval_holdout.txt` + `isolation.json`。返回 manifest dict。

    失败语义（fail-closed，但不再"1 行就全否"）：
      · 剔除比例 > `CFG['ISOLATION_MAX_DROP_FRAC']`（默认 10%）→ 说明 split 配错了
        （典型：把 train 当 eval），必须停下来人工看，不能静默清掉大半评测集；
      · 剔除后剩余段落 < `CFG['ISOLATION_MIN_PARAS']`（默认 20）→ 样本太小，
        NLL 没有意义；
      · 其它情况一律放行，并在 manifest 里写清剔除了什么。
    """
    cfg = cfg or CFG
    calib_path, eval_path = Path(calib_path), Path(eval_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_text = calib_path.read_text(errors="replace")
    eval_text = eval_path.read_text(errors="replace")
    c_par = paragraphs_with_offsets(calib_text)
    e_par = paragraphs_with_offsets(eval_text)

    details = find_overlaps(c_par, e_par)
    drop = {d["eval_index"] for d in details}
    n_eval = len(e_par)
    frac = len(drop) / max(1, n_eval)

    max_frac = float(cfg.get("ISOLATION_MAX_DROP_FRAC", 0.10))
    min_keep = int(cfg.get("ISOLATION_MIN_PARAS", 20))
    if frac > max_frac:
        raise GuardError(
            f"评测集里有 {len(drop)}/{n_eval} 段（{frac:.1%}）与校准集共享前 200 字符，"
            f"超过上限 {max_frac:.0%}。\n"
            "  → 这不像'维基百科里的重复句式'，更像**把 train 当成了 eval**。\n"
            f"  → 先核对 CFG['EVAL_TXT']（当前 {eval_path}）是不是真的 test split；\n"
            "  → 不要调大上限来绕过：那会把评测集清空。")

    kept = [(off, raw) for i, (off, raw) in enumerate(e_par) if i not in drop]
    if len(kept) < min_keep:
        raise GuardError(
            f"clean holdout 只剩 {len(kept)} 段（下限 {min_keep}）—— 评测集太小，"
            "NLL 没有意义。换更大的 eval split，或减小 MIN_PARA_CHARS 之前先问清楚。")

    body = "\n\n".join(raw for _off, raw in kept) + "\n"
    holdout_path = out_dir / "eval_holdout.txt"
    holdout_path.write_text(body, encoding="utf-8")

    removed_sha = [d["eval_full_sha256"] for d in details]
    kept_chars = len(body)
    manifest = {
        "schema": 1,
        "rule": HOLDOUT_RULE,
        "prefix_chars": PREFIX_CHARS,
        "min_para_chars": MIN_PARA_CHARS,
        "normalization": "NFKC + collapse-whitespace + casefold",
        "calib": {
            "path": str(calib_path),
            "sha256": sha256_text(calib_text),
            "paras": len(c_par),
            "chars": len(calib_text),
        },
        "eval_raw": {
            "path": str(eval_path),
            "label": eval_label or CFG.get("EVAL_SOURCE") or None,
            "sha256": sha256_text(eval_text),
            "paras": n_eval,
            "chars": len(eval_text),
        },
        "overlap_raw": {
            "paras": len(details),
            "fraction": round(frac, 6),
            "details": details,
            "note": ("这是**原始**交集，不是结论。`same_prefix200_only=true` 表示"
                     "两段只有前 200 字符相同、整段不同；这类情形多半是维基百科的"
                     "标题/列表项，不能据此断言污染。"),
        },
        "cleaning": {
            "removed_paras": len(drop),
            "removed_eval_indices": sorted(drop)[:200],
            "removed_full_sha256": removed_sha[:200],
            "kept_paras": len(kept),
            "kept_chars": kept_chars,
            "kept_tokens_est": int(kept_chars / CHARS_PER_TOKEN),
            "token_est_method": f"chars/{CHARS_PER_TOKEN:g}（近似，不是分词器计数）",
            "max_drop_frac": max_frac,
            "min_keep_paras": min_keep,
        },
        "holdout": {
            "path": str(holdout_path),
            "sha256": sha256_text(body),
            "paras": len(kept),
            "chars": kept_chars,
        },
        "actual_imatrix_calib_overlap": imatrix_overlap(
            calib_text, details, manifest=imatrix_manifest,
            need_tokens=(imatrix_manifest or {}).get("need_tokens")),
        "ok": True,
        "note": ("两个量化用**同一份** holdout（S6 只切一次组）。"
                 "原始 eval 文本未被修改，holdout 是独立文件，可复核。"),
    }
    write_json(out_dir / "isolation.json", manifest)

    log(f"隔离取证：eval {n_eval} 段，与 calib 共享前 200 字符的 {len(details)} 段"
        f"（{frac:.2%}），其中整段完全相同 "
        f"{sum(1 for d in details if d['full_equal'])} 段")
    if details:
        for d in details[:5]:
            log(f"  · eval#{d['eval_index']} "
                f"{'整段相同' if d['full_equal'] else '仅前200字符相同'}"
                f" ← calib{d['calib_indices']}  「{d['eval_preview'][:60]}」")
    log(f"clean holdout：剔除 {len(drop)} 段，保留 {len(kept)} 段 / "
        f"{kept_chars} 字符（≈{int(kept_chars / CHARS_PER_TOKEN)} token）"
        f" → {holdout_path.name} sha256 {manifest['holdout']['sha256'][:16]}…")
    _ov = manifest["actual_imatrix_calib_overlap"]
    log(f"actual_imatrix_calib_overlap = {_ov['value']}（level {_ov['level']}）"
        + ("" if _ov["value"] != "unknown"
           else " —— 这是诚实标注，不等于『已证明无污染』"))
    return manifest
