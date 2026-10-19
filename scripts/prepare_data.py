"""Create deterministic token-only JSONL windows for Stage 1.

Every version that can silently change the numbers is pinned and recorded:
model repo + commit, tokenizer repo + commit, dataset repo + commit, seed,
window offsets and a SHA256 over each token window and over the whole corpus.

Usage:
    python scripts/prepare_data.py --out data --seq-len 129
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)} windows")


def make_windows(token_ids: list[int], source: str, split: str, count: int,
                 seq_len: int, seed: int) -> list[dict]:
    if len(token_ids) < seq_len:
        raise ValueError(f"{source}:{split} has fewer than {seq_len} tokens")
    max_start = len(token_ids) - seq_len
    offsets = list(range(0, max_start + 1, seq_len))
    rng = random.Random(seed)
    rng.shuffle(offsets)
    starts = sorted(offsets[: min(count, len(offsets))])
    result = []
    for start in starts:
        ids = token_ids[start:start + seq_len]
        result.append({
            "sample_id": f"{source}:{split}:{start}",
            "source": source,
            "split": split,
            "start_offset": start,
            "ids": ids,
            "tokens_sha256": sha_ids(ids),
        })
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-Base-bf16",
                    help="tokenizer is taken from the model repo so they cannot diverge")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq-len", type=int, default=129)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--calibration", type=int, default=64)
    ap.add_argument("--validation", type=int, default=32)
    ap.add_argument("--test", type=int, default=64)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi()
    model_info = api.model_info(args.model)
    ds_info = api.dataset_info(args.dataset)
    model_rev, ds_rev = model_info.sha, ds_info.sha

    tok = AutoTokenizer.from_pretrained(args.model, revision=model_rev, use_fast=True)

    splits = [("train", args.calibration), ("validation", args.validation),
              ("test", args.test)]
    corpus = {}
    for split, count in splits:
        rows = load_dataset(args.dataset, args.config, split=split, revision=ds_rev)
        text = "\n\n".join(r["text"] for r in rows if r["text"].strip())
        ids = tok.encode(text, add_special_tokens=False)
        corpus[split] = {"n_tokens": len(ids), "corpus_sha256": sha_text(text),
                         "n_documents": len(rows)}
        windows = make_windows(ids, "wikitext2", split, count, args.seq_len, args.seed)
        target = {"train": "calibration", "validation": "validation", "test": "test"}[split]
        dump(args.out / f"{target}.jsonl", windows)

    manifest = {
        "model_repo": args.model,
        "model_revision": model_rev,
        "tokenizer_repo": args.model,
        "tokenizer_revision": model_rev,
        "tokenizer_class": type(tok).__name__,
        "vocab_size": tok.vocab_size,
        "dataset": args.dataset,
        "dataset_config": args.config,
        "dataset_revision": ds_rev,
        "seq_len": args.seq_len,
        "n_predicted_tokens": args.seq_len - 1,
        "seed": args.seed,
        "split_targets": {"train": "calibration", "validation": "validation",
                          "test": "test"},
        "requested_windows": {s: c for s, c in splits},
        "corpus": corpus,
        "concat_rule": "join non-empty rows with '\\n\\n', then tokenize without special tokens",
        "window_rule": "non-overlapping windows of seq_len tokens; offsets shuffled by seed; sorted by start",
        "note": "Token ids stay local. Do not commit the sampled corpus to a public repo.",
    }
    (args.out / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"model_revision={model_rev} dataset_revision={ds_rev}")


if __name__ == "__main__":
    main()
