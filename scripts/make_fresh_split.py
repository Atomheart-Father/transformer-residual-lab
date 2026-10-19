"""Draw a fresh, provably disjoint evaluation split.

The Stage 1 / 1.5 test windows have been inspected many times while choosing
plans, so they can no longer serve as an untouched test set. This script draws
new windows from the *same* corpus with a *different* seed and then removes
every offset that appears in the existing files, so the new split cannot
overlap the old one by even a single token.

The offsets are multiples of seq_len, so disjoint offsets imply disjoint token
ranges and therefore disjoint text.

Output:
    <out>/fresh_test.jsonl          new windows
    <out>/fresh_test_manifest.json  seed, excluded offsets, disjointness proof

Usage:
    python scripts/make_fresh_split.py --out data257 --split test --count 64 \
        --seq-len 257 --seed 771013 --exclude data257/test.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data257"))
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-Base-bf16")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seq-len", type=int, default=257)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--seed", type=int, default=771013)
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="existing JSONL whose windows must not be reused")
    ap.add_argument("--name", default="fresh_test")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi()
    model_rev = api.model_info(args.model).sha
    ds_rev = api.dataset_info(args.dataset).sha

    tok = AutoTokenizer.from_pretrained(args.model, revision=model_rev, use_fast=True)
    rows = load_dataset(args.dataset, args.config, split=args.split, revision=ds_rev)
    text = "\n\n".join(r["text"] for r in rows if r["text"].strip())
    ids = tok.encode(text, add_special_tokens=False)

    excluded_offsets, excluded_hashes = set(), set()
    for path in args.exclude:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            excluded_offsets.add(int(rec["start_offset"]))
            excluded_hashes.add(rec["tokens_sha256"])

    max_start = len(ids) - args.seq_len
    pool = [o for o in range(0, max_start + 1, args.seq_len)
            if o not in excluded_offsets]
    if len(pool) < args.count:
        raise SystemExit(f"only {len(pool)} disjoint windows available, "
                         f"{args.count} requested")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    starts = sorted(pool[:args.count])

    windows = []
    for start in starts:
        w = ids[start:start + args.seq_len]
        h = sha_ids(w)
        if h in excluded_hashes:
            raise SystemExit(f"window at {start} duplicates an excluded window")
        windows.append({
            "sample_id": f"wikitext2:{args.split}:{start}",
            "source": "wikitext2",
            "split": args.split,
            "start_offset": start,
            "ids": w,
            "tokens_sha256": h,
        })

    if len({w["start_offset"] for w in windows}) != args.count:
        raise SystemExit("duplicate offsets in the fresh split")
    if {w["start_offset"] for w in windows} & excluded_offsets:
        raise SystemExit("fresh split overlaps the excluded set")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for w in windows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    manifest = {
        "name": args.name,
        "purpose": "untouched evaluation split for Stage 1.6; never used for "
                   "plan selection, alpha fitting, or compensator training",
        "model_revision": model_rev,
        "dataset": args.dataset,
        "dataset_config": args.config,
        "dataset_revision": ds_rev,
        "split": args.split,
        "seq_len": args.seq_len,
        "n_windows": len(windows),
        "seed": args.seed,
        "excluded_files": [str(p) for p in args.exclude],
        "n_excluded_offsets": len(excluded_offsets),
        "n_candidate_offsets": len(pool) + len(excluded_offsets & set(
            range(0, max_start + 1, args.seq_len))),
        "disjointness": "offsets are multiples of seq_len, so distinct offsets "
                        "imply distinct token ranges; also checked by sha256",
        "window_rule": "non-overlapping windows of seq_len tokens; offsets "
                       "shuffled by seed; sorted by start",
        "note": "Token ids stay local. Do not commit the sampled corpus.",
    }
    (args.out / f"{args.name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{path}: {len(windows)} windows, seed={args.seed}, "
          f"{len(excluded_offsets)} offsets excluded")


if __name__ == "__main__":
    main()
