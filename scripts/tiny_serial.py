"""Stage 1.8: teacher-derived lightweight serial path.

The question is *not* "can a small network predict the residual". Stage 1.7
already showed that a randomly initialised gated cell loses to a plain MLP at
the same budget, and that says nothing about whether the serial computation
itself can be replaced.

The question here is: can the serial work done by teacher blocks 3..10 be
carried by a narrow, cheap, *independently executable* path whose weights come
from the teacher?

Graph
-----
    h3 --+--> [ teacher blocks 3..10, parallel group ] --> h11_P
         |
         +--> [ tiny serial path, hidden 128 ]        --> z
                                                        |
                       c_hat = z - (h11_P - h3)   -+--> h11_P + c_hat

The tiny path reads only `h3`, so it never waits for the heavy group. That is
the property Stage 1.7's compensator did not have.

Stages
------
    collect -> cache h3, h11_T, h11_P for a split (teacher states on disk)
    extract -> PCA rotation, channel/neuron selection, weight inheritance
    fit     -> train A1 (serial MLP) / A2 (tiny, random) / A3 (tiny, inherited)
    fuse    -> train the small head that turns z into c_hat
    eval    -> free-run irr-a with the tiny path correcting group 3's exit
    bench   -> latency, params, FLOPs

Every number in the report comes from the files these stages write.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from measure import (
    PINNED_MODEL_REPO,
    PINNED_MODEL_REVISION,
    embed,
    git_sha,
    logits_metrics,
    project_logits,
    resolve_plan,
)
from serial_compensate import run_group

def to_np(x) -> np.ndarray:
    """mlx bf16 buffers are uint8 to numpy; go through fp32 first."""
    return np.asarray(mx.asarray(x).astype(mx.float32))


GROUP = [3, 4, 5, 6, 7, 8, 9, 10]   # the irr-a group whose exit we correct
SITE = 3                            # its index in irr-a
H11 = GROUP[-1] + 1                 # index of the teacher state at the exit


# ---------------------------------------------------------------------------
# tiny model
# ---------------------------------------------------------------------------


class MlpBlock(nn.Module):
    """A1's block: same width, same residual shape, no attention."""

    def __init__(self, dim: int, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x, mask=None):
        h = self.norm(x)
        a, b = self.w1(h), self.w3(h)
        return x + self.w2(a * mx.sigmoid(a) * b)


class TinySerial(nn.Module):
    """Narrow residual stack between two projections.

    `kind="tiny"` uses the teacher's own Qwen3 block class, so RoPE, RMSNorm,
    GQA and the residual layout are identical by construction. `kind="mlp"` is
    the same stack with attention removed.
    """

    def __init__(self, kind: str, dim_in: int, hidden: int, n_layers: int,
                 head_dim: int, n_heads: int, intermediate: int, eps: float,
                 rope_theta: float, max_pos: int):
        super().__init__()
        self.kind = kind
        self.dim_in = dim_in
        self.hidden = hidden
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.intermediate = intermediate

        self.proj_in = nn.Linear(dim_in, hidden, bias=False)
        self.proj_out = nn.Linear(hidden, dim_in, bias=False)
        if kind == "mlp":
            self.blocks = [MlpBlock(hidden, intermediate, eps) for _ in range(n_layers)]
        else:
            from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock

            args = ModelArgs(
                model_type="qwen3", hidden_size=hidden, num_hidden_layers=n_layers,
                intermediate_size=intermediate, num_attention_heads=n_heads,
                rms_norm_eps=eps, vocab_size=1, num_key_value_heads=n_heads,
                max_position_embeddings=max_pos, rope_theta=rope_theta,
                head_dim=head_dim, tie_word_embeddings=True, rope_scaling=None,
            )
            self.blocks = [TransformerBlock(args) for _ in range(n_layers)]

    def __call__(self, x, mask=None):
        """c_hat-shaped output: the predicted *change* of the state.

        Reading the difference of the projected residual instead of the
        projected residual matters more than it looks. A teacher block is a
        near-cancellation: it emits a state of RMS 9.7 whose difference from its
        input is only ~0.35. A narrow model that has not learnt that
        cancellation exactly produces a state off by ~3 and looks broken, when
        the part we actually need (the change) may be fine. Predicting the
        change also starts every variant at exactly zero, which is the honest
        baseline.
        """
        h0 = self.proj_in(x.astype(mx.float32))
        h = h0
        for blk in self.blocks:
            h = blk(h, mask, None) if self.kind != "mlp" else blk(h, mask)
        return self.proj_out(h - h0)

    def n_params(self) -> int:
        return int(sum(np.asarray(v).size for _, v in _flat(self.parameters())))


def build_tiny(args) -> TinySerial:
    return TinySerial(args.kind, args.dim_in, args.hidden, args.n_layers,
                      args.head_dim, args.n_heads, args.intermediate, args.eps,
                      args.rope_theta, args.max_pos)


def build_from_checkpoint(path: Path) -> TinySerial:
    """Rebuild a unit from its own metadata.

    `fuse` / `eval` / `bench` must not trust the CLI for the architecture: A1
    (mlp, intermediate 640) and A2/A3 (tiny, intermediate 512) would silently
    crash or, worse, load a tree into a mismatched shape. The checkpoint meta is
    the single source of truth once a unit has been fitted.
    """
    flat, meta = load_tree(path)
    unit = TinySerial(
        meta["kind"], meta.get("dim_in", 1024), meta["hidden"],
        meta["n_layers"], meta.get("head_dim", 128), meta.get("n_heads", 1),
        meta.get("intermediate", 512), meta.get("eps", 1e-6),
        meta.get("rope_theta", 1_000_000.0), meta.get("max_pos", 32768))
    unit.update(flat_to_tree(flat, unit))
    return unit


def tiny_flops(tiny: TinySerial, seq_len: int) -> int:
    """2 * MACs per token, with the attention term at this sequence length."""
    h, d = tiny.hidden, tiny.head_dim
    macs = 2 * h * tiny.dim_in
    if tiny.kind == "mlp":
        macs += tiny.n_layers * 3 * tiny.intermediate * h
    else:
        macs += tiny.n_layers * (4 * h * h + 2 * seq_len * d
                                 + 3 * tiny.intermediate * h)
    return int(2 * macs)


# ---------------------------------------------------------------------------
# flat parameter trees (npz round-trip)
# ---------------------------------------------------------------------------


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat(v, f"{prefix}.{i}")
    else:
        yield prefix, tree


def save_tree(path: Path, module: nn.Module, meta: dict) -> None:
    flat = {k: to_np(v) for k, v in _flat(module.parameters())}
    np.savez_compressed(path, meta=json.dumps(meta), **flat)


def load_tree(path: Path) -> tuple[dict, dict]:
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"]))
    flat = {k: blob[k] for k in blob.files if k != "meta"}
    return flat, meta


def untree(flat: dict, template) -> dict:
    if isinstance(template, dict):
        return {k: untree(flat, v) for k, v in template.items()}
    if isinstance(template, list):
        return [untree(flat, v) for v in template]
    raise TypeError("template leaf without a path")


def flat_to_tree(flat: dict, module: nn.Module) -> dict:
    """Rebuild the nested tree by walking the module's own parameter shapes."""
    paths = {k: None for k, _ in _flat(module.parameters())}

    def build(prefix, node):
        if isinstance(node, dict):
            return {k: build(f"{prefix}.{k}" if prefix else k, v)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [build(f"{prefix}.{i}", v) for i, v in enumerate(node)]
        if prefix not in paths:
            raise KeyError(f"{prefix} not in the checkpoint")
        return mx.array(flat[prefix])

    return build("", module.parameters())


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def teacher_slice(model, ids, mask):
    """States up to the end of group 3, without running the tail of the model."""
    h = embed(model, ids, mx)
    states = [h]
    for i, blk in enumerate(model.model.layers):
        out = blk(h, mask, None)
        mx.eval(out)
        states.append(out)
        h = out
        if i >= GROUP[-1]:
            break
    return states


def cmd_collect(args):
    if not args.input:
        raise SystemExit("--input is required for collect")
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    # `--data` names the split directory the fit/eval stages will later read; if
    # it is absent fall back to --output. Writing to --output while ignoring
    # --data silently puts the cache where nothing looks for it.
    out_dir = args.data if args.data else args.output

    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    rows = []
    for path in args.input:
        rows.extend(json.loads(l) for l in path.open(encoding="utf-8"))
    sources = [str(p) for p in args.input]
    n = min(len(rows), args.max_samples)
    T = len(rows[0]["ids"])
    D = model.args.hidden_size

    h3 = np.zeros((n, T, D), dtype=np.float16)
    h11T = np.zeros((n, T, D), dtype=np.float16)
    h11P = np.zeros((n, T, D), dtype=np.float16)
    rms = {}
    t0 = time.perf_counter()
    for i, rec in enumerate(rows[:n]):
        ids = mx.array(rec["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states = teacher_slice(model, ids, mask)
        a = states[3]
        par = run_group(model, a, GROUP, mask, mx)
        mx.eval(par)
        h3[i] = to_np(a)[0].astype(np.float16)
        h11T[i] = to_np(states[H11])[0].astype(np.float16)
        h11P[i] = to_np(par)[0].astype(np.float16)
        if not rms:
            for L in range(0, H11 + 1):
                x = to_np(states[L]).reshape(-1, D)
                rms[str(L)] = {"rms_full": float(np.sqrt((x ** 2).mean()))}
        if (i + 1) % 64 == 0:
            print(f"  {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "h3.npy", h3)
    np.save(out_dir / "h11T.npy", h11T)
    np.save(out_dir / "h11P.npy", h11P)
    manifest = {
        "stage": "collect", "input": sources, "n_samples": n,
        "seq_len": T, "hidden": D, "group": GROUP, "arrays":
        ["h3.npy", "h11T.npy", "h11P.npy"], "dtype": "float16",
        "n_predicted_tokens": n * (T - 1), "rms_full": rms,
        "seconds": round(time.perf_counter() - t0, 1),
        "model": args.model, "model_revision": args.revision,
        "git_commit": git_sha(),
    }
    (out_dir / "collect_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rms_full"},
                     indent=2))


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def pca_rotation(H, dim, n_keep, n_tokens=40000, seed=0):
    """Top eigenvectors of the residual covariance at h3.

    SliceGPT in miniature: rotate into a variance-ordered basis, keep the
    leading coordinates. The rotation is orthogonal, so nothing is lost until
    the truncation.
    """
    X = np.asarray(H, dtype=np.float32).reshape(-1, dim)
    if X.shape[0] > n_tokens:
        idx = np.random.default_rng(seed).choice(X.shape[0], n_tokens, replace=False)
        X = X[idx]
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / X.shape[0]
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    pivot = np.argmax(np.abs(evecs), axis=0)
    signs = np.sign(evecs[pivot, np.arange(dim)])
    signs[signs == 0] = 1.0
    R = (evecs * signs).astype(np.float32)
    return R, {
        "n_tokens_used": int(X.shape[0]),
        "energy_share_top_keep": float(evals[:n_keep].sum() / evals.sum()),
        "top_evals": [float(x) for x in evals[:6]],
    }


def neuron_importance(model, teacher_layers, keep_int):
    """Structured neuron score: written hard x read hard x propagated hard.

    LLM-Pruner's criterion without its recovery stage. Data-independent, so it
    cannot leak evaluation data into the choice of channels.
    """
    picks = {}
    for L in teacher_layers:
        mlp = model.model.layers[L].mlp
        wg, wu, wd = (to_np(mlp.gate_proj.weight), to_np(mlp.up_proj.weight),
                      to_np(mlp.down_proj.weight))
        score = (np.linalg.norm(wg, axis=1) * np.linalg.norm(wu, axis=1)
                 * np.linalg.norm(wd, axis=0))
        picks[L] = np.sort(np.argsort(score)[::-1][:keep_int])
    return picks


def head_choice(model, teacher_layers, head_dim):
    """Which query head (and its GQA partner) to keep. Qwen3: 16 q / 8 kv."""
    out = {}
    for L in teacher_layers:
        attn = model.model.layers[L].self_attn
        q, o = to_np(attn.q_proj.weight), to_np(attn.o_proj.weight)
        n_q = q.shape[0] // head_dim
        n_kv = to_np(attn.k_proj.weight).shape[0] // head_dim
        ratio = n_q // n_kv
        scores = []
        for h in range(n_q):
            qh = np.linalg.norm(q[h * head_dim:(h + 1) * head_dim])
            oh = np.linalg.norm(o[:, h * head_dim:(h + 1) * head_dim])
            scores.append(float(qh * oh))
        h = int(np.argmax(scores))
        out[L] = {"q_head": h, "kv_head": h // ratio, "score": scores[h],
                  "n_q_heads": n_q, "n_kv_heads": n_kv}
    return out


def cmd_extract(args):
    """Build the A3 initialisation and write it next to the manifest."""
    if not args.input:
        raise SystemExit("--input is required for extract (calibration windows)")
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    tiny = build_tiny(args)
    teacher_layers = [int(x) for x in args.teacher_layers.split(",")]
    if len(teacher_layers) != args.n_layers:
        raise SystemExit("--teacher-layers must list one layer per tiny block")

    data = Path(args.data)
    h3 = np.load(data / "h3.npy", mmap_mode="r")
    R, rinfo = pca_rotation(h3, args.dim_in, args.hidden, seed=args.seed)

    keep_int = neuron_importance(model, teacher_layers, args.keep_int)
    heads = head_choice(model, teacher_layers, args.head_dim)

    # per-layer rescale factors: the 128-wide norm cannot use a 1024-wide RMS
    rows = []
    for path in args.input:
        rows.extend(json.loads(l) for l in path.open(encoding="utf-8"))
    n = min(len(rows), args.max_samples)
    acc = {L: [0.0, 0.0, 0] for L in teacher_layers}
    Rt = R.T
    for rec in rows[:n]:
        ids = mx.array(rec["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        states = teacher_slice(model, ids, mask)
        for L in teacher_layers:
            x = to_np(states[L]).reshape(-1, args.dim_in)
            xr = x @ Rt.T                      # x_rot = R^T x
            acc[L][0] += float((x ** 2).sum())
            acc[L][1] += float((xr[:, :args.hidden] ** 2).sum())
            acc[L][2] += x.size
    factors = {}
    for L, (sf, sk, cnt) in acc.items():
        rms_full = math.sqrt(sf / cnt)
        rms_kept = math.sqrt(sk / (cnt / args.dim_in * args.hidden))
        factors[L] = {"rms_full": rms_full, "rms_kept": rms_kept,
                      "factor": rms_kept / max(rms_full, 1e-12)}

    # ---- weight inheritance
    tree = {}
    tree["proj_in"] = {"weight": mx.array(R[:, :args.hidden].T)}
    tree["proj_out"] = {"weight": mx.array(R[:, :args.hidden])}
    blocks = []
    Rm = mx.array(R)
    Rt_m = mx.array(Rt)
    for t, L in enumerate(teacher_layers):
        blk = model.model.layers[L]
        hd = args.head_dim
        hq, hkv = heads[L]["q_head"], heads[L]["kv_head"]
        q = (mx.array(to_np(blk.self_attn.q_proj.weight)) @ Rm)
        k = (mx.array(to_np(blk.self_attn.k_proj.weight)) @ Rm)
        v = (mx.array(to_np(blk.self_attn.v_proj.weight)) @ Rm)
        o = (Rt_m @ mx.array(to_np(blk.self_attn.o_proj.weight)))
        gate = (mx.array(to_np(blk.mlp.gate_proj.weight)) @ Rm)
        up = (mx.array(to_np(blk.mlp.up_proj.weight)) @ Rm)
        down = (Rt_m @ mx.array(to_np(blk.mlp.down_proj.weight)))
        nidx = keep_int[L]
        nidx_mx = mx.array(nidx)      # mlx needs an array index, not a np array
        f = factors[L]["factor"]
        g_in = np.asarray(Rt_m @ mx.array(
            to_np(blk.input_layernorm.weight)))[:args.hidden] * f
        g_post = np.asarray(Rt_m @ mx.array(
            to_np(blk.post_attention_layernorm.weight)))[:args.hidden] * f
        blocks.append({
            "self_attn": {
                "q_proj": {"weight": q[hq * hd:(hq + 1) * hd, :args.hidden]},
                "k_proj": {"weight": k[hkv * hd:(hkv + 1) * hd, :args.hidden]},
                "v_proj": {"weight": v[hkv * hd:(hkv + 1) * hd, :args.hidden]},
                "o_proj": {"weight": o[:args.hidden, hq * hd:(hq + 1) * hd]},
                "q_norm": {"weight": mx.array(
                    to_np(blk.self_attn.q_norm.weight))},
                "k_norm": {"weight": mx.array(
                    to_np(blk.self_attn.k_norm.weight))},
            },
            "mlp": {
                "gate_proj": {"weight": gate[nidx_mx][:, :args.hidden]},
                "up_proj": {"weight": up[nidx_mx][:, :args.hidden]},
                "down_proj": {"weight": down[:args.hidden, nidx_mx]},
            },
            "input_layernorm": {"weight": mx.array(g_in)},
            "post_attention_layernorm": {"weight": mx.array(g_post)},
        })
    tree["blocks"] = blocks

    # align with the module's own parameter layout before writing
    flat = {}
    for k, v in _flat(tree):
        flat[k] = np.asarray(v)
    try:
        tiny.update(flat_to_tree(flat, tiny))
    except KeyError as e:
        raise SystemExit(f"extracted tree does not match the module: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "rotation.npy", R)
    save_tree(args.output / "a3_init.npz", tiny, {
        "kind": args.kind, "hidden": args.hidden, "n_layers": args.n_layers,
        "head_dim": args.head_dim, "n_heads": args.n_heads,
        "intermediate": args.intermediate, "dim_in": args.dim_in,
        "eps": args.eps, "rope_theta": args.rope_theta,
        "max_pos": args.max_pos, "init_source": "teacher_weights",
        "teacher_layers": teacher_layers, "n_params": tiny.n_params(),
        "tag": "a3_init", "seed": args.seed,
    })
    manifest = {
        "stage": "extract",
        "method": "PCA rotation of h3 + structured slicing + norm rescale",
        "teacher_layers": teacher_layers,
        "hidden_keep": args.hidden,
        "intermediate_keep": args.keep_int,
        "rotation": rinfo,
        "head_choice": {str(k): v for k, v in heads.items()},
        "norm_rescale": {str(k): v for k, v in factors.items()},
        "n_params": tiny.n_params(),
        "inherited": [
            "input_layernorm / post_attention_layernorm weights (rotated, sliced, rescaled)",
            "q/k/v/o projections (rotated, sliced to one head)",
            "q_norm / k_norm (unchanged, head_dim is already 128)",
            "gate/up/down projections (rotated, sliced to 512 neurons)",
            "proj_in / proj_out (the kept rows/columns of the rotation)",
        ],
        "reinitialised": [],
        "recalibrated": ["RMSNorm weights: x1 scalar per norm, measured, not fitted"],
        "not_faithful": (
            "This is not a function-preserving compression: 1024 -> 128 channels "
            "loses 87.5% of the residual coordinates, and one attention head "
            "replaces 16. Fidelity must be measured, not assumed."),
        "flops_per_token": tiny_flops(tiny, 257),
        "git_commit": git_sha(),
    }
    (args.output / "extract_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("hidden_keep", "intermediate_keep", "n_params",
                       "flops_per_token", "rotation")}, indent=2))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def load_split(path: Path, n_max=None):
    h3 = np.load(path / "h3.npy", mmap_mode="r")
    tgt = np.load(path / "h11T.npy", mmap_mode="r")
    par = np.load(path / "h11P.npy", mmap_mode="r")
    n = h3.shape[0] if n_max is None else min(n_max, h3.shape[0])
    return h3[:n], tgt[:n], par[:n]


def cmd_fit(args):
    # nn.Linear draws from the global key, so seed it: A2 must be reproducible
    mx.random.seed(args.seed)
    tiny = build_tiny(args)
    if args.init_from:
        flat, meta = load_tree(args.init_from)
        tiny.update(flat_to_tree(flat, tiny))
    init_source = args.init_from.name if args.init_from else "random"

    h3, h11T, _ = load_split(Path(args.data))
    n, T, D = h3.shape
    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    target_rms = float(np.sqrt(np.mean(
        (np.asarray(h11T[tr_idx], dtype=np.float32)
         - np.asarray(h3[tr_idx], dtype=np.float32)) ** 2)))

    def tensors(idx):
        a = np.asarray(h3[idx], dtype=np.float32)
        t = np.asarray(h11T[idx], dtype=np.float32) - a
        return mx.array(a), mx.array(t)

    def loss_fn(mod, a, t):
        return mx.mean(((mod(a, "causal") - t) / target_rms) ** 2)

    def measure(mod, idx):
        a, t = tensors(idx)
        pred = mod(a, "causal")
        mx.eval(pred)
        d = np.asarray(pred - t, dtype=np.float32)
        tt = np.asarray(t, dtype=np.float32)
        return {
            "rel_err": float(np.sqrt((d ** 2).mean())
                             / max(np.sqrt((tt ** 2).mean()), 1e-9)),
            "cosine": float((pred * t).sum()
                            / (mx.sqrt((pred * pred).sum() * (t * t).sum()) + 1e-9)),
            "target_rms": float(np.sqrt((tt ** 2).mean())),
            "pred_rms": float(np.sqrt((np.asarray(pred) ** 2).mean())),
        }

    before = {"val": measure(tiny, val_idx), "train": measure(tiny, tr_idx)}
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    grad_fn = nn.value_and_grad(tiny, loss_fn)
    steps = max(1, len(tr_idx) // args.batch_samples)
    history, t0 = [], time.perf_counter()
    for ep in range(args.epochs):
        r2 = np.random.default_rng(args.seed + ep)
        run = 0.0
        for _ in range(steps):
            pick = r2.choice(tr_idx, size=min(args.batch_samples, len(tr_idx)),
                             replace=False)
            a, t = tensors(pick)
            loss, grads = grad_fn(tiny, a, t)
            opt.update(tiny, grads)
            mx.eval(tiny.parameters(), opt.state, loss)
            run += float(loss)
        rec = {"epoch": ep + 1, "train_loss": run / steps}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = measure(tiny, val_idx)
            rec.update({f"val_{k}": v for k, v in m.items()})
            print(f"  ep{ep + 1}: loss={rec['train_loss']:.4f} "
                  f"rel={m['rel_err']:.4f} cos={m['cosine']:.4f}", flush=True)
        history.append(rec)

    after = {"val": measure(tiny, val_idx), "train": measure(tiny, tr_idx)}
    args.output.mkdir(parents=True, exist_ok=True)
    save_tree(args.output / f"{args.tag}.npz", tiny, {
        "kind": args.kind, "hidden": args.hidden, "n_layers": args.n_layers,
        "head_dim": args.head_dim, "n_heads": args.n_heads,
        "intermediate": args.intermediate, "dim_in": args.dim_in,
        "eps": args.eps, "rope_theta": args.rope_theta,
        "max_pos": args.max_pos, "init_source": init_source,
        "n_params": tiny.n_params(), "tag": args.tag, "seed": args.seed,
    })
    report = {
        "tag": args.tag, "kind": args.kind, "init_source": init_source,
        "n_params": tiny.n_params(),
        "n_train_windows": int(len(tr_idx)), "n_val_windows": int(len(val_idx)),
        "n_train_tokens": int(len(tr_idx) * (T - 1)),
        "target_rms": target_rms, "epochs": args.epochs, "lr": args.lr,
        "batch_samples": args.batch_samples, "seed": args.seed,
        "before_training": before, "after_training": after,
        "history": history, "seconds": round(time.perf_counter() - t0, 1),
        "data": str(args.data), "git_commit": git_sha(),
        "flops_per_token": tiny_flops(tiny, T),
    }
    (args.output / f"fit_{args.tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("tag", "n_params", "before_training", "after_training")},
                     indent=2))


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


class FuseHead(nn.Module):
    def __init__(self, dim_in, hidden, dim_out):
        super().__init__()
        self.w1 = nn.Linear(dim_in, hidden, bias=True)
        self.w2 = nn.Linear(hidden, dim_out, bias=True)

    def __call__(self, x):
        return self.w2(nn.gelu(self.w1(x)))


def cmd_fuse(args):
    tiny = build_from_checkpoint(Path(args.compensator[0]))
    h3, h11T, h11P = load_split(Path(args.data))
    n = h3.shape[0]

    Z = np.zeros((n,) + h3.shape[1:], dtype=np.float32)
    for i in range(n):
        z = tiny(mx.array(np.asarray(h3[i:i + 1], dtype=np.float32)), "causal")
        mx.eval(z)
        Z[i] = np.asarray(z, dtype=np.float32)

    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

    mode = args.fusion_mode
    dim_in = h3.shape[-1] * (2 if mode == "trunk" else 3)
    head = FuseHead(dim_in, args.fuse_hidden, h3.shape[-1])
    n_params = int(sum(np.asarray(v).size for _, v in _flat(head.parameters())))

    def feats(idx):
        a = np.asarray(h3[idx], dtype=np.float32)
        p = np.asarray(h11P[idx], dtype=np.float32)
        z = Z[idx]
        if mode == "trunk":
            x = np.concatenate([a, p], axis=-1)
        elif mode == "zero_trunk":
            x = np.concatenate([a, a, z], axis=-1)
        else:
            x = np.concatenate([a, p, z], axis=-1)
        return mx.array(x)

    def target(idx):
        return mx.array(np.asarray(h11T[idx], dtype=np.float32)
                        - np.asarray(h11P[idx], dtype=np.float32))

    def loss_fn(mod, x, y):
        return mx.mean((mod(x) - y) ** 2)

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(head, loss_fn)
    steps = max(1, len(tr_idx) // args.batch_samples)
    hist = []
    for ep in range(args.epochs):
        r2 = np.random.default_rng(args.seed + ep)
        run = 0.0
        for _ in range(steps):
            pick = r2.choice(tr_idx, size=min(args.batch_samples, len(tr_idx)),
                             replace=False)
            x, y = feats(pick), target(pick)
            loss, grads = grad_fn(head, x, y)
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state, loss)
            run += float(loss)
        hist.append({"epoch": ep + 1, "train_mse": run / steps})

    def report(idx):
        x, y = feats(idx), target(idx)
        pred = head(x)
        mx.eval(pred)
        d = np.asarray(pred - y, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32)
        return {"rel_err": float(np.sqrt((d ** 2).mean())
                                 / max(np.sqrt((yy ** 2).mean()), 1e-9)),
                "mse": float((d ** 2).mean()),
                "target_rms": float(np.sqrt((yy ** 2).mean()))}

    args.output.mkdir(parents=True, exist_ok=True)
    save_tree(args.output / f"{args.tag}.npz", head, {
        "mode": mode, "n_params": n_params, "hidden": args.fuse_hidden,
        "dim_in": dim_in, "tiny": Path(args.compensator[0]).stem})
    rep = {"tag": args.tag, "mode": mode, "n_params": n_params,
           "n_train_windows": int(len(tr_idx)), "epochs": args.epochs,
           "train": report(tr_idx), "val": report(val_idx),
           "final_train_mse": hist[-1]["train_mse"], "git_commit": git_sha()}
    (args.output / f"fuse_{args.tag}.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args):
    if not args.input:
        raise SystemExit("--input is required for eval")
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer, config = load(args.model, revision=args.revision,
                                    return_config=True)
    model.eval()
    plan = resolve_plan(len(model.model.layers), args.plan)
    if plan.groups[SITE] != GROUP:
        raise SystemExit(f"plan group {SITE} is {plan.groups[SITE]}, expected {GROUP}")

    # One tiny unit per --compensator, keyed by checkpoint stem. The fusion
    # heads were each trained against one specific tiny, so a single eval run
    # must be able to carry all of them at once.
    tinies: dict[str, TinySerial] = {}
    compensator_meta = {}
    for path in args.compensator:
        tinies[Path(path).stem] = build_from_checkpoint(Path(path))
        compensator_meta[Path(path).stem] = load_tree(Path(path))[1]
    dim_in = next(iter(tinies.values())).dim_in

    fusions = {}
    for path in args.fusion:
        f, fmeta = load_tree(Path(path))
        head = FuseHead(fmeta["dim_in"], fmeta["hidden"], dim_in)
        head.update(flat_to_tree(f, head))
        fusions[Path(path).stem] = {"head": head, "mode": fmeta["mode"],
                                   "tiny": fmeta.get("tiny")}

    rows = []
    for path in args.input:
        rows.extend(json.loads(l) for l in path.open(encoding="utf-8"))
    n = min(len(rows), args.max_samples)
    out_rows, t_tiny = [], 0.0
    for i, rec in enumerate(rows[:n]):
        ids = mx.array(rec["ids"], dtype=mx.int32)[None, :]
        h0 = embed(model, ids, mx)
        mask = create_attention_mask(h0, None)
        ref = model(ids)
        mx.eval(ref)
        states = teacher_slice(model, ids, mask)
        h11_T = states[H11]

        def run(mode, fuse_key=None, tiny_tag=None):
            nonlocal t_tiny
            unit = tinies[tiny_tag] if tiny_tag else None
            h = h0
            for gi, group in enumerate(plan.groups):
                a = h
                if gi == SITE and mode == "oracle":
                    h = h11_T
                    continue
                par = run_group(model, h, group, mask, mx)
                if gi == SITE and mode != "base":
                    t0 = time.perf_counter()
                    z = unit(a, mask)
                    mx.eval(z)
                    if mode == "tiny_sub":
                        c = z - (par - a)
                    else:
                        src = fusions[fuse_key]
                        m = src["mode"]
                        if m == "trunk":
                            x = mx.concatenate([a, par], axis=-1)
                        elif m == "zero_trunk":
                            x = mx.concatenate([a, a, z], axis=-1)
                        else:
                            x = mx.concatenate([a, par, z], axis=-1)
                        c = src["head"](x.astype(mx.float32))
                    mx.eval(c)
                    t_tiny += time.perf_counter() - t0
                    par = par + c.astype(par.dtype)
                    mx.eval(par)
                h = par
            return h

        variants = {}
        base = run("base")
        mx.eval(base)
        variants["irr-a"] = logits_metrics(ref, project_logits(model, base, mx), ids, mx)
        orc = run("oracle")
        mx.eval(orc)
        variants["irr-a+oracle"] = logits_metrics(
            ref, project_logits(model, orc, mx), ids, mx)
        if args.variants_include("tiny_sub"):
            # Which tiny is used for the raw substitution is explicit: the head
            # was trained for one specific unit, and they are NOT interchangeable.
            for tag in tinies:
                h = run("tiny_sub", tiny_tag=tag)
                mx.eval(h)
                variants[f"tiny_sub_{tag}"] = logits_metrics(
                    ref, project_logits(model, h, mx), ids, mx)
        for key in sorted(fusions):
            src = fusions[key]
            h = run("fusion", key, tiny_tag=src["tiny"] or next(iter(tinies)))
            mx.eval(h)
            variants[f"fuse_{key}"] = logits_metrics(
                ref, project_logits(model, h, mx), ids, mx)
        out_rows.append({"sample_id": rec["sample_id"], "variants": variants})
        print(f"[eval {i + 1}/{n}] " + " ".join(
            f"{k}={v['delta_nll']:.3f}" for k, v in variants.items()), flush=True)

    agg = {}
    for v in out_rows[0]["variants"]:
        ds = [r["variants"][v]["delta_nll"] for r in out_rows if v in r["variants"]]
        agg[v] = {
            "n": len(ds), "delta_nll_mean": float(np.mean(ds)),
            "delta_nll_std": float(np.std(ds, ddof=1)),
            "kl_mean": float(np.mean([r["variants"][v]["teacher_to_candidate_kl"]
                                      for r in out_rows if v in r["variants"]])),
            "top1_mean": float(np.mean([r["variants"][v]["top1_agreement"]
                                        for r in out_rows if v in r["variants"]])),
        }
    base_nll = agg["irr-a"]["delta_nll_mean"]
    for v, a in agg.items():
        a["gap_closure"] = (base_nll - a["delta_nll_mean"]) / base_nll
        a["ppl_ratio"] = float(math.exp(a["delta_nll_mean"]))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "stage": "eval", "plan": plan.name, "sites": [SITE], "group": GROUP,
        "input": str(args.input), "n_samples": n, "model": args.model,
        "model_revision": args.revision,
        "compensator": args.compensator, "fusion": args.fusion,
        "compensator_meta": compensator_meta,
        "aggregate": agg,
        "tiny_ms_per_sample": 1000.0 * t_tiny / max(n, 1),
        "git_commit": git_sha(),
        "warning": "Sequential execution. No end-to-end speedup is claimed.",
    }
    (args.output / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(agg, indent=2))


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def cmd_bench(args):
    tiny = build_from_checkpoint(Path(args.compensator[0]))
    T, D = 256, tiny.dim_in
    a = mx.random.normal((1, T, D)).astype(mx.float32)
    mx.eval(a)

    def call():
        mx.eval(tiny(a, "causal"))

    for _ in range(args.warmup):
        call()
    mx.reset_peak_memory()
    ts = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        call()
        ts.append(time.perf_counter() - t0)
    peak = mx.get_peak_memory()
    out = {
        "compensator": str(args.compensator[0]), "kind": tiny.kind,
        "hidden": tiny.hidden, "n_layers": tiny.n_layers,
        "n_params": tiny.n_params(), "tokens": T,
        "tiny_ms": 1000.0 * float(np.median(ts)),
        "tiny_ms_p10": 1000.0 * float(np.percentile(ts, 10)),
        "tiny_ms_p90": 1000.0 * float(np.percentile(ts, 90)),
        "flops_per_token": tiny_flops(tiny, T),
        "peak_memory_mb": peak / 2 ** 20,
        "repeat": args.repeat,
        "note": "one call, 256 tokens, medians after warm-up",
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PINNED_MODEL_REPO)
    ap.add_argument("--revision", default=PINNED_MODEL_REVISION)
    ap.add_argument("--stage", required=True,
                    choices=["collect", "extract", "fit", "fuse", "eval", "bench"])
    ap.add_argument("--input", type=Path, action="append",
                    help="JSONL of windows; repeatable, collect concatenates them")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data", type=Path, help="collect output directory")
    ap.add_argument("--plan", default="irr-a:0|1|2|3-10|11-19|20-25|26-27")
    ap.add_argument("--max-samples", type=int, default=64)

    ap.add_argument("--kind", choices=["tiny", "mlp"], default="tiny")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--dim-in", type=int, default=1024)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--rope-theta", type=float, default=1_000_000.0)
    ap.add_argument("--max-pos", type=int, default=32768)
    ap.add_argument("--teacher-layers", default="3,5,7,9")
    ap.add_argument("--keep-int", type=int, default=512)
    ap.add_argument("--init-from", type=Path, default=None)

    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261019)

    ap.add_argument("--compensator", action="append", default=[])
    ap.add_argument("--fusion", action="append", default=[])
    ap.add_argument("--fusion-mode", choices=["trunk", "tiny", "zero_trunk"],
                    default="tiny")
    ap.add_argument("--fuse-hidden", type=int, default=256)
    ap.add_argument("--with-tiny-sub", action="store_true")
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    args.variants_include = lambda name: (
        name == "tiny_sub" and args.with_tiny_sub)

    if args.revision is None:
        raise SystemExit("refusing to run with an unpinned model revision")
    if args.stage in ("fuse", "eval", "bench") and not args.compensator:
        raise SystemExit("--compensator is required for this stage")
    if args.stage in ("fit", "fuse") and args.data is None:
        raise SystemExit("--data is required")

    {"collect": cmd_collect, "extract": cmd_extract, "fit": cmd_fit,
     "fuse": cmd_fuse, "eval": cmd_eval, "bench": cmd_bench}[args.stage](args)


if __name__ == "__main__":
    main()
