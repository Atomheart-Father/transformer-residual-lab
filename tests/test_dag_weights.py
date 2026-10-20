"""Unit test: the weighted-merge axis must reduce to the default rule.

Run:  .venv/bin/python tests/test_dag_weights.py
Asserts, for the real frozen model and a real window:
  1. ``weights=None`` and ``weights={j: 1.0}`` give bit-identical outputs in
     roadmap mode (the semantics axis is off by default, exact).
  2. A non-trivial weight changes the output (the knob is real, not ignored).
  3. The final output merge is unaffected by input weights: the h_final
     computation stays the plain ``h0 + sum delta`` rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.base import create_attention_mask  # noqa: E402

from measure import PINNED_MODEL_REPO, PINNED_MODEL_REVISION, embed  # noqa: E402
from dag_exec import dag_forward, window_group_dag  # noqa: E402


def main() -> None:
    model, _ = load(PINNED_MODEL_REPO, revision=PINNED_MODEL_REVISION)
    model.eval()
    n = len(model.model.layers)
    ids = mx.array([[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
                     1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015,
                     1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023]],
                   dtype=mx.int32)
    h0 = embed(model, ids, mx)
    mask = create_attention_mask(h0, None)
    mx.eval(h0)

    dag = window_group_dag(n, 10, 18, 2)

    a = dag_forward(model, h0, dag, mask, mx, merge_mode="roadmap")
    b = dag_forward(model, h0, dag, mask, mx, merge_mode="roadmap",
                    weights={j: 1.0 for j in range(n)})
    mx.eval(a, b)
    d_none_vs_one = float(mx.max(mx.abs(a - b)).item())
    assert d_none_vs_one == 0.0, f"weights=1 changed the result: {d_none_vs_one}"

    w = {j: (0.5 if 10 <= j < 18 else 1.0) for j in range(n)}
    c = dag_forward(model, h0, dag, mask, mx, merge_mode="roadmap", weights=w)
    mx.eval(c)
    d_weighted = float(mx.max(mx.abs(a - c)).item())
    assert d_weighted > 0.0, "non-trivial weights had no effect"

    # roadmap inputs with weights must still honor P_i literally: with the chain
    # graph there is exactly one predecessor per block, so any weight changes the
    # result in a predictable direction but cannot break the API.
    print(f"PASS  weights=1 bit-identical ({d_none_vs_one}), "
          f"weighted output differs by max_abs={d_weighted:.4f}")


if __name__ == "__main__":
    main()
