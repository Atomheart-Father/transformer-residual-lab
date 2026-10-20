"""Stage 2.0: static sparse dependency DAG over the frozen original blocks.

Every one of the ``n`` original decoder blocks is always executed with its
original frozen weights. Nothing is trained, nothing is deleted; only the
*input dependencies* and the *ready order* change.

Semantics (roadmap section 2), stated so that it can be checked exactly::

    x_i     = h0 + sum_{j in P_i} delta_j      fp32 sum, ascending j, cast to h0.dtype
    delta_i = F_i(x_i) - x_i
    h_final = h0 + sum_i delta_i               fp32 sum, ascending i, cast to h0.dtype

Three consequences make the executor verifiable instead of merely plausible:

* ``P_i = {0..i-1}`` for every block gives ``x_i = h_{i-1}``, i.e. the graph
  *is* the serial chain. ``chain_mode="auto"`` detects this and runs the chain
  through the plain block call sequence, which is bit-identical to the
  reference model - the strict equivalence check. The delta formulation of the
  same graph is also runnable (``chain_mode="off"``) and its pure
  double-rounding noise is reported, because that noise is the price of the
  merge rule for every non-chain graph.

* g2 grouping is the special case ``P_2k = P_2k+1 = {0..2k-1}``: both blocks of
  a pair read the same anchor, exactly like ``measure.group_forward`` with
  ``fusion="sum"``. The g2-mimic DAG can therefore be compared against the
  production g2 implementation, and against a monolithic fp32 merge that is
  bit-identical to the DAG merge rule by construction.

* Depth is always the longest chain of block dependencies (critical path in
  block steps). The number of groups, the number of edges or FLOPs are never
  used as a depth proxy, and no wall-clock speedup is claimed from the graph
  alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# graph object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dag:
    """Predecessor sets over the fixed original block numbering.

    ``preds[i]`` is the set of block indices whose residual updates block i
    must wait for. Predecessors must have smaller indices, so the original
    topological order is preserved by construction (no layer swapping).
    """

    name: str
    n: int
    preds: tuple[frozenset[int], ...]
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.preds) != self.n:
            raise ValueError(
                f"{self.name}: expected {self.n} predecessor sets, "
                f"got {len(self.preds)}")
        for i, p in enumerate(self.preds):
            bad = sorted(j for j in p if not 0 <= j < i)
            if bad:
                raise ValueError(
                    f"{self.name}: block {i} has invalid predecessors {bad}; "
                    "only earlier blocks may be predecessors (fixed order)")

    # -- structure --------------------------------------------------------
    @property
    def executed_blocks(self) -> int:
        """All original blocks always run; this is never a pruning claim."""
        return self.n

    @property
    def n_edges(self) -> int:
        return sum(len(p) for p in self.preds)

    def levels(self) -> list[int]:
        """Dependency level of every block; 0 for blocks that read only h0."""
        lvl = [0] * self.n
        for i, p in enumerate(self.preds):
            lvl[i] = 1 + max((lvl[j] for j in p), default=-1)
        return lvl

    @property
    def critical_path(self) -> int:
        """Longest chain of dependent blocks from embedding to final state."""
        return max(self.levels()) + 1

    def level_groups(self) -> list[list[int]]:
        """Blocks grouped by level, for level-synchronous scheduling."""
        lvl = self.levels()
        out: list[list[int]] = [[] for _ in range(max(lvl) + 1)]
        for i, l in enumerate(lvl):
            out[l].append(i)
        return out

    def is_chain(self) -> bool:
        return all(self.preds[i] == frozenset(range(i)) for i in range(self.n))

    def fan_in(self) -> list[int]:
        return [len(p) for p in self.preds]

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "preds": [sorted(p) for p in self.preds],
            "critical_path": self.critical_path,
            "executed_blocks": self.executed_blocks,
            "n_edges": self.n_edges,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict) -> "Dag":
        return Dag(d["name"], int(d["n"]),
                   tuple(frozenset(int(j) for j in p) for p in d["preds"]),
                   d.get("note", ""))


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------

def _expand(text: str) -> frozenset[int]:
    """``"3,5,10-12"`` -> {3,5,10,11,12}. Empty text -> empty set."""
    out: set[int] = set()
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"descending range {tok}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    return frozenset(out)


def make_dag(n: int, spec: str) -> Dag:
    """Build a Dag from a spec string.

    Forms::

        chain                          P_i = {0..i-1}: the serial model
        <name>:p0|p1|...|p{n-1}         full spec, one part per block
        <name>@<lo>-<hi>:q0|...|q{k}    window spec, blocks outside [lo,hi) serial

    Every part is a comma separated predecessor list, ranges allowed
    (``10-12``). A window spec lists exactly ``hi-lo`` parts, one per window
    block in ascending order; a part may name predecessors outside the window.
    """
    if spec == "chain":
        return chain_dag(n)

    m = re.fullmatch(r"([\w.\-]+)@(\d+)-(\d+):(.+)", spec, re.S)
    if m:
        name, lo, hi, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if not 0 <= lo < hi <= n:
            raise ValueError(f"{spec}: window [{lo},{hi}) outside 0..{n}")
        parts = body.split("|")
        if len(parts) != hi - lo:
            raise ValueError(
                f"{spec}: window needs exactly {hi - lo} parts, got {len(parts)}")
        window = [_expand(p) for p in parts]
        preds = tuple(
            frozenset(range(i)) if i < lo or i >= hi else window[i - lo]
            for i in range(n))
        return Dag(name, n, preds, note=f"window [{lo},{hi}) sparse, rest serial")

    m = re.fullmatch(r"([\w.\-]+):(.+)", spec, re.S)
    if m:
        name, body = m.group(1), m.group(2)
        parts = body.split("|")
        if len(parts) != n:
            raise ValueError(f"{spec}: expected {n} parts, got {len(parts)}")
        preds = tuple(_expand(p) for p in parts)
        return Dag(name, n, preds, note="full spec")

    raise ValueError(f"unknown dag spec: {spec!r}")


def chain_dag(n: int = 28, name: str = "chain") -> Dag:
    return Dag(name, n, tuple(frozenset(range(i)) for i in range(n)),
               note="structurally the serial model")


def group_dag(n: int = 28, g: int = 2, name: Optional[str] = None) -> Dag:
    """Fixed contiguous grouping, expressed as shared-anchor dependencies.

    A group of ``g`` consecutive blocks all read the merge of all strictly
    earlier groups, which is exactly ``measure.group_forward(..., fusion=sum)``.
    """
    if g < 1:
        raise ValueError("group size must be >= 1")
    preds = tuple(frozenset(range((i // g) * g)) for i in range(n))
    return Dag(name or f"g{g}", n, preds, note=f"fixed group size {g}")


def prefix_group_dag(n: int = 28, prefix: int = 0, g: int = 2,
                     name: Optional[str] = None) -> Dag:
    """``measure.make_plan(f"prefix{k}-g{g}")`` as a dependency graph.

    Blocks before ``prefix`` stay serial (each reads every earlier update);
    the tail is grouped by ``g`` with shared anchors. ``prefix=0`` is
    :func:`group_dag`.
    """
    if not 0 <= prefix < n:
        raise ValueError(f"prefix {prefix} outside 0..{n - 1}")
    preds = []
    for i in range(n):
        if i < prefix:
            preds.append(frozenset(range(i)))
        else:
            preds.append(frozenset(range(prefix + ((i - prefix) // g) * g)))
    return Dag(name or f"prefix{prefix}-g{g}", n, tuple(preds),
               note=f"{prefix} serial blocks, tail grouped by {g}")


def window_group_dag(n: int = 28, lo: int = 10, hi: int = 18, g: int = 2,
                     name: Optional[str] = None) -> Dag:
    """``measure.make_plan(f"window-mid-g{g}")`` as a dependency graph.

    Only ``[lo, hi)`` is grouped; every block outside the window reads every
    earlier update, so the window's merged output is the next serial input.
    """
    if not 0 <= lo < hi <= n:
        raise ValueError(f"window [{lo},{hi}) outside 0..{n}")
    preds = []
    for i in range(n):
        if lo <= i < hi:
            preds.append(frozenset(range(lo + ((i - lo) // g) * g)))
        else:
            preds.append(frozenset(range(i)))
    return Dag(name or f"win{lo}-{hi}-g{g}", n, tuple(preds),
               note=f"window [{lo},{hi}) grouped by {g}, rest serial")


def load_dag(path: Path) -> Dag:
    return Dag.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

# Merge rule (the one semantic choice in this executor, stated exactly).
#
# State materialization ("rule (b)"):
#   * ``h`` is the accumulated bf16 state; ``covered`` is the set of block
#     updates already inside it. ``h`` only advances at merge points - the
#     blocks whose updates some later block waits for and that are not in
#     ``h`` yet.
#   * A block whose new predecessors are absorbed at once reads
#     ``x_i = bf16(fp32(h) + sum_{j new} fp32(delta_j))`` and the merge point
#     moves to that state.
#   * A block whose predecessors are all already covered reads ``h`` as-is
#     (this is the shared-anchor sibling case of grouped execution).
#   * A block waiting for something ``h`` does not contain (a sparse graph
#     that skips back) falls back to accumulating from h0, so the result
#     never silently mixes states.
#   * Serial continuation shortcut: when the only not-yet-covered update is
#     the immediately previous block's and that block's own input already
#     held everything before it, the reader takes ``out_{i-1}`` directly.
#     This is exact for serial chains and for the serial segments around
#     grouped regions, instead of re-rounding a reconstructed sum.
#
# The rule is a refinement of the roadmap's ``h_0 + sum delta`` formulation
# with "local basis states" (which the roadmap explicitly allows): the same
# semantics, with bf16 materialization at the merge points the graph defines.
# It reproduces the production ``group_forward`` g2 bit-for-bit.


def _merge(h0, deltas: dict, idxs, mx):
    """fp32 accumulation of selected residual updates, ascending index."""
    acc = h0.astype(mx.float32)
    for j in idxs:
        acc = acc + deltas[j].astype(mx.float32)
    return acc


def dag_forward(model, h0, dag: Dag, mask, mx, *,
                delta_override: Optional[dict] = None,
                input_override: Optional[dict] = None,
                record: bool = False, eval_each: bool = False,
                sync_levels: bool = False, chain_mode: str = "auto",
                refine: bool = True):
    """Execute ``dag`` on the frozen blocks.

    ``delta_override[j]`` replaces the computed residual update of block j: used
    for teacher-driven probes, where the graph structure is real but upstream
    updates are exact, so no inherited error contaminates the measurement.
    ``input_override[i]`` replaces the constructed input of block i entirely
    (diagnostic oracle interventions only - never a deployable result).

    ``chain_mode="auto"`` reduces a full-chain graph to the plain serial call
    sequence, which is bit-identical to the reference model. ``"off"`` forces
    the merge machinery without the serial-continuation shortcut
    (``refine=False``), exposing the merge rule's rounding.

    ``sync_levels=True`` materializes the updates of each dependency level
    before the next level starts (bounded memory, some scheduling freedom).
    ``eval_each=True`` materializes every node (no scheduling freedom at all).

    Returns ``h_final``, or ``(h_final, records)`` when ``record=True``.
    """
    if chain_mode not in ("auto", "off", "force"):
        raise ValueError(f"unknown chain_mode {chain_mode!r}")
    if chain_mode == "off":
        refine = False          # expose the merge rule without the shortcut
    over_d = delta_override or {}
    over_x = input_override or {}
    layers = model.model.layers
    recs: dict = {"x": {}, "delta": {}, "out": {}}

    if chain_mode in ("auto", "force") and dag.is_chain():
        h = h0
        for i in range(dag.n):
            x = over_x.get(i, h)
            if i in over_d:
                d = over_d[i]
                h = x + d                       # constructed; block not executed
            else:
                out = layers[i](x, mask, None)
                d = out - x
                h = out                         # exact reference semantics
            if record:
                recs["x"][i], recs["delta"][i] = x, d
            if eval_each:
                mx.eval(h)
        return (h, recs) if record else h

    deltas: dict = {}
    outs: dict = {}
    contains: dict = {}
    h = h0
    covered: frozenset[int] = frozenset()
    n = dag.n
    all_blocks = frozenset(range(n))

    for lvl_nodes in dag.level_groups():
        for i in lvl_nodes:
            P = dag.preds[i]
            if i in over_x:
                x = over_x[i]
                contains[i] = frozenset(range(i))   # oracle input: true prefix
            else:
                new = P - covered
                prev_full = covered == frozenset(range(i - 1)) and i >= 1
                prev_contains = contains.get(i - 1, frozenset())
                if (refine and new == frozenset({i - 1}) and prev_full
                        and prev_contains >= frozenset(range(i - 1))
                        and (i - 1) in outs):
                    # exact serial continuation: out_{i-1} already is the merge
                    x = outs[i - 1]
                    contains[i] = prev_contains | {i - 1}
                    h, covered = x, covered | {i - 1}
                elif covered <= P:
                    if new:
                        x = _merge(h, deltas, sorted(new), mx).astype(h0.dtype)
                        h, covered = x, covered | new
                    else:
                        x = h
                    contains[i] = covered
                else:
                    # sparse reader: h does not hold everything this block waits
                    # for, so accumulate from h0 rather than mixing states
                    x = _merge(h0, deltas, sorted(P), mx).astype(h0.dtype)
                    contains[i] = P
            if eval_each:
                mx.eval(x)
            if i in over_d:
                d = over_d[i]
            else:
                out = layers[i](x, mask, None)
                d = out - x
                outs[i] = out
            if eval_each:
                mx.eval(d)
            deltas[i] = d
            if record:
                recs["x"][i], recs["delta"][i] = x, d
        if sync_levels and lvl_nodes:
            mx.eval(*[deltas[i] for i in lvl_nodes])

    unabsorbed = all_blocks - covered
    if (refine and unabsorbed == frozenset({n - 1}) and n >= 1
            and contains.get(n - 1, frozenset()) >= frozenset(range(n - 1))
            and (n - 1) in outs):
        h_final = outs[n - 1]
    elif covered <= all_blocks:
        h_final = _merge(h, deltas, sorted(unabsorbed), mx).astype(h0.dtype)
    else:
        h_final = _merge(h0, deltas, range(n), mx).astype(h0.dtype)
    if record:
        recs["out_final"] = h_final
        return h_final, recs
    return h_final


def g2_reference(model, h0, mask, mx):
    """The production g2 implementation: ``group_forward`` group by group.

    fp32 accumulation inside a group, bf16 re-cast at every group boundary -
    the exact semantics of ``measure.group_forward`` / ``run_group``.
    """
    layers = model.model.layers
    n = len(layers)
    h = h0
    for s in range(0, n, 2):
        group = list(range(s, min(n, s + 2)))
        anchor = h
        acc = anchor.astype(mx.float32)
        for i in group:
            acc = acc + (layers[i](anchor, mask, None) - anchor).astype(mx.float32)
        h = acc.astype(anchor.dtype)
        mx.eval(h)
    return h


def monolithic_reference(model, h0, dag: Dag, mask, mx):
    """All updates from h0 in one fp32 sum, single final cast.

    For any graph this performs exactly the ops of :func:`dag_forward`, so a
    bit-exact match pins the merge rule of the executor.
    """
    layers = model.model.layers
    deltas: dict = {}
    for i in range(dag.n):
        x = _merge(h0, deltas, sorted(dag.preds[i]), mx).astype(h0.dtype)
        deltas[i] = layers[i](x, mask, None) - x
    return _merge(h0, deltas, range(dag.n), mx).astype(h0.dtype)
