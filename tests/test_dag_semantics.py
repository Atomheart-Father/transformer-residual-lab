"""DAG algebra, spec parsing and merge-rule tests (Stage 2.0).

``DagStructure`` runs on plain Python - no MLX, no model. ``MergeRule`` builds
tiny toy blocks on the MLX CPU stream and checks the executor's bookkeeping
(running state, merge points, serial-continuation shortcut, sparse fallback)
against hand-computed references.
"""

import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dag_exec import (  # noqa: E402
    Dag,
    chain_dag,
    dag_forward,
    group_dag,
    make_dag,
    prefix_group_dag,
    window_group_dag,
)

N = 28


class DagStructure(unittest.TestCase):
    def test_chain_is_the_serial_model(self):
        d = chain_dag(N)
        self.assertTrue(d.is_chain())
        self.assertEqual(d.critical_path, N)
        self.assertEqual(d.executed_blocks, N)
        self.assertEqual(d.n_edges, N * (N - 1) // 2)

    def test_group_depth_is_n_over_g(self):
        self.assertEqual(group_dag(N, 2).critical_path, 14)
        self.assertEqual(group_dag(N, 4).critical_path, 7)
        self.assertEqual(group_dag(N, 8).critical_path, 4)
        self.assertEqual(group_dag(N, 1).critical_path, N)   # g1 == chain
        self.assertEqual(group_dag(N, 1).is_chain(), True)

    def test_group_siblings_share_the_anchor(self):
        d = group_dag(N, 2)
        for k in range(0, N, 2):
            self.assertEqual(d.preds[k], frozenset(range(k)))
            if k + 1 < N:
                self.assertEqual(d.preds[k + 1], frozenset(range(k)))

    def test_prefix_then_group(self):
        d = prefix_group_dag(N, 3, 2)
        self.assertEqual(d.critical_path, 3 + 13)     # 3 serial + 13 levels
        self.assertEqual(d.preds[3], frozenset({0, 1, 2}))
        self.assertEqual(d.preds[4], frozenset({0, 1, 2}))
        self.assertEqual(d.preds[5], frozenset(range(5)))

    def test_window_only_restructures_the_window(self):
        d = window_group_dag(N, 10, 18, 2)
        self.assertEqual(d.critical_path, 10 + 4 + 10)  # 24, not "4 groups"
        for i in range(10):
            self.assertEqual(d.preds[i], frozenset(range(i)), i)
        for i in range(18, N):
            self.assertEqual(d.preds[i], frozenset(range(i)), i)
        self.assertEqual(d.preds[10], frozenset(range(10)))
        self.assertEqual(d.preds[11], frozenset(range(10)))
        self.assertEqual(d.preds[17], frozenset(range(16)))

    def test_sparse_window_spec(self):
        # window 10..17: 14 waits for 10 and 12 (skipping 11); 15 for 11 and 13
        spec = "sparse@10-18:0-9|0-9|10|11|10,12|11,13|14|15"
        d = make_dag(N, spec)
        self.assertEqual(d.preds[14], frozenset({10, 12}))
        self.assertEqual(d.preds[15], frozenset({11, 13}))
        self.assertEqual(d.preds[17], frozenset({15}))
        self.assertEqual(d.preds[9], frozenset(range(9)))
        self.assertEqual(d.preds[18], frozenset(range(18)))
        self.assertEqual(d.critical_path, 10 + 4 + 10)

    def test_spec_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):        # wrong part count
            make_dag(N, "x:0|1|2")
        with self.assertRaises(ValueError):        # wrong window part count
            make_dag(N, "x@10-18:0|1|2")
        with self.assertRaises(ValueError):        # forward predecessor
            make_dag(N, "x:" + "|".join("13" if i == 12 else "" for i in range(N)))
        with self.assertRaises(ValueError):        # self loop
            make_dag(N, "x:" + "|".join("5" if i == 5 else "" for i in range(N)))
        with self.assertRaises(ValueError):        # unknown spec
            make_dag(N, "nonsense")
        with self.assertRaises(ValueError):        # window outside model
            make_dag(N, "x@26-30:0|1|2|3")

    def test_full_spec_round_trip(self):
        spec = ("full:" + "|".join(",".join(str(j) for j in range(i)) for i in range(N)))
        d = make_dag(N, spec)
        self.assertTrue(d.is_chain())
        again = Dag.from_dict(d.to_dict())
        self.assertEqual(again.preds, d.preds)
        self.assertEqual(again.critical_path, d.critical_path)

    def test_window_deeper_than_grouping_is_possible(self):
        # a graph may be *worse* than fixed grouping; the object must allow it
        spec = "worse@10-18:0-9|0-9|10,11|12|13|14|15|16"   # window depth 7
        d = make_dag(N, spec)
        self.assertEqual(d.critical_path, 10 + 7 + 10)      # 27 > g2's 24


class MergeRule(unittest.TestCase):
    """Toy blocks on the MLX CPU stream; compare against numpy references."""

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        cls.mx = mx

    def toy(self, scales):
        """blocks f(x) = x + a * x*x, so deltas are closed-form."""
        layers = []
        for a in scales:
            def f(x, mask=None, cache=None, a=a):
                return x + a * (x * x)
            layers.append(f)
        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        return model

    def reference_g2(self, scales, h0):
        """Independent production-style g2: anchor, fp32 sum, bf16-ish cast."""
        x = h0.astype(np.float64)
        n = len(scales)
        for s in range(0, n, 2):
            group = range(s, min(n, s + 2))
            anchor = x
            ds = [(i, (lambda v, a=scales[i]: v + a * v * v)(anchor) - anchor)
                  for i in group]
            acc = anchor
            for _, d in ds:
                acc = acc + d
            x = acc
        return x

    def test_chain_reduces_to_serial_exactly(self):
        mx = self.mx
        scales = [0.05, -0.03, 0.02, 0.01, -0.04]
        model = self.toy(scales)
        h0 = mx.array(np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 3, 4))
        serial = h0
        for a in scales:
            serial = serial + a * (serial * serial)
        got = dag_forward(model, h0, chain_dag(len(scales)), None, mx)
        mx.eval(got)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(serial))

    def test_g2_mimic_matches_production_semantics(self):
        mx = self.mx
        scales = [0.05, -0.03, 0.02, 0.01, -0.04, 0.06]
        model = self.toy(scales)
        h0 = mx.array(np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 3, 4))
        got = dag_forward(model, h0, group_dag(len(scales), 2), None, mx)
        mx.eval(got)
        want = self.reference_g2(scales, np.asarray(h0.astype(mx.float32)))
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                                   rtol=1e-5, atol=1e-6)

    def test_serial_continuation_and_sparse_fallback(self):
        mx = self.mx
        scales = [0.05, -0.03, 0.02, 0.01]
        model = self.toy(scales)
        h0 = mx.array(np.array([[[0.3, -0.7, 0.5, 0.1]]], dtype=np.float32))
        g = np.asarray(h0.astype(mx.float32))

        def f(v, a):
            return v + a * v * v

        # preds: 0:{}, 1:{}, 2:{0,1}, 3:{1}  (3 is sparse: skips 0 and 2)
        d = make_dag(4, "t:|" + "|0,1|1")
        got = dag_forward(model, h0, d, None, mx)
        mx.eval(got)
        d0 = f(g, 0.05) - g
        d1 = f(g, -0.03) - g
        x2 = g + d0 + d1                       # merge point
        d2 = f(x2, 0.02) - x2
        x3 = g + d1                            # sparse reader: h0 + d1 only
        d3 = f(x3, 0.01) - x3
        want = x2 + d2 + d3                    # final merges every update once
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                                   rtol=1e-5, atol=1e-6)

    def test_middle_edge_drop_is_honored_not_a_noop(self):
        """A block that declines an already-covered middle edge must not read
        a state that still contains it (the serial-continuation shortcut is
        only valid for the full prefix)."""
        mx = self.mx
        scales = [0.05, -0.03, 0.02, 0.01]
        model = self.toy(scales)
        h0 = mx.array(np.array([[[0.3, -0.7, 0.5, 0.1]]], dtype=np.float32))
        g = np.asarray(h0.astype(mx.float32))

        def f(v, a):
            return v + a * v * v

        # 0:{}, 1:{0}, 2:{0,1}, 3:{0,2}  (3 declines edge 1, keeps 0 and 2)
        d = make_dag(4, "t:|0|0,1|0,2")
        for mode in ("materialized", "roadmap"):
            got = dag_forward(model, h0, d, None, mx, merge_mode=mode)
            mx.eval(got)
            out0 = f(g, 0.05)
            d0 = out0 - g
            out1 = f(out0, -0.03)
            d1 = out1 - out0
            x2 = out1                          # merge{0,1}
            d2 = f(x2, 0.02) - x2
            x3 = g + d0 + d2                   # honest: h0 + d0 + d2, no d1
            d3 = f(x3, 0.01) - x3
            want = out1 + d2 + d3              # final merges every update once
            np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                                       rtol=1e-5, atol=1e-6, err_msg=mode)

    def test_roadmap_mode_matches_monolithic_merge(self):
        """roadmap inputs are exactly h0 + sum(P_i) deltas, so the g2 graph
        must agree bit-for-bit with the monolithic fp32 merge reference."""
        mx = self.mx
        scales = [0.05, -0.03, 0.02, 0.01, -0.04, 0.06]
        model = self.toy(scales)
        h0 = mx.array(np.linspace(-1, 1, 12, dtype=np.float32).reshape(1, 3, 4))
        d = group_dag(len(scales), 2)
        got = dag_forward(model, h0, d, None, mx, merge_mode="roadmap")
        mx.eval(got)
        from dag_exec import monolithic_reference
        want = monolithic_reference(model, h0, d, None, mx)
        mx.eval(want)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_final_serial_tail_uses_last_output(self):
        mx = self.mx
        scales = [0.05, -0.03, 0.02]
        model = self.toy(scales)
        h0 = mx.array(np.array([[[0.3, -0.7, 0.5]]], dtype=np.float32))
        # 0,1 grouped; 2 waits for everything -> its input is out_1
        d = make_dag(3, "t:||0,1")
        got = dag_forward(model, h0, d, None, mx)
        mx.eval(got)
        g = np.asarray(h0.astype(mx.float32))
        x1 = g                                  # siblings read h0
        d0 = g + 0.05 * g * g - g
        d1 = g + (-0.03) * g * g - g
        x2 = g + d0 + d1
        want = x2 + 0.02 * x2 * x2
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                                   rtol=1e-5, atol=1e-6)

    def test_delta_override_probes_graph_structure_only(self):
        mx = self.mx
        scales = [0.05, -0.03, 0.02]
        model = self.toy(scales)
        h0 = mx.array(np.array([[[0.3, -0.7, 0.5]]], dtype=np.float32))
        g = np.asarray(h0.astype(mx.float32))
        d = group_dag(3, 2)
        # teacher deltas: exact chain deltas
        t0 = (g + 0.05 * g * g) - g
        t1 = (g + t0 + (-0.03) * (g + t0) ** 2) - (g + t0)
        t2 = (g + t0 + t1 + 0.02 * (g + t0 + t1) ** 2) - (g + t0 + t1)
        got = dag_forward(model, h0, d, None, mx,
                          delta_override={0: mx.array(t0), 1: mx.array(t1),
                                          2: mx.array(t2)})
        mx.eval(got)
        want = g + t0 + t1 + t2
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
