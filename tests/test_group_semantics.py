"""Pure-algebra and plan-construction checks. No GPU, no model, no MLX needed."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from measure import (  # noqa: E402
    decompose_error,
    even_keep,
    make_plan,
    resolve_plan,
    stage15_plans,
    stage16_plans,
    stage1_plans,
)


def block_factory(factor):
    return lambda x: x + factor * x * x


def teacher(blocks, x):
    for f in blocks:
        x = f(x)
    return x


def grouping(blocks, x, size):
    for i in range(0, len(blocks), size):
        group = blocks[i:i + size]
        if len(group) == 1:
            x = group[0](x)
        else:
            a = x
            x = a + sum(f(a) - a for f in group)
    return x


class Semantics(unittest.TestCase):
    def test_singleton_groups_identical_to_teacher(self):
        blocks = [block_factory(f) for f in (0.05, -0.02, 0.03, 0.01)]
        self.assertAlmostEqual(teacher(blocks, 0.7), grouping(blocks, 0.7, 1), places=12)

    def test_parallel_update_is_not_sum_of_full_states(self):
        blocks = [block_factory(0.05), block_factory(0.08)]
        a = 0.3
        correct = a + (blocks[0](a) - a) + (blocks[1](a) - a)
        wrong = blocks[0](a) + blocks[1](a)
        self.assertAlmostEqual(correct, grouping(blocks, a, 2), places=12)
        self.assertGreater(abs(correct - wrong), 0.1)

    def test_parallel_differs_from_nonlinear_serial(self):
        blocks = [block_factory(0.8), block_factory(0.7)]
        self.assertGreater(abs(grouping(blocks, 0.4, 2) - teacher(blocks, 0.4)), 0.001)


class Plans(unittest.TestCase):
    n = 28

    def test_group_plan_covers_every_layer_once(self):
        for g in (2, 4, 8, 28):
            plan = make_plan(self.n, f"group{g}")
            flat = [i for grp in plan.groups for i in grp]
            self.assertEqual(flat, list(range(self.n)))
            self.assertEqual(plan.executed_blocks, self.n)
            self.assertEqual(plan.effective_depth, len(plan.groups))
            self.assertEqual(plan.effective_depth, -(-self.n // g))

    def test_group1_reduces_to_serial(self):
        a, b = make_plan(self.n, "group1"), make_plan(self.n, "serial")
        self.assertEqual(a.groups, b.groups)
        self.assertEqual(a.effective_depth, b.effective_depth)

    def test_window_plan_only_touches_its_window(self):
        plan = make_plan(self.n, "window-mid-g4")
        touched = [grp for grp in plan.groups if len(grp) > 1]
        self.assertEqual(touched, [[10, 11, 12, 13], [14, 15, 16, 17]])
        flat = [i for grp in plan.groups for i in grp]
        self.assertEqual(flat, list(range(self.n)))
        self.assertEqual(plan.executed_blocks, self.n)

    def test_delete_plans_are_even_and_reproducible(self):
        for k in (7, 14, 21):
            plan = make_plan(self.n, f"delete-keep{k}")
            kept = [grp[0] for grp in plan.groups]
            self.assertEqual(len(kept), k)
            self.assertEqual(kept, even_keep(self.n, k))
            self.assertEqual(plan.executed_blocks, k)
            self.assertEqual(plan.effective_depth, k)
            self.assertEqual(len(plan.deleted_layers), self.n - k)
            self.assertFalse(set(kept) & set(plan.deleted_layers))
            self.assertEqual(kept[0], 0)
            self.assertEqual(kept[-1], self.n - 1)

    def test_stage1_plans_include_required_controls(self):
        names = stage1_plans(self.n)
        for required in ("serial", "group2", "group4", "group8", "group28",
                         "delete-keep7", "delete-keep14", "delete-keep21",
                         "window-early-g4", "window-mid-g4", "window-late-g4"):
            self.assertIn(required, names)

    def test_stage15_has_a_deletion_baseline_per_depth(self):
        plans = stage15_plans(self.n)
        depths = {p.effective_depth for p in plans if p.kind == "parallel"}
        delete_depths = {p.effective_depth for p in plans if p.kind == "delete"}
        self.assertTrue(depths)
        self.assertTrue(depths <= delete_depths,
                        f"missing deletion baselines at {sorted(depths - delete_depths)}")

    def test_first_parallel_group_skips_serial_prefix(self):
        plan = make_plan(self.n, "prefix3-g8")
        first = next(i for i, g in enumerate(plan.groups) if len(g) > 1)
        self.assertEqual(plan.groups[first][0], 3)


class IrregularGrouping(unittest.TestCase):
    """Stage 1.6: the plan must be a partition, so block count can never drift."""

    n = 28

    def test_custom_spec_is_a_partition(self):
        plan = make_plan(self.n, "x:0|1|2|3-10|11-18|19-25|26-27")
        self.assertEqual(plan.groups,
                         [[0], [1], [2], list(range(3, 11)), list(range(11, 19)),
                          list(range(19, 26)), [26, 27]])
        self.assertEqual(plan.executed_blocks, self.n)
        self.assertEqual(plan.effective_depth, 7)

    def test_gaps_and_repeats_are_rejected(self):
        with self.assertRaises(ValueError):
            make_plan(self.n, "x:0|1|2|4-27")          # layer 3 missing
        with self.assertRaises(ValueError):
            make_plan(self.n, "x:0|1|2|3-27|27")       # 27 twice
        with self.assertRaises(ValueError):
            make_plan(self.n, "x:0|1|2|3-20")          # stops early

    def test_stage16_tail_family_is_depth_matched(self):
        plans = {p.name: p for p in stage16_plans(self.n)}
        for name in ("tail-b27", "tail-b26", "tail-b25", "tail-b24", "tail-b23"):
            self.assertEqual(plans[name].effective_depth, 7, name)
            self.assertEqual(plans[name].executed_blocks, self.n, name)
        # only the last two stages differ inside the family
        base = plans["tail-b27"].groups
        for name in ("tail-b26", "tail-b25", "tail-b24", "tail-b23"):
            self.assertEqual(plans[name].groups[:-2], base[:-2], name)

    def test_stage16_serial_marker_is_explicit(self):
        plans = {p.name: p for p in stage16_plans(self.n)}
        self.assertEqual(plans["tail-2serial"].effective_depth, 8)
        self.assertEqual(plans["d10-b23"].effective_depth, 10)

    def test_resolve_plan_accepts_shorthand_and_spec(self):
        self.assertEqual(resolve_plan(self.n, "tail-b26").effective_depth, 7)
        self.assertEqual(resolve_plan(self.n, "prefix3-g8").groups,
                         resolve_plan(self.n, "tail-b27").groups)


class Decomposition(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.t_s = rng.normal(size=(2, 5, 3))
        self.c_s = self.t_s + rng.normal(scale=0.1, size=self.t_s.shape)
        self.t_e = rng.normal(size=self.t_s.shape)
        self.gp_cs = self.c_s + rng.normal(scale=0.2, size=self.t_s.shape)
        self.gp_ts = self.t_s + rng.normal(scale=0.2, size=self.t_s.shape)

    def test_split_is_exact(self):
        inh, amp, new, total = decompose_error(
            self.c_s, self.t_s, self.gp_cs, self.gp_ts, self.t_e)
        np.testing.assert_allclose(total, inh + amp + new, atol=1e-12)

    def test_correct_entry_has_no_inherited_term(self):
        inh, amp, new, total = decompose_error(
            self.t_s, self.t_s, self.gp_ts, self.gp_ts, self.t_e)
        self.assertLess(np.abs(inh).max(), 1e-12)
        self.assertLess(np.abs(amp).max(), 1e-12)
        np.testing.assert_allclose(total, new, atol=1e-12)

    def test_amplification_is_group_response_minus_input(self):
        _, amp, _, _ = decompose_error(
            self.c_s, self.t_s, self.gp_cs, self.gp_ts, self.t_e)
        expected = (self.gp_cs - self.gp_ts) - (self.c_s - self.t_s)
        np.testing.assert_allclose(amp, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
