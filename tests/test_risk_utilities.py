"""Hand-computed finite-distribution CVaR and cost/reward convention fixtures."""

import math
import unittest

from risk_aware_v2x.risk import (
    empirical_var, lower_cvar, mean_cvar_cost, mean_cvar_reward,
    select_cost_action, upper_cvar,
)


class RiskUtilityTests(unittest.TestCase):
    def test_integer_tail_and_left_inverse_var(self):
        values = tuple(range(1, 11))
        self.assertEqual(empirical_var(values, alpha=.9), 9)
        self.assertAlmostEqual(upper_cvar(values, alpha=.9), 10)
        self.assertAlmostEqual(lower_cvar(values, alpha=.9), 1)

    def test_fractional_boundary_mass(self):
        self.assertAlmostEqual(upper_cvar([0, 10, 20], alpha=.5), 50 / 3)
        self.assertAlmostEqual(lower_cvar([0, 10, 20], alpha=.5), 10 / 3)
        self.assertAlmostEqual(upper_cvar(range(15), alpha=.9), 41 / 3)

    def test_ties_do_not_expand_tail_probability(self):
        values = [1, 2, 2, 10]
        self.assertEqual(empirical_var(values, alpha=.5), 2)
        self.assertAlmostEqual(upper_cvar(values, alpha=.5), 6)
        self.assertNotEqual(upper_cvar(values, alpha=.5), (2 + 2 + 10) / 3)

    def test_var_decimal_cdf_boundary_is_not_rounded_up_by_multiplication(self):
        self.assertEqual(empirical_var(range(1, 101), alpha=.14), 14)

    def test_zero_confidence_is_mean_and_singleton_is_constant(self):
        for function in (upper_cvar, lower_cvar):
            self.assertAlmostEqual(function([-3, 1, 8], alpha=0), 2)
            self.assertEqual(function([7], alpha=.9), 7)
        self.assertEqual(empirical_var([5, 1], alpha=0), 1)

    def test_near_one_confidence_and_negative_costs(self):
        self.assertEqual(upper_cvar([-10, -2], alpha=math.nextafter(1., 0.)), -2)
        self.assertAlmostEqual(upper_cvar([-10, -2], alpha=.5), -2)

    def test_risk_mixture_changes_choice(self):
        risky = [0.] * 9 + [100.]
        safe = [12.] * 10
        self.assertAlmostEqual(mean_cvar_cost(risky, risk_weight=.5), 55)
        self.assertEqual(select_cost_action([risky, safe], [True, True], risk_weight=0), 0)
        self.assertEqual(select_cost_action([risky, safe], [True, True]), 1)

    def test_reward_sign_equivalence_and_lower_tail(self):
        costs = ([0., 1., 80.], [12., 12., 12.], [-2., 5., 50.])
        for alpha in (0., .5, .9):
            for weight in (0., .5, 1.):
                cost_scores = [mean_cvar_cost(row, alpha=alpha, risk_weight=weight) for row in costs]
                reward_scores = [mean_cvar_reward([-value for value in row], alpha=alpha,
                                                 risk_weight=weight) for row in costs]
                self.assertEqual(cost_scores, [-value for value in reward_scores])
                self.assertEqual(min(range(3), key=cost_scores.__getitem__),
                                 max(range(3), key=reward_scores.__getitem__))

    def test_masked_rows_never_selected_or_evaluated(self):
        self.assertEqual(select_cost_action([[3, 4], [float("nan")], [-1e6]],
                                            [True, False, False]), 0)
        self.assertEqual(select_cost_action([[4, 3], [3, 4]], [True, True]), 0)

    def test_order_invariance_and_input_is_not_mutated(self):
        values = [20, 0, 10]
        self.assertEqual(upper_cvar(values, alpha=.5), upper_cvar(sorted(values), alpha=.5))
        self.assertEqual(values, [20, 0, 10])

    def test_invalid_distribution_and_parameters(self):
        for values in ([], [float("nan")], [float("inf")], [True]):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                upper_cvar(values)
        for alpha in (-.1, 1., float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                upper_cvar([1], alpha=alpha)
        with self.assertRaises(ValueError):
            mean_cvar_cost([1], risk_weight=2)

    def test_invalid_action_shapes_and_empty_mask(self):
        for values, mask in (([[1]], [False]), ([[1]], [1]), ([[1]], []),
                             ([[1], [1, 2]], [True, True]), ([[]], [True])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                select_cost_action(values, mask)
