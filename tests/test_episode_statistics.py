"""Hand-computed episode risk and paired, not task-wise, bootstrap fixtures."""

import math
import random
import unittest
from random import Random

from risk_aware_v2x.risk.statistics import episode_cost_summary, paired_mean_interval


class EpisodeStatisticsTests(unittest.TestCase):
    def test_full_episode_cost_risk_has_correct_tail_mass(self):
        result = episode_cost_summary(range(1, 11))
        self.assertEqual(result["mean_episode_total_cost"], 5.5)
        self.assertEqual(result["var_episode_total_cost"], 9)
        self.assertAlmostEqual(result["cvar_episode_total_cost"], 10.)
        self.assertAlmostEqual(result["nominal_tail_mass_in_episodes"], 1.)
        self.assertTrue(result["tail_estimation_warning"])

    def test_shared_pair_resampling_preserves_a_constant_difference(self):
        result = paired_mean_interval({"a": 10., "b": 20.}, {"b": 18., "a": 8.}, rng=Random(4), resamples=100)
        self.assertEqual(result.mean_cost_difference_a_minus_b, 2.)
        self.assertEqual(result.lower, 2.)
        self.assertEqual(result.upper, 2.)
        zero = paired_mean_interval({"a": 2., "b": 5.}, {"b": 5., "a": 2.}, rng=Random(4), resamples=10)
        self.assertEqual(zero.lower, 0.)
        self.assertEqual(zero.upper, 0.)

    def test_seeded_bootstrap_is_key_order_invariant_and_global_rng_independent(self):
        before = random.getstate()
        one = paired_mean_interval({"a": 0., "b": 4.}, {"a": 0., "b": 0.}, rng=Random(8), resamples=100)
        two = paired_mean_interval({"b": 4., "a": 0.}, {"b": 0., "a": 0.}, rng=Random(8), resamples=100)
        self.assertEqual(one, two)
        self.assertEqual(one.mean_cost_difference_a_minus_b, 2.)
        self.assertEqual(one.lower, 0.)
        self.assertEqual(one.upper, 4.)
        self.assertEqual(before, random.getstate())

    def test_near_one_confidence_does_not_pass_rounded_one_into_cvar_api(self):
        result = paired_mean_interval({"a": 1., "b": 3.}, {"a": 0., "b": 0.}, rng=Random(1),
                                      resamples=10, confidence=math.nextafter(1., 0.))
        self.assertLessEqual(result.lower, result.upper)

    def test_invalid_samples_keys_randomness_and_confidence(self):
        for values in ([], [float("nan")], [True]):
            with self.assertRaises((TypeError, ValueError)):
                episode_cost_summary(values)
        for a, b in (({"a": 1}, {"a": 1}), ({"a": 1, "b": 2}, {"a": 1, "c": 2}),
                     ({"a": 1, "b": float("inf")}, {"a": 1, "b": 2})):
            with self.assertRaises(ValueError):
                paired_mean_interval(a, b, rng=Random(1))
        for extra in ({"confidence": 1.}, {"confidence": 0.}, {"resamples": 0}, {"rng": None}):
            with self.assertRaises((TypeError, ValueError)):
                paired_mean_interval({"a": 1., "b": 2.}, {"a": 1., "b": 2.}, **{"rng": Random(1), **extra})
