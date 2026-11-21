"""Hand-computed statistics and strict input checks without learning packages."""

import math
import sys
import unittest
from fractions import Fraction

from experiments.input_statistics import describe_values, feature_statistics


class DescribeValuesTests(unittest.TestCase):
    def test_signed_mean_rms_and_absolute_nearest_rank_are_distinct(self):
        result = describe_values([-3., 0., 4.])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["min"], -3.)
        self.assertEqual(result["max"], 4.)
        self.assertEqual(result["mean"], 1 / 3)
        self.assertAlmostEqual(result["rms"], math.sqrt(25 / 3))
        self.assertEqual(result["max_abs"], 4.)
        self.assertEqual(result["p50_abs"], 3.)
        self.assertEqual(result["p95_abs"], 4.)
        self.assertEqual(result["p99_abs"], 4.)

    def test_exact_nearest_rank_boundaries_without_interpolation(self):
        result = describe_values(range(100))
        self.assertEqual(result["p50_abs"], 49.)
        self.assertEqual(result["p95_abs"], 94.)
        self.assertEqual(result["p99_abs"], 98.)
        self.assertEqual(describe_values([0, 1, 2, 3])["p50_abs"], 1.)
        self.assertEqual(describe_values(range(20))["p95_abs"], 18.)

    def test_empty_singleton_and_zero_data(self):
        empty = describe_values(iter(()))
        self.assertEqual(empty["count"], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != "count"))
        singleton = describe_values([-7.])
        self.assertEqual(singleton["mean"], -7.)
        self.assertEqual(singleton["rms"], 7.)
        for key in ("max_abs", "p50_abs", "p95_abs", "p99_abs"):
            self.assertEqual(singleton[key], 7.)
        zero = describe_values([0., -0., 0])
        self.assertEqual(zero["rms"], 0.)
        self.assertEqual(zero["max_abs"], 0.)

    def test_large_finite_values_do_not_overflow_rms_or_mean(self):
        result = describe_values([1e200, -1e200])
        self.assertEqual(result["mean"], 0.)
        self.assertEqual(result["rms"], 1e200)
        huge = describe_values([1.7e308, 1.7e308])
        self.assertEqual(huge["mean"], 1.7e308)
        self.assertEqual(huge["rms"], 1.7e308)
        self.assertTrue(all(math.isfinite(value) for value in huge.values()))
        # Dividing each largest float by n before fsum can round upward and
        # overflow again. Max rescaling keeps both mean and RMS finite.
        for count in range(2, 20):
            with self.subTest(count=count):
                largest = describe_values([sys.float_info.max] * count)
                self.assertEqual(largest["mean"], sys.float_info.max)
                self.assertEqual(largest["rms"], sys.float_info.max)

    def test_fsum_retains_small_cancellation_residual(self):
        self.assertEqual(describe_values([1e16, 1., -1e16])["mean"], 1 / 3)

    def test_generator_is_consumed_once_and_finite_reals_are_accepted(self):
        seen = []

        def values():
            for value in (Fraction(1, 2), 1, 1.5):
                seen.append(value)
                yield value

        result = describe_values(values())
        self.assertEqual(len(seen), 3)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 1.)

    def test_nonfinite_nonnumeric_boolean_and_overflowing_values_reject(self):
        for value in (math.inf, -math.inf, math.nan, 10**1000):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                describe_values([value])
        for value in (True, False, "1", None, 1j):
            with self.subTest(value=value), self.assertRaises(TypeError):
                describe_values([value])
        for values in ("", "12", b"", {}, {1: 2}, None, 1):
            with self.subTest(values=values), self.assertRaises(TypeError):
                describe_values(values)


class FeatureStatisticsTests(unittest.TestCase):
    def test_column_order_and_hand_computed_results(self):
        observations = [[-3., 1.], [0., 3.], [4., 5.]]
        names = ["signed", "positive"]
        result = feature_statistics(observations, names)
        self.assertEqual([item["name"] for item in result], names)
        self.assertEqual(result[0]["mean"], 1 / 3)
        self.assertAlmostEqual(result[1]["rms"], math.sqrt(35 / 3))
        self.assertEqual(result[1]["p50_abs"], 3.)
        self.assertEqual(observations, [[-3., 1.], [0., 3.], [4., 5.]])
        self.assertEqual(names, ["signed", "positive"])

    def test_empty_data_preserve_named_columns(self):
        result = feature_statistics([], ("a", "b"))
        self.assertEqual([item["name"] for item in result], ["a", "b"])
        for item in result:
            self.assertEqual(item["count"], 0)
            self.assertTrue(all(value is None for key, value in item.items() if key not in ("name", "count")))

    def test_outer_generator_is_consumed_once(self):
        seen = []

        def rows():
            for row in ((1., 2.), (3., 4.)):
                seen.append(row)
                yield row

        result = feature_statistics(rows(), ("a", "b"))
        self.assertEqual(len(seen), 2)
        self.assertEqual([item["mean"] for item in result], [2., 3.])

    def test_invalid_names_reject(self):
        for names in ([], (), [""], [" "], ["a", "a"], ["a", 1]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                feature_statistics([], names)
        for names in ("ab", None, (name for name in ("a", "b"))):
            with self.subTest(names_type=type(names).__name__), self.assertRaises(TypeError):
                feature_statistics([], names)

    def test_ragged_nonsequence_boolean_and_nonfinite_rows_reject(self):
        for rows in ([[1.]], [[1., 2., 3.]], [[1., 2.], [3.]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                feature_statistics(rows, ("a", "b"))
        for rows in (("12",), ((value for value in (1., 2.)),), (None,), [[1., True]]):
            with self.subTest(rows_type=type(rows).__name__), self.assertRaises(TypeError):
                feature_statistics(rows, ("a", "b"))
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                feature_statistics([[1., value]], ("a", "b"))
        for rows in ("", {}, None, 1):
            with self.subTest(rows=rows), self.assertRaises(TypeError):
                feature_statistics(rows, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
