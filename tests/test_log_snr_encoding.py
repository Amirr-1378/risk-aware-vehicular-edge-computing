"""Fixed log-SINR formula and exact legacy/physical representation boundaries."""

import math
import sys
import unittest
from dataclasses import asdict, replace

from risk_aware_v2x.agents import (
    FeatureScales, LogSNRObservationEncoder, ObservationEncoder, normalized_log_snr,
)
from risk_aware_v2x.agents.encoding import (
    ENCODED_OBSERVATION_SCHEMA_VERSION, LOG_SNR_ENCODED_OBSERVATION_SCHEMA_VERSION,
)
from test_slotted_environment import make_env, to_decision


class NormalizedLogSNRTests(unittest.TestCase):
    def test_zero_reference_and_hand_computed_exponents(self):
        self.assertEqual(normalized_log_snr(0., 100.), 0.)
        self.assertEqual(normalized_log_snr(100., 100.), 1.)
        # For reference9, log(1+99)/log(1+9)=log100/log10=2.
        self.assertAlmostEqual(normalized_log_snr(99., 9.), 2.)
        self.assertAlmostEqual(normalized_log_snr(9999., 9.), 4.)
        self.assertAlmostEqual(normalized_log_snr(9999., 99.), 2.)

    def test_monotone_tiny_and_large_inputs_without_clipping(self):
        inputs = (0., 1e-300, 1e-15, 1e-6, 1., 100., 1e8, sys.float_info.max)
        values = tuple(normalized_log_snr(value, 100.) for value in inputs)
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(values[-1], 1.)
        self.assertAlmostEqual(values[1] / 1e-300, 1. / math.log(101.))

    def test_tiny_reference_preserves_identity_and_rejects_result_overflow(self):
        tiny = math.nextafter(0., 1.)
        self.assertEqual(normalized_log_snr(tiny, tiny), 1.)
        self.assertEqual(normalized_log_snr(0., tiny), 0.)
        self.assertTrue(math.isfinite(normalized_log_snr(1., 1e-300)))
        with self.assertRaises(ValueError):
            normalized_log_snr(sys.float_info.max, tiny)

    def test_invalid_input_and_reference_are_rejected(self):
        for value in (-1., math.inf, -math.inf, math.nan, 10**1000):
            with self.subTest(input_type=type(value).__name__), self.assertRaises(ValueError):
                normalized_log_snr(value, 100.)
        for value in (0., -1., math.inf, math.nan, 10**1000):
            with self.subTest(reference_type=type(value).__name__), self.assertRaises(ValueError):
                normalized_log_snr(1., value)
        for value in (True, False, None, "100", 1j):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalized_log_snr(value, 100.)
            with self.subTest(reference=value), self.assertRaises(TypeError):
                normalized_log_snr(1., value)


class LogSNRObservationEncodingTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_env()
        self.environment.reset(seed=4)
        to_decision(self.environment)
        self.raw = self.environment.observations[1]
        self.scales = FeatureScales()
        self.legacy = ObservationEncoder(max_server_vehicles=3, scales=self.scales)
        self.logarithmic = LogSNRObservationEncoder(max_server_vehicles=3, scales=self.scales)

    def test_distinct_schema_names_and_same_dimension(self):
        self.assertEqual(ENCODED_OBSERVATION_SCHEMA_VERSION, 3)
        self.assertEqual(LOG_SNR_ENCODED_OBSERVATION_SCHEMA_VERSION, 4)
        self.assertEqual(self.legacy.encode(self.raw).schema_version, 3)
        self.assertEqual(self.logarithmic.encode(self.raw).schema_version, 4)
        self.assertEqual(self.legacy.dimension, 36 + 15 * 3)
        self.assertEqual(self.legacy.dimension, self.logarithmic.dimension)
        differences = [(old, new) for old, new in zip(self.legacy.feature_names, self.logarithmic.feature_names)
                       if old != new]
        self.assertEqual(differences, [(f"server[{index}].snr", f"server[{index}].snr_log1p")
                                      for index in range(3)])

    def test_only_snr_values_change_and_original_snapshot_remains_exact(self):
        before = asdict(self.raw)
        old = self.legacy.encode(self.raw)
        new = self.logarithmic.encode(self.raw)
        for index, (name, old_value, new_value) in enumerate(zip(
                self.legacy.feature_names, old.values, new.values, strict=True)):
            if name.endswith(".snr"):
                slot = (index - 36) // 15
                server = self.raw.servers[slot]
                expected = normalized_log_snr(server.record.estimated_snr_linear, self.scales.snr_linear) if server else 0.
                self.assertEqual(new_value, expected)
                expected_old = server.record.estimated_snr_linear / self.scales.snr_linear if server else 0.
                self.assertEqual(old_value, expected_old)
            else:
                self.assertEqual(old_value, new_value)
        self.assertEqual(new.action_mask, old.action_mask)
        self.assertEqual((new.user_vehicle_id, new.task_id), (old.user_vehicle_id, old.task_id))
        self.assertEqual(before, asdict(self.raw))
        self.assertEqual(self.legacy.encode(self.raw), old)

    def test_absent_blocks_zero_and_present_masked_information_retained(self):
        encoded = self.logarithmic.encode(self.raw)
        self.assertEqual(encoded.values[-30:], (0.,) * 30)
        masked_raw = replace(self.raw, action_mask=(True, False, False, False))
        masked = self.logarithmic.encode(masked_raw)
        self.assertEqual(masked.values, encoded.values)
        self.assertEqual(masked.action_mask, (True, False, False, False))
        features = dict(zip(self.logarithmic.feature_names, masked.values, strict=True))
        self.assertEqual(features["server[0].present"], 1.)
        self.assertGreater(features["server[0].snr_log1p"], 0.)

    def test_fixed_custom_reference_changes_only_transformed_snr(self):
        custom = FeatureScales(snr_linear=9.)
        encoder = LogSNRObservationEncoder(max_server_vehicles=3, scales=custom)
        features = dict(zip(encoder.feature_names, encoder.encode(self.raw).values, strict=True))
        self.assertEqual(features["server[0].snr_log1p"],
                         normalized_log_snr(self.raw.servers[0].record.estimated_snr_linear, 9.))
        self.assertEqual(encoder.representation_contract(), {
            "schema_version": 1, "kind": "normalized-log1p-snr-v1",
            "input": "cached-large-scale-linear-sinr",
            "formula": "log1p(snr_linear)/log1p(reference_snr_linear)",
            "reference_snr_linear": 9., "fitted": False, "clipped": False})

    def test_legacy_linear_arithmetic_and_names_remain_unchanged(self):
        encoded = self.legacy.encode(self.raw)
        features = dict(zip(self.legacy.feature_names, encoded.values, strict=True))
        self.assertEqual(features["server[0].snr"], self.raw.servers[0].record.estimated_snr_linear / 100.)
        self.assertFalse(any("log1p" in name for name in self.legacy.feature_names))
        self.assertEqual(encoded.schema_version, 3)
        self.assertEqual(self.legacy.scales, FeatureScales())

    def test_no_servers_still_preserves_global_vector(self):
        encoder = LogSNRObservationEncoder(max_server_vehicles=0, scales=self.scales)
        legacy = ObservationEncoder(max_server_vehicles=0, scales=self.scales)
        raw = replace(self.raw, servers=(), action_mask=(True,))
        actual, expected = encoder.encode(raw), legacy.encode(raw)
        self.assertEqual(actual.values, expected.values)
        self.assertEqual(actual.action_mask, expected.action_mask)
        self.assertEqual(len(actual.values), 36)


if __name__ == "__main__":
    unittest.main()
