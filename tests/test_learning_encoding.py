"""Fixed features, zero padding, normalization, and cached-information isolation."""

import math
import unittest
from dataclasses import replace

from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
from test_slotted_environment import make_env, to_decision


class ObservationEncodingTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.env.reset(seed=4)
        to_decision(self.env)
        self.observation = self.env.observations[1]
        self.encoder = ObservationEncoder(max_server_vehicles=3, scales=FeatureScales())

    def test_fixed_dimension_and_all_zero_padding(self):
        encoded = self.encoder.encode(self.observation)
        self.assertEqual(self.encoder.dimension, 38 + 15 * 3)
        self.assertEqual(len(encoded.values), self.encoder.dimension)
        self.assertEqual(encoded.values[-30:], (0.,) * 30)
        self.assertEqual(encoded.action_mask, (True, True, False, False))
        self.assertTrue(all(math.isfinite(value) for value in encoded.values))

    def test_named_normalization_and_no_clipping(self):
        raw = replace(self.observation, deadline_slack_s=-2., input_size_bits=2_000_000)
        features = dict(zip(self.encoder.feature_names, self.encoder.encode(raw).values, strict=True))
        self.assertEqual(features["task.input"], 2.)
        self.assertEqual(features["task.deadline_slack"], -2.)
        self.assertEqual(features["task.criticality_normal"], 1.)
        self.assertEqual(features["task.criticality_critical"], 0.)
        self.assertEqual(features["server[0].present"], 1.)
        self.assertEqual(features["query.cache_valid"], 1.)

    def test_custom_fixed_reference_scales(self):
        encoder = ObservationEncoder(max_server_vehicles=3, scales=FeatureScales(input_bits=1_000))
        self.assertEqual(encoder.encode(self.observation).values[0], 1.)

    def test_heading_is_periodic_not_discontinuous_raw_angle(self):
        one = self.encoder.encode(replace(self.observation, user_heading_rad=.4))
        two = self.encoder.encode(replace(self.observation, user_heading_rad=.4 + 2 * math.pi))
        for first, second in zip(one.values, two.values, strict=True):
            self.assertAlmostEqual(first, second)

    def test_identifiers_are_not_neural_features(self):
        altered = replace(self.observation, user_vehicle_id=99, task_id=123)
        self.assertEqual(self.encoder.encode(self.observation).values, self.encoder.encode(altered).values)
        self.assertEqual(self.encoder.encode(altered).task_id, 123)

    def test_live_resource_changes_cannot_change_encoded_snapshot(self):
        before = self.encoder.encode(self.observation)
        self.env.server_vehicles[9].position_m = (900., 0.)
        self.assertEqual(before, self.encoder.encode(self.observation))

    def test_masked_present_record_is_not_confused_with_padding(self):
        encoded = self.encoder.encode(replace(self.observation, action_mask=(True, False, False, False)))
        features = dict(zip(self.encoder.feature_names, encoded.values, strict=True))
        self.assertEqual(features["server[0].present"], 1.)
        self.assertFalse(encoded.action_mask[1])

    def test_zero_server_capacity_still_encodes_mec(self):
        encoder = ObservationEncoder(max_server_vehicles=0, scales=FeatureScales())
        raw = replace(self.observation, servers=(), action_mask=(True,))
        self.assertEqual(len(encoder.encode(raw).values), 38)

    def test_invalid_schema_capacity_mask_and_nonfinite_values(self):
        for change in ({"schema_version": 2}, {"servers": ()},
                       {"action_mask": (True, True, True, False)},
                       {"action_mask": (False, True, False, False)},
                       {"action_mask": (True,)},
                       {"action_mask": (True, 1, False, False)},
                       {"deadline_slack_s": float("nan")}, {"user_heading_rad": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.encoder.encode(replace(self.observation, **change))
        with self.assertRaises(ValueError):
            FeatureScales(time_s=0)
        with self.assertRaises(TypeError):
            FeatureScales(energy_j=True)
