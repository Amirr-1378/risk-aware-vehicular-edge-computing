"""Immutable split plan: no overlap, outcome-based seed filtering, or role reuse."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.run_scenario_audit import DEFAULT_PROTOCOL
from risk_aware_v2x.scenarios import ScenarioSpec
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, SPLITS


class DevelopmentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = DevelopmentProtocol.load(DEFAULT_PROTOCOL)

    def test_checked_in_configuration_matches_declared_nominal_defaults(self):
        self.assertEqual(self.plan.profile("nominal"), ScenarioSpec())
        self.assertEqual(self.plan.split_counts, (20, 6, 30))
        self.assertEqual(len(self.plan.profiles), 4)

    def test_every_role_and_split_has_a_distinct_seed(self):
        values = []
        for split, count in zip(SPLITS, self.plan.split_counts, strict=True):
            for index in range(count):
                seeds = self.plan.episode_seeds(split, index)
                values.extend((seeds.scenario_seed, seeds.channel_seed, seeds.policy_seed))
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(set(values).isdisjoint(self.plan.agent_seeds))

    def test_seed_access_order_and_profile_changes_do_not_reroll_episode_keys(self):
        original = self.plan.episode_seeds("validation", 2)
        self.plan.episode_seeds("train", 0)
        self.assertEqual(original, self.plan.episode_seeds("validation", 2))
        changed = replace(self.plan, profiles=(replace(ScenarioSpec(), arrival_rate_per_user_hz=4.),))
        self.assertEqual(original, changed.episode_seeds("validation", 2))
        self.assertNotEqual(self.plan.fingerprint, changed.fingerprint)

    def test_protocol_fingerprint_reproduces_and_changes_with_seed_contract(self):
        self.assertEqual(self.plan.fingerprint, DevelopmentProtocol.load(DEFAULT_PROTOCOL).fingerprint)
        self.assertNotEqual(self.plan.fingerprint, replace(self.plan, master_seed=1).fingerprint)

    def test_invalid_keys_counts_profiles_and_duplicate_agent_seeds(self):
        for split, index in (("unknown", 0), ("validation", 6), ("train", -1), ("test", True)):
            with self.subTest(split=split), self.assertRaises((TypeError, ValueError)):
                self.plan.episode_seeds(split, index)
        for changes in ({"agent_seeds": (1, 1)}, {"split_counts": (1, 0, 1)},
                        {"profiles": (ScenarioSpec(), ScenarioSpec())}, {"purpose": "final-paper"},
                        {"cvar_alpha": 1.}, {"bootstrap_confidence": 1.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, **changes)

    def test_duplicate_json_keys_cannot_silently_change_protocol_parameters(self):
        with patch("pathlib.Path.read_text", return_value='{"schema_version":1,"schema_version":2}'):
            with self.assertRaises(ValueError):
                DevelopmentProtocol.load("fixture.json")
