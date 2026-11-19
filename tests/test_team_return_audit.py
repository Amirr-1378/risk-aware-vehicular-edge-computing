"""Matched fixed-policy verification of local inputs and team return ledgers."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.run_team_return_audit import run_team_return_audit
from risk_aware_v2x.scenarios import ScenarioSpec
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


class TeamReturnAuditTests(unittest.TestCase):
    def plan(self, *, empty=False):
        specification = replace(ScenarioSpec(), num_users=2, num_servers=1, max_server_vehicle_slots=2,
            arrival_horizon_slots=4, arrival_rate_per_user_hz=0. if empty else 100.)
        return DevelopmentProtocol("team-return-fixture", 7, (1, 1, 1), (11,),
                                   (specification,), bootstrap_resamples=10)

    def test_all_policy_pairs_preserve_inputs_actions_physics_and_unique_global_cost(self):
        report = run_team_return_audit(protocol=self.plan(), profiles=("nominal",), cases=1, gamma_per_slot=.9)
        self.assertEqual(report["matched_policy_pairs"], 5)
        self.assertEqual(report["physical_episode_executions"], 10)
        self.assertFalse(report["centralized_controller"])
        self.assertFalse(report["training_started"])
        for record in report["episodes"][0]["policies"].values():
            self.assertTrue(record["matched_individual_run_verified"])
            self.assertEqual(record["metrics"]["completed_tasks"], 8.)
            accounting = record["team_accounting"]
            self.assertAlmostEqual(accounting["total_cost"], record["metrics"]["total_cost"])
            self.assertAlmostEqual(accounting["total_cost"], accounting["prefix_cost"] + accounting["closed_interval_cost"])
            self.assertTrue(all(t["team_cost"] >= t["individual_cost"] for t in record["transitions"]))
        json.dumps(report, allow_nan=False)

    def test_empty_case_is_not_dropped_and_has_no_fabricated_actor_transition(self):
        report = run_team_return_audit(protocol=self.plan(empty=True), profiles=("nominal",), cases=1)
        for record in report["episodes"][0]["policies"].values():
            self.assertEqual(record["transitions"], [])
            self.assertEqual(record["team_accounting"]["total_cost"], 0.)

    def test_runner_only_requests_validation_seed_keys(self):
        plan = self.plan(empty=True)
        original = DevelopmentProtocol.episode_seeds
        seen = []

        def guard(instance, split, index):
            self.assertEqual(split, "validation")
            seen.append(split)
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = run_team_return_audit(protocol=plan, profiles=("nominal",), cases=1)
        self.assertEqual(seen, ["validation"])
        self.assertFalse(report["test_split_accessed"])

    def test_bad_prefix_or_profile_is_rejected_before_simulation(self):
        plan = self.plan()
        for changes in ({"cases": 2}, {"cases": True}, {"profiles": ("unknown",)},
                        {"profiles": ["nominal"]}, {"gamma_per_slot": 1.1}):
            with patch("experiments.run_team_return_audit.generate_scenario", side_effect=AssertionError("too soon")):
                with self.assertRaises((ValueError, TypeError)):
                    run_team_return_audit(protocol=plan, **changes)
