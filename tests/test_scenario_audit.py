"""Complete paired baseline runs; held-out test is closed by default."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.run_scenario_audit import DEFAULT_PROTOCOL, run_scenario_audit, save_audit_report
from risk_aware_v2x.scenarios import ScenarioSpec
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


def tiny_plan():
    return DevelopmentProtocol("fixture-v1", 7, (2, 2, 2), (11, 29), (
        replace(ScenarioSpec(), arrival_horizon_slots=30, num_users=2, num_servers=2,
                arrival_rate_per_user_hz=15.),
    ), bootstrap_resamples=50)


class ScenarioAuditTests(unittest.TestCase):
    def test_paired_cases_use_same_scenes_and_complete_every_task_with_full_cost(self):
        report = run_scenario_audit(protocol=tiny_plan(), cases=2)
        self.assertEqual(len(report["policies"]), 5)
        reference = report["policies"]["always_mec"]
        for episodes in report["policies"].values():
            for expected, actual in zip(reference, episodes, strict=True):
                self.assertEqual(expected["scenario_fingerprint"], actual["scenario_fingerprint"])
                self.assertEqual(expected["metrics"]["arrived_tasks"], actual["metrics"]["completed_tasks"])
                self.assertAlmostEqual(actual["metrics"]["total_cost"],
                                       sum(actual["accounting"]["total_cost"].values()))
        self.assertEqual(len(report["paired_mean_cost_difference_vs_always_mec"]), 4)
        json.dumps(report, allow_nan=False)

    def test_whole_seeded_report_reproduces(self):
        self.assertEqual(run_scenario_audit(protocol=tiny_plan()), run_scenario_audit(protocol=tiny_plan()))

    def test_test_split_is_blocked_before_scenario_construction(self):
        with patch("experiments.run_scenario_audit.generate_scenario", side_effect=AssertionError("test leaked")):
            with self.assertRaises(ValueError):
                run_scenario_audit(protocol=tiny_plan(), split="test")
        # Software fixture only, not the checked-in held-out cases.
        result = run_scenario_audit(protocol=tiny_plan(), split="test", allow_test=True, cases=1)
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["paired_mean_cost_difference_vs_always_mec"], {})

    def test_all_development_profiles_drain_on_a_small_validation_fixture(self):
        full = DevelopmentProtocol.load(DEFAULT_PROTOCOL)
        for profile in full.profiles:
            with self.subTest(profile=profile.label):
                plan = replace(full, profiles=(replace(profile, arrival_horizon_slots=25),),
                               split_counts=(1, 1, 1), bootstrap_resamples=10)
                report = run_scenario_audit(protocol=plan, profile=profile.label, cases=1)
                for episodes in report["policies"].values():
                    self.assertEqual(episodes[0]["metrics"]["arrived_tasks"], episodes[0]["metrics"]["completed_tasks"])

    def test_empty_episodes_are_not_filtered_out_of_risk_samples(self):
        plan = replace(tiny_plan(), profiles=(replace(tiny_plan().profiles[0], arrival_rate_per_user_hz=0.),))
        report = run_scenario_audit(protocol=plan)
        for summary in report["episode_total_cost_summaries"].values():
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["cvar_episode_total_cost"], 0.)
        with self.assertRaises(ValueError):
            run_scenario_audit(protocol=tiny_plan(), cases=3)

    def test_report_output_is_finite_exclusive_and_preserves_existing_results(self):
        report = {"schema_version": 1, "cost": 2.}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            save_audit_report(report, path)
            self.assertEqual(json.loads(path.read_text()), report)
            with self.assertRaises(FileExistsError):
                save_audit_report({"cost": 99.}, path)
            self.assertEqual(json.loads(path.read_text()), report)
            with self.assertRaises(ValueError):
                save_audit_report({"cost": float("nan")}, Path(directory) / "invalid.json")
