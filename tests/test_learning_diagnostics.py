"""Hand-computed aggregation and read-only report/pairing checks."""

import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from experiments.summarize_learning_audit import render_markdown, summarize_episodes, summarize_learning_audit


def episode(tasks, cost, energy, latency, misses, tardiness, sv_work, direct, fallback):
    return {"metrics": {"cost_model_version": 2, "arrived_tasks": tasks, "completed_tasks": tasks,
        "total_cost": cost, "mobile_energy_j": energy, "mean_task_latency_s": latency,
        "deadline_misses": misses, "total_task_tardiness_s": tardiness,
        "mean_task_tardiness_s": tardiness / tasks if tasks else None,
        "v2v_completed_workload_fraction": sv_work,
        "mec_completed_workload_fraction": 1. - sv_work if tasks else None,
        "direct_mec_task_fraction": direct, "fallback_task_fraction": fallback, "query_count": tasks}}


class EpisodeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.values = [episode(1, 2., .1, 1., 0, 0., 1., 0., 0.),
                       episode(3, 6., .9, 3., 2, 6., .25, .5, .25),
                       episode(0, 0., 0., None, 0, 0., 0., None, None)]

    def test_cost_risk_includes_empty_episodes_and_fractional_tail_mass(self):
        result = summarize_episodes(self.values, alpha=.5)
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["empty_episodes"], 1)
        self.assertEqual(result["task_instances"], 4)
        risk = result["episode_cost_risk"]
        self.assertAlmostEqual(risk["mean_episode_total_cost"], 8 / 3)
        self.assertAlmostEqual(risk["nominal_tail_mass_in_episodes"], 1.5)
        self.assertAlmostEqual(risk["cvar_episode_total_cost"], 14 / 3)

    def test_task_weighting_and_episode_workload_weighting_have_distinct_denominators(self):
        result = summarize_episodes(self.values, alpha=.9)
        self.assertAlmostEqual(result["task_weighted_latency_s"], 2.5)
        self.assertAlmostEqual(result["task_weighted_mobile_energy_j"], .25)
        self.assertAlmostEqual(result["task_weighted_deadline_miss_fraction"], .5)
        self.assertAlmostEqual(result["task_weighted_tardiness_s"], 1.5)
        self.assertAlmostEqual(result["task_weighted_direct_mec_fraction"], .375)
        self.assertAlmostEqual(result["task_weighted_fallback_fraction"], .1875)
        self.assertAlmostEqual(result["episode_mean_v2v_workload_fraction_nonempty"], .625)
        self.assertAlmostEqual(result["episode_mean_mec_workload_fraction_nonempty"], .375)

    def test_empty_only_sample_has_zero_cost_and_undefined_task_rates(self):
        result = summarize_episodes(self.values[-1:], alpha=.9)
        self.assertEqual(result["episode_cost_risk"]["mean_episode_total_cost"], 0.)
        for key in ("task_weighted_latency_s", "task_weighted_mobile_energy_j",
                    "task_weighted_deadline_miss_fraction", "episode_mean_mec_workload_fraction_nonempty"):
            self.assertIsNone(result[key])

    def test_input_and_global_random_state_are_untouched(self):
        before = copy.deepcopy(self.values)
        state = random.getstate()
        one = summarize_episodes(self.values, alpha=.9)
        self.assertEqual(one, summarize_episodes(self.values, alpha=.9))
        self.assertEqual(before, self.values)
        self.assertEqual(state, random.getstate())

    def test_nonfinite_negative_fractional_counts_old_cost_and_incomplete_drain_are_rejected(self):
        for key, value in (("total_cost", float("nan")), ("mobile_energy_j", -1.),
                           ("arrived_tasks", .5), ("completed_tasks", 0), ("deadline_misses", 2),
                           ("v2v_completed_workload_fraction", 1.1), ("cost_model_version", True),
                           ("cost_model_version", 1), ("mean_task_latency_s", float("inf"))):
            values = copy.deepcopy(self.values)
            values[0]["metrics"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                summarize_episodes(values, alpha=.9)
        with self.assertRaises(ValueError):
            summarize_episodes([], alpha=.9)


HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for diagnostic integration")
class LearningDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.learning_protocol import LearningProtocol
        from experiments.run_learning_audit import DEFAULT_LEARNING_PROTOCOL, run_learning_audit
        from risk_aware_v2x.scenarios import ScenarioSpec
        from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol

        specification = replace(ScenarioSpec(), num_users=1, num_servers=1,
                                max_server_vehicle_slots=2, arrival_horizon_slots=3,
                                arrival_rate_per_user_hz=100.)
        scenarios = DevelopmentProtocol("diagnostics-fixture", 7, (1, 1, 1), (11, 29),
                                        (specification,), bootstrap_resamples=10)
        learning = LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL)
        learning = replace(learning, scenario_protocol_sha256=scenarios.fingerprint,
            algorithms=("qr_dqn_cvar",), agent_seeds=(11, 29), training_episodes=1,
            validation_cases=1, validation_profiles=("nominal",),
            common_agent=replace(learning.common_agent, hidden_sizes=(8,), num_quantiles=5,
                                 batch_size=2, learning_starts=2, replay_capacity=16))
        cls.report = run_learning_audit(learning=learning, scenarios=scenarios)

    def test_one_row_per_model_profile_and_fixed_policies_once_without_seed_pooling(self):
        before = copy.deepcopy(self.report)
        state = random.getstate()
        summary = summarize_learning_audit(self.report)
        self.assertEqual(summary["episode_executions"], 9)  # 2 train + 2 learned val + 5 fixed val
        self.assertEqual(len(summary["validation_rows"]), 7)
        learned = [row for row in summary["validation_rows"] if row["kind"] == "learned"]
        self.assertEqual({row["agent_seed"] for row in learned}, {11, 29})
        self.assertTrue(all(row["episode_cost_risk"]["episodes"] == 1 for row in learned))
        self.assertFalse(summary["main_training_started"])
        self.assertFalse(summary["test_split_accessed"])
        self.assertEqual(self.report, before)
        self.assertEqual(state, random.getstate())
        json.dumps(summary, allow_nan=False)

    def test_training_trace_reconciles_updates_and_uses_physical_slot_epsilon(self):
        summary = summarize_learning_audit(self.report)
        for trace in summary["training_traces"]:
            row = trace["episodes"][0]
            self.assertEqual(row["updates"], 2)
            self.assertEqual(row["transitions"], 3)
            slots = row["training_slots_after"]
            self.assertAlmostEqual(row["epsilon_at_episode_end"], 1. - min(1., slots / 1000) * .95)
            self.assertIsNotNone(row["mean_update_loss"])

    def test_no_optimizer_updates_remain_undefined_instead_of_fabricated_zero_loss(self):
        report = copy.deepcopy(self.report)
        for run in report["runs"]:
            run["gradient_updates"] = 0
            run["training"][0]["updates"] = []
        summary = summarize_learning_audit(report)
        for trace in summary["training_traces"]:
            row = trace["episodes"][0]
            self.assertEqual(row["updates"], 0)
            self.assertIsNone(row["mean_update_loss"])
            self.assertIsNone(row["max_gradient_norm_before_clip"])

    def test_missing_seed_baseline_or_episode_fails_instead_of_omitting_it(self):
        mutations = [lambda r: r["runs"].pop(),
            lambda r: r["runs"].append(copy.deepcopy(r["runs"][0])),
            lambda r: r["runs"][0]["validation"]["nominal"].clear(),
            lambda r: r["analytical_validation"]["nominal"]["policies"].pop("always_mec")]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                summarize_learning_audit(report)

    def test_mismatched_scenario_or_channel_seed_is_rejected(self):
        for section in ("training", "validation"):
            for key in ("scenario_fingerprint", "seeds"):
                report = copy.deepcopy(self.report)
                records = report["runs"][0][section]
                item = records[0] if section == "training" else records["nominal"][0]
                if key == "seeds":
                    item[key]["channel_seed"] += 1
                else:
                    item[key] = "wrong-scene"
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    summarize_learning_audit(report)

    def test_old_cost_open_test_changed_protocol_and_validation_updates_are_rejected(self):
        mutations = [lambda r: r.update(report_schema_version=2),
            lambda r: r.update(test_split_accessed=True),
            lambda r: r["model_contract"].update(cost_model_version=1),
            lambda r: r["scenarios"][0].update(split="test"),
            lambda r: r["learning_protocol"].update(training_episodes=99),
            lambda r: r["runs"][0]["validation"]["nominal"][0].update(training=True),
            lambda r: r["runs"][0]["validation"]["nominal"][0]["actions"][0].update(epsilon=.1),
            lambda r: r["runs"][0].update(gradient_updates=99)]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                summarize_learning_audit(report)

    def test_markdown_records_tail_sample_limit_and_does_not_pool_latency_quantiles(self):
        summary = summarize_learning_audit(self.report)
        markdown = render_markdown(summary)
        self.assertIn("Tail episodes", markdown)
        self.assertIn("not establish convergence", markdown)
        self.assertNotIn("pooled_latency_p95", markdown)
        summary["cvar_alpha"] = .5
        self.assertIn("CVaR 0.5", render_markdown(summary))

    def test_cli_hashes_input_and_never_overwrites_or_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            source.write_text(json.dumps(self.report, allow_nan=False), encoding="utf-8")
            output = Path(directory) / "summary"
            command = [sys.executable, "-m", "experiments.summarize_learning_audit", str(source), "--output-dir", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertFalse(json.loads(first.stdout)["main_training_started"])
            saved = (output / "diagnostics.json").read_bytes()
            self.assertEqual(json.loads(saved)["input_report_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            repeated = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(saved, (output / "diagnostics.json").read_bytes())
