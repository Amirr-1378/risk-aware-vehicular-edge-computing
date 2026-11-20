"""Matched stochastic scope ablation: budgets, pairing, frozen state and output."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.learning_protocol import LearningProtocol
    from experiments.return_scope_protocol import ReturnScopeProtocol
    from experiments.run_return_scope_audit import _episode, run_return_scope_audit
    from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.schedules import EpsilonSchedule
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
    from risk_aware_v2x.scenarios import ScenarioSpec
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


def fixture_plans():
    spec = replace(ScenarioSpec(), num_users=2, num_servers=1, max_server_vehicle_slots=2,
                   arrival_horizon_slots=3, arrival_rate_per_user_hz=100.)
    scenarios = DevelopmentProtocol("scope-fixture", 7, (2, 2, 2), (11, 29),
        (spec, replace(spec, label="harsh", v2v_shadowing_std_db=6.)), bootstrap_resamples=10)
    learning = LearningProtocol("scope-fixture", scenarios.fingerprint, "nominal", ("nominal", "harsh"),
        2, 2, (11, 29), ("qr_dqn", "qr_dqn_cvar"),
        AgentConfig(hidden_sizes=(8,), num_quantiles=5, batch_size=2, learning_starts=2,
                    replay_capacity=64, target_update_interval=3), EpsilonSchedule(decay_slots=100), FeatureScales())
    return ReturnScopeProtocol("scope-fixture", learning), scenarios


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for return-scope audit")
class ReturnScopeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.scenarios = fixture_plans()
        cls.report = run_return_scope_audit(protocol=cls.protocol, scenarios=cls.scenarios)

    def test_complete_grid_same_initial_weights_and_transition_update_budget(self):
        runs = self.report["runs"]
        self.assertEqual(len(runs), 8)
        self.assertEqual(self.report["training_budget"]["offered_tasks_per_model"], 12)
        self.assertEqual(self.report["training_budget"]["expected_updates_per_model"], 11)
        initials = {}
        for run in runs:
            pair = run["algorithm"], run["agent_seed"]
            self.assertEqual(initials.setdefault(pair, run["initial_model_sha256"]), run["initial_model_sha256"])
            self.assertEqual(run["seen_transitions"], 12)
            self.assertEqual(run["gradient_updates"], 11)
            self.assertEqual(run["target_synchronizations"], 3)
            for episode in run["training"] + [e for values in run["validation"].values() for e in values]:
                self.assertEqual(episode["metrics"]["completed_tasks"], 6.)
                self.assertAlmostEqual(sum(episode["accounting"]["total_cost"].values()), episode["metrics"]["total_cost"])
                if run["return_scope"] == "team":
                    self.assertAlmostEqual(episode["team_accounting"]["total_cost"], episode["metrics"]["total_cost"])
        self.assertFalse(self.report["centralized_controller"])
        self.assertFalse(self.report["main_training_started"])

    def test_validation_pairing_and_fixed_policies_are_not_duplicated_across_scopes(self):
        self.assertEqual(self.report["physical_episode_executions"], 68)  # 8*(2 train+4 val)+20 fixed
        for run in self.report["runs"]:
            for profile, audit in self.report["analytical_validation"].items():
                for learned, reference in zip(run["validation"][profile], audit["scenarios"], strict=True):
                    self.assertEqual(learned["seeds"], reference["seeds"])
                    self.assertEqual(learned["scenario_fingerprint"], reference["fingerprint"])
                    self.assertEqual(learned["updates"], [])
                    self.assertEqual(learned["training_slots_before"], run["training_slots"])
                    self.assertEqual(learned["training_slots_after"], run["training_slots"])
        self.assertEqual(len(self.report["analytical_validation"]["nominal"]["policies"]), 5)

    def test_only_train_and_validation_seed_keys_are_requested(self):
        protocol = replace(self.protocol, learning=replace(self.protocol.learning, algorithms=("qr_dqn_cvar",),
            agent_seeds=(11,), training_episodes=1, validation_cases=1, validation_profiles=("nominal",)))
        original = DevelopmentProtocol.episode_seeds
        seen = []

        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            seen.append(split)
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = run_return_scope_audit(protocol=protocol, scenarios=self.scenarios)
        self.assertEqual(set(seen), {"train", "validation"})
        self.assertFalse(report["test_split_accessed"])

    def test_iteration_order_preserves_every_model_run(self):
        learning = replace(self.protocol.learning, algorithms=self.protocol.learning.algorithms[::-1],
            agent_seeds=self.protocol.learning.agent_seeds[::-1], validation_profiles=self.protocol.learning.validation_profiles[::-1])
        reverse = replace(self.protocol, learning=learning, return_scopes=self.protocol.return_scopes[::-1])
        report = run_return_scope_audit(protocol=reverse, scenarios=self.scenarios)
        key = lambda runs: {(r["return_scope"], r["algorithm"], r["agent_seed"]): r for r in runs}
        self.assertEqual(key(report["runs"]), key(self.report["runs"]))

    def test_output_checkpoint_scopes_and_exclusive_creation(self):
        plan = replace(self.protocol, learning=replace(self.protocol.learning, algorithms=("qr_dqn_cvar",),
            agent_seeds=(11,), training_episodes=1, validation_cases=1, validation_profiles=("nominal",)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            report = run_return_scope_audit(protocol=plan, scenarios=self.scenarios, output_directory=output)
            encoder = ObservationEncoder(max_server_vehicles=2, scales=plan.learning.feature_scales)
            for run in report["runs"]:
                loader = TeamNeuralCostAgent if run["return_scope"] == "team" else NeuralCostAgent
                agent, metadata = loader.load_checkpoint(output / run["checkpoint"], encoder)
                self.assertEqual(metadata["return_scope_protocol_sha256"], plan.fingerprint)
                self.assertEqual(metadata["source_sha256"], report["provenance"]["source_sha256"])
                self.assertEqual(metadata["return_scope"], run["return_scope"])
                self.assertEqual(agent.learning_return_contract(), run["learning_return_contract"])
            before = (output / "report.json").read_bytes()
            json.loads(before)
            self.assertTrue((output / "summary.md").exists())
            with self.assertRaises(FileExistsError):
                run_return_scope_audit(protocol=plan, scenarios=self.scenarios, output_directory=output)
            self.assertEqual(before, (output / "report.json").read_bytes())

    def test_validation_mutation_and_guard_failure_abort_the_audit(self):
        small = replace(self.protocol, learning=replace(self.protocol.learning, algorithms=("qr_dqn_cvar",),
            agent_seeds=(11,), training_episodes=1, validation_cases=1, validation_profiles=("nominal",)))

        def mutate(*args, **kwargs):
            report, clock = _episode(*args, **kwargs)
            if not kwargs["training"]:
                args[3].gradient_updates += 1
            return report, clock

        with patch("experiments.run_return_scope_audit._episode", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                run_return_scope_audit(protocol=small, scenarios=self.scenarios)
        with patch("experiments.run_return_scope_audit._episode", side_effect=RuntimeError("fixture guard")) as call:
            with self.assertRaisesRegex(RuntimeError, "fixture guard"):
                run_return_scope_audit(protocol=small, scenarios=self.scenarios)
            self.assertEqual(call.call_count, 1)

    def test_mismatch_fails_before_scene_generation_or_creating_output(self):
        bad = replace(self.protocol, learning=replace(self.protocol.learning, scenario_protocol_sha256="0" * 64))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with patch("experiments.run_return_scope_audit.generate_scenario", side_effect=AssertionError("too soon")):
                with self.assertRaises(ValueError):
                    run_return_scope_audit(protocol=bad, scenarios=self.scenarios, output_directory=output)
            self.assertFalse(output.exists())
