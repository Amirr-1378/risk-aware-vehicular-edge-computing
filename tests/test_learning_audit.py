"""Optional multi-seed train/validation audits, never the held-out test plan."""

import copy
import importlib.util
import json
import math
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.learning_protocol import LearningProtocol
    from experiments.learning_runtime import deterministic_cpu
    from experiments.run_learning_audit import (
        DEFAULT_LEARNING_PROTOCOL, _episode, model_fingerprint, run_learning_audit, summarize_runs,
    )
    from experiments.run_scenario_audit import DEFAULT_PROTOCOL
    from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.schedules import EpsilonSchedule
    from risk_aware_v2x.scenarios import ScenarioSpec, generate_scenario
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


def fixture_plans():
    spec = replace(ScenarioSpec(), num_users=1, num_servers=1, max_server_vehicle_slots=2,
                   arrival_horizon_slots=6, arrival_rate_per_user_hz=100.)
    scenarios = DevelopmentProtocol("learning-fixture-v1", 7, (2, 2, 2), (11, 29),
        (spec, replace(spec, label="harsh", v2v_shadowing_std_db=6.)), bootstrap_resamples=30)
    learning = LearningProtocol("fixture-v1", scenarios.fingerprint, "nominal", ("nominal", "harsh"),
        2, 2, (11, 29), ALGORITHMS,
        AgentConfig(hidden_sizes=(8,), num_quantiles=5, batch_size=2, learning_starts=2,
                    replay_capacity=64, target_update_interval=3),
        EpsilonSchedule(decay_slots=100), FeatureScales())
    return learning, scenarios


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for multi-seed audits")
class LearningProtocolTests(unittest.TestCase):
    def test_checked_in_learning_plan_is_bound_to_exact_physical_plan(self):
        learning = LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL)
        scenarios = DevelopmentProtocol.load(DEFAULT_PROTOCOL)
        learning.validate_against(scenarios)
        self.assertEqual(learning.agent_seeds, (11, 29))
        self.assertEqual(learning.training_episodes, 4)
        self.assertEqual(learning.validation_cases, 2)
        self.assertEqual(learning.selection_rule, "fixed-final-training-episode")
        self.assertEqual(learning.fingerprint, LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL).fingerprint)
        self.assertNotEqual(learning.fingerprint, replace(learning, training_episodes=3).fingerprint)

    def test_algorithms_share_hyperparameters_and_qr_risk_ablation(self):
        learning, _ = fixture_plans()
        for algorithm in ALGORITHMS:
            config = learning.agent_configuration(algorithm)
            self.assertEqual(config, replace(learning.common_agent, algorithm=algorithm))
            self.assertEqual(config.effective_risk_weight, .5 if algorithm == "qr_dqn_cvar" else 0.)

    def test_old_learning_contract_is_preserved_but_rejected_not_silently_upgraded(self):
        with self.assertRaisesRegex(ValueError, "v2"):
            LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL.with_name("learning_pilot_v1.json"))
        learning, _ = fixture_plans()
        for changes in ({"raw_observation_schema_version": 1}, {"encoder_schema_version": 1},
                        {"baseline_suite_version": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(learning, **changes)

    def test_invalid_or_mutable_protocol_settings_are_rejected(self):
        learning, _ = fixture_plans()
        for changes in ({"training_episodes": 0}, {"validation_cases": True},
                        {"agent_seeds": (11, 11)}, {"agent_seeds": (False,)},
                        {"agent_seeds": [11]}, {"validation_profiles": ("nominal", "nominal")},
                        {"algorithms": ("unknown",)}, {"selection_rule": "best-validation"},
                        {"common_agent": replace(learning.common_agent, algorithm="dqn")},
                        {"scenario_protocol_sha256": "wrong"}, {"schema_version": True}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(learning, **changes)

    def test_mismatches_fail_before_generation_and_output_creation(self):
        learning, scenarios = fixture_plans()
        for changes in ({"scenario_protocol_sha256": "0" * 64}, {"training_episodes": 3},
                        {"validation_cases": 3}, {"training_profile": "unknown"},
                        {"agent_seeds": (71,)},
                        {"common_agent": replace(learning.common_agent, alpha=.8)}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "new"
                with patch("experiments.run_learning_audit.generate_scenario", side_effect=AssertionError("too early")):
                    with self.assertRaises(ValueError):
                        run_learning_audit(learning=replace(learning, **changes), scenarios=scenarios,
                                           output_directory=output)
                self.assertFalse(output.exists())

    def test_json_rejects_duplicate_unknown_and_common_algorithm_overrides(self):
        original = DEFAULT_LEARNING_PROTOCOL.read_text(encoding="utf-8")
        data = json.loads(original)
        for text in ('{"revision":"a","revision":"b"}',
                     json.dumps({**data, "allow_test": True}),
                     json.dumps({**data, "common_agent": {**data["common_agent"], "algorithm": "dqn"}})):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises((ValueError, TypeError)):
                    LearningProtocol.load(path)


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for multi-seed audits")
class LearningAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learning, cls.scenarios = fixture_plans()
        cls.report = run_learning_audit(learning=cls.learning, scenarios=cls.scenarios)

    def test_all_methods_seeds_and_profiles_fully_drain_without_missing_cost(self):
        self.assertEqual(len(self.report["runs"]), 8)
        for run in self.report["runs"]:
            self.assertEqual(run["seen_transitions"], 12)
            self.assertEqual(run["gradient_updates"], 11)
            self.assertEqual(run["training_slots"], sum(item["elapsed_slots"] for item in run["training"]))
            episodes = run["training"] + [item for values in run["validation"].values() for item in values]
            for episode in episodes:
                self.assertEqual(episode["metrics"]["arrived_tasks"], 6)
                self.assertEqual(episode["metrics"]["completed_tasks"], 6)
                self.assertAlmostEqual(episode["metrics"]["total_cost"],
                                       sum(episode["accounting"]["total_cost"].values()))
            for values in run["validation"].values():
                for item in values:
                    self.assertFalse(item["training"])
                    self.assertEqual(item["training_slots_after"], run["training_slots"])
                    self.assertEqual(item["updates"], [])
                    self.assertTrue(all(action["epsilon"] == 0. for action in item["actions"]))
        json.dumps(self.report, allow_nan=False)

    def test_learned_and_fixed_policies_share_identical_validation_scenes_and_channel_seeds(self):
        reference_train = self.report["runs"][0]["training"]
        for run in self.report["runs"]:
            self.assertEqual([item["scenario_fingerprint"] for item in run["training"]],
                             [item["scenario_fingerprint"] for item in reference_train])
            for profile, audit in self.report["analytical_validation"].items():
                for learned, fixed in zip(run["validation"][profile], audit["scenarios"], strict=True):
                    self.assertEqual(learned["scenario_fingerprint"], fixed["fingerprint"])
                    self.assertEqual(learned["seeds"], fixed["seeds"])
                    self.assertEqual(learned["environment_seed"], fixed["seeds"]["channel_seed"])

    def test_risk_is_per_trained_policy_not_pooled_across_seeds(self):
        for run in self.report["runs"]:
            for summary in run["validation_cost_summaries"].values():
                self.assertEqual(summary["episodes"], 2)
                self.assertAlmostEqual(summary["nominal_tail_mass_in_episodes"], .2)
        for profile in self.report["conditional_training_seed_summaries"].values():
            self.assertIn("NOT scenario-population", profile["interval_scope"])
            self.assertEqual(len(profile["paired_seed_mean_intervals"]), 9)
            self.assertTrue(all(value["pairs"] == 2 for value in profile["paired_seed_mean_intervals"].values()))
            self.assertNotIn("cvar_episode_total_cost", profile)

    def test_complete_seeded_report_is_reproducible_and_global_rng_is_untouched(self):
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        repeated = run_learning_audit(learning=self.learning, scenarios=self.scenarios)
        self.assertEqual(self.report, repeated)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    def test_seed_statistics_and_paired_difference_use_seed_means_not_grid_cells(self):
        runs = copy.deepcopy(self.report["runs"])
        for run in runs:
            # A deliberately hand-computable seed statistic, independent of
            # task counts or validation episode pseudo-replication.
            value = 2. if run["agent_seed"] == 11 else 4.
            for summary in run["validation_cost_summaries"].values():
                summary["mean_episode_total_cost"] = value + (1. if run["algorithm"] == "qr_dqn_cvar" else 0.)
        result = summarize_runs(runs, self.report["analytical_validation"], self.learning, self.scenarios)
        for profile in result.values():
            qr = profile["per_algorithm"]["qr_dqn"]
            self.assertEqual(qr["mean_of_seed_means"], 3.)
            self.assertAlmostEqual(qr["sample_std_of_seed_means"], math.sqrt(2))
            delta = profile["paired_seed_mean_intervals"]["qr_dqn_cvar_minus_qr_dqn"]
            self.assertEqual(delta["pairs"], 2)
            self.assertEqual(delta["lower"], 1.)
            self.assertEqual(delta["upper"], 1.)
        for incomplete in (runs[:-1], [*runs, runs[0]]):
            with self.assertRaises(ValueError):
                summarize_runs(incomplete, self.report["analytical_validation"], self.learning, self.scenarios)

    def test_execution_requests_only_train_validation_episodes(self):
        original = DevelopmentProtocol.episode_seeds
        seen = []
        def guarded(plan, split, index):
            self.assertNotEqual(split, "test")
            seen.append(split)
            return original(plan, split, index)
        small = replace(self.learning, algorithms=("qr_dqn_cvar",), agent_seeds=(11,),
                        training_episodes=1, validation_cases=1, validation_profiles=("nominal",))
        with patch.object(DevelopmentProtocol, "episode_seeds", guarded):
            report = run_learning_audit(learning=small, scenarios=self.scenarios)
        self.assertEqual(set(seen), {"train", "validation"})
        self.assertFalse(report["test_split_accessed"])
        self.assertEqual(report["conditional_training_seed_summaries"]["nominal"]["paired_seed_mean_intervals"], {})

    def test_empty_episodes_remain_included_with_no_fabricated_learning(self):
        empty = replace(self.scenarios, profiles=(replace(self.scenarios.profile("nominal"),
                                                        arrival_rate_per_user_hz=0.),))
        small = replace(self.learning, scenario_protocol_sha256=empty.fingerprint,
                        algorithms=("dqn",), agent_seeds=(11,), training_episodes=1,
                        validation_cases=1, validation_profiles=("nominal",))
        report = run_learning_audit(learning=small, scenarios=empty)
        run = report["runs"][0]
        self.assertEqual(run["seen_transitions"], 0)
        self.assertEqual(run["gradient_updates"], 0)
        self.assertEqual(run["validation_cost_summaries"]["nominal"]["episodes"], 1)
        self.assertEqual(run["validation_cost_summaries"]["nominal"]["cvar_episode_total_cost"], 0.)
        self.assertIsNone(run["validation"]["nominal"][0]["metrics"]["mean_task_cost"])

    def test_algorithm_seed_and_profile_iteration_order_does_not_change_any_run(self):
        reordered = replace(self.learning, algorithms=self.learning.algorithms[::-1],
                            agent_seeds=self.learning.agent_seeds[::-1],
                            validation_profiles=self.learning.validation_profiles[::-1])
        report = run_learning_audit(learning=reordered, scenarios=self.scenarios)
        index = lambda runs: {(run["algorithm"], run["agent_seed"]): run for run in runs}
        self.assertEqual(index(report["runs"]), index(self.report["runs"]))

    def test_cpu_settings_are_restored_even_if_execution_raises(self):
        before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                  torch.is_deterministic_algorithms_warn_only_enabled())
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with deterministic_cpu():
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                raise RuntimeError("fixture")
        self.assertEqual(before, (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                                 torch.is_deterministic_algorithms_warn_only_enabled()))

    def test_final_checkpoint_contract_and_exclusive_output_preserve_existing_artifacts(self):
        small = replace(self.learning, algorithms=("qr_dqn_cvar",), agent_seeds=(11,),
                        training_episodes=1, validation_cases=1, validation_profiles=("nominal",))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            report = run_learning_audit(learning=small, scenarios=self.scenarios, output_directory=output)
            path = output / report["runs"][0]["checkpoint"]
            encoder = ObservationEncoder(max_server_vehicles=2, scales=small.feature_scales)
            agent, metadata = NeuralCostAgent.load_checkpoint(path, encoder)
            self.assertEqual(metadata["model_sha256"], model_fingerprint(agent))
            self.assertEqual(metadata["learning_protocol_sha256"], small.fingerprint)
            self.assertEqual(metadata["scenario_protocol_sha256"], self.scenarios.fingerprint)
            self.assertEqual(metadata["source_sha256"], report["provenance"]["source_sha256"])
            self.assertEqual(metadata["selection_rule"], "fixed-final-training-episode")
            self.assertEqual(metadata["raw_observation_schema_version"], 2)
            self.assertEqual(metadata["encoder_schema_version"], 2)
            self.assertEqual(metadata["baseline_suite_version"], 2)
            self.assertEqual(agent.gradient_updates, 5)
            saved = (output / "report.json").read_bytes()
            json.loads(saved)
            with self.assertRaises(FileExistsError):
                run_learning_audit(learning=small, scenarios=self.scenarios, output_directory=output)
            self.assertEqual(saved, (output / "report.json").read_bytes())

    def test_new_stochastic_evaluation_preserves_full_agent_and_rng_state(self):
        encoder = ObservationEncoder(max_server_vehicles=2, scales=self.learning.feature_scales)
        keys = [self.scenarios.episode_seeds(split, 0) for split in ("train", "validation")]
        scenes = [generate_scenario(self.scenarios.profile("nominal"), seed=key.scenario_seed) for key in keys]
        with deterministic_cpu():
            agent = NeuralCostAgent(self.learning.common_agent, encoder, seed=11)
            _, clock = _episode(scenes[0], keys[0], encoder, agent, self.learning, 0, training=True)
            replay = copy.deepcopy(agent.replay.state_dict())
            exploration = {key: rng.getstate() for key, rng in agent._exploration.items()}
            networks = [copy.deepcopy(network.state_dict()) for network in (agent.online, agent.target)]
            optimizer = copy.deepcopy(agent.optimizer.state_dict())
            _, unchanged = _episode(scenes[1], keys[1], encoder, agent, self.learning, clock, training=False)
            self.assertEqual(unchanged, clock)
            self.assertEqual(replay, agent.replay.state_dict())
            self.assertEqual(exploration, {key: rng.getstate() for key, rng in agent._exploration.items()})
            for expected, network in zip(networks, (agent.online, agent.target), strict=True):
                for name, tensor in network.state_dict().items():
                    self.assertTrue(torch.equal(expected[name], tensor))
            actual = agent.optimizer.state_dict()
            self.assertEqual(optimizer["param_groups"], actual["param_groups"])
            for key, values in optimizer["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, actual["state"][key][name]))

    def test_failure_aborts_instead_of_dropping_or_replacing_a_run(self):
        small = replace(self.learning, algorithms=("dqn",), agent_seeds=(11,))
        with patch("experiments.run_learning_audit._episode", side_effect=RuntimeError("failure fixture")) as execute:
            with self.assertRaisesRegex(RuntimeError, "failure fixture"):
                run_learning_audit(learning=small, scenarios=self.scenarios)
            self.assertEqual(execute.call_count, 1)
