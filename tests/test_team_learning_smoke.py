"""Team-return end-to-end learning, frozen evaluation and episode-boundary resume."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.learning_runtime import deterministic_cpu, run_episode
    from experiments.run_environment_smoke import _json_value, build_smoke_scenario
    from experiments.run_learning_smoke import run_learning_smoke
    from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.schedules import EpsilonSchedule
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
    from risk_aware_v2x.agents.team_returns import TeamReturnAdapter


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for team training audits")
class TeamLearningSmokeTests(unittest.TestCase):
    def test_all_algorithms_train_with_team_costs_and_report_unique_physical_cost(self):
        for algorithm in ALGORITHMS:
            report = run_learning_smoke(return_scope="team", algorithm=algorithm, episodes=2)
            self.assertEqual(report["report_schema_version"], 3)
            self.assertEqual(report["learning_return_contract"]["return_scope"], "all-users-sum-v1")
            self.assertEqual(report["seen_transitions"], 12)
            self.assertEqual(report["gradient_updates"], 9)
            for episode in report["training"] + report["evaluation"]:
                actual = episode["metrics"]["total_cost"]
                team = episode["team_accounting"]
                self.assertAlmostEqual(actual, team["total_cost"])
                self.assertAlmostEqual(actual, sum(episode["accounting"]["total_cost"].values()))
                self.assertAlmostEqual(actual, team["prefix_cost"] + team["closed_interval_cost"])
                self.assertEqual(episode["metrics"]["completed_tasks"], 6.)
            for episode in report["evaluation"]:
                self.assertEqual(episode["updates"], [])
                self.assertEqual(episode["training_slots_before"], episode["training_slots_after"])
                self.assertTrue(all(a["epsilon"] == 0. for a in episode["actions"]))
            json.dumps(report, default=_json_value, allow_nan=False)

    def test_split_resume_matches_full_model_replay_optimizer_rng_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            full, first, final = (Path(directory) / name for name in ("full.pt", "first.pt", "final.pt"))
            uninterrupted = run_learning_smoke(return_scope="team", episodes=4, checkpoint_path=full)
            run_learning_smoke(return_scope="team", episodes=2, checkpoint_path=first)
            resumed = run_learning_smoke(return_scope="team", episodes=2, resume_path=first, checkpoint_path=final)
            self.assertEqual(uninterrupted["evaluation"], resumed["evaluation"])
            self.assertEqual(uninterrupted["gradient_updates"], resumed["gradient_updates"])
            a, b = torch.load(full, weights_only=True), torch.load(final, weights_only=True)
            self.assertEqual(a["checkpoint_schema_version"], 3)
            self.assertEqual(a["replay"], b["replay"])
            self.assertEqual(a["exploration"], b["exploration"])
            self.assertEqual(a["run_metadata_json"], b["run_metadata_json"])
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for index, values in a["optimizer"]["state"].items():
                for key, value in values.items():
                    self.assertTrue(torch.equal(value, b["optimizer"]["state"][index][key]))
            for net in ("online", "target"):
                for key, value in a[net].items():
                    self.assertTrue(torch.equal(value, b[net][key]))

    def test_complete_seeded_team_report_reproduces(self):
        self.assertEqual(run_learning_smoke(return_scope="team", episodes=2),
                         run_learning_smoke(return_scope="team", episodes=2))

    def test_cross_scope_resume_and_existing_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "individual.pt"
            run_learning_smoke(episodes=0, checkpoint_path=path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                run_learning_smoke(return_scope="team", episodes=1, resume_path=path)
            with self.assertRaises(FileExistsError):
                run_learning_smoke(return_scope="team", episodes=1, checkpoint_path=path)
            self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            run_learning_smoke(return_scope="unknown", episodes=0)

    def test_runtime_rejects_mismatched_adapters_before_reset_or_learning(self):
        environment, _ = build_smoke_scenario()
        encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
        for adapter_kind, agent_kind in ((TeamReturnAdapter, NeuralCostAgent), (DecisionAdapter, TeamNeuralCostAgent)):
            adapter, agent = adapter_kind(environment, encoder), agent_kind(AgentConfig(), encoder, seed=7)
            with patch.object(adapter, "reset", side_effect=AssertionError("reset must not run")):
                with self.assertRaisesRegex(ValueError, "scope"):
                    run_episode(adapter, agent, environment_seed=7, training=True,
                                schedule=EpsilonSchedule(), training_slots=0)
            self.assertEqual(agent.seen_transitions, 0)

    def test_frozen_team_evaluation_preserves_online_target_optimizer_replay_and_rng(self):
        environment, _ = build_smoke_scenario()
        encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
        agent = TeamNeuralCostAgent(AgentConfig(hidden_sizes=(8,), num_quantiles=5,
            batch_size=2, learning_starts=2, replay_capacity=16), encoder, seed=7)
        adapter = TeamReturnAdapter(environment, encoder)
        with deterministic_cpu():
            _, clock = run_episode(adapter, agent, environment_seed=7, training=True,
                                  schedule=EpsilonSchedule(), training_slots=0)
            weights = [copy.deepcopy(net.state_dict()) for net in (agent.online, agent.target)]
            optimizer = copy.deepcopy(agent.optimizer.state_dict())
            replay = agent.replay.state_dict()
            rng = {uid: generator.getstate() for uid, generator in agent._exploration.items()}
            counts = (agent.seen_transitions, agent.gradient_updates)
            _, after = run_episode(adapter, agent, environment_seed=8, training=False,
                                   schedule=EpsilonSchedule(), training_slots=clock)
        self.assertEqual(after, clock)
        self.assertEqual(counts, (agent.seen_transitions, agent.gradient_updates))
        self.assertEqual(replay, agent.replay.state_dict())
        self.assertEqual(rng, {uid: generator.getstate() for uid, generator in agent._exploration.items()})
        for previous, net in zip(weights, (agent.online, agent.target), strict=True):
            for key, value in previous.items():
                self.assertTrue(torch.equal(value, net.state_dict()[key]))
        for index, values in optimizer["state"].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, agent.optimizer.state_dict()["state"][index][key]))

    def test_zero_episode_team_run_does_not_fabricate_training_or_clock_progress(self):
        report = run_learning_smoke(return_scope="team", episodes=0)
        self.assertEqual(report["seen_transitions"], 0)
        self.assertEqual(report["gradient_updates"], 0)
        self.assertEqual(report["replay_size"], 0)
        self.assertEqual(report["training_slots"], 0)
