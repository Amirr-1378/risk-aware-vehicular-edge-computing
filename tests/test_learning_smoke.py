"""Optional end-to-end learning/evaluation and episode-boundary resume audits."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.run_learning_smoke import run_learning_smoke, run_episode
    from experiments.run_environment_smoke import _json_value, build_smoke_scenario
    from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.schedules import EpsilonSchedule


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for the training smoke tests")
class LearningSmokeTests(unittest.TestCase):
    def test_all_four_algorithms_update_and_evaluate_without_missing_cost(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                report = run_learning_smoke(algorithm=algorithm, episodes=2)
                self.assertGreater(report["gradient_updates"], 0)
                self.assertEqual(report["seen_transitions"], 12)
                self.assertEqual(report["gradient_updates"], 9)
                # Drain duration depends on the selected servers/fallbacks;
                # the 25-slot arrival horizon is not a fixed episode length.
                self.assertEqual(report["training_slots"],
                                 sum(episode["elapsed_slots"] for episode in report["training"]))
                self.assertGreaterEqual(report["training_slots"], 50)
                for episode in report["training"] + report["evaluation"]:
                    self.assertEqual(episode["metrics"]["completed_tasks"], 6.)
                    self.assertAlmostEqual(sum(episode["accounting"]["total_cost"].values()),
                                           episode["metrics"]["total_cost"])
                for evaluation in report["evaluation"]:
                    self.assertEqual(evaluation["updates"], [])
                    self.assertEqual(evaluation["training_slots_before"], evaluation["training_slots_after"])
                    self.assertTrue(all(action["epsilon"] == 0. for action in evaluation["actions"]))
                json.dumps(report, default=_json_value, allow_nan=False)

    def test_whole_seeded_report_reproduces(self):
        self.assertEqual(run_learning_smoke(episodes=2), run_learning_smoke(episodes=2))

    def test_split_resume_matches_uninterrupted_weights_replay_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            full, split, resumed = (Path(directory) / name for name in ("full.pt", "split.pt", "resumed.pt"))
            uninterrupted = run_learning_smoke(episodes=4, checkpoint_path=full)
            run_learning_smoke(episodes=2, checkpoint_path=split)
            continued = run_learning_smoke(episodes=2, resume_path=split, checkpoint_path=resumed)
            self.assertEqual(uninterrupted["evaluation"], continued["evaluation"])
            self.assertEqual(uninterrupted["gradient_updates"], continued["gradient_updates"])
            first, second = torch.load(full, weights_only=True), torch.load(resumed, weights_only=True)
            self.assertEqual(first["replay"], second["replay"])
            self.assertEqual(first["exploration"], second["exploration"])
            self.assertEqual(first["run_metadata_json"], second["run_metadata_json"])
            for key in first["online"]:
                self.assertTrue(torch.equal(first["online"][key], second["online"][key]))
                self.assertTrue(torch.equal(first["target"][key], second["target"][key]))

    def test_resume_protocol_mismatch_and_existing_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            run_learning_smoke(episodes=1, checkpoint_path=checkpoint)
            with self.assertRaises(ValueError):
                run_learning_smoke(episodes=1, resume_path=checkpoint, environment_seed=99)
            with self.assertRaises(ValueError):
                run_learning_smoke(algorithm="dqn", episodes=1, resume_path=checkpoint)
            with self.assertRaises(FileExistsError):
                run_learning_smoke(episodes=1, checkpoint_path=checkpoint)
            state = torch.load(checkpoint, weights_only=True)
            metadata = json.loads(state["run_metadata_json"])
            metadata["source_sha256"] = "different-source"
            state["run_metadata_json"] = json.dumps(metadata)
            incompatible = Path(directory) / "different-source.pt"
            torch.save(state, incompatible)
            with self.assertRaises(ValueError):
                run_learning_smoke(episodes=1, resume_path=incompatible)

    def test_no_training_evaluation_leaves_replay_and_counters_untouched(self):
        report = run_learning_smoke(episodes=0)
        self.assertEqual(report["seen_transitions"], 0)
        self.assertEqual(report["gradient_updates"], 0)
        self.assertEqual(report["replay_size"], 0)
        self.assertEqual(report["training_slots"], 0)

    def test_whole_evaluation_episode_does_not_change_agent_or_random_states(self):
        environment, _ = build_smoke_scenario()
        encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
        agent = NeuralCostAgent(AgentConfig(), encoder, seed=8)
        adapter = DecisionAdapter(environment, encoder)
        before = {key: value.clone() for key, value in agent.online.state_dict().items()}
        replay = agent.replay.state_dict()
        report, clock = run_episode(adapter, agent, environment_seed=4, training=False,
                                    schedule=EpsilonSchedule(), training_slots=37)
        self.assertEqual(clock, 37)
        self.assertEqual(report["updates"], [])
        self.assertEqual(replay, agent.replay.state_dict())
        self.assertEqual(agent._exploration, {})
        self.assertEqual(agent.seen_transitions, 0)
        for key, value in agent.online.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))

    def test_seed_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            run_learning_smoke(episodes=1, environment_seed=1_000_007)
