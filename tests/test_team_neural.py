"""Global-cost targets, local inference, strict checkpoint scopes and exact resume."""

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent


def fixture(algorithm="dqn", **changes):
    from test_team_replay import sample
    encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
    config = AgentConfig(**{"algorithm": algorithm, "hidden_sizes": (), "num_quantiles": 15,
        "batch_size": 2, "learning_starts": 2, "replay_capacity": 16,
        "target_update_interval": 2, "gamma_per_slot": .9, **changes})
    return TeamNeuralCostAgent(config, encoder, seed=8), encoder, sample(encoder.dimension)


def constant(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32).flatten())


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for team neural tests")
class TeamNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_team_targets_use_global_interval_cost_and_semimarkov_discount(self):
        for algorithm, expected in (("dqn", 9.12), ("double_dqn", 13.98)):
            agent, _, item = fixture(algorithm)
            constant(agent.online, [9., 1., -1e6])
            constant(agent.target, [2., 8., -1e6])
            self.assertAlmostEqual(float(agent.targets((item,))[0, 0]), expected, places=5)

    def test_risk_and_terminal_targets_keep_existing_quantile_math(self):
        for algorithm in ("qr_dqn", "qr_dqn_cvar"):
            agent, _, item = fixture(algorithm)
            constant(agent.target, [[0.] * 14 + [150.], [12.] * 15, [-1e9] * 15])
            targets = agent.targets((item,))[0]
            expected = torch.tensor([7.5] * 14 + [129.]) if algorithm == "qr_dqn" else torch.full((15,), 17.22)
            self.assertTrue(torch.allclose(targets, expected, atol=1e-5))
            terminal = replace(item, source=replace(item.source, terminated=True, next_observation=None, bootstrap_discount=0.))
            self.assertTrue(torch.equal(agent.targets((terminal,)), torch.full((1, 15), 7.5)))

    def test_inference_initialization_exploration_and_equal_cost_updates_match_individual_agent(self):
        for algorithm in ALGORITHMS:
            team, encoder, item = fixture(algorithm)
            individual = NeuralCostAgent(team.config, encoder, seed=8)
            self.assertEqual(team.preference_diagnostics(item.observation), individual.preference_diagnostics(item.observation))
            self.assertEqual([team.act(item.observation, epsilon=.7) for _ in range(6)],
                             [individual.act(item.observation, epsilon=.7) for _ in range(6)])
            equal_cost = replace(item, cost=item.source.cost, discounted_cost=item.source.discounted_cost)
            for _ in range(2):
                team.observe(equal_cost)
                individual.observe(item.source)
            self.assertEqual(team.learn(), individual.learn())
            for one, two in zip(team.online.parameters(), individual.online.parameters(), strict=True):
                self.assertTrue(torch.equal(one, two))

    def test_mixed_scopes_and_gamma_mismatch_are_rejected_before_replay_insertion(self):
        team, encoder, item = fixture()
        individual = NeuralCostAgent(team.config, encoder, seed=8)
        with self.assertRaises(TypeError):
            team.observe(item.source)
        with self.assertRaises(TypeError):
            individual.observe(item)
        terminal = replace(item, source=replace(item.source, terminated=True, next_observation=None, bootstrap_discount=0.), gamma_per_slot=.8)
        with self.assertRaises(ValueError):
            team.observe(terminal)
        self.assertEqual(team.seen_transitions, 0)
        self.assertEqual(len(team.replay), 0)

    def test_team_checkpoint_roundtrip_reproduces_optimizer_replay_exploration_and_updates(self):
        team, encoder, item = fixture("qr_dqn_cvar")
        team.observe(item)
        team.observe(item)
        team.learn()
        team.act(item.observation, epsilon=.7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "team.pt"
            team.save_checkpoint(path, run_metadata={"scope": "team", "episode": 1})
            restored, metadata = TeamNeuralCostAgent.load_checkpoint(path, encoder)
            self.assertEqual(metadata, {"scope": "team", "episode": 1})
            self.assertEqual(team.replay.state_dict(), restored.replay.state_dict())
            self.assertEqual(team.act(item.observation, epsilon=.7), restored.act(item.observation, epsilon=.7))
            self.assertEqual(team.learn(), restored.learn())
            for one, two in zip(team.online.parameters(), restored.online.parameters(), strict=True):
                self.assertTrue(torch.equal(one, two))
            with self.assertRaises(FileExistsError):
                team.save_checkpoint(path)

    def test_same_dimension_checkpoints_reject_cross_scope_even_when_empty(self):
        team, encoder, _ = fixture()
        individual = NeuralCostAgent(team.config, encoder, seed=8)
        with tempfile.TemporaryDirectory() as directory:
            team_path, individual_path = Path(directory) / "team.pt", Path(directory) / "individual.pt"
            team.save_checkpoint(team_path)
            individual.save_checkpoint(individual_path)
            with self.assertRaises(ValueError):
                TeamNeuralCostAgent.load_checkpoint(individual_path, encoder)
            with self.assertRaises(ValueError):
                NeuralCostAgent.load_checkpoint(team_path, encoder)
            state = torch.load(team_path, weights_only=True)
            state.pop("learning_return_contract")
            missing = Path(directory) / "missing-scope.pt"
            torch.save(state, missing)
            with self.assertRaisesRegex(ValueError, "scope"):
                TeamNeuralCostAgent.load_checkpoint(missing, encoder)
            state = torch.load(team_path, weights_only=True)
            state["replay"] = individual.replay.state_dict()
            wrong = Path(directory) / "wrong-replay.pt"
            torch.save(state, wrong)
            with self.assertRaises(ValueError):
                TeamNeuralCostAgent.load_checkpoint(wrong, encoder)
