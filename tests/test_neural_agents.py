"""CPU network learning, target synchronization, RNG isolation, and checkpoints."""

import importlib.util
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from risk_aware_v2x.agents import DecisionTransition, EncodedObservation, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent


def neural_fixture(algorithm="dqn", **changes):
    encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
    config = AgentConfig(**{"algorithm": algorithm, "hidden_sizes": (), "num_quantiles": 15,
                            "batch_size": 2, "learning_starts": 2, "replay_capacity": 64,
                            "target_update_interval": 2, "gamma_per_slot": .9, **changes})
    agent = NeuralCostAgent(config, encoder, seed=8)
    observation = EncodedObservation(1, (1.,) + (0.,) * (encoder.dimension - 1),
                                     (True, True, False), 1, 1)
    transition = DecisionTransition(observation, 0, 0, 2, 3., 2.9, .81,
                                    replace(observation, task_id=2), False)
    return agent, encoder, transition


def constant_output(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32).flatten())


@unittest.skipUnless(HAS_TORCH, "install learning dependencies for neural agent tests")
class NeuralAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_initialization_is_seeded_and_does_not_mutate_global_rngs(self):
        python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
        first, _, _ = neural_fixture()
        second, _, _ = neural_fixture()
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        for one, two in zip(first.online.parameters(), second.online.parameters(), strict=True):
            self.assertTrue(torch.equal(one, two))

    def test_masks_apply_to_exploration_and_greedy_inference(self):
        agent, _, sample = neural_fixture()
        constant_output(agent.online, [4., -1e6, -1e9])
        self.assertEqual(agent.act(sample.observation), 1)
        draws = [agent.act(sample.observation, epsilon=1.) for _ in range(30)]
        self.assertEqual(set(draws), {0, 1})
        with self.assertRaises(ValueError):
            agent.act(replace(sample.observation, action_mask=(False, False, False)))

    def test_no_exploration_rng_consumption_in_greedy_evaluation(self):
        agent, _, sample = neural_fixture()
        agent.act(sample.observation, epsilon=1.)
        before = {user_id: rng.getstate() for user_id, rng in agent._exploration.items()}
        replay_before = agent.replay.state_dict()
        for _ in range(5):
            agent.act(sample.observation, epsilon=0.)
        self.assertEqual(before, {user_id: rng.getstate() for user_id, rng in agent._exploration.items()})
        self.assertEqual(replay_before, agent.replay.state_dict())

    def test_user_exploration_streams_are_independent_of_interleaving(self):
        first, _, sample = neural_fixture()
        second, _, _ = neural_fixture()
        reference = [first.act(sample.observation, epsilon=1.) for _ in range(20)]
        actual = []
        for _ in range(20):
            second.act(replace(sample.observation, user_vehicle_id=99), epsilon=1.)
            actual.append(second.act(sample.observation, epsilon=1.))
        self.assertEqual(reference, actual)

    def test_agent_uses_semimarkov_discount_and_correct_double_q_network(self):
        for algorithm, expected in (("dqn", 4.52), ("double_dqn", 9.38)):
            agent, _, sample = neural_fixture(algorithm)
            constant_output(agent.online, [9., 1., -1e6])
            constant_output(agent.target, [2., 8., -1e6])
            self.assertAlmostEqual(float(agent.targets((sample,))[0, 0]), expected, places=5)

    def test_rejects_wrong_dimensions_gamma_and_terminal_bootstrap(self):
        agent, _, sample = neural_fixture()
        for changes in ({"bootstrap_discount": .9}, {"terminated": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                agent.observe(replace(sample, **changes))
        with self.assertRaises(ValueError):
            agent.act(replace(sample.observation, values=(1.,)))
        self.assertEqual(agent.seen_transitions, 0)

    def test_qr_target_selection_respects_risk_configuration_and_target_network(self):
        for algorithm in ("qr_dqn", "qr_dqn_cvar"):
            agent, _, sample = neural_fixture(algorithm)
            constant_output(agent.target, [[0.] * 14 + [150.], [12.] * 15, [-1e9] * 15])
            constant_output(agent.online, [[200.] * 15, [1.] * 15, [-1e9] * 15])
            expected = (2.9 + .81 * torch.tensor([[0.] * 14 + [150.]])) if algorithm == "qr_dqn" else torch.full((1, 15), 12.62)
            torch.testing.assert_close(agent.targets((sample,)), expected)

    def test_zero_risk_mixture_exactly_matches_risk_neutral_qr_updates(self):
        neutral, _, sample = neural_fixture("qr_dqn")
        ablated, _, _ = neural_fixture("qr_dqn_cvar", risk_weight=0.)
        terminal = replace(sample, terminated=True, next_observation=None, bootstrap_discount=0.)
        for agent in (neutral, ablated):
            agent.observe(sample)
            agent.observe(terminal)
        for _ in range(3):
            self.assertEqual(neutral.learn(), ablated.learn())
            self.assertEqual(neutral.act(sample.observation), ablated.act(sample.observation))
        for one, two in zip(neutral.online.parameters(), ablated.online.parameters(), strict=True):
            self.assertTrue(torch.equal(one, two))

    def test_gradient_clipping_limits_actual_gradient_norm(self):
        agent, _, sample = neural_fixture(max_grad_norm=.01)
        terminal = replace(sample, cost=100., discounted_cost=100., terminated=True,
                           next_observation=None, bootstrap_discount=0.)
        agent.observe(terminal)
        agent.observe(terminal)
        result = agent.learn()
        actual = sum(float(parameter.grad.square().sum()) for parameter in agent.online.parameters()) ** .5
        self.assertGreater(result.gradient_norm_before_clip, .01)
        self.assertLessEqual(actual, .010001)

    def test_warmup_target_freeze_and_hard_sync_by_gradient_steps(self):
        agent, _, sample = neural_fixture()
        self.assertIsNone(agent.learn())
        terminal = replace(sample, next_observation=None, terminated=True, bootstrap_discount=0.)
        agent.observe(terminal)
        self.assertIsNone(agent.learn())
        agent.observe(terminal)
        initial_target = {key: value.clone() for key, value in agent.target.state_dict().items()}
        first = agent.learn()
        self.assertFalse(first.target_synchronized)
        for key, value in agent.target.state_dict().items():
            self.assertTrue(torch.equal(value, initial_target[key]))
        second = agent.learn()
        self.assertTrue(second.target_synchronized)
        self.assertEqual(second.gradient_updates, 2)
        for online, target in zip(agent.online.parameters(), agent.target.parameters(), strict=True):
            self.assertTrue(torch.equal(online, target))
            self.assertFalse(target.requires_grad)
            self.assertIsNone(target.grad)

    def test_all_algorithms_fit_a_tiny_terminal_cost_problem(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                agent, _, sample = neural_fixture(algorithm, learning_rate=.03)
                low = replace(sample, cost=1., discounted_cost=1., terminated=True,
                              next_observation=None, bootstrap_discount=0.)
                high = replace(low, action=1, cost=3., discounted_cost=3.)
                agent.observe(low)
                agent.observe(high)
                initial = agent.learn().loss
                for _ in range(99):
                    final = agent.learn()
                self.assertLess(final.loss, initial * .2)
                self.assertEqual(agent.act(sample.observation), 0)

    def test_checkpoint_preserves_next_update_exploration_and_optimizer(self):
        agent, encoder, sample = neural_fixture("qr_dqn_cvar")
        terminal = replace(sample, next_observation=None, terminated=True, bootstrap_discount=0.)
        agent.observe(sample)
        agent.observe(terminal)
        for _ in range(4):
            agent.act(sample.observation, epsilon=1.)
        agent.learn()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            agent.save_checkpoint(path, run_metadata={"episode": 3})
            restored, metadata = NeuralCostAgent.load_checkpoint(path, encoder)
            self.assertEqual(metadata, {"episode": 3})
            self.assertEqual(agent.replay.state_dict(), restored.replay.state_dict())
            self.assertEqual(agent.seen_transitions, restored.seen_transitions)
            self.assertEqual(agent.gradient_updates, restored.gradient_updates)
            self.assertEqual(agent.act(sample.observation, epsilon=.7),
                             restored.act(sample.observation, epsilon=.7))
            self.assertEqual(agent.learn(), restored.learn())
            for one, two in zip(agent.online.parameters(), restored.online.parameters(), strict=True):
                self.assertTrue(torch.equal(one, two))
            with self.assertRaises(FileExistsError):
                agent.save_checkpoint(path)

    def test_checkpoint_rejects_different_encoder_scales_and_schema(self):
        agent, encoder, _ = neural_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            agent.save_checkpoint(path)
            incompatible = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales(input_bits=2e6))
            with self.assertRaises(ValueError):
                NeuralCostAgent.load_checkpoint(path, incompatible)
            state = torch.load(path, weights_only=True)
            state["checkpoint_schema_version"] = 99
            bad_path = Path(directory) / "unsupported.pt"
            torch.save(state, bad_path)
            with self.assertRaises(ValueError):
                NeuralCostAgent.load_checkpoint(bad_path, encoder)

    def test_configuration_validation_and_risk_neutral_ablation(self):
        for changes in ({"algorithm": "unknown"}, {"hidden_sizes": [32]},
                        {"num_quantiles": 1}, {"learning_rate": 0}, {"alpha": 1.},
                        {"replay_capacity": 1}, {"target_update_interval": True}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                AgentConfig(**changes)
        self.assertEqual(AgentConfig(algorithm="qr_dqn").effective_risk_weight, 0.)
        self.assertEqual(AgentConfig(algorithm="qr_dqn_cvar").effective_risk_weight, .5)
        self.assertEqual(AgentConfig(algorithm="dqn").atom_count, 1)

    def test_checkpoint_rejects_invalid_optimizer_parameters_and_nonfinite_state(self):
        agent, encoder, sample = neural_fixture()
        agent.observe(sample)
        agent.observe(sample)
        agent.learn()
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.pt"
            agent.save_checkpoint(original)
            bad = Path(directory) / "different-lr.pt"
            state = torch.load(original, weights_only=True)
            state["optimizer"]["param_groups"][0]["lr"] = 1.
            torch.save(state, bad)
            with self.assertRaises(ValueError):
                NeuralCostAgent.load_checkpoint(bad, encoder)
            state = torch.load(original, weights_only=True)
            for parameter_state in state["optimizer"]["state"].values():
                parameter_state["exp_avg"].fill_(float("nan"))
            invalid = Path(directory) / "invalid-moments.pt"
            torch.save(state, invalid)
            with self.assertRaises(ValueError):
                NeuralCostAgent.load_checkpoint(invalid, encoder)
