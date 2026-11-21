"""Explicit encoder meaning guards preserve legacy and opt-in replay state."""

import copy
import importlib.util
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from risk_aware_v2x.agents import (
    DecisionTransition, EncodedObservation, FeatureScales, LogSNRObservationEncoder,
    ObservationEncoder, ReplayBuffer,
)
from risk_aware_v2x.agents.replay import (
    transition_from_record, transition_record, validate_observation, validate_transition,
)
from risk_aware_v2x.agents.team_replay import TeamReplayBuffer, team_transition_record
from risk_aware_v2x.agents.team_returns import TeamDecisionTransition

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent, encoder_signature
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent


def encoders():
    return (ObservationEncoder(max_server_vehicles=1, scales=FeatureScales()),
            LogSNRObservationEncoder(max_server_vehicles=1, scales=FeatureScales()))


def fixture(encoder):
    # At the declared raw reference SNR, both representations equal one. Even
    # identical numeric vectors must retain distinct, explicit schema meaning.
    values = [0.] * encoder.dimension
    values[0] = 1.
    values[encoder.feature_names.index("server[0].present")] = 1.
    snr_name = "server[0].snr" if encoder.schema_version == 3 else "server[0].snr_log1p"
    values[encoder.feature_names.index(snr_name)] = 1.
    observation = EncodedObservation(encoder.schema_version, tuple(values), (True, True), 1, 1)
    transition = DecisionTransition(observation, 0, 0, 2, 3., 2.9, .81,
                                    replace(observation, task_id=2), False)
    return observation, transition


class EncoderReplayCompatibilityTests(unittest.TestCase):
    def test_schema_versions_are_explicit_integers_only(self):
        observation, _ = fixture(encoders()[0])
        for version in (3, 4):
            validate_observation(replace(observation, schema_version=version))
        for version in (True, False, 3., 4., 1, 2, 5, None, "4"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_observation(replace(observation, schema_version=version))

    def test_transition_cannot_mix_current_and_next_schema(self):
        for encoder in encoders():
            _, sample = fixture(encoder)
            validate_transition(sample)
            other = 4 if encoder.schema_version == 3 else 3
            with self.assertRaisesRegex(ValueError, "schema"):
                validate_transition(replace(sample, next_observation=replace(sample.next_observation, schema_version=other)))
            record = transition_record(sample)
            record["next_observation"]["schema_version"] = other
            with self.assertRaisesRegex(ValueError, "schema"):
                transition_from_record(record)

    def test_v3_records_and_container_keys_are_preserved_and_v4_round_trips(self):
        expected_keys = {"schema_version", "cost_model_version", "capacity", "next_index", "items", "rng_state"}
        for encoder in encoders():
            _, sample = fixture(encoder)
            replay = ReplayBuffer(capacity=4, seed=17)
            replay.append(sample)
            self.assertEqual(transition_record(sample), asdict(sample))
            self.assertEqual(transition_from_record(transition_record(sample)), sample)
            state = replay.state_dict()
            self.assertEqual(set(state), expected_keys)
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["items"], [asdict(sample)])
            restored = ReplayBuffer.from_state_dict(state)
            self.assertEqual(restored.state_dict(), state)
            self.assertEqual(restored.sample(1), replay.sample(1))

    def test_append_rejects_mixed_schema_before_ring_replacement(self):
        linear, logarithmic = (fixture(encoder)[1] for encoder in encoders())
        for first, other in ((linear, logarithmic), (logarithmic, linear)):
            replay = ReplayBuffer(capacity=1, seed=17)
            replay.append(first)
            before = replay.state_dict()
            with self.assertRaisesRegex(ValueError, "mix"):
                replay.append(other)
            self.assertEqual(before, replay.state_dict())

    def test_loading_mixed_replay_and_serializing_corrupt_replay_reject(self):
        linear, logarithmic = (fixture(encoder)[1] for encoder in encoders())
        replay = ReplayBuffer(capacity=4, seed=17)
        replay.append(linear)
        state = replay.state_dict()
        mixed = copy.deepcopy(state)
        mixed["items"].append(transition_record(logarithmic))
        mixed["next_index"] = 2
        with self.assertRaisesRegex(ValueError, "mix"):
            ReplayBuffer.from_state_dict(mixed)
        replay._items.append(logarithmic)
        with self.assertRaisesRegex(ValueError, "mix"):
            replay.state_dict()

    def test_team_replay_retains_container_contract_and_rejects_mixed_schemas(self):
        samples = [TeamDecisionTransition(fixture(encoder)[1], 3., 2.9, .9) for encoder in encoders()]
        expected_keys = {"schema_version", "cost_model_version", "team_return_contract",
                         "capacity", "next_index", "items", "rng_state"}
        for first, other in ((samples[0], samples[1]), (samples[1], samples[0])):
            replay = TeamReplayBuffer(capacity=4, seed=17)
            replay.append(first)
            state = replay.state_dict()
            self.assertEqual(set(state), expected_keys)
            self.assertEqual(state["schema_version"], 1)
            self.assertEqual(TeamReplayBuffer.from_state_dict(state).state_dict(), state)
            with self.assertRaisesRegex(ValueError, "mix"):
                replay.append(other)
            self.assertEqual(replay.state_dict(), state)
            mixed = copy.deepcopy(state)
            mixed["items"].append(team_transition_record(other))
            mixed["next_index"] = 2
            with self.assertRaisesRegex(ValueError, "mix"):
                TeamReplayBuffer.from_state_dict(mixed)
            replay._items.append(other)
            with self.assertRaisesRegex(ValueError, "mix"):
                replay.state_dict()


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for encoder checkpoint guards")
class EncoderNeuralCompatibilityTests(unittest.TestCase):
    def config(self, algorithm="qr_dqn_cvar", **changes):
        return AgentConfig(**{"algorithm": algorithm, "hidden_sizes": (8,), "num_quantiles": 5,
            "batch_size": 2, "learning_starts": 2, "replay_capacity": 16,
            "target_update_interval": 2, "gamma_per_slot": .9, **changes})

    def test_legacy_signature_is_exact_and_new_contract_is_explicit(self):
        linear, logarithmic = encoders()
        self.assertEqual(encoder_signature(linear), {
            "schema_version": 3, "dimension": linear.dimension, "max_server_vehicles": 1,
            "feature_names": linear.feature_names, "scales": asdict(linear.scales)})
        new = encoder_signature(logarithmic)
        self.assertEqual(new["schema_version"], 4)
        self.assertEqual(new["dimension"], linear.dimension)
        self.assertEqual(new["representation_contract"], logarithmic.representation_contract())
        self.assertNotEqual(new, encoder_signature(linear))

    def test_representation_does_not_change_initialization_rng_or_network_dimensions(self):
        linear, logarithmic = encoders()
        before = torch.get_rng_state().clone()
        first = NeuralCostAgent(self.config(), linear, seed=17)
        second = NeuralCostAgent(self.config(), logarithmic, seed=17)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first.input_dimension, second.input_dimension)
        for name, value in first.online.state_dict().items():
            self.assertTrue(torch.equal(value, second.online.state_dict()[name]))
        self.assertEqual(first.replay.state_dict(), second.replay.state_dict())

    def test_cross_schema_input_is_rejected_before_exploration_or_learning(self):
        for encoder in encoders():
            observation, sample = fixture(encoder)
            agent = NeuralCostAgent(self.config(), encoder, seed=17)
            other_version = 4 if encoder.schema_version == 3 else 3
            wrong = replace(observation, schema_version=other_version)
            with self.assertRaisesRegex(ValueError, "schema"):
                agent.act(wrong, epsilon=1.)
            self.assertEqual(agent._exploration, {})
            wrong_sample = replace(sample, observation=wrong,
                                   next_observation=replace(sample.next_observation, schema_version=other_version))
            with self.assertRaisesRegex(ValueError, "schema"):
                agent.observe(wrong_sample)
            with self.assertRaisesRegex(ValueError, "schema"):
                agent.targets((sample, wrong_sample))
            self.assertEqual(agent.seen_transitions, 0)
            self.assertEqual(len(agent.replay), 0)

    def test_empty_and_populated_checkpoints_reject_cross_encoder_both_directions(self):
        linear, logarithmic = encoders()
        with tempfile.TemporaryDirectory() as directory:
            for encoder, other in ((linear, logarithmic), (logarithmic, linear)):
                agent = NeuralCostAgent(self.config(), encoder, seed=17)
                for populated in (False, True):
                    if populated:
                        _, sample = fixture(encoder)
                        agent.observe(sample)
                    path = Path(directory) / f"schema{encoder.schema_version}-{populated}.pt"
                    agent.save_checkpoint(path)
                    before = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "encoder"):
                        NeuralCostAgent.load_checkpoint(path, other)
                    self.assertEqual(path.read_bytes(), before)
                    state = torch.load(path, weights_only=True)
                    self.assertEqual(state["checkpoint_schema_version"], 2)

    def test_same_encoder_checkpoint_preserves_next_update_rng_optimizer_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            for encoder in encoders():
                for algorithm in ("qr_dqn", "qr_dqn_cvar"):
                    observation, sample = fixture(encoder)
                    agent = NeuralCostAgent(self.config(algorithm), encoder, seed=17)
                    terminal = replace(sample, terminated=True, next_observation=None, bootstrap_discount=0.)
                    agent.observe(sample)
                    agent.observe(terminal)
                    agent.act(observation, epsilon=.7)
                    agent.learn()
                    path = Path(directory) / f"schema{encoder.schema_version}-{algorithm}.pt"
                    agent.save_checkpoint(path, run_metadata={"variant": encoder.schema_version})
                    restored, metadata = NeuralCostAgent.load_checkpoint(path, encoder)
                    self.assertEqual(metadata, {"variant": encoder.schema_version})
                    self.assertEqual(restored.replay.state_dict(), agent.replay.state_dict())
                    torch.testing.assert_close(restored.targets((sample, terminal)), agent.targets((sample, terminal)), rtol=0., atol=0.)
                    self.assertEqual(restored.act(observation, epsilon=.7), agent.act(observation, epsilon=.7))
                    self.assertEqual(restored.learn(), agent.learn())
                    for key, value in restored.online.state_dict().items():
                        self.assertTrue(torch.equal(value, agent.online.state_dict()[key]))

    def test_log_encoder_mask_and_beta_do_not_change_greedy_selector_definition(self):
        _, encoder = encoders()
        observation, _ = fixture(encoder)
        first = NeuralCostAgent(self.config(sigmoid_beta=1.), encoder, seed=17)
        second = NeuralCostAgent(self.config(sigmoid_beta=100.), encoder, seed=17)
        for agent in (first, second):
            with torch.no_grad():
                for parameter in agent.online.parameters():
                    parameter.zero_()
                agent.online.layers[-1].bias.copy_(torch.tensor([2.] * 5 + [1.] * 5))
        self.assertEqual(first.act(observation), 1)
        self.assertEqual(second.act(observation), 1)
        self.assertEqual(first.preference_diagnostics(observation)["selected_action"], 1)
        masked = replace(observation, action_mask=(True, False))
        self.assertEqual(first.act(masked), 0)
        self.assertEqual(second.act(masked), 0)

    def test_team_agent_uses_log_encoder_with_scope_and_schema_guards(self):
        linear, logarithmic = encoders()
        observation, sample = fixture(logarithmic)
        agent = TeamNeuralCostAgent(self.config(), logarithmic, seed=17)
        team = TeamDecisionTransition(sample, sample.cost, sample.discounted_cost, .9)
        terminal_source = replace(sample, terminated=True, next_observation=None, bootstrap_discount=0.)
        terminal = TeamDecisionTransition(terminal_source, terminal_source.cost, terminal_source.discounted_cost, .9)
        agent.observe(team)
        agent.observe(terminal)
        self.assertIsNotNone(agent.learn())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "team-log.pt"
            agent.save_checkpoint(path)
            restored, _ = TeamNeuralCostAgent.load_checkpoint(path, logarithmic)
            self.assertEqual(restored.replay.state_dict(), agent.replay.state_dict())
            with self.assertRaisesRegex(ValueError, "encoder"):
                TeamNeuralCostAgent.load_checkpoint(path, linear)
            with self.assertRaisesRegex(ValueError, "schema"):
                restored.act(replace(observation, schema_version=3))


if __name__ == "__main__":
    unittest.main()
