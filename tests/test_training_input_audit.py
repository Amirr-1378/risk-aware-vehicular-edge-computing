"""Training-only numerical diagnostics preserve samples, learners and files."""

import importlib.util
import json
import math
import random
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import hashlib
    import numpy
    import torch
    from experiments.run_resumable_development import run_resumable_development
    from experiments.run_return_scope_audit import _same_state, _training_state
    from experiments.run_training_input_audit import audit_agent_inputs, run_training_input_audit
    from risk_aware_v2x.agents import DecisionTransition, EncodedObservation, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.replay import ReplayBuffer
    from risk_aware_v2x.environment import SlottedEnvironment
    from test_return_scope_audit import fixture_plans


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for training input audits")
class TrainingInputNumericsTests(unittest.TestCase):
    def fixture(self, *, gamma=1.):
        encoder = ObservationEncoder(max_server_vehicles=1, scales=FeatureScales())
        config = AgentConfig(hidden_sizes=(), num_quantiles=2, batch_size=1, learning_starts=1,
                             replay_capacity=4, gamma_per_slot=gamma)
        agent = NeuralCostAgent(config, encoder, seed=8)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[0].weight[0, 0] = 2.
            agent.online.layers[0].weight[1, encoder.feature_names.index("server[0].snr")] = 1.
        return agent, encoder

    def observation(self, encoder, *, task=1, first=3., snr=4., valid=False):
        values = [0.] * encoder.dimension
        values[0] = first
        values[encoder.feature_names.index("server[0].present")] = 1.
        values[encoder.feature_names.index("server[0].snr")] = snr
        return EncodedObservation(3, tuple(values), (True, valid), 1, task)

    def terminal(self, observation):
        return DecisionTransition(observation, 0, 0, 1, 1., 1., 0., None, True)

    def test_input_mass_and_first_layer_contributions_by_hand_keep_masked_inputs(self):
        agent, encoder = self.fixture()
        samples = (self.terminal(self.observation(encoder)),
                   self.terminal(self.observation(encoder, task=2, first=0.)))
        before = _training_state(agent)
        report = audit_agent_inputs(agent, samples, encoder.feature_names)
        # SNR squares32; feature0 squares9; presence squares2. Invalid SV's
        # received values remain inputs to the shared fully connected network.
        geometry = report["input_geometry"]["current"]
        self.assertAlmostEqual(geometry["snr_squared_mass_share"], 32. / 43.)
        # Absolute weighted contributions: feature0=3*2=6, SNR=(4+4)*1=8.
        self.assertAlmostEqual(geometry["snr_first_linear_absolute_contribution_share"], 4. / 7.)
        self.assertEqual(report["server_presence"]["current"],
                         [{"slot": 0, "present_count": 2, "action_valid_count": 0}])
        self.assertEqual(report["predictions"]["valid_action_atoms"]["count"], 4)
        self.assertEqual(report["predictions"]["valid_action_atoms"]["max_abs"], 6.)
        self.assertEqual(report["targets"]["terminal_or_zero_discount"]["count"], 4)
        self.assertEqual(report["targets"]["all"]["max_abs"], 1.)
        self.assertTrue(_same_state(before, _training_state(agent)))

    def test_terminal_and_zero_discount_next_states_are_excluded_from_feature_stats(self):
        agent, encoder = self.fixture(gamma=0.)
        observation = self.observation(encoder)
        unused_next = self.observation(encoder, task=2, snr=1e20)
        nonterminal = DecisionTransition(observation, 0, 0, 1, 2., 2., 0., unused_next, False)
        report = audit_agent_inputs(agent, (self.terminal(observation), nonterminal), encoder.feature_names)
        self.assertEqual(report["next_bootstrap_count"], 0)
        self.assertTrue(all(item["count"] == 0 for item in report["next_bootstrap_features"]))
        self.assertEqual(report["targets"]["bootstrap_active"]["count"], 0)
        self.assertEqual(report["targets"]["all"]["max_abs"], 2.)

    def test_bootstrap_target_uses_existing_masked_target_network(self):
        agent, encoder = self.fixture()
        observation = self.observation(encoder)
        with torch.no_grad():
            for parameter in agent.target.parameters():
                parameter.zero_()
            # The invalid SV's tempting negative outputs must be ignored.
            agent.target.layers[-1].bias.copy_(torch.tensor([3., 3., -1e6, -1e6]))
        following = replace(observation, task_id=2)
        sample = DecisionTransition(observation, 0, 0, 1, 2., 2., 1., following, False)
        report = audit_agent_inputs(agent, (sample,), encoder.feature_names)
        self.assertEqual(report["next_bootstrap_count"], 1)
        self.assertEqual(report["targets"]["bootstrap_active"]["min"], 5.)
        self.assertEqual(report["targets"]["bootstrap_active"]["max"], 5.)

    def test_empty_replay_does_not_compute_empty_targets(self):
        agent, encoder = self.fixture()
        with patch.object(agent, "targets", side_effect=AssertionError("empty targets forbidden")):
            report = audit_agent_inputs(agent, (), encoder.feature_names)
        self.assertEqual(report["sample_count"], 0)
        self.assertIsNone(report["fixed_cohort_loss"])
        self.assertIsNone(report["input_geometry"]["current"]["snr_squared_mass_share"])
        self.assertEqual(report["predictions"]["valid_action_atoms"]["count"], 0)

    def test_float32_overflow_is_rejected_without_mutating_agent(self):
        agent, encoder = self.fixture()
        sample = self.terminal(self.observation(encoder, snr=1e100))
        before = _training_state(agent)
        settings = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
        with self.assertRaises((ValueError, OverflowError)):
            audit_agent_inputs(agent, (sample,), encoder.feature_names)
        self.assertTrue(_same_state(before, _training_state(agent)))
        self.assertEqual(settings, (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()))

    def test_zero_input_or_zero_weights_have_undefined_mass_shares(self):
        agent, encoder = self.fixture()
        zero = EncodedObservation(3, (0.,) * encoder.dimension, (True, False), 1, 1)
        result = audit_agent_inputs(agent, (self.terminal(zero),), encoder.feature_names)
        geometry = result["input_geometry"]["current"]
        self.assertIsNone(geometry["snr_squared_mass_share"])
        self.assertIsNone(geometry["snr_first_linear_absolute_contribution_share"])
        self.assertTrue(all(item["share"] is None for item in geometry["squared_mass_shares"]))
        with torch.no_grad():
            agent.online.layers[0].weight.zero_()
        result = audit_agent_inputs(agent, (self.terminal(self.observation(encoder)),), encoder.feature_names)
        geometry = result["input_geometry"]["current"]
        self.assertIsNotNone(geometry["snr_squared_mass_share"])
        self.assertIsNone(geometry["snr_first_linear_absolute_contribution_share"])

    def test_fixed_cohort_loss_weights_partial_last_batch_by_sample_count(self):
        agent, encoder = self.fixture()
        agent.config = replace(agent.config, batch_size=2)
        samples = tuple(self.terminal(self.observation(encoder, task=index + 1, first=first, snr=snr))
                        for index, (first, snr) in enumerate(((0., 0.), (0., 0.), (3., 4.))))
        # Quantile losses for the three terminal samples are .5,.5,4.0.
        # Averaging batch means without weighting would incorrectly give2.25.
        result = audit_agent_inputs(agent, samples, encoder.feature_names)
        self.assertAlmostEqual(result["fixed_cohort_loss"], 5. / 3.)


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for training session audits")
class TrainingInputSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        paired, cls.scenarios = fixture_plans()
        cls.learning = replace(paired.learning, algorithms=("qr_dqn_cvar",), agent_seeds=(11,),
                               validation_profiles=("nominal",), validation_cases=1)
        cls.session = cls.root / "original"
        run_resumable_development(output_directory=cls.session, learning=cls.learning,
                                  scenarios=cls.scenarios, max_additional_episodes=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def snapshot(self, session):
        return {path.relative_to(session).as_posix(): path.read_bytes()
                for path in session.rglob("*") if path.is_file()}

    def test_audit_forbids_training_sampling_simulation_and_preserves_rng_and_files(self):
        before = self.snapshot(self.session)
        python_rng, numpy_rng, torch_rng = random.getstate(), numpy.random.get_state(), torch.get_rng_state().clone()
        settings = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
        forbidden = AssertionError("read-only audit must not train/sample/step")
        with patch.object(NeuralCostAgent, "act", side_effect=forbidden), \
                patch.object(NeuralCostAgent, "observe", side_effect=forbidden), \
                patch.object(NeuralCostAgent, "learn", side_effect=forbidden), \
                patch.object(ReplayBuffer, "sample", side_effect=forbidden), \
                patch.object(torch.optim.Adam, "step", side_effect=forbidden), \
                patch.object(SlottedEnvironment, "step", side_effect=forbidden), \
                patch("risk_aware_v2x.scenarios.generate_scenario", side_effect=forbidden):
            report = run_training_input_audit(session_directory=self.session)
        self.assertEqual(before, self.snapshot(self.session))
        self.assertEqual(python_rng, random.getstate())
        current_numpy = numpy.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        self.assertTrue(numpy.array_equal(numpy_rng[1], current_numpy[1]))
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(settings, (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                                    torch.is_deterministic_algorithms_warn_only_enabled()))
        self.assertEqual(report["replay"]["retained_transitions"], 12)
        self.assertTrue(report["replay"]["complete_training_replay"])
        self.assertEqual(report["initial_network"]["sample_count"], 12)
        self.assertEqual(report["final_network"]["sample_count"], 12)
        self.assertEqual(len(report["training_update_trace"]), 2)
        for flag in ("training_executed", "test_split_accessed", "validation_analytics_used", "feature_scales_changed"):
            self.assertIs(report["flags"][flag], False)
        self.assertTrue(all(report["mutation_checks"].values()))

    def test_output_exclusive_and_outside_session(self):
        path = self.root / "new-diagnostic.json"
        report = run_training_input_audit(session_directory=self.session, output_path=path)
        saved = path.read_bytes()
        self.assertEqual(json.loads(saved), json.loads(json.dumps(report)))
        with self.assertRaises(FileExistsError):
            run_training_input_audit(session_directory=self.session, output_path=path)
        self.assertEqual(saved, path.read_bytes())
        with self.assertRaises(ValueError):
            run_training_input_audit(session_directory=self.session, output_path=self.session / "bad.json")
        self.assertFalse((self.session / "bad.json").exists())

    def test_corrupt_artifact_rejected_before_output(self):
        copy = self.root / "corrupt"
        shutil.copytree(self.session, copy)
        (copy / "steps" / "000002" / "agent.pt").write_bytes(b"truncated")
        output = self.root / "absent-corrupt.json"
        with self.assertRaises(ValueError):
            run_training_input_audit(session_directory=copy, output_path=output)
        self.assertFalse(output.exists())

    def test_checkpoint_identity_mismatch_rejected_before_output(self):
        original = NeuralCostAgent.load_checkpoint

        def wrong_identity(*args, **kwargs):
            agent, metadata = original(*args, **kwargs)
            return agent, {**metadata, "session_contract_sha256": "0" * 64}

        output = self.root / "absent-identity.json"
        with patch.object(NeuralCostAgent, "load_checkpoint", side_effect=wrong_identity):
            with self.assertRaises(ValueError):
                run_training_input_audit(session_directory=self.session, output_path=output)
        self.assertFalse(output.exists())

    def test_validation_values_never_enter_input_or_learning_statistics(self):
        report = run_training_input_audit(session_directory=self.session)
        copy = self.root / "validation-values"
        shutil.copytree(self.session, copy)
        final = copy / "steps" / "000003"
        path = final / "report.json"
        record = json.loads(path.read_text())
        record["validation"]["nominal"][0]["metrics"]["total_cost"] = 1e250
        path.write_text(json.dumps(record), encoding="utf-8")
        commit_path = final / "commit.json"
        commit = json.loads(commit_path.read_text())
        commit["artifacts"]["report.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        commit_path.write_text(json.dumps(commit), encoding="utf-8")
        # This is a temporary test journal with a newly hashed final report.
        # Changing validation outcomes must not change training-only analytics.
        changed = run_training_input_audit(session_directory=copy)
        for key in ("initial_network", "final_network", "replay", "training_update_trace"):
            self.assertEqual(report[key], changed[key])

    def test_retained_ring_window_is_not_reported_as_all_training_observations(self):
        output = self.root / "small-replay"
        learning = replace(self.learning, common_agent=replace(self.learning.common_agent, replay_capacity=2))
        run_resumable_development(output_directory=output, learning=learning,
                                  scenarios=self.scenarios, max_additional_episodes=2)
        report = run_training_input_audit(session_directory=output)
        self.assertEqual(report["replay"]["seen_transitions"], 12)
        self.assertEqual(report["replay"]["retained_transitions"], 2)
        self.assertFalse(report["replay"]["complete_training_replay"])
        self.assertEqual(report["final_network"]["sample_count"], 2)

    def test_initialized_session_has_empty_training_statistics(self):
        output = self.root / "initialized"
        with patch("experiments.run_resumable_development._episode", side_effect=RuntimeError("before first episode")):
            with self.assertRaises(RuntimeError):
                run_resumable_development(output_directory=output, learning=self.learning, scenarios=self.scenarios)
        report = run_training_input_audit(session_directory=output)
        self.assertEqual(report["training_update_trace"], [])
        self.assertEqual(report["replay"]["retained_transitions"], 0)
        self.assertIsNone(report["final_network"]["fixed_cohort_loss"])


if __name__ == "__main__":
    unittest.main()
