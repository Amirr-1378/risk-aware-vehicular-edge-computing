"""Stable single preference curve and beta-independent discrete QR control."""

import copy
import importlib.util
import json
import math
import random
import unittest
from dataclasses import replace

from risk_aware_v2x.risk import sigmoid_mec_preference

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from risk_aware_v2x.agents import EncodedObservation, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent
    from test_neural_agents import constant_output, neural_fixture


class SigmoidCurveTests(unittest.TestCase):
    def test_direction_midpoint_monotonicity_and_same_function_for_all_servers(self):
        values = [sigmoid_mec_preference(delta, beta=2.) for delta in (-4., -1., 0., 1., 4.)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], .01)
        self.assertEqual(values[2], .5)
        self.assertGreater(values[-1], .99)
        self.assertAlmostEqual(values[0] + values[-1], 1.)
        self.assertAlmostEqual(sigmoid_mec_preference(1., beta=2.), 1. / (1. + math.exp(-2.)))

    def test_extreme_gaps_and_product_overflow_saturate_without_exp_overflow(self):
        for magnitude in (1_000., 1e308):
            for beta in (1., 1e308):
                self.assertEqual(sigmoid_mec_preference(magnitude, beta=beta), 1.)
                self.assertEqual(sigmoid_mec_preference(-magnitude, beta=beta), 0.)
        self.assertEqual(sigmoid_mec_preference(0., beta=1e308), .5)
        self.assertEqual(sigmoid_mec_preference(-0., beta=1e308), .5)

    def test_beta_changes_steepness_not_center_and_no_global_rng_is_used(self):
        before = random.getstate()
        self.assertLess(sigmoid_mec_preference(1., beta=.1), sigmoid_mec_preference(1., beta=10.))
        self.assertGreater(sigmoid_mec_preference(-1., beta=.1), sigmoid_mec_preference(-1., beta=10.))
        for beta in (.01, 1., 100.):
            self.assertEqual(sigmoid_mec_preference(0., beta=beta), .5)
        self.assertEqual(before, random.getstate())

    def test_invalid_beta_or_gap_is_rejected(self):
        for beta in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(beta=beta), self.assertRaises((TypeError, ValueError)):
                sigmoid_mec_preference(1., beta=beta)
        for delta in (float("nan"), float("inf"), True):
            with self.subTest(delta=delta), self.assertRaises((TypeError, ValueError)):
                sigmoid_mec_preference(delta)


@unittest.skipUnless(HAS_TORCH, "install learning dependencies for learned preference tests")
class LearnedPreferenceTests(unittest.TestCase):
    def fixture(self, *, beta=1.):
        encoder = ObservationEncoder(max_server_vehicles=3, scales=FeatureScales())
        agent = NeuralCostAgent(AgentConfig(hidden_sizes=(), num_quantiles=10, sigmoid_beta=beta), encoder, seed=11)
        observation = EncodedObservation(3, (0.,) * encoder.dimension, (True, True, True, False), 1, 1)
        # MEC J=12; SV1 mean=10 but upper tail=100 => J=55; SV2 J=8.
        constant_output(agent.online, [[12.] * 10, [0.] * 9 + [100.], [8.] * 10, [-1e6] * 10])
        return agent, observation

    def test_multiple_servers_mask_upper_cost_tail_and_preference_sign(self):
        agent, observation = self.fixture()
        report = agent.preference_diagnostics(observation)
        self.assertEqual(agent.act(observation), 2)
        self.assertEqual(report["selected_action"], 2)
        self.assertEqual([item["action"] for item in report["valid_actions"]], [0, 1, 2])
        mec, risky, safe = report["valid_actions"]
        self.assertAlmostEqual(risky["mean_cost"], 10.)
        self.assertAlmostEqual(risky["upper_cvar_cost"], 100.)
        self.assertAlmostEqual(risky["risk_cost"], 55.)
        self.assertAlmostEqual(risky["delta_vs_mec"], 43.)
        self.assertGreater(risky["mec_preference_score"], .5)
        self.assertAlmostEqual(safe["delta_vs_mec"], -4.)
        self.assertLess(safe["mec_preference_score"], .5)
        self.assertIsNone(mec["mec_preference_score"])
        json.dumps(report, allow_nan=False)

    def test_beta_does_not_change_ranking_action_or_exploration(self):
        reports, decisions = [], []
        for beta in (1e-6, 1., 1e308):
            agent, observation = self.fixture(beta=beta)
            report = agent.preference_diagnostics(observation)
            reports.append([item["risk_cost"] for item in report["valid_actions"]])
            self.assertEqual(report["selected_action"], 2)
            self.assertEqual(agent.act(observation), 2)
            decisions.append([agent.act(observation, epsilon=1.) for _ in range(20)])
        self.assertTrue(all(row == reports[0] for row in reports))
        self.assertTrue(all(row == decisions[0] for row in decisions))

    def test_sigmoid_saturation_cannot_cause_a_tie_in_original_score_selection(self):
        agent, observation = self.fixture(beta=1e308)
        constant_output(agent.online, [[12.] * 10, [8.] * 10, [7.] * 10, [-1e6] * 10])
        report = agent.preference_diagnostics(observation)
        self.assertEqual([item["mec_preference_score"] for item in report["valid_actions"]][1:], [0., 0.])
        self.assertEqual(agent.act(observation), 2)

    def test_ties_use_lowest_action_index_and_absent_servers_force_mec(self):
        agent, observation = self.fixture()
        constant_output(agent.online, [[5.] * 10] * 4)
        self.assertEqual(agent.act(observation), 0)
        report = agent.preference_diagnostics(observation)
        self.assertEqual(report["valid_actions"][1]["mec_preference_score"], .5)
        only_mec = replace(observation, action_mask=(True, False, False, False))
        self.assertEqual(agent.act(only_mec), 0)
        self.assertEqual(len(agent.preference_diagnostics(only_mec)["valid_actions"]), 1)
        constant_output(agent.online, [[5.] * 10, [float("nan")] * 10, [float("inf")] * 10, [-1e6] * 10])
        self.assertEqual(agent.preference_diagnostics(only_mec)["selected_action"], 0)

    def test_preference_diagnostic_preserves_weights_replay_rng_and_optimizer(self):
        agent, observation = self.fixture()
        weights = copy.deepcopy(agent.online.state_dict())
        replay = agent.replay.state_dict()
        python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        optimizer = copy.deepcopy(agent.optimizer.state_dict())
        for _ in range(3):
            self.assertEqual(agent.preference_diagnostics(observation)["selected_action"], 2)
        self.assertEqual(replay, agent.replay.state_dict())
        self.assertEqual(optimizer, agent.optimizer.state_dict())
        self.assertEqual(agent._exploration, {})
        self.assertEqual(agent.seen_transitions, 0)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for name, value in weights.items():
            self.assertTrue(torch.equal(value, agent.online.state_dict()[name]))

    def test_beta_never_enters_td_targets_loss_or_quantile_update(self):
        agents = []
        for beta in (.01, 100.):
            agent, _, sample = neural_fixture("qr_dqn_cvar", sigmoid_beta=beta)
            agent.observe(sample)
            agent.observe(replace(sample, action=1))
            targets = agent.targets((sample,))
            update = agent.learn()
            agents.append((agent, targets, update))
        first, second = agents
        torch.testing.assert_close(first[1], second[1], rtol=0., atol=0.)
        self.assertEqual(first[2], second[2])
        for a, b in zip(first[0].online.parameters(), second[0].online.parameters(), strict=True):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(first[0].config.risk_weight, .5)
