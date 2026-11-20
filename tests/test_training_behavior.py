"""Causal teacher collection is training-only and cannot replace real TD costs."""

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(x) is not None for x in ("torch", "numpy"))
if HAS_LEARNING:
    from experiments.comparison_policies import comparison_policies
    from experiments.learning_runtime import deterministic_cpu, run_episode
    from experiments.run_environment_smoke import build_smoke_scenario
    from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
    from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent
    from risk_aware_v2x.agents.schedules import EpsilonSchedule
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
    from risk_aware_v2x.agents.team_returns import TeamReturnAdapter


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for behavior-boundary tests")
class TrainingBehaviorTests(unittest.TestCase):
    def setup_run(self, team=False):
        env, _ = build_smoke_scenario()
        encoder = ObservationEncoder(max_server_vehicles=2, scales=FeatureScales())
        config = AgentConfig(hidden_sizes=(8,), num_quantiles=5, batch_size=64, learning_starts=64, replay_capacity=128)
        agent = (TeamNeuralCostAgent if team else NeuralCostAgent)(config, encoder, seed=7)
        adapter = (TeamReturnAdapter if team else DecisionAdapter)(env, encoder)
        policies, _ = comparison_policies(env.config, env._templates[0], env._templates[1], seed=17)
        return adapter, agent, policies["minimum_estimated_combined_cost"]

    def test_teacher_is_rejected_in_evaluation_before_reset_rng_or_actions(self):
        adapter, agent, teacher = self.setup_run()
        with patch.object(adapter, "reset", side_effect=AssertionError("reset forbidden")), patch.object(teacher, "act") as act:
            with self.assertRaisesRegex(ValueError, "evaluation"):
                run_episode(adapter, agent, environment_seed=7, training=False, schedule=EpsilonSchedule(),
                            training_slots=0, behavior_policy=teacher)
            act.assert_not_called()
        self.assertEqual(agent.seen_transitions, 0)
        self.assertEqual(agent._exploration, {})

    def test_unknown_behavior_type_is_rejected_before_reset(self):
        adapter, agent, _ = self.setup_run()
        with patch.object(adapter, "reset", side_effect=AssertionError("reset forbidden")):
            with self.assertRaises(TypeError):
                run_episode(adapter, agent, environment_seed=7, training=True, schedule=EpsilonSchedule(),
                            training_slots=0, behavior_policy=object())

    def test_teacher_receives_exact_local_raw_inputs_and_no_learner_exploration(self):
        adapter, agent, teacher = self.setup_run()
        original = teacher.act
        seen = []

        def local_only(raw):
            self.assertIs(raw, adapter.raw_observations[raw.user_vehicle_id])
            seen.append(raw)
            return original(raw)

        with deterministic_cpu(), patch.object(teacher, "act", side_effect=local_only), patch.object(agent, "act", side_effect=AssertionError("no learner behavior during teacher phase")):
            report, clock = run_episode(adapter, agent, environment_seed=7, training=True,
                schedule=EpsilonSchedule(), training_slots=0, behavior_policy=teacher)
        self.assertEqual(len(seen), 6)
        self.assertEqual(agent._exploration, {})
        self.assertEqual(clock, report["elapsed_slots"])
        self.assertEqual(report["training_behavior"]["actions"], 6)
        self.assertFalse(report["training_behavior"]["scheduled_epsilon_applied"])
        for action in report["actions"]:
            self.assertEqual(action["epsilon"], 0.)
            self.assertGreater(action["scheduled_epsilon"], 0.)
            self.assertFalse(action["scheduled_epsilon_applied"])
        # Teacher contact estimates never replace the received learner masks.
        self.assertEqual([item.observation.action_mask for item in agent.replay],
                         [adapter.encoder.encode(raw).action_mask for raw in seen])

    def test_projected_teacher_cost_cannot_enter_individual_or_team_replay(self):
        for team in (False, True):
            adapter, agent, teacher = self.setup_run(team)
            original = teacher.estimate
            with deterministic_cpu(), patch.object(teacher, "estimate", side_effect=lambda raw, action: replace(original(raw, action), cost=1e9 + action)):
                report, _ = run_episode(adapter, agent, environment_seed=7, training=True,
                    schedule=EpsilonSchedule(), training_slots=0, behavior_policy=teacher)
            self.assertEqual(agent.seen_transitions, 6)
            self.assertTrue(all(item.cost < 1e6 for item in agent.replay))
            self.assertAlmostEqual(report["metrics"]["total_cost"], sum(report["accounting"]["total_cost"].values()))
            if team:
                self.assertAlmostEqual(report["team_accounting"]["total_cost"], report["metrics"]["total_cost"])
            else:
                self.assertAlmostEqual(report["metrics"]["total_cost"], sum(item.cost for item in agent.replay)
                                       + sum(report["accounting"]["prefix_cost"].values()))

    def test_evaluation_after_teacher_uses_only_learned_argmin_and_preserves_replay(self):
        adapter, agent, teacher = self.setup_run()
        with deterministic_cpu():
            _, clock = run_episode(adapter, agent, environment_seed=7, training=True,
                schedule=EpsilonSchedule(), training_slots=0, behavior_policy=teacher)
            replay = agent.replay.state_dict()
            with patch.object(teacher, "act", side_effect=AssertionError("teacher forbidden")):
                report, unchanged = run_episode(adapter, agent, environment_seed=8, training=False,
                    schedule=EpsilonSchedule(), training_slots=clock)
        self.assertEqual(replay, agent.replay.state_dict())
        self.assertEqual(unchanged, clock)
        self.assertNotIn("training_behavior", report)
        self.assertTrue(all(x["action"] == x["preference_diagnostics"]["selected_action"] for x in report["actions"]))
