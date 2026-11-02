"""Cost-model v2 regression: real task outcomes, pure EDF, no label leakage."""

import unittest
from dataclasses import asdict, replace

from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
from risk_aware_v2x.contracts import COST_MODEL_VERSION, model_contract
from risk_aware_v2x.environment.accounting import final_task_cost, incremental_cost
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.entities import ExecutorKind
from risk_aware_v2x.scenarios import ScenarioSpec, generate_scenario
from test_slotted_environment import make_config, make_env, make_server, make_task, drain, to_decision


class RevisedObjectiveTests(unittest.TestCase):
    def check_final_cost(self, env):
        metrics = terminal_metrics(env)
        self.assertEqual(metrics["cost_model_version"], COST_MODEL_VERSION)
        for task in env.tasks.values():
            ledger = env.ledgers[task.task_id]
            elapsed = task.completion_time_s - task.arrival_slot * env.config.slot_duration_s
            expected = .6 * elapsed / task.deadline_s + .25 * ledger.energy.total_j / task.energy_budget_j
            expected += .15 * (task.final_executor_kind is ExecutorKind.MEC)
            self.assertAlmostEqual(final_task_cost(task, ledger, env.config), expected)
            self.assertAlmostEqual(ledger.total_cost, expected)
        return metrics

    def test_direct_mec_and_v2v_use_full_arrival_origin_without_deadline_surcharge(self):
        for action in (0, 1):
            with self.subTest(action=action):
                env = make_env(config=make_config(arrival_horizon_slots=6),
                               tasks=[make_task(arrival_slot=5, deadline_s=.001)])
                env.reset(seed=4)
                drain(env, action=action)
                result = self.check_final_cost(env)
                self.assertEqual(result["deadline_miss_fraction"], 1.)
                self.assertGreater(result["mean_task_tardiness_s"], 0.)
                task = env.tasks[1]
                self.assertEqual(task.final_executor_kind, ExecutorKind.MEC if action == 0 else ExecutorKind.SERVER_VEHICLE)
                self.assertEqual(sum(event.kind == "mec_selected" for event in env.events), int(action == 0))

    def test_fallback_keeps_waste_and_counts_one_mec_indicator(self):
        env = make_env(servers=[make_server()],
                       tasks=[make_task(deadline_s=.001)])
        env.reset(seed=3)
        to_decision(env)
        # The cached candidate was in range at generation; force a physical
        # interruption after the valid action without leaking truth to policy.
        env.step({1: 1})
        env.server_vehicles[9].position_m = (200., 0.)
        drain(env, action=0)
        self.check_final_cost(env)
        task, ledger = env.tasks[1], env.ledgers[1]
        self.assertTrue(task.fallback_used)
        self.assertGreater(ledger.latency.interrupted_attempt_s, 0.)
        self.assertGreater(ledger.energy.interrupted_attempt_j, 0.)
        self.assertEqual(sum(event.kind == "mec_selected" for event in env.events), 1)
        self.assertEqual(task.remaining_cpu_cycles, 0.)

    def test_miss_flags_cannot_change_cost_and_large_ratios_are_not_clipped(self):
        task, config = make_task(deadline_s=.01, energy_budget_j=.1), make_config()
        expected = .6 * 100 + .25 * 20 + .15
        a = incremental_cost(task, config, elapsed_s=1., energy_j=2., mec_event=True)
        task.deadline_missed = True
        b = incremental_cost(task, config, elapsed_s=1., energy_j=2., mec_event=True)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, expected)
        self.assertGreater(a, 1.)
        with self.assertRaises(TypeError):
            incremental_cost(task, config, deadline_event=True)

    def test_label_cannot_enter_task_observation_encoder_or_scenario_manifest(self):
        env = make_env()
        env.reset()
        to_decision(env)
        observation = env.observations[1]
        encoder = ObservationEncoder(max_server_vehicles=3, scales=FeatureScales())
        self.assertEqual(encoder.dimension, 36 + 15 * 3)
        self.assertEqual(encoder.encode(observation).schema_version, 3)
        for record in (asdict(env.tasks[1]), asdict(observation)):
            self.assertFalse(any("criticality" in key or "priority" in key for key in record))
        self.assertFalse(any("criticality" in name or "priority" in name for name in encoder.feature_names))
        manifest = generate_scenario(ScenarioSpec(), seed=4).manifest()
        self.assertTrue(all("criticality" not in task for task in manifest["tasks"]))
        self.assertFalse(manifest["model_contract"]["manual_task_labels"])
        with self.assertRaises(TypeError):
            make_task(criticality=4)

    def test_fixed_objective_and_cost_version_are_validated(self):
        self.assertEqual(model_contract()["objective_weights"], [.6, .25, .15])
        with self.assertRaises(ValueError):
            make_config(cost_model_version=1)
        with self.assertRaises(TypeError):
            make_config(deadline_penalty=2.)
        with self.assertRaises(ValueError):
            make_config(latency_weight=.7)

    def test_reference_reconstruction_detects_corruption_not_just_total_conservation(self):
        env = make_env()
        env.reset()
        drain(env)
        env.ledgers[1].total_cost += 2.
        with self.assertRaisesRegex(RuntimeError, "reference"):
            terminal_metrics(env)
