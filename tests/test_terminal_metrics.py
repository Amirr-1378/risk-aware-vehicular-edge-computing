"""Terminal-only metrics, correct denominators, and no invented empty averages."""

import json
import unittest

from risk_aware_v2x.entities import Criticality
from risk_aware_v2x.environment.evaluation import terminal_metrics
from test_slotted_environment import make_env, make_task, drain


class TerminalMetricTests(unittest.TestCase):
    def test_direct_mec_metrics_conserve_cost_energy_and_latency(self):
        environment = make_env()
        environment.reset()
        drain(environment)
        result = terminal_metrics(environment)
        self.assertEqual(result["mec_completed_workload_fraction"], 1.)
        self.assertEqual(result["direct_mec_task_fraction"], 1.)
        self.assertEqual(result["fallback_task_fraction"], 0.)
        self.assertIsNone(result["critical_deadline_miss_fraction"])
        self.assertEqual(result["mean_task_cost"], result["total_cost"])
        self.assertEqual(result["mean_task_latency_s"], environment.ledgers[1].latency.total_s)
        self.assertEqual(result["pooled_task_latency_p99_s"], result["mean_task_latency_s"])
        self.assertGreater(result["mean_query_round_trip_s"], 0.)

    def test_v2v_workload_and_critical_deadline_denominators(self):
        environment = make_env(tasks=[make_task(criticality=Criticality.CRITICAL, deadline_s=.001)])
        environment.reset()
        drain(environment, action=1)
        result = terminal_metrics(environment)
        self.assertEqual(result["mec_completed_workload_fraction"], 0.)
        self.assertEqual(result["v2v_completed_workload_fraction"], 1.)
        self.assertEqual(result["critical_task_count"], 1)
        self.assertEqual(result["critical_deadline_miss_fraction"], 1.)
        self.assertEqual(result["on_time_task_fraction"], 0.)

    def test_empty_episode_preserves_zero_cost_and_null_undefined_metrics(self):
        environment = make_env(tasks=[])
        environment.reset()
        drain(environment)
        result = terminal_metrics(environment)
        self.assertEqual(result["total_cost"], 0.)
        for key in ("mean_task_cost", "mean_task_latency_s", "mec_completed_workload_fraction",
                    "critical_deadline_miss_fraction", "mean_query_round_trip_s"):
            self.assertIsNone(result[key])
        json.dumps(result, allow_nan=False)

    def test_unfinished_episodes_cannot_enter_terminal_evaluation(self):
        environment = make_env()
        environment.reset()
        with self.assertRaises(ValueError):
            terminal_metrics(environment)
