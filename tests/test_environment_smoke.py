"""The public synthetic entry point must drain under both demonstration policies."""

import json
import unittest

from experiments.run_environment_smoke import _json_value, run_smoke


class EnvironmentSmokeTests(unittest.TestCase):
    def test_both_smoke_policies_finish_every_task_and_export_finite_json(self) -> None:
        for policy in ("mec", "first_cached_server"):
            with self.subTest(policy=policy):
                report = run_smoke(seed=7, policy=policy)
                self.assertEqual(report["metrics"]["completed_tasks"], 6.0)
                self.assertAlmostEqual(report["sum_returned_cost"], report["metrics"]["total_cost"])
                json.dumps(report, default=_json_value, allow_nan=False)
                if policy == "first_cached_server":
                    self.assertGreater(report["metrics"]["interruption_count"], 0.0)
                    self.assertGreater(report["metrics"]["v2v_completed_workload_fraction"], 0.0)
                else:
                    self.assertEqual(report["metrics"]["interruption_count"], 0.0)
