"""Common policy runner: finite JSON, reproducibility, and no missing costs."""

import json
import unittest

from experiments.run_baseline_smoke import run_comparison
from experiments.run_environment_smoke import _json_value


class BaselineSmokeTests(unittest.TestCase):
    def test_all_policies_finish_and_conserve_discounted_and_raw_cost(self):
        report = run_comparison(gamma_per_slot=.99)
        self.assertEqual(len(report["policies"]), 4)
        for policy in report["policies"].values():
            self.assertEqual(policy["metrics"]["completed_tasks"], 6.)
            self.assertEqual(len(policy["transitions"]), 6)
            accounting = policy["accounting"]
            self.assertAlmostEqual(sum(accounting["total_cost"].values()), policy["metrics"]["total_cost"])
            for user_id, total in accounting["total_cost"].items():
                self.assertAlmostEqual(total, accounting["prefix_cost"][user_id]
                                       + accounting["closed_transition_cost"][user_id])
                self.assertAlmostEqual(accounting["discounted_total_cost"][user_id],
                                       accounting["discounted_prefix_cost"][user_id]
                                       + accounting["discounted_closed_transition_cost"][user_id])
        json.dumps(report, default=_json_value, allow_nan=False)

    def test_seeded_full_report_reproduces(self):
        self.assertEqual(run_comparison(seed=2, policy_seed=3), run_comparison(seed=2, policy_seed=3))

    def test_deterministic_policies_ignore_random_policy_seed(self):
        one, two = run_comparison(policy_seed=1), run_comparison(policy_seed=2)
        for name in ("always_mec", "always_v2v", "minimum_estimated_latency"):
            self.assertEqual(one["policies"][name], two["policies"][name])
