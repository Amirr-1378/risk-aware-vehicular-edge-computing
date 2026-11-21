"""Hand-computed representation deltas retain per-model risk and pairing."""

import builtins
import copy
import importlib
import json
import math
import random
import unittest
from unittest.mock import patch

from experiments import snr_representation_statistics as statistics


def fixture():
    runs = []
    for seed in (11, 29):
        costs = {"linear_v3": [2., 6.] if seed == 11 else [10., 14.],
                 "log_snr_v4": [1., 3.] if seed == 11 else [12., 16.]}
        for variant in ("linear_v3", "log_snr_v4"):
            episodes = [{"episode_id": f"nominal:validation:{index}",
                "scenario_fingerprint": f"scene-{index}", "training": False, "updates": [],
                "seeds": {"split": "validation", "index": index, "scenario_seed": 100 + index,
                          "channel_seed": 200 + index, "policy_seed": 300 + index},
                "metrics": {"cost_model_version": 2, "total_cost": cost,
                            "arrived_tasks": 1 if index == 0 else 9, "completed_tasks": 1 if index == 0 else 9}}
                for index, cost in enumerate(costs[variant])]
            runs.append({"representation_variant": variant, "return_scope": "individual",
                         "algorithm": "qr_dqn_cvar", "agent_seed": seed, "validation": {"nominal": episodes}})
    return runs


def summarize(runs, **changes):
    return statistics.summarize_snr_representation_runs(runs, **{
        "variants": ("linear_v3", "log_snr_v4"), "algorithms": ("qr_dqn_cvar",),
        "agent_seeds": (11, 29), "profiles": ("nominal",), "validation_cases": 2, **changes})


class SNRRepresentationStatisticsTests(unittest.TestCase):
    def test_risk_rows_keep_models_separate_and_episode_weights_equal(self):
        result = summarize(fixture())
        self.assertEqual(len(result["model_cost_risks"]), 4)
        row = next(row for row in result["model_cost_risks"]
                   if row["agent_seed"] == 11 and row["representation_variant"] == "linear_v3")
        self.assertEqual(row["episode_cost_risk"]["mean_episode_total_cost"], 4.)
        self.assertAlmostEqual(row["episode_cost_risk"]["cvar_episode_total_cost"], 6.)
        self.assertAlmostEqual(row["episode_cost_risk"]["nominal_tail_mass_in_episodes"], .2)
        self.assertNotIn("cvar_episode_total_cost", result["conditional_seed_differences"][0])

    def test_mean_tail_and_conditional_seed_differences_by_hand(self):
        result = summarize(fixture())
        paired = result["paired_variant_differences"]
        self.assertEqual([row["mean_log_minus_linear_cost"] for row in paired], [-2., 2.])
        self.assertEqual([row["cvar_log_minus_linear_cost"] for row in paired], [-3., 2.])
        self.assertEqual([row["log_minus_linear_cost"] for row in paired[0]["episode_differences"]], [-1., -3.])
        seeds = result["conditional_seed_differences"][0]
        self.assertEqual(seeds["seed_mean_differences"], {"11": -2., "29": 2.})
        self.assertEqual(seeds["mean_of_seed_differences"], 0.)
        self.assertAlmostEqual(seeds["sample_std_of_seed_differences"], math.sqrt(8))
        self.assertEqual(seeds["trained_agent_seed_pairs"], 2)

    def test_cvar_difference_is_not_the_cvar_of_paired_differences(self):
        runs = [run for run in fixture() if run["agent_seed"] == 11]
        for run in runs:
            costs = [0., 10.] if run["representation_variant"] == "linear_v3" else [9., 1.]
            for episode, cost in zip(run["validation"]["nominal"], costs, strict=True):
                episode["metrics"]["total_cost"] = cost
        result = summarize(runs, agent_seeds=(11,))
        pair = result["paired_variant_differences"][0]
        self.assertEqual(pair["mean_log_minus_linear_cost"], 0.)
        self.assertAlmostEqual(pair["cvar_log_minus_linear_cost"], -1.)
        self.assertEqual([item["log_minus_linear_cost"] for item in pair["episode_differences"]], [9., -9.])
        self.assertIsNone(result["conditional_seed_differences"][0]["sample_std_of_seed_differences"])

    def test_order_input_and_global_randomness_are_unchanged(self):
        runs, before_rng = fixture(), random.getstate()
        before = copy.deepcopy(runs)
        result = summarize(runs)
        reversed_runs = copy.deepcopy(runs[::-1])
        for run in reversed_runs:
            run["validation"]["nominal"].reverse()
        self.assertEqual(result, summarize(reversed_runs, variants=("log_snr_v4", "linear_v3")))
        self.assertEqual(before, runs)
        self.assertEqual(before_rng, random.getstate())
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_and_unexpected_runs_are_rejected(self):
        for mutate in (lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
                       lambda r: r[0].update(representation_variant="unknown"),
                       lambda r: r[0].update(return_scope="team"), lambda r: r[0].update(algorithm="dqn")):
            runs = fixture()
            mutate(runs)
            with self.assertRaises(ValueError):
                summarize(runs)

    def test_all_models_algorithms_and_seeds_must_share_validation_scenes(self):
        for field in ("scenario_fingerprint", "seeds"):
            runs = fixture()
            episode = runs[-1]["validation"]["nominal"][0]
            if field == "seeds":
                episode[field]["channel_seed"] += 1
            else:
                episode[field] = "other-scene"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unpaired"):
                summarize(runs)
        runs = fixture()
        additional = copy.deepcopy(runs)
        for run in additional:
            run["algorithm"] = "qr_dqn"
        runs.extend(additional)
        self.assertEqual(len(summarize(runs, algorithms=("qr_dqn", "qr_dqn_cvar"))["model_cost_risks"]), 8)
        runs[-1]["validation"]["nominal"][1]["seeds"]["scenario_seed"] += 1
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize(runs, algorithms=("qr_dqn", "qr_dqn_cvar"))

    def test_frozen_validation_and_complete_episode_prefix_are_required(self):
        mutations = [lambda e: e.pop(), lambda e: e.__setitem__(1, copy.deepcopy(e[0])),
                     lambda e: e[0].update(episode_id="nominal:test:0"),
                     lambda e: e[0].update(training=True), lambda e: e[0].update(updates=[{}]),
                     lambda e: e[0].update(training_behavior={}),
                     lambda e: e[0].update(actions=[{"epsilon": .1}]),
                     lambda e: e[0]["metrics"].update(completed_tasks=0),
                     lambda e: e[0]["metrics"].update(total_cost=float("nan")),
                     lambda e: e[0]["seeds"].update(index=True)]
        for mutate in mutations:
            runs = fixture()
            mutate(runs[0]["validation"]["nominal"])
            with self.assertRaises((ValueError, TypeError)):
                summarize(runs)

    def test_empty_episodes_are_retained_in_equal_episode_risk(self):
        runs = fixture()
        for run in runs:
            run["validation"]["nominal"][0]["metrics"].update(total_cost=0., arrived_tasks=0, completed_tasks=0)
        result = summarize(runs)
        row = next(row for row in result["model_cost_risks"]
                   if row["agent_seed"] == 11 and row["representation_variant"] == "linear_v3")
        self.assertEqual(row["episode_cost_risk"]["episodes"], 2)
        self.assertEqual(row["episode_cost_risk"]["mean_episode_total_cost"], 3.)
        runs[0]["validation"]["nominal"][0]["metrics"]["total_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize(runs)

    def test_bad_axes_confidence_counts_and_profiles_reject(self):
        for changes in ({"variants": ("linear_v3",)}, {"variants": ("linear_v3", "linear_v3")},
                        {"variants": ["linear_v3", "log_snr_v4"]}, {"algorithms": ("dqn",)},
                        {"algorithms": ("qr_dqn", "qr_dqn")}, {"agent_seeds": (11, 11)},
                        {"agent_seeds": (True,)}, {"profiles": ("unknown",)},
                        {"validation_cases": 0}, {"validation_cases": True}, {"alpha": 1.}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                summarize(fixture(), **changes)

    def test_statistics_reload_without_torch_or_numpy_and_markdown_limits(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "numpy"):
                raise AssertionError("statistics must not import learning libraries")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            importlib.reload(statistics)
            result = summarize(fixture(), alpha=.5)
        markdown = statistics.render_markdown(result)
        for text in ("CVaR 0.5", "Tail episodes", "Negative paired", "conditional on", "No confidence interval"):
            self.assertIn(text, markdown)


if __name__ == "__main__":
    unittest.main()
