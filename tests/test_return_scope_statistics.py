"""Hand-computed paired episode differences without seed/CVaR pooling."""

import copy
import json
import math
import random
import unittest

from experiments.return_scope_statistics import render_markdown, summarize_return_scope_runs


def fixture():
    runs = []
    for seed in (11, 29):
        costs = {"individual": [2., 6.] if seed == 11 else [10., 14.],
                 "team": [1., 3.] if seed == 11 else [12., 16.]}
        for scope in ("individual", "team"):
            episodes = [{"episode_id": f"nominal:validation:{index}",
                "scenario_fingerprint": f"scene-{index}",
                "seeds": {"split": "validation", "index": index, "scenario_seed": 100 + index,
                          "channel_seed": 200 + index, "policy_seed": 300 + index},
                "metrics": {"cost_model_version": 2, "total_cost": cost,
                            "arrived_tasks": 1 if index == 0 else 9, "completed_tasks": 1 if index == 0 else 9}}
                for index, cost in enumerate(costs[scope])]
            runs.append({"return_scope": scope, "algorithm": "qr_dqn_cvar", "agent_seed": seed,
                         "validation": {"nominal": episodes}})
    return runs


def summarize(runs, **changes):
    return summarize_return_scope_runs(runs, **{"algorithms": ("qr_dqn_cvar",), "agent_seeds": (11, 29),
        "profiles": ("nominal",), "validation_cases": 2, **changes})


class ReturnScopeStatisticsTests(unittest.TestCase):
    def test_risk_is_per_model_and_episode_weighted_despite_different_task_counts(self):
        result = summarize(fixture())
        self.assertEqual(len(result["model_cost_risks"]), 4)
        risk = next(row["episode_cost_risk"] for row in result["model_cost_risks"]
                    if row["agent_seed"] == 11 and row["return_scope"] == "individual")
        self.assertEqual(risk["mean_episode_total_cost"], 4.)
        self.assertEqual(risk["episodes"], 2)
        self.assertAlmostEqual(risk["cvar_episode_total_cost"], 6.)
        self.assertAlmostEqual(risk["nominal_tail_mass_in_episodes"], .2)
        self.assertNotIn("cvar_episode_total_cost", result["conditional_seed_differences"][0])

    def test_paired_scope_deltas_and_conditional_seed_mean_std_are_hand_computed(self):
        result = summarize(fixture())
        rows = result["paired_scope_differences"]
        self.assertEqual([row["mean_team_minus_individual_cost"] for row in rows], [-2., 2.])
        self.assertEqual([row["team_minus_individual_cost"] for row in rows[0]["episode_differences"]], [-1., -3.])
        seeds = result["conditional_seed_differences"][0]
        self.assertEqual(seeds["seed_mean_differences"], {"11": -2., "29": 2.})
        self.assertEqual(seeds["mean_of_seed_differences"], 0.)
        self.assertAlmostEqual(seeds["sample_std_of_seed_differences"], math.sqrt(8))
        self.assertEqual(seeds["trained_agent_seed_pairs"], 2)

    def test_single_seed_has_undefined_sample_std_and_empty_episodes_stay_in_risk(self):
        runs = [run for run in fixture() if run["agent_seed"] == 11]
        for run in runs:
            run["validation"]["nominal"][0]["metrics"].update(total_cost=0., arrived_tasks=0, completed_tasks=0)
        result = summarize(runs, agent_seeds=(11,))
        self.assertIsNone(result["conditional_seed_differences"][0]["sample_std_of_seed_differences"])
        self.assertEqual(result["model_cost_risks"][0]["episode_cost_risk"]["episodes"], 2)
        self.assertEqual(result["model_cost_risks"][0]["episode_cost_risk"]["mean_episode_total_cost"], 3.)
        self.assertIn("n/a", render_markdown(result))
        runs[0]["validation"]["nominal"][0]["metrics"]["total_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize(runs, agent_seeds=(11,))

    def test_input_run_episode_order_and_global_rng_do_not_affect_summary(self):
        runs, state = fixture(), random.getstate()
        before = copy.deepcopy(runs)
        result = summarize(runs)
        reordered = copy.deepcopy(runs[::-1])
        for run in reordered:
            run["validation"]["nominal"].reverse()
        self.assertEqual(result, summarize(reordered))
        self.assertEqual(before, runs)
        self.assertEqual(state, random.getstate())
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_and_unexpected_runs_are_rejected(self):
        for mutate in (lambda runs: runs.pop(), lambda runs: runs.append(copy.deepcopy(runs[0])),
                       lambda runs: runs[0].update(return_scope="unknown"),
                       lambda runs: runs[0].update(algorithm="unexpected")):
            runs = fixture()
            mutate(runs)
            with self.assertRaises(ValueError):
                summarize(runs)

    def test_missing_duplicate_or_unexpected_validation_episode_ids_are_rejected(self):
        for mutate in (lambda episodes: episodes.pop(), lambda episodes: episodes.__setitem__(1, copy.deepcopy(episodes[0])),
                       lambda episodes: episodes[0].update(episode_id="nominal:test:0")):
            runs = fixture()
            mutate(runs[0]["validation"]["nominal"])
            with self.assertRaises(ValueError):
                summarize(runs)

    def test_pairing_is_checked_across_models_seeds_and_scopes(self):
        for key in ("scenario_fingerprint", "seeds"):
            runs = fixture()
            entry = runs[-1]["validation"]["nominal"][0]
            if key == "seeds":
                entry[key]["channel_seed"] += 1
            else:
                entry[key] = "different-scenario"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unpaired"):
                summarize(runs)

    def test_pairing_also_covers_different_algorithms_and_profiles_stay_separate(self):
        runs = fixture()
        additional = copy.deepcopy(runs)
        for run in additional:
            run["algorithm"] = "qr_dqn"
        runs.extend(additional)
        for run in runs:
            shifted = copy.deepcopy(run["validation"]["nominal"])
            for item in shifted:
                item["episode_id"] = item["episode_id"].replace("nominal", "harsh")
                item["metrics"]["total_cost"] += 100.
            run["validation"]["harsh"] = shifted
        result = summarize(runs, algorithms=("qr_dqn_cvar", "qr_dqn"), profiles=("nominal", "harsh"))
        self.assertEqual(len(result["model_cost_risks"]), 16)
        self.assertEqual(len(result["conditional_seed_differences"]), 4)
        runs[-1]["validation"]["nominal"][0]["scenario_fingerprint"] = "different-algorithm-scene"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize(runs, algorithms=("qr_dqn_cvar", "qr_dqn"), profiles=("nominal", "harsh"))

    def test_bad_seed_role_metadata_and_incomplete_or_nonfinite_costs_are_rejected(self):
        for location, key, value in (("seeds", "split", "test"), ("seeds", "index", True),
                ("seeds", "scenario_seed", -1), ("metrics", "total_cost", float("nan")),
                ("metrics", "total_cost", -1.), ("metrics", "total_cost", True),
                ("metrics", "completed_tasks", 0), ("metrics", "cost_model_version", 1)):
            runs = fixture()
            runs[0]["validation"]["nominal"][0][location][key] = value
            with self.subTest(location=location, key=key), self.assertRaises((ValueError, TypeError)):
                summarize(runs)

    def test_invalid_axes_confidence_count_and_missing_profile_are_rejected(self):
        for changes in ({"algorithms": []}, {"algorithms": ("x", "x")}, {"agent_seeds": (True,)},
                        {"agent_seeds": (11, 11)}, {"profiles": ("nominal", "other")},
                        {"validation_cases": 0}, {"validation_cases": True}, {"alpha": 1.}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                summarize(fixture(), **changes)

    def test_markdown_identifies_sample_units_tail_mass_and_descriptive_limits(self):
        result = summarize(fixture(), alpha=.5)
        text = render_markdown(result)
        self.assertIn("CVaR 0.5", text)
        self.assertIn("Tail episodes", text)
        self.assertIn("Negative paired differences favor team", text)
        self.assertIn("conditional on", text)
        self.assertIn("No confidence interval", text)
        self.assertIn("not identical learned actions", text)
