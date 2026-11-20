"""Hand-computed learned-policy outcome comparisons without sample pooling."""

import copy
import json
import math
import random
import unittest

from experiments.behavior_warmup_statistics import render_markdown, summarize_behavior_variants


def fixture():
    runs = []
    for variant in ("standard", "greedy_warmup"):
        for scope in ("individual", "team"):
            for seed in (11, 29):
                costs = [2., 6.] if seed == 11 else [10., 14.]
                if variant == "greedy_warmup":
                    costs = [1., 3.] if seed == 11 else [12., 16.]
                if scope == "team":
                    costs = [value + 100. for value in costs]
                episodes = [{"episode_id": f"nominal:validation:{index}", "training": False,
                    "scenario_fingerprint": f"scene-{index}",
                    "seeds": {"split": "validation", "index": index, "scenario_seed": 100 + index,
                              "channel_seed": 200 + index, "policy_seed": 300 + index},
                    "metrics": {"cost_model_version": 2, "total_cost": value,
                                "arrived_tasks": 1 if index == 0 else 9,
                                "completed_tasks": 1 if index == 0 else 9}}
                    for index, value in enumerate(costs)]
                runs.append({"behavior_variant": variant, "return_scope": scope,
                    "algorithm": "qr_dqn_cvar", "agent_seed": seed, "validation": {"nominal": episodes}})
    return runs


def summarize(runs, **changes):
    return summarize_behavior_variants(runs, **{"variants": ("standard", "greedy_warmup"),
        "return_scopes": ("individual", "team"), "algorithms": ("qr_dqn_cvar",),
        "agent_seeds": (11, 29), "profiles": ("nominal",), "validation_cases": 2, **changes})


class BehaviorWarmupStatisticsTests(unittest.TestCase):
    def test_risks_remain_per_model_and_episode_weighted_not_task_weighted(self):
        result = summarize(fixture())
        self.assertEqual(len(result["model_cost_risks"]), 8)
        row = next(x for x in result["model_cost_risks"] if x["return_scope"] == "individual"
                   and x["behavior_variant"] == "standard" and x["agent_seed"] == 11)
        risk = row["episode_cost_risk"]
        self.assertEqual(risk["mean_episode_total_cost"], 4.)
        self.assertEqual(risk["cvar_episode_total_cost"], 6.)
        self.assertAlmostEqual(risk["nominal_tail_mass_in_episodes"], .2)
        self.assertEqual(risk["episodes"], 2)
        self.assertNotIn("cvar_episode_total_cost", result["conditional_seed_differences"][0])

    def test_paired_deltas_and_conditional_seed_statistics_match_hand_arithmetic(self):
        result = summarize(fixture())
        rows = [row for row in result["paired_variant_differences"] if row["return_scope"] == "individual"]
        self.assertEqual([row["mean_greedy_warmup_minus_standard_cost"] for row in rows], [-2., 2.])
        self.assertEqual([row["greedy_warmup_minus_standard_cost"] for row in rows[0]["episode_differences"]], [-1., -3.])
        seeds = result["conditional_seed_differences"][0]
        self.assertEqual(seeds["seed_mean_differences"], {"11": -2., "29": 2.})
        self.assertEqual(seeds["mean_of_seed_differences"], 0.)
        self.assertAlmostEqual(seeds["sample_std_of_seed_differences"], math.sqrt(8))
        self.assertEqual(seeds["trained_agent_seed_pairs"], 2)

    def test_single_seed_empty_episodes_remain_in_risk_without_invented_std(self):
        runs = [run for run in fixture() if run["agent_seed"] == 11]
        for run in runs:
            run["validation"]["nominal"][0]["metrics"].update(total_cost=0., arrived_tasks=0, completed_tasks=0)
        result = summarize(runs, agent_seeds=(11,))
        self.assertTrue(all(row["sample_std_of_seed_differences"] is None for row in result["conditional_seed_differences"]))
        risk = result["model_cost_risks"][0]["episode_cost_risk"]
        self.assertEqual(risk["episodes"], 2)
        self.assertEqual(risk["mean_episode_total_cost"], 3.)
        self.assertIn("n/a", render_markdown(result))
        runs[0]["validation"]["nominal"][0]["metrics"]["total_cost"] = 1.
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize(runs, agent_seeds=(11,))

    def test_input_order_axes_order_and_global_random_state_do_not_change_results(self):
        runs, state = fixture(), random.getstate()
        before = copy.deepcopy(runs)
        result = summarize(runs)
        reverse = copy.deepcopy(runs[::-1])
        for run in reverse:
            run["validation"]["nominal"].reverse()
        self.assertEqual(result, summarize(reverse, variants=("greedy_warmup", "standard"),
            return_scopes=("team", "individual"), agent_seeds=(29, 11)))
        self.assertEqual(before, runs)
        self.assertEqual(state, random.getstate())
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_unknown_variant_or_scope_runs_are_rejected(self):
        for mutate in (lambda runs: runs.pop(), lambda runs: runs.append(copy.deepcopy(runs[0])),
                       lambda runs: runs[0].update(behavior_variant="unknown"),
                       lambda runs: runs[0].update(return_scope="unknown"),
                       lambda runs: runs[0].update(algorithm="unknown")):
            runs = fixture()
            mutate(runs)
            with self.assertRaises(ValueError):
                summarize(runs)

    def test_missing_duplicate_or_nonvalidation_episode_identifiers_are_rejected(self):
        for mutate in (lambda values: values.pop(), lambda values: values.__setitem__(1, copy.deepcopy(values[0])),
                       lambda values: values[0].update(episode_id="nominal:test:0")):
            runs = fixture()
            mutate(runs[0]["validation"]["nominal"])
            with self.assertRaises(ValueError):
                summarize(runs)

    def test_pairing_mismatch_is_rejected_even_when_all_warmup_models_share_it(self):
        for field in ("scenario_fingerprint", "seeds"):
            runs = fixture()
            for run in runs:
                if run["behavior_variant"] == "greedy_warmup":
                    episode = run["validation"]["nominal"][0]
                    if field == "seeds":
                        episode["seeds"]["channel_seed"] += 1
                    else:
                        episode[field] = "different-future"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unpaired"):
                summarize(runs)

    def test_pairing_covers_algorithms_seeds_scopes_and_profiles_stay_separate(self):
        runs = fixture()
        extra = copy.deepcopy(runs)
        for run in extra:
            run["algorithm"] = "qr_dqn"
        runs.extend(extra)
        for run in runs:
            shifted = copy.deepcopy(run["validation"]["nominal"])
            for episode in shifted:
                episode["episode_id"] = episode["episode_id"].replace("nominal", "harsh")
                episode["metrics"]["total_cost"] += 1000.
            run["validation"]["harsh"] = shifted
        result = summarize(runs, algorithms=("qr_dqn", "qr_dqn_cvar"), profiles=("nominal", "harsh"))
        self.assertEqual(len(result["model_cost_risks"]), 32)
        self.assertEqual(len(result["paired_variant_differences"]), 16)
        self.assertEqual(len(result["conditional_seed_differences"]), 8)
        runs[-1]["validation"]["nominal"][0]["scenario_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            summarize(runs, algorithms=("qr_dqn", "qr_dqn_cvar"), profiles=("nominal", "harsh"))

    def test_nonfinite_cost_bad_seed_metadata_and_incomplete_drain_are_rejected(self):
        for where, key, value in (("metrics", "total_cost", float("inf")), ("metrics", "total_cost", -1.),
                ("metrics", "completed_tasks", 0), ("metrics", "cost_model_version", 1),
                ("seeds", "split", "test"), ("seeds", "index", True), ("seeds", "policy_seed", -1)):
            runs = fixture()
            runs[0]["validation"]["nominal"][0][where][key] = value
            with self.subTest(where=where, key=key), self.assertRaises((ValueError, TypeError)):
                summarize(runs)
        runs = fixture()
        runs[0]["validation"]["nominal"][0]["training"] = True
        with self.assertRaisesRegex(ValueError, "frozen"):
            summarize(runs)

    def test_invalid_axes_case_count_alpha_and_missing_profiles_are_rejected(self):
        for changes in ({"variants": ("standard",)}, {"variants": ("standard", "standard")},
                {"variants": ["standard", "greedy_warmup"]}, {"return_scopes": ("team",)},
                {"return_scopes": ("individual", "other")}, {"algorithms": ("x", "x")},
                {"agent_seeds": (True,)}, {"agent_seeds": (11, 11)}, {"validation_cases": 0},
                {"validation_cases": True}, {"alpha": 1.}, {"profiles": ("other",)}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                summarize(fixture(), **changes)

    def test_markdown_reports_tail_mass_and_training_variant_boundaries(self):
        result = summarize(fixture(), alpha=.5)
        markdown = render_markdown(result)
        self.assertIn("CVaR 0.5", markdown)
        self.assertIn("Tail episodes", markdown)
        self.assertIn("teacher is unavailable during evaluation", markdown)
        self.assertIn("not identical original training", markdown)
        self.assertIn("No confidence interval", markdown)
        self.assertIn("conditional on", markdown)
