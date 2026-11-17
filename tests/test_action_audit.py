"""Counterfactual cost arithmetic, cloned state transitions and no-test access."""

import copy
import json
import math
import pickle
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.comparison_policies import comparison_policies
from experiments.run_action_audit import audit_decision, compare_branches, run_action_audit
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.scenarios import ScenarioSpec
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from test_slotted_environment import make_env, make_task, make_user, to_decision


class BranchComparisonTests(unittest.TestCase):
    def test_owner_improvement_can_increase_global_cost_and_externality_is_retained(self):
        branches = [{"action": 0, "remaining_cost_by_user": {1: 3., 2: 7.}},
                    {"action": 1, "remaining_cost_by_user": {1: 2., 2: 10.}},
                    {"action": 2, "remaining_cost_by_user": {1: 5., 2: 3.}}]
        result = compare_branches(branches, greedy_action=0, owner_id=1)
        self.assertEqual(result["owner_best_action_ex_post"], 1)
        self.assertEqual(result["system_best_action_ex_post"], 2)
        self.assertEqual(result["owner_gain_vs_greedy_ex_post"], 1.)
        self.assertEqual(result["system_gain_vs_greedy_ex_post"], 2.)
        self.assertEqual(result["owner_best_change_to_other_users_cost"], 3.)
        self.assertEqual(result["opposite_owner_system_preferences"], [1, 2])

    def test_exact_ties_use_action_order_and_tolerance_does_not_modify_ranking(self):
        branches = [{"action": 2, "remaining_cost_by_user": {1: 1.}},
                    {"action": 0, "remaining_cost_by_user": {1: 1.}}]
        self.assertEqual(compare_branches(branches, greedy_action=2, owner_id=1)["system_best_action_ex_post"], 0)
        branches[0]["remaining_cost_by_user"][1] = math.nextafter(1., 0.)
        result = compare_branches(branches, greedy_action=0, owner_id=1)
        self.assertEqual(result["system_best_action_ex_post"], 2)
        self.assertGreater(result["system_gain_vs_greedy_ex_post"], 0.)

    def test_invalid_costs_missing_users_duplicate_actions_and_missing_reference_are_rejected(self):
        for branches, reference, owner in (([], 0, 1),
                ([{"action": 0, "remaining_cost_by_user": {1: float("nan")}}], 0, 1),
                ([{"action": 0, "remaining_cost_by_user": {1: -1.}}], 0, 1),
                ([{"action": 0, "remaining_cost_by_user": {1: 1.}}], 1, 1),
                ([{"action": 0, "remaining_cost_by_user": {1: 1.}}], 0, 2),
                ([{"action": 0, "remaining_cost_by_user": {1: 1.}},
                  {"action": 0, "remaining_cost_by_user": {1: 1.}}], 0, 1),
                ([{"action": 0, "remaining_cost_by_user": {1: 1.}},
                  {"action": 1, "remaining_cost_by_user": {2: 1.}}], 0, 1)):
            with self.assertRaises(ValueError):
                compare_branches(branches, greedy_action=reference, owner_id=owner)


class ActionAuditTests(unittest.TestCase):
    def setUp(self):
        self.environment = make_env(users=[make_user(), make_user(vehicle_id=2)],
            tasks=[make_task(), make_task(task_id=2, owner_vehicle_id=2)])
        self.environment.reset(seed=7)
        to_decision(self.environment)
        policies, _ = comparison_policies(self.environment.config, self.environment.users.values(),
                                          self.environment.server_vehicles.values(), seed=17)
        self.policy = policies["minimum_estimated_combined_cost"]

    def test_branching_preserves_all_source_state_and_global_rng(self):
        before, rng = pickle.dumps(self.environment), random.getstate()
        result = audit_decision(self.environment, self.policy, owner_id=1)
        self.assertEqual(before, pickle.dumps(self.environment))
        self.assertEqual(rng, random.getstate())
        self.assertEqual({branch["action"] for branch in result["branches"]}, {0, 1})
        self.assertEqual(set(result["reference_action_batch"]), {1, 2})
        for branch in result["branches"]:
            self.assertEqual(set(branch["remaining_cost_by_user"]), {1, 2})
            self.assertEqual(branch["terminal_metrics"]["arrived_tasks"], 2.)
            self.assertEqual(branch["terminal_metrics"]["completed_tasks"], 2.)
        json.dumps(result, allow_nan=False)

    def test_branch_order_does_not_change_fading_rng_outcomes_or_action_comparisons(self):
        forward = audit_decision(self.environment, self.policy, owner_id=1)
        backward = audit_decision(self.environment, self.policy, owner_id=1, action_order=(1, 0))
        self.assertEqual(forward, backward)

    def test_full_cost_minus_predecision_sunk_cost_equals_remaining_cost(self):
        sunk = math.fsum(ledger.total_cost for ledger in self.environment.ledgers.values())
        self.assertGreater(sunk, 0.)  # Query time and query energy have already been incurred.
        result = audit_decision(self.environment, self.policy, owner_id=1)
        for branch in result["branches"]:
            self.assertAlmostEqual(math.fsum(branch["remaining_cost_by_user"].values()) + sunk,
                                   branch["terminal_metrics"]["total_cost"])

    def test_reference_branch_matches_unmodified_simulation_and_other_simultaneous_action_is_fixed(self):
        result = audit_decision(self.environment, self.policy, owner_id=1)
        for branch in result["branches"]:
            manual = copy.deepcopy(self.environment)
            manual.step({**result["reference_action_batch"], 1: branch["action"]})
            while not manual.terminated:
                manual.step({uid: self.policy.act(obs) for uid, obs in manual.observations.items()})
            self.assertEqual(terminal_metrics(manual), branch["terminal_metrics"])

    def test_failed_rollout_aborts_without_mutating_source_or_dropping_branch(self):
        before = pickle.dumps(self.environment)
        with patch("experiments.run_action_audit._finish_branch", side_effect=RuntimeError("guard fixture")) as finish:
            with self.assertRaisesRegex(RuntimeError, "guard fixture"):
                audit_decision(self.environment, self.policy, owner_id=1)
            self.assertEqual(finish.call_count, 1)
        self.assertEqual(before, pickle.dumps(self.environment))
        with self.assertRaises(ValueError):
            audit_decision(self.environment, self.policy, owner_id=1, action_order=(0, 0))

    def test_runner_uses_predeclared_prefix_and_never_requests_test_seeds(self):
        specification = replace(ScenarioSpec(), num_users=2, num_servers=1,
            max_server_vehicle_slots=2, arrival_horizon_slots=4, arrival_rate_per_user_hz=100.)
        plan = DevelopmentProtocol("action-fixture", 7, (1, 2, 1), (11,),
                                   (specification,), bootstrap_resamples=10)
        original = DevelopmentProtocol.episode_seeds
        accessed = []

        def guard(instance, split, index):
            accessed.append(split)
            self.assertEqual(split, "validation")
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = run_action_audit(protocol=plan, profiles=("nominal",), cases=2, decisions_per_episode=2)
        self.assertEqual(set(accessed), {"validation"})
        self.assertFalse(report["training_started"])
        self.assertFalse(report["test_split_accessed"])
        self.assertEqual(report["summary"]["inspected_decisions"], 4)
        for episode in report["episodes"]:
            decisions = episode["inspected_decisions"]
            self.assertEqual(len(decisions), 2)
            self.assertEqual([(d["slot"], d["user_vehicle_id"]) for d in decisions],
                             sorted((d["slot"], d["user_vehicle_id"]) for d in decisions))
            self.assertEqual(episode["reference_metrics"]["completed_tasks"], 8.)
        json.dumps(report, allow_nan=False)

    def test_invalid_plan_prefix_fails_before_any_scenario_generation(self):
        plan = DevelopmentProtocol("action-fixture", 7, (1, 1, 1), (11,), (ScenarioSpec(),))
        for arguments in ({"cases": 2}, {"profiles": ("unknown",)}, {"profiles": ("nominal", "nominal")},
                          {"decisions_per_episode": 0}, {"decisions_per_episode": True}):
            with patch("experiments.run_action_audit.generate_scenario", side_effect=AssertionError("too early")):
                with self.assertRaises((TypeError, ValueError)):
                    run_action_audit(protocol=plan, **arguments)
