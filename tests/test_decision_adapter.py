"""Decision gaps, prefix/drain accounting, and semi-Markov discount fixtures."""

import math
import unittest

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from test_slotted_environment import make_config, make_env, make_mec, make_task, make_user


def adapter_for(environment, gamma=1.):
    return DecisionAdapter(environment, ObservationEncoder(
        max_server_vehicles=environment.config.max_server_vehicles, scales=FeatureScales(),
    ), gamma_per_slot=gamma)


def run_adapter(adapter, seed=9):
    adapter.reset(seed=seed)
    transitions, costs = [], []
    while not adapter.environment.terminated:
        result = adapter.step({user_id: 0 for user_id in adapter.observations})
        transitions.extend(result.transitions)
        costs.append(result.costs)
    return transitions, costs, adapter.accounting


class DecisionAdapterTests(unittest.TestCase):
    def test_query_prefix_and_all_drain_costs_are_preserved(self):
        adapter = adapter_for(make_env())
        transitions, costs, accounting = run_adapter(adapter)
        self.assertEqual(len(transitions), 1)
        item = transitions[0]
        self.assertGreater(item.start_slot, 0)
        self.assertGreater(accounting.prefix_cost[1], 0.)
        self.assertGreater(item.duration_slots, 1)
        self.assertEqual(item.start_slot + item.duration_slots, len(costs))
        self.assertTrue(item.terminated)
        self.assertIsNone(item.next_observation)
        self.assertEqual(item.bootstrap_discount, 0.)
        self.assertEqual(item.discounted_reward, -item.discounted_cost)
        self.assertAlmostEqual(accounting.total_cost[1], math.fsum(row[1] for row in costs))
        self.assertAlmostEqual(accounting.total_cost[1], accounting.prefix_cost[1] + item.cost)
        self.assertAlmostEqual(accounting.total_cost[1], adapter.environment.metrics()["total_cost"])

    def test_multiple_irregular_intervals_use_gamma_to_slot_duration(self):
        env = make_env(config=make_config(arrival_horizon_slots=15), tasks=[
            make_task(1), make_task(2), make_task(3, arrival_slot=14),
        ])
        gamma = .9
        adapter = adapter_for(env, gamma)
        transitions, slot_costs, accounting = run_adapter(adapter)
        self.assertEqual(len(transitions), 3)
        self.assertEqual(sum(item.terminated for item in transitions), 1)
        self.assertTrue(any(item.duration_slots > 1 and not item.terminated for item in transitions))
        for index, item in enumerate(transitions):
            stop = item.start_slot + item.duration_slots
            expected = math.fsum(gamma ** offset * slot_costs[item.start_slot + offset][1]
                                 for offset in range(item.duration_slots))
            self.assertAlmostEqual(item.discounted_cost, expected)
            self.assertAlmostEqual(item.cost, math.fsum(row[1] for row in slot_costs[item.start_slot:stop]))
            if not item.terminated:
                self.assertAlmostEqual(item.bootstrap_discount, gamma ** item.duration_slots)
                self.assertEqual(stop, transitions[index + 1].start_slot)
                self.assertEqual(item.next_observation, transitions[index + 1].observation)
                self.assertTrue(item.next_observation.action_mask[0])
        expected_total = math.fsum(gamma ** slot * row[1] for slot, row in enumerate(slot_costs))
        self.assertAlmostEqual(accounting.discounted_total_cost[1], expected_total)
        self.assertAlmostEqual(expected_total, accounting.discounted_prefix_cost[1]
                               + accounting.discounted_closed_transition_cost[1])

    def test_multi_user_and_user_with_no_tasks_have_separate_accounting(self):
        env = make_env(users=[make_user(1), make_user(2), make_user(3)],
                       config=make_config(arrival_horizon_slots=2),
                       tasks=[make_task(1), make_task(2, owner_vehicle_id=2, arrival_slot=1)])
        transitions, _, accounting = run_adapter(adapter_for(env))
        self.assertEqual({item.observation.user_vehicle_id for item in transitions}, {1, 2})
        self.assertEqual(accounting.total_cost[3], 0.)
        for user_id in (1, 2, 3):
            self.assertAlmostEqual(accounting.total_cost[user_id],
                                   accounting.prefix_cost[user_id] + accounting.closed_transition_cost[user_id])
        self.assertAlmostEqual(sum(accounting.total_cost.values()), env.metrics()["total_cost"])

    def test_no_task_episode_has_no_fabricated_transition(self):
        transitions, costs, accounting = run_adapter(adapter_for(make_env(tasks=[])))
        self.assertEqual(transitions, [])
        self.assertTrue(costs)
        self.assertEqual(accounting.total_cost, {1: 0.})

    def test_immediate_decision_can_have_zero_prefix(self):
        env = make_env(config=make_config(query_request_bits=0, query_header_bits=0,
                                         query_record_bits=0, query_build_energy_j=0),
                       mec=make_mec(query_processing_time_s=0))
        transitions, _, accounting = run_adapter(adapter_for(env))
        self.assertEqual(transitions[0].start_slot, 0)
        self.assertEqual(accounting.prefix_cost[1], 0.)

    def test_zero_discount_is_well_defined(self):
        adapter = adapter_for(make_env(), 0.)
        transitions, costs, accounting = run_adapter(adapter)
        self.assertEqual(accounting.discounted_total_cost[1], costs[0][1])
        self.assertEqual(transitions[0].bootstrap_discount, 0.)
        self.assertEqual(transitions[0].discounted_cost, costs[transitions[0].start_slot][1])

    def test_invalid_action_batch_does_not_mutate_adapter_or_environment(self):
        adapter = adapter_for(make_env())
        adapter.reset()
        while not adapter.observations:
            adapter.step({})
        before = (adapter.environment.slot, adapter.observations, adapter.accounting)
        with self.assertRaises(ValueError):
            adapter.step({1: 3})
        self.assertEqual(before, (adapter.environment.slot, adapter.observations, adapter.accounting))

    def test_reset_reproducibility_and_copied_observation_mapping(self):
        adapter = adapter_for(make_env(), .95)
        first = run_adapter(adapter)
        self.assertEqual(first, run_adapter(adapter))
        adapter.reset()
        while not adapter.observations:
            adapter.step({})
        copy = adapter.observations
        copy.clear()
        self.assertTrue(adapter.observations)

    def test_lifecycle_errors_are_explicit(self):
        adapter = adapter_for(make_env())
        with self.assertRaises(RuntimeError):
            adapter.step({})
        run_adapter(adapter)
        with self.assertRaises(RuntimeError):
            adapter.step({})
        adapter.reset()
        adapter.environment.step({})
        with self.assertRaises(RuntimeError):
            adapter.step({})
        with self.assertRaises(ValueError):
            adapter_for(make_env(), 1.1)
        with self.assertRaises(ValueError):
            DecisionAdapter(make_env(), ObservationEncoder(max_server_vehicles=0, scales=FeatureScales()))

    def test_guard_failure_does_not_fabricate_a_terminal_transition(self):
        adapter = adapter_for(make_env(config=make_config(max_episode_slots=5)))
        adapter.reset()
        returned = []
        while adapter.environment.slot < 5:
            returned.extend(adapter.step({user_id: 0 for user_id in adapter.observations}).transitions)
        self.assertFalse(adapter.environment.terminated)
        self.assertEqual(returned, [])
        before = adapter.accounting
        self.assertGreater(before.total_cost[1], before.prefix_cost[1])
        self.assertEqual(before.closed_transition_cost[1], 0.)
        with self.assertRaises(RuntimeError):
            adapter.step({user_id: 0 for user_id in adapter.observations})
        self.assertEqual(before, adapter.accounting)
