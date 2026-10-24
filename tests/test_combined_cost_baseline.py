"""Hand-computed myopic costs, cached contact, and no-oracle boundaries."""

import math
import random
import unittest
from dataclasses import asdict, replace

from experiments.comparison_policies import comparison_policies
from risk_aware_v2x.baselines import GreedyCostParameters, MinimumEstimatedCombinedCost, MinimumEstimatedLatency
from test_slotted_environment import make_env, to_decision


class CombinedCostBaselineTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.env.reset(seed=7)
        to_decision(self.env)
        raw = self.env.observations[1]
        server = replace(raw.servers[0], record_age_s=0., record=replace(raw.servers[0].record,
            available_cpu_hz=100_000., queue_workload_cycles=100_000., estimated_rate_bps=1_000.,
            estimated_downlink_rate_bps=1_000., estimated_contact_time_s=10.))
        self.raw = replace(raw, input_size_bits=1_000, output_size_bits=100,
            required_cpu_cycles=100_000., remaining_cpu_cycles=100_000.,
            deadline_s=10., deadline_slack_s=10., energy_budget_j=10., tx_power_fraction=1.,
            criticality_weight=1, servers=(server, None, None), mec=replace(raw.mec,
            available_cpu_hz=100_000., estimated_queue_delay_s=.25,
            estimated_uplink_rate_bps=1_000., estimated_downlink_rate_bps=1_000.))
        self.parameters = GreedyCostParameters(.6, .25, .15, 2., 1., .5, 1., .5, 1e-16)
        self.policy = MinimumEstimatedCombinedCost(self.parameters)

    def test_mec_cost_uses_user_only_energy_and_queue_delay_once(self):
        estimate = self.policy.estimate(self.raw, 0)
        self.assertAlmostEqual(estimate.service_time_s, 2.35)
        self.assertAlmostEqual(estimate.mobile_energy_j, 1.05)
        self.assertAlmostEqual(estimate.cost, .6 * 2.35 / 10 + .25 * 1.05 / 10 + .15)
        self.assertIsNone(estimate.remaining_cached_contact_s)
        self.assertFalse(estimate.new_deadline_miss)

    def test_v2v_cost_contains_both_mobile_radios_and_task_cpu_only(self):
        estimate = self.policy.estimate(self.raw, 1)
        self.assertAlmostEqual(estimate.service_time_s, 3.1)
        # User 1.05 + SV RX .5 + SV TX .1 + kappa*C*f^2=.1 J.
        self.assertAlmostEqual(estimate.mobile_energy_j, 1.75)
        self.assertAlmostEqual(estimate.cost, .6 * 3.1 / 10 + .25 * 1.75 / 10)
        self.assertTrue(estimate.admissible)
        self.assertEqual(estimate.remaining_cached_contact_s, 10.)

    def test_combined_cost_can_choose_differently_from_latency_only(self):
        self.assertEqual(MinimumEstimatedLatency().act(self.raw), 0)
        self.assertEqual(self.policy.act(self.raw), 1)

    def test_energy_weight_and_deadline_penalty_can_change_the_decision(self):
        expensive = MinimumEstimatedCombinedCost(replace(self.parameters, server_cpu_energy_coefficient=1e-14))
        self.assertEqual(expensive.act(self.raw), 0)
        urgent = replace(self.raw, deadline_slack_s=2.5)
        self.assertFalse(self.policy.estimate(urgent, 0).new_deadline_miss)
        self.assertTrue(self.policy.estimate(urgent, 1).new_deadline_miss)
        self.assertEqual(self.policy.act(urgent), 0)

    def test_strict_deadline_boundary_and_no_second_penalty_for_overdue_task(self):
        duration = self.policy.estimate(self.raw, 1).service_time_s
        exact = replace(self.raw, deadline_slack_s=duration)
        self.assertFalse(self.policy.estimate(exact, 1).new_deadline_miss)
        crossed = replace(exact, deadline_slack_s=math.nextafter(duration, 0.))
        self.assertTrue(self.policy.estimate(crossed, 1).new_deadline_miss)
        self.assertTrue(self.policy.estimate(replace(self.raw, deadline_slack_s=0.), 1).new_deadline_miss)
        overdue = self.policy.estimate(replace(self.raw, deadline_slack_s=-1.), 1)
        self.assertFalse(overdue.new_deadline_miss)
        self.assertEqual(overdue.cost, self.policy.estimate(self.raw, 1).cost)

    def test_original_deadline_is_available_at_exact_deadline_no_zero_over_zero(self):
        raw = replace(self.raw, deadline_slack_s=0., elapsed_deadline_fraction=1.)
        estimate = self.policy.estimate(raw, 1)
        self.assertAlmostEqual(estimate.cost, .6 * 3.1 / 10 + .25 * 1.75 / 10 + 2.)

    def test_criticality_multiplies_cost_not_hardware_energy(self):
        normal = self.policy.estimate(self.raw, 1)
        critical = self.policy.estimate(replace(self.raw, criticality_weight=4), 1)
        self.assertEqual(critical.mobile_energy_j, normal.mobile_energy_j)
        self.assertEqual(critical.cost, 4 * normal.cost)

    def test_already_incurred_time_is_not_charged_again_in_remaining_score(self):
        # Holding slack/normalization fixed isolates the sunk elapsed feature.
        self.assertEqual(self.policy.estimate(self.raw, 1),
                         self.policy.estimate(replace(self.raw, elapsed_deadline_fraction=.5), 1))

    def test_contact_guard_ages_cached_estimate_and_does_not_change_shared_mask(self):
        aged = replace(self.raw.servers[0], record_age_s=7.)
        raw = replace(self.raw, servers=(aged, None, None))
        estimate = self.policy.estimate(raw, 1)
        self.assertEqual(estimate.remaining_cached_contact_s, 3.)
        self.assertFalse(estimate.admissible)
        self.assertEqual(estimate.cost, math.inf)
        self.assertEqual(self.policy.act(raw), 0)
        self.assertTrue(raw.action_mask[1])
        unguarded = MinimumEstimatedCombinedCost(replace(self.parameters, require_cached_contact=False))
        self.assertEqual(unguarded.act(raw), 1)

    def test_contact_exact_boundary_is_allowed_and_expired_estimate_is_not(self):
        duration = self.policy.estimate(self.raw, 1).service_time_s
        server = self.raw.servers[0]
        exact = replace(self.raw, servers=(replace(server, record=replace(server.record,
                                                          estimated_contact_time_s=duration)), None, None))
        self.assertTrue(self.policy.estimate(exact, 1).admissible)
        expired = replace(self.raw, servers=(replace(server, record_age_s=11.), None, None))
        self.assertEqual(self.policy.estimate(expired, 1).remaining_cached_contact_s, 0.)
        self.assertEqual(self.policy.act(expired), 0)

    def test_zero_rate_or_cpu_is_infinite_unavailable_not_nan_even_with_zero_weights(self):
        parameters = replace(self.parameters, latency_weight=0., energy_weight=0., mec_weight=0., deadline_penalty=0.)
        policy = MinimumEstimatedCombinedCost(parameters)
        server = self.raw.servers[0]
        for changes in ({"estimated_rate_bps": 0.}, {"available_cpu_hz": 0.}):
            raw = replace(self.raw, servers=(replace(server, record=replace(server.record, **changes)), None, None),
                          mec=replace(self.raw.mec, estimated_uplink_rate_bps=0.))
            for action in (0, 1):
                self.assertEqual(policy.estimate(raw, action).cost, math.inf)
                self.assertFalse(policy.estimate(raw, action).admissible)
            self.assertEqual(policy.act(raw), 0)

    def test_zero_output_needs_no_downlink_rate_and_actual_tx_fraction_is_used(self):
        mec = replace(self.raw.mec, estimated_downlink_rate_bps=0.)
        raw = replace(self.raw, mec=mec, output_size_bits=0, tx_power_fraction=.25)
        estimate = self.policy.estimate(raw, 0)
        self.assertAlmostEqual(estimate.service_time_s, 2.25)
        self.assertAlmostEqual(estimate.mobile_energy_j, .25)

    def test_mask_padding_action_types_and_invalid_parameters_are_rejected(self):
        for action in (True, 2, 3, -1, 1.0):
            with self.subTest(action=action), self.assertRaises((TypeError, ValueError)):
                self.policy.estimate(self.raw, action)
        for changes in ({"schema_version": 1}, {"deadline_s": 0.}, {"energy_budget_j": 0.},
                        {"deadline_slack_s": float("nan")}, {"criticality_weight": True},
                        {"criticality_weight": 3}, {"tx_power_fraction": 2.}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                self.policy.estimate(replace(self.raw, **changes), 0)
        for changes in ({"energy_weight": -1.}, {"server_rx_power_w": float("nan")},
                        {"server_cpu_energy_coefficient": True}, {"require_cached_contact": 1}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(self.parameters, **changes)

    def test_deterministic_ties_and_zero_candidates_use_mec(self):
        policy = MinimumEstimatedCombinedCost(replace(self.parameters, latency_weight=0., energy_weight=0.,
                                                      mec_weight=0., deadline_penalty=0.))
        self.assertEqual(policy.act(self.raw), 0)
        raw = replace(self.raw, servers=(None, None, None), action_mask=(True, False, False, False))
        self.assertEqual(self.policy.act(raw), 0)

    def test_live_server_truth_and_global_rng_do_not_affect_frozen_decision(self):
        before = random.getstate()
        action = self.policy.act(self.raw)
        self.env.server_vehicles[9].position_m = (10_000., 0.)
        self.env.server_vehicles[9].battery_energy_j = 1.
        self.assertEqual(self.policy.act(self.raw), action)
        self.assertEqual(random.getstate(), before)

    def test_registry_uses_shared_objective_and_frozen_public_static_constants(self):
        config = replace(self.env.config, latency_weight=.7, energy_weight=.2, mec_weight=.1, deadline_penalty=4.)
        policies, contract = comparison_policies(config, self.env.users.values(), self.env.server_vehicles.values(), seed=3)
        self.assertEqual(len(policies), 5)
        self.assertEqual(contract["baseline_suite_version"], 2)
        params = policies["minimum_estimated_combined_cost"].parameters
        for name in ("latency_weight", "energy_weight", "mec_weight", "deadline_penalty"):
            self.assertEqual(getattr(params, name), getattr(config, name))
        self.assertEqual(contract["minimum_estimated_combined_cost"], asdict(params))
        self.assertEqual(params.server_cpu_energy_coefficient, self.env.server_vehicles[9].cpu_energy_coefficient)

    def test_registry_rejects_heterogeneous_hardware_and_accepts_empty_server_cohort(self):
        users, servers = list(self.env.users.values()), list(self.env.server_vehicles.values())
        other = replace(users[0], vehicle_id=99, rx_power_w=users[0].rx_power_w * 2)
        with self.assertRaisesRegex(ValueError, "heterogeneous"):
            comparison_policies(self.env.config, [users[0], other], servers, seed=1)
        with self.assertRaises(ValueError):
            comparison_policies(self.env.config, [], servers, seed=1)
        policies, contract = comparison_policies(self.env.config, users, [], seed=1)
        self.assertEqual(contract["minimum_estimated_combined_cost"]["server_cpu_energy_coefficient"], 0.)
        raw = replace(self.raw, servers=(), action_mask=(True,))
        self.assertEqual(policies["minimum_estimated_combined_cost"].act(raw), 0)
