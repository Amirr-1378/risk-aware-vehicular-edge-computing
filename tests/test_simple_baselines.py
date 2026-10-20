"""Policies use received observations only, with shared action semantics."""

import random
import unittest
from dataclasses import replace

from risk_aware_v2x.baselines import (
    AlwaysMEC, AlwaysV2V, MinimumEstimatedLatency, UniformRandom, estimated_service_time_s,
)
from test_slotted_environment import make_env, to_decision


class SimpleBaselineTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.env.reset(seed=7)
        to_decision(self.env)
        self.observation = self.env.observations[1]

    def test_mec_and_v2v_rules_and_missing_candidates(self):
        self.assertEqual(AlwaysMEC().act(self.observation), 0)
        self.assertEqual(AlwaysV2V().act(self.observation), 1)
        masked = replace(self.observation, action_mask=(True, False, False, False))
        self.assertEqual(AlwaysV2V().act(masked), 0)
        self.assertEqual(MinimumEstimatedLatency().act(masked), 0)

    def test_cached_service_time_has_correct_units_and_no_double_backlog(self):
        server = self.observation.servers[0]
        server = replace(server, record=replace(server.record, available_cpu_hz=100_000.,
                         queue_workload_cycles=100_000., estimated_rate_bps=1_000.,
                         estimated_downlink_rate_bps=1_000.))
        mec = replace(self.observation.mec, available_cpu_hz=100_000.,
                      queue_workload_cycles=100_000., estimated_queue_delay_s=.25,
                      estimated_uplink_rate_bps=1_000., estimated_downlink_rate_bps=1_000.)
        raw = replace(self.observation, mec=mec, servers=(server, None, None))
        self.assertAlmostEqual(estimated_service_time_s(raw, 1), 3.1)
        self.assertAlmostEqual(estimated_service_time_s(raw, 0), 2.35)
        self.assertEqual(MinimumEstimatedLatency().act(raw), 0)
        self.assertEqual(AlwaysV2V().act(raw), 1)

    def test_v2v_ties_resolve_by_observed_slot(self):
        server = self.observation.servers[0]
        other = replace(server, record=replace(server.record, server_vehicle_id=11))
        raw = replace(self.observation, servers=(server, other, None), action_mask=(True, True, True, False))
        self.assertEqual(AlwaysV2V().act(raw), 1)

    def test_zero_rate_is_infinite_estimate_not_division_error(self):
        raw = replace(self.observation, mec=replace(self.observation.mec, estimated_uplink_rate_bps=0))
        self.assertEqual(estimated_service_time_s(raw, 0), float("inf"))
        self.assertEqual(MinimumEstimatedLatency().act(raw), 1)

    def test_live_truth_is_not_policy_input(self):
        policy = AlwaysV2V()
        before = policy.act(self.observation)
        self.env.server_vehicles[9].position_m = (1_000., 0.)
        self.assertEqual(policy.act(self.observation), before)

    def test_random_is_seeded_valid_and_global_rng_independent(self):
        before = random.getstate()
        first, second = UniformRandom(seed=5), UniformRandom(seed=5)
        sequence = [first.act(self.observation) for _ in range(100)]
        self.assertEqual(sequence, [second.act(self.observation) for _ in range(100)])
        self.assertEqual(set(sequence), {0, 1})
        self.assertEqual(before, random.getstate())
        only_mec = replace(self.observation, action_mask=(True, False, False, False))
        self.assertEqual(first.act(only_mec), 0)

    def test_random_users_do_not_consume_each_others_streams(self):
        first, second = UniformRandom(seed=2), UniformRandom(seed=2)
        other = replace(self.observation, user_vehicle_id=2)
        reference = [first.act(self.observation) for _ in range(20)]
        actual = []
        for _ in range(20):
            second.act(other)
            actual.append(second.act(self.observation))
        self.assertEqual(reference, actual)

    def test_masked_action_and_padding_are_rejected(self):
        with self.assertRaises(ValueError):
            estimated_service_time_s(self.observation, 2)
        with self.assertRaises(ValueError):
            AlwaysMEC().act(replace(self.observation, action_mask=(True, True, True, False)))
