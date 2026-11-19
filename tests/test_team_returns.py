"""Global slot conservation and intentionally overlapping independent actor views."""

import copy
import json
import math
import pickle
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from risk_aware_v2x.agents import DecisionAdapter, EncodedObservation, FeatureScales, ObservationEncoder, ReplayBuffer
from risk_aware_v2x.agents.team_returns import TeamDecisionTransition, TeamReturnAdapter, team_return_contract
from risk_aware_v2x.environment import StepResult
from test_slotted_environment import make_config, make_env, make_mec, make_task, make_user


class IdentityEncoder:
    max_server_vehicles = 0

    def encode(self, observation):
        return observation


class CostTraceEnvironment:
    """Exact cost/decision trace for hand arithmetic; no physical-model mock claims."""

    config = SimpleNamespace(max_server_vehicles=0)
    users = {1: None, 2: None, 3: None}

    def __init__(self):
        self.trace = ({1: 1., 2: 1., 3: 0.}, {1: 1., 2: 2., 3: 0.},
                      {1: 2., 2: 3., 3: 0.}, {1: 3., 2: 4., 3: 0.}, {1: 5., 2: 6., 3: 0.})
        self.decisions = {1: (1,), 2: (2,), 3: (1, 2)}

    @property
    def observations(self):
        return {uid: EncodedObservation(3, (1.,), (True,), uid, self.slot * 10 + uid)
                for uid in self.decisions.get(self.slot, ())} if not self.terminated else {}

    def reset(self, *, seed=0):
        self.slot, self.terminated = 0, False
        return self.observations

    def step(self, actions):
        if self.terminated or set(actions) != set(self.observations) or any(a != 0 for a in actions.values()):
            raise ValueError("invalid scripted batch")
        slot, costs = self.slot, self.trace[self.slot]
        self.slot += 1
        self.terminated = self.slot == len(self.trace)
        return StepResult(slot, self.observations, dict(costs), {u: -c for u, c in costs.items()}, self.terminated, ())


def run(adapter, seed=7):
    adapter.reset(seed=seed)
    transitions, intervals, slots, batches = [], [], [], []
    while not adapter.environment.terminated:
        batch = {uid: 0 for uid in adapter.observations}
        batches.append((adapter.environment.slot, batch, adapter.observations))
        step = adapter.step(batch)
        transitions.extend(step.transitions)
        slots.append(step.costs)
        if hasattr(step, "global_interval") and step.global_interval is not None:
            intervals.append(step.global_interval)
    return transitions, intervals, slots, batches


class TeamReturnFormulaTests(unittest.TestCase):
    def test_team_cost_views_and_unique_global_intervals_have_different_sum_rules(self):
        adapter = TeamReturnAdapter(CostTraceEnvironment(), IdentityEncoder())
        transitions, intervals, _, _ = run(adapter)
        # Global costs [2,3,5,7,11], user epochs: u1=(1,3), u2=(2,3).
        views = {(t.observation.user_vehicle_id, t.source.start_slot): t.cost for t in transitions}
        self.assertEqual(views, {(1, 1): 8., (2, 2): 5., (1, 3): 18., (2, 3): 18.})
        self.assertEqual([(t.start_slot, t.duration_slots, t.cost) for t in intervals],
                         [(1, 1, 3.), (2, 1, 5.), (3, 2, 18.)])
        accounting = adapter.accounting
        self.assertEqual(accounting.total_cost, 28.)
        self.assertEqual(accounting.prefix_cost, 2.)
        self.assertEqual(accounting.closed_interval_cost, 26.)
        self.assertEqual(accounting.per_user_prefix_cost, {1: 2., 2: 5., 3: 28.})
        self.assertEqual(accounting.per_user_closed_cost, {1: 26., 2: 23., 3: 0.})
        self.assertNotEqual(sum(t.cost for t in transitions), accounting.closed_interval_cost)
        for uid in (1, 2, 3):
            self.assertEqual(accounting.per_user_prefix_cost[uid] + accounting.per_user_closed_cost[uid], 28.)

    def test_nonunit_discount_uses_physical_slot_duration_and_preserves_absolute_origin(self):
        gamma = .5
        adapter = TeamReturnAdapter(CostTraceEnvironment(), IdentityEncoder(), gamma_per_slot=gamma)
        transitions, intervals, _, _ = run(adapter)
        values = [2., 3., 5., 7., 11.]
        for item in transitions:
            source = item.source
            segment = values[source.start_slot:source.start_slot + source.duration_slots]
            self.assertEqual(item.discounted_cost, sum(gamma ** k * cost for k, cost in enumerate(segment)))
            self.assertEqual(source.bootstrap_discount, 0. if source.terminated else gamma ** source.duration_slots)
            self.assertEqual(item.discounted_reward, -item.discounted_cost)
        accounting = adapter.accounting
        expected = 2 + .5 * 3 + .25 * 5 + .125 * 7 + .0625 * 11
        self.assertEqual(accounting.discounted_total_cost, expected)
        self.assertEqual(accounting.discounted_prefix_cost + accounting.discounted_closed_interval_cost, expected)
        for uid in (1, 2, 3):
            self.assertEqual(accounting.per_user_discounted_prefix_cost[uid]
                             + accounting.per_user_discounted_closed_cost[uid], expected)
        self.assertEqual(intervals[-1].discounted_cost, 7 + .5 * 11)

    def test_zero_discount_and_user_without_actions_have_no_fabricated_bootstrap(self):
        adapter = TeamReturnAdapter(CostTraceEnvironment(), IdentityEncoder(), gamma_per_slot=0.)
        transitions, _, _, _ = run(adapter)
        self.assertTrue(all(t.source.bootstrap_discount == 0. for t in transitions))
        self.assertEqual(adapter.accounting.discounted_total_cost, 2.)
        self.assertEqual(adapter.accounting.discounted_closed_interval_cost, 0.)
        self.assertNotIn(3, {t.observation.user_vehicle_id for t in transitions})

    def test_new_return_type_is_rejected_by_legacy_replay_and_has_explicit_contract(self):
        adapter = TeamReturnAdapter(CostTraceEnvironment(), IdentityEncoder())
        transitions, _, _, _ = run(adapter)
        with self.assertRaises(TypeError):
            ReplayBuffer(capacity=10, seed=7).append(transitions[0])
        self.assertEqual(team_return_contract()["return_scope"], "all-users-sum-v1")
        self.assertEqual(transitions[0].schema_version, 1)
        json.dumps(asdict(transitions[0]), allow_nan=False)

    def test_bad_scope_schema_cost_discount_and_missing_owner_cost_are_rejected(self):
        adapter = TeamReturnAdapter(CostTraceEnvironment(), IdentityEncoder(), gamma_per_slot=.5)
        transitions, _, _, _ = run(adapter)
        first = transitions[0]
        for changes in ({"schema_version": True}, {"schema_version": 2}, {"return_scope": "individual"},
                        {"cost": float("nan")}, {"discounted_cost": float("inf")},
                        {"cost": 0.}, {"discounted_cost": first.cost + 1.}, {"gamma_per_slot": 1.}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(first, **changes)


class TeamReturnIntegrationTests(unittest.TestCase):
    def fixture(self):
        return make_env(users=[make_user(1), make_user(2), make_user(3)],
            config=make_config(arrival_horizon_slots=15), tasks=[make_task(1), make_task(2),
                make_task(3, owner_vehicle_id=2, arrival_slot=1), make_task(4, owner_vehicle_id=2, arrival_slot=14)])

    def adapter(self, environment, gamma=1.):
        return TeamReturnAdapter(environment, ObservationEncoder(
            max_server_vehicles=environment.config.max_server_vehicles, scales=FeatureScales()), gamma_per_slot=gamma)

    def test_local_actor_inputs_actions_physics_and_individual_records_are_identical(self):
        team = self.adapter(self.fixture(), .9)
        reference = DecisionAdapter(self.fixture(), team._individual.encoder, gamma_per_slot=.9)
        team_transitions, _, team_slots, team_batches = run(team)
        original, _, original_slots, original_batches = run(reference)
        self.assertEqual(team_batches, original_batches)
        self.assertEqual(team_slots, original_slots)
        self.assertEqual([t.source for t in team_transitions], original)
        self.assertEqual(team.individual_accounting, reference.accounting)
        self.assertEqual(team.environment.metrics(), reference.environment.metrics())
        self.assertAlmostEqual(team.accounting.total_cost, team.environment.metrics()["total_cost"])
        self.assertAlmostEqual(team.accounting.total_cost,
                               team.accounting.prefix_cost + team.accounting.closed_interval_cost)
        self.assertTrue(any(t.cost > t.source.cost for t in team_transitions))
        for uid in (1, 2, 3):
            self.assertAlmostEqual(team.accounting.discounted_total_cost,
                team.accounting.per_user_discounted_prefix_cost[uid] + team.accounting.per_user_discounted_closed_cost[uid])

    def test_no_task_episode_has_only_empty_batches_and_zero_cost(self):
        team = self.adapter(make_env(tasks=[]))
        transitions, intervals, _, batches = run(team)
        self.assertEqual(transitions, [])
        self.assertEqual(intervals, [])
        self.assertTrue(all(not actions for _, actions, _ in batches))
        self.assertEqual(team.accounting.total_cost, 0.)

    def test_immediate_decision_has_zero_prefix_and_terminal_interval_keeps_drain(self):
        team = self.adapter(make_env(config=make_config(query_request_bits=0, query_header_bits=0,
            query_record_bits=0, query_build_energy_j=0), mec=make_mec(query_processing_time_s=0)))
        transitions, intervals, _, _ = run(team)
        self.assertEqual(team.accounting.prefix_cost, 0.)
        self.assertEqual(intervals[0].start_slot, 0)
        self.assertEqual(intervals[-1].start_slot + intervals[-1].duration_slots, team.environment.slot)
        self.assertTrue(transitions[-1].source.terminated)

    def test_invalid_batch_is_atomic_and_reset_preserves_exact_seeded_outputs(self):
        team = self.adapter(self.fixture())
        first = run(team)
        self.assertEqual(first, run(team))
        team.reset(seed=7)
        while not team.observations:
            team.step({})
        before = pickle.dumps(team)
        with self.assertRaises(ValueError):
            team.step({})
        self.assertEqual(before, pickle.dumps(team))
        received = team.observations
        received.clear()
        self.assertTrue(team.observations)

    def test_lifecycle_and_external_step_guards_remain_active(self):
        team = self.adapter(self.fixture())
        with self.assertRaises(RuntimeError):
            team.accounting
        with self.assertRaises(RuntimeError):
            team.step({})
        run(team)
        with self.assertRaises(RuntimeError):
            team.step({})
        team.reset()
        team.environment.step({})
        with self.assertRaises(RuntimeError):
            team.step({})

    def test_episode_guard_keeps_open_team_costs_without_fabricating_terminal_success(self):
        team = self.adapter(make_env(config=make_config(max_episode_slots=5)))
        team.reset()
        returned = []
        while team.environment.slot < 5:
            returned.extend(team.step({uid: 0 for uid in team.observations}).transitions)
        self.assertEqual(returned, [])
        self.assertGreater(team.accounting.total_cost, team.accounting.prefix_cost)
        before = copy.deepcopy(team.accounting)
        with self.assertRaises(RuntimeError):
            team.step({uid: 0 for uid in team.observations})
        self.assertEqual(team.accounting, before)
