"""Dependency-free replay validation, reproducibility, and epsilon-clock fixtures."""

import random
import unittest
from dataclasses import replace

from risk_aware_v2x.agents import DecisionTransition, EncodedObservation, ReplayBuffer
from risk_aware_v2x.agents.replay import transition_from_record, transition_record
from risk_aware_v2x.agents.schedules import EpsilonSchedule


def replay_sample(task_id=1, *, terminal=False):
    observation = EncodedObservation(1, (float(task_id), .5), (True, True, False), 1, task_id)
    following = None if terminal else replace(observation, task_id=task_id + 1)
    return DecisionTransition(observation, 1, 3, 2, 3., 2.9, 0. if terminal else .81,
                              following, terminal)


class ReplayAndScheduleTests(unittest.TestCase):
    def test_ring_evicts_only_oldest_and_restores_cursor(self):
        replay = ReplayBuffer(capacity=3, seed=8)
        for task_id in range(1, 6):
            replay.append(replay_sample(task_id))
        self.assertEqual(len(replay), 3)
        self.assertEqual({item.observation.task_id for item in replay}, {3, 4, 5})
        restored = ReplayBuffer.from_state_dict(replay.state_dict())
        restored.append(replay_sample(6))
        self.assertEqual({item.observation.task_id for item in restored}, {4, 5, 6})

    def test_sample_without_replacement_with_no_global_randomness(self):
        before = random.getstate()
        first, second = ReplayBuffer(capacity=10, seed=3), ReplayBuffer(capacity=10, seed=3)
        for task_id in range(10):
            first.append(replay_sample(task_id))
            second.append(replay_sample(task_id))
        self.assertEqual(first.sample(5), second.sample(5))
        sample = first.sample(10)
        self.assertEqual(len({item.observation.task_id for item in sample}), 10)
        self.assertEqual(before, random.getstate())

    def test_checkpoint_restores_next_random_draw_not_just_initial_seed(self):
        replay = ReplayBuffer(capacity=10, seed=6)
        for task_id in range(10):
            replay.append(replay_sample(task_id))
        replay.sample(3)
        restored = ReplayBuffer.from_state_dict(replay.state_dict())
        for _ in range(5):
            self.assertEqual(replay.sample(4), restored.sample(4))

    def test_primitive_transition_round_trip_terminal_and_nonterminal(self):
        for terminal in (False, True):
            original = replay_sample(terminal=terminal)
            self.assertEqual(transition_from_record(transition_record(original)), original)

    def test_rejects_invalid_original_action_and_inconsistent_terminal(self):
        for changes in ({"action": True}, {"action": 2}, {"action": 3},
                        {"terminated": True}, {"next_observation": None},
                        {"duration_slots": 0}, {"start_slot": -1},
                        {"cost": float("nan")}, {"discounted_cost": -1.},
                        {"bootstrap_discount": 2.}, {"terminated": 1}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                ReplayBuffer(capacity=2, seed=1).append(replace(replay_sample(), **changes))

    def test_rejects_mutable_nonfinite_or_mismatched_observations(self):
        original = replay_sample()
        for observation in (replace(original.observation, values=[1., 2.]),
                            replace(original.observation, values=(float("inf"),)),
                            replace(original.observation, action_mask=(False, True, False))):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                ReplayBuffer(capacity=2, seed=1).append(replace(original, observation=observation))
        for changes in ({"user_vehicle_id": 99}, {"values": (.1,)}, {"action_mask": (True,)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ReplayBuffer(capacity=2, seed=1).append(replace(
                    original, next_observation=replace(original.next_observation, **changes)))

    def test_invalid_capacity_sample_and_checkpoint_cursor(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0, seed=1)
        replay = ReplayBuffer(capacity=2, seed=1)
        with self.assertRaises(ValueError):
            replay.sample(1)
        with self.assertRaises(TypeError):
            replay.sample(True)
        state = replay.state_dict()
        state["next_index"] = 1
        with self.assertRaises(ValueError):
            ReplayBuffer.from_state_dict(state)

    def test_linear_epsilon_uses_slots_and_clamps_after_decay(self):
        schedule = EpsilonSchedule(initial=1., final=.1, decay_slots=100)
        self.assertEqual(schedule.value(0), 1.)
        self.assertAlmostEqual(schedule.value(25), .775)
        self.assertAlmostEqual(schedule.value(50), .55)
        self.assertAlmostEqual(schedule.value(100), .1)
        self.assertAlmostEqual(schedule.value(1_000), .1)
        with self.assertRaises(ValueError):
            schedule.value(-1)
        with self.assertRaises(ValueError):
            EpsilonSchedule(initial=.1, final=.5)
        with self.assertRaises(ValueError):
            EpsilonSchedule(decay_slots=0)
