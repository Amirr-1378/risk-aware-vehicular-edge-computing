"""Typed team replay, strict serialization, uniform sampling and exact resume."""

import copy
import json
import random
import unittest
from dataclasses import replace

from risk_aware_v2x.agents import DecisionTransition, EncodedObservation, ReplayBuffer
from risk_aware_v2x.agents.team_returns import TeamDecisionTransition
from risk_aware_v2x.agents.team_replay import TeamReplayBuffer, team_transition_from_record, team_transition_record


def sample(dimension=1):
    observation = EncodedObservation(3, (1.,) + (0.,) * (dimension - 1), (True, True, False), 1, 1)
    source = DecisionTransition(observation, 0, 0, 2, 3., 2.9, .81, replace(observation, task_id=2), False)
    return TeamDecisionTransition(source, 8., 7.5, .9)


class TeamReplayTests(unittest.TestCase):
    def test_json_roundtrip_preserves_both_cost_scopes_and_local_observation(self):
        item = sample()
        record = json.loads(json.dumps(team_transition_record(item)))
        result = team_transition_from_record(record)
        self.assertEqual(result, item)
        self.assertEqual(result.cost, 8.)
        self.assertEqual(result.source.cost, 3.)
        self.assertEqual(result.observation, result.source.observation)

    def test_missing_wrong_scope_version_and_malformed_nested_cost_are_rejected(self):
        original = team_transition_record(sample())
        for key, value in (("schema_version", None), ("schema_version", True), ("return_scope", "individual"),
                           ("cost", -1.), ("discounted_cost", float("inf"))):
            record = copy.deepcopy(original)
            if value is None:
                record.pop(key)
            else:
                record[key] = value
            with self.subTest(key=key), self.assertRaises((TypeError, ValueError)):
                team_transition_from_record(record)
        record = copy.deepcopy(original)
        record["source"].pop("cost_model_version")
        with self.assertRaises(ValueError):
            team_transition_from_record(record)

    def test_ring_sampling_and_rng_resume_reproduce_without_global_randomness(self):
        replay = TeamReplayBuffer(capacity=3, seed=7)
        state = random.getstate()
        for task_id in range(5):
            item = sample()
            replay.append(replace(item, source=replace(item.source, observation=replace(item.observation, task_id=task_id))))
        self.assertEqual({x.observation.task_id for x in replay}, {2, 3, 4})
        restored = TeamReplayBuffer.from_state_dict(replay.state_dict())
        self.assertEqual(replay.state_dict(), restored.state_dict())
        for _ in range(4):
            self.assertEqual(replay.sample(2), restored.sample(2))
        replay.append(sample())
        restored.append(sample())
        self.assertEqual(replay.state_dict(), restored.state_dict())
        self.assertEqual(random.getstate(), state)

    def test_individual_and_team_replays_reject_each_others_records_and_headers(self):
        individual, team = ReplayBuffer(capacity=3, seed=7), TeamReplayBuffer(capacity=3, seed=7)
        with self.assertRaises(TypeError):
            individual.append(sample())
        with self.assertRaises(TypeError):
            team.append(sample().source)
        with self.assertRaises(ValueError):
            TeamReplayBuffer.from_state_dict(individual.state_dict())
        with self.assertRaises(ValueError):
            ReplayBuffer.from_state_dict(team.state_dict())

    def test_bad_header_cursor_capacity_and_sampling_bounds_fail(self):
        replay = TeamReplayBuffer(capacity=3, seed=7)
        replay.append(sample())
        for changes in ({"schema_version": True}, {"cost_model_version": 2.}, {"team_return_contract": {}},
                        {"next_index": 3}, {"next_index": 2}, {"capacity": 0}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                TeamReplayBuffer.from_state_dict({**replay.state_dict(), **changes})
        for size in (0, 2, True):
            with self.assertRaises((TypeError, ValueError)):
                replay.sample(size)
