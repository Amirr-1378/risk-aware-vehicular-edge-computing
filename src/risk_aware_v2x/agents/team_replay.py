"""Typed team-return replay; individual records are never silently converted."""

from dataclasses import asdict
from random import Random

from ..contracts import COST_MODEL_VERSION
from ..entities._validation import require_non_negative_int, require_positive_int
from ..models.randomness import RandomStreams
from .replay import transition_from_record, validate_replay_schema
from .team_returns import TEAM_RETURN_SCHEMA_VERSION, TEAM_RETURN_SCOPE, TeamDecisionTransition, team_return_contract


def validate_team_transition(transition):
    if not isinstance(transition, TeamDecisionTransition):
        raise TypeError("team replay requires TeamDecisionTransition records")
    transition.__post_init__()


def team_transition_record(transition):
    validate_team_transition(transition)
    return asdict(transition)


def team_transition_from_record(record):
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record.get("schema_version") != TEAM_RETURN_SCHEMA_VERSION
            or record.get("return_scope") != TEAM_RETURN_SCOPE):
        raise ValueError("explicit team return schema/scope is required")
    data = dict(record)
    data["source"] = transition_from_record(data["source"])
    return TeamDecisionTransition(**data)


class TeamReplayBuffer:
    """Uniform ring, explicit owned RNG; sampling design matches user replay.

    The RNG label is deliberately the same as individual replay to permit
    matched return-scope ablations. Stored record types/contracts differ.
    """

    def __init__(self, *, capacity, seed):
        self.capacity = require_positive_int("capacity", capacity)
        self._rng = RandomStreams(seed).stream("replay")
        self._items: list[TeamDecisionTransition] = []
        self._next = 0

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def append(self, transition):
        validate_team_transition(transition)
        if self._items and self._items[0].observation.schema_version != transition.observation.schema_version:
            raise ValueError("team replay cannot mix encoded observation schemas")
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def sample(self, size):
        size = require_positive_int("sample size", size)
        if size > len(self._items):
            raise ValueError("not enough team replay samples")
        return tuple(self._rng.sample(self._items, size))

    def state_dict(self):
        validate_replay_schema(self._items)
        return {"schema_version": 1, "cost_model_version": COST_MODEL_VERSION,
            "team_return_contract": team_return_contract(), "capacity": self.capacity,
            "next_index": self._next, "items": [team_transition_record(item) for item in self._items],
            "rng_state": self._rng.getstate()}

    @classmethod
    def from_state_dict(cls, state):
        if (type(state.get("schema_version")) is not int or state.get("schema_version") != 1
                or type(state.get("cost_model_version")) is not int or state.get("cost_model_version") != COST_MODEL_VERSION
                or state.get("team_return_contract") != team_return_contract()):
            raise ValueError("incompatible team replay schema/cost/return scope")
        result = cls(capacity=state["capacity"], seed=0)
        if not isinstance(state["items"], list):
            raise TypeError("team replay items must be a list")
        items = [team_transition_from_record(record) for record in state["items"]]
        validate_replay_schema(items)
        index = require_non_negative_int("next_index", state["next_index"])
        if (len(items) > result.capacity or index >= result.capacity
                or (len(items) < result.capacity and index != len(items))):
            raise ValueError("invalid team replay size/cursor")
        rng = Random(0)
        rng.setstate(state["rng_state"])
        result._items, result._next, result._rng = items, index, rng
        return result
