"""Validated uniform replay with explicit RNG and primitive checkpoint records."""

from dataclasses import asdict
from random import Random

from ..entities._validation import (
    require_finite_float, require_fraction, require_non_negative_float,
    require_non_negative_int, require_positive_int,
)
from ..models.randomness import RandomStreams
from .encoding import EncodedObservation
from .transitions import DecisionTransition


def validate_observation(observation: EncodedObservation) -> None:
    if not isinstance(observation, EncodedObservation) or observation.schema_version != 1:
        raise ValueError("expected encoded observation schema v1")
    require_non_negative_int("user_vehicle_id", observation.user_vehicle_id)
    require_non_negative_int("task_id", observation.task_id)
    if not isinstance(observation.values, tuple) or not observation.values:
        raise ValueError("encoded values must be a nonempty immutable tuple")
    for value in observation.values:
        require_finite_float("feature", value)
    mask = observation.action_mask
    if (not isinstance(mask, tuple) or not mask or mask[0] is not True
            or not all(isinstance(flag, bool) for flag in mask)):
        raise ValueError("encoded mask must be an immutable boolean tuple with valid MEC")


def validate_transition(transition: DecisionTransition) -> None:
    if not isinstance(transition, DecisionTransition):
        raise TypeError("replay accepts DecisionTransition records only")
    validate_observation(transition.observation)
    require_non_negative_int("start_slot", transition.start_slot)
    require_positive_int("duration_slots", transition.duration_slots)
    action = require_non_negative_int("action", transition.action)
    mask = transition.observation.action_mask
    if action >= len(mask) or not mask[action]:
        raise ValueError("transition action must have been valid in its original observation")
    require_non_negative_float("cost", transition.cost)
    require_non_negative_float("discounted_cost", transition.discounted_cost)
    require_fraction("bootstrap_discount", transition.bootstrap_discount)
    if not isinstance(transition.terminated, bool):
        raise TypeError("terminated must be boolean")
    if transition.terminated:
        if transition.next_observation is not None or transition.bootstrap_discount != 0.:
            raise ValueError("terminal samples have no next observation and zero bootstrap")
    else:
        validate_observation(transition.next_observation)
        current, following = transition.observation, transition.next_observation
        if (following.user_vehicle_id != current.user_vehicle_id
                or len(following.values) != len(current.values)
                or len(following.action_mask) != len(current.action_mask)):
            raise ValueError("next observation must preserve user, vector, and action dimensions")


def transition_record(transition: DecisionTransition) -> dict:
    validate_transition(transition)
    return asdict(transition)


def transition_from_record(record: dict) -> DecisionTransition:
    def observation(data):
        if data is None:
            return None
        data = dict(data)
        data["values"], data["action_mask"] = tuple(data["values"]), tuple(data["action_mask"])
        return EncodedObservation(**data)
    data = dict(record)
    data["observation"] = observation(data["observation"])
    data["next_observation"] = observation(data["next_observation"])
    result = DecisionTransition(**data)
    validate_transition(result)
    return result


class ReplayBuffer:
    """Fixed-capacity ring; sample uniformly without replacement, never global RNG."""
    def __init__(self, *, capacity: int, seed: int) -> None:
        self.capacity = require_positive_int("capacity", capacity)
        self._rng = RandomStreams(seed).stream("replay")
        self._items: list[DecisionTransition] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def append(self, transition: DecisionTransition) -> None:
        validate_transition(transition)
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def sample(self, size: int) -> tuple[DecisionTransition, ...]:
        size = require_positive_int("sample size", size)
        if size > len(self._items):
            raise ValueError("not enough replay samples")
        return tuple(self._rng.sample(self._items, size))

    def state_dict(self) -> dict:
        return {"schema_version": 1, "capacity": self.capacity, "next_index": self._next,
                "items": [transition_record(item) for item in self._items],
                "rng_state": self._rng.getstate()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ReplayBuffer":
        if state["schema_version"] != 1:
            raise ValueError("unsupported replay schema")
        result = cls(capacity=state["capacity"], seed=0)
        items = [transition_from_record(record) for record in state["items"]]
        index = require_non_negative_int("next_index", state["next_index"])
        if (len(items) > result.capacity or index >= result.capacity
                or (len(items) < result.capacity and index != len(items))):
            raise ValueError("invalid replay size/cursor")
        rng = Random(0)
        rng.setstate(state["rng_state"])
        result._items, result._next, result._rng = items, index, rng
        return result
