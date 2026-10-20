"""Per-user decision-epoch transitions over a physical-slot simulator.

Each action remains open until that user's next decision or global termination.
Costs from every intervening slot are retained, including query and drain costs.
Costs before a user's first action are an explicit pre-decision prefix.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..entities._validation import require_fraction
from ..environment import OffloadingObservation, SlottedEnvironment
from .encoding import EncodedObservation, ObservationEncoder


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    observation: EncodedObservation
    action: int
    start_slot: int
    duration_slots: int
    cost: float
    discounted_cost: float
    bootstrap_discount: float
    next_observation: EncodedObservation | None
    terminated: bool

    @property
    def discounted_reward(self) -> float:
        return -self.discounted_cost


@dataclass(frozen=True, slots=True)
class LearningStep:
    slot: int
    observations: dict[int, EncodedObservation]
    transitions: tuple[DecisionTransition, ...]
    costs: dict[int, float]
    rewards: dict[int, float]
    terminated: bool


@dataclass(frozen=True, slots=True)
class EpisodeAccounting:
    """Per-user totals; prefix + closed transitions equals total after full drain.

    Discounted prefix/total/closed amounts use the absolute episode slot origin.
    A transition's own discounted_cost instead uses its start_slot as origin.
    """
    prefix_cost: dict[int, float]
    discounted_prefix_cost: dict[int, float]
    closed_transition_cost: dict[int, float]
    discounted_closed_transition_cost: dict[int, float]
    total_cost: dict[int, float]
    discounted_total_cost: dict[int, float]


@dataclass(slots=True)
class _Pending:
    observation: EncodedObservation
    action: int
    start_slot: int
    costs: list[float] = field(default_factory=list)


class DecisionAdapter:
    """No learned no-op action and no lost cost in slots without decisions.

    Use reset/step ONLY through this adapter while an episode is active. Gamma
    is per physical slot, not per decision. Each user has its own transition;
    cost attribution is per user, not exclusively to the task in its observation.
    """
    def __init__(self, environment: SlottedEnvironment, encoder: ObservationEncoder, *,
                 gamma_per_slot: float = 1.0) -> None:
        if encoder.max_server_vehicles != environment.config.max_server_vehicles:
            raise ValueError("encoder and environment action capacities must match")
        self.environment = environment
        self.encoder = encoder
        self.gamma_per_slot = require_fraction("gamma_per_slot", gamma_per_slot)
        self._initialized = False

    @property
    def observations(self) -> dict[int, EncodedObservation]:
        self._require_reset()
        return dict(self._observations)

    @property
    def raw_observations(self) -> dict[int, OffloadingObservation]:
        """Same received information for analytical baselines, never diagnostic truth."""
        self._require_reset()
        return dict(self._raw)

    @property
    def accounting(self) -> EpisodeAccounting:
        self._require_reset()
        return EpisodeAccounting(*(
            {user_id: math.fsum(values) for user_id, values in bucket.items()}
            for bucket in (self._prefix, self._discounted_prefix, self._closed,
                           self._discounted_closed, self._total, self._discounted_total)
        ))

    def _require_reset(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset before using the decision adapter")

    def reset(self, *, seed: int = 0) -> dict[int, EncodedObservation]:
        self._raw = self.environment.reset(seed=seed)
        self._observations = {user_id: self.encoder.encode(observation)
                              for user_id, observation in self._raw.items()}
        self._pending: dict[int, _Pending] = {}
        self._prefix = {user_id: [] for user_id in self.environment.users}
        self._discounted_prefix = {user_id: [] for user_id in self.environment.users}
        self._closed = {user_id: [] for user_id in self.environment.users}
        self._discounted_closed = {user_id: [] for user_id in self.environment.users}
        self._total = {user_id: [] for user_id in self.environment.users}
        self._discounted_total = {user_id: [] for user_id in self.environment.users}
        self._has_acted: set[int] = set()
        self._expected_slot = 0
        self._initialized = True
        return self.observations

    def step(self, actions: Mapping[int, int]) -> LearningStep:
        self._require_reset()
        if self.environment.slot != self._expected_slot:
            raise RuntimeError("environment was stepped outside its decision adapter")
        # The kernel validates the entire batch before either layer changes state.
        result = self.environment.step(actions)
        for user_id, action in actions.items():
            self._pending[user_id] = _Pending(self._observations[user_id], action, result.slot)
            self._has_acted.add(user_id)
        for user_id, cost in result.costs.items():
            absolute_discounted = self.gamma_per_slot ** result.slot * cost
            self._total[user_id].append(cost)
            self._discounted_total[user_id].append(absolute_discounted)
            if user_id in self._pending:
                self._pending[user_id].costs.append(cost)
            elif user_id not in self._has_acted:
                self._prefix[user_id].append(cost)
                self._discounted_prefix[user_id].append(absolute_discounted)
            else:
                raise RuntimeError("an acted user must have an open decision interval")
        next_observations = {user_id: self.encoder.encode(observation)
                             for user_id, observation in result.observations.items()}
        transitions = []
        for user_id in sorted(tuple(self._pending)):
            if not result.terminated and user_id not in next_observations:
                continue
            pending = self._pending.pop(user_id)
            duration = result.slot + 1 - pending.start_slot
            cost = math.fsum(pending.costs)
            discounted_cost = math.fsum(self.gamma_per_slot ** index * value
                                        for index, value in enumerate(pending.costs))
            transitions.append(DecisionTransition(
                pending.observation, pending.action, pending.start_slot, duration,
                cost, discounted_cost,
                0.0 if result.terminated else self.gamma_per_slot ** duration,
                None if result.terminated else next_observations[user_id], result.terminated,
            ))
            self._closed[user_id].append(cost)
            self._discounted_closed[user_id].append(self.gamma_per_slot ** pending.start_slot * discounted_cost)
        self._raw, self._observations = result.observations, next_observations
        self._expected_slot = result.slot + 1
        return LearningStep(result.slot, self.observations, tuple(transitions),
                            result.costs, result.rewards, result.terminated)
