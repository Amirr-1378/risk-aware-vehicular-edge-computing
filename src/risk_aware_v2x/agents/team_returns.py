"""Training-only team returns for independent local-observation actors.

The physical global ledger is counted once. Each user's decision intervals
provide an overlapping view of the SAME team costs for cooperative learning;
those views must never be summed across users as an episode objective.
No controller, extra actor observation, or legacy learner/replay change.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..contracts import COST_MODEL_VERSION
from ..entities._validation import (
    require_fraction, require_non_negative_float, require_positive_int,
)
from .transitions import DecisionAdapter, DecisionTransition, EpisodeAccounting
from .replay import validate_transition
from .encoding import EncodedObservation

TEAM_RETURN_SCHEMA_VERSION = 1
TEAM_RETURN_SCOPE = "all-users-sum-v1"


def team_return_contract() -> dict:
    return {"schema_version": TEAM_RETURN_SCHEMA_VERSION, "cost_model_version": COST_MODEL_VERSION,
        "return_scope": TEAM_RETURN_SCOPE, "actors": "independent-local-observation",
        "aggregation": "sum physical slot costs over all users, no average",
        "intervals": "per-user decision epochs; overlapping across users",
        "episode_objective": "unique global slot ledger, never sum actor views"}


@dataclass(frozen=True, slots=True)
class TeamDecisionTransition:
    """Distinct type: legacy replay/learners must reject these team targets.

    source retains the unchanged local observation/action and individual cost
    for audit. cost/discounted_cost here are dimensionless TEAM costs.
    """

    source: DecisionTransition
    cost: float
    discounted_cost: float
    gamma_per_slot: float
    schema_version: int = TEAM_RETURN_SCHEMA_VERSION
    return_scope: str = TEAM_RETURN_SCOPE

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version != TEAM_RETURN_SCHEMA_VERSION or self.return_scope != TEAM_RETURN_SCOPE:
            raise ValueError("unsupported team return contract")
        validate_transition(self.source)
        require_non_negative_float("team cost", self.cost)
        require_non_negative_float("discounted team cost", self.discounted_cost)
        gamma = require_fraction("gamma_per_slot", self.gamma_per_slot)
        expected = 0. if self.source.terminated else gamma ** self.source.duration_slots
        if not math.isclose(self.source.bootstrap_discount, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("source bootstrap discount differs from team slot discount")
        if self.cost + 1e-12 < self.source.cost or self.discounted_cost + 1e-12 < self.source.discounted_cost:
            raise ValueError("team sum cannot exclude the owner's cost")
        if self.discounted_cost > self.cost + 1e-12:
            raise ValueError("nonnegative discounted costs cannot exceed raw costs")
        if gamma == 1. and not math.isclose(self.cost, self.discounted_cost, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("unit discount must preserve raw team cost")

    @property
    def observation(self) -> EncodedObservation:
        return self.source.observation

    @property
    def discounted_reward(self) -> float:
        return -self.discounted_cost


@dataclass(frozen=True, slots=True)
class TeamCostInterval:
    """One disjoint GLOBAL interval, for accounting only; no joint actor input."""

    start_slot: int
    duration_slots: int
    cost: float
    discounted_cost: float
    terminated: bool


@dataclass(frozen=True, slots=True)
class TeamReturnAccounting:
    """Global conservation plus separate per-user views of the team return.

    At drain, global prefix + global closed intervals = global total. For EACH
    user separately, team prefix + that user's closed views = global total.
    A user who never acts has all cost in its prefix and no fabricated action.
    """

    total_cost: float
    discounted_total_cost: float
    prefix_cost: float
    discounted_prefix_cost: float
    closed_interval_cost: float
    discounted_closed_interval_cost: float
    per_user_prefix_cost: dict[int, float]
    per_user_discounted_prefix_cost: dict[int, float]
    per_user_closed_cost: dict[int, float]
    per_user_discounted_closed_cost: dict[int, float]


@dataclass(frozen=True, slots=True)
class TeamLearningStep:
    slot: int
    observations: dict[int, EncodedObservation]
    transitions: tuple[TeamDecisionTransition, ...]
    global_interval: TeamCostInterval | None
    costs: dict[int, float]
    team_cost: float
    team_reward: float
    terminated: bool


class TeamReturnAdapter:
    """Wrap unchanged individual decisions with team-cost training records.

    The actor API remains a separate local observation for each eligible user.
    Empty batches advance physical time, without creating actions/transitions.
    This is an independent-learning data interface, not a centralized MDP or
    a convergence/CVaR-optimality claim. Never feed it to legacy user replay.
    """

    def __init__(self, environment, encoder, *, gamma_per_slot=1.):
        self._individual = DecisionAdapter(environment, encoder, gamma_per_slot=gamma_per_slot)
        self._initialized = False

    @property
    def environment(self):
        return self._individual.environment

    @property
    def gamma_per_slot(self):
        return self._individual.gamma_per_slot

    @property
    def observations(self):
        return self._individual.observations

    @property
    def raw_observations(self):
        return self._individual.raw_observations

    @property
    def individual_accounting(self) -> EpisodeAccounting:
        return self._individual.accounting

    def reset(self, *, seed=0):
        observations = self._individual.reset(seed=seed)
        self._slots: list[float] = []
        self._first_actions: dict[int, int] = {}
        self._global_start: int | None = None
        self._intervals: list[TeamCostInterval] = []
        self._closed = {user_id: [] for user_id in self.environment.users}
        self._discounted_closed = {user_id: [] for user_id in self.environment.users}
        self._initialized = True
        return observations

    def _require_reset(self):
        if not self._initialized:
            raise RuntimeError("call reset before using the team return adapter")

    def _discounted(self, costs):
        return math.fsum(self.gamma_per_slot ** index * value for index, value in enumerate(costs))

    @property
    def accounting(self) -> TeamReturnAccounting:
        self._require_reset()
        first = min(self._first_actions.values(), default=len(self._slots))
        prefix = self._slots[:first]
        prefixes = {uid: self._slots[:self._first_actions.get(uid, len(self._slots))] for uid in self._closed}
        return TeamReturnAccounting(
            math.fsum(self._slots), self._discounted(self._slots),
            math.fsum(prefix), self._discounted(prefix),
            math.fsum(item.cost for item in self._intervals),
            math.fsum(self.gamma_per_slot ** item.start_slot * item.discounted_cost for item in self._intervals),
            {uid: math.fsum(values) for uid, values in prefixes.items()},
            {uid: self._discounted(values) for uid, values in prefixes.items()},
            {uid: math.fsum(values) for uid, values in self._closed.items()},
            {uid: math.fsum(values) for uid, values in self._discounted_closed.items()},
        )

    def step(self, actions: Mapping[int, int]) -> TeamLearningStep:
        self._require_reset()
        # The individual adapter/kernel validate before ANY team state changes.
        result = self._individual.step(actions)
        team_cost = math.fsum(result.costs.values())
        self._slots.append(team_cost)
        if actions:
            for user_id in actions:
                self._first_actions.setdefault(user_id, result.slot)
            if self._global_start is not None:
                raise RuntimeError("global cost interval was not closed at the preceding decision")
            self._global_start = result.slot
        transitions = []
        for source in result.transitions:
            values = self._slots[source.start_slot:source.start_slot + source.duration_slots]
            if len(values) != source.duration_slots:
                raise RuntimeError("team interval does not align with the physical clock")
            transition = TeamDecisionTransition(source, math.fsum(values), self._discounted(values), self.gamma_per_slot)
            transitions.append(transition)
            user_id = source.observation.user_vehicle_id
            self._closed[user_id].append(transition.cost)
            self._discounted_closed[user_id].append(self.gamma_per_slot ** source.start_slot * transition.discounted_cost)
        interval = None
        if self._global_start is not None and (result.observations or result.terminated):
            start = self._global_start
            values = self._slots[start:]
            interval = TeamCostInterval(start, len(values), math.fsum(values), self._discounted(values), result.terminated)
            self._intervals.append(interval)
            self._global_start = None
        return TeamLearningStep(result.slot, result.observations, tuple(transitions), interval,
                                result.costs, team_cost, -team_cost, result.terminated)
