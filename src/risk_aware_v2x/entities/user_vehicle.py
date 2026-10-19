"""User-vehicle entity with task queue and timestamped MEC information cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
from typing import ClassVar

from ._validation import (
    require_finite_float,
    require_mobility_profile_id,
    require_non_negative_float,
    require_non_negative_int,
    require_position_m,
    require_positive_float,
    require_positive_int,
    require_unique_ids,
)
from .metadata import MECSnapshot


class QueryStatus(Enum):
    """State of one user's cached information or in-flight metadata query."""

    NO_CACHE = "no_cache"
    CACHE_VALID = "cache_valid"
    CACHE_EXPIRED = "cache_expired"
    QUERY_REQUEST_UPLINK = "query_request_uplink"
    QUERY_MEC_PROCESSING = "query_mec_processing"
    QUERY_RESPONSE_DOWNLINK = "query_response_downlink"


_QUERY_PHASE_TRANSITIONS = {
    QueryStatus.QUERY_REQUEST_UPLINK: QueryStatus.QUERY_MEC_PROCESSING,
    QueryStatus.QUERY_MEC_PROCESSING: QueryStatus.QUERY_RESPONSE_DOWNLINK,
}


@dataclass(slots=True)
class UserVehicle:
    """A requesting vehicle that never executes tasks locally.

    Position is in metres, speed in m/s, acceleration in m/s², heading in
    radians, power in watts, and energy in joules.
    """

    vehicle_id: int
    max_tx_power_w: float
    rx_power_w: float
    idle_power_w: float
    battery_capacity_j: float
    radio_antenna_gain_linear: float
    max_task_queue_size: int
    mobility_profile_id: int | str

    position_m: tuple[float, float] = (0.0, 0.0)
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    acceleration_mps2: float = 0.0
    battery_energy_j: float | None = None
    tx_power_w: float | None = None
    task_queue: list[int] = field(default_factory=list)

    cached_mec_snapshot: MECSnapshot | None = field(default=None, init=False)
    snapshot_generated_time_s: float | None = field(default=None, init=False)
    snapshot_received_time_s: float | None = field(default=None, init=False)
    query_status: QueryStatus = field(default=QueryStatus.NO_CACHE, init=False)
    query_remaining_time_s: float = field(default=0.0, init=False)
    query_owner_task_id: int | None = field(default=None, init=False)
    _configuration_locked: bool = field(default=False, init=False, repr=False)

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "vehicle_id",
            "max_tx_power_w",
            "rx_power_w",
            "idle_power_w",
            "battery_capacity_j",
            "radio_antenna_gain_linear",
            "max_task_queue_size",
            "mobility_profile_id",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_configuration_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(
                f"{name} is immutable after UserVehicle initialization"
            )
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.vehicle_id = require_non_negative_int("vehicle_id", self.vehicle_id)
        self.max_tx_power_w = require_positive_float(
            "max_tx_power_w", self.max_tx_power_w
        )
        self.rx_power_w = require_non_negative_float("rx_power_w", self.rx_power_w)
        self.idle_power_w = require_non_negative_float(
            "idle_power_w", self.idle_power_w
        )
        self.battery_capacity_j = require_positive_float(
            "battery_capacity_j", self.battery_capacity_j
        )
        self.radio_antenna_gain_linear = require_positive_float(
            "radio_antenna_gain_linear", self.radio_antenna_gain_linear
        )
        self.max_task_queue_size = require_positive_int(
            "max_task_queue_size", self.max_task_queue_size
        )
        self.mobility_profile_id = require_mobility_profile_id(
            self.mobility_profile_id
        )
        self.position_m = require_position_m(self.position_m)
        self.speed_mps = require_non_negative_float("speed_mps", self.speed_mps)
        self.heading_rad = require_finite_float("heading_rad", self.heading_rad)
        self.acceleration_mps2 = require_finite_float(
            "acceleration_mps2", self.acceleration_mps2
        )
        if self.battery_energy_j is None:
            self.battery_energy_j = self.battery_capacity_j
        else:
            self.battery_energy_j = require_non_negative_float(
                "battery_energy_j", self.battery_energy_j
            )
        if self.battery_energy_j > self.battery_capacity_j:
            raise ValueError("battery_energy_j cannot exceed battery_capacity_j")
        if self.tx_power_w is None:
            self.tx_power_w = self.max_tx_power_w
        else:
            self.tx_power_w = require_non_negative_float(
                "tx_power_w", self.tx_power_w
            )
        if self.tx_power_w > self.max_tx_power_w:
            raise ValueError("tx_power_w cannot exceed max_tx_power_w")
        self.task_queue = require_unique_ids("task_queue", self.task_queue)
        if len(self.task_queue) > self.max_task_queue_size:
            raise ValueError("task_queue exceeds max_task_queue_size")
        self._configuration_locked = True

    @property
    def query_in_flight(self) -> bool:
        """Whether one of the three query phases is currently active."""
        return self.query_status in {
            QueryStatus.QUERY_REQUEST_UPLINK,
            QueryStatus.QUERY_MEC_PROCESSING,
            QueryStatus.QUERY_RESPONSE_DOWNLINK,
        }

    def enqueue_task(self, task_id: int) -> None:
        """Append a unique task identifier without exceeding queue capacity."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id in self.task_queue:
            raise ValueError("task_id is already in the user queue")
        if len(self.task_queue) >= self.max_task_queue_size:
            raise OverflowError("user task queue is full")
        self.task_queue.append(validated_id)

    def remove_task(self, task_id: int) -> None:
        """Remove an existing task identifier from the local admission queue."""
        validated_id = require_non_negative_int("task_id", task_id)
        try:
            self.task_queue.remove(validated_id)
        except ValueError as error:
            raise KeyError(f"task_id {validated_id} is not queued") from error

    def information_age_s(self, current_time_s: float) -> float | None:
        """Return snapshot age from generation time, or ``None`` without a cache."""
        if self.snapshot_generated_time_s is None:
            return None
        now_s = require_non_negative_float("current_time_s", current_time_s)
        if now_s < self.snapshot_generated_time_s:
            raise ValueError("current_time_s cannot precede snapshot generation")
        # Decimal timestamp subtraction preserves slot-grid TTL boundaries
        # without an epsilon that would accept a genuinely expired snapshot.
        return float(Decimal(str(now_s)) - Decimal(str(self.snapshot_generated_time_s)))

    def is_cache_valid(self, current_time_s: float, lifetime_s: float) -> bool:
        """Return true through the exact lifetime boundary (age <= lifetime)."""
        lifetime = require_non_negative_float("lifetime_s", lifetime_s)
        age_s = self.information_age_s(current_time_s)
        return age_s is not None and age_s <= lifetime

    def refresh_cache_status(self, current_time_s: float, lifetime_s: float) -> None:
        """Update a non-query cache state from its timestamped information age."""
        if self.query_in_flight:
            return
        if self.cached_mec_snapshot is None:
            self.query_status = QueryStatus.NO_CACHE
        elif self.is_cache_valid(current_time_s, lifetime_s):
            self.query_status = QueryStatus.CACHE_VALID
        else:
            self.query_status = QueryStatus.CACHE_EXPIRED

    def begin_query(self, owner_task_id: int, request_time_s: float) -> None:
        """Start the single permitted in-flight query for a waiting queued task."""
        validated_id = require_non_negative_int("owner_task_id", owner_task_id)
        if validated_id not in self.task_queue:
            raise ValueError("query owner must be a queued task of this user")
        if self.query_in_flight:
            raise ValueError("at most one query may be in flight per user")
        if self.query_status not in {QueryStatus.NO_CACHE, QueryStatus.CACHE_EXPIRED}:
            raise ValueError("a query can only start without a valid cache")
        require_non_negative_float("request_time_s", request_time_s)
        self.query_owner_task_id = validated_id
        self.query_remaining_time_s = 0.0
        self.query_status = QueryStatus.QUERY_REQUEST_UPLINK

    def advance_query_phase(
        self, next_status: QueryStatus, remaining_time_s: float
    ) -> None:
        """Advance request to MEC processing or processing to response downlink."""
        if not isinstance(next_status, QueryStatus):
            raise TypeError("next_status must be a QueryStatus value")
        expected = _QUERY_PHASE_TRANSITIONS.get(self.query_status)
        if next_status is not expected:
            raise ValueError(
                f"illegal query transition: {self.query_status.value} -> "
                f"{next_status.value}"
            )
        self.query_remaining_time_s = require_non_negative_float(
            "remaining_time_s", remaining_time_s
        )
        self.query_status = next_status

    def set_query_remaining_time_s(self, remaining_time_s: float) -> None:
        """Update remaining phase time while ensuring a query is active."""
        if not self.query_in_flight:
            raise ValueError("no query is in flight")
        self.query_remaining_time_s = require_non_negative_float(
            "remaining_time_s", remaining_time_s
        )

    def install_snapshot(self, snapshot: MECSnapshot, received_time_s: float) -> None:
        """Install a fully received response and clear the query owner."""
        if self.query_status is not QueryStatus.QUERY_RESPONSE_DOWNLINK:
            raise ValueError("a snapshot can only arrive from the response phase")
        if not isinstance(snapshot, MECSnapshot):
            raise TypeError("snapshot must be an MECSnapshot")
        received_s = require_non_negative_float("received_time_s", received_time_s)
        if received_s < snapshot.generated_time_s:
            raise ValueError("snapshot cannot be received before it is generated")
        self.cached_mec_snapshot = snapshot
        self.snapshot_generated_time_s = snapshot.generated_time_s
        self.snapshot_received_time_s = received_s
        self.query_remaining_time_s = 0.0
        self.query_owner_task_id = None
        self.query_status = QueryStatus.CACHE_VALID

    def can_spend_energy(self, energy_j: float, protected_reserve_j: float = 0.0) -> bool:
        """Check an energy spend while preserving the configured fallback reserve."""
        amount_j = require_non_negative_float("energy_j", energy_j)
        reserve_j = require_non_negative_float(
            "protected_reserve_j", protected_reserve_j
        )
        if reserve_j > self.battery_capacity_j:
            raise ValueError("protected_reserve_j cannot exceed battery capacity")
        return self.battery_energy_j - amount_j >= reserve_j

    def consume_energy(
        self, energy_j: float, protected_reserve_j: float = 0.0
    ) -> None:
        """Deduct mobile energy without violating a protected fallback reserve."""
        amount_j = require_non_negative_float("energy_j", energy_j)
        if not self.can_spend_energy(amount_j, protected_reserve_j):
            raise ValueError("energy spend would violate the protected reserve")
        self.battery_energy_j -= amount_j

    def update_motion(
        self,
        *,
        position_m: tuple[float, float],
        speed_mps: float,
        heading_rad: float,
        acceleration_mps2: float,
    ) -> None:
        """Install a mobility-model result after validating its SI quantities."""
        self.position_m = require_position_m(position_m)
        self.speed_mps = require_non_negative_float("speed_mps", speed_mps)
        self.heading_rad = require_finite_float("heading_rad", heading_rad)
        self.acceleration_mps2 = require_finite_float(
            "acceleration_mps2", acceleration_mps2
        )
