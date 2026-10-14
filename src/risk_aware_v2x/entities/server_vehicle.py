"""Volunteer server-vehicle entity and finite mobile-resource invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(slots=True)
class ServerVehicle:
    """A mobile compute provider within an MEC coverage region.

    CPU rates are in cycle/s, position in metres, motion in SI units, power in
    watts, and energy in joules. Price, certificates, and blockchain state are
    intentionally absent from the core entity.
    """

    vehicle_id: int
    cpu_capacity_hz: float
    min_self_reserved_cpu_hz: float
    max_concurrent_tasks: int
    max_queue_size: int
    max_tx_power_w: float
    rx_power_w: float
    cpu_energy_coefficient: float
    battery_capacity_j: float
    min_reserve_energy_j: float
    radio_antenna_gain_linear: float
    mobility_profile_id: int | str

    position_m: tuple[float, float] = (0.0, 0.0)
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    acceleration_mps2: float = 0.0
    battery_energy_j: float | None = None
    self_workload_cpu_hz: float = 0.0
    service_queue: list[int] = field(default_factory=list)
    active_task_ids: set[int] = field(default_factory=set)
    cpu_allocations_hz: dict[int, float] = field(default_factory=dict)
    availability: bool = True
    _configuration_locked: bool = field(default=False, init=False, repr=False)

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "vehicle_id",
            "cpu_capacity_hz",
            "min_self_reserved_cpu_hz",
            "max_concurrent_tasks",
            "max_queue_size",
            "max_tx_power_w",
            "rx_power_w",
            "cpu_energy_coefficient",
            "battery_capacity_j",
            "min_reserve_energy_j",
            "radio_antenna_gain_linear",
            "mobility_profile_id",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_configuration_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(
                f"{name} is immutable after ServerVehicle initialization"
            )
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.vehicle_id = require_non_negative_int("vehicle_id", self.vehicle_id)
        self.cpu_capacity_hz = require_positive_float(
            "cpu_capacity_hz", self.cpu_capacity_hz
        )
        self.min_self_reserved_cpu_hz = require_non_negative_float(
            "min_self_reserved_cpu_hz", self.min_self_reserved_cpu_hz
        )
        if self.min_self_reserved_cpu_hz > self.cpu_capacity_hz:
            raise ValueError("reserved CPU cannot exceed CPU capacity")
        self.max_concurrent_tasks = require_positive_int(
            "max_concurrent_tasks", self.max_concurrent_tasks
        )
        self.max_queue_size = require_positive_int(
            "max_queue_size", self.max_queue_size
        )
        self.max_tx_power_w = require_positive_float(
            "max_tx_power_w", self.max_tx_power_w
        )
        self.rx_power_w = require_non_negative_float("rx_power_w", self.rx_power_w)
        self.cpu_energy_coefficient = require_positive_float(
            "cpu_energy_coefficient", self.cpu_energy_coefficient
        )
        self.battery_capacity_j = require_positive_float(
            "battery_capacity_j", self.battery_capacity_j
        )
        self.min_reserve_energy_j = require_non_negative_float(
            "min_reserve_energy_j", self.min_reserve_energy_j
        )
        if self.min_reserve_energy_j >= self.battery_capacity_j:
            raise ValueError("min_reserve_energy_j must be below battery capacity")
        self.radio_antenna_gain_linear = require_positive_float(
            "radio_antenna_gain_linear", self.radio_antenna_gain_linear
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
            raise ValueError("battery_energy_j cannot exceed battery capacity")
        self.self_workload_cpu_hz = require_non_negative_float(
            "self_workload_cpu_hz", self.self_workload_cpu_hz
        )
        if (
            self.min_self_reserved_cpu_hz + self.self_workload_cpu_hz
            > self.cpu_capacity_hz
        ):
            raise ValueError("reserved and self-workload CPU exceed capacity")
        self.service_queue = require_unique_ids("service_queue", self.service_queue)
        self.active_task_ids = set(
            require_unique_ids("active_task_ids", self.active_task_ids)
        )
        if len(self.service_queue) > self.max_queue_size:
            raise ValueError("service_queue exceeds max_queue_size")
        if len(self.active_task_ids) > self.max_concurrent_tasks:
            raise ValueError("active tasks exceed max_concurrent_tasks")
        if set(self.service_queue) & self.active_task_ids:
            raise ValueError("a task cannot be both queued and active")
        normalized_allocations: dict[int, float] = {}
        for task_id, allocation_hz in self.cpu_allocations_hz.items():
            validated_id = require_non_negative_int("allocation task_id", task_id)
            if validated_id not in self.active_task_ids:
                raise ValueError("CPU allocations require active tasks")
            normalized_allocations[validated_id] = require_non_negative_float(
                "allocation_hz", allocation_hz
            )
        self.cpu_allocations_hz = normalized_allocations
        if self.available_service_cpu_hz < 0.0:
            raise ValueError("initial CPU allocations exceed service capacity")
        if not isinstance(self.availability, bool):
            raise TypeError("availability must be a boolean")
        self._configuration_locked = True

    @property
    def available_service_cpu_hz(self) -> float:
        """CPU capacity left after reservation, self load, and task allocations."""
        return (
            self.cpu_capacity_hz
            - self.min_self_reserved_cpu_hz
            - self.self_workload_cpu_hz
            - sum(self.cpu_allocations_hz.values())
        )

    @property
    def can_accept_task(self) -> bool:
        """Whether visible state permits admission to the waiting queue."""
        return (
            self.availability
            and len(self.service_queue) < self.max_queue_size
            and self.battery_energy_j > self.min_reserve_energy_j
            and self.available_service_cpu_hz > 0.0
        )

    def enqueue_task(self, task_id: int) -> None:
        """Add a unique task to the finite waiting queue."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id in self.service_queue or validated_id in self.active_task_ids:
            raise ValueError("task is already assigned to this server vehicle")
        if len(self.service_queue) >= self.max_queue_size:
            raise OverflowError("server-vehicle queue is full")
        self.service_queue.append(validated_id)

    def activate_task(self, task_id: int) -> None:
        """Move a waiting task into an available execution slot."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.service_queue:
            raise KeyError(f"task_id {validated_id} is not queued")
        if len(self.active_task_ids) >= self.max_concurrent_tasks:
            raise OverflowError("all server-vehicle execution slots are occupied")
        self.service_queue.remove(validated_id)
        self.active_task_ids.add(validated_id)

    def set_cpu_allocation(self, task_id: int, allocation_hz: float) -> None:
        """Set one active task's allocation without overcommitting CPU capacity."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise ValueError("CPU can only be allocated to an active task")
        allocation = require_non_negative_float("allocation_hz", allocation_hz)
        previous = self.cpu_allocations_hz.get(validated_id, 0.0)
        available_with_previous = self.available_service_cpu_hz + previous
        if allocation > available_with_previous:
            raise ValueError("CPU allocation exceeds available service capacity")
        if allocation == 0.0:
            self.cpu_allocations_hz.pop(validated_id, None)
        else:
            self.cpu_allocations_hz[validated_id] = allocation

    def preempt_task(self, task_id: int) -> None:
        """Return an active task to the queue at a slot boundary."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise KeyError(f"task_id {validated_id} is not active")
        if len(self.service_queue) >= self.max_queue_size:
            raise OverflowError("cannot preempt into a full waiting queue")
        self.active_task_ids.remove(validated_id)
        self.cpu_allocations_hz.pop(validated_id, None)
        self.service_queue.append(validated_id)

    def release_task(self, task_id: int) -> None:
        """Remove a completed or interrupted active task and its allocation."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise KeyError(f"task_id {validated_id} is not active")
        self.active_task_ids.remove(validated_id)
        self.cpu_allocations_hz.pop(validated_id, None)

    def can_spend_energy(self, energy_j: float) -> bool:
        """Whether a spend preserves the server vehicle's minimum reserve."""
        amount_j = require_non_negative_float("energy_j", energy_j)
        return self.battery_energy_j - amount_j >= self.min_reserve_energy_j

    def consume_energy(self, energy_j: float) -> None:
        """Deduct energy or reject a spend that would violate the reserve."""
        amount_j = require_non_negative_float("energy_j", energy_j)
        if not self.can_spend_energy(amount_j):
            raise ValueError("energy spend would violate min_reserve_energy_j")
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
