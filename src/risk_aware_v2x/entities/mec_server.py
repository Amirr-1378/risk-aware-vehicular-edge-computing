"""MEC server entity with finite compute/radio resources and registry state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from ._validation import (
    require_non_negative_float,
    require_non_negative_int,
    require_position_m,
    require_positive_float,
    require_positive_int,
    require_unique_ids,
)
from .metadata import MECMetadataRecord, MECSnapshot, ServerVehicleMetadata


@dataclass(slots=True)
class MECServer:
    """Always-available edge compute and information service.

    CPU rates are in cycle/s, bandwidths in Hz, power in watts, position and
    coverage in metres, and query processing time in seconds. MEC energy is not
    represented because it is excluded from the primary objective.
    """

    server_id: int
    position_m: tuple[float, float]
    coverage_radius_m: float
    cpu_capacity_hz: float
    max_concurrent_tasks: int
    total_uplink_bandwidth_hz: float
    total_downlink_bandwidth_hz: float
    max_downlink_power_w: float
    query_processing_time_s: float

    service_queue: list[int] = field(default_factory=list)
    active_task_ids: set[int] = field(default_factory=set)
    cpu_allocations_hz: dict[int, float] = field(default_factory=dict)
    uplink_allocations_hz: dict[int, float] = field(default_factory=dict)
    downlink_allocations_hz: dict[int, float] = field(default_factory=dict)
    background_cpu_load_hz: float = 0.0
    registered_server_vehicle_ids: set[int] = field(default_factory=set)
    latest_server_information: dict[int, ServerVehicleMetadata] = field(
        default_factory=dict
    )
    _configuration_locked: bool = field(default=False, init=False, repr=False)

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "server_id",
            "position_m",
            "coverage_radius_m",
            "cpu_capacity_hz",
            "max_concurrent_tasks",
            "total_uplink_bandwidth_hz",
            "total_downlink_bandwidth_hz",
            "max_downlink_power_w",
            "query_processing_time_s",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_configuration_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(f"{name} is immutable after MECServer initialization")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.server_id = require_non_negative_int("server_id", self.server_id)
        self.position_m = require_position_m(self.position_m)
        self.coverage_radius_m = require_positive_float(
            "coverage_radius_m", self.coverage_radius_m
        )
        self.cpu_capacity_hz = require_positive_float(
            "cpu_capacity_hz", self.cpu_capacity_hz
        )
        self.max_concurrent_tasks = require_positive_int(
            "max_concurrent_tasks", self.max_concurrent_tasks
        )
        self.total_uplink_bandwidth_hz = require_positive_float(
            "total_uplink_bandwidth_hz", self.total_uplink_bandwidth_hz
        )
        self.total_downlink_bandwidth_hz = require_positive_float(
            "total_downlink_bandwidth_hz", self.total_downlink_bandwidth_hz
        )
        self.max_downlink_power_w = require_positive_float(
            "max_downlink_power_w", self.max_downlink_power_w
        )
        self.query_processing_time_s = require_non_negative_float(
            "query_processing_time_s", self.query_processing_time_s
        )
        self.service_queue = require_unique_ids("service_queue", self.service_queue)
        self.active_task_ids = set(
            require_unique_ids("active_task_ids", self.active_task_ids)
        )
        if set(self.service_queue) & self.active_task_ids:
            raise ValueError("a task cannot be both queued and active")
        if len(self.active_task_ids) > self.max_concurrent_tasks:
            raise ValueError("active tasks exceed max_concurrent_tasks")
        self.background_cpu_load_hz = require_non_negative_float(
            "background_cpu_load_hz", self.background_cpu_load_hz
        )
        if self.background_cpu_load_hz > self.cpu_capacity_hz:
            raise ValueError("background CPU load cannot exceed capacity")
        self.cpu_allocations_hz = self._normalize_allocations(
            self.cpu_allocations_hz, "cpu_allocations_hz", self.active_task_ids
        )
        self.uplink_allocations_hz = self._normalize_allocations(
            self.uplink_allocations_hz, "uplink_allocations_hz"
        )
        self.downlink_allocations_hz = self._normalize_allocations(
            self.downlink_allocations_hz, "downlink_allocations_hz"
        )
        if self.available_cpu_hz < 0.0:
            raise ValueError("initial CPU allocations exceed MEC capacity")
        if self.available_uplink_bandwidth_hz < 0.0:
            raise ValueError("initial uplink allocations exceed MEC bandwidth")
        if self.available_downlink_bandwidth_hz < 0.0:
            raise ValueError("initial downlink allocations exceed MEC bandwidth")
        self.registered_server_vehicle_ids = set(
            require_unique_ids(
                "registered_server_vehicle_ids", self.registered_server_vehicle_ids
            )
        )
        normalized_information: dict[int, ServerVehicleMetadata] = {}
        for vehicle_id, record in self.latest_server_information.items():
            validated_id = require_non_negative_int("server_vehicle_id", vehicle_id)
            if not isinstance(record, ServerVehicleMetadata):
                raise TypeError(
                    "latest_server_information values must be ServerVehicleMetadata"
                )
            if record.server_vehicle_id != validated_id:
                raise ValueError("server information key and record ID must match")
            normalized_information[validated_id] = record
            self.registered_server_vehicle_ids.add(validated_id)
        self.latest_server_information = normalized_information
        self._configuration_locked = True

    @property
    def available_cpu_hz(self) -> float:
        """Unallocated MEC CPU capacity in cycle/s."""
        return (
            self.cpu_capacity_hz
            - self.background_cpu_load_hz
            - sum(self.cpu_allocations_hz.values())
        )

    @property
    def available_uplink_bandwidth_hz(self) -> float:
        """Unallocated aggregate MEC uplink bandwidth in Hz."""
        return self.total_uplink_bandwidth_hz - sum(
            self.uplink_allocations_hz.values()
        )

    @property
    def available_downlink_bandwidth_hz(self) -> float:
        """Unallocated aggregate MEC downlink bandwidth in Hz."""
        return self.total_downlink_bandwidth_hz - sum(
            self.downlink_allocations_hz.values()
        )

    def covers(self, position_m: tuple[float, float]) -> bool:
        """Return whether a two-dimensional position lies inside the MEC region."""
        position = require_position_m(position_m)
        return math.dist(self.position_m, position) <= self.coverage_radius_m

    def enqueue_task(self, task_id: int) -> None:
        """Admit a task to the unbounded MEC waiting queue."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id in self.service_queue or validated_id in self.active_task_ids:
            raise ValueError("task is already assigned to the MEC")
        self.service_queue.append(validated_id)

    def activate_task(self, task_id: int) -> None:
        """Move a queued task into one of the finite MEC execution slots."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.service_queue:
            raise KeyError(f"task_id {validated_id} is not queued")
        if len(self.active_task_ids) >= self.max_concurrent_tasks:
            raise OverflowError("all MEC execution slots are occupied")
        self.service_queue.remove(validated_id)
        self.active_task_ids.add(validated_id)

    def set_cpu_allocation(self, task_id: int, allocation_hz: float) -> None:
        """Set an active task's CPU allocation without exceeding MEC capacity."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise ValueError("CPU can only be allocated to an active task")
        allocation = require_non_negative_float("allocation_hz", allocation_hz)
        previous = self.cpu_allocations_hz.get(validated_id, 0.0)
        if allocation > self.available_cpu_hz + previous:
            raise ValueError("CPU allocation exceeds available MEC capacity")
        if allocation == 0.0:
            self.cpu_allocations_hz.pop(validated_id, None)
        else:
            self.cpu_allocations_hz[validated_id] = allocation

    def set_uplink_allocation(self, user_vehicle_id: int, bandwidth_hz: float) -> None:
        """Set one user's MEC uplink allocation in Hz."""
        self._set_bandwidth_allocation(
            self.uplink_allocations_hz,
            self.total_uplink_bandwidth_hz,
            user_vehicle_id,
            bandwidth_hz,
        )

    def set_downlink_allocation(
        self, user_vehicle_id: int, bandwidth_hz: float
    ) -> None:
        """Set one user's MEC downlink allocation in Hz."""
        self._set_bandwidth_allocation(
            self.downlink_allocations_hz,
            self.total_downlink_bandwidth_hz,
            user_vehicle_id,
            bandwidth_hz,
        )

    def preempt_task(self, task_id: int) -> None:
        """Return an active MEC task to the waiting queue at a slot boundary."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise KeyError(f"task_id {validated_id} is not active")
        self.active_task_ids.remove(validated_id)
        self.cpu_allocations_hz.pop(validated_id, None)
        self.service_queue.append(validated_id)

    def release_task(self, task_id: int) -> None:
        """Remove a completed active task and release its CPU allocation."""
        validated_id = require_non_negative_int("task_id", task_id)
        if validated_id not in self.active_task_ids:
            raise KeyError(f"task_id {validated_id} is not active")
        self.active_task_ids.remove(validated_id)
        self.cpu_allocations_hz.pop(validated_id, None)

    def update_server_information(self, record: ServerVehicleMetadata) -> None:
        """Register or refresh one server-vehicle control-plane record."""
        if not isinstance(record, ServerVehicleMetadata):
            raise TypeError("record must be ServerVehicleMetadata")
        previous = self.latest_server_information.get(record.server_vehicle_id)
        if previous is not None and record.record_time_s < previous.record_time_s:
            raise ValueError("an older record cannot replace latest server information")
        self.registered_server_vehicle_ids.add(record.server_vehicle_id)
        self.latest_server_information[record.server_vehicle_id] = record

    def remove_registered_server_vehicle(self, server_vehicle_id: int) -> None:
        """Remove a server vehicle and its latest control-plane record."""
        validated_id = require_non_negative_int(
            "server_vehicle_id", server_vehicle_id
        )
        if validated_id not in self.registered_server_vehicle_ids:
            raise KeyError(f"server vehicle {validated_id} is not registered")
        self.registered_server_vehicle_ids.remove(validated_id)
        self.latest_server_information.pop(validated_id, None)

    def build_snapshot(
        self,
        *,
        generated_time_s: float,
        queue_workload_cycles: float,
        estimated_queue_delay_s: float,
        server_records: tuple[ServerVehicleMetadata, ...] | None = None,
    ) -> MECSnapshot:
        """Freeze the current MEC state and chosen user-specific server records."""
        generated_s = require_non_negative_float(
            "generated_time_s", generated_time_s
        )
        records = (
            tuple(
                self.latest_server_information[vehicle_id]
                for vehicle_id in sorted(self.latest_server_information)
            )
            if server_records is None
            else tuple(server_records)
        )
        return MECSnapshot(
            generated_time_s=generated_s,
            mec=MECMetadataRecord(
                server_id=self.server_id,
                cpu_capacity_hz=self.cpu_capacity_hz,
                available_cpu_hz=self.available_cpu_hz,
                queue_workload_cycles=require_non_negative_float(
                    "queue_workload_cycles", queue_workload_cycles
                ),
                estimated_queue_delay_s=require_non_negative_float(
                    "estimated_queue_delay_s", estimated_queue_delay_s
                ),
                available_uplink_bandwidth_hz=self.available_uplink_bandwidth_hz,
                available_downlink_bandwidth_hz=self.available_downlink_bandwidth_hz,
            ),
            server_vehicles=records,
        )

    @staticmethod
    def _normalize_allocations(
        values: dict[int, float],
        name: str,
        permitted_ids: set[int] | None = None,
    ) -> dict[int, float]:
        result: dict[int, float] = {}
        for resource_id, allocation in values.items():
            validated_id = require_non_negative_int(f"{name} key", resource_id)
            if permitted_ids is not None and validated_id not in permitted_ids:
                raise ValueError(f"{name} requires an active task")
            result[validated_id] = require_non_negative_float(name, allocation)
        return result

    @staticmethod
    def _set_bandwidth_allocation(
        allocations: dict[int, float],
        capacity_hz: float,
        user_vehicle_id: int,
        bandwidth_hz: float,
    ) -> None:
        validated_id = require_non_negative_int("user_vehicle_id", user_vehicle_id)
        allocation = require_non_negative_float("bandwidth_hz", bandwidth_hz)
        previous = allocations.get(validated_id, 0.0)
        if allocation > capacity_hz - sum(allocations.values()) + previous:
            raise ValueError("bandwidth allocation exceeds aggregate capacity")
        if allocation == 0.0:
            allocations.pop(validated_id, None)
        else:
            allocations[validated_id] = allocation
