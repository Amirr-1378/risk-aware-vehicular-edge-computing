"""Computation-task entity and its explicit lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from decimal import Decimal
from typing import ClassVar

from ._validation import (
    require_finite_float,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
)


class Criticality(IntEnum):
    """Task criticality and its primary-objective multiplicative weight."""

    NORMAL = 1
    HIGH = 2
    CRITICAL = 4


class ExecutorKind(Enum):
    """Permitted remote execution targets; local execution is absent."""

    MEC = "mec"
    SERVER_VEHICLE = "server_vehicle"


class TaskStatus(Enum):
    """All valid non-terminal and terminal task states."""

    CREATED = "created"
    WAITING_FOR_INFORMATION = "waiting_for_information"
    READY_FOR_OFFLOADING = "ready_for_offloading"
    UPLOADING_TO_MEC = "uploading_to_mec"
    UPLOADING_TO_SERVER_VEHICLE = "uploading_to_server_vehicle"
    MEC_QUEUED = "mec_queued"
    V2V_QUEUED = "v2v_queued"
    MEC_EXECUTING = "mec_executing"
    V2V_EXECUTING = "v2v_executing"
    DOWNLOADING_FROM_MEC = "downloading_from_mec"
    DOWNLOADING_FROM_SERVER_VEHICLE = "downloading_from_server_vehicle"
    V2V_SERVICE_INTERRUPTED = "v2v_service_interrupted"
    FALLBACK_PENDING = "fallback_pending"
    COMPLETED_ON_TIME = "completed_on_time"
    COMPLETED_LATE = "completed_late"


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {TaskStatus.WAITING_FOR_INFORMATION, TaskStatus.READY_FOR_OFFLOADING}
    ),
    TaskStatus.WAITING_FOR_INFORMATION: frozenset({TaskStatus.READY_FOR_OFFLOADING}),
    TaskStatus.READY_FOR_OFFLOADING: frozenset(
        {TaskStatus.UPLOADING_TO_MEC, TaskStatus.UPLOADING_TO_SERVER_VEHICLE,
         TaskStatus.WAITING_FOR_INFORMATION}
    ),
    TaskStatus.UPLOADING_TO_MEC: frozenset({TaskStatus.MEC_QUEUED}),
    TaskStatus.UPLOADING_TO_SERVER_VEHICLE: frozenset(
        {TaskStatus.V2V_QUEUED, TaskStatus.V2V_SERVICE_INTERRUPTED}
    ),
    TaskStatus.MEC_QUEUED: frozenset({TaskStatus.MEC_EXECUTING}),
    TaskStatus.V2V_QUEUED: frozenset(
        {TaskStatus.V2V_EXECUTING, TaskStatus.V2V_SERVICE_INTERRUPTED}
    ),
    TaskStatus.MEC_EXECUTING: frozenset(
        {TaskStatus.MEC_QUEUED, TaskStatus.DOWNLOADING_FROM_MEC}
    ),
    TaskStatus.V2V_EXECUTING: frozenset(
        {
            TaskStatus.V2V_QUEUED,
            TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE,
            TaskStatus.V2V_SERVICE_INTERRUPTED,
        }
    ),
    TaskStatus.DOWNLOADING_FROM_MEC: frozenset(
        {TaskStatus.COMPLETED_ON_TIME, TaskStatus.COMPLETED_LATE}
    ),
    TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE: frozenset(
        {
            TaskStatus.COMPLETED_ON_TIME,
            TaskStatus.COMPLETED_LATE,
            TaskStatus.V2V_SERVICE_INTERRUPTED,
        }
    ),
    TaskStatus.V2V_SERVICE_INTERRUPTED: frozenset({TaskStatus.FALLBACK_PENDING}),
    TaskStatus.FALLBACK_PENDING: frozenset({TaskStatus.UPLOADING_TO_MEC}),
    TaskStatus.COMPLETED_ON_TIME: frozenset(),
    TaskStatus.COMPLETED_LATE: frozenset(),
}


@dataclass(slots=True)
class Task:
    """An indivisible remote-computation task.

    Size fields are in bits, CPU work is in cycles, time is in seconds, and
    energy is in joules. A fallback restarts the complete task at the MEC but
    deliberately preserves already accumulated delay and energy.
    """

    task_id: int
    owner_vehicle_id: int
    arrival_slot: int
    input_size_bits: int
    output_size_bits: int
    cpu_cycles_per_bit: float
    deadline_s: float
    criticality: Criticality
    energy_budget_j: float

    status: TaskStatus = field(default=TaskStatus.CREATED, init=False)
    initial_target_id: int | None = field(default=None, init=False)
    initial_target_kind: ExecutorKind | None = field(default=None, init=False)
    final_executor_id: int | None = field(default=None, init=False)
    final_executor_kind: ExecutorKind | None = field(default=None, init=False)
    remaining_upload_bits: float = field(init=False)
    remaining_cpu_cycles: float = field(init=False)
    remaining_download_bits: float = field(init=False)
    query_wait_time_s: float = field(default=0.0, init=False)
    query_energy_j: float = field(default=0.0, init=False)
    v2v_attempt_delay_s: float = field(default=0.0, init=False)
    v2v_attempt_energy_j: float = field(default=0.0, init=False)
    fallback_used: bool = field(default=False, init=False)
    deadline_missed: bool = field(default=False, init=False)
    completion_time_s: float | None = field(default=None, init=False)
    _configuration_locked: bool = field(default=False, init=False, repr=False)

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "task_id",
            "owner_vehicle_id",
            "arrival_slot",
            "input_size_bits",
            "output_size_bits",
            "cpu_cycles_per_bit",
            "deadline_s",
            "criticality",
            "energy_budget_j",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_configuration_locked", False)
            and name in self._IMMUTABLE_FIELDS
        ):
            raise AttributeError(f"{name} is immutable after Task initialization")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        self.task_id = require_non_negative_int("task_id", self.task_id)
        self.owner_vehicle_id = require_non_negative_int(
            "owner_vehicle_id", self.owner_vehicle_id
        )
        self.arrival_slot = require_non_negative_int("arrival_slot", self.arrival_slot)
        self.input_size_bits = require_positive_int(
            "input_size_bits", self.input_size_bits
        )
        self.output_size_bits = require_non_negative_int(
            "output_size_bits", self.output_size_bits
        )
        self.cpu_cycles_per_bit = require_positive_float(
            "cpu_cycles_per_bit", self.cpu_cycles_per_bit
        )
        self.deadline_s = require_positive_float("deadline_s", self.deadline_s)
        if not isinstance(self.criticality, Criticality):
            raise TypeError("criticality must be a Criticality value")
        self.energy_budget_j = require_positive_float(
            "energy_budget_j", self.energy_budget_j
        )
        self.remaining_upload_bits = float(self.input_size_bits)
        self.remaining_cpu_cycles = self.required_cpu_cycles
        self.remaining_download_bits = float(self.output_size_bits)
        self._configuration_locked = True

    @property
    def required_cpu_cycles(self) -> float:
        """Return ``input_size_bits * cpu_cycles_per_bit`` in CPU cycles."""
        return self.input_size_bits * self.cpu_cycles_per_bit

    @property
    def is_terminal(self) -> bool:
        """Whether this task has completed, either on time or late."""
        return self.status in {
            TaskStatus.COMPLETED_ON_TIME,
            TaskStatus.COMPLETED_LATE,
        }

    def absolute_deadline_s(self, slot_duration_s: float) -> float:
        """Return the absolute deadline from slot zero, in seconds."""
        duration_s = require_positive_float("slot_duration_s", slot_duration_s)
        return float(Decimal(self.arrival_slot) * Decimal(str(duration_s)) + Decimal(str(self.deadline_s)))

    def scheduling_key(self, slot_duration_s: float) -> tuple[int, float, int, int]:
        """Return the deterministic criticality-aware EDF ordering key."""
        return (
            -int(self.criticality),
            self.absolute_deadline_s(slot_duration_s),
            self.arrival_slot,
            self.task_id,
        )

    def transition_to(self, next_status: TaskStatus) -> None:
        """Apply one legal lifecycle transition or raise ``ValueError``."""
        if not isinstance(next_status, TaskStatus):
            raise TypeError("next_status must be a TaskStatus value")
        if next_status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(
                f"illegal task transition: {self.status.value} -> {next_status.value}"
            )
        self.status = next_status

    def assign_target(self, kind: ExecutorKind, target_id: int) -> None:
        """Record the first offloading target and begin its upload phase."""
        if self.status is not TaskStatus.READY_FOR_OFFLOADING:
            raise ValueError("a target can only be assigned to a ready task")
        if self.initial_target_id is not None:
            raise ValueError("the task already has an initial target")
        if not isinstance(kind, ExecutorKind):
            raise TypeError("kind must be an ExecutorKind value")
        validated_id = require_non_negative_int("target_id", target_id)
        self.initial_target_kind = kind
        self.initial_target_id = validated_id
        self.transition_to(
            TaskStatus.UPLOADING_TO_MEC
            if kind is ExecutorKind.MEC
            else TaskStatus.UPLOADING_TO_SERVER_VEHICLE
        )

    def add_query_cost(self, wait_time_s: float, energy_j: float) -> None:
        """Accumulate query waiting time and query energy for this task."""
        self.query_wait_time_s += require_non_negative_float(
            "wait_time_s", wait_time_s
        )
        self.query_energy_j += require_non_negative_float("energy_j", energy_j)

    def serve_upload(self, transmitted_bits: float) -> float:
        """Consume upload work and return the number of bits actually served."""
        return self._serve_remaining("remaining_upload_bits", transmitted_bits, "bits")

    def serve_cpu(self, executed_cycles: float) -> float:
        """Consume computation work and return the cycles actually executed."""
        return self._serve_remaining(
            "remaining_cpu_cycles", executed_cycles, "executed_cycles"
        )

    def serve_download(self, transmitted_bits: float) -> float:
        """Consume result-download work and return the bits actually served."""
        return self._serve_remaining(
            "remaining_download_bits", transmitted_bits, "bits"
        )

    def mark_v2v_interrupted(self, delay_s: float, energy_j: float) -> None:
        """Record a recoverable V2V interruption without discarding its cost."""
        if self.status not in {
            TaskStatus.UPLOADING_TO_SERVER_VEHICLE,
            TaskStatus.V2V_QUEUED,
            TaskStatus.V2V_EXECUTING,
            TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE,
        }:
            raise ValueError("only an active V2V phase can be interrupted")
        self.v2v_attempt_delay_s += require_non_negative_float("delay_s", delay_s)
        self.v2v_attempt_energy_j += require_non_negative_float("energy_j", energy_j)
        self.transition_to(TaskStatus.V2V_SERVICE_INTERRUPTED)

    def prepare_mec_fallback(self) -> None:
        """Restore all original work and begin a full MEC re-execution."""
        if self.status is TaskStatus.V2V_SERVICE_INTERRUPTED:
            self.transition_to(TaskStatus.FALLBACK_PENDING)
        if self.status is not TaskStatus.FALLBACK_PENDING:
            raise ValueError("MEC fallback requires an interrupted V2V attempt")
        self.remaining_upload_bits = float(self.input_size_bits)
        self.remaining_cpu_cycles = self.required_cpu_cycles
        self.remaining_download_bits = float(self.output_size_bits)
        self.fallback_used = True
        self.transition_to(TaskStatus.UPLOADING_TO_MEC)

    def mark_deadline_if_crossed(
        self, current_time_s: float, slot_duration_s: float
    ) -> bool:
        """Set the one-time miss flag only after the absolute deadline is crossed."""
        now_s = require_finite_float("current_time_s", current_time_s)
        crossed = now_s > self.absolute_deadline_s(slot_duration_s)
        if crossed and not self.deadline_missed:
            self.deadline_missed = True
            return True
        return False

    def complete(
        self,
        completion_time_s: float,
        slot_duration_s: float,
        executor_id: int,
    ) -> None:
        """Finish a download and classify completion using exact deadline semantics."""
        if self.status is TaskStatus.DOWNLOADING_FROM_MEC:
            kind = ExecutorKind.MEC
        elif self.status is TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE:
            kind = ExecutorKind.SERVER_VEHICLE
        else:
            raise ValueError("a task can only complete from a download phase")
        completed_s = require_finite_float("completion_time_s", completion_time_s)
        duration = require_positive_float("slot_duration_s", slot_duration_s)
        arrival_s = float(Decimal(self.arrival_slot) * Decimal(str(duration)))
        if completed_s < arrival_s:
            raise ValueError("completion_time_s cannot precede task arrival")
        self.final_executor_kind = kind
        self.final_executor_id = require_non_negative_int("executor_id", executor_id)
        self.completion_time_s = completed_s
        on_time = completed_s <= self.absolute_deadline_s(slot_duration_s)
        if not on_time:
            self.deadline_missed = True
        self.transition_to(
            TaskStatus.COMPLETED_ON_TIME if on_time else TaskStatus.COMPLETED_LATE
        )

    def _serve_remaining(self, field_name: str, amount: float, label: str) -> float:
        available = getattr(self, field_name)
        requested = require_non_negative_float(label, amount)
        served = min(available, requested)
        setattr(self, field_name, available - served)
        return served
