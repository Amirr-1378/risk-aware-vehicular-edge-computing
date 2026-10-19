"""Per-task ledgers, one-time events, and slot cost aggregation."""

from dataclasses import dataclass, field, replace
import math

from ..entities import Task
from ..entities._validation import require_non_negative_float
from ..models.energy import MobileEnergyBreakdown
from ..models.latency import LatencyBreakdown
from .config import EnvironmentConfig


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """An auditable event timestamp in seconds, without exposing it to a policy."""

    kind: str
    time_s: float
    task_id: int | None = None
    user_vehicle_id: int | None = None
    detail: str = ""


@dataclass(slots=True)
class TaskLedger:
    """Accumulated SI-valued components for a single task."""

    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    energy: MobileEnergyBreakdown = field(default_factory=MobileEnergyBreakdown)
    total_cost: float = 0.0
    mec_charged: bool = False
    deadline_charged: bool = False
    attempt_start_s: float | None = None

    def add_time(self, bucket: str, duration_s: float) -> None:
        self.latency = replace(self.latency, **{bucket: getattr(self.latency, bucket) + duration_s})

    def add_energy(self, bucket: str, energy_j: float) -> None:
        self.energy = replace(self.energy, **{bucket: getattr(self.energy, bucket) + energy_j})

    def discard_v2v_attempt(self) -> None:
        """Move old-attempt components to retained buckets, without adding cost again."""
        duration = sum(getattr(self.latency, name) for name in (
            "queue_wait_s", "upload_s", "compute_s", "download_s", "slot_boundary_wait_s",
        ))
        self.latency = replace(
            self.latency, queue_wait_s=0.0, upload_s=0.0, compute_s=0.0,
            download_s=0.0, slot_boundary_wait_s=0.0,
            interrupted_attempt_s=self.latency.interrupted_attempt_s + duration,
        )
        energy = math.fsum(getattr(self.energy, name) for name in (
            "user_tx_j", "user_rx_j", "user_idle_j", "server_rx_j", "server_cpu_j", "server_tx_j",
        ))
        self.energy = MobileEnergyBreakdown(
            query_j=self.energy.query_j,
            interrupted_attempt_j=self.energy.interrupted_attempt_j + energy,
        )


def incremental_cost(
    task: Task, config: EnvironmentConfig, *, elapsed_s: float = 0.0,
    energy_j: float = 0.0, mec_event: bool = False, deadline_event: bool = False,
) -> float:
    """The approved weighted combined cost increment (dimensionless)."""
    elapsed_s = require_non_negative_float("elapsed_s", elapsed_s)
    energy_j = require_non_negative_float("energy_j", energy_j)
    if not isinstance(mec_event, bool) or not isinstance(deadline_event, bool):
        raise TypeError("one-time event flags must be booleans")
    return int(task.criticality) * (
        config.latency_weight * elapsed_s / task.deadline_s
        + config.energy_weight * energy_j / task.energy_budget_j
        + config.mec_weight * mec_event
        + config.deadline_penalty * deadline_event
    )
