"""Versioned raw observations built only from own state and received snapshots."""

import math
from dataclasses import dataclass
from collections.abc import Mapping

from ..entities import MECServer, QueryStatus, Task, UserVehicle
from ..entities.metadata import MECMetadataRecord, ServerVehicleMetadata
from ..models.latency import computation_time_s, transmission_time_s
from ..models.mobility import distance_m
from .config import EnvironmentConfig

RAW_OBSERVATION_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ServerObservation:
    """A frozen server record and task-specific estimates; all times are seconds."""

    record: ServerVehicleMetadata
    record_age_s: float
    contact_margin: float


@dataclass(frozen=True, slots=True)
class OffloadingObservation:
    """Raw schema v2; None server slots are padding, MEC is always action zero.

    Physical suffixes specify SI units. No live server object or future trace
    is referenced. The learning adapter encodes this received information only.
    """

    schema_version: int
    user_vehicle_id: int
    task_id: int
    input_size_bits: int
    output_size_bits: int
    required_cpu_cycles: float
    remaining_cpu_cycles: float
    deadline_s: float
    deadline_slack_s: float
    criticality_weight: int
    energy_budget_j: float
    elapsed_deadline_fraction: float
    user_battery_fraction: float
    user_speed_mps: float
    user_heading_rad: float
    distance_to_mec_m: float
    tx_power_fraction: float
    unassigned_task_count: int
    unassigned_workload_cycles: float
    snapshot_age_s: float
    response_age_s: float
    remaining_lifetime_s: float
    normalized_information_age: float
    query_status: QueryStatus
    query_remaining_time_s: float
    mec: MECMetadataRecord
    mec_utilization: float
    servers: tuple[ServerObservation | None, ...]
    action_mask: tuple[bool, ...]

    def server_id_for_action(self, action: int) -> int:
        """Resolve a non-padded V2V slot; the caller validates the action mask."""
        if isinstance(action, bool) or not isinstance(action, int) or action <= 0:
            raise ValueError("a V2V action must be a positive integer")
        if action >= len(self.action_mask) or self.servers[action - 1] is None:
            raise ValueError("action addresses an absent server slot")
        return self.servers[action - 1].record.server_vehicle_id


def observed_valid_actions(observation: OffloadingObservation) -> tuple[int, ...]:
    """Validate the received mask without consulting any live resource state."""
    mask = observation.action_mask
    if len(mask) != len(observation.servers) + 1 or not mask or mask[0] is not True:
        raise ValueError("observation must have a valid MEC action and fixed server slots")
    if not all(isinstance(flag, bool) for flag in mask):
        raise ValueError("action mask must be boolean")
    if any(mask[index + 1] and server is None
           for index, server in enumerate(observation.servers)):
        raise ValueError("padding cannot be an available action")
    return tuple(index for index, flag in enumerate(mask) if flag)


def build_observation(
    user: UserVehicle, task: Task, tasks: Mapping[int, Task], mec: MECServer,
    config: EnvironmentConfig, current_time_s: float,
) -> OffloadingObservation:
    """Read live own-user features, but only cached MEC/server features and masks."""
    snapshot = user.cached_mec_snapshot
    if snapshot is None or not user.is_cache_valid(current_time_s, config.information_lifetime_s):
        raise ValueError("offloading observations require a received, valid snapshot")
    slots: list[ServerObservation | None] = []
    mask = [True]
    for record in sorted(snapshot.server_vehicles, key=lambda entry: entry.server_vehicle_id):
        estimate_s = (
            transmission_time_s(task.input_size_bits, record.estimated_rate_bps)
            + computation_time_s(record.queue_workload_cycles + task.required_cpu_cycles, record.available_cpu_hz)
            + transmission_time_s(task.output_size_bits, record.estimated_downlink_rate_bps)
        )
        margin = record.estimated_contact_time_s / estimate_s if math.isfinite(estimate_s) and estimate_s > 0 else 0.0
        slots.append(ServerObservation(record, current_time_s - record.record_time_s, margin))
        mask.append(
            record.same_mec_region and record.available
            and record.estimated_distance_m <= config.v2v_range_m
            and record.estimated_snr_linear >= config.min_v2v_snr_linear
            and record.available_cpu_hz > 0.0
            and record.battery_fraction > record.reserve_energy_fraction
        )
    if len(slots) > config.max_server_vehicles:
        raise ValueError("snapshot exceeds fixed server action capacity")
    slots.extend([None] * (config.max_server_vehicles - len(slots)))
    mask.extend([False] * (config.max_server_vehicles - len(mask) + 1))
    age_s = user.information_age_s(current_time_s)
    arrival_s = task.arrival_slot * config.slot_duration_s
    return OffloadingObservation(
        RAW_OBSERVATION_SCHEMA_VERSION, user.vehicle_id, task.task_id, task.input_size_bits, task.output_size_bits,
        task.required_cpu_cycles, task.remaining_cpu_cycles,
        task.deadline_s,
        task.absolute_deadline_s(config.slot_duration_s) - current_time_s,
        int(task.criticality), task.energy_budget_j,
        (current_time_s - arrival_s) / task.deadline_s,
        user.battery_energy_j / user.battery_capacity_j, user.speed_mps, user.heading_rad,
        distance_m(user.position_m, mec.position_m), user.tx_power_w / user.max_tx_power_w,
        len(user.task_queue), sum(tasks[task_id].required_cpu_cycles for task_id in user.task_queue),
        age_s, current_time_s - user.snapshot_received_time_s,
        max(0.0, config.information_lifetime_s - age_s),
        age_s / config.information_lifetime_s, user.query_status, user.query_remaining_time_s,
        snapshot.mec, 1.0 - snapshot.mec.available_cpu_hz / snapshot.mec.cpu_capacity_hz,
        tuple(slots), tuple(mask),
    )
