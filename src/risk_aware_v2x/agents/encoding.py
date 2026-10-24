"""Versioned, fixed-size, dimensionless encoding of received observations only."""

import math
from dataclasses import dataclass, fields

from ..entities import QueryStatus
from ..entities._validation import require_finite_float, require_non_negative_int, require_positive_float
from ..environment.observations import RAW_OBSERVATION_SCHEMA_VERSION, OffloadingObservation, observed_valid_actions

ENCODED_OBSERVATION_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class FeatureScales:
    """Fixed reference scales, NOT sample statistics or clipping thresholds.

    Defaults are illustrative SI references, not calibrated paper parameters.
    Freeze these before training; persist the entire object with checkpoints.
    """
    input_bits: float = 1e6
    output_bits: float = 1e5
    cpu_cycles: float = 1e9
    time_s: float = 1.0
    energy_j: float = 1.0
    speed_mps: float = 30.0
    distance_m: float = 100.0
    cpu_hz: float = 1e9
    bandwidth_hz: float = 1e6
    rate_bps: float = 1e7
    snr_linear: float = 100.0
    task_count: float = 10.0
    contact_margin: float = 10.0

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, require_positive_float(field.name, getattr(self, field.name)))


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Only values and mask go into a network; identifiers are diagnostic labels."""
    schema_version: int
    values: tuple[float, ...]
    action_mask: tuple[bool, ...]
    user_vehicle_id: int
    task_id: int


class ObservationEncoder:
    """Raw schema v2 -> vector schema v2. No running statistics or simulator access.

    No clipping: negative deadline slack and unusually large workloads remain
    distinguishable. Non-finite inputs/results are errors. Absent server blocks
    are entirely zero, including presence; present-but-masked blocks are kept.
    """
    schema_version = ENCODED_OBSERVATION_SCHEMA_VERSION
    _query_order = (
        QueryStatus.NO_CACHE, QueryStatus.CACHE_VALID, QueryStatus.CACHE_EXPIRED,
        QueryStatus.QUERY_REQUEST_UPLINK, QueryStatus.QUERY_MEC_PROCESSING,
        QueryStatus.QUERY_RESPONSE_DOWNLINK,
    )
    _global_names = (
        "task.input", "task.output", "task.required_cycles", "task.remaining_cycles",
        "task.deadline", "task.deadline_slack", "task.energy_budget", "task.elapsed_deadline_fraction",
        "task.criticality_normal", "task.criticality_high", "task.criticality_critical",
        "user.battery_fraction", "user.speed", "user.heading_sin", "user.heading_cos",
        "user.distance_to_mec", "user.tx_power_fraction", "user.unassigned_count",
        "user.unassigned_cycles", "cache.snapshot_age", "cache.response_age",
        "cache.remaining_lifetime", "cache.normalized_age", "query.remaining_time",
        "mec.cpu_capacity", "mec.available_cpu", "mec.queue_cycles", "mec.queue_delay",
        "mec.uplink_bandwidth", "mec.downlink_bandwidth", "mec.uplink_rate",
        "mec.downlink_rate", "mec.utilization",
    )
    _server_names = (
        "present", "available_cpu", "queue_cycles", "battery_fraction", "snr",
        "uplink_rate", "downlink_rate", "distance", "contact_time", "available",
        "reserve_fraction", "relative_speed", "same_mec_region", "record_age", "contact_margin",
    )

    def __init__(self, *, max_server_vehicles: int, scales: FeatureScales) -> None:
        self.max_server_vehicles = require_non_negative_int("max_server_vehicles", max_server_vehicles)
        if not isinstance(scales, FeatureScales):
            raise TypeError("scales must be FeatureScales")
        self.scales = scales

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (self._global_names + tuple(f"query.{status.value}" for status in self._query_order)
                + tuple(f"server[{index}].{name}" for index in range(self.max_server_vehicles)
                        for name in self._server_names))

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def encode(self, observation: OffloadingObservation) -> EncodedObservation:
        if observation.schema_version != RAW_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported raw observation schema")
        if len(observation.servers) != self.max_server_vehicles:
            raise ValueError("observation server capacity differs from encoder")
        observed_valid_actions(observation)
        if observation.criticality_weight not in (1, 2, 4):
            raise ValueError("unsupported task criticality")
        if observation.query_status not in self._query_order:
            raise ValueError("unknown query status")
        s, o, m = self.scales, observation, observation.mec
        # Validate heading before trigonometry, which could otherwise obscure NaN.
        heading = require_finite_float("user_heading_rad", o.user_heading_rad)
        values = [
            o.input_size_bits / s.input_bits, o.output_size_bits / s.output_bits,
            o.required_cpu_cycles / s.cpu_cycles, o.remaining_cpu_cycles / s.cpu_cycles,
            require_positive_float("deadline_s", o.deadline_s) / s.time_s,
            o.deadline_slack_s / s.time_s, o.energy_budget_j / s.energy_j,
            o.elapsed_deadline_fraction, *(float(o.criticality_weight == level) for level in (1, 2, 4)),
            o.user_battery_fraction, o.user_speed_mps / s.speed_mps,
            math.sin(heading), math.cos(heading), o.distance_to_mec_m / s.distance_m,
            o.tx_power_fraction, o.unassigned_task_count / s.task_count,
            o.unassigned_workload_cycles / s.cpu_cycles,
            o.snapshot_age_s / s.time_s, o.response_age_s / s.time_s,
            o.remaining_lifetime_s / s.time_s, o.normalized_information_age,
            o.query_remaining_time_s / s.time_s, m.cpu_capacity_hz / s.cpu_hz,
            m.available_cpu_hz / s.cpu_hz, m.queue_workload_cycles / s.cpu_cycles,
            m.estimated_queue_delay_s / s.time_s,
            m.available_uplink_bandwidth_hz / s.bandwidth_hz,
            m.available_downlink_bandwidth_hz / s.bandwidth_hz,
            m.estimated_uplink_rate_bps / s.rate_bps, m.estimated_downlink_rate_bps / s.rate_bps,
            o.mec_utilization, *(float(o.query_status is status) for status in self._query_order),
        ]
        for server in o.servers:
            if server is None:
                values.extend([0.0] * len(self._server_names))
                continue
            r = server.record
            values.extend((
                1.0, r.available_cpu_hz / s.cpu_hz, r.queue_workload_cycles / s.cpu_cycles,
                r.battery_fraction, r.estimated_snr_linear / s.snr_linear,
                r.estimated_rate_bps / s.rate_bps, r.estimated_downlink_rate_bps / s.rate_bps,
                r.estimated_distance_m / s.distance_m, r.estimated_contact_time_s / s.time_s,
                float(r.available), r.reserve_energy_fraction, r.relative_speed_mps / s.speed_mps,
                float(r.same_mec_region), server.record_age_s / s.time_s, server.contact_margin / s.contact_margin,
            ))
        encoded = tuple(require_finite_float(name, value) for name, value in zip(self.feature_names, values, strict=True))
        return EncodedObservation(self.schema_version, encoded, o.action_mask, o.user_vehicle_id, o.task_id)
