"""Immutable metadata records exchanged between the MEC and user vehicles."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    require_fraction,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
)


@dataclass(frozen=True, slots=True)
class MECMetadataRecord:
    """MEC resource information frozen at snapshot generation time.

    CPU values are measured in cycle/s, workloads in cycles, bandwidths in Hz,
    and delays in seconds.
    """

    server_id: int
    cpu_capacity_hz: float
    available_cpu_hz: float
    queue_workload_cycles: float
    estimated_queue_delay_s: float
    available_uplink_bandwidth_hz: float
    available_downlink_bandwidth_hz: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "server_id", require_non_negative_int("server_id", self.server_id)
        )
        object.__setattr__(
            self,
            "cpu_capacity_hz",
            require_positive_float("cpu_capacity_hz", self.cpu_capacity_hz),
        )
        object.__setattr__(
            self,
            "available_cpu_hz",
            require_non_negative_float("available_cpu_hz", self.available_cpu_hz),
        )
        if self.available_cpu_hz > self.cpu_capacity_hz:
            raise ValueError("available_cpu_hz cannot exceed cpu_capacity_hz")
        for field_name in (
            "queue_workload_cycles",
            "estimated_queue_delay_s",
            "available_uplink_bandwidth_hz",
            "available_downlink_bandwidth_hz",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_negative_float(field_name, getattr(self, field_name)),
            )


@dataclass(frozen=True, slots=True)
class ServerVehicleMetadata:
    """User-specific cached record for one candidate server vehicle.

    SNR is a linear ratio, rate is in bit/s, distances are in metres, and time
    values are in seconds. Rate, SNR, and distance are produced by the channel
    model; this record only stores their consistent result.
    """

    server_vehicle_id: int
    record_time_s: float
    available_cpu_hz: float
    queue_workload_cycles: float
    battery_fraction: float
    estimated_snr_linear: float
    estimated_rate_bps: float
    estimated_distance_m: float
    estimated_contact_time_s: float
    available: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "server_vehicle_id",
            require_non_negative_int("server_vehicle_id", self.server_vehicle_id),
        )
        object.__setattr__(
            self,
            "record_time_s",
            require_non_negative_float("record_time_s", self.record_time_s),
        )
        for field_name in (
            "available_cpu_hz",
            "queue_workload_cycles",
            "estimated_snr_linear",
            "estimated_rate_bps",
            "estimated_distance_m",
            "estimated_contact_time_s",
        ):
            object.__setattr__(
                self,
                field_name,
                require_non_negative_float(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "battery_fraction",
            require_fraction("battery_fraction", self.battery_fraction),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")


@dataclass(frozen=True, slots=True)
class MECSnapshot:
    """Immutable MEC response generated at ``generated_time_s``.

    Reception time is intentionally not stored here: each user records it when
    the response arrives, while information age is measured from generation.
    """

    generated_time_s: float
    mec: MECMetadataRecord
    server_vehicles: tuple[ServerVehicleMetadata, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generated_time_s",
            require_non_negative_float("generated_time_s", self.generated_time_s),
        )
        if not isinstance(self.mec, MECMetadataRecord):
            raise TypeError("mec must be an MECMetadataRecord")
        records = tuple(self.server_vehicles)
        if not all(isinstance(record, ServerVehicleMetadata) for record in records):
            raise TypeError("server_vehicles must contain ServerVehicleMetadata records")
        ids = [record.server_vehicle_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("server_vehicles must not contain duplicate identifiers")
        object.__setattr__(self, "server_vehicles", records)

    def server_record(self, server_vehicle_id: int) -> ServerVehicleMetadata | None:
        """Return the cached record with the requested identifier, if present."""
        target_id = require_non_negative_int("server_vehicle_id", server_vehicle_id)
        return next(
            (
                record
                for record in self.server_vehicles
                if record.server_vehicle_id == target_id
            ),
            None,
        )
