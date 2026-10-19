"""Explicit, versionable parameters for the first slotted simulator."""

from dataclasses import dataclass

from ..entities._validation import (
    require_non_negative_float, require_non_negative_int,
    require_positive_float, require_positive_int,
)
from ..models.channel import LogDistanceChannel


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """SI-valued scenario inputs; no traffic/radio calibration is implicit.

    Arrival horizon and guard are in slots. The first simulator requires
    deterministic V2I channels so finite conservative fallback-energy bounds
    can be verified without imposing an artificial rate floor.
    """

    arrival_horizon_slots: int
    max_server_vehicles: int
    v2i_uplink: LogDistanceChannel
    v2i_downlink: LogDistanceChannel
    v2v: LogDistanceChannel
    noise_density_w_per_hz: float
    v2v_bandwidth_hz: float
    v2v_range_m: float
    min_v2v_snr_linear: float
    contact_prediction_horizon_s: float
    query_request_bits: int
    query_header_bits: int
    query_record_bits: int
    query_build_energy_j: float
    max_tasks_per_user: int
    max_task_input_bits: int
    max_task_output_bits: int
    slot_duration_s: float = 0.010
    information_lifetime_s: float = 0.100
    outage_persistence_slots: int = 1
    max_episode_slots: int = 100_000
    latency_weight: float = 0.60
    energy_weight: float = 0.25
    mec_weight: float = 0.15
    deadline_penalty: float = 2.0

    def __post_init__(self) -> None:
        for name in ("arrival_horizon_slots", "outage_persistence_slots", "max_episode_slots",
                     "max_tasks_per_user", "max_task_input_bits"):
            object.__setattr__(self, name, require_positive_int(name, getattr(self, name)))
        for name in ("max_server_vehicles", "query_request_bits", "query_header_bits",
                     "query_record_bits", "max_task_output_bits"):
            object.__setattr__(self, name, require_non_negative_int(name, getattr(self, name)))
        for name in (
            "noise_density_w_per_hz", "v2v_bandwidth_hz", "v2v_range_m",
            "contact_prediction_horizon_s", "slot_duration_s", "information_lifetime_s",
        ):
            object.__setattr__(self, name, require_positive_float(name, getattr(self, name)))
        for name in (
            "min_v2v_snr_linear", "query_build_energy_j", "latency_weight",
            "energy_weight", "mec_weight", "deadline_penalty",
        ):
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))
        if self.max_episode_slots < self.arrival_horizon_slots:
            raise ValueError("episode guard cannot precede the arrival horizon")
        for name in ("v2i_uplink", "v2i_downlink", "v2v"):
            if not isinstance(getattr(self, name), LogDistanceChannel):
                raise TypeError(f"{name} must be a LogDistanceChannel")
        for channel in (self.v2i_uplink, self.v2i_downlink):
            if channel.shadowing_std_db != 0.0 or channel.rayleigh_fading:
                raise ValueError("guaranteed MEC service currently requires deterministic V2I")
