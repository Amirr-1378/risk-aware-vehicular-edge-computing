"""Observation-only comparison rules; not the paper best-response baseline."""

from typing import Protocol

from ..environment.observations import OffloadingObservation, observed_valid_actions as valid_actions
from ..models.latency import computation_time_s, transmission_time_s
from ..models.randomness import RandomStreams


class OffloadingPolicy(Protocol):
    def act(self, observation: OffloadingObservation) -> int:
        """Return one action from the received observation's mask."""
        ...


def estimated_service_time_s(observation: OffloadingObservation, action: int) -> float:
    """Cached UL + backlog/CPU + task/CPU + DL, not a completion guarantee.

    Excludes already incurred query delay, future arrivals, slot-boundary gates,
    and future contention. Infinity is allowed for an unavailable cached rate.
    """
    if isinstance(action, bool) or action not in valid_actions(observation):
        raise ValueError("action must be a valid integer index")
    if not isinstance(action, int):
        raise TypeError("action must be an integer")
    if action == 0:
        record = observation.mec
        uplink, downlink = record.estimated_uplink_rate_bps, record.estimated_downlink_rate_bps
        # Use the provided backlog delay once; do not count backlog twice.
        wait = record.estimated_queue_delay_s
        cpu_hz = record.available_cpu_hz
    else:
        record = observation.servers[action - 1].record
        uplink, downlink = record.estimated_rate_bps, record.estimated_downlink_rate_bps
        wait = computation_time_s(record.queue_workload_cycles, record.available_cpu_hz)
        cpu_hz = record.available_cpu_hz
    return (transmission_time_s(observation.input_size_bits, uplink) + wait
            + computation_time_s(observation.required_cpu_cycles, cpu_hz)
            + transmission_time_s(observation.output_size_bits, downlink))


class AlwaysMEC:
    def act(self, observation: OffloadingObservation) -> int:
        valid_actions(observation)
        return 0


class AlwaysV2V:
    """Prefer any observed-valid V2V; choose minimum cached service time among SVs.

    If none are observed-valid, select MEC. Any subsequent true-state rejection
    and full-restart MEC recovery belong to the common simulator, not this rule.
    """
    def act(self, observation: OffloadingObservation) -> int:
        candidates = valid_actions(observation)[1:]
        if not candidates:
            return 0
        return min(candidates, key=lambda action: (
            estimated_service_time_s(observation, action), action,
        ))


class MinimumEstimatedLatency:
    """Supplementary latency-only heuristic, NOT greedy combined-cost control."""
    def act(self, observation: OffloadingObservation) -> int:
        return min(valid_actions(observation), key=lambda action: (
            estimated_service_time_s(observation, action), action,
        ))


class UniformRandom:
    """Uniform over observed-valid actions with explicit independent per-user RNGs."""
    def __init__(self, *, seed: int) -> None:
        self._streams = RandomStreams(seed)

    def act(self, observation: OffloadingObservation) -> int:
        return self._streams.stream(f"baseline-uniform:user:{observation.user_vehicle_id}").choice(
            valid_actions(observation)
        )
