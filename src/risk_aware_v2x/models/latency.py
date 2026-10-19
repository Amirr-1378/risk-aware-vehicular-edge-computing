"""Continuous-time estimates and exact per-phase slot progress, in seconds."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..entities._validation import require_non_negative_float, require_non_negative_int


def transmission_time_s(size_bits: float, rate_bps: float) -> float:
    """Return S/R in seconds: no data needs zero time; positive data at R=0 needs inf.

    Infinity is an estimate of unavailable service, never a valid elapsed-time
    state. Inputs must be finite and non-negative even for a zero-size transfer.
    """
    size = require_non_negative_float("size_bits", size_bits)
    rate = require_non_negative_float("rate_bps", rate_bps)
    if size == 0.0:
        return 0.0
    return size / rate if rate > 0.0 else math.inf


def computation_time_s(workload_cycles: float, allocated_cpu_hz: float) -> float:
    """Return CPU cycles/(cycle/s), using the same zero-service semantics as radio."""
    work = require_non_negative_float("workload_cycles", workload_cycles)
    cpu = require_non_negative_float("allocated_cpu_hz", allocated_cpu_hz)
    if work == 0.0:
        return 0.0
    return work / cpu if cpu > 0.0 else math.inf


def query_response_size_bits(header_size_bits: int, record_size_bits: int, record_count: int) -> int:
    """Return header + N*record, all sizes in bits and record count dimensionless."""
    header = require_non_negative_int("header_size_bits", header_size_bits)
    record = require_non_negative_int("record_size_bits", record_size_bits)
    count = require_non_negative_int("record_count", record_count)
    return header + count * record


def query_latency_s(
    *, request_size_bits: float, response_size_bits: float,
    uplink_rate_bps: float, downlink_rate_bps: float, processing_time_s: float,
) -> float:
    """Request UL time + MEC processing time + response DL time in seconds."""
    request_s = transmission_time_s(request_size_bits, uplink_rate_bps)
    response_s = transmission_time_s(response_size_bits, downlink_rate_bps)
    processing_s = require_non_negative_float("processing_time_s", processing_time_s)
    return request_s + processing_s + response_s


@dataclass(frozen=True, slots=True)
class WorkProgress:
    """Served/remaining work in bits or cycles, and actual busy time in seconds."""

    served_work: float
    remaining_work: float
    active_time_s: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))


def progress_work(remaining_work: float, service_rate_per_s: float, duration_s: float) -> WorkProgress:
    """Serve at most the remaining bits/cycles at a constant rate over one interval.

    The actual active time avoids charging a whole slot of radio/CPU energy
    when work finishes early. No work crosses into the next phase automatically.
    """
    remaining = require_non_negative_float("remaining_work", remaining_work)
    rate = require_non_negative_float("service_rate_per_s", service_rate_per_s)
    duration = require_non_negative_float("duration_s", duration_s)
    if remaining == 0.0 or rate == 0.0 or duration == 0.0:
        return WorkProgress(0.0, remaining, 0.0)
    if remaining / rate <= duration:
        return WorkProgress(remaining, 0.0, remaining / rate)
    served = rate * duration
    return WorkProgress(served, remaining - served, duration)


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Non-overlapping measured durations, all in seconds.

    The upload/queue/compute/download fields describe the final successful
    attempt. ``interrupted_attempt_s`` is all elapsed time in the discarded V2V
    attempt, with no repeated MEC service or query time. ``admission_wait_s``
    includes ready-but-not-selected waiting and any fallback admission gap.
    """

    query_wait_s: float = 0.0
    admission_wait_s: float = 0.0
    queue_wait_s: float = 0.0
    upload_s: float = 0.0
    compute_s: float = 0.0
    download_s: float = 0.0
    interrupted_attempt_s: float = 0.0
    slot_boundary_wait_s: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))

    @property
    def total_s(self) -> float:
        """Total physical elapsed time; each interval is attributed exactly once."""
        return require_non_negative_float("total_s", math.fsum(getattr(self, name) for name in self.__dataclass_fields__))


def elapsed_latency_s(arrival_time_s: float, completion_time_s: float) -> float:
    """Ground-truth latency from arrival to completion, in seconds."""
    arrival = require_non_negative_float("arrival_time_s", arrival_time_s)
    completion = require_non_negative_float("completion_time_s", completion_time_s)
    if completion < arrival:
        raise ValueError("completion cannot precede arrival")
    return completion - arrival
