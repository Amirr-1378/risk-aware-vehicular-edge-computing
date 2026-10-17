"""Mobile radio/CPU energy accounting; infrastructure MEC energy is excluded."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..entities._validation import require_non_negative_float


def radio_energy_j(power_w: float, active_time_s: float) -> float:
    """Energy P*T in joules for a radio TX/RX or an idle interval."""
    power = require_non_negative_float("power_w", power_w)
    duration = require_non_negative_float("active_time_s", active_time_s)
    return require_non_negative_float("energy_j", power * duration)


def cpu_energy_j(
    executed_cycles: float, allocated_cpu_hz: float, cpu_energy_coefficient: float
) -> float:
    """Return kappa*C*f² in joules; kappa is J/(cycle*Hz²), C is actual work.

    Zero work costs zero dynamic CPU energy. Executing positive work at zero
    frequency is invalid. Leakage and server self-workload energy are excluded.
    """
    cycles = require_non_negative_float("executed_cycles", executed_cycles)
    cpu = require_non_negative_float("allocated_cpu_hz", allocated_cpu_hz)
    coefficient = require_non_negative_float("cpu_energy_coefficient", cpu_energy_coefficient)
    if cycles > 0.0 and cpu == 0.0:
        raise ValueError("positive executed cycles require positive CPU allocation")
    if cycles == 0.0 or coefficient == 0.0:
        return 0.0
    return require_non_negative_float("cpu_energy_j", coefficient * cycles * cpu * cpu)


def query_energy_j(
    *, build_energy_j: float, tx_power_w: float, request_time_s: float,
    idle_power_w: float, processing_time_s: float,
    rx_power_w: float, response_time_s: float,
) -> float:
    """User query build + request TX + processing wait + response RX energy.

    This sum is charged once to the user and attributed to its query-owner
    task; it must not also be included in that task's other radio/idle buckets.
    """
    build = require_non_negative_float("build_energy_j", build_energy_j)
    return require_non_negative_float("query_energy_j", math.fsum((
        build, radio_energy_j(tx_power_w, request_time_s),
        radio_energy_j(idle_power_w, processing_time_s),
        radio_energy_j(rx_power_w, response_time_s),
    )))


@dataclass(frozen=True, slots=True)
class MobileEnergyBreakdown:
    """Non-overlapping task energy buckets in J, for MEC, V2V, or fallback.

    User/server buckets describe the final attempt; ``interrupted_attempt_j``
    retains all mobile energy from the discarded V2V attempt. Query energy
    belongs only to the query-triggering task. For direct MEC service all
    server-vehicle fields are zero. There is no MEC-energy field.
    """

    query_j: float = 0.0
    user_tx_j: float = 0.0
    user_rx_j: float = 0.0
    user_idle_j: float = 0.0
    server_rx_j: float = 0.0
    server_cpu_j: float = 0.0
    server_tx_j: float = 0.0
    interrupted_attempt_j: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))

    @property
    def total_j(self) -> float:
        """All task-attributable mobile energy, with no infrastructure energy."""
        return require_non_negative_float("total_j", math.fsum(getattr(self, name) for name in self.__dataclass_fields__))
