"""Myopic plug-in combined-cost heuristic, NOT a learned expectation/oracle."""

import math
from dataclasses import dataclass, fields

from ..entities._validation import (
    require_finite_float, require_fraction, require_non_negative_float,
    require_non_negative_int, require_positive_float,
)
from ..environment.observations import RAW_OBSERVATION_SCHEMA_VERSION, OffloadingObservation, observed_valid_actions
from ..models.energy import cpu_energy_j, radio_energy_j
from ..models.latency import transmission_time_s
from .simple import estimated_service_time_s


@dataclass(frozen=True, slots=True)
class GreedyCostParameters:
    """Public static model assumptions, frozen before action selection.

    Weights are dimensionless; powers are W; kappa is J/(cycle*Hz^2).
    The experiment must supply matching shared objective/hardware constants.
    Heterogeneous hardware requires a separately specified observation/model.
    """
    latency_weight: float
    energy_weight: float
    mec_weight: float
    deadline_penalty: float
    user_max_tx_power_w: float
    user_rx_power_w: float
    server_tx_power_w: float
    server_rx_power_w: float
    server_cpu_energy_coefficient: float
    require_cached_contact: bool = True

    def __post_init__(self):
        for item in fields(self):
            if item.name != "require_cached_contact":
                object.__setattr__(self, item.name, require_non_negative_float(item.name, getattr(self, item.name)))
        if not isinstance(self.require_cached_contact, bool):
            raise TypeError("require_cached_contact must be boolean")


@dataclass(frozen=True, slots=True)
class EstimatedActionCost:
    """Point projection, not a probability or actual task ledger.

    Infinite service/energy/cost denote unavailable service estimates. Cost is
    also infinite when the optional cached-contact constraint fails. Such
    estimates must not be serialized as finite scientific outcome samples.
    """
    service_time_s: float
    mobile_energy_j: float
    remaining_cached_contact_s: float | None
    new_deadline_miss: bool
    admissible: bool
    cost: float


class MinimumEstimatedCombinedCost:
    """Minimize an observed plug-in remaining-task cost; ties use action index.

    Future contention, arrivals, fading, and fallback are not predicted. The
    optional contact guard uses only aged cached contact estimates, not truth.
    Already incurred latency/query energy and deadline penalties are sunk;
    they remain in the simulator outcome, not in the action-dependent score.
    """
    def __init__(self, parameters: GreedyCostParameters):
        if not isinstance(parameters, GreedyCostParameters):
            raise TypeError("parameters must be GreedyCostParameters")
        self.parameters = parameters

    def estimate(self, observation: OffloadingObservation, action: int) -> EstimatedActionCost:
        if observation.schema_version != RAW_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("combined-cost estimates require raw observation schema v2")
        # This also validates the original mask/action without querying truth.
        service_s = estimated_service_time_s(observation, action)
        deadline_s = require_positive_float("deadline_s", observation.deadline_s)
        budget_j = require_positive_float("energy_budget_j", observation.energy_budget_j)
        slack_s = require_finite_float("deadline_slack_s", observation.deadline_slack_s)
        criticality = require_non_negative_int("criticality_weight", observation.criticality_weight)
        if criticality not in (1, 2, 4):
            raise ValueError("unsupported criticality weight")
        p = self.parameters
        fraction = require_fraction("tx_power_fraction", observation.tx_power_fraction)
        contact_s = None
        if action == 0:
            up_rate, down_rate = observation.mec.estimated_uplink_rate_bps, observation.mec.estimated_downlink_rate_bps
        else:
            server = observation.servers[action - 1]
            record = server.record
            age_s = require_non_negative_float("record_age_s", server.record_age_s)
            contact_s = max(0., record.estimated_contact_time_s - age_s)
            up_rate, down_rate = record.estimated_rate_bps, record.estimated_downlink_rate_bps
        new_miss = 0. <= slack_s < service_s
        if not math.isfinite(service_s):
            return EstimatedActionCost(service_s, math.inf, contact_s, new_miss, False, math.inf)
        up_s = transmission_time_s(observation.input_size_bits, up_rate)
        down_s = transmission_time_s(observation.output_size_bits, down_rate)
        energies = [radio_energy_j(p.user_max_tx_power_w * fraction, up_s),
                    radio_energy_j(p.user_rx_power_w, down_s)]
        if action != 0:
            energies.extend((radio_energy_j(p.server_rx_power_w, up_s),
                             radio_energy_j(p.server_tx_power_w, down_s),
                             cpu_energy_j(observation.required_cpu_cycles, record.available_cpu_hz,
                                          p.server_cpu_energy_coefficient)))
        energy_j = require_non_negative_float("estimated_mobile_energy_j", math.fsum(energies))
        admissible = action == 0 or not p.require_cached_contact or service_s <= contact_s
        cost = criticality * math.fsum((p.latency_weight * service_s / deadline_s,
            p.energy_weight * energy_j / budget_j, p.mec_weight * (action == 0),
            p.deadline_penalty * new_miss)) if admissible else math.inf
        if admissible:
            cost = require_non_negative_float("estimated_cost", cost)
        return EstimatedActionCost(service_s, energy_j, contact_s, new_miss, admissible, cost)

    def act(self, observation: OffloadingObservation) -> int:
        # MEC is always retained; an all-infinite point projection resolves to
        # MEC, whose eventual positive service is a kernel assumption.
        return min(observed_valid_actions(observation), key=lambda action: (
            self.estimate(observation, action).cost, action,
        ))
