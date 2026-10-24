"""Versioned fixed-policy suite and public static hardware assumptions."""

from dataclasses import asdict

from risk_aware_v2x.baselines import (
    AlwaysMEC, AlwaysV2V, GreedyCostParameters, MinimumEstimatedCombinedCost,
    MinimumEstimatedLatency, UniformRandom,
)

BASELINE_SUITE_VERSION = 2


def comparison_policies(config, users, servers, *, seed: int) -> tuple[dict, dict]:
    """Extract only fixed hardware constants, never live queues/positions/energy.

    The current homogeneous-radio/kappa assumption is validated explicitly;
    do not silently use the first node as a proxy for heterogeneous hardware.
    """
    users, servers = tuple(users), tuple(servers)
    if not users:
        raise ValueError("at least one user is required")

    def common(nodes, name, *, empty=0.):
        values = {getattr(node, name) for node in nodes}
        if not values:
            return empty  # No server action exists when the SV cohort is empty.
        if len(values) != 1:
            raise ValueError(f"heterogeneous {name} requires an explicit per-node observation/model")
        return next(iter(values))

    parameters = GreedyCostParameters(
        latency_weight=config.latency_weight, energy_weight=config.energy_weight,
        mec_weight=config.mec_weight, deadline_penalty=config.deadline_penalty,
        user_max_tx_power_w=common(users, "max_tx_power_w"),
        user_rx_power_w=common(users, "rx_power_w"),
        server_tx_power_w=common(servers, "max_tx_power_w"),
        server_rx_power_w=common(servers, "rx_power_w"),
        server_cpu_energy_coefficient=common(servers, "cpu_energy_coefficient"),
        require_cached_contact=True,
    )
    return {
        "always_mec": AlwaysMEC(), "always_v2v": AlwaysV2V(),
        "minimum_estimated_latency": MinimumEstimatedLatency(),
        "uniform_random": UniformRandom(seed=seed),
        "minimum_estimated_combined_cost": MinimumEstimatedCombinedCost(parameters),
    }, {
        "baseline_suite_version": BASELINE_SUITE_VERSION,
        "minimum_estimated_combined_cost": asdict(parameters),
        "information": "received snapshot and own task/user state plus public static hardware/objective constants",
        "tie_break": "smallest observation action index; MEC=0",
        "query_protocol": "mandatory shared query/cache; NOT query-free MEC",
    }
