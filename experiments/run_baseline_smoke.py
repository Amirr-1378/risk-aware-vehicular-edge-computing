"""Common-scenario policy/adapter integration audit, NOT a scientific benchmark."""

import argparse
import json
import math
import platform
from dataclasses import asdict

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from .run_environment_smoke import _json_value, build_smoke_scenario
from .comparison_policies import comparison_policies


def run_comparison(*, seed: int = 7, policy_seed: int = 17,
                   gamma_per_slot: float = 1.0) -> dict:
    environment, scenario = build_smoke_scenario()
    scales = FeatureScales()
    encoder = ObservationEncoder(max_server_vehicles=environment.config.max_server_vehicles, scales=scales)
    adapter = DecisionAdapter(environment, encoder, gamma_per_slot=gamma_per_slot)
    adapter.reset(seed=seed)
    policies, policy_configuration = comparison_policies(environment.config, environment.users.values(),
        environment.server_vehicles.values(), seed=policy_seed)
    reports = {}
    for name, policy in policies.items():
        adapter.reset(seed=seed)
        transitions, actions_taken = [], []
        while not environment.terminated:
            actions = {user_id: policy.act(observation)
                       for user_id, observation in sorted(adapter.raw_observations.items())}
            actions_taken.extend({"slot": environment.slot, "user_vehicle_id": user_id,
                                  "task_id": adapter.raw_observations[user_id].task_id,
                                  "action": action} for user_id, action in actions.items())
            transitions.extend(adapter.step(actions).transitions)
        accounting = adapter.accounting
        for user_id, total in accounting.total_cost.items():
            if not math.isclose(total, accounting.prefix_cost[user_id] + accounting.closed_transition_cost[user_id],
                                rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("raw prefix/transition accounting mismatch")
            if not math.isclose(accounting.discounted_total_cost[user_id],
                                accounting.discounted_prefix_cost[user_id]
                                + accounting.discounted_closed_transition_cost[user_id],
                                rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("discounted prefix/transition accounting mismatch")
        reports[name] = {
            "elapsed_slots": environment.slot, "metrics": environment.metrics(),
            "accounting": asdict(accounting), "actions": actions_taken,
            "transitions": [{
                "user_vehicle_id": item.observation.user_vehicle_id,
                "task_id": item.observation.task_id, "start_slot": item.start_slot,
                "action": item.action, "duration_slots": item.duration_slots,
                "cost": item.cost, "discounted_cost": item.discounted_cost,
                "bootstrap_discount": item.bootstrap_discount, "terminated": item.terminated,
                "next_task_id": None if item.next_observation is None else item.next_observation.task_id,
            } for item in transitions],
        }
    return {
        "report_schema_version": 2, "purpose": "synthetic policy/adapter integration audit, not a paper benchmark",
        "policy_configuration": policy_configuration,
        "environment_seed": seed, "policy_seed": policy_seed,
        "python_version": platform.python_version(), "gamma_per_slot": adapter.gamma_per_slot,
        "configuration": asdict(environment.config), "scenario": scenario,
        "encoder": {"schema_version": encoder.schema_version, "dimension": encoder.dimension,
                    "scales": asdict(scales), "feature_names": encoder.feature_names},
        "policies": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--gamma-per-slot", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run_comparison(seed=args.seed, policy_seed=args.policy_seed,
                                    gamma_per_slot=args.gamma_per_slot),
                     indent=2, default=_json_value, allow_nan=False))
