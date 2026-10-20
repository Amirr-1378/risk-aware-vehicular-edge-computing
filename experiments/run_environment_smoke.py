"""Synthetic integration demonstration; not a paper benchmark or trained baseline.

PowerShell: $env:PYTHONPATH = "src"
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict

from risk_aware_v2x.entities import Criticality, MECServer, ServerVehicle, Task, UserVehicle
from risk_aware_v2x.environment import EnvironmentConfig, SlottedEnvironment
from risk_aware_v2x.models.channel import LogDistanceChannel


def build_smoke_scenario() -> tuple[SlottedEnvironment, dict]:
    """Build shared scenario prototypes and a pre-reset reproducibility manifest."""
    channel = LogDistanceChannel(5.9e9, 2.0)
    config = EnvironmentConfig(
        arrival_horizon_slots=25, max_server_vehicles=2,
        v2i_uplink=channel, v2i_downlink=channel, v2v=channel,
        noise_density_w_per_hz=4e-21, v2v_bandwidth_hz=1e6,
        v2v_range_m=100.0, min_v2v_snr_linear=1.0,
        contact_prediction_horizon_s=5.0,
        query_request_bits=256, query_header_bits=256, query_record_bits=512,
        query_build_energy_j=1e-6, max_tasks_per_user=3,
        max_task_input_bits=50_000, max_task_output_bits=1_000,
    )
    users = [UserVehicle(
        vehicle_id=user_id, max_tx_power_w=1.0, rx_power_w=.5, idle_power_w=.1,
        battery_capacity_j=100.0, radio_antenna_gain_linear=1.0,
        max_task_queue_size=10, mobility_profile_id="stationary",
        position_m=(5.0 * (user_id - 1), 0.0),
    ) for user_id in (1, 2)]
    mec = MECServer(
        server_id=0, position_m=(0.0, 0.0), coverage_radius_m=200.0,
        cpu_capacity_hz=2e9, max_concurrent_tasks=2,
        total_uplink_bandwidth_hz=1e6, total_downlink_bandwidth_hz=1e6,
        max_downlink_power_w=10.0, query_processing_time_s=.002,
    )
    servers = [ServerVehicle(
        vehicle_id=server_id, cpu_capacity_hz=5e8, min_self_reserved_cpu_hz=5e7,
        max_concurrent_tasks=1, max_queue_size=10, max_tx_power_w=1.0,
        rx_power_w=.5, cpu_energy_coefficient=1e-27, battery_capacity_j=100.0,
        min_reserve_energy_j=1.0, radio_antenna_gain_linear=1.0,
        mobility_profile_id="outbound" if server_id == 9 else "stationary",
        position_m=(95.0, 0.0) if server_id == 9 else (-20.0, 0.0),
        speed_mps=100.0 if server_id == 9 else 0.0,
    ) for server_id in (9, 11)]
    trace = [Task(
        task_id=2 * index + user_id, owner_vehicle_id=user_id,
        arrival_slot=8 * index, input_size_bits=30_000 + 10_000 * index,
        output_size_bits=1_000, cpu_cycles_per_bit=500.0,
        deadline_s=.15, criticality=Criticality.CRITICAL if index == 1 else Criticality.NORMAL,
        energy_budget_j=.1,
    ) for index in range(3) for user_id in (1, 2)]
    env = SlottedEnvironment(config=config, users=users, server_vehicles=servers,
                             mec=mec, task_trace=trace)
    scenario = {
        "users": [_initial_record(user) for user in users],
        "servers": [_initial_record(server) for server in servers],
        "mec": _initial_record(mec), "tasks": [_initial_record(task) for task in trace],
    }
    return env, scenario


def run_smoke(*, seed: int = 7, policy: str = "first_cached_server") -> dict:
    """Return a complete reproducibility manifest and auditable terminal ledgers."""
    if policy not in {"mec", "first_cached_server"}:
        raise ValueError("unknown smoke policy")
    env, scenario = build_smoke_scenario()
    env.reset(seed=seed)
    total_returned_cost = 0.0
    while not env.terminated:
        actions = {}
        for user_id, observation in env.observations.items():
            available = [index for index, valid in enumerate(observation.action_mask) if index > 0 and valid]
            actions[user_id] = available[0] if policy == "first_cached_server" and available else 0
        total_returned_cost += sum(env.step(actions).costs.values())
    return {
        "report_schema_version": 1, "purpose": "synthetic integration smoke test",
        "seed": seed, "python_version": platform.python_version(), "policy": policy,
        "configuration": asdict(env.config), "scenario": scenario,
        "elapsed_slots": env.slot, "metrics": env.metrics(),
        "sum_returned_cost": total_returned_cost,
        "tasks": [{
            "task_id": task_id, "status": task.status.value,
            "final_executor_kind": task.final_executor_kind.value,
            "final_executor_id": task.final_executor_id, "fallback_used": task.fallback_used,
            "completion_time_s": task.completion_time_s,
            "latency": asdict(env.ledgers[task_id].latency),
            "mobile_energy": asdict(env.ledgers[task_id].energy),
            "total_cost": env.ledgers[task_id].total_cost,
        } for task_id, task in sorted(env.tasks.items())],
        "events": [asdict(event) for event in env.events],
    }


def _initial_record(entity) -> dict:
    return {name: value for name, value in asdict(entity).items() if not name.startswith("_")}


def _json_value(value):
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("mec", "first_cached_server"), default="first_cached_server")
    args = parser.parse_args()
    print(json.dumps(run_smoke(seed=args.seed, policy=args.policy), indent=2, default=_json_value, allow_nan=False))
