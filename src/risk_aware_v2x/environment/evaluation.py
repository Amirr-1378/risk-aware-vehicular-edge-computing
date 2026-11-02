"""Terminal diagnostics, never passed to an offloading policy."""

import math

from ..entities import ExecutorKind
from ..contracts import COST_MODEL_VERSION
from ..risk.tails import empirical_var
from .simulator import SlottedEnvironment
from .accounting import final_task_cost


def terminal_metrics(environment: SlottedEnvironment) -> dict:
    if not environment.terminated or any(not task.is_terminal for task in environment.tasks.values()):
        raise ValueError("complete drain is required before scientific episode metrics")
    tasks = tuple(environment.tasks.values())
    for task in tasks:
        ledger = environment.ledgers[task.task_id]
        expected = final_task_cost(task, ledger, environment.config)
        if not math.isclose(expected, ledger.total_cost, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("task cost violates the reference three-term objective")
    n = len(tasks)
    work = math.fsum(task.required_cpu_cycles for task in tasks)
    tardiness = [max(0., task.completion_time_s - task.absolute_deadline_s(environment.config.slot_duration_s))
                 for task in tasks]
    latency = [environment.ledgers[task.task_id].latency.total_s for task in tasks]
    metrics = environment.metrics()
    starts, query_durations = {}, []
    for event in environment.events:
        if event.kind == "query_started":
            starts[event.user_vehicle_id] = event.time_s
        elif event.kind == "snapshot_received":
            query_durations.append(event.time_s - starts.pop(event.user_vehicle_id))
    return {
        **metrics,
        "mean_task_cost": metrics["total_cost"] / n if n else None,
        "mec_completed_workload_fraction": math.fsum(task.required_cpu_cycles for task in tasks
            if task.final_executor_kind is ExecutorKind.MEC) / work if work else None,
        "direct_mec_task_fraction": sum(task.initial_target_kind is ExecutorKind.MEC for task in tasks) / n if n else None,
        "fallback_task_fraction": sum(task.fallback_used for task in tasks) / n if n else None,
        "on_time_task_fraction": sum(not task.deadline_missed for task in tasks) / n if n else None,
        "cost_model_version": COST_MODEL_VERSION,
        "deadline_miss_fraction": sum(task.deadline_missed for task in tasks) / n if n else None,
        "total_task_tardiness_s": math.fsum(tardiness),
        "mean_task_tardiness_s": math.fsum(tardiness) / n if n else None,
        "max_task_tardiness_s": max(tardiness) if n else None,
        "mean_task_latency_s": math.fsum(latency) / n if n else None,
        "pooled_task_latency_p95_s": empirical_var(latency, alpha=.95) if latency else None,
        "pooled_task_latency_p99_s": empirical_var(latency, alpha=.99) if latency else None,
        "mean_mobile_energy_per_task_j": metrics["mobile_energy_j"] / n if n else None,
        "mean_query_round_trip_s": math.fsum(query_durations) / len(query_durations) if query_durations else None,
        "sample_unit_for_cost_risk": "one full-drain episode total; tasks within an episode are not independent samples",
    }
