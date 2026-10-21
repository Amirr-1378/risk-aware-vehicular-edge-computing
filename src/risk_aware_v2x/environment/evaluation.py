"""Terminal diagnostics, never passed to an offloading policy."""

import math

from ..entities import Criticality, ExecutorKind
from ..risk.tails import empirical_var
from .simulator import SlottedEnvironment


def terminal_metrics(environment: SlottedEnvironment) -> dict:
    if not environment.terminated or any(not task.is_terminal for task in environment.tasks.values()):
        raise ValueError("complete drain is required before scientific episode metrics")
    tasks = tuple(environment.tasks.values())
    n = len(tasks)
    work = math.fsum(task.required_cpu_cycles for task in tasks)
    critical = [task for task in tasks if task.criticality is Criticality.CRITICAL]
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
        "critical_task_count": len(critical),
        "critical_deadline_miss_fraction": sum(task.deadline_missed for task in critical) / len(critical) if critical else None,
        "mean_task_latency_s": math.fsum(latency) / n if n else None,
        "pooled_task_latency_p95_s": empirical_var(latency, alpha=.95) if latency else None,
        "pooled_task_latency_p99_s": empirical_var(latency, alpha=.99) if latency else None,
        "mean_mobile_energy_per_task_j": metrics["mobile_energy_j"] / n if n else None,
        "mean_query_round_trip_s": math.fsum(query_durations) / len(query_durations) if query_durations else None,
        "sample_unit_for_cost_risk": "one full-drain episode total; tasks within an episode are not independent samples",
    }
