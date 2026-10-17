"""Criticality-aware EDF planning and explicit CPU queue workload in cycles."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..entities import MECServer, ServerVehicle, Task, TaskStatus
from ..entities._validation import (
    require_non_negative_float,
    require_positive_float,
    require_positive_int,
    require_unique_ids,
)


def ordered_tasks(tasks: Iterable[Task], slot_duration_s: float) -> tuple[Task, ...]:
    """Order by criticality, absolute deadline, arrival slot, then stable task ID."""
    duration = require_positive_float("slot_duration_s", slot_duration_s)
    items = tuple(tasks)
    if not all(isinstance(task, Task) for task in items):
        raise TypeError("tasks must contain Task instances")
    require_unique_ids("task_id", (task.task_id for task in items))
    return tuple(sorted(items, key=lambda task: task.scheduling_key(duration)))


def queue_workload_cycles(tasks: Iterable[Task]) -> float:
    """Sum remaining CPU cycles in a caller-selected queue, including preempted work."""
    return require_non_negative_float("queue_workload_cycles", math.fsum(
        require_non_negative_float("remaining_cpu_cycles", task.remaining_cpu_cycles)
        for task in tasks
    ))


def next_queue_workload_cycles(
    current_cycles: float, served_cycles: float, arrived_cycles: float
) -> float:
    """Apply max(0, Q - served) + arrivals, with service before new arrivals."""
    current = require_non_negative_float("current_cycles", current_cycles)
    served = require_non_negative_float("served_cycles", served_cycles)
    arrived = require_non_negative_float("arrived_cycles", arrived_cycles)
    return require_non_negative_float("next_queue_workload_cycles", max(0.0, current - served) + arrived)


@dataclass(frozen=True, slots=True)
class CPUAllocationPlan:
    """A deterministic slot-boundary schedule; allocations are in cycle/s."""

    active_task_ids: tuple[int, ...]
    waiting_task_ids: tuple[int, ...]
    allocations_hz: tuple[tuple[int, float], ...]


def plan_cpu_schedule(
    tasks: Iterable[Task], *, service_cpu_hz: float,
    max_concurrent_tasks: int, slot_duration_s: float,
) -> CPUAllocationPlan:
    """Select the first k EDF tasks and share all service CPU equally between them.

    The final allocation receives any floating-point residual. Zero service
    CPU selects no active tasks. Plans do not advance time or mutate tasks.
    The caller supplies only tasks ready for computation at this resource.
    """
    cpu = require_non_negative_float("service_cpu_hz", service_cpu_hz)
    concurrency = require_positive_int("max_concurrent_tasks", max_concurrent_tasks)
    ordered = ordered_tasks(tasks, slot_duration_s)
    for task in ordered:
        require_positive_float("remaining_cpu_cycles", task.remaining_cpu_cycles)
    count = min(concurrency, len(ordered)) if cpu > 0.0 else 0
    selected = ordered[:count]
    allocations: list[tuple[int, float]] = []
    for index, task in enumerate(selected):
        allocation = cpu / count if index < count - 1 else cpu - sum(value for _, value in allocations)
        require_positive_float("allocation_hz", allocation)
        allocations.append((task.task_id, allocation))
    return CPUAllocationPlan(
        tuple(task.task_id for task in selected),
        tuple(task.task_id for task in ordered[count:]),
        tuple(allocations),
    )


def reschedule_resource(
    resource: MECServer | ServerVehicle, tasks: Mapping[int, Task], *, slot_duration_s: float,
) -> CPUAllocationPlan:
    """Apply EDF preemption at a slot boundary without changing work or executor.

    This function must be called by the environment only at slot boundaries.
    All validation happens before task/resource mutation. Queue/active exchange
    is applied together, so a critical task can preempt through a full waiting
    queue without needing a temporary extra queue slot.
    """
    if isinstance(resource, MECServer):
        service_cpu_hz = resource.cpu_capacity_hz - resource.background_cpu_load_hz
        queued_status, executing_status = TaskStatus.MEC_QUEUED, TaskStatus.MEC_EXECUTING
    elif isinstance(resource, ServerVehicle):
        service_cpu_hz = resource.cpu_capacity_hz - resource.min_self_reserved_cpu_hz - resource.self_workload_cpu_hz
        queued_status, executing_status = TaskStatus.V2V_QUEUED, TaskStatus.V2V_EXECUTING
        if not resource.availability or resource.battery_energy_j <= resource.min_reserve_energy_j:
            raise ValueError("unavailable server requires MEC fallback before CPU scheduling")
    else:
        raise TypeError("resource must be an MECServer or ServerVehicle")
    active_ids = set(resource.active_task_ids)
    candidate_ids = require_unique_ids(
        "resource task_id", [*resource.service_queue, *sorted(active_ids)]
    )
    candidates: list[Task] = []
    for task_id in candidate_ids:
        task = tasks[task_id]
        if task.task_id != task_id:
            raise ValueError("task mapping key does not match task ID")
        expected = executing_status if task_id in active_ids else queued_status
        if task.status is not expected:
            raise ValueError("resource membership and task computation status disagree")
        candidates.append(task)
    plan = plan_cpu_schedule(
        candidates, service_cpu_hz=service_cpu_hz,
        max_concurrent_tasks=resource.max_concurrent_tasks, slot_duration_s=slot_duration_s,
    )
    if isinstance(resource, ServerVehicle) and len(plan.waiting_task_ids) > resource.max_queue_size:
        raise OverflowError("schedule exceeds finite server queue; resolve interruption before scheduling")
    selected_ids = set(plan.active_task_ids)
    for task in candidates:
        next_status = executing_status if task.task_id in selected_ids else queued_status
        if task.status is not next_status:
            task.transition_to(next_status)
    resource.service_queue = list(plan.waiting_task_ids)
    resource.active_task_ids = selected_ids
    resource.cpu_allocations_hz = dict(plan.allocations_hz)
    return plan
