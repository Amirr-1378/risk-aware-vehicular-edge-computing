"""EDF order, atomic preemption, CPU conservation, and queue workload fixtures."""

import unittest

from risk_aware_v2x.entities import (
    Criticality, ExecutorKind, MECServer, ServerVehicle, Task, TaskStatus,
)
from risk_aware_v2x.models.latency import progress_work
from risk_aware_v2x.models.queueing import (
    next_queue_workload_cycles, ordered_tasks, plan_cpu_schedule,
    queue_workload_cycles, reschedule_resource,
)


def queued_task(
    task_id: int, *, criticality: Criticality = Criticality.NORMAL,
    deadline_s: float = 1.0, arrival_slot: int = 0, v2v: bool = False,
) -> Task:
    task = Task(
        task_id=task_id, owner_vehicle_id=1, arrival_slot=arrival_slot,
        input_size_bits=10, output_size_bits=1, cpu_cycles_per_bit=10.0,
        deadline_s=deadline_s, criticality=criticality, energy_budget_j=1.0,
    )
    task.transition_to(TaskStatus.READY_FOR_OFFLOADING)
    task.assign_target(ExecutorKind.SERVER_VEHICLE if v2v else ExecutorKind.MEC, 9 if v2v else 0)
    task.serve_upload(10)
    task.transition_to(TaskStatus.V2V_QUEUED if v2v else TaskStatus.MEC_QUEUED)
    return task


def mec_resource() -> MECServer:
    return MECServer(
        server_id=0, position_m=(0.0, 0.0), coverage_radius_m=100.0,
        cpu_capacity_hz=100.0, max_concurrent_tasks=1,
        total_uplink_bandwidth_hz=1e6, total_downlink_bandwidth_hz=1e6,
        max_downlink_power_w=10.0, query_processing_time_s=0.001,
        background_cpu_load_hz=20.0,
    )


def server_resource() -> ServerVehicle:
    return ServerVehicle(
        vehicle_id=9, cpu_capacity_hz=100.0, min_self_reserved_cpu_hz=20.0,
        max_concurrent_tasks=1, max_queue_size=1, max_tx_power_w=1.0,
        rx_power_w=0.1, cpu_energy_coefficient=1e-27,
        battery_capacity_j=100.0, min_reserve_energy_j=10.0,
        radio_antenna_gain_linear=1.0, mobility_profile_id="straight",
        self_workload_cpu_hz=10.0,
    )


class QueueingModelTests(unittest.TestCase):
    def test_criticality_precedes_deadline(self) -> None:
        normal = queued_task(1, deadline_s=0.001)
        critical = queued_task(2, criticality=Criticality.CRITICAL, deadline_s=10.0)
        self.assertEqual([task.task_id for task in ordered_tasks([normal, critical], 0.01)], [2, 1])

    def test_absolute_deadline_arrival_and_id_break_ties(self) -> None:
        tasks = [
            queued_task(9, deadline_s=0.5, arrival_slot=50),
            queued_task(3, deadline_s=1.0),
            queued_task(2, deadline_s=1.0),
            queued_task(1, deadline_s=0.9),
        ]
        self.assertEqual([task.task_id for task in ordered_tasks(tasks, 0.01)], [1, 2, 3, 9])
        self.assertEqual(ordered_tasks(reversed(tasks), 0.01), ordered_tasks(tasks, 0.01))

    def test_queue_workload_tracks_remaining_cycles_and_clamps_service(self) -> None:
        tasks = [queued_task(1), queued_task(2)]
        tasks[0].serve_cpu(25.0)
        self.assertEqual(queue_workload_cycles(tasks), 175.0)
        self.assertEqual(next_queue_workload_cycles(100.0, 30.0, 20.0), 90.0)
        self.assertEqual(next_queue_workload_cycles(100.0, 500.0, 20.0), 20.0)
        self.assertEqual(queue_workload_cycles([]), 0.0)

    def test_cpu_plan_shares_capacity_and_is_deterministic(self) -> None:
        tasks = [queued_task(3), queued_task(1), queued_task(2), queued_task(4)]
        plan = plan_cpu_schedule(tasks, service_cpu_hz=100.0, max_concurrent_tasks=3, slot_duration_s=0.01)
        self.assertEqual(plan.active_task_ids, (1, 2, 3))
        self.assertEqual(plan.waiting_task_ids, (4,))
        self.assertAlmostEqual(sum(dict(plan.allocations_hz).values()), 100.0)
        for _, allocation in plan.allocations_hz:
            self.assertAlmostEqual(allocation, 100.0 / 3.0)

    def test_no_cpu_selects_no_active_tasks_and_empty_plan_is_valid(self) -> None:
        plan = plan_cpu_schedule([queued_task(1)], service_cpu_hz=0.0, max_concurrent_tasks=1, slot_duration_s=0.01)
        self.assertEqual(plan.active_task_ids, ())
        self.assertEqual(plan.waiting_task_ids, (1,))
        empty = plan_cpu_schedule([], service_cpu_hz=100.0, max_concurrent_tasks=1, slot_duration_s=0.01)
        self.assertEqual(empty.allocations_hz, ())

    def test_mec_preemption_preserves_remaining_work_and_executor(self) -> None:
        resource = mec_resource()
        normal = queued_task(1)
        critical = queued_task(2, criticality=Criticality.CRITICAL)
        tasks = {1: normal, 2: critical}
        resource.enqueue_task(1)
        reschedule_resource(resource, tasks, slot_duration_s=0.01)
        progress = progress_work(normal.remaining_cpu_cycles, resource.cpu_allocations_hz[1], 0.01)
        normal.serve_cpu(progress.served_work)
        remaining_before = normal.remaining_cpu_cycles
        resource.enqueue_task(2)
        reschedule_resource(resource, tasks, slot_duration_s=0.01)
        self.assertEqual(normal.status, TaskStatus.MEC_QUEUED)
        self.assertEqual(critical.status, TaskStatus.MEC_EXECUTING)
        self.assertEqual(normal.remaining_cpu_cycles, remaining_before)
        self.assertEqual(normal.initial_target_id, 0)
        self.assertEqual(resource.cpu_allocations_hz, {2: 80.0})
        resource.release_task(2)
        reschedule_resource(resource, tasks, slot_duration_s=0.01)
        self.assertEqual(normal.status, TaskStatus.MEC_EXECUTING)
        self.assertEqual(normal.remaining_cpu_cycles, remaining_before)

    def test_server_preempts_through_a_full_waiting_queue_without_dropping(self) -> None:
        resource = server_resource()
        normal = queued_task(1, v2v=True)
        critical = queued_task(2, criticality=Criticality.CRITICAL, v2v=True)
        tasks = {1: normal, 2: critical}
        resource.enqueue_task(1)
        reschedule_resource(resource, tasks, slot_duration_s=0.01)
        resource.enqueue_task(2)
        self.assertEqual(len(resource.service_queue), resource.max_queue_size)
        reschedule_resource(resource, tasks, slot_duration_s=0.01)
        self.assertEqual(resource.service_queue, [1])
        self.assertEqual(resource.active_task_ids, {2})
        self.assertEqual(resource.cpu_allocations_hz, {2: 70.0})
        self.assertEqual({*resource.service_queue, *resource.active_task_ids}, {1, 2})
        self.assertEqual(normal.initial_target_id, 9)
        self.assertEqual(normal.remaining_cpu_cycles, 100.0)

    def test_repeated_scheduling_reuses_total_service_capacity_not_residual_cpu(self) -> None:
        resource = mec_resource()
        tasks = {1: queued_task(1)}
        resource.enqueue_task(1)
        first = reschedule_resource(resource, tasks, slot_duration_s=0.01)
        second = reschedule_resource(resource, tasks, slot_duration_s=0.01)
        self.assertEqual(first, second)
        self.assertEqual(resource.cpu_allocations_hz, {1: 80.0})

    def test_invalid_schedule_does_not_partially_mutate_resource_or_tasks(self) -> None:
        resource = mec_resource()
        task = queued_task(1, v2v=True)
        resource.enqueue_task(1)
        before = (list(resource.service_queue), set(resource.active_task_ids), dict(resource.cpu_allocations_hz), task.status)
        with self.assertRaises(ValueError):
            reschedule_resource(resource, {1: task}, slot_duration_s=0.01)
        self.assertEqual((resource.service_queue, resource.active_task_ids, resource.cpu_allocations_hz, task.status), before)

    def test_duplicate_tasks_negative_work_and_unavailable_server_are_rejected(self) -> None:
        task = queued_task(1)
        with self.assertRaises(ValueError):
            ordered_tasks([task, task], 0.01)
        with self.assertRaises(ValueError):
            next_queue_workload_cycles(1.0, -1.0, 0.0)
        resource = server_resource()
        resource.availability = False
        with self.assertRaises(ValueError):
            reschedule_resource(resource, {}, slot_duration_s=0.01)
