"""Unit tests for task invariants and lifecycle transitions."""

from __future__ import annotations

import math
import unittest

from risk_aware_v2x.entities import (
    Criticality,
    ExecutorKind,
    Task,
    TaskStatus,
)


def make_task(**overrides: object) -> Task:
    values: dict[str, object] = {
        "task_id": 7,
        "owner_vehicle_id": 3,
        "arrival_slot": 2,
        "input_size_bits": 1_000,
        "output_size_bits": 100,
        "cpu_cycles_per_bit": 500.0,
        "deadline_s": 0.05,
        "criticality": Criticality.HIGH,
        "energy_budget_j": 2.0,
    }
    values.update(overrides)
    return Task(**values)  # type: ignore[arg-type]


class TaskEntityTests(unittest.TestCase):
    def test_decimal_slot_deadline_does_not_accept_the_next_float_after_boundary(self) -> None:
        task = make_task(arrival_slot=35, deadline_s=.01)
        self.assertEqual(task.absolute_deadline_s(.01), .36)
        self.assertFalse(task.mark_deadline_if_crossed(.36, .01))
        self.assertTrue(task.mark_deadline_if_crossed(math.nextafter(.36, math.inf), .01))

    def test_workload_is_derived_and_progress_never_becomes_negative(self) -> None:
        task = make_task()

        self.assertEqual(task.required_cpu_cycles, 500_000.0)
        self.assertEqual(task.serve_cpu(499_000.0), 499_000.0)
        self.assertEqual(task.serve_cpu(5_000.0), 1_000.0)
        self.assertEqual(task.remaining_cpu_cycles, 0.0)

    def test_static_values_reject_invalid_units_and_numbers(self) -> None:
        with self.assertRaises(ValueError):
            make_task(input_size_bits=0)
        with self.assertRaises(ValueError):
            make_task(deadline_s=math.inf)
        with self.assertRaises(TypeError):
            make_task(task_id=True)
        with self.assertRaises(TypeError):
            make_task(criticality="high")

        task = make_task()
        with self.assertRaises(AttributeError):
            task.deadline_s = 1.0

    def test_deadline_boundary_is_on_time_and_crossing_is_one_shot(self) -> None:
        task = make_task()
        absolute_deadline_s = 0.07

        self.assertFalse(task.mark_deadline_if_crossed(absolute_deadline_s, 0.01))
        self.assertTrue(task.mark_deadline_if_crossed(0.070_001, 0.01))
        self.assertFalse(task.mark_deadline_if_crossed(0.08, 0.01))

        on_time = make_task(task_id=8)
        on_time.transition_to(TaskStatus.READY_FOR_OFFLOADING)
        on_time.assign_target(ExecutorKind.MEC, target_id=1)
        on_time.transition_to(TaskStatus.MEC_QUEUED)
        on_time.transition_to(TaskStatus.MEC_EXECUTING)
        on_time.transition_to(TaskStatus.DOWNLOADING_FROM_MEC)
        on_time.complete(absolute_deadline_s, 0.01, executor_id=1)
        self.assertEqual(on_time.status, TaskStatus.COMPLETED_ON_TIME)
        self.assertFalse(on_time.deadline_missed)

    def test_v2v_interruption_preserves_cost_and_restarts_full_work_at_mec(self) -> None:
        task = make_task()
        task.transition_to(TaskStatus.READY_FOR_OFFLOADING)
        task.assign_target(ExecutorKind.SERVER_VEHICLE, target_id=9)
        task.serve_upload(250.0)
        task.transition_to(TaskStatus.V2V_QUEUED)
        task.transition_to(TaskStatus.V2V_EXECUTING)
        task.serve_cpu(10_000.0)

        task.mark_v2v_interrupted(delay_s=0.02, energy_j=0.3)
        task.prepare_mec_fallback()

        self.assertEqual(task.status, TaskStatus.UPLOADING_TO_MEC)
        self.assertTrue(task.fallback_used)
        self.assertEqual(task.initial_target_kind, ExecutorKind.SERVER_VEHICLE)
        self.assertEqual(task.initial_target_id, 9)
        self.assertEqual(task.remaining_upload_bits, 1_000.0)
        self.assertEqual(task.remaining_cpu_cycles, 500_000.0)
        self.assertEqual(task.remaining_download_bits, 100.0)
        self.assertEqual(task.v2v_attempt_delay_s, 0.02)
        self.assertEqual(task.v2v_attempt_energy_j, 0.3)

    def test_illegal_transition_and_second_assignment_are_rejected(self) -> None:
        task = make_task()
        with self.assertRaises(ValueError):
            task.transition_to(TaskStatus.MEC_EXECUTING)
        task.transition_to(TaskStatus.READY_FOR_OFFLOADING)
        task.assign_target(ExecutorKind.MEC, 1)
        with self.assertRaises(ValueError):
            task.assign_target(ExecutorKind.SERVER_VEHICLE, 2)

    def test_scheduling_key_is_criticality_aware_edf(self) -> None:
        normal = make_task(task_id=1, criticality=Criticality.NORMAL, deadline_s=0.01)
        critical = make_task(
            task_id=2, criticality=Criticality.CRITICAL, deadline_s=1.0
        )
        earlier_high = make_task(task_id=3, deadline_s=0.1)
        later_high = make_task(task_id=4, deadline_s=0.2)

        ordered = sorted(
            [normal, later_high, critical, earlier_high],
            key=lambda item: item.scheduling_key(0.01),
        )
        self.assertEqual([task.task_id for task in ordered], [2, 3, 4, 1])


if __name__ == "__main__":
    unittest.main()
