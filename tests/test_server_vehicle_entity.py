"""Unit tests for mobile server compute, queue, and energy invariants."""

from __future__ import annotations

import unittest

from risk_aware_v2x.entities import ServerVehicle


def make_server_vehicle(**overrides: object) -> ServerVehicle:
    values: dict[str, object] = {
        "vehicle_id": 9,
        "cpu_capacity_hz": 100.0,
        "min_self_reserved_cpu_hz": 20.0,
        "max_concurrent_tasks": 2,
        "max_queue_size": 2,
        "max_tx_power_w": 2.0,
        "rx_power_w": 0.5,
        "cpu_energy_coefficient": 1e-27,
        "battery_capacity_j": 100.0,
        "min_reserve_energy_j": 10.0,
        "radio_antenna_gain_linear": 1.0,
        "mobility_profile_id": 1,
        "self_workload_cpu_hz": 10.0,
    }
    values.update(overrides)
    return ServerVehicle(**values)  # type: ignore[arg-type]


class ServerVehicleEntityTests(unittest.TestCase):
    def test_available_cpu_matches_documented_formula(self) -> None:
        server = make_server_vehicle()
        self.assertEqual(server.available_service_cpu_hz, 70.0)

        server.enqueue_task(1)
        server.activate_task(1)
        server.set_cpu_allocation(1, 40.0)
        self.assertEqual(server.available_service_cpu_hz, 30.0)
        with self.assertRaises(ValueError):
            server.set_cpu_allocation(1, 80.0)

    def test_queue_capacity_concurrency_and_preemption_are_explicit(self) -> None:
        server = make_server_vehicle(max_concurrent_tasks=1)
        server.enqueue_task(1)
        server.enqueue_task(2)
        with self.assertRaises(OverflowError):
            server.enqueue_task(3)

        server.activate_task(1)
        with self.assertRaises(OverflowError):
            server.activate_task(2)
        server.set_cpu_allocation(1, 20.0)
        server.preempt_task(1)
        self.assertNotIn(1, server.active_task_ids)
        self.assertNotIn(1, server.cpu_allocations_hz)
        self.assertEqual(server.service_queue, [2, 1])

    def test_energy_reserve_controls_availability(self) -> None:
        server = make_server_vehicle(battery_energy_j=12.0)
        self.assertTrue(server.can_spend_energy(2.0))
        server.consume_energy(2.0)
        self.assertEqual(server.battery_energy_j, 10.0)
        self.assertFalse(server.can_accept_task)
        with self.assertRaises(ValueError):
            server.consume_energy(0.01)

    def test_invalid_cpu_and_energy_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_server_vehicle(min_self_reserved_cpu_hz=101.0)
        with self.assertRaises(ValueError):
            make_server_vehicle(min_reserve_energy_j=100.0)
        server = make_server_vehicle()
        with self.assertRaises(AttributeError):
            server.cpu_capacity_hz = 200.0


if __name__ == "__main__":
    unittest.main()
