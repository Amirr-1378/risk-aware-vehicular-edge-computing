"""Unit tests for user queues, query state, cache age, and energy reserve."""

from __future__ import annotations

import unittest
import math

from risk_aware_v2x.entities import (
    MECMetadataRecord,
    MECSnapshot,
    QueryStatus,
    UserVehicle,
)


def make_user(**overrides: object) -> UserVehicle:
    values: dict[str, object] = {
        "vehicle_id": 3,
        "max_tx_power_w": 2.0,
        "rx_power_w": 0.5,
        "idle_power_w": 0.1,
        "battery_capacity_j": 100.0,
        "radio_antenna_gain_linear": 1.0,
        "max_task_queue_size": 2,
        "mobility_profile_id": "urban-eastbound",
    }
    values.update(overrides)
    return UserVehicle(**values)  # type: ignore[arg-type]


def make_snapshot(generated_time_s: float = 0.04) -> MECSnapshot:
    return MECSnapshot(
        generated_time_s=generated_time_s,
        mec=MECMetadataRecord(
            server_id=1,
            cpu_capacity_hz=10e9,
            available_cpu_hz=8e9,
            queue_workload_cycles=1e6,
            estimated_queue_delay_s=0.001,
            available_uplink_bandwidth_hz=10e6,
            available_downlink_bandwidth_hz=20e6,
        ),
    )


class UserVehicleEntityTests(unittest.TestCase):
    def test_decimal_slot_grid_cache_boundary_has_no_epsilon_extension(self) -> None:
        user = make_user(task_queue=[1])
        user.begin_query(owner_task_id=1, request_time_s=0.0)
        user.advance_query_phase(QueryStatus.QUERY_MEC_PROCESSING, 0.0)
        user.advance_query_phase(QueryStatus.QUERY_RESPONSE_DOWNLINK, 0.0)
        user.install_snapshot(make_snapshot(0.06), received_time_s=0.07)
        self.assertTrue(user.is_cache_valid(0.07, 0.01))
        self.assertFalse(user.is_cache_valid(math.nextafter(0.07, math.inf), 0.01))

    def test_queue_is_unique_and_bounded(self) -> None:
        user = make_user()
        user.enqueue_task(1)
        with self.assertRaises(ValueError):
            user.enqueue_task(1)
        user.enqueue_task(2)
        with self.assertRaises(OverflowError):
            user.enqueue_task(3)
        user.remove_task(1)
        self.assertEqual(user.task_queue, [2])

    def test_query_snapshot_age_uses_generation_not_reception_time(self) -> None:
        user = make_user(task_queue=[1])
        user.begin_query(owner_task_id=1, request_time_s=0.0)
        user.advance_query_phase(QueryStatus.QUERY_MEC_PROCESSING, 0.01)
        user.advance_query_phase(QueryStatus.QUERY_RESPONSE_DOWNLINK, 0.02)
        user.install_snapshot(make_snapshot(0.04), received_time_s=0.06)

        self.assertEqual(user.snapshot_generated_time_s, 0.04)
        self.assertEqual(user.snapshot_received_time_s, 0.06)
        self.assertAlmostEqual(user.information_age_s(0.06), 0.02)
        self.assertIsNone(user.query_owner_task_id)

        self.assertTrue(user.is_cache_valid(0.14, lifetime_s=0.1))
        user.refresh_cache_status(0.140_001, lifetime_s=0.1)
        self.assertEqual(user.query_status, QueryStatus.CACHE_EXPIRED)

    def test_only_one_query_can_be_in_flight(self) -> None:
        user = make_user(task_queue=[1, 2])
        user.begin_query(owner_task_id=1, request_time_s=0.0)
        with self.assertRaises(ValueError):
            user.begin_query(owner_task_id=2, request_time_s=0.01)
        with self.assertRaises(ValueError):
            user.install_snapshot(make_snapshot(), received_time_s=0.06)

    def test_user_energy_spend_preserves_fallback_reserve(self) -> None:
        user = make_user(battery_energy_j=10.0)
        self.assertTrue(user.can_spend_energy(7.0, protected_reserve_j=3.0))
        user.consume_energy(7.0, protected_reserve_j=3.0)
        self.assertEqual(user.battery_energy_j, 3.0)
        with self.assertRaises(ValueError):
            user.consume_energy(0.01, protected_reserve_j=3.0)

    def test_radio_and_battery_values_cannot_exceed_capacities(self) -> None:
        with self.assertRaises(ValueError):
            make_user(tx_power_w=2.1)
        with self.assertRaises(ValueError):
            make_user(battery_energy_j=101.0)
        user = make_user()
        with self.assertRaises(AttributeError):
            user.max_tx_power_w = 3.0


if __name__ == "__main__":
    unittest.main()
