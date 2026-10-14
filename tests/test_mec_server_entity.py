"""Unit tests for MEC coverage, resource accounting, and snapshots."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from risk_aware_v2x.entities import MECServer, ServerVehicleMetadata


def make_mec(**overrides: object) -> MECServer:
    values: dict[str, object] = {
        "server_id": 1,
        "position_m": (0.0, 0.0),
        "coverage_radius_m": 100.0,
        "cpu_capacity_hz": 100.0,
        "max_concurrent_tasks": 2,
        "total_uplink_bandwidth_hz": 20.0,
        "total_downlink_bandwidth_hz": 40.0,
        "max_downlink_power_w": 10.0,
        "query_processing_time_s": 0.002,
        "background_cpu_load_hz": 20.0,
    }
    values.update(overrides)
    return MECServer(**values)  # type: ignore[arg-type]


def make_server_record() -> ServerVehicleMetadata:
    return ServerVehicleMetadata(
        server_vehicle_id=9,
        record_time_s=0.03,
        available_cpu_hz=50.0,
        queue_workload_cycles=1_000.0,
        battery_fraction=0.8,
        estimated_snr_linear=10.0,
        estimated_rate_bps=5e6,
        estimated_distance_m=25.0,
        estimated_contact_time_s=2.0,
        available=True,
    )


class MECServerEntityTests(unittest.TestCase):
    def test_coverage_includes_exact_boundary(self) -> None:
        mec = make_mec()
        self.assertTrue(mec.covers((60.0, 80.0)))
        self.assertFalse(mec.covers((60.0, 80.001)))

    def test_cpu_and_bandwidth_allocations_cannot_overcommit(self) -> None:
        mec = make_mec()
        mec.enqueue_task(1)
        mec.activate_task(1)
        mec.set_cpu_allocation(1, 50.0)
        self.assertEqual(mec.available_cpu_hz, 30.0)
        with self.assertRaises(ValueError):
            mec.set_cpu_allocation(1, 81.0)

        mec.set_uplink_allocation(3, 12.0)
        mec.set_uplink_allocation(4, 8.0)
        self.assertEqual(mec.available_uplink_bandwidth_hz, 0.0)
        with self.assertRaises(ValueError):
            mec.set_uplink_allocation(5, 0.1)

    def test_registry_and_snapshot_are_timestamped_and_immutable(self) -> None:
        mec = make_mec()
        record = make_server_record()
        mec.update_server_information(record)
        snapshot = mec.build_snapshot(
            generated_time_s=0.04,
            queue_workload_cycles=2_000.0,
            estimated_queue_delay_s=0.01,
        )

        self.assertEqual(snapshot.generated_time_s, 0.04)
        self.assertEqual(snapshot.server_record(9), record)
        self.assertEqual(snapshot.mec.available_cpu_hz, 80.0)
        mec.background_cpu_load_hz = 30.0
        self.assertEqual(snapshot.mec.available_cpu_hz, 80.0)
        with self.assertRaises(FrozenInstanceError):
            snapshot.generated_time_s = 1.0  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            mec.coverage_radius_m = 200.0

    def test_mec_queue_has_no_rejection_capacity(self) -> None:
        mec = make_mec()
        for task_id in range(100):
            mec.enqueue_task(task_id)
        self.assertEqual(len(mec.service_queue), 100)


if __name__ == "__main__":
    unittest.main()
