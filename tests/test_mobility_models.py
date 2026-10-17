"""Deterministic geometry, braking, contact, and vehicle-state transition tests."""

import math
import unittest

from risk_aware_v2x.entities import UserVehicle
from risk_aware_v2x.models.mobility import (
    MotionState, advance_straight_line, advance_vehicle, distance_m, estimate_contact_time_s,
)


class MobilityModelTests(unittest.TestCase):
    def test_distance_has_known_geometry_and_zero_separation(self) -> None:
        self.assertEqual(distance_m((1.0, 2.0), (4.0, 6.0)), 5.0)
        self.assertEqual(distance_m((1.0, 2.0), (1.0, 2.0)), 0.0)

    def test_kinematic_update_matches_constant_acceleration(self) -> None:
        state = MotionState((0.0, 0.0), 10.0, 0.0, 2.0)
        moved = advance_straight_line(state, 2.0)
        self.assertEqual(moved.position_m, (24.0, 0.0))
        self.assertEqual(moved.speed_mps, 14.0)
        self.assertEqual(state.position_m, (0.0, 0.0))
        north = advance_straight_line(MotionState((0.0, 0.0), 5.0, math.pi / 2), 2.0)
        self.assertAlmostEqual(north.position_m[0], 0.0)
        self.assertEqual(north.position_m[1], 10.0)

    def test_braking_stops_without_reverse_motion_and_step_partition_agrees(self) -> None:
        state = MotionState((0.0, 0.0), 10.0, 0.0, -5.0)
        stopped = advance_straight_line(state, 4.0)
        partitioned = advance_straight_line(advance_straight_line(state, 1.0), 3.0)
        self.assertEqual(stopped.position_m, (10.0, 0.0))
        self.assertEqual(stopped.speed_mps, 0.0)
        self.assertEqual(stopped, partitioned)
        self.assertEqual(advance_straight_line(stopped, 3.0), stopped)

    def test_contact_prediction_matches_separating_and_approaching_pairs(self) -> None:
        user = MotionState((0.0, 0.0), 0.0, 0.0)
        away = MotionState((30.0, 0.0), 10.0, 0.0)
        toward = MotionState((30.0, 0.0), 10.0, math.pi)
        self.assertAlmostEqual(estimate_contact_time_s(user, away, contact_range_m=100.0, horizon_s=100.0), 7.0)
        self.assertAlmostEqual(estimate_contact_time_s(user, toward, contact_range_m=100.0, horizon_s=100.0), 13.0)

    def test_contact_handles_boundary_horizon_stationary_and_outside(self) -> None:
        user = MotionState((0.0, 0.0), 0.0, 0.0)
        cases = [
            (MotionState((100.0, 0.0), 1.0, 0.0), 0.0),
            (MotionState((100.0, 0.0), 1.0, math.pi), 20.0),
            (MotionState((100.0, 0.0), 0.0, 0.0), 20.0),
            (MotionState((101.0, 0.0), 1.0, math.pi), 0.0),
            (MotionState((0.0, 0.0), 0.0, 0.0), 20.0),
        ]
        for server, expected in cases:
            with self.subTest(server=server):
                self.assertEqual(estimate_contact_time_s(user, server, contact_range_m=100.0, horizon_s=20.0), expected)

    def test_mobility_installs_validated_vehicle_state(self) -> None:
        vehicle = UserVehicle(
            vehicle_id=3, max_tx_power_w=1.0, rx_power_w=0.5, idle_power_w=0.1,
            battery_capacity_j=100.0, radio_antenna_gain_linear=1.0,
            max_task_queue_size=10, mobility_profile_id="straight", speed_mps=10.0,
        )
        state = advance_vehicle(vehicle, 0.01)
        self.assertEqual(vehicle.position_m, (0.1, 0.0))
        self.assertEqual(vehicle.position_m, state.position_m)
        before = vehicle.position_m
        with self.assertRaises(ValueError):
            advance_vehicle(vehicle, -0.1)
        self.assertEqual(vehicle.position_m, before)

    def test_invalid_motion_is_rejected(self) -> None:
        for args in (((0.0,), 0.0, 0.0), ((0.0, 0.0), -1.0, 0.0), ((0.0, math.inf), 0.0, 0.0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    MotionState(*args)
