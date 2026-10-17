"""Latency and mobile-energy fixtures, including zero service and fallback."""

import math
import unittest

from risk_aware_v2x.models.energy import (
    MobileEnergyBreakdown, cpu_energy_j, query_energy_j, radio_energy_j,
)
from risk_aware_v2x.models.latency import (
    LatencyBreakdown, computation_time_s, elapsed_latency_s, progress_work,
    query_latency_s, query_response_size_bits, transmission_time_s,
)


class LatencyModelTests(unittest.TestCase):
    def test_known_upload_computation_and_download_durations(self) -> None:
        self.assertEqual(transmission_time_s(1_000.0, 10_000.0), 0.1)
        self.assertEqual(computation_time_s(1e6, 2e9), 0.0005)
        self.assertEqual(transmission_time_s(100.0, 10_000.0), 0.01)

    def test_zero_work_and_zero_service_have_explicit_semantics(self) -> None:
        self.assertEqual(transmission_time_s(0.0, 0.0), 0.0)
        self.assertEqual(computation_time_s(0.0, 0.0), 0.0)
        self.assertEqual(transmission_time_s(1.0, 0.0), math.inf)
        self.assertEqual(computation_time_s(1.0, 0.0), math.inf)

    def test_query_response_scales_with_record_count_and_latency_adds_phases(self) -> None:
        response_bits = query_response_size_bits(100, 200, 3)
        self.assertEqual(response_bits, 700)
        self.assertEqual(query_response_size_bits(100, 200, 0), 100)
        total_s = query_latency_s(
            request_size_bits=100, response_size_bits=response_bits,
            uplink_rate_bps=1_000.0, downlink_rate_bps=7_000.0, processing_time_s=0.02,
        )
        self.assertAlmostEqual(total_s, 0.22)

    def test_progress_is_conserved_and_early_finish_uses_only_actual_active_time(self) -> None:
        partial = progress_work(100.0, 1_000.0, 0.01)
        self.assertEqual((partial.served_work, partial.remaining_work, partial.active_time_s), (10.0, 90.0, 0.01))
        finished = progress_work(3.0, 1_000.0, 0.01)
        self.assertEqual((finished.served_work, finished.remaining_work, finished.active_time_s), (3.0, 0.0, 0.003))
        self.assertEqual(finished.served_work + finished.remaining_work, 3.0)
        self.assertAlmostEqual(radio_energy_j(2.0, finished.active_time_s), 0.006)

    def test_progress_with_no_service_or_no_work_is_idle(self) -> None:
        for remaining, rate, duration, expected_remaining in (
            (100.0, 0.0, 0.01, 100.0), (0.0, 100.0, 0.01, 0.0), (100.0, 100.0, 0.0, 100.0),
        ):
            with self.subTest(remaining=remaining, rate=rate, duration=duration):
                result = progress_work(remaining, rate, duration)
                self.assertEqual(result.served_work, 0.0)
                self.assertEqual(result.active_time_s, 0.0)
                self.assertEqual(result.remaining_work, expected_remaining)

    def test_latency_fallback_is_counted_once_and_matches_elapsed_time(self) -> None:
        breakdown = LatencyBreakdown(
            query_wait_s=0.02, admission_wait_s=0.01, queue_wait_s=0.03,
            upload_s=0.04, compute_s=0.05, download_s=0.01, interrupted_attempt_s=0.07,
        )
        self.assertAlmostEqual(breakdown.total_s, 0.23)
        self.assertAlmostEqual(breakdown.total_s, elapsed_latency_s(1.0, 1.23))

    def test_invalid_durations_and_non_finite_physical_state_are_rejected(self) -> None:
        for operation in (
            lambda: transmission_time_s(-1.0, 1.0),
            lambda: transmission_time_s(0.0, -1.0),
            lambda: computation_time_s(1.0, math.inf),
            lambda: progress_work(1.0, 1.0, -0.1),
            lambda: LatencyBreakdown(compute_s=math.inf),
            lambda: elapsed_latency_s(2.0, 1.0),
            lambda: query_response_size_bits(1, 1, -1),
        ):
            with self.assertRaises((TypeError, ValueError)):
                operation()


class EnergyModelTests(unittest.TestCase):
    def test_radio_and_cpu_formula_fixtures(self) -> None:
        self.assertAlmostEqual(radio_energy_j(2.0, 0.1), 0.2)
        self.assertAlmostEqual(cpu_energy_j(1e6, 2e9, 1e-27), 0.004)
        self.assertEqual(cpu_energy_j(0.0, 0.0, 1e-27), 0.0)
        low = cpu_energy_j(1e6, 1e9, 1e-27)
        self.assertAlmostEqual(cpu_energy_j(1e6, 2e9, 1e-27), 4.0 * low)

    def test_query_energy_sums_build_tx_idle_and_rx_once(self) -> None:
        cost_j = query_energy_j(
            build_energy_j=0.01, tx_power_w=2.0, request_time_s=0.1,
            idle_power_w=0.5, processing_time_s=0.02,
            rx_power_w=1.0, response_time_s=0.2,
        )
        self.assertAlmostEqual(cost_j, 0.42)

    def test_mobile_breakdown_supports_mec_v2v_and_preserves_interrupted_energy(self) -> None:
        mec = MobileEnergyBreakdown(query_j=0.1, user_tx_j=0.2, user_rx_j=0.3)
        v2v = MobileEnergyBreakdown(
            query_j=0.1, user_tx_j=0.2, user_rx_j=0.3,
            server_rx_j=0.1, server_cpu_j=0.2, server_tx_j=0.1,
        )
        fallback = MobileEnergyBreakdown(
            query_j=0.1, user_tx_j=0.2, user_rx_j=0.3, interrupted_attempt_j=0.4,
        )
        self.assertAlmostEqual(mec.total_j, 0.6)
        self.assertAlmostEqual(v2v.total_j, 1.0)
        self.assertAlmostEqual(fallback.total_j, 1.0)
        self.assertEqual(mec.server_cpu_j, 0.0)

    def test_invalid_energy_inputs_are_rejected(self) -> None:
        for operation in (
            lambda: radio_energy_j(-1.0, 1.0),
            lambda: radio_energy_j(1.0, math.inf),
            lambda: cpu_energy_j(1.0, 0.0, 1e-27),
            lambda: cpu_energy_j(1.0, 1.0, -1e-27),
            lambda: MobileEnergyBreakdown(server_cpu_j=-1.0),
        ):
            with self.assertRaises(ValueError):
                operation()
