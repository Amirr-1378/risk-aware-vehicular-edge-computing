"""Hand-computed radio fixtures, consistent estimates, and stochastic isolation."""

import math
import unittest
from random import Random

from risk_aware_v2x.models.channel import (
    SPEED_OF_LIGHT_MPS, LogDistanceChannel, noise_power_w,
    received_sinr_linear, shannon_rate_bps,
)
from risk_aware_v2x.models.randomness import RandomStreams
from risk_aware_v2x.models.units import (
    bytes_to_bits, db_to_linear, dbm_to_watts, linear_to_db, watts_to_dbm,
)


class UnitConversionTests(unittest.TestCase):
    def test_known_power_and_size_conversions(self) -> None:
        self.assertEqual(db_to_linear(10.0), 10.0)
        self.assertEqual(linear_to_db(100.0), 20.0)
        self.assertEqual(dbm_to_watts(30.0), 1.0)
        self.assertEqual(dbm_to_watts(0.0), 0.001)
        self.assertEqual(watts_to_dbm(1.0), 30.0)
        self.assertEqual(bytes_to_bits(125), 1_000)

    def test_power_round_trips(self) -> None:
        for power_db in (-120.0, -3.0, 0.0, 6.0, 40.0):
            with self.subTest(power_db=power_db):
                self.assertAlmostEqual(linear_to_db(db_to_linear(power_db)), power_db)
                self.assertAlmostEqual(watts_to_dbm(dbm_to_watts(power_db)), power_db)

    def test_invalid_units_and_non_finite_inputs_are_rejected(self) -> None:
        for operation, value in (
            (linear_to_db, 0.0), (watts_to_dbm, -1.0),
            (db_to_linear, math.inf), (db_to_linear, math.nan),
            (db_to_linear, 1e10), (db_to_linear, -1e10),
            (bytes_to_bits, -1), (bytes_to_bits, 1.5), (bytes_to_bits, True),
        ):
            with self.subTest(operation=operation.__name__, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    operation(value)


class ChannelModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channel = LogDistanceChannel(SPEED_OF_LIGHT_MPS, path_loss_exponent=2.0)

    def test_free_space_fixture_and_distance_monotonicity(self) -> None:
        self.assertAlmostEqual(
            self.channel.large_scale_gain_linear(10.0), (1.0 / (40.0 * math.pi)) ** 2,
        )
        self.assertAlmostEqual(
            self.channel.large_scale_gain_linear(20.0),
            self.channel.large_scale_gain_linear(10.0) / 4.0,
        )
        self.assertEqual(
            self.channel.large_scale_gain_linear(0.0), self.channel.reference_gain_linear,
        )

    def test_reference_distance_and_exponent_are_not_confused(self) -> None:
        channel = LogDistanceChannel(SPEED_OF_LIGHT_MPS, 3.0, reference_distance_m=2.0)
        self.assertAlmostEqual(channel.reference_gain_linear, (1 / (8 * math.pi)) ** 2)
        self.assertAlmostEqual(channel.large_scale_gain_linear(4.0), channel.reference_gain_linear / 8)

    def test_noise_sinr_and_shannon_fixtures(self) -> None:
        self.assertAlmostEqual(noise_power_w(2e-20, 1e6, 2.0), 4e-14)
        sinr = received_sinr_linear(
            tx_power_w=2.0, channel_gain_linear=0.01, noise_w=0.001,
            interference_w=0.001, tx_antenna_gain_linear=2.0,
        )
        self.assertAlmostEqual(sinr, 20.0)
        self.assertEqual(shannon_rate_bps(10.0, 3.0), 20.0)
        self.assertEqual(shannon_rate_bps(0.0, 100.0), 0.0)
        self.assertEqual(shannon_rate_bps(10.0, 0.0), 0.0)
        self.assertGreater(shannon_rate_bps(1.0, 1e-20), 0.0)

    def test_interference_lowers_rate_and_zero_tx_produces_no_service(self) -> None:
        inputs = dict(distance_m=10.0, tx_power_w=1.0, bandwidth_hz=1e6, noise_w=1e-9)
        clean = self.channel.realize(**inputs)
        noisy = self.channel.realize(**inputs, interference_w=1e-8)
        self.assertLess(noisy.rate_bps, clean.rate_bps)
        silent = self.channel.realize(**{**inputs, "tx_power_w": 0.0})
        self.assertEqual(silent.rate_bps, 0.0)

    def test_filtered_distance_inversion_removes_fading_but_not_shadowing(self) -> None:
        link = self.channel.realize(
            distance_m=50.0, tx_power_w=2.0, bandwidth_hz=1e6,
            noise_w=1e-9, interference_w=1e-9, fading_power_gain_linear=0.01,
        )
        estimate_inputs = dict(tx_power_w=2.0, noise_w=1e-9, interference_w=1e-9)
        self.assertAlmostEqual(self.channel.estimate_distance_m(
            filtered_sinr_linear=link.large_scale_sinr_linear, **estimate_inputs,
        ), 50.0)
        self.assertAlmostEqual(link.sinr_linear, link.large_scale_sinr_linear / 100.0)
        self.assertAlmostEqual(link.rate_bps, shannon_rate_bps(1e6, link.sinr_linear))
        shadowed = self.channel.realize(
            distance_m=50.0, tx_power_w=2.0, bandwidth_hz=1e6, noise_w=1e-9,
            interference_w=1e-9, shadowing_db=10.0,
        )
        estimate = self.channel.estimate_distance_m(
            filtered_sinr_linear=shadowed.large_scale_sinr_linear, **estimate_inputs,
        )
        self.assertAlmostEqual(estimate, 50.0 * math.sqrt(10.0))

    def test_seeded_channel_repeats_and_fading_draws_do_not_change_shadowing(self) -> None:
        channel = LogDistanceChannel(5.9e9, 2.5, shadowing_std_db=4.0, rayleigh_fading=True)
        first, second = RandomStreams(7), RandomStreams(7)
        args = dict(distance_m=50.0, tx_power_w=1.0, bandwidth_hz=1e6, noise_w=1e-9)
        link1 = channel.sample(**args, shadowing_rng=first.stream("shadowing"), fading_rng=first.stream("fading"))
        link2 = channel.sample(**args, shadowing_rng=second.stream("shadowing"), fading_rng=second.stream("fading"))
        self.assertEqual(link1, link2)
        for _ in range(100):
            first.stream("fading").random()
        next1 = channel.sample(**args, shadowing_rng=first.stream("shadowing"), fading_rng=first.stream("fading"))
        next2 = channel.sample(**args, shadowing_rng=second.stream("shadowing"), fading_rng=second.stream("fading"))
        self.assertEqual(next1.large_scale_gain_linear, next2.large_scale_gain_linear)
        self.assertNotEqual(next1.fading_power_gain_linear, next2.fading_power_gain_linear)

    def test_rayleigh_power_has_unit_mean_and_can_represent_deep_fades(self) -> None:
        channel = LogDistanceChannel(5.9e9, 2.0, rayleigh_fading=True)
        rng = Random(1234)
        gains = [channel.sample(
            distance_m=20.0, tx_power_w=1.0, bandwidth_hz=1e6, noise_w=1e-9,
            fading_rng=rng,
        ).fading_power_gain_linear for _ in range(10_000)]
        self.assertAlmostEqual(sum(gains) / len(gains), 1.0, delta=0.03)
        self.assertLess(min(gains), 0.001)

    def test_enabled_stochastic_models_require_separate_explicit_rngs(self) -> None:
        channel = LogDistanceChannel(5.9e9, 2.0, shadowing_std_db=2.0, rayleigh_fading=True)
        args = dict(distance_m=10.0, tx_power_w=1.0, bandwidth_hz=1.0, noise_w=1.0)
        with self.assertRaises(TypeError):
            channel.sample(**args)
        shared = Random(1)
        with self.assertRaises(ValueError):
            channel.sample(**args, shadowing_rng=shared, fading_rng=shared)

    def test_invalid_channel_configuration_and_estimates(self) -> None:
        for kwargs in ({"path_loss_exponent": 0.0}, {"shadowing_std_db": -1.0}, {"rayleigh_fading": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    LogDistanceChannel(**{"carrier_frequency_hz": 5.9e9, "path_loss_exponent": 2.0, **kwargs})
        with self.assertRaises(ValueError):
            self.channel.estimate_distance_m(filtered_sinr_linear=0.0, tx_power_w=1.0, noise_w=1.0)
        with self.assertRaises(ValueError):
            self.channel.realize(distance_m=-1.0, tx_power_w=1.0, bandwidth_hz=1.0, noise_w=1.0)
        with self.assertRaises(ValueError):
            received_sinr_linear(tx_power_w=1.0, channel_gain_linear=1.0, noise_w=0.0)
        with self.assertRaises(ValueError):
            noise_power_w(1.0, 1.0, 0.5)
