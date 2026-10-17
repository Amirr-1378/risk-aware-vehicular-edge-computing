"""Scalar log-distance channel primitives; parameters belong to experiments.

This is a configurable analytical channel, not a full 3GPP channel model.
Shadowing samples and Rayleigh power gains have no temporal correlation here;
the environment decides when to refresh or retain them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from ..entities._validation import (
    require_finite_float,
    require_non_negative_float,
    require_positive_float,
)
from .units import db_to_linear

SPEED_OF_LIGHT_MPS = 299_792_458.0


def shannon_rate_bps(bandwidth_hz: float, sinr_linear: float) -> float:
    """Ideal bit/s = B log2(1 + SINR); zero bandwidth or SINR gives zero rate."""
    bandwidth = require_non_negative_float("bandwidth_hz", bandwidth_hz)
    sinr = require_non_negative_float("sinr_linear", sinr_linear)
    return require_non_negative_float(
        "rate_bps", bandwidth * (math.log1p(sinr) / math.log(2.0))
    )


def noise_power_w(
    density_w_per_hz: float, bandwidth_hz: float, noise_figure_linear: float = 1.0
) -> float:
    """Receiver thermal-noise power N0 B F in W, with linear noise figure F >= 1."""
    density = require_positive_float("density_w_per_hz", density_w_per_hz)
    bandwidth = require_non_negative_float("bandwidth_hz", bandwidth_hz)
    figure = require_positive_float("noise_figure_linear", noise_figure_linear)
    if figure < 1.0:
        raise ValueError("noise_figure_linear must be at least one")
    return require_non_negative_float("noise_power_w", density * bandwidth * figure)


def received_sinr_linear(
    *, tx_power_w: float, channel_gain_linear: float,
    noise_w: float, interference_w: float = 0.0,
    tx_antenna_gain_linear: float = 1.0, rx_antenna_gain_linear: float = 1.0,
) -> float:
    """Compute P_t G_t G_r g / (N + I), using watts and linear gains."""
    power = require_non_negative_float("tx_power_w", tx_power_w)
    gain = require_non_negative_float("channel_gain_linear", channel_gain_linear)
    noise = require_positive_float("noise_w", noise_w)
    interference = require_non_negative_float("interference_w", interference_w)
    tx_gain = require_positive_float("tx_antenna_gain_linear", tx_antenna_gain_linear)
    rx_gain = require_positive_float("rx_antenna_gain_linear", rx_antenna_gain_linear)
    denominator = require_positive_float("noise_plus_interference_w", noise + interference)
    return require_non_negative_float("sinr_linear", power * tx_gain * rx_gain * gain / denominator)


@dataclass(frozen=True, slots=True)
class ChannelRealization:
    """One physical realization, with separately accessible large/small-scale parts.

    ``rate_bps`` uses instantaneous SINR; ``large_scale_rate_bps`` removes only
    fast fading and serves as an ideal filtered-rate estimate. Shadowing is
    present in both. Neither is an independently sampled feature.
    """

    large_scale_gain_linear: float
    fading_power_gain_linear: float
    large_scale_sinr_linear: float
    sinr_linear: float
    bandwidth_hz: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))

    @property
    def rate_bps(self) -> float:
        return shannon_rate_bps(self.bandwidth_hz, self.sinr_linear)

    @property
    def large_scale_rate_bps(self) -> float:
        return shannon_rate_bps(self.bandwidth_hz, self.large_scale_sinr_linear)


@dataclass(frozen=True, slots=True)
class LogDistanceChannel:
    """Reference-distance path loss with optional shadowing and Rayleigh fading.

    Gain is g0 * (max(d, d0)/d0)^(-alpha) * 10^(-X/10), where
    g0 = (c/(4*pi*f*d0))². Distances are in m, carrier in Hz, X in dB.
    The reference distance regularizes zero/short separation. Setting alpha=2
    and X=0 recovers free-space gain for d >= d0.
    """

    carrier_frequency_hz: float
    path_loss_exponent: float
    reference_distance_m: float = 1.0
    shadowing_std_db: float = 0.0
    rayleigh_fading: bool = False

    def __post_init__(self) -> None:
        for name in ("carrier_frequency_hz", "path_loss_exponent", "reference_distance_m"):
            object.__setattr__(self, name, require_positive_float(name, getattr(self, name)))
        object.__setattr__(self, "shadowing_std_db", require_non_negative_float("shadowing_std_db", self.shadowing_std_db))
        if not isinstance(self.rayleigh_fading, bool):
            raise TypeError("rayleigh_fading must be a boolean")
        if self.reference_gain_linear > 1.0:
            raise ValueError("reference distance is too small for the free-space reference gain")

    @property
    def reference_gain_linear(self) -> float:
        """Dimensionless reference gain g0, excluding antenna gains."""
        wavelength_m = SPEED_OF_LIGHT_MPS / self.carrier_frequency_hz
        return require_positive_float("reference_gain_linear", (wavelength_m / (4.0 * math.pi * self.reference_distance_m)) ** 2)

    def large_scale_gain_linear(self, distance_m: float, shadowing_db: float = 0.0) -> float:
        """Evaluate path gain; positive shadowing_db means additional attenuation."""
        distance = require_non_negative_float("distance_m", distance_m)
        shadowing = require_finite_float("shadowing_db", shadowing_db)
        normalized_distance = max(distance, self.reference_distance_m) / self.reference_distance_m
        gain = self.reference_gain_linear * normalized_distance ** (-self.path_loss_exponent) * db_to_linear(-shadowing)
        return require_positive_float("large_scale_gain_linear", gain)

    def realize(
        self, *, distance_m: float, tx_power_w: float, bandwidth_hz: float,
        noise_w: float, interference_w: float = 0.0,
        tx_antenna_gain_linear: float = 1.0, rx_antenna_gain_linear: float = 1.0,
        shadowing_db: float = 0.0, fading_power_gain_linear: float = 1.0,
    ) -> ChannelRealization:
        """Combine a supplied physical state, with no random draws or hidden cache."""
        gain = self.large_scale_gain_linear(distance_m, shadowing_db)
        fading = require_non_negative_float("fading_power_gain_linear", fading_power_gain_linear)
        sinr = received_sinr_linear(
            tx_power_w=tx_power_w, channel_gain_linear=gain, noise_w=noise_w,
            interference_w=interference_w, tx_antenna_gain_linear=tx_antenna_gain_linear,
            rx_antenna_gain_linear=rx_antenna_gain_linear,
        )
        return ChannelRealization(gain, fading, sinr, sinr * fading, bandwidth_hz)

    def sample(
        self, *, distance_m: float, tx_power_w: float, bandwidth_hz: float,
        noise_w: float, interference_w: float = 0.0,
        tx_antenna_gain_linear: float = 1.0, rx_antenna_gain_linear: float = 1.0,
        shadowing_rng: Random | None = None, fading_rng: Random | None = None,
    ) -> ChannelRealization:
        """Sample one log-normal shadow factor and unit-mean exponential power gain.

        Each enabled stochastic mechanism requires its own injected RNG. A
        deterministic experiment can use the defaults with no RNGs.
        """
        if self.shadowing_std_db > 0.0 and not isinstance(shadowing_rng, Random):
            raise TypeError("enabled shadowing requires an explicit Random instance")
        if self.rayleigh_fading and not isinstance(fading_rng, Random):
            raise TypeError("enabled fading requires an explicit Random instance")
        if self.shadowing_std_db > 0.0 and self.rayleigh_fading and shadowing_rng is fading_rng:
            raise ValueError("shadowing and fading require independent RNG streams")
        shadowing_db = (
            shadowing_rng.gauss(0.0, self.shadowing_std_db)
            if self.shadowing_std_db > 0.0 else 0.0
        )
        fading = fading_rng.expovariate(1.0) if self.rayleigh_fading else 1.0
        return self.realize(
            distance_m=distance_m, tx_power_w=tx_power_w, bandwidth_hz=bandwidth_hz,
            noise_w=noise_w, interference_w=interference_w,
            tx_antenna_gain_linear=tx_antenna_gain_linear,
            rx_antenna_gain_linear=rx_antenna_gain_linear,
            shadowing_db=shadowing_db, fading_power_gain_linear=fading,
        )

    def estimate_distance_m(
        self, *, filtered_sinr_linear: float, tx_power_w: float, noise_w: float,
        interference_w: float = 0.0, tx_antenna_gain_linear: float = 1.0,
        rx_antenna_gain_linear: float = 1.0,
    ) -> float:
        """Invert filtered SINR with nominal path loss, leaving shadowing unknown.

        Interference must be known or estimated consistently with the SINR.
        The estimate is clamped below at d0. Zero SINR provides no finite
        distance estimate and raises ValueError; an unavailable link is masked
        by the environment rather than assigned a fabricated distance.
        """
        filtered = require_positive_float("filtered_sinr_linear", filtered_sinr_linear)
        require_positive_float("tx_power_w", tx_power_w)
        reference_sinr = received_sinr_linear(
            tx_power_w=tx_power_w, channel_gain_linear=self.reference_gain_linear,
            noise_w=noise_w, interference_w=interference_w,
            tx_antenna_gain_linear=tx_antenna_gain_linear,
            rx_antenna_gain_linear=rx_antenna_gain_linear,
        )
        estimate = self.reference_distance_m * (reference_sinr / filtered) ** (1.0 / self.path_loss_exponent)
        return require_positive_float("estimated_distance_m", max(self.reference_distance_m, estimate))
