"""Explicit unit conversions; dB conversions always apply to power ratios."""

import math

from ..entities._validation import (
    require_finite_float,
    require_non_negative_int,
    require_positive_float,
)


def db_to_linear(value_db: float) -> float:
    """Convert a finite power ratio in dB to a positive linear ratio."""
    value = require_finite_float("value_db", value_db)
    try:
        result = 10.0 ** (value / 10.0)
    except OverflowError as error:
        raise ValueError("dB conversion exceeds floating-point range") from error
    return require_positive_float("linear ratio", result)


def linear_to_db(value_linear: float) -> float:
    """Convert a strictly positive power ratio to dB (zero has no finite dB)."""
    return 10.0 * math.log10(require_positive_float("value_linear", value_linear))


def dbm_to_watts(power_dbm: float) -> float:
    """Convert dBm (relative to 1 mW) to watts."""
    return db_to_linear(require_finite_float("power_dbm", power_dbm) - 30.0)


def watts_to_dbm(power_w: float) -> float:
    """Convert positive watts to dBm."""
    return linear_to_db(require_positive_float("power_w", power_w)) + 30.0


def bytes_to_bits(size_bytes: int) -> int:
    """Convert a non-negative integer byte count to bits, without rounding."""
    return 8 * require_non_negative_int("size_bytes", size_bytes)
