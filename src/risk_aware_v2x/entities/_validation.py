"""Shared validation helpers for simulation entities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Real


def require_non_negative_int(name: str, value: object) -> int:
    """Return *value* as an integer after rejecting booleans and negatives."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def require_positive_int(name: str, value: object) -> int:
    """Return *value* after checking that it is a strictly positive integer."""
    result = require_non_negative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def require_finite_float(name: str, value: object) -> float:
    """Return a finite floating-point representation of a real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_non_negative_float(name: str, value: object) -> float:
    """Return a finite, non-negative floating-point value."""
    result = require_finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def require_positive_float(name: str, value: object) -> float:
    """Return a finite, strictly positive floating-point value."""
    result = require_finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def require_fraction(name: str, value: object) -> float:
    """Return a finite fraction in the closed interval [0, 1]."""
    result = require_finite_float(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def require_position_m(value: object) -> tuple[float, float]:
    """Validate and normalize a two-dimensional position measured in metres."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("position_m must be a two-dimensional sequence")
    if len(value) != 2:
        raise ValueError("position_m must contain exactly two coordinates")
    return (
        require_finite_float("position_m[0]", value[0]),
        require_finite_float("position_m[1]", value[1]),
    )


def require_mobility_profile_id(value: object) -> int | str:
    """Validate a stable integer or non-empty string mobility-profile identifier."""
    if isinstance(value, bool):
        raise TypeError("mobility_profile_id must be an integer or string")
    if isinstance(value, int):
        return require_non_negative_int("mobility_profile_id", value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("mobility_profile_id must be non-negative or a non-empty string")


def require_unique_ids(name: str, values: Iterable[object]) -> list[int]:
    """Validate a collection of non-negative, unique integer identifiers."""
    result = [require_non_negative_int(name, value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate identifiers")
    return result
