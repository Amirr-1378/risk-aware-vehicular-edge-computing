"""Descriptive dimensionless input statistics; no fitting or transformation.

Callers supply training observations and enforce their split/provenance bounds.
Absolute empirical percentiles use nearest rank: the value at rank ceil(p*n),
with p = .50, .95 or .99 and one-based ranks. They are descriptive sample
statistics, not confidence intervals, thresholds or proposed feature scales.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real


def _finite_real(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("input values must be real numbers, excluding booleans")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("input values must have a finite float representation") from error
    if not math.isfinite(result):
        raise ValueError("input values must be finite")
    return result


def describe_values(values) -> dict:
    """Describe a one-pass iterable of finite real dimensionless values.

    Empty data return count zero and None for every other statistic. RMS uses
    max-absolute rescaling so finite values such as 1e200 are never squared
    directly. Mean uses math.fsum; if its intermediate sum overflows, summands
    are max-absolute rescaled before averaging. No RNG or learned statistics enter.
    """
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("values must be a numeric iterable")
    data = [_finite_real(value) for value in values]
    count = len(data)
    if not count:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "rms": None, "max_abs": None, "p50_abs": None,
                "p95_abs": None, "p99_abs": None}
    absolute = sorted(abs(value) for value in data)
    largest = absolute[-1]
    try:
        mean = math.fsum(data) / count
    except OverflowError:
        mean = largest * (math.fsum(value / largest for value in data) / count)
    rms = largest * math.sqrt(math.fsum((value / largest) ** 2 for value in data) / count) if largest else 0.

    def percentile(numerator):
        # Exact rational ceilings avoid a floating-point rank-boundary error.
        rank = (numerator * count + 99) // 100
        return absolute[rank - 1]

    return {"count": count, "min": min(data), "max": max(data), "mean": mean,
            "rms": rms, "max_abs": largest, "p50_abs": percentile(50),
            "p95_abs": percentile(95), "p99_abs": percentile(99)}


def feature_statistics(observations, names) -> list[dict]:
    """Describe columns of finite dimensionless training observations.

    ``observations`` can be a one-shot outer iterable; each row must be a
    sequence with exactly one value per name. Names are a nonempty sequence
    of unique, nonblank strings. Empty data preserve the names with count zero.
    The original name order and every numeric value are retained; no clipping,
    feature selection, policy decision or adaptive scale fitting is performed.
    """
    if isinstance(names, (str, bytes, bytearray)) or not isinstance(names, Sequence):
        raise TypeError("feature names must be a sequence of strings")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if isinstance(observations, (str, bytes, bytearray, Mapping)):
        raise TypeError("observations must be an iterable of numeric rows")
    columns = [[] for _ in names]
    for row in observations:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise TypeError("each observation must be a numeric sequence")
        if len(row) != len(names):
            raise ValueError("observation width differs from feature names")
        for column, value in zip(columns, row, strict=True):
            column.append(_finite_real(value))
    return [{"name": name, **describe_values(column)}
            for name, column in zip(names, columns, strict=True)]
