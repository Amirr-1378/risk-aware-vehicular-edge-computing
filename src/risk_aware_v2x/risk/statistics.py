"""Episode-level descriptive risk and paired bootstrap mean-cost differences."""

import math
from dataclasses import dataclass
from random import Random
from collections.abc import Mapping, Sequence

from ..entities._validation import require_finite_float, require_fraction, require_positive_int
from .tails import empirical_var, upper_cvar


def _finite_samples(values):
    values = tuple(require_finite_float("sample", value) for value in values)
    if not values:
        raise ValueError("at least one episode sample is required")
    return values


def episode_cost_summary(costs: Sequence[float], *, alpha: float = .9) -> dict:
    values = _finite_samples(costs)
    return {"episodes": len(values), "mean_episode_total_cost": math.fsum(values) / len(values),
            "var_episode_total_cost": empirical_var(values, alpha=alpha),
            "cvar_episode_total_cost": upper_cvar(values, alpha=alpha), "alpha": alpha,
            "nominal_tail_mass_in_episodes": len(values) * (1. - alpha),
            "tail_estimation_warning": "descriptive_only; a small or dependent sample cannot establish reliable tail risk"}


@dataclass(frozen=True, slots=True)
class PairedMeanInterval:
    pairs: int
    mean_cost_difference_a_minus_b: float
    lower: float
    upper: float
    confidence: float
    resamples: int


def paired_mean_interval(a: Mapping[str, float], b: Mapping[str, float], *,
                         rng: Random, resamples: int = 2_000,
                         confidence: float = .95) -> PairedMeanInterval:
    """Resample matching independent episode keys jointly, never individual tasks.

    Percentile endpoints use the left-inverse empirical bootstrap CDF. A-B<0
    favors A for cost. This is not a p-value or a hierarchical trained-agent CI.
    """
    if set(a) != set(b) or len(a) < 2 or not all(isinstance(key, str) for key in a):
        raise ValueError("two or more identically keyed episode pairs are required")
    if not isinstance(rng, Random):
        raise TypeError("an explicit random.Random instance is required")
    require_positive_int("resamples", resamples)
    confidence = require_fraction("confidence", confidence)
    if confidence in (0., 1.):
        raise ValueError("confidence must be strictly between zero and one")
    differences = _finite_samples(require_finite_float("cost_a", a[key])
                                  - require_finite_float("cost_b", b[key]) for key in sorted(a))
    n = len(differences)
    draws = [math.fsum(differences[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(resamples)]
    tail = (1. - confidence) / 2
    return PairedMeanInterval(n, math.fsum(differences) / n,
                              empirical_var(draws, alpha=tail),
                              max(draws) if 1. - tail == 1. else empirical_var(draws, alpha=1. - tail),
                              confidence, resamples)
