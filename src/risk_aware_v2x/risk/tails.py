"""Exact tail integrals for finite equally weighted empirical distributions.

QR-DQN atoms have equal probability mass, not probability proportional to their
value. Fractional mass at the tail boundary is essential when N*(1-alpha) is
not an integer. These functions also accept negative returns.
"""

import math
from collections.abc import Sequence

from ..entities._validation import require_finite_float, require_fraction


def _atoms(values: Sequence[float]) -> tuple[float, ...]:
    atoms = tuple(require_finite_float("atom", value) for value in values)
    if not atoms:
        raise ValueError("a distribution needs at least one atom")
    return atoms


def _confidence(alpha: float) -> float:
    alpha = require_fraction("alpha", alpha)
    if alpha == 1.0:
        raise ValueError("alpha must be strictly less than one")
    return alpha


def empirical_var(values: Sequence[float], *, alpha: float = .9) -> float:
    """Left-inverse empirical CDF; at alpha=0 return the minimum atom."""
    alpha = _confidence(alpha)
    atoms = sorted(_atoms(values))
    # Compare CDF probabilities directly: ceil(alpha*N) can round an exact
    # decimal grid boundary upward, e.g. float(.14)*100 > 14.
    return next(value for index, value in enumerate(atoms, start=1)
                if index / len(atoms) >= alpha)


def upper_cvar(values: Sequence[float], *, alpha: float = .9) -> float:
    """Average of the worst (largest) 1-alpha probability mass, including ties."""
    alpha = _confidence(alpha)
    atoms = sorted(_atoms(values), reverse=True)
    mass = (1.0 - alpha) * len(atoms)
    # Clamp only roundoff in counts, never the requested probability itself.
    full = min(len(atoms), math.floor(mass))
    fraction = mass - full
    terms = [value / mass for value in atoms[:full]]
    if full < len(atoms):
        terms.append(atoms[full] * (fraction / mass))
    return math.fsum(terms)


def lower_cvar(values: Sequence[float], *, alpha: float = .9) -> float:
    """Average of the smallest 1-alpha mass (reward convention, same confidence)."""
    return -upper_cvar(tuple(-value for value in _atoms(values)), alpha=alpha)


def mean_cvar_cost(values: Sequence[float], *, alpha: float = .9,
                   risk_weight: float = .5) -> float:
    """Cost score to MINIMIZE: (1-lambda)*mean + lambda*upper-CVaR."""
    atoms = _atoms(values)
    weight = require_fraction("risk_weight", risk_weight)
    _confidence(alpha)
    mean = math.fsum(value / len(atoms) for value in atoms)
    return (1.0 - weight) * mean + weight * upper_cvar(atoms, alpha=alpha)


def mean_cvar_reward(values: Sequence[float], *, alpha: float = .9,
                     risk_weight: float = .5) -> float:
    """Reward score to MAXIMIZE; exactly sign-equivalent to negated costs."""
    return -mean_cvar_cost(tuple(-value for value in _atoms(values)),
                          alpha=alpha, risk_weight=risk_weight)


def select_cost_action(quantiles: Sequence[Sequence[float]], mask: Sequence[bool], *,
                       alpha: float = .9, risk_weight: float = .5) -> int:
    """Select a valid action, ties by lowest index. Masked rows are never read.

    Each valid action must have the same positive atom count. At inference we
    sort a copy for integration even if a network's quantiles cross; training
    must retain the original quantile fractions and is not implemented here.
    """
    _confidence(alpha)
    require_fraction("risk_weight", risk_weight)
    if len(quantiles) != len(mask) or not all(isinstance(flag, bool) for flag in mask):
        raise ValueError("mask must be boolean and match the action dimension")
    valid = [index for index, flag in enumerate(mask) if flag]
    if not valid:
        raise ValueError("at least one action must be valid")
    count = len(quantiles[valid[0]])
    if count == 0 or any(len(quantiles[index]) != count for index in valid):
        raise ValueError("valid actions must have an equal, positive atom count")
    return min(valid, key=lambda index: (
        mean_cvar_cost(quantiles[index], alpha=alpha, risk_weight=risk_weight), index,
    ))
