"""Smooth, non-probabilistic MEC preference of a learned scalar cost gap."""

import math

from ..entities._validation import require_finite_float, require_positive_float


def sigmoid_mec_preference(delta_cost: float, *, beta: float = 1.) -> float:
    """sigmoid(beta*(J_SV-J_MEC)): 0 favors SV, .5 tie, 1 favors MEC.

    beta is positive and has inverse-score units (dimensionless here). This
    is not an action probability and must NEVER select/sample an action or
    modify a real cost/TD target. Stable branches allow product overflow to
    saturate at the correct limit without exponential overflow. Finite float
    saturation can collapse distinct gaps: always rank the original J values.
    """
    delta = require_finite_float("delta_cost", delta_cost)
    slope = require_positive_float("sigmoid_beta", beta)
    x = slope * delta
    if x >= 0.:
        return 1. / (1. + math.exp(-x))
    exponential = math.exp(x)
    return exponential / (1. + exponential)
