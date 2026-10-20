"""Distributional-risk and CVaR utilities (no learning dependencies)."""

from .tails import (
    empirical_var, lower_cvar, mean_cvar_cost, mean_cvar_reward,
    select_cost_action, upper_cvar,
)

__all__ = ["empirical_var", "lower_cvar", "mean_cvar_cost", "mean_cvar_reward",
           "select_cost_action", "upper_cvar"]
