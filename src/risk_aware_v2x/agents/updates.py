"""CPU cost-minimizing DQN/Double-Q targets and pairwise quantile Huber loss.

Import this module only with the optional learning dependency installed.
"""

import torch
from torch import Tensor

from ..entities._validation import require_fraction, require_positive_float


def cost_scores(atoms: Tensor, *, alpha: float = .9, risk_weight: float = 0.) -> Tensor:
    """[batch, actions, atoms] -> mean/upper-CVaR scores; lower is preferable.

    Validation of only mask-valid rows belongs to masked_cost_actions. This
    differentiable helper is used for inference/targets under no_grad.
    """
    alpha, weight = require_fraction("alpha", alpha), require_fraction("risk_weight", risk_weight)
    if (alpha == 1. or atoms.ndim != 3 or min(atoms.shape) < 1
            or not atoms.is_floating_point()):
        raise ValueError("expected nonempty atoms and confidence below one")
    mean = atoms.mean(dim=-1)
    if weight == 0.:
        return mean
    count = atoms.shape[-1]
    mass = count * (1. - alpha)
    ordered = atoms.sort(dim=-1, descending=True).values
    # Each descending atom contributes min(1, max(0, q-index)) units of mass.
    weights = (mass - torch.arange(count, dtype=atoms.dtype, device=atoms.device)).clamp(0., 1.) / mass
    tail = (ordered * weights).sum(dim=-1)
    return (1. - weight) * mean + weight * tail


def masked_cost_actions(atoms: Tensor, masks: Tensor, *, alpha: float = .9,
                        risk_weight: float = 0.) -> Tensor:
    if (atoms.ndim != 3 or masks.dtype != torch.bool or masks.shape != atoms.shape[:2]
            or not bool(masks.any(dim=1).all())):
        raise ValueError("each batch row needs a boolean mask and a valid action")
    if not bool(torch.isfinite(atoms[masks]).all()):
        raise ValueError("valid action atoms must be finite")
    # Remove masked NaN/Inf before computing scores, not just after reduction.
    safe = atoms.masked_fill(~masks.unsqueeze(-1), 0.)
    scores = cost_scores(safe, alpha=alpha, risk_weight=risk_weight)
    return scores.masked_fill(~masks, float("inf")).argmin(dim=1)


@torch.no_grad()
def bootstrap_cost_targets(costs: Tensor, discounts: Tensor, target_atoms: Tensor,
                           next_masks: Tensor, *, selector_atoms: Tensor | None = None,
                           alpha: float = .9, risk_weight: float = 0.) -> Tensor:
    """Select on target net (DQN/QR) or online net (Double DQN), evaluate target.

    Terminal/zero-discount rows are not selected or read at all; they may have
    all-false masks and NaN placeholders. Target result shape is [batch, atoms].
    """
    if (target_atoms.ndim != 3 or target_atoms.shape[-1] < 1 or costs.ndim != 1
            or costs.shape != discounts.shape or target_atoms.shape[0] != len(costs)
            or next_masks.shape != target_atoms.shape[:2] or next_masks.dtype != torch.bool):
        raise ValueError("inconsistent target batch shapes")
    if (not bool(torch.isfinite(costs).all()) or not bool(torch.isfinite(discounts).all())
            or bool(((discounts < 0.) | (discounts > 1.)).any())):
        raise ValueError("finite costs and discounts in [0,1] are required")
    source = target_atoms if selector_atoms is None else selector_atoms
    if source.shape != target_atoms.shape:
        raise ValueError("selector and target atom shapes must match")
    targets = costs[:, None].expand(-1, target_atoms.shape[-1]).clone()
    active = discounts > 0.
    if bool(active.any()):
        actions = masked_cost_actions(source[active], next_masks[active], alpha=alpha, risk_weight=risk_weight)
        chosen = target_atoms[active][torch.arange(int(active.sum()), device=target_atoms.device), actions]
        if not bool(torch.isfinite(chosen).all()):
            raise ValueError("selected target atoms must be finite")
        targets[active] += discounts[active, None] * chosen
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("non-finite Bellman targets")
    return targets


def quantile_huber_loss(predictions: Tensor, targets: Tensor, *, kappa: float = 1.) -> Tensor:
    """Mean batch, SUM predicted quantiles, mean target atoms; unsorted tau_i.

    rho_tau(delta)=|tau-I(delta<0)|*Huber_kappa(delta), without division by
    kappa. Original QR-DQN scaling: tau_i=(i+.5)/N; all N*M pairs are used.
    Targets and the sign indicator are detached from the gradient.
    """
    kappa = require_positive_float("kappa", kappa)
    if (predictions.ndim != 2 or targets.ndim != 2 or predictions.shape[0] != targets.shape[0]
            or min(*predictions.shape, *targets.shape) < 1):
        raise ValueError("predictions and targets must be nonempty [batch, atoms]")
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("loss inputs must be finite")
    delta = targets.detach()[:, None, :] - predictions[:, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= kappa, .5 * delta.square(), kappa * (absolute - .5 * kappa))
    tau = (torch.arange(predictions.shape[1], device=predictions.device, dtype=predictions.dtype) + .5) / predictions.shape[1]
    weight = (tau[None, :, None] - (delta.detach() < 0.).to(predictions.dtype)).abs()
    return (weight * huber).mean(dim=2).sum(dim=1).mean()
