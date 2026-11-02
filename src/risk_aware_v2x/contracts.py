"""Scientific compatibility boundary for the approved label-free objective.

Model v2 is not cost-compatible with historical criticality/penalty runs.
Weights are dimensionless shared objective coefficients, never task labels.
"""

from .entities._validation import require_non_negative_float

COST_MODEL_VERSION = 2
OBJECTIVE_WEIGHTS = (.60, .25, .15)
QUEUE_DISCIPLINE = "edf-absolute-deadline-arrival-task-id-v2"


def validate_objective_weights(latency: float, energy: float, mec: float) -> None:
    values = tuple(require_non_negative_float(name, value) for name, value in
                   zip(("latency_weight", "energy_weight", "mec_weight"), (latency, energy, mec), strict=True))
    if values != OBJECTIVE_WEIGHTS:
        raise ValueError("cost model v2 requires shared weights (0.60, 0.25, 0.15)")


def model_contract() -> dict:
    return {"cost_model_version": COST_MODEL_VERSION, "objective_weights": list(OBJECTIVE_WEIGHTS),
            "queue_discipline": QUEUE_DISCIPLINE, "manual_task_labels": False,
            "deadline_miss_penalty": False, "cost_or_reward": "cost-minimize; reward=-cost"}
