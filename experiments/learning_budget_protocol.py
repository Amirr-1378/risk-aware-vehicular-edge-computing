"""Predeclared development learning budgets; planning never starts training."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from risk_aware_v2x.entities._validation import require_non_negative_int, require_positive_int
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .learning_protocol import LearningProtocol
from .return_scope_protocol import _learning_from_dict
from .snr_representation_protocol import SNRRepresentationProtocol, VARIANTS


DEFAULT_LEARNING_BUDGET_PROTOCOL = Path(__file__).resolve().parent / "configs" / "learning_budget_v1.json"
DEFAULT_BUDGET_SCENARIO_PROTOCOL = Path(__file__).resolve().parent / "configs" / "development_budget_v1.json"
FIXED_POLICY_NAMES = ("always_mec", "always_v2v", "minimum_estimated_latency",
                      "uniform_random", "minimum_estimated_combined_cost")


@dataclass(frozen=True, slots=True)
class LearningBudgetProtocol:
    """One continuing chain per representation/algorithm/seed, fixed checkpoints.

    The nested v3 selection rule applies to the maximum training budget.
    Intermediate budgets are mandatory descriptive evaluations, never a
    validation-based stopping or best-checkpoint selection rule.
    """

    revision: str
    learning: LearningProtocol
    budgets: tuple[int, ...] = (20, 100, 500)
    variants: tuple[str, ...] = VARIANTS
    checkpoint_interval_episodes: int = 10
    return_scope: str = "individual"
    schema_version: int = 1
    purpose: str = "development-learning-budget-study-not-final-paper-protocol"
    selection_rule: str = "fixed-final-budget-no-validation-selection"

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if (self.schema_version != 1
                or self.purpose != "development-learning-budget-study-not-final-paper-protocol"):
            raise ValueError("requires learning-budget development protocol v1; main training is separate")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("learning-budget protocol revision is required")
        if not isinstance(self.learning, LearningProtocol):
            raise TypeError("learning must be an immutable LearningProtocol budget template")
        if not isinstance(self.budgets, tuple) or not self.budgets:
            raise ValueError("budgets must be a nonempty immutable tuple")
        for budget in self.budgets:
            require_positive_int("budget", budget)
        if any(left >= right for left, right in zip(self.budgets, self.budgets[1:])):
            raise ValueError("budgets must be strictly increasing without duplicates")
        if self.budgets[-1] != self.learning.training_episodes:
            raise ValueError("the final budget must equal the declared training episode count")
        require_positive_int("checkpoint_interval_episodes", self.checkpoint_interval_episodes)
        if self.selection_rule != "fixed-final-budget-no-validation-selection":
            raise ValueError("all budgets are predeclared; validation must not select or stop training")
        self._representation_controls()

    def _representation_controls(self):
        # Composition retains the exact approved QR pair, local actor scope,
        # gamma/alpha/lambda and fixed reference; its fingerprint is NOT the
        # identity of this study with its different budgets and checkpoint rule.
        return SNRRepresentationProtocol(self.revision + ":representation-controls", self.learning,
                                         variants=self.variants, return_scope=self.return_scope)

    def validate_against(self, scenarios: DevelopmentProtocol) -> None:
        """Check expanded trace capacity/binding before creating any output."""
        self._representation_controls().validate_against(scenarios)

    def encoder_for(self, variant: str, *, max_server_vehicles: int):
        return self._representation_controls().encoder_for(variant, max_server_vehicles=max_server_vehicles)

    def variant_contract(self, variant: str) -> dict:
        return self._representation_controls().variant_contract(variant)

    def next_budget(self, completed: int) -> int | None:
        """First strictly later evaluation boundary, or None at the final one."""
        completed = require_non_negative_int("completed_training_episodes", completed)
        if completed > self.budgets[-1]:
            raise ValueError("completed episodes exceed the fixed total budget")
        return next((budget for budget in self.budgets if budget > completed), None)

    def execution_plan(self) -> dict:
        """Count declared episode executions without generating any trajectory.

        One chain is trained once to the largest budget. Fixed policies run
        once per validation case/profile, not once per checkpoint or model.
        This does not estimate convergence, compute time or storage capacity.
        """
        chains = len(self.variants) * len(self.learning.algorithms) * len(self.learning.agent_seeds)
        validation_per_budget = len(self.learning.validation_profiles) * self.learning.validation_cases
        train = chains * self.budgets[-1]
        learned_validation = chains * len(self.budgets) * validation_per_budget
        fixed = len(FIXED_POLICY_NAMES) * validation_per_budget
        return {"plan_schema_version": 1, "planning_only": True,
            "model_chains": chains, "budgets": list(self.budgets),
            "training_episodes_per_chain": self.budgets[-1],
            "validation_episodes_per_budget_per_chain": validation_per_budget,
            "training_episode_executions": train,
            "learned_validation_episode_executions": learned_validation,
            "fixed_policy_episode_executions": fixed,
            "total_physical_episode_executions": train + learned_validation + fixed,
            "fixed_policy_names": list(FIXED_POLICY_NAMES),
            "checkpoint_interval_episodes": self.checkpoint_interval_episodes,
            "fixed_validation_cohort_across_budgets": True,
            "selection_rule": self.selection_rule,
            "main_training_started": False, "test_split_accessed": False}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data.get("schema_version") != 1
                or data.get("purpose") != "development-learning-budget-study-not-final-paper-protocol"):
            raise ValueError("requires explicit learning-budget development protocol v1")
        if set(data) != {"revision", "learning", "budgets", "variants", "checkpoint_interval_episodes",
                         "return_scope", "schema_version", "purpose", "selection_rule"}:
            raise ValueError("all and only learning-budget protocol fields must be explicit")
        data = dict(data)
        for name in ("budgets", "variants"):
            if not isinstance(data[name], list):
                raise TypeError(f"{name} must be a JSON array")
            data[name] = tuple(data[name])
        data["learning"] = _learning_from_dict(data["learning"])
        return cls(**data)
