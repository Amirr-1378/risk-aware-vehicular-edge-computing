"""Versioned paired individual/team-return development plan, never main training."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from risk_aware_v2x.agents import FeatureScales
from risk_aware_v2x.agents.neural import AgentConfig
from risk_aware_v2x.agents.schedules import EpsilonSchedule
from risk_aware_v2x.contracts import COST_MODEL_VERSION
from risk_aware_v2x.entities._validation import require_positive_int
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .learning_protocol import LearningProtocol

DEFAULT_RETURN_SCOPE_PROTOCOL = Path(__file__).resolve().parent / "configs" / "return_scope_pilot_v1.json"


def _learning_from_dict(record: dict) -> LearningProtocol:
    """Read a full v3 shared-budget template without reinterpreting old JSON."""
    if not isinstance(record, dict):
        raise TypeError("learning must contain a full learning-protocol object")
    data = dict(record)
    if (type(data.get("schema_version")) is not int or data.get("schema_version") != 3
            or type(data.get("cost_model_version")) is not int
            or data.get("cost_model_version") != COST_MODEL_VERSION):
        raise ValueError("nested learning requires protocol v3 with explicit cost model v2")
    for name in ("validation_profiles", "agent_seeds", "algorithms"):
        if not isinstance(data[name], list):
            raise TypeError(f"nested {name} must be a JSON array")
        data[name] = tuple(data[name])
    common = dict(data["common_agent"])
    if "algorithm" in common:
        raise ValueError("common_agent must not override the per-run algorithm")
    common["hidden_sizes"] = tuple(common["hidden_sizes"])
    data["common_agent"] = AgentConfig(**common)
    data["epsilon_schedule"] = EpsilonSchedule(**data["epsilon_schedule"])
    data["feature_scales"] = FeatureScales(**data["feature_scales"])
    return LearningProtocol(**data)


@dataclass(frozen=True, slots=True)
class ReturnScopeProtocol:
    """Both return scopes share one immutable learning/scenario budget.

    The nested v3 record is a shared hyperparameter/budget template. It does
    not change the original individual-only v3 runner or its saved artifacts.
    """

    revision: str
    learning: LearningProtocol
    return_scopes: tuple[str, ...] = ("individual", "team")
    schema_version: int = 1
    purpose: str = "development-return-scope-ablation-not-final-paper-protocol"

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if (self.schema_version != 1
                or self.purpose != "development-return-scope-ablation-not-final-paper-protocol"):
            raise ValueError("requires return-scope development protocol v1; main training is separate")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("return-scope protocol revision is required")
        if not isinstance(self.learning, LearningProtocol):
            raise TypeError("learning must be an immutable LearningProtocol shared-budget template")
        if (not isinstance(self.return_scopes, tuple) or len(self.return_scopes) != 2
                or set(self.return_scopes) != {"individual", "team"}):
            raise ValueError("both individual and team return scopes are required exactly once")
        self._validate_control_contract()

    def _validate_control_contract(self):
        common = self.learning.common_agent
        if common.gamma_per_slot != 1. or common.alpha != .9 or common.risk_weight != .5:
            raise ValueError("paired development requires gamma=1, alpha=.9 and template risk_weight=.5")

    def validate_against(self, scenarios: DevelopmentProtocol) -> None:
        """Reject plan mismatches/excessive split budgets before any execution."""
        self.learning.validate_against(scenarios)
        self._validate_control_contract()

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data.get("schema_version") != 1
                or data.get("purpose") != "development-return-scope-ablation-not-final-paper-protocol"):
            raise ValueError("requires explicit return-scope protocol schema v1; old plans are not upgraded")
        data = dict(data)
        if not {"revision", "learning", "return_scopes", "schema_version", "purpose"} <= set(data):
            raise ValueError("all return-scope protocol fields must be explicit")
        if not isinstance(data["return_scopes"], list):
            raise TypeError("return_scopes must be a JSON array")
        data["return_scopes"] = tuple(data["return_scopes"])
        data["learning"] = _learning_from_dict(data["learning"])
        return cls(**data)
