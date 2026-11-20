"""Fixed-budget behavior-warmup development plan; no change to the learned method."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from risk_aware_v2x.entities._validation import require_positive_int
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .return_scope_protocol import ReturnScopeProtocol, _learning_from_dict

DEFAULT_BEHAVIOR_WARMUP_PROTOCOL = Path(__file__).resolve().parent / "configs" / "behavior_warmup_pilot_v1.json"


def _paired_from_dict(record: dict) -> ReturnScopeProtocol:
    """Read an explicit nested paired plan, without upgrading older formats."""
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
            or record.get("purpose") != "development-return-scope-ablation-not-final-paper-protocol"):
        raise ValueError("paired must contain an explicit return-scope protocol v1")
    data = dict(record)
    if not {"revision", "learning", "return_scopes", "schema_version", "purpose"} <= set(data):
        raise ValueError("all nested return-scope fields must be explicit")
    if not isinstance(data["return_scopes"], list):
        raise TypeError("nested return_scopes must be a JSON array")
    data["return_scopes"] = tuple(data["return_scopes"])
    data["learning"] = _learning_from_dict(data["learning"])
    return ReturnScopeProtocol(**data)


@dataclass(frozen=True, slots=True)
class BehaviorWarmupProtocol:
    """Compare ordinary training and causal greedy behavior on the same budget.

    Teacher episodes replace only the first declared training episodes' behavior;
    they are not extra environment interactions or extra optimizer updates.
    Observation, physical costs, QR targets/loss, CVaR and inference stay intact.
    """

    revision: str
    paired: ReturnScopeProtocol
    teacher_episodes: int = 2
    variants: tuple[str, ...] = ("standard", "greedy_warmup")
    schema_version: int = 1
    purpose: str = "development-behavior-warmup-ablation-not-final-paper-protocol"

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if (self.schema_version != 1
                or self.purpose != "development-behavior-warmup-ablation-not-final-paper-protocol"):
            raise ValueError("requires behavior-warmup development protocol v1; main training is separate")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("behavior-warmup protocol revision is required")
        if not isinstance(self.paired, ReturnScopeProtocol):
            raise TypeError("paired must be an immutable ReturnScopeProtocol")
        require_positive_int("teacher_episodes", self.teacher_episodes)
        if self.teacher_episodes >= self.paired.learning.training_episodes:
            raise ValueError("teacher_episodes must leave at least one ordinary learner-behavior episode")
        if (not isinstance(self.variants, tuple) or len(self.variants) != 2
                or set(self.variants) != {"standard", "greedy_warmup"}):
            raise ValueError("both standard and greedy_warmup variants are required exactly once")

    def validate_against(self, scenarios: DevelopmentProtocol) -> None:
        """Validate the shared scene/budget contract before execution or output."""
        self.paired.validate_against(scenarios)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data.get("schema_version") != 1
                or data.get("purpose") != "development-behavior-warmup-ablation-not-final-paper-protocol"):
            raise ValueError("requires explicit behavior-warmup protocol v1; old plans are not upgraded")
        data = dict(data)
        if not {"revision", "paired", "teacher_episodes", "variants", "schema_version", "purpose"} <= set(data):
            raise ValueError("all behavior-warmup protocol fields must be explicit")
        if not isinstance(data["variants"], list):
            raise TypeError("variants must be a JSON array")
        data["variants"] = tuple(data["variants"])
        data["paired"] = _paired_from_dict(data["paired"])
        return cls(**data)
