"""Fixed paired SNR-representation development plan, never main training."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from risk_aware_v2x.agents import ObservationEncoder
from risk_aware_v2x.agents.encoding import LogSNRObservationEncoder
from risk_aware_v2x.entities._validation import require_positive_int
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .learning_protocol import LearningProtocol
from .return_scope_protocol import _learning_from_dict

VARIANTS = ("linear_v3", "log_snr_v4")
QR_ALGORITHMS = ("qr_dqn", "qr_dqn_cvar")
DEFAULT_SNR_REPRESENTATION_PROTOCOL = Path(__file__).resolve().parent / "configs" / "snr_representation_pilot_v1.json"


@dataclass(frozen=True, slots=True)
class SNRRepresentationProtocol:
    """One shared v3 budget template, with explicit per-arm encoder contracts.

    The nested v3 encoder field remains the legacy control template. It does
    not relabel schema-4 observations or make old weights compatible with them.
    """

    revision: str
    learning: LearningProtocol
    variants: tuple[str, ...] = VARIANTS
    return_scope: str = "individual"
    schema_version: int = 1
    purpose: str = "development-snr-representation-ablation-not-final-paper-protocol"

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1 or self.purpose != "development-snr-representation-ablation-not-final-paper-protocol":
            raise ValueError("requires SNR-representation development protocol v1; main training is separate")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("SNR-representation protocol revision is required")
        if not isinstance(self.learning, LearningProtocol):
            raise TypeError("learning must be an immutable LearningProtocol budget template")
        if (not isinstance(self.variants, tuple) or len(self.variants) != 2
                or any(not isinstance(value, str) for value in self.variants)
                or set(self.variants) != set(VARIANTS)):
            raise ValueError("both linear_v3 and log_snr_v4 are required exactly once")
        if self.return_scope != "individual":
            raise ValueError("representation comparison keeps individual returns fixed")
        self._validate_controls()

    def _validate_controls(self):
        learning, common = self.learning, self.learning.common_agent
        if len(learning.algorithms) != 2 or set(learning.algorithms) != set(QR_ALGORITHMS):
            raise ValueError("both risk-neutral QR and QR-CVaR controls are required")
        if common.gamma_per_slot != 1. or common.alpha != .9 or common.risk_weight != .5:
            raise ValueError("representation comparison requires gamma=1, alpha=.9 and risk_weight=.5")
        if learning.feature_scales.snr_linear != 100.:
            raise ValueError("the pre-existing fixed linear SNR reference must remain 100")
        if (learning.schema_version != 3 or learning.raw_observation_schema_version != 3
                or learning.encoder_schema_version != 3 or learning.cost_model_version != 2):
            raise ValueError("shared controls must remain learning v3 / raw v3 / encoder v3 / cost v2")

    def validate_against(self, scenarios: DevelopmentProtocol) -> None:
        self.learning.validate_against(scenarios)
        self._validate_controls()

    def encoder_for(self, variant: str, *, max_server_vehicles: int):
        if variant not in self.variants:
            raise ValueError("representation variant is outside the declared plan")
        kind = ObservationEncoder if variant == "linear_v3" else LogSNRObservationEncoder
        return kind(max_server_vehicles=max_server_vehicles, scales=self.learning.feature_scales)

    def variant_contract(self, variant: str) -> dict:
        encoder = self.encoder_for(variant, max_server_vehicles=0)
        representation = (encoder.representation_contract() if variant == "log_snr_v4" else
            {"schema_version": 1, "kind": "linear-snr-v3", "input": "cached-large-scale-linear-sinr",
             "formula": "snr_linear/reference_snr_linear", "reference_snr_linear": 100.,
             "fitted": False, "clipped": False})
        return {"representation_variant": variant, "encoder_schema_version": encoder.schema_version,
                **representation}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data.get("schema_version") != 1
                or data.get("purpose") != "development-snr-representation-ablation-not-final-paper-protocol"):
            raise ValueError("requires explicit SNR-representation protocol schema v1")
        if set(data) != {"revision", "learning", "variants", "return_scope", "schema_version", "purpose"}:
            raise ValueError("all and only SNR-representation protocol fields must be explicit")
        if not isinstance(data["variants"], list):
            raise TypeError("variants must be a JSON array")
        data = dict(data)
        data["variants"] = tuple(data["variants"])
        data["learning"] = _learning_from_dict(data["learning"])
        return cls(**data)
