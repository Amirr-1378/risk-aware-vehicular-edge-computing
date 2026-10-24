"""Versioned, bounded neural development runs, separate from physical models."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from risk_aware_v2x.agents import FeatureScales
from risk_aware_v2x.agents.encoding import ENCODED_OBSERVATION_SCHEMA_VERSION
from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig
from risk_aware_v2x.agents.schedules import EpsilonSchedule
from risk_aware_v2x.entities._validation import require_non_negative_int, require_positive_int
from risk_aware_v2x.environment.observations import RAW_OBSERVATION_SCHEMA_VERSION
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .comparison_policies import BASELINE_SUITE_VERSION


@dataclass(frozen=True, slots=True)
class LearningProtocol:
    revision: str
    scenario_protocol_sha256: str
    training_profile: str
    validation_profiles: tuple[str, ...]
    training_episodes: int
    validation_cases: int
    agent_seeds: tuple[int, ...]
    algorithms: tuple[str, ...]
    common_agent: AgentConfig
    epsilon_schedule: EpsilonSchedule
    feature_scales: FeatureScales
    selection_rule: str = "fixed-final-training-episode"
    schema_version: int = 2
    purpose: str = "development-learning-audit-not-final-paper-protocol"
    raw_observation_schema_version: int = RAW_OBSERVATION_SCHEMA_VERSION
    encoder_schema_version: int = ENCODED_OBSERVATION_SCHEMA_VERSION
    baseline_suite_version: int = BASELINE_SUITE_VERSION

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 2 or self.purpose != "development-learning-audit-not-final-paper-protocol":
            raise ValueError("use development learning protocol v2; v1 input/baseline contracts are not interchangeable")
        for name, version in (("raw_observation_schema_version", RAW_OBSERVATION_SCHEMA_VERSION),
                              ("encoder_schema_version", ENCODED_OBSERVATION_SCHEMA_VERSION),
                              ("baseline_suite_version", BASELINE_SUITE_VERSION)):
            require_positive_int(name, getattr(self, name))
            if getattr(self, name) != version:
                raise ValueError(f"unsupported {name}")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("learning protocol revision is required")
        if (not isinstance(self.scenario_protocol_sha256, str)
                or len(self.scenario_protocol_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.scenario_protocol_sha256)):
            raise ValueError("the exact scenario protocol SHA256 is required")
        if not isinstance(self.training_profile, str) or not self.training_profile.strip():
            raise ValueError("one training profile is required")
        for name in ("validation_profiles", "agent_seeds", "algorithms"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name} are not allowed")
        if any(not isinstance(label, str) or not label.strip() for label in self.validation_profiles):
            raise ValueError("validation profile names are required")
        for seed in self.agent_seeds:
            require_non_negative_int("agent seed", seed)
        if any(algorithm not in ALGORITHMS for algorithm in self.algorithms):
            raise ValueError("unknown learning algorithm")
        for name in ("training_episodes", "validation_cases"):
            require_positive_int(name, getattr(self, name))
        if self.selection_rule != "fixed-final-training-episode":
            raise ValueError("validation-based selection/early stopping is not supported")
        for value, kind in ((self.common_agent, AgentConfig), (self.epsilon_schedule, EpsilonSchedule),
                            (self.feature_scales, FeatureScales)):
            if not isinstance(value, kind):
                raise TypeError(f"expected {kind.__name__}")
        if self.common_agent.algorithm != "qr_dqn_cvar":
            raise ValueError("common_agent is a QR-CVaR template; only algorithm is replaced per run")

    def validate_against(self, scenarios: DevelopmentProtocol) -> None:
        """Reject every mismatch before constructing scenes or creating output."""
        if self.scenario_protocol_sha256 != scenarios.fingerprint:
            raise ValueError("learning/scenario protocol fingerprint mismatch")
        if not set(self.agent_seeds) <= set(scenarios.agent_seeds):
            raise ValueError("agent seeds must be predeclared in the scenario plan")
        if self.common_agent.alpha != scenarios.cvar_alpha:
            raise ValueError("learning and evaluation CVaR confidence must agree")
        for label in (self.training_profile, *self.validation_profiles):
            scenarios.profile(label)
        scenarios.episode_seeds("train", self.training_episodes - 1)
        scenarios.episode_seeds("validation", self.validation_cases - 1)

    def agent_configuration(self, algorithm: str) -> AgentConfig:
        if algorithm not in self.algorithms:
            raise ValueError("algorithm is outside the declared run plan")
        return replace(self.common_agent, algorithm=algorithm)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        for name in ("validation_profiles", "agent_seeds", "algorithms"):
            data[name] = tuple(data[name])
        common = dict(data["common_agent"])
        if "algorithm" in common:
            raise ValueError("common_agent must not override the per-run algorithm")
        common["hidden_sizes"] = tuple(common["hidden_sizes"])
        data["common_agent"] = AgentConfig(**common)
        data["epsilon_schedule"] = EpsilonSchedule(**data["epsilon_schedule"])
        data["feature_scales"] = FeatureScales(**data["feature_scales"])
        return cls(**data)
