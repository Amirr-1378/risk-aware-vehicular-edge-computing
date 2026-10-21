"""Immutable development split/seed plan with a canonical configuration hash."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..entities._validation import require_fraction, require_non_negative_int, require_positive_int
from ..models.randomness import RandomStreams
from .generator import canonical_json
from .specification import ScenarioSpec

SPLITS = ("train", "validation", "test")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class EpisodeSeeds:
    split: str
    index: int
    scenario_seed: int
    channel_seed: int
    policy_seed: int


@dataclass(frozen=True, slots=True)
class DevelopmentProtocol:
    revision: str
    master_seed: int
    split_counts: tuple[int, int, int]
    agent_seeds: tuple[int, ...]
    profiles: tuple[ScenarioSpec, ...]
    cvar_alpha: float = .9
    bootstrap_resamples: int = 2_000
    bootstrap_confidence: float = .95
    schema_version: int = 1
    purpose: str = "development-only-not-final-paper-protocol"

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1 or self.purpose != "development-only-not-final-paper-protocol":
            raise ValueError("only development protocol v1 is supported")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("protocol revision is required")
        require_non_negative_int("master_seed", self.master_seed)
        if not isinstance(self.split_counts, tuple) or len(self.split_counts) != 3:
            raise ValueError("counts must be an immutable train/validation/test triple")
        for count in self.split_counts:
            require_positive_int("split count", count)
        if not isinstance(self.agent_seeds, tuple) or not self.agent_seeds:
            raise ValueError("explicit immutable agent seeds are required")
        for seed in self.agent_seeds:
            require_non_negative_int("agent seed", seed)
        if len(set(self.agent_seeds)) != len(self.agent_seeds):
            raise ValueError("agent seeds must be unique")
        if (not isinstance(self.profiles, tuple) or not self.profiles
                or not all(isinstance(profile, ScenarioSpec) for profile in self.profiles)
                or len({profile.label for profile in self.profiles}) != len(self.profiles)):
            raise ValueError("immutable, uniquely named scenario profiles are required")
        if len({profile.max_server_vehicle_slots for profile in self.profiles}) != 1:
            raise ValueError("all profiles must preserve a common neural action capacity")
        if require_fraction("cvar_alpha", self.cvar_alpha) == 1.:
            raise ValueError("CVaR confidence must be below one")
        require_positive_int("bootstrap_resamples", self.bootstrap_resamples)
        if require_fraction("bootstrap_confidence", self.bootstrap_confidence) in (0., 1.):
            raise ValueError("bootstrap confidence must lie strictly between zero and one")
        # Guard against accidental seed collisions in this finite plan; no
        # reroll based on measured outcomes and no seed-set filtering.
        used = set()
        for split, count in zip(SPLITS, self.split_counts, strict=True):
            for index in range(count):
                seeds = self.episode_seeds(split, index)
                for seed in (seeds.scenario_seed, seeds.channel_seed, seeds.policy_seed):
                    if seed in used or seed in self.agent_seeds:
                        raise ValueError("seed-role collision in finite plan")
                    used.add(seed)

    def episode_seeds(self, split: str, index: int) -> EpisodeSeeds:
        if split not in SPLITS:
            raise ValueError("unknown split")
        require_non_negative_int("episode index", index)
        if index >= self.split_counts[SPLITS.index(split)]:
            raise ValueError("episode index lies outside its predeclared split")
        streams = RandomStreams(self.master_seed)
        values = [streams.stream(f"development-plan-v1:{split}:{index}:{role}").getrandbits(63)
                  for role in ("scenario", "channel", "policy")]
        return EpisodeSeeds(split, index, *values)

    def profile(self, label: str) -> ScenarioSpec:
        for profile in self.profiles:
            if profile.label == label:
                return profile
        raise ValueError("unknown profile")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        base, profiles = data.pop("base_scenario"), data.pop("profile_overrides")
        counts = data.pop("split_counts")
        if set(counts) != set(SPLITS):
            raise ValueError("all and only the three declared splits are required")
        data["split_counts"] = tuple(counts[split] for split in SPLITS)
        data["agent_seeds"] = tuple(data["agent_seeds"])
        data["profiles"] = tuple(ScenarioSpec.from_dict({**base, **overrides, "label": label})
                                 for label, overrides in sorted(profiles.items()))
        return cls(**data)
