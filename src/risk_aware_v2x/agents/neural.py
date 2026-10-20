"""Optional CPU PyTorch learning agents; all outputs represent COST, not reward.

The CVaR variant is a risk-distorted QR controller, not an exact static-CVaR
optimizer. Neural imports are intentionally absent from agents/__init__.py.
"""

import copy
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

from ..entities._validation import (
    require_fraction, require_non_negative_int, require_positive_float, require_positive_int,
)
from ..models.randomness import RandomStreams
from .encoding import EncodedObservation, ObservationEncoder
from .replay import ReplayBuffer, validate_observation, validate_transition
from .transitions import DecisionTransition
from .updates import bootstrap_cost_targets, masked_cost_actions, quantile_huber_loss

ALGORITHMS = ("dqn", "double_dqn", "qr_dqn", "qr_dqn_cvar")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    algorithm: str = "qr_dqn_cvar"
    hidden_sizes: tuple[int, ...] = (128, 128)
    num_quantiles: int = 51
    learning_rate: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 10_000
    learning_starts: int = 128
    target_update_interval: int = 100
    huber_kappa: float = 1.
    max_grad_norm: float = 10.
    gamma_per_slot: float = 1.
    alpha: float = .9
    risk_weight: float = .5

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError("unknown learning algorithm")
        if not isinstance(self.hidden_sizes, tuple):
            raise TypeError("hidden_sizes must be an immutable tuple")
        for width in self.hidden_sizes:
            require_positive_int("hidden width", width)
        for name in ("num_quantiles", "batch_size", "replay_capacity", "learning_starts", "target_update_interval"):
            require_positive_int(name, getattr(self, name))
        if self.replay_capacity < max(self.batch_size, self.learning_starts):
            raise ValueError("replay must accommodate batch size and learning warm-up")
        for name in ("learning_rate", "huber_kappa", "max_grad_norm"):
            object.__setattr__(self, name, require_positive_float(name, getattr(self, name)))
        for name in ("gamma_per_slot", "alpha", "risk_weight"):
            object.__setattr__(self, name, require_fraction(name, getattr(self, name)))
        if self.alpha == 1.:
            raise ValueError("CVaR confidence must be below one")
        if self.algorithm.startswith("qr_") and self.num_quantiles < 2:
            raise ValueError("QR agents need at least two quantiles")

    @property
    def atom_count(self) -> int:
        return self.num_quantiles if self.algorithm.startswith("qr_") else 1

    @property
    def effective_risk_weight(self) -> float:
        return self.risk_weight if self.algorithm == "qr_dqn_cvar" else 0.


class CostNetwork(nn.Module):
    def __init__(self, input_dimension: int, action_count: int, atom_count: int,
                 hidden_sizes: tuple[int, ...]):
        super().__init__()
        widths = (input_dimension, *hidden_sizes, action_count * atom_count)
        layers = []
        for index, (source, target) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layers.append(nn.Linear(source, target, device="cpu", dtype=torch.float32))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.action_count, self.atom_count = action_count, atom_count

    def forward(self, values):
        return self.layers(values).reshape(-1, self.action_count, self.atom_count)


@dataclass(frozen=True, slots=True)
class UpdateStats:
    loss: float
    gradient_norm_before_clip: float
    gradient_updates: int
    target_synchronized: bool


def encoder_signature(encoder: ObservationEncoder) -> dict:
    return {"schema_version": encoder.schema_version, "dimension": encoder.dimension,
            "max_server_vehicles": encoder.max_server_vehicles,
            "feature_names": encoder.feature_names, "scales": asdict(encoder.scales)}


class NeuralCostAgent:
    """Shared per-user policy, uniform replay, Adam, hard-synchronized target net.

    epsilon is supplied by the experiment (same value for a complete action
    batch). Greedy evaluation consumes no exploration/replay randomness.
    """
    def __init__(self, config: AgentConfig, encoder: ObservationEncoder, *, seed: int):
        if not isinstance(config, AgentConfig):
            raise TypeError("config must be AgentConfig")
        self.config, self.seed = config, require_non_negative_int("seed", seed)
        self.signature = encoder_signature(encoder)
        self.input_dimension, self.action_count = encoder.dimension, encoder.max_server_vehicles + 1
        streams = RandomStreams(seed)
        initialization_seed = streams.stream("network-initialization").getrandbits(63)
        # nn.Linear initializes from torch's RNG; isolate and restore that state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.online = CostNetwork(self.input_dimension, self.action_count,
                                      config.atom_count, config.hidden_sizes)
        self.target = copy.deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate,
                                          foreach=False, fused=False)
        self.replay = ReplayBuffer(capacity=config.replay_capacity,
                                   seed=streams.stream("replay-seed").getrandbits(63))
        self._exploration_seed = streams.stream("exploration-seed").getrandbits(63)
        self._exploration: dict[int, Random] = {}
        self.seen_transitions = self.gradient_updates = 0

    def _validate_input(self, observation: EncodedObservation):
        validate_observation(observation)
        if len(observation.values) != self.input_dimension or len(observation.action_mask) != self.action_count:
            raise ValueError("encoded observation does not match network dimensions")

    @torch.no_grad()
    def act(self, observation: EncodedObservation, *, epsilon: float = 0.) -> int:
        self._validate_input(observation)
        epsilon = require_fraction("epsilon", epsilon)
        if epsilon > 0.:
            user_id = observation.user_vehicle_id
            if user_id not in self._exploration:
                self._exploration[user_id] = RandomStreams(self._exploration_seed).stream(f"user:{user_id}")
            rng = self._exploration[user_id]
            if rng.random() < epsilon:
                return rng.choice([index for index, valid in enumerate(observation.action_mask) if valid])
        atoms = self.online(torch.tensor([observation.values], dtype=torch.float32))
        masks = torch.tensor([observation.action_mask], dtype=torch.bool)
        return int(masked_cost_actions(atoms, masks, alpha=self.config.alpha,
                                      risk_weight=self.config.effective_risk_weight)[0])

    def _validate_sample(self, transition: DecisionTransition):
        validate_transition(transition)
        self._validate_input(transition.observation)
        if transition.next_observation is not None:
            self._validate_input(transition.next_observation)
        expected = 0. if transition.terminated else self.config.gamma_per_slot ** transition.duration_slots
        if not math.isclose(transition.bootstrap_discount, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("transition slot discount differs from agent configuration")

    def observe(self, transition: DecisionTransition):
        self._validate_sample(transition)
        self.replay.append(transition)
        self.seen_transitions += 1

    @torch.no_grad()
    def targets(self, batch: tuple[DecisionTransition, ...]):
        """Public numerical diagnostic; no gradient or replay/RNG mutation."""
        if not batch:
            raise ValueError("target batch cannot be empty")
        for item in batch:
            self._validate_sample(item)
        next_values = torch.zeros((len(batch), self.input_dimension), dtype=torch.float32)
        next_masks = torch.zeros((len(batch), self.action_count), dtype=torch.bool)
        for index, item in enumerate(batch):
            if item.next_observation is not None and item.bootstrap_discount > 0.:
                next_values[index] = torch.tensor(item.next_observation.values, dtype=torch.float32)
                next_masks[index] = torch.tensor(item.next_observation.action_mask, dtype=torch.bool)
        target_atoms = self.target(next_values)
        selector = self.online(next_values) if self.config.algorithm == "double_dqn" else None
        return bootstrap_cost_targets(
            torch.tensor([item.discounted_cost for item in batch], dtype=torch.float32),
            torch.tensor([item.bootstrap_discount for item in batch], dtype=torch.float32),
            target_atoms, next_masks, selector_atoms=selector, alpha=self.config.alpha,
            risk_weight=self.config.effective_risk_weight,
        )

    def learn(self) -> UpdateStats | None:
        if len(self.replay) < max(self.config.batch_size, self.config.learning_starts):
            return None
        batch = self.replay.sample(self.config.batch_size)
        values = torch.tensor([item.observation.values for item in batch], dtype=torch.float32)
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64)
        predictions = self.online(values)[torch.arange(len(batch)), actions]
        targets = self.targets(batch)
        if self.config.atom_count == 1:
            loss = F.huber_loss(predictions, targets, delta=self.config.huber_kappa)
        else:
            loss = quantile_huber_loss(predictions, targets, kappa=self.config.huber_kappa)
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in self.online.parameters()):
            raise ValueError("non-finite parameters after optimizer update")
        self.gradient_updates += 1
        synchronized = self.gradient_updates % self.config.target_update_interval == 0
        if synchronized:
            self.target.load_state_dict(self.online.state_dict())
        return UpdateStats(float(loss.detach()), float(norm), self.gradient_updates, synchronized)

    def save_checkpoint(self, path: str | Path, *, run_metadata: dict | None = None):
        """Exclusive creation: never overwrite a checkpoint. No simulator is pickled."""
        if run_metadata is not None and not isinstance(run_metadata, dict):
            raise TypeError("run_metadata must be a JSON-compatible dictionary")
        payload = {
            "checkpoint_schema_version": 1, "torch_version": str(torch.__version__),
            "python_version": platform.python_version(), "config": asdict(self.config),
            "encoder": self.signature, "seed": self.seed,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "seen_transitions": self.seen_transitions, "gradient_updates": self.gradient_updates,
            "exploration": {user_id: rng.getstate() for user_id, rng in self._exploration.items()},
            "run_metadata_json": json.dumps(run_metadata or {}, sort_keys=True, allow_nan=False),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as destination:
            torch.save(payload, destination)

    @classmethod
    def load_checkpoint(cls, path: str | Path, encoder: ObservationEncoder):
        """Load only a trusted local checkpoint; restricted tensor/primitive loader."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["checkpoint_schema_version"] != 1 or state["encoder"] != encoder_signature(encoder):
            raise ValueError("checkpoint schema or encoder configuration mismatch")
        if state["torch_version"] != str(torch.__version__):
            raise ValueError("exact PyTorch version required for reproducible training resume")
        config = dict(state["config"])
        config["hidden_sizes"] = tuple(config["hidden_sizes"])
        result = cls(AgentConfig(**config), encoder, seed=state["seed"])
        replay = ReplayBuffer.from_state_dict(state["replay"])
        if replay.capacity != result.config.replay_capacity:
            raise ValueError("checkpoint replay capacity mismatch")
        for item in replay:
            result._validate_sample(item)
        expected_groups = [{key: value for key, value in group.items() if key != "params"}
                           for group in result.optimizer.state_dict()["param_groups"]]
        restored_groups = [{key: value for key, value in group.items() if key != "params"}
                           for group in state["optimizer"]["param_groups"]]
        if restored_groups != expected_groups:
            raise ValueError("checkpoint optimizer hyperparameters differ from agent configuration")
        result.online.load_state_dict(state["online"], strict=True)
        result.target.load_state_dict(state["target"], strict=True)
        result.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in result.optimizer.state.values():
            if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                   for value in optimizer_state.values()):
                raise ValueError("checkpoint contains non-finite optimizer state")
        for network in (result.online, result.target):
            if not all(bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                raise ValueError("checkpoint contains non-finite model parameters")
        result.replay = replay
        result.seen_transitions = require_non_negative_int("seen_transitions", state["seen_transitions"])
        result.gradient_updates = require_non_negative_int("gradient_updates", state["gradient_updates"])
        if result.seen_transitions < len(replay):
            raise ValueError("checkpoint transition count is inconsistent")
        for user_id, rng_state in state["exploration"].items():
            require_non_negative_int("user_vehicle_id", user_id)
            rng = Random(0)
            rng.setstate(rng_state)
            result._exploration[user_id] = rng
        return result, json.loads(state["run_metadata_json"])
