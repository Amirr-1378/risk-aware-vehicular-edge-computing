"""Read-only training-replay input diagnostics, without fitting or new training.

Archived checkpoint identity is distinct from this auditor's provenance. Loading
for numerical inspection never relaxes the experiment runner's resume guards.
"""

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from risk_aware_v2x.agents import FeatureScales, ObservationEncoder
from risk_aware_v2x.agents.neural import AgentConfig, NeuralCostAgent, encoder_signature
from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
from risk_aware_v2x.agents.updates import cost_scores, quantile_huber_loss
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_non_negative_float, require_non_negative_int, require_positive_int
from .input_statistics import describe_values, feature_statistics
from .learning_runtime import deterministic_cpu
from .provenance import source_provenance
from .run_learning_audit import model_fingerprint
from .run_return_scope_audit import _same_state, _training_state
from .runtime_preflight import runtime_identity
from .session_journal import inspect_session


SESSION_PURPOSE = "resumable-single-model-development-not-main-training"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON: {value}")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=_reject_constant)


def _global_state():
    kind, keys, position, has_gauss, cached = numpy.random.get_state()
    return {"python": random.getstate(), "numpy": (kind, keys.tobytes(), position, has_gauss, cached),
        "torch": torch.get_rng_state().clone(), "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def _agent_state(agent):
    return {"training": _training_state(agent),
        "gradients": [None if p.grad is None else p.grad.detach().clone() for p in agent.online.parameters()],
        "requires_grad": [p.requires_grad for p in agent.online.parameters()],
        "config": asdict(agent.config), "signature": copy.deepcopy(agent.signature),
        "seed": agent.seed, "exploration_seed": agent._exploration_seed}


def _geometry(rows, names, network):
    """Stable descriptive mass, not variance, attribution or causal influence."""
    largest = max((abs(value) for row in rows for value in row), default=0.)
    mass = [math.fsum((row[index] / largest) ** 2 for row in rows) if largest else 0.
            for index in range(len(names))]
    layer = next((module for module in network.modules() if isinstance(module, nn.Linear)), None)
    if layer is None or layer.in_features != len(names):
        raise ValueError("first linear layer must match input width")
    if not bool(torch.isfinite(layer.weight).all()):
        raise ValueError("first linear layer weights must be finite")
    weights = layer.weight.detach().double().abs().tolist()
    weight_scale = max((value for row in weights for value in row), default=0.)
    contributions = [(math.fsum(abs(row[index]) / largest for row in rows)
                      * math.fsum(row[index] / weight_scale for row in weights))
                     if largest and weight_scale else 0. for index in range(len(names))]

    def shares(values):
        total = math.fsum(values)
        return [value / total if total else None for value in values]

    squared, linear = shares(mass), shares(contributions)
    snr_indices = [index for index, name in enumerate(names) if name.startswith("server[") and name.endswith(".snr")]
    return {"squared_mass_shares": [{"name": name, "share": value} for name, value in zip(names, squared, strict=True)],
        "snr_squared_mass_share": math.fsum(squared[i] for i in snr_indices) if math.fsum(mass) else None,
        "first_linear_absolute_contribution_shares": [
            {"name": name, "share": value} for name, value in zip(names, linear, strict=True)],
        "snr_first_linear_absolute_contribution_share": math.fsum(linear[i] for i in snr_indices) if math.fsum(contributions) else None,
        "interpretation": "Squared input mass and online-first-layer sum_obs(abs(x_f))*sum_units(abs(W_uf)); bias excluded; not causal attribution."}


def _presence(observations, names):
    result = []
    for index, name in enumerate(names):
        if name.startswith("server[") and name.endswith(".present"):
            slot = int(name.split("[", 1)[1].split("]", 1)[0])
            if any(o.values[index] not in (0., 1.) for o in observations):
                raise ValueError("server presence must be zero or one")
            result.append({"slot": slot, "present_count": sum(o.values[index] == 1. for o in observations),
                           "action_valid_count": sum(o.action_mask[slot + 1] for o in observations)})
    return result


def audit_agent_inputs(agent, samples: tuple, feature_names: tuple) -> dict:
    """Inspect an ordered retained TRAINING cohort; no actions, sampling or updates.

    Fixed-cohort loss is retrospective, not the recorded historical minibatch
    loss. Current and active-next observations are separate correlated samples.
    """
    if not isinstance(samples, tuple) or not isinstance(feature_names, tuple):
        raise TypeError("samples and feature names must be immutable tuples")
    if len(feature_names) != agent.input_dimension or feature_names != tuple(agent.signature["feature_names"]):
        raise ValueError("feature names differ from network input width")
    for sample in samples:
        agent._validate_sample(sample)
    frozen, global_before = _agent_state(agent), _global_state()
    current = tuple(item.observation for item in samples)
    following = tuple(item.next_observation for item in samples
                      if item.next_observation is not None and item.bootstrap_discount > 0.)
    current_rows, next_rows = [o.values for o in current], [o.values for o in following]
    predictions = {key: [] for key in ("valid_action_atoms", "valid_action_mean_cost",
        "valid_action_upper_cvar_cost", "valid_action_risk_cost", "recorded_action_atoms")}
    targets = {key: [] for key in ("all", "bootstrap_active", "terminal_or_zero_discount")}
    losses = []
    with deterministic_cpu(), torch.no_grad():
        for observations in (current, following):
            if observations and not bool(torch.isfinite(torch.tensor([o.values for o in observations], dtype=torch.float32)).all()):
                raise ValueError("finite encoded values overflow float32 network inputs")
        for start in range(0, len(samples), agent.config.batch_size):
            batch = samples[start:start + agent.config.batch_size]
            values = torch.tensor([item.observation.values for item in batch], dtype=torch.float32)
            masks = torch.tensor([item.observation.action_mask for item in batch], dtype=torch.bool)
            atoms = agent.online(values)
            if not bool(torch.isfinite(atoms[masks]).all()):
                raise ValueError("non-finite valid-action predictions")
            safe = atoms.masked_fill(~masks.unsqueeze(-1), 0.)
            predictions["valid_action_atoms"].extend(atoms[masks].reshape(-1).tolist())
            for key, weight in (("valid_action_mean_cost", 0.), ("valid_action_upper_cvar_cost", 1.),
                                ("valid_action_risk_cost", agent.config.effective_risk_weight)):
                predictions[key].extend(cost_scores(safe, alpha=agent.config.alpha, risk_weight=weight)[masks].tolist())
            chosen = atoms[torch.arange(len(batch)), torch.tensor([item.action for item in batch])]
            predictions["recorded_action_atoms"].extend(chosen.reshape(-1).tolist())
            target = agent.targets(batch)
            targets["all"].extend(target.reshape(-1).tolist())
            for index, item in enumerate(batch):
                key = "bootstrap_active" if item.bootstrap_discount > 0. else "terminal_or_zero_discount"
                targets[key].extend(target[index].tolist())
            loss = (F.huber_loss(chosen, target, delta=agent.config.huber_kappa) if agent.config.atom_count == 1
                    else quantile_huber_loss(chosen, target, kappa=agent.config.huber_kappa))
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite fixed-cohort diagnostic loss")
            losses.append(float(loss) * len(batch))
        result = {"sample_count": len(samples), "next_bootstrap_count": len(following),
            "feature_names": list(feature_names),
            "current_features": feature_statistics(current_rows, feature_names),
            "next_bootstrap_features": feature_statistics(next_rows, feature_names),
            "input_geometry": {"current": _geometry(current_rows, feature_names, agent.online),
                               "next_bootstrap": _geometry(next_rows, feature_names, agent.online)},
            "server_presence": {"current": _presence(current, feature_names),
                                "next_bootstrap": _presence(following, feature_names)},
            "costs": {"interval": describe_values(item.cost for item in samples),
                      "discounted_interval": describe_values(item.discounted_cost for item in samples)},
            "predictions": {key: describe_values(values) for key, values in predictions.items()},
            "targets": {key: describe_values(values) for key, values in targets.items()},
            "fixed_cohort_loss": math.fsum(losses) / len(samples) if samples else None}
    if not _same_state(frozen, _agent_state(agent)) or not _same_state(global_before, _global_state()):
        raise RuntimeError("input audit mutated learner, RNG or execution settings")
    result["mutation_checks"] = {"learner_state_unchanged": True,
        "global_rng_and_execution_settings_unchanged": True}
    return result


def _validated_contract(contract):
    for key in ("session_schema_version", "updates_per_new_transition"):
        require_positive_int(key, contract[key])
    source = contract["source_sha256"]
    if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("invalid archived source fingerprint")
    for key, expected in (("session_schema_version", 1), ("purpose", SESSION_PURPOSE),
                          ("model_contract", model_contract()), ("training_behavior", "standard-learned-epsilon"),
                          ("selection_rule", "fixed-final-training-episode"), ("updates_per_new_transition", 1),
                          ("warmup_counting", "total-transitions"), ("main_training_started", False),
                          ("test_split_accessed", False), ("centralized_controller", False)):
        if contract.get(key) != expected or isinstance(expected, bool) and contract.get(key) is not expected:
            raise ValueError(f"unsupported session contract: {key}")
    learning, scenarios = contract["learning_protocol"], contract["scenario_protocol"]
    if (learning.get("schema_version") != 3 or scenarios.get("schema_version") != 2
            or learning.get("cost_model_version") != 2 or scenarios.get("cost_model_version") != 2
            or learning.get("purpose") != "development-learning-audit-not-final-paper-protocol"
            or scenarios.get("purpose") != "development-only-not-final-paper-protocol"
            or learning.get("selection_rule") != contract["selection_rule"]):
        raise ValueError("incompatible scientific protocol")
    for name, plan in (("learning_protocol", learning), ("scenario_protocol", scenarios)):
        if _digest(plan) != contract[name + "_sha256"]:
            raise ValueError("scientific protocol fingerprint mismatch")
    if learning["scenario_protocol_sha256"] != contract["scenario_protocol_sha256"]:
        raise ValueError("learning/scenario binding mismatch")
    config_data = dict(contract["agent_configuration"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    config = AgentConfig(**config_data)
    common = dict(learning["common_agent"])
    common["algorithm"] = contract["algorithm"]
    if (_canonical(common) != _canonical(asdict(config)) or config.algorithm != contract["algorithm"]
            or config.algorithm not in learning["algorithms"] or contract["agent_seed"] not in learning["agent_seeds"]
            or contract["agent_seed"] not in scenarios["agent_seeds"]):
        raise ValueError("session algorithm/configuration/seed mismatch")
    require_non_negative_int("agent_seed", contract["agent_seed"])
    encoder = ObservationEncoder(max_server_vehicles=contract["encoder"]["max_server_vehicles"],
                                 scales=FeatureScales(**contract["encoder"]["scales"]))
    if (_canonical(encoder_signature(encoder)) != _canonical(contract["encoder"])
            or contract["encoder"]["scales"] != learning["feature_scales"]):
        raise ValueError("session encoder mismatch")
    scope = contract["return_scope"]
    if scope not in ("individual", "team"):
        raise ValueError("unsupported return scope")
    kind = TeamNeuralCostAgent if scope == "team" else NeuralCostAgent
    if contract["learning_return_contract"] != kind.learning_return_contract():
        raise ValueError("learning return contract mismatch")
    for split, count_key, sequence_key in (("train", "training_episodes", "training_sequence"),
                                           ("validation", "validation_cases", "validation_sequence")):
        count = require_positive_int(count_key, learning[count_key])
        sequence = contract[sequence_key]
        if not isinstance(sequence, list) or len(sequence) != count:
            raise ValueError("declared sequence budget mismatch")
        for index, seed in enumerate(sequence):
            if seed["split"] != split or type(seed["index"]) is not int or seed["index"] != index:
                raise ValueError("declared split/index order mismatch")
            for key in ("scenario_seed", "channel_seed", "policy_seed"):
                require_non_negative_int(key, seed[key])
    return config, encoder, kind


def _training_history(session, entries, contract, config):
    if not entries:
        raise ValueError("session needs a committed initialization checkpoint")
    planned, previous, trace = contract["learning_protocol"]["training_episodes"], None, []
    progress_names = ("completed_training_episodes", "training_slots", "seen_transitions", "gradient_updates", "replay_size")
    for index, entry in enumerate(entries):
        meta = entry["metadata"]
        for key in progress_names:
            require_non_negative_int(key, meta[key])
        for key in ("model_sha256", "initial_model_sha256"):
            value = meta[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid model fingerprint")
        phase = "initialized" if index == 0 else "training" if index <= planned else "validation"
        if (index > planned + 1 or meta["phase"] != phase
                or meta["completed_training_episodes"] != min(index, planned)
                or set(entry["artifacts"]) != ({"report.json"} if phase == "validation" else {"report.json", "agent.pt"})):
            raise ValueError("session phase/budget/artifact sequence mismatch")
        if phase == "validation":
            if any(meta[key] != previous[key] for key in previous if key != "phase"):
                raise ValueError("validation changed learner progress")
        else:
            report = _read_json(session / "steps" / f"{index:06d}" / "report.json")
            if phase == "initialized":
                if (any(meta[key] != 0 for key in progress_names) or meta["model_sha256"] != meta["initial_model_sha256"]
                        or report.get("purpose") != SESSION_PURPOSE):
                    raise ValueError("invalid initialization state")
            else:
                episode = report["episode"]
                count = require_non_negative_int("transition_count", episode["transition_count"])
                elapsed = require_positive_int("elapsed_slots", episode["elapsed_slots"])
                for key in ("environment_seed", "training_slots_before", "training_slots_after"):
                    require_non_negative_int(key, episode[key])
                seen = previous["seen_transitions"] + count
                updates = max(0, seen - max(config.batch_size, config.learning_starts) + 1)
                if (episode["training"] is not True or episode["seeds"] != contract["training_sequence"][index - 1]
                        or episode["episode_id"] != f'{contract["learning_protocol"]["training_profile"]}:train:{index - 1}'
                        or episode["environment_seed"] != episode["seeds"]["channel_seed"]
                        or episode["training_slots_before"] != previous["training_slots"]
                        or episode["training_slots_after"] != previous["training_slots"] + elapsed
                        or meta["training_slots"] != episode["training_slots_after"]
                        or meta["seen_transitions"] != seen or meta["gradient_updates"] != updates
                        or meta["replay_size"] != min(seen, config.replay_capacity)
                        or episode["metrics"]["cost_model_version"] != model_contract()["cost_model_version"]
                        or episode["metrics"]["arrived_tasks"] != count or episode["metrics"]["completed_tasks"] != count
                        or len(episode["updates"]) != updates - previous["gradient_updates"]
                        or "training_behavior" in episode or any("behavior_source" in action for action in episode["actions"])):
                    raise ValueError("training episode counters/clock/behavior mismatch")
                for number, update in enumerate(episode["updates"], previous["gradient_updates"] + 1):
                    require_non_negative_int("gradient_updates", update["gradient_updates"])
                    require_non_negative_float("loss", update["loss"])
                    require_non_negative_float("gradient_norm_before_clip", update["gradient_norm_before_clip"])
                    if (update["gradient_updates"] != number
                            or update["target_synchronized"] is not (number % config.target_update_interval == 0)):
                        raise ValueError("training update trace mismatch")
                trace.append({"episode_id": episode["episode_id"], "transitions": count,
                    "training_slots_before": episode["training_slots_before"], "training_slots_after": meta["training_slots"],
                    "cumulative_transitions": seen, "gradient_updates": updates, "update_count": len(episode["updates"]),
                    "loss": describe_values(update["loss"] for update in episode["updates"]),
                    "gradient_norm_before_clip": describe_values(update["gradient_norm_before_clip"] for update in episode["updates"]),
                    "target_synchronizations": sum(update["target_synchronized"] for update in episode["updates"])})
            if previous and meta["initial_model_sha256"] != previous["initial_model_sha256"]:
                raise ValueError("initial model identity changed")
        previous = meta
    return trace


def run_training_input_audit(*, session_directory, output_path=None) -> dict:
    """Read immutable checkpoint training data; validation bytes are integrity-only."""
    session = Path(session_directory)
    destination = None if output_path is None else Path(output_path)
    if destination is not None:
        if destination.resolve().is_relative_to(session.resolve()):
            raise ValueError("audit output must be outside the source session")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("choose a new diagnostic report path")
    global_before = _global_state()
    contract = _read_json(session / "contract.json")
    entries = inspect_session(session, contract)
    config, encoder, kind = _validated_contract(contract)
    trace = _training_history(session, entries, contract, config)
    checkpoints = [entry for entry in entries if "agent.pt" in entry["artifacts"]]
    agents = []
    for entry in (checkpoints[0], checkpoints[-1]):
        path = session / "steps" / f'{entry["step_index"]:06d}' / "agent.pt"
        agent, saved = kind.load_checkpoint(path, encoder)
        expected = {"run_kind": SESSION_PURPOSE, "session_contract_sha256": _digest(contract),
                    "step_index": entry["step_index"], **entry["metadata"]}
        meta = entry["metadata"]
        if (saved != expected or agent.config != config or agent.seed != contract["agent_seed"]
                or agent.seen_transitions != meta["seen_transitions"] or agent.gradient_updates != meta["gradient_updates"]
                or len(agent.replay) != meta["replay_size"] or model_fingerprint(agent) != meta["model_sha256"]):
            raise ValueError("checkpoint identity/state differs from session metadata")
        agents.append(agent)
    frozen = [_agent_state(agent) for agent in agents]
    samples = tuple(agents[-1].replay)
    initial = audit_agent_inputs(agents[0], samples, encoder.feature_names)
    final = audit_agent_inputs(agents[-1], samples, encoder.feature_names)
    if not all(_same_state(before, _agent_state(agent)) for before, agent in zip(frozen, agents, strict=True)):
        raise RuntimeError("checkpoint analysis mutated learner state")
    if entries != inspect_session(session, contract) or not _same_state(global_before, _global_state()):
        raise RuntimeError("audit mutated source journal, global RNG or execution settings")
    report = {"report_schema_version": 1, "purpose": "training-input-audit-not-training",
        "model_contract": model_contract(), "return_scope": contract["return_scope"],
        "archived_session": {"source_sha256": contract["source_sha256"], "runtime_identity": contract["runtime_identity"],
            "contract_sha256": _digest(contract), "last_commit_sha256": entries[-1]["commit_sha256"],
            "initial_checkpoint_sha256": checkpoints[0]["artifacts"]["agent.pt"],
            "final_checkpoint_sha256": checkpoints[-1]["artifacts"]["agent.pt"]},
        "auditor": {"provenance": source_provenance(), "runtime_identity": runtime_identity()},
        "replay": {"retained_transitions": len(samples), "seen_transitions": agents[-1].seen_transitions,
            "complete_training_replay": len(samples) == agents[-1].seen_transitions,
            "order": "fixed replay storage order; may differ from chronology after ring rollover"},
        "initial_network": initial, "final_network": final, "training_update_trace": trace,
        "flags": {"training_executed": False, "test_split_accessed": False,
                  "validation_analytics_used": False, "feature_scales_changed": False},
        "mutation_checks": {"both_learner_states_unchanged": True, "source_journal_unchanged": True,
                            "global_rng_and_execution_settings_unchanged": True},
        "limits": ["Retrospective descriptive training cohort, not feature fitting or causal attribution.",
                   "Fixed-cohort recomputed loss is distinct from recorded sampled update losses.",
                   "No validation selection, performance comparison, convergence or exact cross-runtime resume claim."]}
    if not _same_state(global_before, _global_state()):
        raise RuntimeError("audit provenance collection mutated RNG or execution settings")
    if destination is not None:
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_training_input_audit(session_directory=args.session_dir, output_path=args.output)
    print(json.dumps({"output": str(args.output), "purpose": report["purpose"], "replay": report["replay"]}))


if __name__ == "__main__":
    main()
