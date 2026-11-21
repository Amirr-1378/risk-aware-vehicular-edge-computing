"""Single-model, episode-boundary development sessions; never main training.

Reuse the existing local actors, physics and learning runtime. Only persistence
and invocation boundaries differ. One writer per output directory is required.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents import ObservationEncoder
from risk_aware_v2x.agents.neural import ALGORITHMS, NeuralCostAgent, encoder_signature
from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_non_negative_int, require_positive_int
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol, _unique_object
from .learning_protocol import LearningProtocol
from .learning_runtime import deterministic_cpu
from .provenance import source_provenance
from .run_learning_audit import DEFAULT_LEARNING_PROTOCOL, _episode, model_fingerprint
from .run_return_scope_audit import _same_state, _training_state
from .run_scenario_audit import DEFAULT_PROTOCOL
from .runtime_preflight import collect_preflight
from .session_journal import append_step, create_session, inspect_session


PURPOSE = "resumable-single-model-development-not-main-training"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_report(output, entry):
    return json.loads((output / "steps" / f'{entry["step_index"]:06d}' / "report.json").read_text(
        encoding="utf-8"), object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))


def _progress(agent, completed, clock, initial):
    return {"completed_training_episodes": completed, "training_slots": clock,
        "seen_transitions": agent.seen_transitions, "gradient_updates": agent.gradient_updates,
        "replay_size": len(agent.replay), "model_sha256": model_fingerprint(agent),
        "initial_model_sha256": initial}


def _validate_progress(metadata):
    for key in ("completed_training_episodes", "training_slots", "seen_transitions",
                "gradient_updates", "replay_size"):
        require_non_negative_int(key, metadata[key])
    for key in ("model_sha256", "initial_model_sha256"):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")


def _check_episode(report, scene, key, *, clock, training):
    """Validate persisted scene identity, complete drain and physical slot clock."""
    if (report["seeds"] != asdict(key) or report["scenario_fingerprint"] != scene.fingerprint
            or report["episode_id"] != f"{scene.specification.label}:{key.split}:{key.index}"):
        raise ValueError("persisted scene/seed/order mismatch")
    elapsed = require_positive_int("elapsed_slots", report["elapsed_slots"])
    count = require_non_negative_int("transition_count", report["transition_count"])
    require_non_negative_int("environment_seed", report["environment_seed"])
    require_non_negative_int("training_slots_before", report["training_slots_before"])
    require_non_negative_int("training_slots_after", report["training_slots_after"])
    if (report["training"] is not training or report["environment_seed"] != key.channel_seed
            or count != len(scene.tasks) or report["metrics"]["completed_tasks"] != len(scene.tasks)
            or report["training_slots_before"] != clock
            or report["training_slots_after"] != clock + (elapsed if training else 0)
            or "training_behavior" in report
            or any("behavior_source" in action for action in report["actions"])):
        raise ValueError("persisted episode budget/clock/behavior mismatch")
    if not training and (report["updates"] or any(action["epsilon"] != 0. for action in report["actions"])):
        raise ValueError("persisted validation contains training/exploration")


def _validate_history(output, entries, learning, scenarios, train_keys, val_keys):
    """Semantic checks complement the journal's byte hashes, before any write."""
    previous = None
    for index, entry in enumerate(entries):
        meta = entry["metadata"]
        _validate_progress(meta)
        report = _read_report(output, entry)
        phase = "initialized" if index == 0 else "training" if index <= learning.training_episodes else "validation"
        if (meta["phase"] != phase or index > learning.training_episodes + 1
                or set(entry["artifacts"]) != ({"report.json"} if phase == "validation" else {"report.json", "agent.pt"})):
            raise ValueError("invalid session phase/artifact sequence")
        expected_completed = min(index, learning.training_episodes)
        if meta["completed_training_episodes"] != expected_completed:
            raise ValueError("completed episode count mismatch")
        if phase == "initialized":
            if (any(meta[name] != 0 for name in ("training_slots", "seen_transitions", "gradient_updates", "replay_size"))
                    or meta["model_sha256"] != meta["initial_model_sha256"]
                    or report.get("purpose") != PURPOSE):
                raise ValueError("invalid initial session state")
        else:
            if meta["initial_model_sha256"] != previous["initial_model_sha256"]:
                raise ValueError("initial model identity changed")
            if phase == "training":
                key = train_keys[index - 1]
                scene = generate_scenario(scenarios.profile(learning.training_profile), seed=key.scenario_seed)
                episode = report["episode"]
                _check_episode(episode, scene, key, clock=previous["training_slots"], training=True)
                seen = previous["seen_transitions"] + len(scene.tasks)
                updates = max(0, seen - max(learning.common_agent.batch_size, learning.common_agent.learning_starts) + 1)
                if (meta["training_slots"] != episode["training_slots_after"]
                        or meta["seen_transitions"] != seen or meta["gradient_updates"] != updates
                        or len(episode["updates"]) != updates - previous["gradient_updates"]
                        or meta["replay_size"] != min(seen, learning.common_agent.replay_capacity)
                        or report["scenario_manifest"] != json.loads(_canonical(scene.manifest()))):
                    raise ValueError("persisted training counters/manifest mismatch")
            else:
                if any(meta[name] != previous[name] for name in previous if name != "phase"):
                    raise ValueError("validation changed learner progress")
                validation = report["validation"]
                if set(validation) != set(learning.validation_profiles):
                    raise ValueError("incomplete validation profiles")
                for profile, episodes in validation.items():
                    if len(episodes) != len(val_keys):
                        raise ValueError("incomplete validation cases")
                    for key, episode in zip(val_keys, episodes, strict=True):
                        scene = generate_scenario(scenarios.profile(profile), seed=key.scenario_seed)
                        _check_episode(episode, scene, key, clock=meta["training_slots"], training=False)
        previous = meta


def run_resumable_development(*, output_directory: str | Path,
                              learning: LearningProtocol | None = None,
                              scenarios: DevelopmentProtocol | None = None,
                              algorithm: str = "qr_dqn_cvar", agent_seed: int = 11,
                              return_scope: str = "individual", resume: bool = False,
                              max_additional_episodes: int = 1) -> dict:
    """Advance a fixed plan, saving each complete episode and final validation.

    The per-invocation cap pauses the plan; it never extends its total budget.
    Interrupted unpublished steps fail closed. Recovery copies the verified
    prefix to a new directory; no simulator state is resumed within an episode.
    """
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    require_positive_int("max_additional_episodes", max_additional_episodes)
    require_non_negative_int("agent_seed", agent_seed)
    if return_scope not in ("individual", "team"):
        raise ValueError("return_scope must be individual or team")
    learning = learning if learning is not None else LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL)
    scenarios = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    learning.validate_against(scenarios)
    config = learning.agent_configuration(algorithm)
    if agent_seed not in learning.agent_seeds:
        raise ValueError("agent seed is outside the declared development plan")
    preflight = collect_preflight(require_learning=True)
    if not preflight["ok"]:
        raise ValueError(f'incompatible runtime: {preflight["missing"] + preflight["incompatible"]}')
    encoder = ObservationEncoder(max_server_vehicles=scenarios.profile(learning.training_profile).max_server_vehicle_slots,
                                 scales=learning.feature_scales)
    kind = TeamNeuralCostAgent if return_scope == "team" else NeuralCostAgent
    train_keys = [scenarios.episode_seeds("train", i) for i in range(learning.training_episodes)]
    val_keys = [scenarios.episode_seeds("validation", i) for i in range(learning.validation_cases)]
    contract = json.loads(_canonical({"session_schema_version": 1, "purpose": PURPOSE,
        "model_contract": model_contract(), "source_sha256": preflight["provenance"]["source_sha256"],
        "runtime_identity": preflight["runtime_identity"], "learning_protocol": asdict(learning),
        "learning_protocol_sha256": learning.fingerprint, "scenario_protocol": asdict(scenarios),
        "scenario_protocol_sha256": scenarios.fingerprint, "algorithm": algorithm, "agent_seed": agent_seed,
        "return_scope": return_scope, "learning_return_contract": kind.learning_return_contract(),
        "encoder": encoder_signature(encoder), "agent_configuration": asdict(config),
        "training_behavior": "standard-learned-epsilon", "updates_per_new_transition": 1,
        "warmup_counting": "total-transitions", "training_sequence": [asdict(k) for k in train_keys],
        "validation_sequence": [asdict(k) for k in val_keys], "selection_rule": learning.selection_rule,
        "main_training_started": False, "test_split_accessed": False, "centralized_controller": False}))
    identity = _digest(contract)
    output = Path(output_directory)
    entries = inspect_session(output, contract) if resume else []
    if not resume and output.exists():
        raise FileExistsError("choose a new session directory or use explicit resume")
    _validate_history(output, entries, learning, scenarios, train_keys, val_keys)
    with deterministic_cpu():
        if entries:
            checkpoint_entry = entries[-2] if entries[-1]["metadata"]["phase"] == "validation" else entries[-1]
            checkpoint = output / "steps" / f'{checkpoint_entry["step_index"]:06d}' / "agent.pt"
            agent, saved = kind.load_checkpoint(checkpoint, encoder)
            expected = {"run_kind": PURPOSE, "session_contract_sha256": identity,
                        "step_index": checkpoint_entry["step_index"], **checkpoint_entry["metadata"]}
            if saved != expected or agent.config != config or agent.seed != agent_seed:
                raise ValueError("checkpoint session identity/configuration mismatch")
            progress = checkpoint_entry["metadata"]
            initial = progress["initial_model_sha256"]
            completed, clock = progress["completed_training_episodes"], progress["training_slots"]
            if _progress(agent, completed, clock, initial) != {k: v for k, v in progress.items() if k != "phase"}:
                raise ValueError("checkpoint learner state/counter mismatch")
        else:
            agent = kind(config, encoder, seed=agent_seed)
            completed, clock, initial = 0, 0, model_fingerprint(agent)
            if not resume:
                create_session(output, contract)
            meta = {"phase": "initialized", **_progress(agent, completed, clock, initial)}
            entry = append_step(output, contract, meta,
                {"purpose": PURPOSE, "preflight": preflight, "provenance": source_provenance()},
                lambda path: agent.save_checkpoint(path, run_metadata={"run_kind": PURPOSE,
                    "session_contract_sha256": identity, "step_index": 0, **meta}))
            entries.append(entry)
        stop = min(learning.training_episodes, completed + max_additional_episodes)
        for index in range(completed, stop):
            key = train_keys[index]
            scene = generate_scenario(scenarios.profile(learning.training_profile), seed=key.scenario_seed)
            report, clock = _episode(scene, key, encoder, agent, learning, clock,
                                     training=True, return_scope=return_scope)
            _check_episode(report, scene, key, clock=entries[-1]["metadata"]["training_slots"], training=True)
            completed = index + 1
            meta = {"phase": "training", **_progress(agent, completed, clock, initial)}
            seen = entries[-1]["metadata"]["seen_transitions"] + len(scene.tasks)
            expected_updates = max(0, seen - max(config.batch_size, config.learning_starts) + 1)
            if agent.seen_transitions != seen or agent.gradient_updates != expected_updates:
                raise RuntimeError("training transition/update budget mismatch")
            step_index = len(entries)
            # Native reports contain integer vehicle-ID keys. Normalize once at
            # the JSON boundary; journal object keys are deliberately strings.
            persisted = json.loads(_canonical({"episode": report, "scenario_manifest": scene.manifest()}))
            entry = append_step(output, contract, meta, persisted,
                lambda path: agent.save_checkpoint(path, run_metadata={"run_kind": PURPOSE,
                    "session_contract_sha256": identity, "step_index": step_index, **meta}))
            entries.append(entry)
        if completed == learning.training_episodes and entries[-1]["metadata"]["phase"] != "validation":
            frozen = _training_state(agent)
            validation = {}
            for profile in learning.validation_profiles:
                validation[profile] = []
                for key in val_keys:
                    scene = generate_scenario(scenarios.profile(profile), seed=key.scenario_seed)
                    report, unchanged = _episode(scene, key, encoder, agent, learning, clock,
                                                 training=False, return_scope=return_scope)
                    _check_episode(report, scene, key, clock=clock, training=False)
                    if unchanged != clock:
                        raise RuntimeError("validation advanced training clock")
                    validation[profile].append(report)
            if not _same_state(frozen, _training_state(agent)):
                raise RuntimeError("validation mutated learner/replay/RNG/optimizer state")
            entries.append(append_step(output, contract,
                {"phase": "validation", **_progress(agent, completed, clock, initial)},
                json.loads(_canonical({"validation": validation}))))
        complete = entries[-1]["metadata"]["phase"] == "validation"
        checkpoint_index = len(entries) - (2 if complete else 1)
        return {"report_schema_version": 1, "purpose": PURPOSE, "status": "complete" if complete else "incomplete",
            "session_contract_sha256": identity, "source_sha256": contract["source_sha256"],
            "runtime_identity": contract["runtime_identity"], "model_contract": model_contract(),
            "algorithm": algorithm, "agent_seed": agent_seed, "return_scope": return_scope,
            "training_behavior": contract["training_behavior"], **_progress(agent, completed, clock, initial),
            "planned_training_episodes": learning.training_episodes,
            "planned_validation_episodes": len(val_keys) * len(learning.validation_profiles),
            "completed_validation_episodes": len(val_keys) * len(learning.validation_profiles) if complete else 0,
            "checkpoint": f"steps/{checkpoint_index:06d}/agent.pt", "last_commit_sha256": entries[-1]["commit_sha256"],
            "validation": _read_report(output, entries[-1])["validation"] if complete else None,
            "main_training_started": False, "test_split_accessed": False, "centralized_controller": False,
            "limits": "Single synthetic development model, fixed final checkpoint; no performance comparison or convergence claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-protocol", type=Path, default=DEFAULT_LEARNING_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="qr_dqn_cvar")
    parser.add_argument("--agent-seed", type=int, default=11)
    parser.add_argument("--return-scope", choices=("individual", "team"), default="individual")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-additional-episodes", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_resumable_development(output_directory=args.output_dir,
        learning=LearningProtocol.load(args.learning_protocol), scenarios=DevelopmentProtocol.load(args.scenario_protocol),
        algorithm=args.algorithm, agent_seed=args.agent_seed, return_scope=args.return_scope,
        resume=args.resume, max_additional_episodes=args.max_additional_episodes)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
