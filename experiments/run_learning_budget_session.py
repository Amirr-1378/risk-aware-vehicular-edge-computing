"""Chunked, resumable development learning curves; one fixed model per session.

Each committed training chunk ends at complete episodes and never crosses a
declared evaluation budget. Validation cannot select a checkpoint or alter the
learner. One writer per directory is required; Drive durability is not assured.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents.neural import NeuralCostAgent, encoder_signature
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import (
    require_non_negative_float, require_non_negative_int, require_positive_int,
)
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .learning_budget_protocol import (
    DEFAULT_BUDGET_SCENARIO_PROTOCOL, DEFAULT_LEARNING_BUDGET_PROTOCOL, LearningBudgetProtocol,
)
from .learning_runtime import deterministic_cpu
from .provenance import source_provenance
from .run_learning_audit import _episode, model_fingerprint
from .run_resumable_development import (
    _canonical, _check_episode, _digest, _progress, _read_report, _validate_progress,
)
from .run_return_scope_audit import _same_state, _training_state
from .runtime_preflight import collect_preflight, runtime_identity
from .session_journal import append_step, create_session, inspect_session


RUN_KIND = "resumable-learning-budget-session-v1"
PROGRESS_KEYS = ("completed_training_episodes", "training_slots", "seen_transitions",
                 "gradient_updates", "replay_size", "model_sha256", "initial_model_sha256")


def _progress_only(metadata):
    return {key: metadata[key] for key in PROGRESS_KEYS}


def _assert_identity_unchanged(contract):
    if (source_provenance()["source_sha256"] != contract["source_sha256"]
            or runtime_identity() != contract["runtime_identity"]):
        raise RuntimeError("source or runtime changed during the budget session")


def _check_updates(episode, *, seen_before, updates_before, config):
    seen = seen_before + episode["transition_count"]
    updates = max(0, seen - max(config.batch_size, config.learning_starts) + 1)
    if len(episode["updates"]) != updates - updates_before:
        raise ValueError("training episode update budget mismatch")
    for number, update in enumerate(episode["updates"], updates_before + 1):
        require_non_negative_int("gradient_updates", update["gradient_updates"])
        require_non_negative_float("loss", update["loss"])
        require_non_negative_float("gradient_norm_before_clip", update["gradient_norm_before_clip"])
        if (update["gradient_updates"] != number
                or update["target_synchronized"] is not (number % config.target_update_interval == 0)):
            raise ValueError("training update trace/counter mismatch")
    return seen, updates


def _validate_budget_history(output, entries, plan, scenarios, train_keys, val_keys, config):
    """Validate a state machine, not a step-index/episode-index coincidence."""
    validated, previous = [], None
    latest_validation = None
    for index, entry in enumerate(entries):
        meta, report = entry["metadata"], _read_report(output, entry)
        _validate_progress(meta)
        phase = meta["phase"]
        allowed = {"phase", *PROGRESS_KEYS} | ({"budget"} if phase == "validation" else set())
        if set(meta) != allowed:
            raise ValueError("unexpected budget-session metadata")
        artifacts = {"report.json"} if phase == "validation" else {"report.json", "agent.pt"}
        if set(entry["artifacts"]) != artifacts:
            raise ValueError("invalid budget-session artifact set")
        if index == 0:
            if (phase != "initialized" or report.get("purpose") != RUN_KIND
                    or any(meta[key] != 0 for key in PROGRESS_KEYS[:5])
                    or meta["model_sha256"] != meta["initial_model_sha256"]):
                raise ValueError("invalid initial budget-session state")
        elif phase == "training_chunk":
            completed = previous["completed_training_episodes"]
            if completed in plan.budgets and completed not in validated:
                raise ValueError("training crossed a budget without frozen validation")
            following_budget = plan.next_budget(completed)
            records = report.get("episodes")
            if (following_budget is None or not isinstance(records, list) or not records
                    or len(records) > plan.checkpoint_interval_episodes
                    or completed + len(records) > following_budget
                    or meta["completed_training_episodes"] != completed + len(records)):
                raise ValueError("invalid training chunk boundary or episode count")
            clock, seen, updates = (previous["training_slots"], previous["seen_transitions"],
                                    previous["gradient_updates"])
            for offset, item in enumerate(records):
                if not isinstance(item, dict) or set(item) != {"episode", "scenario_manifest"}:
                    raise ValueError("invalid training chunk episode record")
                key = train_keys[completed + offset]
                scene = generate_scenario(scenarios.profile(plan.learning.training_profile), seed=key.scenario_seed)
                episode = item["episode"]
                _check_episode(episode, scene, key, clock=clock, training=True)
                if (episode["metrics"]["arrived_tasks"] != len(scene.tasks)
                        or item["scenario_manifest"] != json.loads(_canonical(scene.manifest()))):
                    raise ValueError("training chunk scene/manifest mismatch")
                seen, updates = _check_updates(episode, seen_before=seen, updates_before=updates, config=config)
                clock = episode["training_slots_after"]
            if (meta["training_slots"] != clock or meta["seen_transitions"] != seen
                    or meta["gradient_updates"] != updates
                    or meta["replay_size"] != min(seen, config.replay_capacity)):
                raise ValueError("persisted training chunk learner counters mismatch")
        elif phase == "validation":
            budget = require_positive_int("validation budget", meta["budget"])
            if (budget not in plan.budgets or budget in validated
                    or budget != previous["completed_training_episodes"]
                    or list(plan.budgets[:len(validated)]) != validated
                    or budget != plan.budgets[len(validated)]
                    or report.get("budget") != budget
                    or _progress_only(meta) != _progress_only(previous)):
                raise ValueError("invalid validation budget/order or mutated learner progress")
            validation = report.get("validation")
            if not isinstance(validation, dict) or set(validation) != set(plan.learning.validation_profiles):
                raise ValueError("incomplete validation profiles")
            for profile, episodes in validation.items():
                if not isinstance(episodes, list) or len(episodes) != len(val_keys):
                    raise ValueError("incomplete validation cohort")
                for key, episode in zip(val_keys, episodes, strict=True):
                    scene = generate_scenario(scenarios.profile(profile), seed=key.scenario_seed)
                    _check_episode(episode, scene, key, clock=meta["training_slots"], training=False)
            validated.append(budget)
            latest_validation = report["validation"]
        else:
            raise ValueError("unexpected budget-session phase")
        if previous is not None and meta["initial_model_sha256"] != previous["initial_model_sha256"]:
            raise ValueError("initial model identity changed")
        previous = meta
    return validated, latest_validation


def run_learning_budget_session(*, output_directory,
                                 protocol: LearningBudgetProtocol | None = None,
                                 scenarios: DevelopmentProtocol | None = None,
                                 variant="log_snr_v4", algorithm="qr_dqn_cvar", agent_seed=11,
                                 resume=False, max_additional_episodes=10) -> dict:
    """Advance a single fixed chain; pause limits do not extend its total plan.

    Interrupted uncommitted chunks rerun the same seed prefix from the last
    complete checkpoint. Partial final journal writes need explicit recovery
    into a new directory via session_journal.copy_verified_prefix.
    """
    if not isinstance(resume, bool):
        raise TypeError("resume must be boolean")
    require_positive_int("max_additional_episodes", max_additional_episodes)
    require_non_negative_int("agent_seed", agent_seed)
    plan = protocol if protocol is not None else LearningBudgetProtocol.load(DEFAULT_LEARNING_BUDGET_PROTOCOL)
    scenarios = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_BUDGET_SCENARIO_PROTOCOL)
    plan.validate_against(scenarios)
    config = plan.learning.agent_configuration(algorithm)
    if agent_seed not in plan.learning.agent_seeds:
        raise ValueError("agent seed is outside the declared study")
    encoder = plan.encoder_for(variant, max_server_vehicles=scenarios.profile(plan.learning.training_profile).max_server_vehicle_slots)
    preflight = collect_preflight(require_learning=True)
    if not preflight["ok"]:
        raise ValueError(f'incompatible runtime: {preflight["missing"] + preflight["incompatible"]}')
    train_keys = [scenarios.episode_seeds("train", i) for i in range(plan.learning.training_episodes)]
    val_keys = [scenarios.episode_seeds("validation", i) for i in range(plan.learning.validation_cases)]
    contract = json.loads(_canonical({"session_schema_version": 1, "purpose": RUN_KIND,
        "model_contract": model_contract(), "source_sha256": preflight["provenance"]["source_sha256"],
        "runtime_identity": preflight["runtime_identity"], "learning_budget_protocol": asdict(plan),
        "learning_budget_protocol_sha256": plan.fingerprint, "scenario_protocol": asdict(scenarios),
        "scenario_protocol_sha256": scenarios.fingerprint, "algorithm": algorithm, "agent_seed": agent_seed,
        "representation_variant": variant, "representation_contract": plan.variant_contract(variant),
        "return_scope": "individual", "learning_return_contract": None, "encoder": encoder_signature(encoder),
        "agent_configuration": asdict(config), "training_behavior": "standard-learned-epsilon",
        "updates_per_new_transition": 1, "warmup_counting": "total-transitions",
        "training_sequence": [asdict(key) for key in train_keys],
        "validation_sequence": [asdict(key) for key in val_keys],
        "selection_rule": "fixed-final-budget-no-validation-selection",
        "main_training_started": False, "test_split_accessed": False, "centralized_controller": False}))
    identity, output = _digest(contract), Path(output_directory)
    if not resume and output.exists():
        raise FileExistsError("choose a new budget-session directory or explicitly resume")
    entries = inspect_session(output, contract) if resume else []
    validated, latest_validation = _validate_budget_history(output, entries, plan, scenarios, train_keys, val_keys, config)
    with deterministic_cpu():
        if entries:
            checkpoint_entry = next(entry for entry in reversed(entries) if "agent.pt" in entry["artifacts"])
            checkpoint = output / "steps" / f'{checkpoint_entry["step_index"]:06d}' / "agent.pt"
            agent, saved = NeuralCostAgent.load_checkpoint(checkpoint, encoder)
            expected = {"run_kind": RUN_KIND, "session_contract_sha256": identity,
                        "step_index": checkpoint_entry["step_index"], **checkpoint_entry["metadata"]}
            if saved != expected or agent.config != config or agent.seed != agent_seed:
                raise ValueError("checkpoint budget-session identity/configuration mismatch")
            progress = checkpoint_entry["metadata"]
            completed, clock, initial = (progress["completed_training_episodes"], progress["training_slots"],
                                          progress["initial_model_sha256"])
            if _progress(agent, completed, clock, initial) != _progress_only(progress):
                raise ValueError("checkpoint learner state/counter mismatch")
        else:
            agent = NeuralCostAgent(config, encoder, seed=agent_seed)
            completed, clock, initial = 0, 0, model_fingerprint(agent)
            _assert_identity_unchanged(contract)
            if not resume:
                create_session(output, contract)
            meta = {"phase": "initialized", **_progress(agent, completed, clock, initial)}
            entry = append_step(output, contract, meta,
                {"purpose": RUN_KIND, "preflight": preflight, "provenance": source_provenance()},
                lambda path: agent.save_checkpoint(path, run_metadata={"run_kind": RUN_KIND,
                    "session_contract_sha256": identity, "step_index": 0, **meta}))
            entries.append(entry)
        stop = min(plan.learning.training_episodes, completed + max_additional_episodes)
        while True:
            # Every due validation is completed before either pausing or training.
            if completed in plan.budgets and completed not in validated:
                frozen, validation = _training_state(agent), {}
                for profile in plan.learning.validation_profiles:
                    validation[profile] = []
                    for key in val_keys:
                        scene = generate_scenario(scenarios.profile(profile), seed=key.scenario_seed)
                        report, unchanged = _episode(scene, key, encoder, agent, plan.learning, clock,
                                                     training=False, return_scope="individual")
                        _check_episode(report, scene, key, clock=clock, training=False)
                        if unchanged != clock:
                            raise RuntimeError("validation advanced the training clock")
                        validation[profile].append(report)
                if not _same_state(frozen, _training_state(agent)):
                    raise RuntimeError("validation mutated learner, replay, RNG or optimizer")
                _assert_identity_unchanged(contract)
                meta = {"phase": "validation", "budget": completed, **_progress(agent, completed, clock, initial)}
                entries.append(append_step(output, contract, meta,
                    json.loads(_canonical({"budget": completed, "validation": validation}))))
                validated.append(completed)
                latest_validation = json.loads(_canonical(validation))
            if completed >= stop:
                break
            end = min(stop, completed + plan.checkpoint_interval_episodes, plan.next_budget(completed))
            records, seen, updates = [], agent.seen_transitions, agent.gradient_updates
            for index in range(completed, end):
                key = train_keys[index]
                scene = generate_scenario(scenarios.profile(plan.learning.training_profile), seed=key.scenario_seed)
                before = clock
                report, clock = _episode(scene, key, encoder, agent, plan.learning, clock,
                                         training=True, return_scope="individual")
                _check_episode(report, scene, key, clock=before, training=True)
                seen, updates = _check_updates(report, seen_before=seen, updates_before=updates, config=config)
                if agent.seen_transitions != seen or agent.gradient_updates != updates:
                    raise RuntimeError("training transition/update budget mismatch")
                records.append({"episode": report, "scenario_manifest": scene.manifest()})
            completed = end
            _assert_identity_unchanged(contract)
            meta = {"phase": "training_chunk", **_progress(agent, completed, clock, initial)}
            step_index = len(entries)
            entries.append(append_step(output, contract, meta,
                json.loads(_canonical({"episodes": records})),
                lambda path: agent.save_checkpoint(path, run_metadata={"run_kind": RUN_KIND,
                    "session_contract_sha256": identity, "step_index": step_index, **meta})))
        checkpoint_entry = next(entry for entry in reversed(entries) if "agent.pt" in entry["artifacts"])
        complete = completed == plan.learning.training_episodes and validated == list(plan.budgets)
        validation_per_budget = len(val_keys) * len(plan.learning.validation_profiles)
        return {"report_schema_version": 1, "purpose": RUN_KIND,
            "status": "complete" if complete else "incomplete", "session_contract_sha256": identity,
            "source_sha256": contract["source_sha256"], "runtime_identity": contract["runtime_identity"],
            "learning_budget_protocol_sha256": plan.fingerprint, "model_contract": model_contract(),
            "representation_variant": variant, "algorithm": algorithm, "agent_seed": agent_seed,
            "return_scope": "individual", "training_behavior": contract["training_behavior"],
            **_progress(agent, completed, clock, initial), "planned_training_episodes": plan.learning.training_episodes,
            "completed_budget_points": validated, "planned_budget_points": list(plan.budgets),
            "latest_validation_budget": validated[-1] if validated else None, "latest_validation": latest_validation,
            "completed_validation_episodes": len(validated) * validation_per_budget,
            "planned_validation_episodes": len(plan.budgets) * validation_per_budget,
            "checkpoint": f'steps/{checkpoint_entry["step_index"]:06d}/agent.pt',
            "last_commit_sha256": entries[-1]["commit_sha256"],
            "main_training_started": False, "test_split_accessed": False, "centralized_controller": False,
            "limits": "One fixed development chain; intermediate budgets are descriptive, never validation-selected stopping/model choices."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_LEARNING_BUDGET_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_BUDGET_SCENARIO_PROTOCOL)
    parser.add_argument("--plan", action="store_true", help="print the declared study only; create no output or learner")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=("linear_v3", "log_snr_v4"), default="log_snr_v4")
    parser.add_argument("--algorithm", choices=("qr_dqn", "qr_dqn_cvar"), default="qr_dqn_cvar")
    parser.add_argument("--agent-seed", type=int, default=11)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-additional-episodes", type=int, default=10)
    args = parser.parse_args(argv)
    protocol = LearningBudgetProtocol.load(args.protocol)
    scenarios = DevelopmentProtocol.load(args.scenario_protocol)
    protocol.validate_against(scenarios)
    if args.plan:
        print(json.dumps(protocol.execution_plan(), indent=2, allow_nan=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --plan is supplied")
    result = run_learning_budget_session(output_directory=args.output_dir, protocol=protocol, scenarios=scenarios,
        variant=args.variant, algorithm=args.algorithm, agent_seed=args.agent_seed, resume=args.resume,
        max_additional_episodes=args.max_additional_episodes)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
