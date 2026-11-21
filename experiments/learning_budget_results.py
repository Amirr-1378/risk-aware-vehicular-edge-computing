"""Fixed references and complete-grid learning-budget summaries; never learning."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.agents.neural import encoder_signature
from risk_aware_v2x.entities._validation import require_non_negative_int, require_positive_int
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .learning_budget_protocol import (
    DEFAULT_BUDGET_SCENARIO_PROTOCOL, DEFAULT_LEARNING_BUDGET_PROTOCOL,
    FIXED_POLICY_NAMES, LearningBudgetProtocol,
)
from .learning_runtime import deterministic_cpu
from .provenance import source_provenance
from .run_learning_budget_session import RUN_KIND, PROGRESS_KEYS, _check_updates
from .run_resumable_development import _canonical, _validate_progress
from .runtime_preflight import runtime_identity
from .session_journal import inspect_session
from .snr_representation_statistics import summarize_snr_representation_runs
from .summarize_learning_audit import summarize_episodes
from .return_scope_statistics import _episodes
from .run_scenario_audit import run_scenario_audit, save_audit_report


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate result JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"non-finite result JSON: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=reject)


def _plans(protocol, scenarios):
    plan = protocol if protocol is not None else LearningBudgetProtocol.load(DEFAULT_LEARNING_BUDGET_PROTOCOL)
    scenes = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_BUDGET_SCENARIO_PROTOCOL)
    plan.validate_against(scenes)
    return plan, scenes


def run_learning_budget_references(*, output_directory, protocol=None, scenarios=None):
    """Evaluate the five unchanged fixed policies once on the declared cohort."""
    plan, scenes = _plans(protocol, scenarios)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    identity = runtime_identity()
    with deterministic_cpu():
        references = {profile: run_scenario_audit(protocol=scenes, profile=profile, split="validation",
            cases=plan.learning.validation_cases) for profile in plan.learning.validation_profiles}
    if (source_provenance()["source_sha256"] != provenance["source_sha256"]
            or runtime_identity() != identity):
        raise RuntimeError("source or runtime changed while evaluating fixed references")
    if any(set(value["policies"]) != set(FIXED_POLICY_NAMES) for value in references.values()):
        raise RuntimeError("fixed policy suite differs from the declared study")
    report = {"report_schema_version": 1, "purpose": "learning-budget-fixed-references-not-training",
        "model_contract": model_contract(), "learning_budget_protocol_sha256": plan.fingerprint,
        "scenario_protocol_sha256": scenes.fingerprint, "source_sha256": provenance["source_sha256"],
        "provenance": provenance, "runtime_identity": identity, "references": references,
        "physical_episode_executions": len(FIXED_POLICY_NAMES) * len(plan.learning.validation_profiles) * plan.learning.validation_cases,
        "training_executed": False, "test_split_accessed": False, "main_training_started": False}
    save_audit_report(report, output / "report.json")
    return report


def _validation_reports(session, contract, plan, scenes):
    """Verify immutable records; summarize no checkpoint until its whole cohort exists."""
    entries = inspect_session(session, contract)
    if not entries:
        raise ValueError("a session must contain committed initialization")
    completed, validated, previous, reports = 0, [], None, {}
    config = plan.learning.agent_configuration(contract["algorithm"])
    for index, entry in enumerate(entries):
        meta = entry["metadata"]
        _validate_progress(meta)
        phase = meta["phase"]
        expected_keys = {"phase", *PROGRESS_KEYS} | ({"budget"} if phase == "validation" else set())
        if set(meta) != expected_keys:
            raise ValueError("unexpected session progress fields")
        record = _read(session / "steps" / f'{entry["step_index"]:06d}' / "report.json")
        if index == 0:
            if (phase != "initialized" or any(meta[key] != 0 for key in PROGRESS_KEYS[:5])
                    or meta["model_sha256"] != meta["initial_model_sha256"]
                    or record.get("purpose") != RUN_KIND
                    or set(entry["artifacts"]) != {"report.json", "agent.pt"}):
                raise ValueError("invalid initialization")
        elif phase == "training_chunk":
            if completed in plan.budgets and completed not in validated:
                raise ValueError("missing required validation before further training")
            limit = plan.next_budget(completed)
            new_completed = meta["completed_training_episodes"]
            if (limit is None or not completed < new_completed <= min(limit, completed + plan.checkpoint_interval_episodes)
                    or set(entry["artifacts"]) != {"report.json", "agent.pt"}):
                raise ValueError("invalid training chunk boundary")
            items = record.get("episodes")
            if not isinstance(items, list) or len(items) != new_completed - completed:
                raise ValueError("training chunk episode count mismatch")
            clock, seen, updates = (previous["training_slots"], previous["seen_transitions"], previous["gradient_updates"])
            for number, item in enumerate(items, completed):
                if not isinstance(item, dict) or set(item) != {"episode", "scenario_manifest"}:
                    raise ValueError("invalid training chunk record")
                episode = item["episode"]
                _episode_identity(episode, scenes, plan.learning.training_profile, "train", number, clock, True)
                seen, updates = _check_updates(episode, seen_before=seen, updates_before=updates, config=config)
                clock = episode["training_slots_after"]
            if (meta["training_slots"] != clock or meta["seen_transitions"] != seen
                    or meta["gradient_updates"] != updates or meta["replay_size"] != min(seen, config.replay_capacity)):
                raise ValueError("training chunk counters differ from its episodes")
            completed = new_completed
        elif phase == "validation":
            if (meta.get("budget") != completed or completed not in plan.budgets
                    or completed in validated or plan.budgets[len(validated)] != completed
                    or any(meta[key] != previous[key] for key in PROGRESS_KEYS)
                    or set(entry["artifacts"]) != {"report.json"}):
                raise ValueError("validation changed progress or budget order")
            validation = record["validation"]
            if record["budget"] != completed or set(validation) != set(plan.learning.validation_profiles):
                raise ValueError("incomplete validation profiles")
            for profile, episodes in validation.items():
                _episodes(episodes, profile=profile, count=plan.learning.validation_cases)
                for number, episode in enumerate(episodes):
                    _episode_identity(episode, scenes, profile, "validation", number, meta["training_slots"], False)
            reports[completed] = validation
            validated.append(completed)
        else:
            raise ValueError("unexpected session phase")
        if meta["completed_training_episodes"] != completed:
            raise ValueError("session episode count mismatch")
        if previous is not None and meta["initial_model_sha256"] != previous["initial_model_sha256"]:
            raise ValueError("initial model identity changed")
        previous = meta
    return entries, validated, reports


def _episode_identity(episode, scenes, profile, split, number, clock, training):
    """Check recorded seeds/clocks, without reconstructing trajectories or tensors."""
    key = scenes.episode_seeds(split, number)
    elapsed = require_positive_int("elapsed_slots", episode["elapsed_slots"])
    count = require_non_negative_int("transition_count", episode["transition_count"])
    if (episode["seeds"] != asdict(key) or episode["episode_id"] != f"{profile}:{split}:{number}"
            or episode["environment_seed"] != key.channel_seed or episode["training"] is not training
            or episode["training_slots_before"] != clock
            or episode["training_slots_after"] != clock + (elapsed if training else 0)
            or episode["metrics"]["arrived_tasks"] != count or episode["metrics"]["completed_tasks"] != count
            or "training_behavior" in episode
            or any("behavior_source" in action for action in episode["actions"])
            or (not training and (episode["updates"] or any(action["epsilon"] != 0. for action in episode["actions"])))):
        raise ValueError("recorded episode seed/clock/training boundary mismatch")


def summarize_learning_budget(*, study_directory, protocol=None, scenarios=None, output_path=None):
    """Collect ALL declared chains; incomplete grids produce progress only.

    No network, environment episode, checkpoint deserialization, source change
    or validation-driven model/budget selection is performed.
    """
    plan, scenes = _plans(protocol, scenarios)
    study = Path(study_directory)
    destination = None if output_path is None else Path(output_path)
    paths = {(variant, algorithm, seed): study / variant / algorithm / f"seed-{seed}"
        for variant in plan.variants for algorithm in plan.learning.algorithms for seed in plan.learning.agent_seeds}
    if destination is not None:
        if destination.exists():
            raise FileExistsError("choose a new summary path")
        if any(destination.resolve().is_relative_to(path.resolve()) for path in [*paths.values(), study / "references"]):
            raise ValueError("summary output must not modify source sessions/references")
    progress, collected, source, runtime = [], {}, None, None
    for key, session in paths.items():
        identity = dict(zip(("representation_variant", "algorithm", "agent_seed"), key, strict=True))
        if not session.exists():
            progress.append({**identity, "status": "missing", "completed_budget_points": []})
            continue
        contract = _read(session / "contract.json")
        expected = {"purpose": RUN_KIND, "learning_budget_protocol_sha256": plan.fingerprint,
            "scenario_protocol_sha256": scenes.fingerprint, "model_contract": model_contract(),
            "learning_budget_protocol": json.loads(_canonical(asdict(plan))),
            "scenario_protocol": json.loads(_canonical(asdict(scenes))), "session_schema_version": 1,
            "encoder": json.loads(_canonical(encoder_signature(plan.encoder_for(key[0], max_server_vehicles=scenes.profile(plan.learning.training_profile).max_server_vehicle_slots)))),
            "agent_configuration": json.loads(_canonical(asdict(plan.learning.agent_configuration(key[1])))),
            "training_sequence": [asdict(scenes.episode_seeds("train", i)) for i in range(plan.learning.training_episodes)],
            "validation_sequence": [asdict(scenes.episode_seeds("validation", i)) for i in range(plan.learning.validation_cases)],
            "representation_contract": plan.variant_contract(key[0]), "return_scope": "individual",
            "learning_return_contract": None, "updates_per_new_transition": 1, "warmup_counting": "total-transitions",
            "training_behavior": "standard-learned-epsilon", "selection_rule": plan.selection_rule,
            "main_training_started": False, "test_split_accessed": False, "centralized_controller": False, **identity}
        if any(contract.get(name) != value for name, value in expected.items()):
            raise ValueError("session study/variant/algorithm identity mismatch")
        if source is None:
            source, runtime = contract["source_sha256"], contract["runtime_identity"]
        if contract["source_sha256"] != source or contract["runtime_identity"] != runtime:
            raise ValueError("all study chains must share exact source/runtime identity")
        entries, validated, values = _validation_reports(session, contract, plan, scenes)
        complete = validated == list(plan.budgets)
        progress.append({**identity, "status": "complete" if complete else "incomplete",
            "completed_training_episodes": entries[-1]["metadata"]["completed_training_episodes"],
            "completed_budget_points": validated, "last_commit_sha256": entries[-1]["commit_sha256"]})
        collected[key] = values
    grid_complete = all(row["status"] == "complete" for row in progress)
    report = {"report_schema_version": 1, "purpose": "learning-budget-complete-grid-summary-not-training",
        "status": "incomplete", "learning_budget_protocol_sha256": plan.fingerprint,
        "model_contract": model_contract(), "scenario_protocol_sha256": scenes.fingerprint,
        "selection_rule": plan.selection_rule, "progress": progress, "model_curves": None,
        "representation_comparisons": None, "fixed_references": None,
        "producer_source_sha256": source, "producer_runtime_identity": runtime,
        "auditor_provenance": source_provenance(), "main_training_started": False, "test_split_accessed": False,
        "limits": ["One continuing model per chain; checkpoint rows are correlated.",
                   "Risks remain separate per model/budget/profile; no seed/policy tail pooling.",
                   "Journal hashes and recorded identities are verified; physics and checkpoint tensors are not replayed by this collector.",
                   "No best-checkpoint selection, early stopping, significance or convergence claim."]}
    if grid_complete:
        curves, comparisons, paired = [], {}, {}
        for budget in plan.budgets:
            runs = []
            for key, values in collected.items():
                validation = values[budget]
                runs.append({"representation_variant": key[0], "algorithm": key[1], "agent_seed": key[2],
                             "return_scope": "individual", "validation": validation})
                for profile, episodes in validation.items():
                    scene_keys = [(e["scenario_fingerprint"], e["seeds"]) for e in episodes]
                    if profile not in paired:
                        paired[profile] = scene_keys
                    if paired[profile] != scene_keys:
                        raise ValueError("validation cohort changed across budgets/models")
                    curves.append({"representation_variant": key[0], "algorithm": key[1], "agent_seed": key[2],
                        "budget": budget, "profile": profile, "metrics": summarize_episodes(episodes, alpha=scenes.cvar_alpha)})
            comparisons[str(budget)] = summarize_snr_representation_runs(runs, variants=plan.variants,
                algorithms=plan.learning.algorithms, agent_seeds=plan.learning.agent_seeds,
                profiles=plan.learning.validation_profiles, validation_cases=plan.learning.validation_cases, alpha=scenes.cvar_alpha)
        report.update(status="awaiting_references", model_curves=curves, representation_comparisons=comparisons)
        reference_path = study / "references" / "report.json"
        if reference_path.exists():
            reference = _read(reference_path)
            if (reference["purpose"] != "learning-budget-fixed-references-not-training"
                    or reference["report_schema_version"] != 1 or reference["model_contract"] != model_contract()
                    or reference["main_training_started"] is not False
                    or reference["physical_episode_executions"] != len(FIXED_POLICY_NAMES) * len(plan.learning.validation_profiles) * plan.learning.validation_cases
                    or reference["learning_budget_protocol_sha256"] != plan.fingerprint
                    or reference["scenario_protocol_sha256"] != scenes.fingerprint
                    or reference["source_sha256"] != source or reference["runtime_identity"] != runtime
                    or reference["training_executed"] is not False or reference["test_split_accessed"] is not False
                    or set(reference["references"]) != set(plan.learning.validation_profiles)):
                raise ValueError("fixed references differ from study identity")
            references = {}
            for profile, audit in reference["references"].items():
                if (audit["split"] != "validation" or audit["cases"] != plan.learning.validation_cases
                        or audit["profile"] != profile or audit["model_contract"] != model_contract()
                        or audit["protocol_sha256"] != scenes.fingerprint
                        or audit["provenance"]["source_sha256"] != source
                        or set(audit["policies"]) != set(FIXED_POLICY_NAMES)
                        or [(s["fingerprint"], s["seeds"]) for s in audit["scenarios"]] != paired[profile]):
                    raise ValueError("fixed references used different validation scenes")
                for episodes in audit["policies"].values():
                    if (not isinstance(episodes, list) or len(episodes) != plan.learning.validation_cases
                            or [(episode["episode_id"], episode["scenario_fingerprint"]) for episode in episodes]
                            != [(scene["episode_id"], scene["fingerprint"]) for scene in audit["scenarios"]]):
                        raise ValueError("fixed policy episode/cohort mismatch")
                references[profile] = {name: summarize_episodes(episodes, alpha=scenes.cvar_alpha)
                                       for name, episodes in audit["policies"].items()}
            report.update(status="complete", fixed_references=references)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_LEARNING_BUDGET_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_BUDGET_SCENARIO_PROTOCOL)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--references", action="store_true", help="explicitly run only fixed-policy validation references")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan, scenes = _plans(LearningBudgetProtocol.load(args.protocol), DevelopmentProtocol.load(args.scenario_protocol))
    if args.references:
        result = run_learning_budget_references(output_directory=args.study_dir / "references", protocol=plan, scenarios=scenes)
    else:
        result = summarize_learning_budget(study_directory=args.study_dir, protocol=plan, scenarios=scenes, output_path=args.output)
    print(json.dumps({"purpose": result["purpose"], "status": result.get("status"), "output": str(args.output)}, allow_nan=False))


if __name__ == "__main__":
    main()
