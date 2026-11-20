"""Matched stochastic individual/team-return development ablation.

Independent local actors, fixed budgets/final checkpoints, full drain, and
validation only. This is not main training, convergence or superiority evidence.
"""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch

from risk_aware_v2x.agents import ObservationEncoder
from risk_aware_v2x.agents.neural import NeuralCostAgent, encoder_signature
from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .learning_runtime import deterministic_cpu, runtime_metadata
from .provenance import source_provenance
from .return_scope_protocol import DEFAULT_RETURN_SCOPE_PROTOCOL, ReturnScopeProtocol
from .return_scope_statistics import render_markdown, summarize_return_scope_runs
from .run_learning_audit import _episode, model_fingerprint
from .run_scenario_audit import DEFAULT_PROTOCOL, run_scenario_audit, save_audit_report


def _training_state(agent):
    """Snapshot all mutable learner state before frozen evaluation, without RNG use."""
    return {"online": copy.deepcopy(agent.online.state_dict()), "target": copy.deepcopy(agent.target.state_dict()),
        "optimizer": copy.deepcopy(agent.optimizer.state_dict()), "replay": agent.replay.state_dict(),
        "exploration": {uid: rng.getstate() for uid, rng in agent._exploration.items()},
        "seen_transitions": agent.seen_transitions, "gradient_updates": agent.gradient_updates,
        "online_training": agent.online.training, "target_training": agent.target.training}


def _same_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_state(x, y) for x, y in zip(a, b))
    return a == b


def _fit_model(*, learning, train_scenes, train_keys, val_scenes, val_keys, encoder,
               scope, algorithm, seed, output, initializations, experiment_metadata,
               checkpoint_prefix="checkpoints", behavior_factory=None):
    """One fixed-budget final-model run; behavior can change training collection only."""
    kind = TeamNeuralCostAgent if scope == "team" else NeuralCostAgent
    agent = kind(learning.agent_configuration(algorithm), encoder, seed=seed)
    initial = model_fingerprint(agent)
    pair = algorithm, seed
    if pair in initializations and initial != initializations[pair]:
        raise RuntimeError("matched experiments have different initial weights")
    initializations[pair] = initial
    training, clock = [], 0
    for index, (key, scene) in enumerate(zip(train_keys, train_scenes, strict=True)):
        policy = None if behavior_factory is None else behavior_factory(index, scene, key)
        episode, clock = _episode(scene, key, encoder, agent, learning, clock,
                                  training=True, return_scope=scope, behavior_policy=policy)
        training.append(episode)
    offered = sum(len(scene.tasks) for scene in train_scenes)
    updates = max(0, offered - max(learning.common_agent.batch_size, learning.common_agent.learning_starts) + 1)
    if agent.seen_transitions != offered or agent.gradient_updates != updates:
        raise RuntimeError("matched task/transition/update budget violated; never drop a failed task")
    final = model_fingerprint(agent)
    checkpoint = f"{checkpoint_prefix}/{scope}/{algorithm}-seed-{seed}-final.pt"
    metadata = {**experiment_metadata, "model_contract": model_contract(),
        "return_scope": scope, "learning_return_contract": agent.learning_return_contract(),
        "learning_protocol_sha256": learning.fingerprint, "selection_rule": learning.selection_rule,
        "completed_training_episodes": learning.training_episodes, "training_slots": clock,
        "model_sha256": final, "initial_model_sha256": initial,
        "training_scenarios": [scene.fingerprint for scene in train_scenes]}
    if "training_behavior_contract" in metadata:
        count = metadata["training_behavior_contract"]["teacher_episodes"]
        metadata["teacher_completed_at_training_slot"] = training[count - 1]["training_slots_after"] if count else 0
        metadata["teacher_action_count"] = sum(e.get("training_behavior", {}).get("actions", 0) for e in training)
        metadata["behavior_phase_at_checkpoint"] = "learned_epsilon"
    if output is not None:
        agent.save_checkpoint(output / checkpoint, run_metadata=metadata)
    frozen = _training_state(agent)
    validation = {}
    for profile, scenes in val_scenes.items():
        validation[profile] = []
        for key, scene in zip(val_keys, scenes, strict=True):
            # There is intentionally no behavior policy on ANY evaluation call.
            episode, unchanged = _episode(scene, key, encoder, agent, learning, clock,
                                          training=False, return_scope=scope)
            if unchanged != clock:
                raise RuntimeError("validation advanced the training clock")
            validation[profile].append(episode)
    if not _same_state(frozen, _training_state(agent)):
        raise RuntimeError("validation mutated model, target, optimizer, replay, RNG or counters")
    return {"return_scope": scope, "learning_return_contract": agent.learning_return_contract(),
        "algorithm": algorithm, "agent_seed": seed, "agent_configuration": asdict(agent.config),
        "initial_model_sha256": initial, "model_sha256": final, "training_slots": clock,
        "seen_transitions": agent.seen_transitions, "gradient_updates": agent.gradient_updates,
        "target_synchronizations": sum(u["target_synchronized"] for e in training for u in e["updates"]),
        "epsilon_at_training_end": learning.epsilon_schedule.value(clock),
        "replay_size": len(agent.replay), "training": training, "validation": validation,
        "checkpoint": checkpoint if output is not None else None}


def run_return_scope_audit(*, protocol: ReturnScopeProtocol | None = None,
                           scenarios: DevelopmentProtocol | None = None,
                           output_directory: str | Path | None = None) -> dict:
    plan = protocol if protocol is not None else ReturnScopeProtocol.load(DEFAULT_RETURN_SCOPE_PROTOCOL)
    scenes_plan = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    plan.validate_against(scenes_plan)
    learning = plan.learning
    # Resolve every selected key before creating output or constructing scenes.
    train_keys = [scenes_plan.episode_seeds("train", i) for i in range(learning.training_episodes)]
    val_keys = [scenes_plan.episode_seeds("validation", i) for i in range(learning.validation_cases)]
    output = None if output_directory is None else Path(output_directory)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    train_scenes = [generate_scenario(scenes_plan.profile(learning.training_profile), seed=key.scenario_seed) for key in train_keys]
    val_scenes = {profile: [generate_scenario(scenes_plan.profile(profile), seed=key.scenario_seed) for key in val_keys]
                  for profile in learning.validation_profiles}
    scene_records = [{"split": "train", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
                     for key, scene in zip(train_keys, train_scenes, strict=True)]
    scene_records.extend({"split": "validation", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
                         for scenes in val_scenes.values() for key, scene in zip(val_keys, scenes, strict=True))
    encoder = ObservationEncoder(max_server_vehicles=scenes_plan.profile(learning.training_profile).max_server_vehicle_slots,
                                 scales=learning.feature_scales)
    offered_train_tasks = sum(len(scene.tasks) for scene in train_scenes)
    warmup = max(learning.common_agent.batch_size, learning.common_agent.learning_starts)
    expected_updates = max(0, offered_train_tasks - warmup + 1)
    initializations, runs = {}, []
    with deterministic_cpu():
        for scope in plan.return_scopes:
            for algorithm in learning.algorithms:
                for seed in learning.agent_seeds:
                    runs.append(_fit_model(learning=learning, train_scenes=train_scenes, train_keys=train_keys,
                        val_scenes=val_scenes, val_keys=val_keys, encoder=encoder, scope=scope, algorithm=algorithm,
                        seed=seed, output=output, initializations=initializations, experiment_metadata={
                        "run_kind": "stochastic-return-scope-audit-v1", "return_scope_protocol_sha256": plan.fingerprint,
                        "scenario_protocol_sha256": scenes_plan.fingerprint, "source_sha256": provenance["source_sha256"],
                    }))
        runtime = runtime_metadata()
    analytical = {profile: run_scenario_audit(protocol=scenes_plan, profile=profile, split="validation",
                                             cases=learning.validation_cases) for profile in learning.validation_profiles}
    for profile, audit in analytical.items():
        if [item["fingerprint"] for item in audit["scenarios"]] != [scene.fingerprint for scene in val_scenes[profile]]:
            raise RuntimeError("fixed policies used different validation scenes")
    summary = summarize_return_scope_runs(runs, algorithms=learning.algorithms, agent_seeds=learning.agent_seeds,
        profiles=learning.validation_profiles, validation_cases=learning.validation_cases, alpha=scenes_plan.cvar_alpha)
    executions = len(runs) * (learning.training_episodes + len(learning.validation_profiles) * learning.validation_cases)
    executions += sum(len(values) for audit in analytical.values() for values in audit["policies"].values())
    report = {"report_schema_version": 1, "run_kind": "stochastic-return-scope-audit-v1",
        "model_contract": model_contract(), "return_scope_protocol": asdict(plan), "return_scope_protocol_sha256": plan.fingerprint,
        "learning_protocol": asdict(learning), "learning_protocol_sha256": learning.fingerprint,
        "scenario_protocol": asdict(scenes_plan), "scenario_protocol_sha256": scenes_plan.fingerprint,
        "provenance": provenance, "runtime": runtime, "encoder": encoder_signature(encoder),
        "test_split_accessed": False, "main_training_started": False, "centralized_controller": False,
        "selection_rule": learning.selection_rule, "training_budget": {"offered_tasks_per_model": offered_train_tasks,
            "expected_updates_per_model": expected_updates, "unit": "same independent arrival traces, once each; physical slots may differ"},
        "physical_episode_executions": executions, "scenarios": scene_records, "runs": runs,
        "analytical_validation": analytical, "return_scope_summary": summary,
        "warning": "synthetic development ablation; no held-out test, convergence or superiority claim; actor views are correlated"}
    if output is not None:
        save_audit_report(report, output / "report.json")
        with (output / "summary.md").open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(render_markdown(summary))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_RETURN_SCOPE_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_return_scope_audit(protocol=ReturnScopeProtocol.load(args.protocol),
        scenarios=DevelopmentProtocol.load(args.scenario_protocol), output_directory=args.output_dir)
    print(json.dumps({"output_directory": str(args.output_dir), "models": len(report["runs"]),
        "physical_episode_executions": report["physical_episode_executions"], "main_training_started": False}))
