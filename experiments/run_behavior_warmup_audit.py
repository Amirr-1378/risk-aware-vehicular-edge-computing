"""Opt-in causal greedy behavior warmup; unchanged QR/CVaR learning and inference.

Teacher episodes REPLACE early normal episodes inside one fixed budget. No
expert loss/labels, altered reward, extra data/updates, or teacher evaluation.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents import ObservationEncoder
from risk_aware_v2x.agents.neural import NeuralCostAgent
from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
from risk_aware_v2x.baselines import GreedyCostParameters
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_non_negative_int
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .behavior_warmup_protocol import DEFAULT_BEHAVIOR_WARMUP_PROTOCOL, BehaviorWarmupProtocol
from .behavior_warmup_statistics import render_markdown, summarize_behavior_variants
from .comparison_policies import comparison_policies
from .learning_runtime import deterministic_cpu, runtime_metadata
from .provenance import source_provenance
from .run_return_scope_audit import _fit_model
from .run_scenario_audit import DEFAULT_PROTOCOL, run_scenario_audit, save_audit_report


def _teacher(scene, key):
    policies, _ = comparison_policies(scene.configuration, scene.users, scene.servers, seed=key.policy_seed)
    return policies["minimum_estimated_combined_cost"]


def _behavior_contract(variant, count, parameters):
    return {"variant": variant, "teacher": "cached_combined_cost_greedy" if variant == "greedy_warmup" else None,
        "teacher_episodes": count if variant == "greedy_warmup" else 0,
        "teacher_parameters": parameters if variant == "greedy_warmup" else None,
        "phase_rule": "first declared training episodes only; no evaluation teacher",
        "epsilon_clock": "all physical training slots including teacher/drain, no reset",
        "learner_exploration_during_teacher": "not consumed; later RNG draw alignment differs from standard"}


def load_behavior_checkpoint(path, encoder, *, protocol, variant, scope, algorithm, seed):
    """Verify final-model experiment identity; this is not a sweep-resume driver."""
    if variant not in protocol.variants or scope not in protocol.paired.return_scopes:
        raise ValueError("checkpoint variant/scope is outside the declared protocol")
    kind = TeamNeuralCostAgent if scope == "team" else NeuralCostAgent
    agent, meta = kind.load_checkpoint(path, encoder)
    count = protocol.teacher_episodes if variant == "greedy_warmup" else 0
    contract = meta.get("training_behavior_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("checkpoint training behavior contract must be explicit")
    if (meta.get("run_kind") != "stochastic-behavior-warmup-audit-v1"
            or meta.get("behavior_warmup_protocol_sha256") != protocol.fingerprint
            or meta.get("return_scope_protocol_sha256") != protocol.paired.fingerprint
            or meta.get("learning_protocol_sha256") != protocol.paired.learning.fingerprint
            or meta.get("scenario_protocol_sha256") != protocol.paired.learning.scenario_protocol_sha256
            or meta.get("behavior_variant") != variant or meta.get("return_scope") != scope
            or type(meta.get("completed_training_episodes")) is not int
            or meta.get("completed_training_episodes") != protocol.paired.learning.training_episodes
            or meta.get("behavior_phase_at_checkpoint") != "learned_epsilon"
            or contract.get("variant") != variant or type(contract.get("teacher_episodes")) is not int
            or contract.get("teacher_episodes") != count
            or contract.get("teacher") != ("cached_combined_cost_greedy" if count else None)
            or agent.config != protocol.paired.learning.agent_configuration(algorithm) or agent.seed != seed):
        raise ValueError("checkpoint behavior variant/protocol/phase/configuration mismatch")
    boundary = require_non_negative_int("teacher boundary slot", meta.get("teacher_completed_at_training_slot"))
    clock = require_non_negative_int("training slots", meta.get("training_slots"))
    actions = require_non_negative_int("teacher action count", meta.get("teacher_action_count"))
    if boundary > clock or (count > 0 and boundary == 0) or (count == 0 and (boundary != 0 or actions != 0)):
        raise ValueError("checkpoint teacher boundary/counter mismatch")
    if count:
        GreedyCostParameters(**contract["teacher_parameters"])
    elif contract.get("teacher_parameters") is not None:
        raise ValueError("standard checkpoint must not declare teacher parameters")
    if contract != _behavior_contract(variant, protocol.teacher_episodes, contract.get("teacher_parameters")):
        raise ValueError("checkpoint full training behavior contract mismatch")
    return agent, meta


def run_behavior_warmup_audit(*, protocol: BehaviorWarmupProtocol | None = None,
                              scenarios: DevelopmentProtocol | None = None,
                              output_directory: str | Path | None = None) -> dict:
    plan = protocol if protocol is not None else BehaviorWarmupProtocol.load(DEFAULT_BEHAVIOR_WARMUP_PROTOCOL)
    scenes_plan = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    plan.validate_against(scenes_plan)
    paired, learning = plan.paired, plan.paired.learning
    train_keys = [scenes_plan.episode_seeds("train", i) for i in range(learning.training_episodes)]
    val_keys = [scenes_plan.episode_seeds("validation", i) for i in range(learning.validation_cases)]
    output = None if output_directory is None else Path(output_directory)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    train_scenes = [generate_scenario(scenes_plan.profile(learning.training_profile), seed=key.scenario_seed) for key in train_keys]
    val_scenes = {profile: [generate_scenario(scenes_plan.profile(profile), seed=key.scenario_seed) for key in val_keys]
                  for profile in learning.validation_profiles}
    records = [{"split": "train", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
               for key, scene in zip(train_keys, train_scenes, strict=True)]
    records.extend({"split": "validation", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
                   for scenes in val_scenes.values() for key, scene in zip(val_keys, scenes, strict=True))
    parameters = asdict(_teacher(train_scenes[0], train_keys[0]).parameters)

    def teacher_factory(index, scene, key):
        if index >= plan.teacher_episodes:
            return None
        policy = _teacher(scene, key)
        if asdict(policy.parameters) != parameters:
            raise ValueError("teacher public static parameters changed between training scenes")
        return policy

    encoder = ObservationEncoder(max_server_vehicles=scenes_plan.profile(learning.training_profile).max_server_vehicle_slots,
                                 scales=learning.feature_scales)
    initializations, runs = {}, []
    with deterministic_cpu():
        for variant in plan.variants:
            behavior_contract = _behavior_contract(variant, plan.teacher_episodes, parameters)
            for scope in paired.return_scopes:
                for algorithm in learning.algorithms:
                    for seed in learning.agent_seeds:
                        run = _fit_model(learning=learning, train_scenes=train_scenes, train_keys=train_keys,
                            val_scenes=val_scenes, val_keys=val_keys, encoder=encoder, scope=scope, algorithm=algorithm,
                            seed=seed, output=output, initializations=initializations,
                            checkpoint_prefix=f"checkpoints/{variant}",
                            behavior_factory=teacher_factory if variant == "greedy_warmup" else None,
                            experiment_metadata={"run_kind": "stochastic-behavior-warmup-audit-v1", "behavior_variant": variant,
                                "behavior_warmup_protocol_sha256": plan.fingerprint, "return_scope_protocol_sha256": paired.fingerprint,
                                "scenario_protocol_sha256": scenes_plan.fingerprint, "source_sha256": provenance["source_sha256"],
                                "training_behavior_contract": behavior_contract})
                        for index, episode in enumerate(run["training"]):
                            expected = variant == "greedy_warmup" and index < plan.teacher_episodes
                            if ("training_behavior" in episode) != expected:
                                raise RuntimeError("teacher behavior crossed its declared training phase")
                        run.update(behavior_variant=variant, training_behavior_contract=behavior_contract,
                            teacher_action_count=sum(e.get("training_behavior", {}).get("actions", 0) for e in run["training"]),
                            teacher_completed_at_training_slot=(run["training"][plan.teacher_episodes - 1]["training_slots_after"]
                                                               if variant == "greedy_warmup" else 0))
                        runs.append(run)
        runtime = runtime_metadata()
    analytical = {profile: run_scenario_audit(protocol=scenes_plan, profile=profile, split="validation",
                                             cases=learning.validation_cases) for profile in learning.validation_profiles}
    for profile, audit in analytical.items():
        if [x["fingerprint"] for x in audit["scenarios"]] != [x.fingerprint for x in val_scenes[profile]]:
            raise RuntimeError("fixed baselines used different validation scenes")
    summary = summarize_behavior_variants(runs, variants=plan.variants, return_scopes=paired.return_scopes,
        algorithms=learning.algorithms, agent_seeds=learning.agent_seeds, profiles=learning.validation_profiles,
        validation_cases=learning.validation_cases, alpha=scenes_plan.cvar_alpha)
    offered = sum(len(scene.tasks) for scene in train_scenes)
    executions = len(runs) * (learning.training_episodes + len(learning.validation_profiles) * learning.validation_cases)
    executions += sum(len(values) for a in analytical.values() for values in a["policies"].values())
    report = {"report_schema_version": 1, "run_kind": "stochastic-behavior-warmup-audit-v1",
        "purpose": "optional training-behavior ablation, not replacement of the QR-DQN+CVaR method",
        "model_contract": model_contract(), "behavior_warmup_protocol": asdict(plan), "behavior_warmup_protocol_sha256": plan.fingerprint,
        "return_scope_protocol_sha256": paired.fingerprint, "learning_protocol_sha256": learning.fingerprint,
        "scenario_protocol": asdict(scenes_plan), "scenario_protocol_sha256": scenes_plan.fingerprint,
        "provenance": provenance, "runtime": runtime, "scenarios": records, "runs": runs,
        "training_budget": {"offered_tasks_per_model": offered, "expected_updates_per_model": max(0, offered - max(
            learning.common_agent.batch_size, learning.common_agent.learning_starts) + 1),
            "teacher_episodes_replace_normal_episodes": True, "extra_teacher_episodes": 0, "extra_optimizer_updates": 0},
        "physical_episode_executions": executions, "analytical_validation": analytical, "behavior_summary": summary,
        "test_split_accessed": False, "main_training_started": False, "centralized_controller": False,
        "deployment_selector": "learned masked argmin J only; teacher forbidden at evaluation",
        "warning": "synthetic development; changed training collection is explicitly optional; no superiority/convergence guarantee"}
    if output is not None:
        save_audit_report(report, output / "report.json")
        with (output / "summary.md").open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(render_markdown(summary))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_BEHAVIOR_WARMUP_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_behavior_warmup_audit(protocol=BehaviorWarmupProtocol.load(args.protocol),
        scenarios=DevelopmentProtocol.load(args.scenario_protocol), output_directory=args.output_dir)
    print(json.dumps({"output_directory": str(args.output_dir), "models": len(report["runs"]),
                      "physical_episode_executions": report["physical_episode_executions"], "main_training_started": False}))
