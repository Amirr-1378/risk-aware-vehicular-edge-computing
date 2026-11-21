"""Matched linear/log-SNR development comparison; no main or test training."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents.neural import NeuralCostAgent, encoder_signature
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_non_negative_int
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .learning_runtime import deterministic_cpu, runtime_metadata
from .provenance import source_provenance
from .run_learning_audit import model_fingerprint
from .run_return_scope_audit import _fit_model
from .run_scenario_audit import DEFAULT_PROTOCOL, run_scenario_audit, save_audit_report
from .run_training_input_audit import audit_agent_inputs
from .snr_representation_protocol import DEFAULT_SNR_REPRESENTATION_PROTOCOL, SNRRepresentationProtocol
from .snr_representation_statistics import render_markdown, summarize_snr_representation_runs


RUN_KIND = "stochastic-snr-representation-audit-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_snr_representation_checkpoint(path, encoder, *, protocol, variant, algorithm,
                                       seed, source_sha256=None):
    """Inspect a trusted final model with explicit variant identity; no sweep resume."""
    if variant not in protocol.variants or algorithm not in protocol.learning.algorithms:
        raise ValueError("checkpoint variant/algorithm outside the declared protocol")
    require_non_negative_int("agent seed", seed)
    if seed not in protocol.learning.agent_seeds:
        raise ValueError("checkpoint seed outside the declared protocol")
    expected_encoder = protocol.encoder_for(variant, max_server_vehicles=encoder.max_server_vehicles)
    if encoder_signature(encoder) != encoder_signature(expected_encoder):
        raise ValueError("encoder does not match the representation variant")
    agent, metadata = NeuralCostAgent.load_checkpoint(path, encoder)
    expected = {"run_kind": RUN_KIND, "snr_representation_protocol_sha256": protocol.fingerprint,
        "representation_variant": variant, "representation_contract": protocol.variant_contract(variant),
        "encoder_schema_version": encoder.schema_version, "return_scope": "individual",
        "learning_return_contract": None, "model_contract": model_contract(),
        "learning_protocol_sha256": protocol.learning.fingerprint,
        "scenario_protocol_sha256": protocol.learning.scenario_protocol_sha256,
        "selection_rule": protocol.learning.selection_rule,
        "completed_training_episodes": protocol.learning.training_episodes}
    if (any(metadata.get(key) != value for key, value in expected.items())
            or type(metadata.get("completed_training_episodes")) is not int
            or agent.config != protocol.learning.agent_configuration(algorithm) or agent.seed != seed):
        raise ValueError("checkpoint representation/protocol/configuration identity mismatch")
    require_non_negative_int("training slots", metadata.get("training_slots"))
    recorded_source = metadata.get("source_sha256")
    if (not isinstance(recorded_source, str) or len(recorded_source) != 64
            or any(c not in "0123456789abcdef" for c in recorded_source)
            or source_sha256 is not None and recorded_source != source_sha256):
        raise ValueError("checkpoint source identity mismatch")
    expected_updates = max(0, agent.seen_transitions - max(agent.config.batch_size, agent.config.learning_starts) + 1)
    if (metadata.get("model_sha256") != model_fingerprint(agent)
            or agent.gradient_updates != expected_updates):
        raise ValueError("checkpoint actual model/update counters mismatch")
    return agent, metadata


def run_snr_representation_audit(*, output_directory,
                                  protocol: SNRRepresentationProtocol | None = None,
                                  scenarios: DevelopmentProtocol | None = None) -> dict:
    """Train matched final models under fixed, predeclared development controls.

    Reports/checkpoints are mandatory and exclusive. Each actor receives only
    its local encoded snapshot. No old checkpoint is reinterpreted or continued.
    """
    plan = protocol if protocol is not None else SNRRepresentationProtocol.load(DEFAULT_SNR_REPRESENTATION_PROTOCOL)
    scenes_plan = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    plan.validate_against(scenes_plan)
    learning = plan.learning
    train_keys = [scenes_plan.episode_seeds("train", i) for i in range(learning.training_episodes)]
    val_keys = [scenes_plan.episode_seeds("validation", i) for i in range(learning.validation_cases)]
    max_servers = scenes_plan.profile(learning.training_profile).max_server_vehicle_slots
    encoders = {variant: plan.encoder_for(variant, max_server_vehicles=max_servers) for variant in plan.variants}
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance()
    train_scenes = [generate_scenario(scenes_plan.profile(learning.training_profile), seed=key.scenario_seed) for key in train_keys]
    val_scenes = {profile: [generate_scenario(scenes_plan.profile(profile), seed=key.scenario_seed) for key in val_keys]
                  for profile in learning.validation_profiles}
    records = [{"split": "train", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
               for key, scene in zip(train_keys, train_scenes, strict=True)]
    records.extend({"split": "validation", "seeds": asdict(key), "fingerprint": scene.fingerprint, "manifest": scene.manifest()}
                   for scenes in val_scenes.values() for key, scene in zip(val_keys, scenes, strict=True))
    initializations, runs = {}, []
    with deterministic_cpu():
        for variant in plan.variants:
            encoder, representation = encoders[variant], plan.variant_contract(variant)
            for algorithm in learning.algorithms:
                for seed in learning.agent_seeds:
                    run = _fit_model(learning=learning, train_scenes=train_scenes, train_keys=train_keys,
                        val_scenes=val_scenes, val_keys=val_keys, encoder=encoder, scope="individual",
                        algorithm=algorithm, seed=seed, output=output, initializations=initializations,
                        checkpoint_prefix=f"checkpoints/{variant}", experiment_metadata={
                            "run_kind": RUN_KIND, "snr_representation_protocol_sha256": plan.fingerprint,
                            "representation_variant": variant, "representation_contract": representation,
                            "encoder_schema_version": encoder.schema_version,
                            "scenario_protocol_sha256": scenes_plan.fingerprint,
                            "source_sha256": provenance["source_sha256"]})
                    final, metadata = load_snr_representation_checkpoint(output / run["checkpoint"], encoder,
                        protocol=plan, variant=variant, algorithm=algorithm, seed=seed,
                        source_sha256=provenance["source_sha256"])
                    if metadata["training_slots"] != run["training_slots"]:
                        raise RuntimeError("loaded checkpoint training clock mismatch")
                    initial = NeuralCostAgent(learning.agent_configuration(algorithm), encoder, seed=seed)
                    if model_fingerprint(initial) != run["initial_model_sha256"]:
                        raise RuntimeError("diagnostic initial model differs from matched initialization")
                    samples = tuple(final.replay)
                    run.update(representation_variant=variant, representation_contract=representation,
                        encoder=encoder_signature(encoder), input_diagnostics={
                            "cohort": "same final retained TRAINING replay for initial/final networks within this model; policy-dependent across variants",
                            "seen_transitions": final.seen_transitions, "retained_transitions": len(samples),
                            "complete_training_replay": len(samples) == final.seen_transitions,
                            "initial_network": audit_agent_inputs(initial, samples, encoder.feature_names),
                            "final_network": audit_agent_inputs(final, samples, encoder.feature_names)})
                    runs.append(run)
        runtime = runtime_metadata()
    analytical = {profile: run_scenario_audit(protocol=scenes_plan, profile=profile, split="validation",
                                            cases=learning.validation_cases) for profile in learning.validation_profiles}
    for profile, audit in analytical.items():
        if [item["fingerprint"] for item in audit["scenarios"]] != [scene.fingerprint for scene in val_scenes[profile]]:
            raise RuntimeError("fixed baselines used different validation scenes")
    summary = summarize_snr_representation_runs(runs, variants=plan.variants, algorithms=learning.algorithms,
        agent_seeds=learning.agent_seeds, profiles=learning.validation_profiles,
        validation_cases=learning.validation_cases, alpha=scenes_plan.cvar_alpha)
    offered = sum(len(scene.tasks) for scene in train_scenes)
    executions = len(runs) * (learning.training_episodes + len(learning.validation_profiles) * learning.validation_cases)
    executions += sum(len(episodes) for audit in analytical.values() for episodes in audit["policies"].values())
    if source_provenance()["source_sha256"] != provenance["source_sha256"]:
        raise RuntimeError("implementation changed during the representation comparison")
    report = {"report_schema_version": 1, "run_kind": RUN_KIND,
        "purpose": "paired numerical representation development study; not main training or a frozen paper protocol",
        "model_contract": model_contract(), "snr_representation_protocol": asdict(plan),
        "snr_representation_protocol_sha256": plan.fingerprint, "learning_protocol_sha256": learning.fingerprint,
        "scenario_protocol": asdict(scenes_plan), "scenario_protocol_sha256": scenes_plan.fingerprint,
        "provenance": provenance, "runtime": runtime,
        "encoders": {variant: encoder_signature(encoder) for variant, encoder in encoders.items()},
        "training_budget": {"offered_tasks_per_model": offered,
            "expected_updates_per_model": max(0, offered - max(learning.common_agent.batch_size, learning.common_agent.learning_starts) + 1),
            "teacher_episodes": 0, "extra_optimizer_updates": 0,
            "unit": "matched exogenous traces and update counts; physical slots and realized exploration can differ"},
        "scenarios": records, "runs": runs, "analytical_validation": analytical,
        "representation_summary": summary, "physical_episode_executions": executions,
        "test_split_accessed": False, "main_training_started": False, "centralized_controller": False,
        "selection_rule": learning.selection_rule,
        "warning": "synthetic thin-sample study; no automatic variant adoption, convergence, pooled CVaR or universal superiority claim"}
    save_audit_report(report, output / "report.json")
    with (output / "summary.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_markdown(summary))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_SNR_REPRESENTATION_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_snr_representation_audit(protocol=SNRRepresentationProtocol.load(args.protocol),
        scenarios=DevelopmentProtocol.load(args.scenario_protocol), output_directory=args.output_dir)
    print(json.dumps({"output_directory": str(args.output_dir), "models": len(report["runs"]),
        "physical_episode_executions": report["physical_episode_executions"], "main_training_started": False}))
