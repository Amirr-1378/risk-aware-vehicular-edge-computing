"""Multi-seed synthetic neural audit. Train/validation only; no test access."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import stdev

from risk_aware_v2x.agents import DecisionAdapter, ObservationEncoder
from risk_aware_v2x.agents.neural import NeuralCostAgent, encoder_signature
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.models.randomness import RandomStreams
from risk_aware_v2x.risk.statistics import episode_cost_summary, paired_mean_interval
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .learning_protocol import LearningProtocol
from .learning_runtime import deterministic_cpu, run_episode, runtime_metadata
from .provenance import source_provenance
from .run_scenario_audit import DEFAULT_PROTOCOL, run_scenario_audit, save_audit_report

DEFAULT_LEARNING_PROTOCOL = Path(__file__).resolve().parent / "configs" / "learning_pilot_v1.json"


def model_fingerprint(agent: NeuralCostAgent) -> str:
    """Hash actual online weights, not checkpoint serialization/path metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(agent.online.state_dict().items()):
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _episode(scene, seeds, encoder, agent, learning, training_slots, *, training):
    pristine = scene.fingerprint
    adapter = DecisionAdapter(scene.environment(), encoder, gamma_per_slot=agent.config.gamma_per_slot)
    report, clock = run_episode(adapter, agent, environment_seed=seeds.channel_seed,
                                training=training, schedule=learning.epsilon_schedule,
                                training_slots=training_slots)
    metrics = terminal_metrics(adapter.environment)
    if (scene.fingerprint != pristine
            or not math.isclose(math.fsum(report["accounting"]["total_cost"].values()),
                                metrics["total_cost"], rel_tol=1e-12, abs_tol=1e-12)):
        raise RuntimeError("scenario or cost-conservation invariant failed")
    return {**report, "metrics": metrics, "seeds": asdict(seeds),
            "episode_id": f"{scene.specification.label}:{seeds.split}:{seeds.index}",
            "scenario_fingerprint": pristine}, clock


def summarize_runs(runs: list[dict], analytical: dict, learning: LearningProtocol,
                   scenarios: DevelopmentProtocol) -> dict:
    """Seed variation conditional on the fixed train/validation traces.

    Each seed contributes ONE mean over validation episodes. Do not treat the
    seed x episode grid as IID, or pool per-policy tails into a mixture CVaR.
    """
    grid = [(run["algorithm"], run["agent_seed"]) for run in runs]
    expected_grid = {(algorithm, seed) for algorithm in learning.algorithms for seed in learning.agent_seeds}
    if len(grid) != len(expected_grid) or set(grid) != expected_grid:
        raise ValueError("incomplete or duplicated algorithm/seed grid")
    result = {}
    for profile in learning.validation_profiles:
        means = {algorithm: {str(run["agent_seed"]): run["validation_cost_summaries"][profile]["mean_episode_total_cost"]
                             for run in runs if run["algorithm"] == algorithm}
                 for algorithm in learning.algorithms}
        expected = {str(seed) for seed in learning.agent_seeds}
        if any(set(values) != expected for values in means.values()):
            raise ValueError("incomplete algorithm/seed grid; failed runs must not be omitted")
        reference = analytical[profile]["episode_total_cost_summaries"]["always_mec"]["mean_episode_total_cost"]
        comparisons = {}
        if len(expected) >= 2:
            pairs = [(algorithm, "always_mec") for algorithm in learning.algorithms]
            if {"qr_dqn", "qr_dqn_cvar"} <= set(means):
                pairs.append(("qr_dqn_cvar", "qr_dqn"))
            for a, b in pairs:
                other = {seed: reference for seed in expected} if b == "always_mec" else means[b]
                comparisons[f"{a}_minus_{b}"] = asdict(paired_mean_interval(
                    means[a], other,
                    rng=RandomStreams(scenarios.master_seed).stream(f"learning-seed-bootstrap:{profile}:{a}:{b}"),
                    resamples=scenarios.bootstrap_resamples, confidence=scenarios.bootstrap_confidence,
                ))
        result[profile] = {
            "per_algorithm": {name: {"seed_mean_episode_costs": values,
                "mean_of_seed_means": math.fsum(values.values()) / len(values),
                "sample_std_of_seed_means": stdev(values.values()) if len(values) >= 2 else None}
                for name, values in means.items()},
            "paired_seed_mean_intervals": comparisons,
            "interval_unit": "one independently trained agent seed, paired across algorithms",
            "interval_scope": "conditional on fixed training and validation traces; NOT scenario-population uncertainty",
            "warning": "few-seed descriptive pilot; no reliable tail/generalization/superiority claim; no pooled CVaR",
        }
    return result


def run_learning_audit(*, learning: LearningProtocol | None = None,
                       scenarios: DevelopmentProtocol | None = None,
                       output_directory: str | Path | None = None) -> dict:
    learning = learning if learning is not None else LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL)
    scenarios = scenarios if scenarios is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    learning.validate_against(scenarios)
    # Resolve ALL keys before scene construction/output. This runner has no
    # test option, no validation-driven stopping, and no best-seed filtering.
    train_keys = [scenarios.episode_seeds("train", index) for index in range(learning.training_episodes)]
    val_keys = [scenarios.episode_seeds("validation", index) for index in range(learning.validation_cases)]
    output = None if output_directory is None else Path(output_directory)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    training_scenes = [generate_scenario(scenarios.profile(learning.training_profile), seed=key.scenario_seed)
                       for key in train_keys]
    validation_scenes = {profile: [generate_scenario(scenarios.profile(profile), seed=key.scenario_seed)
                                  for key in val_keys] for profile in learning.validation_profiles}
    scene_records = [{"split": "train", "seeds": asdict(key), "fingerprint": scene.fingerprint,
                      "manifest": scene.manifest()} for key, scene in zip(train_keys, training_scenes, strict=True)]
    scene_records.extend({"split": "validation", "seeds": asdict(key), "fingerprint": scene.fingerprint,
                          "manifest": scene.manifest()}
                         for scenes in validation_scenes.values()
                         for key, scene in zip(val_keys, scenes, strict=True))
    encoder = ObservationEncoder(max_server_vehicles=scenarios.profile(learning.training_profile).max_server_vehicle_slots,
                                 scales=learning.feature_scales)
    provenance = source_provenance()
    runs = []
    with deterministic_cpu():
        for algorithm in learning.algorithms:
            for seed in learning.agent_seeds:
                agent = NeuralCostAgent(learning.agent_configuration(algorithm), encoder, seed=seed)
                training_reports, training_slots = [], 0
                for key, scene in zip(train_keys, training_scenes, strict=True):
                    episode, training_slots = _episode(scene, key, encoder, agent, learning,
                                                       training_slots, training=True)
                    training_reports.append(episode)
                final_fingerprint = model_fingerprint(agent)
                checkpoint = f"checkpoints/{algorithm}-seed-{seed}-final.pt"
                if output is not None:
                    agent.save_checkpoint(output / checkpoint, run_metadata={
                        "run_kind": "stochastic-learning-audit-v1",
                        "learning_protocol_sha256": learning.fingerprint,
                        "scenario_protocol_sha256": scenarios.fingerprint,
                        "source_sha256": provenance["source_sha256"],
                        "selection_rule": learning.selection_rule,
                        "completed_training_episodes": learning.training_episodes,
                        "training_slots": training_slots, "model_sha256": final_fingerprint,
                        "training_scenarios": [scene.fingerprint for scene in training_scenes],
                    })
                before = (agent.seen_transitions, agent.gradient_updates, len(agent.replay))
                validation, summaries = {}, {}
                for profile, scenes in validation_scenes.items():
                    validation[profile] = []
                    for key, scene in zip(val_keys, scenes, strict=True):
                        episode, clock = _episode(scene, key, encoder, agent, learning,
                                                  training_slots, training=False)
                        if clock != training_slots:
                            raise RuntimeError("validation changed the exploration schedule")
                        validation[profile].append(episode)
                    summaries[profile] = episode_cost_summary(
                        [episode["metrics"]["total_cost"] for episode in validation[profile]],
                        alpha=scenarios.cvar_alpha)
                if (before != (agent.seen_transitions, agent.gradient_updates, len(agent.replay))
                        or model_fingerprint(agent) != final_fingerprint):
                    raise RuntimeError("validation mutated the trained model or replay counters")
                runs.append({"algorithm": algorithm, "agent_seed": seed,
                    "agent_configuration": asdict(agent.config), "model_sha256": final_fingerprint,
                    "training_slots": training_slots, "seen_transitions": agent.seen_transitions,
                    "gradient_updates": agent.gradient_updates, "replay_size": len(agent.replay),
                    "training": training_reports, "validation": validation,
                    "validation_cost_summaries": summaries,
                    "checkpoint": checkpoint if output is not None else None})
        runtime = runtime_metadata()
    analytical = {profile: run_scenario_audit(protocol=scenarios, profile=profile, split="validation",
                                             cases=learning.validation_cases)
                  for profile in learning.validation_profiles}
    for profile, audit in analytical.items():
        if [item["fingerprint"] for item in audit["scenarios"]] != [scene.fingerprint for scene in validation_scenes[profile]]:
            raise RuntimeError("analytical and learned policies used different validation scenes")
    report = {
        "report_schema_version": 1, "purpose": "synthetic multi-seed learning audit; not a paper benchmark",
        "learning_protocol": asdict(learning), "learning_protocol_sha256": learning.fingerprint,
        "scenario_protocol": asdict(scenarios), "scenario_protocol_sha256": scenarios.fingerprint,
        "provenance": provenance, "runtime": runtime, "encoder": encoder_signature(encoder),
        "test_split_accessed": False, "selection_rule": learning.selection_rule,
        "training_budget_unit": "fixed arrival traces/episodes; drain slots and updates may differ by policy",
        "updates_per_new_eligible_transition": 1,
        "scenarios": scene_records, "runs": runs, "analytical_validation": analytical,
        "conditional_training_seed_summaries": summarize_runs(runs, analytical, learning, scenarios),
        "warning": "uncalibrated few-run pilot; no performance claim; validation is development data, not held-out test evidence",
    }
    if output is not None:
        save_audit_report(report, output / "report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-protocol", type=Path, default=DEFAULT_LEARNING_PROTOCOL)
    parser.add_argument("--scenario-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_learning_audit(learning=LearningProtocol.load(args.learning_protocol),
        scenarios=DevelopmentProtocol.load(args.scenario_protocol), output_directory=args.output_dir),
        indent=2, allow_nan=False))
