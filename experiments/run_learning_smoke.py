"""Finite CPU training/checkpoint audit; not calibrated learning-performance evidence."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent, encoder_signature
from risk_aware_v2x.agents.schedules import EpsilonSchedule
from risk_aware_v2x.entities._validation import require_non_negative_int
from .run_environment_smoke import _json_value, build_smoke_scenario
from .provenance import source_provenance
from .learning_runtime import deterministic_cpu, run_episode, runtime_metadata


def run_learning_smoke(*, algorithm: str = "qr_dqn_cvar", episodes: int = 4,
                       environment_seed: int = 7, agent_seed: int = 17,
                       checkpoint_path: str | Path | None = None,
                       resume_path: str | Path | None = None) -> dict:
    """Train additional episodes, optionally resume a complete episode boundary."""
    episodes = require_non_negative_int("episodes", episodes)
    require_non_negative_int("environment_seed", environment_seed)
    require_non_negative_int("agent_seed", agent_seed)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        raise FileExistsError("checkpoint destination already exists; choose a new filename")
    environment, scenario = build_smoke_scenario()
    encoder = ObservationEncoder(max_server_vehicles=environment.config.max_server_vehicles,
                                 scales=FeatureScales())
    config = AgentConfig(algorithm=algorithm, hidden_sizes=(32, 32), num_quantiles=15,
                         batch_size=4, learning_starts=4, replay_capacity=256,
                         target_update_interval=5)
    schedule = EpsilonSchedule(decay_slots=200)
    configuration = asdict(environment.config)
    provenance = source_provenance()
    scenario_hash = hashlib.sha256(json.dumps(scenario, sort_keys=True, default=_json_value,
                                              allow_nan=False).encode("utf-8")).hexdigest()
    evaluation_seeds = (1_000_007, 1_000_008)
    contract = {"run_kind": "synthetic-learning-smoke-v1", "configuration": configuration,
                "scenario_sha256": scenario_hash, "environment_seed_base": environment_seed,
                "epsilon_schedule": asdict(schedule), "updates_per_new_transition": 1,
                "warmup_counting": "total-transitions",
                "cpu_threads": 1, "deterministic_algorithms": True,
                "evaluation_seeds": list(evaluation_seeds),
                "source_sha256": provenance["source_sha256"]}
    with deterministic_cpu():
        if resume_path is None:
            agent = NeuralCostAgent(config, encoder, seed=agent_seed)
            completed, training_slots = 0, 0
        else:
            agent, metadata = NeuralCostAgent.load_checkpoint(resume_path, encoder)
            if agent.config != config or agent.seed != agent_seed:
                raise ValueError("resume algorithm/configuration/agent seed mismatch")
            if any(metadata.get(key) != value for key, value in contract.items()):
                raise ValueError("resume scenario or training-protocol mismatch")
            completed = require_non_negative_int("completed_training_episodes", metadata["completed_training_episodes"])
            training_slots = require_non_negative_int("training_slots", metadata["training_slots"])
        start_episode = completed
        if any(environment_seed <= seed < environment_seed + completed + episodes
               for seed in evaluation_seeds):
            raise ValueError("training and evaluation seeds must not overlap")
        adapter = DecisionAdapter(environment, encoder, gamma_per_slot=config.gamma_per_slot)
        training_reports = []
        for episode in range(completed, completed + episodes):
            report, training_slots = run_episode(
                adapter, agent, environment_seed=environment_seed + episode, training=True,
                schedule=schedule, training_slots=training_slots,
            )
            report["episode_index"] = episode
            training_reports.append(report)
        completed += episodes
        if checkpoint_path is not None:
            agent.save_checkpoint(checkpoint_path, run_metadata={
                **contract, "completed_training_episodes": completed, "training_slots": training_slots,
            })
        evaluations = []
        # These are disjoint seed namespaces; the tiny deterministic physical
        # smoke scenario is NOT a held-out stochastic generalization benchmark.
        for seed in evaluation_seeds:
            report, unchanged = run_episode(adapter, agent, environment_seed=seed,
                                            training=False, schedule=schedule,
                                            training_slots=training_slots)
            if unchanged != training_slots:
                raise RuntimeError("evaluation mutated the exploration schedule")
            evaluations.append(report)
        return {
            "report_schema_version": 1, "purpose": "synthetic neural training audit; not a paper benchmark",
            "algorithm": algorithm, "agent_seed": agent_seed, "environment_seed_base": environment_seed,
            "agent_configuration": asdict(config), "encoder": encoder_signature(encoder),
            "configuration": configuration, "scenario": scenario, "scenario_sha256": scenario_hash,
            "provenance": provenance,
            **runtime_metadata(),
            "deterministic_algorithms": True, "epsilon_schedule": asdict(schedule),
            "start_training_episode": start_episode, "completed_training_episodes": completed,
            "training_slots": training_slots, "seen_transitions": agent.seen_transitions,
            "gradient_updates": agent.gradient_updates, "replay_size": len(agent.replay),
            "training": training_reports, "evaluation": evaluations,
            "checkpoint": None if checkpoint_path is None else str(Path(checkpoint_path).resolve()),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="qr_dqn_cvar")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--agent-seed", type=int, default=17)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_learning_smoke(algorithm=args.algorithm, episodes=args.episodes,
                                       environment_seed=args.environment_seed, agent_seed=args.agent_seed,
                                       checkpoint_path=args.checkpoint, resume_path=args.resume),
                     indent=2, default=_json_value, allow_nan=False))
