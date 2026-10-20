"""Finite CPU training/checkpoint audit; not calibrated learning-performance evidence."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from importlib.metadata import distributions
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy
import torch

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from risk_aware_v2x.agents.neural import ALGORITHMS, AgentConfig, NeuralCostAgent, encoder_signature
from risk_aware_v2x.agents.schedules import EpsilonSchedule
from risk_aware_v2x.entities._validation import require_non_negative_int
from .run_environment_smoke import _json_value, build_smoke_scenario


def source_provenance() -> dict:
    """Fingerprint actual Python sources, including uncommitted implementation."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted([*(root / "src").rglob("*.py"), *(root / "experiments").glob("*.py")])
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    commit, dirty = None, None
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=5, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5, check=False)
        if revision.returncode == 0:
            commit = revision.stdout.strip()
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass  # Source hash still works when Git is unavailable in an export.
    return {"source_sha256": digest.hexdigest(), "git_commit": commit, "git_dirty": dirty}


@contextmanager
def deterministic_cpu():
    """Local experiment settings; restore process flags/threads on exit."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def run_episode(adapter, agent, *, environment_seed: int, training: bool,
                schedule: EpsilonSchedule, training_slots: int) -> tuple[dict, int]:
    """All users act before a service/update batch. One update per new transition.

    Evaluation never calls observe/learn, advances the schedule, or consumes
    exploration RNG. Episode costs include the explicit pre-decision prefix.
    """
    adapter.reset(seed=environment_seed)
    losses, actions_taken, transition_count = [], [], 0
    slot_start = training_slots
    was_training = agent.online.training
    agent.online.train(training)
    try:
        while not adapter.environment.terminated:
            epsilon = schedule.value(training_slots) if training else 0.
            actions = {user_id: agent.act(observation, epsilon=epsilon)
                       for user_id, observation in sorted(adapter.observations.items())}
            actions_taken.extend({"slot": adapter.environment.slot, "user_vehicle_id": user_id,
                                  "task_id": adapter.observations[user_id].task_id,
                                  "action": action, "epsilon": epsilon}
                                 for user_id, action in actions.items())
            result = adapter.step(actions)
            transition_count += len(result.transitions)
            if training:
                training_slots += 1
                warmup = max(agent.config.batch_size, agent.config.learning_starts)
                eligible_before = max(0, agent.seen_transitions - warmup + 1)
                for item in result.transitions:
                    agent.observe(item)
                eligible_after = max(0, agent.seen_transitions - warmup + 1)
                # Exact post-warm-up transition budget, independent of how many
                # users close intervals simultaneously in the threshold slot.
                for _ in range(eligible_after - eligible_before):
                    update = agent.learn()
                    if update is not None:
                        losses.append(asdict(update))
    finally:
        agent.online.train(was_training)
    accounting = asdict(adapter.accounting)
    for user_id, total in accounting["total_cost"].items():
        for total_key, prefix_key, closed_key in (
            ("total_cost", "prefix_cost", "closed_transition_cost"),
            ("discounted_total_cost", "discounted_prefix_cost", "discounted_closed_transition_cost"),
        ):
            if not math.isclose(accounting[total_key][user_id],
                                accounting[prefix_key][user_id] + accounting[closed_key][user_id],
                                rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("learning run lost prefix or decision-interval costs")
    return {
        "environment_seed": environment_seed, "training": training,
        "training_slots_before": slot_start, "training_slots_after": training_slots,
        "elapsed_slots": adapter.environment.slot, "transition_count": transition_count,
        "updates": losses, "metrics": adapter.environment.metrics(),
        "accounting": accounting, "actions": actions_taken,
    }, training_slots


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
            "python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "numpy_version": numpy.__version__, "device": "cpu", "threads": 1,
            "platform": platform.platform(), "machine": platform.machine(),
            "installed_packages": dict(sorted((package.metadata["Name"], package.version)
                                               for package in distributions())),
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
