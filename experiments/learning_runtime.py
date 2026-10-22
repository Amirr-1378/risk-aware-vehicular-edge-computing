"""Shared CPU episode execution for neural experiment entry points."""

import math
import platform
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import distributions

import numpy
import torch


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


def runtime_metadata() -> dict:
    return {
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "numpy_version": numpy.__version__, "device": "cpu", "threads": 1,
        "platform": platform.platform(), "machine": platform.machine(),
        "installed_packages": dict(sorted((package.metadata["Name"], package.version)
                                           for package in distributions())),
        "deterministic_algorithms": True,
    }


def run_episode(adapter, agent, *, environment_seed: int, training: bool,
                schedule, training_slots: int) -> tuple[dict, int]:
    """Choose the complete user batch before service/updates; fully drain.

    One update per newly eligible transition, after all transitions in that
    slot have been inserted. Evaluation does not advance training slots or
    consume exploration/replay RNG. Costs include the pre-decision prefix.
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
                for _ in range(eligible_after - eligible_before):
                    update = agent.learn()
                    if update is not None:
                        losses.append(asdict(update))
    finally:
        agent.online.train(was_training)
    accounting = asdict(adapter.accounting)
    for user_id in accounting["total_cost"]:
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
