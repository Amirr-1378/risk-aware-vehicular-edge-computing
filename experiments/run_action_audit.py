"""Offline action comparisons with greedy continuation; no training or test access.

Clones include hidden state/future arrivals for diagnostic simulation ONLY.
Ex-post branch minima are not deployable actions, expected Q values, or an
optimal-policy bound. Gains at different reference states cannot be summed.
"""

import argparse
import copy
import json
import math
from dataclasses import asdict

from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_non_negative_float, require_positive_int
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.environment.observations import observed_valid_actions
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .comparison_policies import comparison_policies
from .provenance import source_provenance
from .run_scenario_audit import DEFAULT_PROTOCOL, save_audit_report

COMPARISON_TOLERANCE = 1e-9  # Dimensionless cost; counts only, never action ranking.


def compare_branches(branches: list[dict], *, greedy_action: int, owner_id: int) -> dict:
    """Contrast one user's remaining cost with the sum over all users.

    Negative branch-minus-greedy differences favor the branch. Exact ties use
    the lowest action index; tolerance only labels strict diagnostic gains.
    """
    actions = [branch["action"] for branch in branches]
    if not branches or len(set(actions)) != len(actions) or greedy_action not in actions:
        raise ValueError("unique branches including the reference action are required")
    users = set(branches[0]["remaining_cost_by_user"])
    if owner_id not in users:
        raise ValueError("missing owner cost")
    for branch in branches:
        if set(branch["remaining_cost_by_user"]) != users:
            raise ValueError("every branch must retain every user")
        for value in branch["remaining_cost_by_user"].values():
            require_non_negative_float("remaining user cost", value)
    total = lambda branch: math.fsum(branch["remaining_cost_by_user"].values())
    own = lambda branch: branch["remaining_cost_by_user"][owner_id]
    reference = next(branch for branch in branches if branch["action"] == greedy_action)
    owner_best = min(branches, key=lambda branch: (own(branch), branch["action"]))
    system_best = min(branches, key=lambda branch: (total(branch), branch["action"]))
    deltas = [{"action": branch["action"], "owner_delta_vs_greedy": own(branch) - own(reference),
               "system_delta_vs_greedy": total(branch) - total(reference)} for branch in branches]
    conflicts = [item["action"] for item in deltas
        if (item["owner_delta_vs_greedy"] < -COMPARISON_TOLERANCE and item["system_delta_vs_greedy"] > COMPARISON_TOLERANCE)
        or (item["owner_delta_vs_greedy"] > COMPARISON_TOLERANCE and item["system_delta_vs_greedy"] < -COMPARISON_TOLERANCE)]
    return {"owner_best_action_ex_post": owner_best["action"],
        "system_best_action_ex_post": system_best["action"],
        "owner_gain_vs_greedy_ex_post": own(reference) - own(owner_best),
        "system_gain_vs_greedy_ex_post": total(reference) - total(system_best),
        "owner_best_change_to_other_users_cost": total(owner_best) - total(reference) - own(owner_best) + own(reference),
        "opposite_owner_system_preferences": sorted(conflicts), "branch_deltas": deltas}


def _greedy_batch(environment, policy):
    # The policy sees only received observations, never the cloned truth state.
    return {user_id: policy.act(observation) for user_id, observation in sorted(environment.observations.items())}


def _finish_branch(environment, policy, first_actions, sunk_cost_by_task):
    environment.step(first_actions)
    while not environment.terminated:
        environment.step(_greedy_batch(environment, policy))
    metrics = terminal_metrics(environment)
    remaining = {user_id: math.fsum(environment.ledgers[task_id].total_cost - sunk_cost_by_task.get(task_id, 0.)
        for task_id, task in environment.tasks.items() if task.owner_vehicle_id == user_id)
        for user_id in environment.users}
    return {"remaining_cost_by_user": remaining, "terminal_metrics": metrics,
            "terminal_slot": environment.slot}


def audit_decision(environment, policy, *, owner_id: int, action_order=None) -> dict:
    """Fork all observed-valid actions; preserve the source environment and RNGs.

    Only the chosen user's current action changes. Other users' simultaneous
    actions are fixed to the same pre-admission greedy batch in every branch.
    Later actions are recomputed from each branch's received observations.
    """
    observations = environment.observations
    observation = observations[owner_id]
    valid = observed_valid_actions(observation)
    order = valid if action_order is None else tuple(action_order)
    if len(order) != len(valid) or set(order) != set(valid):
        raise ValueError("branch order must contain each valid action exactly once")
    batch = _greedy_batch(environment, policy)
    sunk = {task_id: ledger.total_cost for task_id, ledger in environment.ledgers.items()}
    branches = []
    for action in order:
        estimate = policy.estimate(observation, action)
        outcome = _finish_branch(copy.deepcopy(environment), policy, {**batch, owner_id: action}, sunk)
        branches.append({"action": action,
            "estimated_myopic_cost": estimate.cost if math.isfinite(estimate.cost) else None,
            "greedy_contact_admissible": estimate.admissible,
            **outcome})
    branches.sort(key=lambda branch: branch["action"])
    return {"slot": environment.slot, "user_vehicle_id": owner_id, "task_id": observation.task_id,
        "snapshot_age_s": observation.snapshot_age_s, "reference_action_batch": batch,
        "greedy_action": batch[owner_id], "branches": branches,
        **compare_branches(branches, greedy_action=batch[owner_id], owner_id=owner_id)}


def run_action_audit(*, protocol: DevelopmentProtocol | None = None,
                     profiles: tuple[str, ...] = ("nominal", "high_load"),
                     cases: int = 2, decisions_per_episode: int = 8) -> dict:
    """Inspect a declared decision prefix on greedy reference trajectories.

    No outcomes affect selection of inspected states. Reference episodes and
    every branch must fully drain; failures abort rather than remove a case.
    There is deliberately no split/test option or learned-agent dependency.
    """
    require_positive_int("cases", cases)
    require_positive_int("decisions_per_episode", decisions_per_episode)
    if not isinstance(profiles, tuple) or not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("profiles must be an immutable nonempty unique tuple")
    plan = protocol if protocol is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    specifications = [plan.profile(profile) for profile in profiles]
    keys = [plan.episode_seeds("validation", index) for index in range(cases)]
    provenance = source_provenance()
    episodes = []
    for specification in specifications:
        for key in keys:
            scene = generate_scenario(specification, seed=key.scenario_seed)
            fingerprint = scene.fingerprint
            environment = scene.environment()
            environment.reset(seed=key.channel_seed)
            policies, contract = comparison_policies(scene.configuration, scene.users, scene.servers, seed=key.policy_seed)
            policy = policies["minimum_estimated_combined_cost"]
            decisions, total_decisions = [], 0
            while not environment.terminated:
                batch = _greedy_batch(environment, policy)
                for owner_id in sorted(batch):
                    total_decisions += 1
                    if len(decisions) < decisions_per_episode:
                        decisions.append(audit_decision(environment, policy, owner_id=owner_id))
                environment.step(batch)
            reference_metrics = terminal_metrics(environment)
            for decision in decisions:
                reference = next(branch for branch in decision["branches"] if branch["action"] == decision["greedy_action"])
                if reference["terminal_metrics"] != reference_metrics or reference["terminal_slot"] != environment.slot:
                    raise RuntimeError("unchanged-action branch did not reproduce the greedy reference")
            if scene.fingerprint != fingerprint:
                raise RuntimeError("scenario prototypes were changed by branch simulation")
            episodes.append({"episode_id": f"{specification.label}:validation:{key.index}",
                "seeds": asdict(key), "scenario_fingerprint": fingerprint, "manifest": scene.manifest(),
                "reference_metrics": reference_metrics, "policy_configuration": contract,
                "reference_decisions": total_decisions, "inspected_decisions": decisions})
    decisions = [decision for episode in episodes for decision in episode["inspected_decisions"]]
    mean = lambda field: math.fsum(decision[field] for decision in decisions) / len(decisions) if decisions else None
    return {"report_schema_version": 1, "model_contract": model_contract(),
        "purpose": "ex-post single-action diagnostics with greedy continuation; not policy performance",
        "protocol": asdict(plan), "protocol_sha256": plan.fingerprint, "provenance": provenance,
        "test_split_accessed": False, "training_started": False,
        "selection": {"profiles": profiles, "cases": cases, "decisions_per_episode": decisions_per_episode,
                      "rule": "first decisions in slot/user-ID order on the unchanged greedy path"},
        "comparison_tolerance": COMPARISON_TOLERANCE, "episodes": episodes,
        "summary": {"reference_episodes": len(episodes), "inspected_decisions": len(decisions),
            "branch_rollouts": sum(len(decision["branches"]) for decision in decisions),
            "decisions_with_owner_gain": sum(decision["owner_gain_vs_greedy_ex_post"] > COMPARISON_TOLERANCE for decision in decisions),
            "decisions_with_system_gain": sum(decision["system_gain_vs_greedy_ex_post"] > COMPARISON_TOLERANCE for decision in decisions),
            "decisions_with_opposite_owner_system_preferences": sum(bool(decision["opposite_owner_system_preferences"]) for decision in decisions),
            "mean_owner_gain_per_inspected_decision_ex_post": mean("owner_gain_vs_greedy_ex_post"),
            "mean_system_gain_per_inspected_decision_ex_post": mean("system_gain_vs_greedy_ex_post")},
        "limitations": [
            "Hidden-state/future-trajectory clones are offline diagnostics; never policy inputs or replay labels.",
            "Each branch uses one realized exogenous path, not an expected action value or a CVaR estimate.",
            "Taking the ex-post minimum introduces hindsight selection; no deployable improvement is measured.",
            "The fixed greedy continuation is not an optimal-policy bound; gains at different states cannot be added.",
            "One user's own future cost and global summed future cost can rank actions differently.",
            "Local gain opportunities on a small validation prefix do not establish learnable superiority.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--profiles", nargs="+", default=("nominal", "high_load"))
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--decisions-per-episode", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Check existing output before expensive branching; exclusive save also guards a race.
    from pathlib import Path
    if Path(args.output).exists():
        raise FileExistsError("choose a new report path; historical results are never overwritten")
    report = run_action_audit(protocol=DevelopmentProtocol.load(args.protocol), profiles=tuple(args.profiles),
                              cases=args.cases, decisions_per_episode=args.decisions_per_episode)
    save_audit_report(report, args.output)
    print(json.dumps({"output": args.output, **report["summary"], "training_started": False}, allow_nan=False))


if __name__ == "__main__":
    main()
