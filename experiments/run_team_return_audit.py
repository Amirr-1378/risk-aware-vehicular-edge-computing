"""Verify team-return accounting with unchanged independent fixed-policy actors.

Matched individual/team-interface runs must have identical local inputs,
actions, physical outcomes, and individual ledgers. No neural training, no
centralized decision-maker, and no test access.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from risk_aware_v2x.agents.team_returns import TeamReturnAdapter, team_return_contract
from risk_aware_v2x.contracts import model_contract
from risk_aware_v2x.entities._validation import require_fraction, require_positive_int
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .comparison_policies import comparison_policies
from .provenance import source_provenance
from .run_scenario_audit import DEFAULT_PROTOCOL, save_audit_report

POLICIES = ("always_mec", "always_v2v", "uniform_random",
            "minimum_estimated_latency", "minimum_estimated_combined_cost")


def _run(scene, key, policy_name, gamma, *, team):
    environment = scene.environment()
    encoder = ObservationEncoder(max_server_vehicles=environment.config.max_server_vehicles, scales=FeatureScales())
    kind = TeamReturnAdapter if team else DecisionAdapter
    adapter = kind(environment, encoder, gamma_per_slot=gamma)
    adapter.reset(seed=key.channel_seed)
    policies, configuration = comparison_policies(scene.configuration, scene.users, scene.servers, seed=key.policy_seed)
    policy = policies[policy_name]
    digest, actions_taken, transitions, intervals = hashlib.sha256(), [], [], []
    while not environment.terminated:
        local = adapter.observations
        # Hash local actor inputs, including masks; no team state is an actor input.
        digest.update(canonical_json({uid: asdict(obs) for uid, obs in sorted(local.items())}).encode("utf-8") + b"\0")
        actions = {uid: policy.act(observation) for uid, observation in sorted(adapter.raw_observations.items())}
        actions_taken.extend({"slot": environment.slot, "user_vehicle_id": uid,
                              "task_id": local[uid].task_id, "action": action} for uid, action in actions.items())
        step = adapter.step(actions)
        for transition in step.transitions:
            source = transition.source if team else transition
            item = {"user_vehicle_id": source.observation.user_vehicle_id, "task_id": source.observation.task_id,
                "start_slot": source.start_slot, "duration_slots": source.duration_slots, "action": source.action,
                "individual_cost": source.cost, "discounted_individual_cost": source.discounted_cost,
                "bootstrap_discount": source.bootstrap_discount, "terminated": source.terminated}
            if team:
                item.update(team_cost=transition.cost, discounted_team_cost=transition.discounted_cost,
                            team_reward=transition.discounted_reward, return_scope=transition.return_scope)
            transitions.append(item)
        if team and step.global_interval is not None:
            intervals.append(asdict(step.global_interval))
    metrics = terminal_metrics(environment)
    individual = adapter.individual_accounting if team else adapter.accounting
    if not math.isclose(math.fsum(individual.total_cost.values()), metrics["total_cost"], rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("individual ledgers do not match actual terminal cost")
    if team:
        accounting = adapter.accounting
        if (not math.isclose(accounting.total_cost, metrics["total_cost"], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(accounting.total_cost, accounting.prefix_cost + accounting.closed_interval_cost,
                                    rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(accounting.discounted_total_cost,
                                    accounting.discounted_prefix_cost + accounting.discounted_closed_interval_cost,
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise RuntimeError("unique global team ledger lost or duplicated costs")
        for uid in environment.users:
            if (not math.isclose(accounting.total_cost,
                    accounting.per_user_prefix_cost[uid] + accounting.per_user_closed_cost[uid], rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(accounting.discounted_total_cost,
                    accounting.per_user_discounted_prefix_cost[uid] + accounting.per_user_discounted_closed_cost[uid],
                    rel_tol=1e-12, abs_tol=1e-12)):
                raise RuntimeError("an independent actor view lost part of the team return")
    return {"metrics": metrics, "elapsed_slots": environment.slot, "policy_configuration": configuration,
            "actor_inputs_sha256": digest.hexdigest(), "actions": actions_taken, "transitions": transitions,
            "individual_accounting": asdict(individual), "team_accounting": asdict(adapter.accounting) if team else None,
            "global_intervals": intervals}


def run_team_return_audit(*, protocol: DevelopmentProtocol | None = None,
                         profiles=("nominal", "high_load"), cases=2, gamma_per_slot=1.) -> dict:
    require_positive_int("cases", cases)
    gamma = require_fraction("gamma_per_slot", gamma_per_slot)
    if not isinstance(profiles, tuple) or not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("profiles must be an immutable nonempty unique tuple")
    plan = protocol if protocol is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    specifications = [plan.profile(profile) for profile in profiles]
    keys = [plan.episode_seeds("validation", index) for index in range(cases)]
    provenance = source_provenance()
    records = []
    for specification in specifications:
        for key in keys:
            scene = generate_scenario(specification, seed=key.scenario_seed)
            fingerprint = scene.fingerprint
            policies = {}
            for name in POLICIES:
                individual = _run(scene, key, name, gamma, team=False)
                team = _run(scene, key, name, gamma, team=True)
                for field in ("metrics", "elapsed_slots", "policy_configuration", "actor_inputs_sha256", "actions", "individual_accounting"):
                    if individual[field] != team[field]:
                        raise RuntimeError(f"team interface changed independent actor behavior or physics: {field}")
                stripped = [{k: v for k, v in item.items() if k not in
                             ("team_cost", "discounted_team_cost", "team_reward", "return_scope")}
                            for item in team["transitions"]]
                if stripped != individual["transitions"] or scene.fingerprint != fingerprint:
                    raise RuntimeError("individual transitions or scenario prototypes were changed")
                policies[name] = {"matched_individual_run_verified": True, **team}
            records.append({"episode_id": f"{specification.label}:validation:{key.index}", "seeds": asdict(key),
                "scenario_fingerprint": fingerprint, "manifest": scene.manifest(), "policies": policies})
    return {"report_schema_version": 1, "model_contract": model_contract(), "team_return_contract": team_return_contract(),
        "purpose": "cooperative return interface audit for independent actors; not learning performance",
        "test_split_accessed": False, "training_started": False, "centralized_controller": False,
        "gamma_per_slot": gamma, "protocol": asdict(plan), "protocol_sha256": plan.fingerprint,
        "provenance": provenance, "episodes": records,
        "matched_policy_pairs": len(records) * len(POLICIES),
        "physical_episode_executions": 2 * len(records) * len(POLICIES),
        "limits": ["Team-cost views overlap across independent users and must not be summed as episode cost.",
                   "Current QR agents/replay/checkpoints remain individual-return implementations; team records are a distinct type.",
                   "Independent local-observation learning remains partially observed and potentially nonstationary.",
                   "No reduction in cost, convergence, global optimality or deployment coordination is established."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--profiles", nargs="+", default=("nominal", "high_load"))
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--gamma-per-slot", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new report path; do not overwrite historical evidence")
    report = run_team_return_audit(protocol=DevelopmentProtocol.load(args.protocol), profiles=tuple(args.profiles),
        cases=args.cases, gamma_per_slot=args.gamma_per_slot)
    save_audit_report(report, args.output)
    print(json.dumps({"output": str(args.output), "matched_policy_pairs": report["matched_policy_pairs"],
        "physical_episode_executions": report["physical_episode_executions"], "training_started": False}))


if __name__ == "__main__":
    main()
