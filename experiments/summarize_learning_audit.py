"""Read-only development diagnostics from a completed v3 learning audit.

No Torch, checkpoint loading, training, test access, model selection, or RNG.
Tail costs stay separate for every trained model; fixed policies appear once.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from risk_aware_v2x.contracts import COST_MODEL_VERSION, model_contract
from risk_aware_v2x.entities._validation import require_fraction, require_non_negative_float
from risk_aware_v2x.risk.statistics import episode_cost_summary
from risk_aware_v2x.scenarios.generator import canonical_json
from risk_aware_v2x.scenarios.protocol import _unique_object


def summarize_episodes(episodes: list[dict], *, alpha: float) -> dict:
    """Episode-weighted cost risk; task-weighted latency/energy/miss rates.

    Workload fractions are averaged across NONEMPTY episodes: the input does
    not contain total CPU cycles, so a pooled workload fraction is unavailable.
    Empty episodes stay in cost risk, with undefined task/workload rates.
    """
    if not episodes:
        raise ValueError("no episodes; missing runs must not be dropped")
    metrics = [episode["metrics"] for episode in episodes]
    for item in metrics:
        if type(item["cost_model_version"]) is not int or item["cost_model_version"] != COST_MODEL_VERSION:
            raise ValueError("incompatible episode cost model")
        for key in ("arrived_tasks", "completed_tasks", "deadline_misses", "query_count"):
            value = require_non_negative_float(key, item[key])
            if not value.is_integer():
                raise ValueError(f"{key} must be an integer count")
        if item["arrived_tasks"] != item["completed_tasks"]:
            raise ValueError("incomplete drain; do not omit failed tasks")
        if item["deadline_misses"] > item["arrived_tasks"]:
            raise ValueError("misses exceed arrived tasks")
        for key in ("total_cost", "mobile_energy_j", "total_task_tardiness_s"):
            require_non_negative_float(key, item[key])
        if item["arrived_tasks"]:
            for key in ("mean_task_latency_s", "mean_task_tardiness_s"):
                require_non_negative_float(key, item[key])
            for key in ("direct_mec_task_fraction", "fallback_task_fraction",
                        "v2v_completed_workload_fraction", "mec_completed_workload_fraction"):
                require_fraction(key, item[key])
            if not math.isclose(item["v2v_completed_workload_fraction"]
                                + item["mec_completed_workload_fraction"], 1., abs_tol=1e-12):
                raise ValueError("completed workload fractions must sum to one")
        elif any(item[key] != 0. for key in ("total_cost", "mobile_energy_j", "total_task_tardiness_s")):
            raise ValueError("empty episodes cannot have task costs/energy/tardiness")
    nonempty = [item for item in metrics if item["arrived_tasks"]]
    tasks = int(sum(item["arrived_tasks"] for item in metrics))

    def task_mean(key):
        return math.fsum(item[key] * item["arrived_tasks"] for item in nonempty) / tasks if tasks else None

    def workload_mean(key):
        return math.fsum(item[key] for item in nonempty) / len(nonempty) if nonempty else None

    return {
        "episodes": len(episodes), "empty_episodes": len(metrics) - len(nonempty),
        "task_instances": tasks,
        "episode_cost_risk": episode_cost_summary([item["total_cost"] for item in metrics], alpha=alpha),
        "task_weighted_latency_s": task_mean("mean_task_latency_s"),
        "task_weighted_mobile_energy_j": math.fsum(item["mobile_energy_j"] for item in metrics) / tasks if tasks else None,
        "task_weighted_deadline_miss_fraction": sum(item["deadline_misses"] for item in metrics) / tasks if tasks else None,
        "task_weighted_tardiness_s": math.fsum(item["total_task_tardiness_s"] for item in metrics) / tasks if tasks else None,
        "task_weighted_direct_mec_fraction": task_mean("direct_mec_task_fraction"),
        "task_weighted_fallback_fraction": task_mean("fallback_task_fraction"),
        "episode_mean_v2v_workload_fraction_nonempty": workload_mean("v2v_completed_workload_fraction"),
        "episode_mean_mec_workload_fraction_nonempty": workload_mean("mec_completed_workload_fraction"),
        "mean_queries_per_episode": math.fsum(item["query_count"] for item in metrics) / len(metrics),
    }


def _paired_keys(episodes, profile, split, count):
    if [item["episode_id"] for item in episodes] != [f"{profile}:{split}:{i}" for i in range(count)]:
        raise ValueError("missing, duplicate, or reordered episode identifiers")
    return [(item["episode_id"], item.get("scenario_fingerprint", item.get("fingerprint")))
            for item in episodes]


def _training_trace(run, learning):
    cumulative_updates, cumulative_transitions, slots = 0, 0, 0
    result = []
    for episode in run["training"]:
        if (episode["training"] is not True or episode["training_slots_before"] != slots
                or episode["training_slots_after"] != slots + episode["elapsed_slots"]):
            raise ValueError("inconsistent training clock")
        slots = episode["training_slots_after"]
        losses, gradients = [], []
        for update in episode["updates"]:
            cumulative_updates += 1
            if update["gradient_updates"] != cumulative_updates:
                raise ValueError("inconsistent gradient update counter")
            losses.append(require_non_negative_float("loss", update["loss"]))
            gradients.append(require_non_negative_float("gradient norm", update["gradient_norm_before_clip"]))
        cumulative_transitions += episode["transition_count"]
        epsilon = learning["epsilon_schedule"]
        end_epsilon = epsilon["initial"] + min(1., slots / epsilon["decay_slots"]) * (epsilon["final"] - epsilon["initial"])
        result.append({"episode_id": episode["episode_id"], "training_slots_after": slots,
            "transitions": episode["transition_count"], "updates": len(losses),
            "mean_update_loss": math.fsum(losses) / len(losses) if losses else None,
            "max_gradient_norm_before_clip": max(gradients) if gradients else None,
            "epsilon_at_episode_end": end_epsilon, "total_cost": episode["metrics"]["total_cost"]})
    if (run["training_slots"], run["seen_transitions"], run["gradient_updates"]) != (slots, cumulative_transitions, cumulative_updates):
        raise ValueError("run counters disagree with episode trace")
    return result


def summarize_learning_audit(report: dict) -> dict:
    """Fail closed on old/missing runs; never retrain or choose a winning seed."""
    if type(report["report_schema_version"]) is not int or report["report_schema_version"] != 3:
        raise ValueError("requires learning report schema v3")
    if report["model_contract"] != model_contract() or report["test_split_accessed"] is not False:
        raise ValueError("requires current cost contract and unopened test split")
    learning, scenarios = report["learning_protocol"], report["scenario_protocol"]
    for plan, key in ((learning, "learning_protocol_sha256"), (scenarios, "scenario_protocol_sha256")):
        if hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest() != report[key]:
            raise ValueError("protocol fingerprint mismatch")
    if (learning["schema_version"] != 3 or scenarios["schema_version"] != 2
            or learning["cost_model_version"] != 2 or scenarios["cost_model_version"] != 2
            or learning["purpose"] != "development-learning-audit-not-final-paper-protocol"
            or scenarios["purpose"] != "development-only-not-final-paper-protocol"
            or learning["scenario_protocol_sha256"] != report["scenario_protocol_sha256"]
            or learning["selection_rule"] != "fixed-final-training-episode"
            or report["selection_rule"] != learning["selection_rule"]):
        raise ValueError("incompatible protocol or selection rule")
    if any(item["split"] not in ("train", "validation") for item in report["scenarios"]):
        raise ValueError("unexpected split in development report")
    expected = {(algorithm, seed) for algorithm in learning["algorithms"] for seed in learning["agent_seeds"]}
    runs = report["runs"]
    if len(runs) != len(expected) or {(run["algorithm"], run["agent_seed"]) for run in runs} != expected:
        raise ValueError("incomplete or duplicate algorithm/seed grid")
    profiles = learning["validation_profiles"]
    for values in (learning["algorithms"], learning["agent_seeds"], profiles):
        if not values or len(values) != len(set(values)):
            raise ValueError("protocol axes must be nonempty and unique")
    if set(report["analytical_validation"]) != set(profiles):
        raise ValueError("missing validation profile")
    rows, traces, reference_train = [], [], None
    alpha = scenarios["cvar_alpha"]
    for run in sorted(runs, key=lambda item: (item["algorithm"], item["agent_seed"])):
        train = run["training"]
        train_keys = _paired_keys(train, learning["training_profile"], "train", learning["training_episodes"])
        signature = (train_keys, [item["seeds"] for item in train])
        if reference_train is not None and signature != reference_train:
            raise ValueError("unpaired training scenarios or seeds")
        reference_train = signature
        summarize_episodes(train, alpha=alpha)  # Check complete drain/finite metrics in training too.
        traces.append({"algorithm": run["algorithm"], "agent_seed": run["agent_seed"],
                       "episodes": _training_trace(run, learning)})
        if set(run["validation"]) != set(profiles):
            raise ValueError("missing learned validation profile")
        for profile in profiles:
            values = run["validation"][profile]
            reference = report["analytical_validation"][profile]["scenarios"]
            if (_paired_keys(values, profile, "validation", learning["validation_cases"])
                    != _paired_keys(reference, profile, "validation", learning["validation_cases"])
                    or [item["seeds"] for item in values] != [item["seeds"] for item in reference]):
                raise ValueError("unpaired validation scenarios or seeds")
            if any(item["training"] is not False or item["updates"]
                   or item["training_slots_before"] != run["training_slots"]
                   or item["training_slots_after"] != run["training_slots"]
                   or any(action["epsilon"] != 0. for action in item["actions"]) for item in values):
                raise ValueError("validation must be frozen and greedy")
            rows.append({"kind": "learned", "profile": profile, "policy": run["algorithm"],
                         "agent_seed": run["agent_seed"], **summarize_episodes(values, alpha=alpha)})
    fixed_policies = {"always_mec", "always_v2v", "uniform_random", "minimum_estimated_latency",
                      "minimum_estimated_combined_cost"}
    for profile in profiles:
        audit = report["analytical_validation"][profile]
        if audit["split"] != "validation" or audit["model_contract"] != model_contract():
            raise ValueError("incompatible analytical validation")
        if set(audit["policies"]) != fixed_policies:
            raise ValueError("incomplete fixed-policy suite")
        reference = _paired_keys(audit["scenarios"], profile, "validation", learning["validation_cases"])
        for policy, episodes in sorted(audit["policies"].items()):
            if _paired_keys(episodes, profile, "validation", learning["validation_cases"]) != reference:
                raise ValueError("unpaired fixed-policy scenarios")
            rows.append({"kind": "fixed", "profile": profile, "policy": policy, "agent_seed": None,
                         **summarize_episodes(episodes, alpha=alpha)})
    return {"diagnostic_schema_version": 1, "purpose": "bounded development diagnostics; not main training",
        "model_contract": model_contract(), "source_provenance": report["provenance"],
        "learning_protocol_sha256": report["learning_protocol_sha256"],
        "scenario_protocol_sha256": report["scenario_protocol_sha256"],
        "test_split_accessed": False, "main_training_started": False, "cvar_alpha": alpha,
        "training_traces": traces, "validation_rows": rows,
        "episode_executions": sum(len(run["training"]) for run in runs) + sum(row["episodes"] for row in rows),
        "interpretation": [
            "Each learned row is ONE fixed final model; never pool seeds into a policy CVaR.",
            "Fixed policies are evaluated once per scenario, not replicated across agent seeds.",
            "Task instances repeat shared traces and are not independent statistical samples.",
            "Cost risk includes empty episodes; task rates exclude zero task denominators.",
            "Workload fractions are episode means over nonempty episodes, not pooled CPU cycles.",
            "Do not average episode p95/p99 values into a pooled latency quantile.",
            "Loss scales differ between scalar and QR heads; low loss does not prove good control.",
            "Tiny validation tails and two seeds do not establish convergence or superiority.",
            "Main training requires a separate calibrated protocol and execution in Colab.",
        ]}


def render_markdown(summary: dict) -> str:
    lines = ["# Development learning diagnostics", "", "Main training has not started. Test split remains closed.",
        "", "Every learned row is a separate final model. CVaR is over episode-total cost.",
        "Fractions below are displayed as percentages; latency is task-weighted in seconds.",
        "SV work is an episode mean over nonempty episodes. Empty task rates are n/a.", "",
        f"| Profile | Policy | Seed | Episodes | Mean cost | CVaR {summary['cvar_alpha']:g} | Tail episodes | Latency s | Miss % | SV work % | Fallback % |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def number(value, scale=1.):
        return "n/a" if value is None else f"{value * scale:.4f}"

    for row in summary["validation_rows"]:
        risk = row["episode_cost_risk"]
        seed = "—" if row["agent_seed"] is None else str(row["agent_seed"])
        values = [row["profile"], row["policy"], seed, str(row["episodes"]),
            number(risk["mean_episode_total_cost"]), number(risk["cvar_episode_total_cost"]),
            number(risk["nominal_tail_mass_in_episodes"]), number(row["task_weighted_latency_s"]),
            number(row["task_weighted_deadline_miss_fraction"], 100),
            number(row["episode_mean_v2v_workload_fraction_nonempty"], 100),
            number(row["task_weighted_fallback_fraction"], 100)]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", *[f"- {line}" for line in summary["interpretation"]], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwrite results")
    args = parser.parse_args()
    payload = args.report.read_bytes()
    report = json.loads(payload, object_pairs_hook=_unique_object)
    summary = summarize_learning_audit(report)
    summary["input_report_sha256"] = hashlib.sha256(payload).hexdigest()
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    markdown = render_markdown(summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "diagnostics.json").write_text(serialized, encoding="utf-8")
    (args.output_dir / "diagnostics.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_directory": str(args.output_dir), "episode_executions": summary["episode_executions"],
                      "validation_rows": len(summary["validation_rows"]), "main_training_started": False}))


if __name__ == "__main__":
    main()
