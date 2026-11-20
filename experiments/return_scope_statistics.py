"""Read-only episode-cost comparisons for individual/team return ablations.

No Torch, simulation, model selection, RNG, bootstrap or pooled-policy CVaR.
Exogenous validation pairing is checked; training pairing belongs to the
experiment producer because this function does not receive training traces.
"""

import math
from statistics import stdev

from risk_aware_v2x.contracts import COST_MODEL_VERSION
from risk_aware_v2x.entities._validation import (
    require_fraction, require_non_negative_float, require_non_negative_int,
    require_positive_int,
)
from risk_aware_v2x.risk.statistics import episode_cost_summary

SCOPES = ("individual", "team")


def _labels(name, values):
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a nonempty immutable tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name}")
    return tuple(sorted(values))


def _episodes(values, *, profile, count):
    """Canonicalize a complete validation prefix without dropping any record."""
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("the declared validation episode count must be complete")
    expected = {f"{profile}:validation:{index}": index for index in range(count)}
    records = {}
    for episode in values:
        if not isinstance(episode, dict):
            raise TypeError("validation episodes must be dictionaries")
        identifier = episode["episode_id"]
        if identifier not in expected or identifier in records:
            raise ValueError("missing, duplicate or unexpected validation episode ID")
        fingerprint = episode["scenario_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("a scenario fingerprint is required for pairing")
        seeds = episode["seeds"]
        if (not isinstance(seeds, dict) or seeds.get("split") != "validation"
                or type(seeds.get("index")) is not int or seeds["index"] != expected[identifier]):
            raise ValueError("episode seed split/index must match its validation ID")
        for role in ("scenario_seed", "channel_seed", "policy_seed"):
            require_non_negative_int(role, seeds[role])
        if "training" in episode and episode["training"] is not False:
            raise ValueError("cost comparison requires frozen validation episodes")
        metrics = episode["metrics"]
        cost = require_non_negative_float("episode total cost", metrics["total_cost"])
        if "cost_model_version" in metrics:
            if type(metrics["cost_model_version"]) is not int or metrics["cost_model_version"] != COST_MODEL_VERSION:
                raise ValueError("incompatible episode cost model")
        if "arrived_tasks" in metrics or "completed_tasks" in metrics:
            counts = [require_non_negative_float(name, metrics[name]) for name in ("arrived_tasks", "completed_tasks")]
            if any(not value.is_integer() for value in counts) or counts[0] != counts[1]:
                raise ValueError("episode metrics must retain complete task drain")
            if counts[0] == 0. and cost != 0.:
                raise ValueError("empty task episodes cannot have physical task cost")
        records[identifier] = {"cost": cost, "fingerprint": fingerprint, "seeds": seeds}
    return records


def summarize_return_scope_runs(runs, *, algorithms: tuple, agent_seeds: tuple,
                                profiles: tuple, validation_cases: int, alpha: float = .9) -> dict:
    """Compare equal-weight episode totals, keeping models and seeds separate.

    Per-seed deltas are means over matched validation episodes. Across-seed
    statistics average those seed deltas, conditional on the producer's fixed
    train/validation traces; they do not estimate scenario-population uncertainty.
    """
    algorithms, profiles = _labels("algorithms", algorithms), _labels("profiles", profiles)
    if not isinstance(agent_seeds, tuple) or not agent_seeds:
        raise ValueError("agent_seeds must be a nonempty immutable tuple")
    for seed in agent_seeds:
        require_non_negative_int("agent seed", seed)
    if len(set(agent_seeds)) != len(agent_seeds):
        raise ValueError("duplicate agent seeds")
    agent_seeds = tuple(sorted(agent_seeds))
    require_positive_int("validation_cases", validation_cases)
    alpha = require_fraction("alpha", alpha)
    if alpha == 1.:
        raise ValueError("CVaR confidence must be below one")
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    expected = {(scope, algorithm, seed) for scope in SCOPES for algorithm in algorithms for seed in agent_seeds}
    index = {}
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("runs must contain dictionaries")
        key = (run["return_scope"], run["algorithm"], run["agent_seed"])
        require_non_negative_int("agent seed", run["agent_seed"])
        if key not in expected or key in index:
            raise ValueError("unexpected or duplicate return-scope/algorithm/seed run")
        if set(run["validation"]) != set(profiles):
            raise ValueError("all declared validation profiles must be present")
        index[key] = {profile: _episodes(run["validation"][profile], profile=profile, count=validation_cases)
                      for profile in profiles}
    if set(index) != expected:
        raise ValueError("incomplete return-scope/algorithm/seed grid; failed runs cannot be omitted")

    risks, paired, seed_rows = [], [], []
    for profile in profiles:
        identifiers = [f"{profile}:validation:{number}" for number in range(validation_cases)]
        reference = index[(SCOPES[0], algorithms[0], agent_seeds[0])][profile]
        for records_by_profile in index.values():
            records = records_by_profile[profile]
            if any(records[key]["fingerprint"] != reference[key]["fingerprint"]
                   or records[key]["seeds"] != reference[key]["seeds"] for key in identifiers):
                raise ValueError("unpaired validation scenario fingerprints or exogenous seeds")
        for algorithm in algorithms:
            deltas_by_seed = {}
            for seed in agent_seeds:
                costs_by_scope = {}
                for scope in SCOPES:
                    costs = [index[(scope, algorithm, seed)][profile][key]["cost"] for key in identifiers]
                    costs_by_scope[scope] = costs
                    risks.append({"return_scope": scope, "algorithm": algorithm, "agent_seed": seed,
                                  "profile": profile, "episode_cost_risk": episode_cost_summary(costs, alpha=alpha)})
                differences = [{"episode_id": identifier, "individual_cost": individual,
                    "team_cost": team, "team_minus_individual_cost": team - individual}
                    for identifier, individual, team in zip(identifiers, costs_by_scope["individual"], costs_by_scope["team"], strict=True)]
                delta = math.fsum(item["team_minus_individual_cost"] for item in differences) / validation_cases
                deltas_by_seed[str(seed)] = delta
                paired.append({"algorithm": algorithm, "agent_seed": seed, "profile": profile,
                    "episode_differences": differences, "mean_team_minus_individual_cost": delta})
            values = tuple(deltas_by_seed.values())
            seed_rows.append({"algorithm": algorithm, "profile": profile,
                "seed_mean_differences": deltas_by_seed,
                "mean_of_seed_differences": math.fsum(values) / len(values),
                "sample_std_of_seed_differences": stdev(values) if len(values) >= 2 else None,
                "trained_agent_seed_pairs": len(values)})

    return {"summary_schema_version": 1, "cvar_alpha": alpha,
        "algorithms": list(algorithms), "agent_seeds": list(agent_seeds), "profiles": list(profiles),
        "validation_cases": validation_cases, "cost_unit": "dimensionless full-drain physical episode-total cost",
        "model_cost_risks": risks, "paired_scope_differences": paired,
        "conditional_seed_differences": seed_rows,
        "validation_scene_pairing_verified": True,
        "training_pairing": "producer responsibility; training traces are not inputs to this summarizer",
        "interpretation": [
            "Every risk row is one fixed final model/profile; never pool seeds or actor-return samples into a policy CVaR.",
            "Costs have equal episode weights, including empty episodes; task counts do not weight cost risk.",
            "Profiles stay separate; reusing exogenous seed keys across known shifts does not create independent replications.",
            "Team-minus-individual cost below zero favors team returns on these observed matched episodes.",
            "Scope deltas pair exogenous scenarios/seeds, not identical learned actions, queues or outcomes.",
            "Seed summaries are conditional on the producer's fixed training/validation traces, not scenario-population uncertainty.",
            "Tiny or dependent validation tails and few trained seeds cannot establish reliable risk improvement, convergence or superiority.",
            "No confidence interval, p-value, significance test, bootstrap or model/seed selection is performed.",
        ]}


def render_markdown(summary) -> str:
    """Render descriptive per-model risk and matched scope differences only."""
    lines = ["# Return-scope development comparison", "",
        "Costs are dimensionless full-drain episode totals, with equal episode weights.",
        "Each risk row is one fixed final model. Negative paired differences favor team returns.", "",
        f"| Profile | Algorithm | Seed | Return scope | Episodes | Mean cost | CVaR {summary['cvar_alpha']:g} | Tail episodes |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary["model_cost_risks"]:
        risk = row["episode_cost_risk"]
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['agent_seed']} | {row['return_scope']} | "
            f"{risk['episodes']} | {risk['mean_episode_total_cost']:.6f} | {risk['cvar_episode_total_cost']:.6f} | "
            f"{risk['nominal_tail_mass_in_episodes']:.4f} |")
    lines.extend(["", "| Profile | Algorithm | Seed | Mean team minus individual cost |",
                  "| --- | --- | ---: | ---: |"])
    for row in summary["paired_scope_differences"]:
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['agent_seed']} | {row['mean_team_minus_individual_cost']:.6f} |")
    lines.extend(["", "Across-seed differences below are conditional on the fixed producer training/validation traces.", "",
        "| Profile | Algorithm | Seed pairs | Mean seed difference | Sample std of seed differences |",
        "| --- | --- | ---: | ---: | ---: |"])
    for row in summary["conditional_seed_differences"]:
        spread = row["sample_std_of_seed_differences"]
        spread_text = "n/a" if spread is None else f"{spread:.6f}"
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['trained_agent_seed_pairs']} | "
            f"{row['mean_of_seed_differences']:.6f} | {spread_text} |")
    lines.extend(["", *[f"- {note}" for note in summary["interpretation"]], ""])
    return "\n".join(lines)
