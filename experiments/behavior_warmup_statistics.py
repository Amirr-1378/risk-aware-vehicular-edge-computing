"""Read-only comparisons of standard versus optional greedy training behavior.

Only frozen learned-policy validation outcomes are compared. No Torch, RNG,
model selection, actor-return pooling, bootstrap or significance inference.
Training-data/budget pairing remains the experiment producer's responsibility.
"""

import math
from statistics import stdev

from risk_aware_v2x.entities._validation import require_fraction, require_non_negative_int, require_positive_int
from risk_aware_v2x.risk.statistics import episode_cost_summary
from .return_scope_statistics import SCOPES, _episodes, _labels

VARIANTS = ("standard", "greedy_warmup")


def summarize_behavior_variants(runs, *, variants: tuple, return_scopes: tuple,
                                algorithms: tuple, agent_seeds: tuple,
                                profiles: tuple, validation_cases: int,
                                alpha: float = .9) -> dict:
    """Equal-weight physical episode costs, separate fixed model risks.

    Warmup-minus-standard deltas pair one scope/algorithm/seed on the same
    exogenous validation scenes. Seed mean/std summaries are conditional on
    one fixed producer cohort, not scenario-population uncertainty estimates.
    """
    if set(_labels("variants", variants)) != set(VARIANTS):
        raise ValueError("both standard and greedy_warmup variants are required exactly once")
    if set(_labels("return_scopes", return_scopes)) != set(SCOPES):
        raise ValueError("both individual and team return scopes are required exactly once")
    algorithms, profiles = _labels("algorithms", algorithms), _labels("profiles", profiles)
    if not isinstance(agent_seeds, tuple) or not agent_seeds:
        raise ValueError("agent_seeds must be a nonempty immutable tuple")
    for seed in agent_seeds:
        require_non_negative_int("agent seed", seed)
    if len(agent_seeds) != len(set(agent_seeds)):
        raise ValueError("duplicate agent seeds")
    agent_seeds = tuple(sorted(agent_seeds))
    require_positive_int("validation_cases", validation_cases)
    alpha = require_fraction("alpha", alpha)
    if alpha == 1.:
        raise ValueError("CVaR confidence must be below one")
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    expected = {(variant, scope, algorithm, seed) for variant in VARIANTS for scope in SCOPES
                for algorithm in algorithms for seed in agent_seeds}
    index = {}
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("runs must contain dictionaries")
        require_non_negative_int("agent seed", run["agent_seed"])
        key = run["behavior_variant"], run["return_scope"], run["algorithm"], run["agent_seed"]
        if key not in expected or key in index:
            raise ValueError("unexpected or duplicate variant/scope/algorithm/seed run")
        if set(run["validation"]) != set(profiles):
            raise ValueError("all declared validation profiles must be present")
        index[key] = {profile: _episodes(run["validation"][profile], profile=profile, count=validation_cases)
                      for profile in profiles}
    if set(index) != expected:
        raise ValueError("incomplete variant/scope/algorithm/seed grid; failed runs cannot be omitted")

    risks, paired, seed_rows = [], [], []
    for profile in profiles:
        identifiers = [f"{profile}:validation:{i}" for i in range(validation_cases)]
        reference = index[(VARIANTS[0], SCOPES[0], algorithms[0], agent_seeds[0])][profile]
        for records_by_profile in index.values():
            records = records_by_profile[profile]
            if any(records[key]["fingerprint"] != reference[key]["fingerprint"]
                   or records[key]["seeds"] != reference[key]["seeds"] for key in identifiers):
                raise ValueError("unpaired validation fingerprints or exogenous seeds across variants/models")
        for scope in SCOPES:
            for algorithm in algorithms:
                seed_deltas = {}
                for seed in agent_seeds:
                    variant_costs = {}
                    for variant in VARIANTS:
                        costs = [index[(variant, scope, algorithm, seed)][profile][key]["cost"] for key in identifiers]
                        variant_costs[variant] = costs
                        risks.append({"behavior_variant": variant, "return_scope": scope,
                            "algorithm": algorithm, "agent_seed": seed, "profile": profile,
                            "episode_cost_risk": episode_cost_summary(costs, alpha=alpha)})
                    differences = [{"episode_id": identifier, "standard_cost": standard,
                        "greedy_warmup_cost": warmup, "greedy_warmup_minus_standard_cost": warmup - standard}
                        for identifier, standard, warmup in zip(identifiers, variant_costs["standard"],
                                                               variant_costs["greedy_warmup"], strict=True)]
                    mean_delta = math.fsum(item["greedy_warmup_minus_standard_cost"] for item in differences) / validation_cases
                    seed_deltas[str(seed)] = mean_delta
                    paired.append({"return_scope": scope, "algorithm": algorithm, "agent_seed": seed,
                        "profile": profile, "episode_differences": differences,
                        "mean_greedy_warmup_minus_standard_cost": mean_delta})
                values = tuple(seed_deltas.values())
                seed_rows.append({"return_scope": scope, "algorithm": algorithm, "profile": profile,
                    "seed_mean_differences": seed_deltas, "mean_of_seed_differences": math.fsum(values) / len(values),
                    "sample_std_of_seed_differences": stdev(values) if len(values) >= 2 else None,
                    "trained_agent_seed_pairs": len(values)})

    return {"summary_schema_version": 1, "variants": list(VARIANTS), "return_scopes": list(SCOPES),
        "algorithms": list(algorithms), "agent_seeds": list(agent_seeds), "profiles": list(profiles),
        "validation_cases": validation_cases, "cvar_alpha": alpha,
        "cost_unit": "dimensionless full-drain physical episode-total cost",
        "model_cost_risks": risks, "paired_variant_differences": paired,
        "conditional_seed_differences": seed_rows, "validation_scene_pairing_verified": True,
        "training_pairing": "producer responsibility; training traces and phase budgets are not inputs to this summarizer",
        "interpretation": [
            "Each risk row is one fixed learned model/profile; never pool variants, seeds, scopes or overlapping actor-return samples into a policy CVaR.",
            "Costs have equal episode weights, including empty episodes; task counts do not weight risk or paired cost differences.",
            "Negative greedy_warmup-minus-standard differences favor warmup on these observed matched validation episodes.",
            "Pairing shares exogenous scenes/seeds, not identical actions, queues, trajectories or outcomes.",
            "The teacher changes training data collection only in the declared variant; evaluation uses the learned masked J selector without a teacher.",
            "Warmup is an optional training-behavior variant, not identical original training, weight initialization, or behavior cloning.",
            "Seed mean/std summaries are conditional on one fixed producer training/validation cohort, not scenario-population uncertainty.",
            "Known shifted profiles reuse seed keys and remain separate; they are not independent population replications.",
            "Tiny/dependent validation tails and few trained seeds do not establish reliable risk improvement, convergence or superiority.",
            "No confidence interval, p-value, significance test, bootstrap, best-seed/checkpoint or variant selection is performed.",
        ]}


def render_markdown(summary) -> str:
    """Render descriptive fixed-model risks and paired behavior-variant costs."""
    lines = ["# Training-behavior warmup development comparison", "",
        "Validation uses learned policies only; the teacher is unavailable during evaluation.",
        "Costs are dimensionless full-drain episode totals with equal episode weights.",
        "Negative paired greedy_warmup-minus-standard differences favor warmup on this cohort.", "",
        f"| Profile | Algorithm | Return scope | Seed | Behavior variant | Episodes | Mean cost | CVaR {summary['cvar_alpha']:g} | Tail episodes |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary["model_cost_risks"]:
        risk = row["episode_cost_risk"]
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['return_scope']} | {row['agent_seed']} | "
            f"{row['behavior_variant']} | {risk['episodes']} | {risk['mean_episode_total_cost']:.6f} | "
            f"{risk['cvar_episode_total_cost']:.6f} | {risk['nominal_tail_mass_in_episodes']:.4f} |")
    lines.extend(["", "| Profile | Algorithm | Return scope | Seed | Mean warmup minus standard cost |",
                  "| --- | --- | --- | ---: | ---: |"])
    for row in summary["paired_variant_differences"]:
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['return_scope']} | {row['agent_seed']} | "
                     f"{row['mean_greedy_warmup_minus_standard_cost']:.6f} |")
    lines.extend(["", "Across-seed differences are conditional on the fixed producer training/validation cohort.", "",
        "| Profile | Algorithm | Return scope | Seed pairs | Mean seed difference | Sample std of seed differences |",
        "| --- | --- | --- | ---: | ---: | ---: |"])
    for row in summary["conditional_seed_differences"]:
        spread = row["sample_std_of_seed_differences"]
        spread_text = "n/a" if spread is None else f"{spread:.6f}"
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['return_scope']} | "
            f"{row['trained_agent_seed_pairs']} | {row['mean_of_seed_differences']:.6f} | {spread_text} |")
    lines.extend(["", *[f"- {note}" for note in summary["interpretation"]], ""])
    return "\n".join(lines)
