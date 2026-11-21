"""Frozen per-model SNR representation comparisons, without pooled-policy risk.

No Torch, simulation, model selection, RNG or significance inference. Exogenous
validation pairing is verified; training pairing belongs to the producer.
"""

import math
from statistics import stdev

from risk_aware_v2x.entities._validation import require_fraction, require_non_negative_int, require_positive_int
from risk_aware_v2x.risk.statistics import episode_cost_summary
from .return_scope_statistics import _episodes, _labels

VARIANTS = ("linear_v3", "log_snr_v4")
QR_ALGORITHMS = ("qr_dqn", "qr_dqn_cvar")


def summarize_snr_representation_runs(runs, *, variants: tuple, algorithms: tuple,
                                      agent_seeds: tuple, profiles: tuple,
                                      validation_cases: int, alpha: float = .9) -> dict:
    """Equal-weight episode risk per model; log-minus-linear paired differences.

    The CVaR difference subtracts the two model CVaRs. It is not the CVaR of
    paired episode differences, and seeds are never pooled into a policy tail.
    """
    if set(_labels("variants", variants)) != set(VARIANTS):
        raise ValueError("both representation variants are required exactly once")
    algorithms, profiles = _labels("algorithms", algorithms), _labels("profiles", profiles)
    if not set(algorithms) <= set(QR_ALGORITHMS):
        raise ValueError("only QR and QR-CVaR algorithms are supported")
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
    expected = {(variant, algorithm, seed) for variant in VARIANTS
                for algorithm in algorithms for seed in agent_seeds}
    index = {}
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("runs must contain dictionaries")
        if run["return_scope"] != "individual":
            raise ValueError("representation comparisons require fixed individual returns")
        require_non_negative_int("agent seed", run["agent_seed"])
        key = run["representation_variant"], run["algorithm"], run["agent_seed"]
        if key not in expected or key in index:
            raise ValueError("unexpected or duplicate representation/algorithm/seed run")
        if set(run["validation"]) != set(profiles):
            raise ValueError("all declared validation profiles must be present")
        for episodes in run["validation"].values():
            for episode in episodes:
                if ("updates" in episode and episode["updates"] != []
                        or "training_behavior" in episode
                        or any(action.get("epsilon") != 0. or "behavior_source" in action
                               for action in episode.get("actions", []))):
                    raise ValueError("representation comparison requires frozen teacher-free validation")
        index[key] = {profile: _episodes(run["validation"][profile], profile=profile, count=validation_cases)
                      for profile in profiles}
    if set(index) != expected:
        raise ValueError("incomplete representation/algorithm/seed grid; failed runs cannot be omitted")

    risks, paired, seed_rows = [], [], []
    for profile in profiles:
        identifiers = [f"{profile}:validation:{index}" for index in range(validation_cases)]
        reference = index[(VARIANTS[0], algorithms[0], agent_seeds[0])][profile]
        for records_by_profile in index.values():
            records = records_by_profile[profile]
            if any(records[key]["fingerprint"] != reference[key]["fingerprint"]
                   or records[key]["seeds"] != reference[key]["seeds"] for key in identifiers):
                raise ValueError("unpaired validation fingerprints or exogenous seeds across representations/models")
        for algorithm in algorithms:
            seed_deltas = {}
            for seed in agent_seeds:
                variant_costs, variant_risks = {}, {}
                for variant in VARIANTS:
                    costs = [index[(variant, algorithm, seed)][profile][key]["cost"] for key in identifiers]
                    risk = episode_cost_summary(costs, alpha=alpha)
                    variant_costs[variant], variant_risks[variant] = costs, risk
                    risks.append({"representation_variant": variant, "return_scope": "individual",
                        "algorithm": algorithm, "agent_seed": seed, "profile": profile,
                        "episode_cost_risk": risk})
                differences = [{"episode_id": identifier, "linear_v3_cost": linear,
                    "log_snr_v4_cost": logarithmic, "log_minus_linear_cost": logarithmic - linear}
                    for identifier, linear, logarithmic in zip(identifiers, variant_costs["linear_v3"],
                                                               variant_costs["log_snr_v4"], strict=True)]
                mean_delta = math.fsum(item["log_minus_linear_cost"] for item in differences) / validation_cases
                seed_deltas[str(seed)] = mean_delta
                paired.append({"algorithm": algorithm, "agent_seed": seed, "profile": profile,
                    "episode_differences": differences, "mean_log_minus_linear_cost": mean_delta,
                    "cvar_log_minus_linear_cost": variant_risks["log_snr_v4"]["cvar_episode_total_cost"]
                                                   - variant_risks["linear_v3"]["cvar_episode_total_cost"],
                    "var_log_minus_linear_cost": variant_risks["log_snr_v4"]["var_episode_total_cost"]
                                                  - variant_risks["linear_v3"]["var_episode_total_cost"]})
            values = tuple(seed_deltas.values())
            seed_rows.append({"algorithm": algorithm, "profile": profile,
                "seed_mean_differences": seed_deltas,
                "mean_of_seed_differences": math.fsum(values) / len(values),
                "sample_std_of_seed_differences": stdev(values) if len(values) >= 2 else None,
                "trained_agent_seed_pairs": len(values)})

    return {"summary_schema_version": 1, "variants": list(VARIANTS), "return_scope": "individual",
        "algorithms": list(algorithms), "agent_seeds": list(agent_seeds), "profiles": list(profiles),
        "validation_cases": validation_cases, "cvar_alpha": alpha,
        "cost_unit": "dimensionless full-drain physical episode-total cost",
        "model_cost_risks": risks, "paired_variant_differences": paired,
        "conditional_seed_differences": seed_rows, "validation_scene_pairing_verified": True,
        "training_pairing": "producer responsibility; training traces are not inputs to this summarizer",
        "interpretation": [
            "Each risk row is one fixed final model/profile; never pool representations, algorithms or seeds into a policy CVaR.",
            "Costs have equal episode weights, including empty episodes; task counts do not weight cost risk.",
            "Negative log-minus-linear differences favor log representation on these observed matched episodes.",
            "The CVaR difference subtracts per-model CVaRs; it is not CVaR of the episode differences.",
            "Pairing concerns exogenous scenarios/seeds, not identical actions, queues or realized exploration histories.",
            "Seed means/std are conditional on fixed producer traces, not scenario-population uncertainty.",
            "Known shifted profiles and reused seed keys do not provide independent replications.",
            "Few episodes/seeds cannot establish reliable tail improvement, convergence or superiority.",
            "No confidence interval, p-value, significance test, RNG, bootstrap or model/seed selection is performed.",
        ]}


def render_markdown(summary) -> str:
    lines = ["# SNR representation development comparison", "",
        "Each row is one fixed model/profile with equal episode weights. No policy/seed CVaR pooling.", "",
        f"| Profile | Algorithm | Seed | Representation | Episodes | Mean cost | CVaR {summary['cvar_alpha']:g} | Tail episodes |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary["model_cost_risks"]:
        risk = row["episode_cost_risk"]
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['agent_seed']} | {row['representation_variant']} | "
                     f"{risk['episodes']} | {risk['mean_episode_total_cost']:.6f} | "
                     f"{risk['cvar_episode_total_cost']:.6f} | {risk['nominal_tail_mass_in_episodes']:.4f} |")
    lines.extend(["", "Negative paired differences favor log representation on the observed cohort.", "",
        "| Profile | Algorithm | Seed | Mean log minus linear cost | CVaR log minus linear cost |",
        "| --- | --- | ---: | ---: | ---: |"])
    for row in summary["paired_variant_differences"]:
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['agent_seed']} | "
                     f"{row['mean_log_minus_linear_cost']:.6f} | {row['cvar_log_minus_linear_cost']:.6f} |")
    lines.extend(["", "Across-seed summaries are conditional on the fixed training/validation traces.", "",
        "| Profile | Algorithm | Seed pairs | Mean seed difference | Sample std |",
        "| --- | --- | ---: | ---: | ---: |"])
    for row in summary["conditional_seed_differences"]:
        spread = row["sample_std_of_seed_differences"]
        spread_text = "n/a" if spread is None else f"{spread:.6f}"
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['trained_agent_seed_pairs']} | "
                     f"{row['mean_of_seed_differences']:.6f} | {spread_text} |")
    lines.extend(["", *[f"- {note}" for note in summary["interpretation"]], ""])
    return "\n".join(lines)
