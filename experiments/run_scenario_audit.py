"""Paired stochastic development-scenario audit, not a final paper benchmark."""

import argparse
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

from risk_aware_v2x.agents import DecisionAdapter, FeatureScales, ObservationEncoder
from risk_aware_v2x.baselines import AlwaysMEC, AlwaysV2V, MinimumEstimatedLatency, UniformRandom
from risk_aware_v2x.entities._validation import require_positive_int
from risk_aware_v2x.environment.evaluation import terminal_metrics
from risk_aware_v2x.models.randomness import RandomStreams
from risk_aware_v2x.risk.statistics import episode_cost_summary, paired_mean_interval
from risk_aware_v2x.scenarios import generate_scenario
from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
from .provenance import source_provenance

DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "configs" / "development_v1.json"


def save_audit_report(report: dict, path: str | Path) -> None:
    """Save finite JSON exclusively; never replace a user's existing report."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(payload)


def run_scenario_audit(*, protocol: DevelopmentProtocol | None = None, profile: str = "nominal",
                       split: str = "validation", cases: int = 2,
                       allow_test: bool = False) -> dict:
    require_positive_int("cases", cases)
    if not isinstance(allow_test, bool):
        raise TypeError("allow_test must be boolean")
    if split == "test" and not allow_test:
        raise ValueError("test split is closed by default; do not use test data for calibration")
    plan = protocol if protocol is not None else DevelopmentProtocol.load(DEFAULT_PROTOCOL)
    spec = plan.profile(profile)
    # Validate the complete selected prefix before constructing any scene.
    keys = [plan.episode_seeds(split, index) for index in range(cases)]
    records, scenes = {}, []
    for episode in keys:
        scene = generate_scenario(spec, seed=episode.scenario_seed)
        identifier = f"{profile}:{split}:{episode.index}"
        fingerprint = scene.fingerprint
        scenes.append({"episode_id": identifier, "seeds": asdict(episode),
                       "fingerprint": fingerprint, "manifest": scene.manifest()})
        policies = {"always_mec": AlwaysMEC(), "always_v2v": AlwaysV2V(),
                    "minimum_estimated_latency": MinimumEstimatedLatency(),
                    "uniform_random": UniformRandom(seed=episode.policy_seed)}
        for name, policy in policies.items():
            environment = scene.environment()
            encoder = ObservationEncoder(max_server_vehicles=spec.max_server_vehicle_slots, scales=FeatureScales())
            adapter = DecisionAdapter(environment, encoder, gamma_per_slot=1.)
            adapter.reset(seed=episode.channel_seed)
            while not environment.terminated:
                actions = {user_id: policy.act(observation)
                           for user_id, observation in sorted(adapter.raw_observations.items())}
                adapter.step(actions)
            accounting = asdict(adapter.accounting)
            metrics = terminal_metrics(environment)
            if (not math.isclose(math.fsum(accounting["total_cost"].values()), metrics["total_cost"],
                                 rel_tol=1e-12, abs_tol=1e-12)
                    or scene.fingerprint != fingerprint):
                raise RuntimeError("cost conservation or pristine scenario invariant failed")
            for user_id, total in accounting["total_cost"].items():
                if not math.isclose(total, accounting["prefix_cost"][user_id]
                                    + accounting["closed_transition_cost"][user_id],
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError("prefix/transition cost mismatch")
            records.setdefault(name, []).append({
                "episode_id": identifier, "scenario_fingerprint": fingerprint,
                "elapsed_slots": environment.slot, "metrics": metrics, "accounting": accounting,
            })
    summaries = {name: episode_cost_summary([entry["metrics"]["total_cost"] for entry in episodes],
                                          alpha=plan.cvar_alpha)
                 for name, episodes in records.items()}
    reference = {entry["episode_id"]: entry["metrics"]["total_cost"] for entry in records["always_mec"]}
    intervals = {}
    if cases >= 2:
        for name, episodes in records.items():
            if name == "always_mec":
                continue
            intervals[name] = asdict(paired_mean_interval(
                {entry["episode_id"]: entry["metrics"]["total_cost"] for entry in episodes}, reference,
                rng=RandomStreams(plan.master_seed).stream(f"bootstrap:{profile}:{split}:{name}"),
                resamples=plan.bootstrap_resamples, confidence=plan.bootstrap_confidence,
            ))
    return {
        "report_schema_version": 1, "purpose": "stochastic development audit; no calibrated performance claim",
        "protocol": asdict(plan), "protocol_sha256": plan.fingerprint,
        "profile": profile, "split": split, "cases": cases,
        "python_version": platform.python_version(), "provenance": source_provenance(),
        "scenarios": scenes, "policies": records,
        "episode_total_cost_summaries": summaries,
        "paired_mean_cost_difference_vs_always_mec": intervals,
        "warning": "small-sample descriptive pilot; paired baseline episode intervals are not learned-agent seed uncertainty",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("choose a new report filename; existing results are never overwritten")
    report = run_scenario_audit(protocol=DevelopmentProtocol.load(args.protocol), profile=args.profile,
                                split=args.split, cases=args.cases, allow_test=args.allow_test)
    if args.output is not None:
        save_audit_report(report, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
