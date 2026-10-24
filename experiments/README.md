# Experiments

Experiment entry points and versioned configurations will live here.
Generated figures, checkpoints, and reports are intentionally ignored by Git.

`run_environment_smoke.py` is a synthetic integration demonstration with two
fixed rules (`mec` and `first_cached_server`). It prints a reproducibility
manifest and complete task ledgers as JSON. It is not a calibrated benchmark
or an algorithm-performance claim. Run it from the repository root with the
package installed, or set PowerShell's `$env:PYTHONPATH = "src"` first:

```text
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
```

`run_baseline_smoke.py` runs AlwaysMEC, AlwaysV2V, UniformRandom, and the
supplementary MinimumEstimatedLatency rule on the same synthetic scenario.
It audits the learning adapter's raw and slot-discounted prefix/transition
decomposition and records explicit encoder configuration and independent seeds:

```text
python -m experiments.run_baseline_smoke --seed 7 --policy-seed 17 --gamma-per-slot 1
```

No learned policy or statistical performance claim is included. See
[`../docs/learning_interface.md`](../docs/learning_interface.md) for limitations
and the distinction between minimum latency and combined-cost control.

`run_learning_smoke.py` trains/evaluates an optional CPU neural agent and can
save/resume complete episode boundaries. It is also an integration audit, not
a calibrated benchmark. Install the `learning` dependencies first:

```text
python -m experiments.run_learning_smoke --algorithm qr_dqn_cvar --episodes 4
python -m experiments.run_learning_smoke --episodes 2 --checkpoint artifacts/learning/first.pt
python -m experiments.run_learning_smoke --episodes 2 --resume artifacts/learning/first.pt --checkpoint artifacts/learning/continued.pt
```

Choose `dqn`, `double_dqn`, `qr_dqn`, or `qr_dqn_cvar`. Reports distinguish
training updates from frozen greedy evaluation and preserve complete costs,
including pre-decision prefixes and variable drain. Separate seed labels in
this deterministic smoke scene are not independent stochastic evaluation
trials. See [`../docs/neural_agents.md`](../docs/neural_agents.md).

`run_scenario_audit.py` evaluates five analytical policies on shared stochastic
development scenarios from `configs/development_v1.json`. It has no neural
dependency. The default split is validation; test access is closed unless
explicitly enabled. No offered task or difficult episode is silently removed.

```text
python -m experiments.run_scenario_audit --profile nominal --cases 2
python -m experiments.run_scenario_audit --profile high_load --cases 2 --output reports/development-high-load-v1.json
```

Other profiles are `fast_servers` and `harsh_channel`. Outputs include scenario
manifests, paired seeds, full-drain metrics, episode-level cost risk, and paired
mean-cost bootstrap intervals. These uncalibrated small development samples
are not learned-policy, standards-compliance, or final paper evidence. See
[`../docs/development_protocol.md`](../docs/development_protocol.md).

`run_learning_audit.py` connects the stochastic plan to 4 neural algorithms x
2 agent seeds. The explicit `configs/learning_pilot_v2.json` fixes 4 nominal
training episodes, final-model-only selection, and 2 validation cases in each
profile. It never constructs/evaluates the test split. Install `learning` first:

```text
python -m experiments.run_learning_audit --output-dir reports/learning-pilot-v2
```

Choose an unused output directory; reports/checkpoints are never overwritten.
The report preserves per-model episode risk and separates conditional training
seed variation from scenario uncertainty. There is no pooled-policy CVaR,
best-seed selection, convergence claim, or sweep-resume driver. The core CPU
loop is shared with the old smoke runner in `learning_runtime.py`. See
[`../docs/learning_development_protocol.md`](../docs/learning_development_protocol.md)
for exact budgets, pairing, artifacts, and methodological limitations.

`comparison_policies.py` centralizes fixed-policy suite v2, including the new
`MinimumEstimatedCombinedCost`. It validates homogeneous static radio/kappa
constants and records matching environment weights. Raw/encoded observations
are version 2 with the original task deadline exposed to all policies. Old
v1 learning configuration/checkpoints remain on disk but are rejected by the
current contract. See [`../docs/combined_cost_baseline.md`](../docs/combined_cost_baseline.md).
