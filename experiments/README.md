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
development scenarios from `configs/development_v2.json`. It has no neural
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
2 agent seeds. The explicit `configs/learning_pilot_v3.json` fixes 4 nominal
training episodes, final-model-only selection, and 2 validation cases in each
profile. It never constructs/evaluates the test split. Install `learning` first:

```text
python -m experiments.run_learning_audit --output-dir reports/learning-pilot-v3
```

Choose an unused output directory; reports/checkpoints are never overwritten.
The report preserves per-model episode risk and separates conditional training
seed variation from scenario uncertainty. There is no pooled-policy CVaR,
best-seed selection, convergence claim, or sweep-resume driver. The core CPU
loop is shared with the old smoke runner in `learning_runtime.py`. See
[`../docs/learning_development_protocol.md`](../docs/learning_development_protocol.md)
for exact budgets, pairing, artifacts, and methodological limitations.

`comparison_policies.py` centralizes fixed-policy suite v3, including
`MinimumEstimatedCombinedCost`. It validates homogeneous static radio/kappa
constants and records matching environment weights. Raw/encoded observations
are version 3, retaining the deadline but removing manual label inputs. Shared
cost model 2 has exactly three terms and no independent deadline penalty;
misses/tardiness remain metrics. Old
v1/v2 learning configurations/checkpoints remain on disk but are rejected by the
current contract. See [`../docs/combined_cost_baseline.md`](../docs/combined_cost_baseline.md).

The canonical control contract is
[`../docs/objective_and_preferences.md`](../docs/objective_and_preferences.md).
QR-CVaR keeps lambda=.5; positive `sigmoid_beta=1` only controls the reported
MEC preference curve. Evaluation still chooses masked argmin J, never samples
from sigmoid values, and never transforms actual costs/TD targets with them.

`summarize_learning_audit.py` reads a completed v3 report without importing
Torch or running training. It writes per-model/per-profile cost risk, task
latency/energy/miss rates, episode-mean workload shares, and training-update
traces to a new directory, preserving input/source/protocol hashes:

```text
python -m experiments.summarize_learning_audit reports/learning-pilot-v3/report.json --output-dir reports/NEW-diagnostics
```

It rejects missing runs, incompatible contracts, mismatched paired scenes,
incomplete drain, and validation updates. Seeds are not pooled into a policy
CVaR; fixed-policy results are not duplicated across agent seeds. The complete
v3 pilot remains development training. Main training must run in Colab, and
Amir must be notified before that milestone starts. See
[`../docs/learning_diagnostics.md`](../docs/learning_diagnostics.md).

`run_action_audit.py` performs offline action comparisons on a declared prefix
of unchanged greedy validation trajectories. Each branch changes one user's
action, fixes other users' simultaneous actions, and then fully drains with
greedy continuation. It preserves cloned state/RNGs and records owner versus
global cost differences. There is no neural dependency, training or test option:

```text
python -m experiments.run_action_audit --profiles nominal high_load --cases 2 --decisions-per-episode 8 --output reports/NEW-action-audit/report.json
```

Ex-post minima use hidden future trajectories for offline diagnosis. They are
not deployable decisions, expected Q values, CVaR estimates, or improvement
bounds. Gains at different forks cannot be added. See
[`../docs/greedy_improvement_plan.md`](../docs/greedy_improvement_plan.md).

`run_team_return_audit.py` verifies an opt-in cooperative-return data interface
while keeping decisions independent in each User Vehicle. Five fixed policies
run through matched individual/team interfaces with identical local inputs,
actions and physics; global ledger costs are counted once, while per-user
team-return intervals may overlap and are never summed as episode cost:

```text
python -m experiments.run_team_return_audit --output reports/NEW-team-return-audit/report.json
```

This interface audit has no neural training, controller at MEC, extra actor
input or test option. Team records have a distinct type/scope and are rejected
by legacy replay; see [`../docs/team_returns.md`](../docs/team_returns.md).

The neural smoke runner now has an explicit `--return-scope team` path:

```text
python -m experiments.run_learning_smoke --return-scope team --algorithm qr_dqn_cvar --episodes 4 --checkpoint artifacts/team/NEW-first.pt
python -m experiments.run_learning_smoke --return-scope team --episodes 2 --resume artifacts/team/NEW-first.pt --checkpoint artifacts/team/NEW-continued.pt
```

The default remains `individual`; the stochastic learning v3 JSON/protocol
remains individual-return only. Team smoke run kind v1/report schema 3 binds
the return contract, source/configuration and seeds, uses team replay schema 1
and checkpoint schema 3, and rejects individual checkpoints/replay. Resume is
only at complete episode boundaries with the exact PyTorch/source/protocol.
The inherited DQN/QR loss, mask/risk/argmin and local actor input are unchanged.
Reports use unique physical team accounting, not summed overlapping replay
costs. This is a bounded mechanics check, not main training or evidence of
beating greedy. See [`../docs/team_learning_validation_20261120.md`](../docs/team_learning_validation_20261120.md).
