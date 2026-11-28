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

`run_return_scope_audit.py` implements the explicit paired stochastic return-
scope protocol in `configs/return_scope_pilot_v1.json`: QR/QR-CVaR x individual/
team returns x seeds11/29, with 20 shared nominal training scenes and four
validation cases per known profile. All controls/hyperparameters are shared;
the current v3 learning JSON remains unchanged. Run with a new output directory:

```text
python -m experiments.run_return_scope_audit --output-dir reports/NEW-return-scope-pilot
```

Checkpoints are saved before frozen greedy validation. Complete model-state
guards, pristine scene pairing, exact offered-task/update budgets and exclusive
outputs are enforced. Fixed policies run once per validation scene, not per
scope/seed. `report.json` includes analytical baselines; `summary.md` compares
per-model risk and paired scope differences. No seed pooling, significance,
test access, adaptive stopping or sweep resume is implemented. This remains
bounded synthetic development, not main Colab training. See
[`../docs/return_scope_development.md`](../docs/return_scope_development.md).

`run_behavior_warmup_audit.py` adds a separately named optional training-
behavior ablation. Its `greedy_warmup` arm uses ordinary causal cached greedy
actions only during the first two of the same 20 training episodes, followed
by ordinary learned epsilon behavior. The standard arm remains unchanged.
QR/CVaR losses/targets, actual costs, masks and learned evaluation stay the same:

```text
python -m experiments.run_behavior_warmup_audit --output-dir reports/NEW-behavior-warmup-pilot
```

Both variants receive identical total task/update budgets and initial weight
streams; no extra teacher episodes or offline updates are added. Teacher
parameters, action source, scheduled-but-unused epsilon and the phase clock
are recorded. Runtime rejects teachers at evaluation. The new protocol/loader
binds behavior variant and checkpoint phase; existing configurations/checkpoints
and default collection remain supported. This experiment does not implicitly
adopt a new final paper method. See
[`../docs/behavior_warmup_development.md`](../docs/behavior_warmup_development.md).

`runtime_preflight.py` checks compatible installed package metadata without
importing learning libraries or starting training. `run_resumable_development.py`
advances one standard-behavior model under an existing fixed development plan:

```text
python -m experiments.runtime_preflight --require-learning
python -m experiments.run_resumable_development --output-dir reports/NEW-session --max-additional-episodes 1
python -m experiments.run_resumable_development --output-dir reports/NEW-session --resume --max-additional-episodes 1
```

Each complete episode has an immutable report/checkpoint/commit; final validation
is frozen and published only after the fixed training budget. Source, expanded
protocols, encoder, scope, seed and exact CPU runtime must match on resume.
Interrupted final writes fail closed; explicit verified-prefix recovery copies
to a fresh directory and preserves original evidence. This path has no teacher,
held-out test, automatic best-model selection or main-training launcher. See
[`../docs/colab_development.md`](../docs/colab_development.md) and the opt-in Colab
notebook. The paper's main protocol is still pending.

`run_training_input_audit.py` inspects a previously saved session's retained
training replay, feature magnitudes, first-layer input contributions and
initial/final cost predictions/targets on the same fixed replay cohort:

```text
python -m experiments.run_training_input_audit --session-dir reports/RECEIVED/session --output reports/NEW-training-input-audit/report.json
```

It validates checkpoint/journal identities, preserves all learner and RNG state,
and excludes validation outcomes from analysis. No new training, sampling,
feature transformation or scale fitting is performed. Archived training
provenance and auditor provenance remain separate. See
[`../docs/training_input_diagnostics.md`](../docs/training_input_diagnostics.md).

`run_snr_representation_audit.py` compares the unchanged linear schema-3
encoder with opt-in normalized-log1p schema 4 under an explicit shared plan:

```text
python -m experiments.run_snr_representation_audit --output-dir reports/NEW-snr-representation-pilot
```

Both QR and QR-CVaR are retrained from paired initial weights, with the same
scenes, task/update budgets and frozen final-model validation. The existing
greedy and other fixed policies remain unchanged. The new outer protocol
records actual per-run encoders separately from the nested learning-v3 budget
template. No fitted statistics, teacher, main training or held-out evaluation
is added. See [`../docs/snr_representation_development.md`](../docs/snr_representation_development.md).

`learning_budget_protocol.py` declares a separate development study in
`configs/learning_budget_v1.json`, bound to expanded synthetic traces in
`configs/development_budget_v1.json`. Two SNR representations x QR/QR-CVaR x
seeds 11/29/47 create 12 continuing chains, each evaluated after 20, 100 and
500 training episodes on the same 20 validation cases in each of four profiles.
The legacy encoder remains the default elsewhere; both representation arms
remain required here. Inspect the plan without constructing trajectories or
a learner:

```text
python -m experiments.run_learning_budget_session --plan
```

`run_learning_budget_session.py` advances only one selected chain, normally in
chunks of at most ten complete training episodes:

```text
python -m experiments.run_learning_budget_session --variant log_snr_v4 --algorithm qr_dqn_cvar --agent-seed 11 --output-dir reports/NEW-budget-study/log_snr_v4/qr_dqn_cvar/seed-11 --max-additional-episodes 10
python -m experiments.run_learning_budget_session --variant log_snr_v4 --algorithm qr_dqn_cvar --agent-seed 11 --output-dir reports/NEW-budget-study/log_snr_v4/qr_dqn_cvar/seed-11 --resume --max-additional-episodes 10
```

The cap counts training episodes only: a call reaching a budget point also
runs its complete 80-episode frozen validation cohort. Source/runtime/protocol,
encoder and model identities are strict resume inputs. Interrupted unpublished
chunks repeat the same scene seeds from the last committed checkpoint; no
scene is skipped or rerolled. Existing journals are never overwritten.

`learning_budget_results.py` runs the five unchanged shared references once
and separately collects a complete grid without learning or scene generation:

```text
python -m experiments.learning_budget_results --study-dir reports/NEW-budget-study --references
python -m experiments.learning_budget_results --study-dir reports/NEW-budget-study --output reports/NEW-budget-study/summary.json
```

Incomplete grids produce progress only. All budgets are predeclared; there is
no validation-driven stopping, extension, best-checkpoint/seed selection or
pooled-policy CVaR. The full plan counts 6000 training, 2880 learned validation
and 400 fixed-policy episodes, 9280 total; the larger study has not been run
during preparation. Its manual one-chain
[`Colab notebook`](../notebooks/colab_learning_budget.ipynb) keeps all action
gates false. Commit/push the verified milestone with Amir's authorization
before obtaining a pinned source ZIP for Colab. This remains synthetic
development; main training, calibration and held-out evaluation are later work.
See [`../docs/learning_budget_development.md`](../docs/learning_budget_development.md).
The local planning and fixture-test record is
[`../docs/learning_budget_preparation_20261121.md`](../docs/learning_budget_preparation_20261121.md).
The received `log_snr_v4` and `linear_v3` QR-CVaR/seed-11 chains each completed
500 training episodes and all 240 declared validation executions. Their
[first-chain verification](../docs/colab_budget_chain_1_20261122.md) and
[linear-chain/input-pairing verification](../docs/colab_budget_chain_2_20261122.md)
preserve archive/source/session identities and confirm all 102 checkpoints.
The 500 training inputs, fixed validation cohort and initial states match.
The [linear risk-neutral QR/seed-11 control](../docs/colab_budget_chain_3_20261122.md)
also passed archive/history/all-51-checkpoint checks at the same budgets, with
effective risk weight zero and inputs/initial states paired to the linear
CVaR chain. Its [matched log-SNR QR control](../docs/colab_budget_chain_4_20261122.md)
also passed all checks, including exact recorded runtime identity after fresh
setup. All four seed-11 combinations and 204 checkpoints are verified, with
matched initial states and inputs across the factor grid. Progress records
are combined after separate checks, retaining null performance curves and
comparisons for the incomplete grid. The third chain's browser disconnection and complete
output on reconnect were reported without rerunning the training cell;
preserved journal integrity does not establish runtime interruption or recovery.
The [log-SNR QR-CVaR/seed-29 audit](../docs/colab_budget_chain_5_20261122.md)
also passed all archive, history and 51-checkpoint checks. Only the learner
seed differs from the matched seed-11 contract: exogenous inputs are shared,
while initial weights and replay RNG differ. A failed
auxiliary cross-runtime bitwise-initialization assumption was diagnosed and
preserved: first-layer differences reach `7.45e-9`, with later layers exact.
Archive tensor fingerprints and exact producer-runtime guards were unchanged;
no training rerun or partial-grid performance comparison was performed.
The [linear QR-CVaR/seed-29 audit](../docs/colab_budget_chain_6_20261122.md)
also passed all archive/history/51-checkpoint checks. Initial producer states
and inputs exactly match the log-SNR counterpart, without a Windows-reference
initialization comparison.

The [seventh-chain audit](../docs/colab_budget_chain_7_20261128.md) verifies
`linear_v3 / qr_dqn / seed 29`: 500 training episodes, all three budget points,
240 validation executions, 104206 training slots, 8930 transitions and 8899
updates. Its archive, recorded history and all 51 checkpoints passed checks.
Inputs and initial producer states match the linear CVaR/seed-29 chain;
only algorithm differs in the nested configuration. Effective risk weight
is zero for this QR control. The final verified
checkpoint is `steps/000052/agent.pt`, with model fingerprint
`0d610b07d24d89db44171c75c4a12ae9b0803164e937a7d68b37765df9ddddb4`.

The [eighth-chain audit](../docs/colab_budget_chain_8_20261128.md) verifies
`log_snr_v4 / qr_dqn / seed 29`: 500 training episodes, the fixed 20/100/500
budget points, 240 validation executions, 103623 training slots, 8930
transitions and 8899 updates. Archive, recorded history and all 51 checkpoints
passed checks. Its final verified checkpoint is `steps/000052/agent.pt`,
with model fingerprint
`fa10b23373c42c22a122d3be40ec15096ee14f3893d2b44b8de6c9f68228378e`.
The four seed-29 factor combinations share exogenous inputs and initial
producer network/optimizer/replay/RNG states; full contracts differ only by
the declared representation/algorithm factors. Effective risk weight is
zero for this QR control. This completed verification of all four factor
combinations at seeds 11 and 29.

The [ninth-chain audit](../docs/colab_budget_chain_9_20261128.md) verifies
`log_snr_v4 / qr_dqn_cvar / seed 47`: 500 training episodes, the fixed
20/100/500 budget points, 240 validation executions, 103341 training slots,
8930 transitions and 8899 updates. Archive, recorded history and all 51
checkpoints passed checks. Its final verified checkpoint is
`steps/000052/agent.pt`, with model fingerprint
`e2badfc2fc26d75a1180bd8b8ec13d36acc473c35c8b3dc1c3fa72d8febdfdc6`.
Full contracts differ from the matched seed-11 and seed-29 CVaR controls only
by learner seed. Exogenous inputs match; initial weights and replay RNG
differ. Effective risk weight remains .5.

The [tenth-chain audit](../docs/colab_budget_chain_10_20261128.md) verifies
`linear_v3 / qr_dqn_cvar / seed 47`: 500 training episodes, the fixed
20/100/500 budget points, 240 validation executions, 104128 training slots,
8930 transitions and 8899 updates. Archive, recorded history and all 51
checkpoints passed checks. Its final verified checkpoint is
`steps/000052/agent.pt`, with model fingerprint
`e7c33dbd1bdb5506b7cce1707431b383b290d3669adf428917b342162aa7840d`.
Inputs and initial producer network/optimizer/replay/RNG states exactly match
the log-SNR CVaR/seed-47 chain; full contracts differ only in the three
representation fields. Against the matched linear CVaR seed-11 and seed-29
controls, only learner seed differs in the contract and inputs are shared.

The [eleventh-chain audit](../docs/colab_budget_chain_11_20261128.md) verifies
`linear_v3 / qr_dqn / seed 47`: 500 training episodes, the fixed 20/100/500
budget points, 240 validation executions, 104170 training slots, 8930
transitions and 8899 updates. Archive, recorded history and all 51 checkpoints
passed checks. Its final verified checkpoint is `steps/000052/agent.pt`,
with model fingerprint
`db76dc634faabea59887d79422a6663946d606c8c84236773b0a38797025c9d6`.
Inputs and initial producer states match the linear CVaR/seed-47 counterpart;
only algorithm differs in the nested configuration, with effective risk
weight zero for this QR control. Against the matched neutral seed-11 and
seed-29 contracts, only learner seed differs and inputs are shared.

The [twelfth-chain audit](../docs/colab_budget_chain_12_20261128.md) verifies
`log_snr_v4 / qr_dqn / seed 47`: 500 training episodes, the fixed 20/100/500
budget points, 240 validation executions, 103497 training slots, 8930
transitions and 8899 updates. Archive, recorded history and all 51 checkpoints
passed checks. Its final verified checkpoint is `steps/000052/agent.pt`,
with model fingerprint
`e232f2c32b120883e79c65bf0c484b72189e5795a4ed8c7c891ba1974e990c9d`.
The four seed-47 factor combinations share exogenous inputs and initial
producer network/optimizer/replay/RNG states; contracts differ only by the
declared representation/algorithm factors. Effective risk weight is zero
for this QR control. All twelve chains and 612 checkpoints are verified,
covering 6000 training episodes and 2880 validation executions. At that
milestone the merged progress status was `awaiting_references`.

The [shared-reference handoff](../docs/colab_budget_references_handoff_20261128.md)
prepares one explicit Colab invocation of the unchanged five-policy reference
runner. An external launcher verifies exact source/runtime/protocol identity,
Drive readiness and the twelve chain-contract identities before creating a
fresh `references` directory. Guard and saved-byte checks used a mocked
evaluator only: zero real evaluation episodes or training updates ran.
The first reported Colab invocation reached an erroneous post-save comparison:
the launcher compared JSON lists with the in-memory tuple-bearing protocol.
This was reproduced using the actual protocol and fixed with exact serializer
byte comparison. A regression fixture checks real protocol tuples, the eleven
pre-execution guards and rejection of altered saved bytes. No scientific
source, greedy policy, objective or runtime identity changed. The already
saved report was retrieved without repeating evaluation. No real episodes
ran locally during the diagnosis, correction or received-report audit.

The [fixed-reference verification](../docs/colab_budget_references_verification_20261128.md)
passed all 400 records, 80 static manifests and cohort pairings with all
twelve chains and 2880 learned validation executions. Recorded costs and
task accounting reconcile; all 12745 offered reference task instances
eventually completed. The received bytes also confirm the original
tuple/list comparison diagnosis. All 9280 planned physical executions are
verified, with 612 learning checkpoints. Current merged progress is
`executions_verified_awaiting_analysis`; full-grid performance collection
and interpretation remain pending, without checkpoint/seed selection or a
greedy-improvement claim. Main training and held-out paper evaluation remain
unstarted; source, physical models and protocols remain unchanged.
