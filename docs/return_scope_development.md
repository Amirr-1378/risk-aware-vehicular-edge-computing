# Stochastic Individual/Team-Return Development Ablation

The versioned `return_scope_pilot_v1.json` compares two training signals under
the same physical task objective: individual user costs and the team sum.
Each User Vehicle still independently chooses MEC or a cached SV from its
own received observation. MEC provides information only. This is bounded
synthetic development, not main training or a frozen paper protocol.
The [2026-11-20 execution record](return_scope_validation_20261120.md) contains
the complete observed comparison, test results, artifacts and fingerprints.

## Declared grid and budget

`ReturnScopeProtocol` schema 1 wraps an immutable v3 `LearningProtocol` as a
**shared budget template** and requires both return scopes. The original v3
runner/configurations remain individual-return experiments and are not silently
upgraded. The outer plan binds the exact expanded scenario-plan fingerprint,
controls and both scope labels. All budget, seed and profile keys are validated
before output or scenario generation. Duplicate JSON keys/unknown settings and
missing scope axes are rejected. Historical configurations remain unchanged.

| Setting | Declared development choice |
| --- | --- |
| Training scopes | individual, team |
| Algorithms | QR-DQN, QR-DQN + CVaR |
| Agent seeds | 11, 29 |
| Models | 2 scopes x 2 algorithms x 2 seeds = 8 |
| Training data | nominal, train indices 0–19, once each |
| Validation | indices 0–3 for nominal, high_load, fast_servers, harsh_channel |
| Checkpoint selection | fixed final training episode only |
| Physical/scenario plan | unchanged `development_v2.json`, cost model 2 |
| Task objective | .60*T/D + .25*E_mobile/E_budget + .15*M |
| Architecture / quantiles | 64/64 ReLU / 51 |
| Adam rate / batch / warm-up / replay | .001 / 16 / 32 / 2048 |
| Target refresh / gradient clipping | every 50 updates / norm 10 |
| Physical-slot discount | 1 |
| Risk control | alpha=.9; QR-CVaR lambda=.5; neutral QR effective lambda=0 |
| Exploration | 1 to .05 over 1000 physical training slots |

The 20 training scenes use the declared existing development-plan capacity.
This improves the mechanics budget beyond the earlier four-episode pilot; it
does not establish convergence or an adequate final paper sample size. These
settings are explicit engineering choices, not empirically calibrated norms.
The QR pair isolates return scope and risk distortion with matching heads.
Scalar DQN/Double DQN remain implemented and available in the other audits;
this ablation does not remove them from the eventual comparison design.

All scopes/methods get the same offered task/workload traces and exogenous
channel seeds. Each model starts fresh with its paired seed: actual initial
online-weight hashes are checked across scopes. Initialization, replay and
exploration designs/configuration are shared. Different decisions can produce
different queues, observations, fallback, drain lengths and replay contents.
Those differences are outcomes, not pairing failures.

Training must complete every offered task. Exactly one transition is emitted
per offloading decision/task under the existing kernel, so the producer checks
the total transition budget against the offered tasks and the update budget
against the declared warm-up. A failed scene/model aborts the audit; no seed,
task or poor run is dropped or replaced. Shared physical-slot epsilon schedules
do **not** imply identical epsilon at every later task: action-dependent drain
changes each model's elapsed physical training clock.

## Execution, frozen evaluation and output

```text
python -m experiments.run_return_scope_audit --output-dir reports/NEW-return-scope-pilot
```

Optional `--protocol` and `--scenario-protocol` paths select explicit versioned
files; the CLI has no hidden per-method overrides, test access or automatic
budget extension. Output directories must be new. Reports/checkpoints remain
ignored by Git. There is no resumable sweep driver in this milestone.

The default grid executes 160 training episodes, 128 learned validation
episodes and 80 fixed-policy validation episodes: **368 physical executions**.
Five fixed policies are evaluated once per validation profile/case, not once
per learned seed or return scope. They use matching pristine scene fingerprints
and channel seeds, preserving the shared mandatory query/cache protocol.

Final checkpoints are saved before validation under
`checkpoints/<scope>/<algorithm>-seed-<seed>-final.pt`. Metadata binds outer,
shared-budget, scenario and actual-source fingerprints, scope, selection rule,
initial/final model hashes, counters and training scene identities. Individual
and team checkpoint/replay contracts remain distinct. The runner is an
orchestrator over existing agent/adapter implementations; it does not change
physical progress, cost weights, loss or observation features.

Validation uses epsilon=0, no updates/replay insertions, and an unchanged
training clock. Full before/after snapshots compare online/target tensors,
optimizer moments, replay/cursor/RNG, exploration RNG, counters and network
training-mode flags. A mutation raises an error. Scenario manifests and final
truth metrics are report diagnostics, never actor inputs.

`report.json` contains all model episodes, scope contracts, per-task-derived
physical metrics, individual accounting, team accounting where applicable,
paired baselines, protocol/source/runtime provenance and comparative summaries.
`summary.md` presents the return-scope comparisons; analytical baseline results
are in the JSON. Physical episode cost is always the unique actual task sum,
never a sum of overlapping team actor returns.

## Statistical interpretation

The pure `return_scope_statistics` module uses terminal **episode-total costs**
with equal episode weights, including empty episodes. It checks the complete
scope/algorithm/seed grid, exact case IDs/profile coverage and matching scene
fingerprints/exogenous seeds across every model. Training pairing is checked
by the producer rather than inferred by the summary from validation outcomes.

Risk remains separate for every fixed model and profile. For each matching
algorithm/agent seed/profile, report per-episode `team - individual` cost and
its mean; negative favors team on these observations. Across trained seeds,
report the raw per-seed mean differences, their mean and sample standard
deviation. These describe training-seed variation conditional on one fixed
training/validation cohort. They are not scenario-population uncertainty,
confidence intervals, significance tests, or a pooled-policy CVaR.

At alpha=.9 and four validation episodes, nominal tail mass is **.4 episodes**:
empirical upper CVaR equals the largest episode cost. It cannot support a
reliable tail-improvement claim. The four known shifted profiles reuse seed
keys and are not pooled as independent populations. Team actor views overlap
and replay samples can be correlated; shared costs can increase variance.
Neither convergence nor global/static-CVaR optimality follows from this setup.

All outcomes, including a worse team-return model, must remain reported.
Changing weights, reducing the baseline's information or continuing only losing
methods until they win would change the comparison. No validation-based seed,
checkpoint, early-stop or best-profile selection is performed here.

Main training remains a later Colab milestone and must be announced to Amir
before execution. Calibration, missing paper/query-free MEC comparisons,
adequate independent evaluation and a frozen final protocol remain open.
At the end of this verified stage, remind Amir about commit/push before
advancing; the implementation request does not itself authorize pushing.
