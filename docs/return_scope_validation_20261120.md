# Stochastic Return-Scope Ablation: Verification, 2026-11-20

The complete declared [development protocol](return_scope_development.md) was
executed once. This is a paired synthetic development study, not main training,
held-out evidence, or a demonstration that the method beats greedy. User
Vehicles still decide independently from their own received information;
there is no MEC action coordinator or extra actor input.

## Actual execution and checks

```text
python -m experiments.run_return_scope_audit --output-dir reports/return-scope-pilot-v1-20261120
```

The new config fixed two return scopes, the QR/QR-CVaR pair and seeds 11/29,
with 20 nominal training scenes and four validation cases in each of four
profiles. No setting, seed, budget or task distribution was changed after
inspecting results. Selection was the final training episode for every model.

- **8 models**, with matched initial online-weight hashes across scopes.
- **160 training, 128 learned validation and 80 fixed-policy validation
  episodes: 368 complete physical executions**. Fixed policies ran once per
  case/profile, not per trained seed or scope.
- Each model saw **347 offered tasks/transitions**, performed **316 gradient
  updates** and **6 target refreshes**. Training clock ranged from 4135 to
  4185 physical slots; final epsilon was .05 in every run. Greedy evaluation
  used epsilon=0 and did not advance that clock.
- Full validation-state snapshots preserved online/target parameters,
  optimizer, replay/cursor/RNG, exploration RNG, counters and mode flags.
- Every model and fixed policy used matching validation scene fingerprints
  and exogenous seed keys. Actor decisions/queues need not match between
  learned methods; they are the outcomes being compared.
- Every admitted task completed. There were 224 recorded deadline misses
  across repeated learned validation executions; completion is not an on-time
  guarantee, and this count is not an independent failure-probability sample.
- All eight final checkpoints reloaded through their correct scope-specific
  loader with matching model/source/outer-protocol hashes.
- **357 tests passed** in the full CPU learning runtime. Standard-library
  runtime: 357 discovered, **262 passed**, 95 optional learning tests skipped.
  Whitespace checks passed. No test-split trajectory was accessed.

## All observed mean-cost results

Every learned row below represents one fixed final model. Each cell is the
mean of **four actual full-drain episode-total costs**, with equal episode
weights. Fixed-policy rows appear once. Lower is preferable for this metric.
The JSON and generated `summary.md` retain separate per-model CVaR and all
paired episode differences; no seeds are pooled into a policy CVaR.

| Policy / return scope | Agent seed | Nominal | High load | Fast servers | Harsh channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| QR-DQN / individual | 11 | 4.7564 | 24.6144 | 4.9897 | 4.9186 |
| QR-DQN / individual | 29 | 4.7560 | 22.2438 | 4.6186 | 4.5957 |
| QR-DQN + CVaR / individual | 11 | 4.9055 | 21.1122 | 4.8392 | 4.6890 |
| QR-DQN + CVaR / individual | 29 | 5.1382 | 22.4034 | 5.1040 | 5.1482 |
| QR-DQN / team | 11 | 5.0992 | 22.7570 | 5.0114 | 4.7165 |
| QR-DQN / team | 29 | 4.7075 | 20.7585 | 4.6712 | 4.5591 |
| QR-DQN + CVaR / team | 11 | 5.0354 | 23.1396 | 5.1200 | 5.2403 |
| QR-DQN + CVaR / team | 29 | 4.9095 | 21.7920 | 5.3902 | 4.9550 |
| AlwaysMEC | n/a | 5.7898 | 37.1140 | 5.7898 | 5.7898 |
| AlwaysV2V | n/a | 4.5466 | 21.1133 | 4.5034 | 4.5071 |
| MinimumEstimatedCombinedCost | n/a | 4.2261 | 18.7401 | 4.1612 | 4.1803 |
| MinimumEstimatedLatency | n/a | 5.4048 | 19.6899 | 5.3873 | 5.4283 |
| UniformRandom | n/a | 4.8305 | 18.8856 | 4.7801 | 4.4760 |

Team returns had lower observed mean cost in **8 of 16** matched
algorithm/seed/profile comparisons with individual returns. Empirical CVaR
was lower in **12 of 16** such comparisons. These counts are dependent,
descriptive comparisons, not independent wins or significance evidence. At
alpha=.9 with four cases, CVaR is simply the maximum observed episode cost;
the nominal tail mass is only .4 episodes.

The cached combined-cost greedy baseline had lower observed mean cost than
**every learned model in all four profile samples**. The larger mechanics
budget and team-cost signal therefore did not close that gap in this study.
No losing model, profile, seed or admitted task was omitted. The shared-cost
signal did not uniformly help, and this result does not justify switching the
default return scope or choosing a favorable seed automatically.

Do not compare these means directly to the earlier four-training/two-validation
pilot as a learning curve: both the training budget and validation case count
changed. A budget-learning-curve study needs the same evaluation cohort and
predeclared checkpoints, without post-hoc best-checkpoint selection.

## Artifacts and provenance

Ignored local output is `reports/return-scope-pilot-v1-20261120/`, containing
`report.json` (12,330,002 bytes), `summary.md`, and eight final checkpoints in
scope-specific subdirectories. Reports preserve full episode traces, physical
metrics/accounting, scope contracts, all baseline comparisons, frozen settings,
and actual source/runtime provenance. Historical configuration, report and
checkpoint files were not overwritten.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `85fe053860b3a5ea9e547f54a32292a0bebb06872b539c0396d86990e080b403` |
| Outer return-scope protocol | `876e36d9904f283406e5ab7f60a4837570a93a3da61d704ab14911092b8cc8fe` |
| Nested shared learning budget | `64236c2e679444dfa87b8cb73c051f82aa187b9bf6b90868671319fc3ed6c263` |
| Expanded scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Exact report bytes | `78770bfa6a3a995b2569e9eadc827a917fbb154af89411316f9daeeef5595ea1` |

Execution provenance records Git revision
`ef1c2f62dcdc5c7aa46adf6b9105afa9ef2df0ce` and `git_dirty=true`, correctly
identifying the new uncommitted implementation at execution time.

## Interpretation before the next stage

This removes the earlier zero-target-refresh limitation and provides a
controlled return-scope comparison; it does not prove convergence, calibrated
tail quality, global/CVaR optimality or generalization. There are two trained
seeds conditional on one fixed training/validation cohort. Known shifted
profiles reuse seed keys, and team actor samples can be correlated.

A useful next hypothesis is whether the learner needs a more sample-efficient
start from ordinary causal greedy experience. Any such experiment must use
training-only data, declare its variant and total data/update budgets in
advance, and include matched with/without-initialization comparisons for both
neutral and risk-sensitive QR. Ex-post validation branch winners are not
training labels. No demonstration prefill, reward bonus, objective reweighting
or additional training was added after this result.

Calibration, the adapted paper/query-free MEC comparisons and a frozen final
protocol remain open. Main training has not started; it remains planned for
Colab with prior notice to Amir. This completed stage is ready for the requested
commit/push reminder before further implementation.
