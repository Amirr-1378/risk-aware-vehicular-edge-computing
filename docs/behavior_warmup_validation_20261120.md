# Greedy behavior warmup validation

The optional experiment completed on 2026-11-20. The QR-DQN + CVaR network,
TD/quantile updates, cost weights, alpha/lambda, masks and learned deployment
selector were preserved. Only the first two training episodes in the treatment
arm used causal greedy actions. The result does not support adopting this
setting into the main method: warmup did not consistently improve cost, and
the cached combined-cost greedy baseline retained a lower observed mean.

The [experiment contract](behavior_warmup_development.md) defines the fixed
budgets and method boundaries. No teacher is used at evaluation, no imitation
loss or reward shaping is introduced, and no change was made to `src/`.

## Execution and method checks

```text
python -m experiments.run_behavior_warmup_audit --output-dir reports/behavior-warmup-pilot-v1-20261120
```

The declared grid used standard/greedy_warmup behavior, individual/team
returns, QR/QR-CVaR, and seeds 11/29: **16 models**. Every model used the same
20 nominal training scenes and four validation cases per known profile.

- **320 training, 256 learned validation and 80 fixed-policy validation
  episodes: 656 complete physical executions**. All admitted tasks completed.
- Every model received **347 transitions**, performed **316 optimizer updates**
  and **six target refreshes**, and ended at scheduled epsilon .05.
- Each warmup model collected **35 teacher actions** in the first two scenes.
  The teacher phase ended after **418 physical training slots** and included
  four of the existing 316 updates. The phase added no episodes, transitions,
  offline optimizer passes or additional update budget.
- Teacher actions used only the current local received observation and the
  declared static baseline constants. Actual adapter costs, not projected
  greedy estimates, fed replay. Teacher contact screening did not modify masks.
- Evaluation used the learned masked argmin J at epsilon=0. Full learner-state
  snapshots remained frozen; teacher injection is rejected before evaluation
  reset. All 16 checkpoints reloaded with matching variant, phase, return scope,
  source/protocol metadata and final online-weight fingerprints.
- All eight ordinary-training checkpoint payloads exactly matched the earlier
  return-scope experiment except `run_metadata_json`: online/target weights,
  optimizer moments, replay/cursor/RNG, exploration RNGs, counters, encoder,
  configuration and container contracts were identical. This verifies that the
  original training path was preserved by the implementation changes.
- **390 tests passed** in the full CPU learning runtime. Standard-library
  runtime: 390 discovered, **273 passed**, 117 optional learning tests skipped.
  Tests include teacher rejection in evaluation, exact local inputs, actual-cost
  preservation despite manipulated projected scores, fixed phase/budgets,
  control reproduction, strict checkpoint identity and complete paired grids.

## Observed episode costs

Every cell below is one fixed final model's mean of four full-drain episode
total costs. Seeds remain separate. The standard rows reproduce the earlier
control; all warmup outcomes are included. The generated summary retains
per-model CVaR and all matched episode differences.

| Behavior / return scope / algorithm | Seed | Nominal | High load | Fast servers | Harsh channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard / individual / QR | 11 | 4.7564 | 24.6144 | 4.9897 | 4.9186 |
| Standard / individual / QR | 29 | 4.7560 | 22.2438 | 4.6186 | 4.5957 |
| Standard / individual / QR-CVaR | 11 | 4.9055 | 21.1122 | 4.8392 | 4.6890 |
| Standard / individual / QR-CVaR | 29 | 5.1382 | 22.4034 | 5.1040 | 5.1482 |
| Standard / team / QR | 11 | 5.0992 | 22.7570 | 5.0114 | 4.7165 |
| Standard / team / QR | 29 | 4.7075 | 20.7585 | 4.6712 | 4.5591 |
| Standard / team / QR-CVaR | 11 | 5.0354 | 23.1396 | 5.1200 | 5.2403 |
| Standard / team / QR-CVaR | 29 | 4.9095 | 21.7920 | 5.3902 | 4.9550 |
| Warmup / individual / QR | 11 | 4.8961 | 21.3675 | 4.8758 | 4.6964 |
| Warmup / individual / QR | 29 | 5.0762 | 23.6878 | 5.0803 | 5.1113 |
| Warmup / individual / QR-CVaR | 11 | 5.3300 | 25.8956 | 5.0604 | 5.0542 |
| Warmup / individual / QR-CVaR | 29 | 4.9246 | 20.9434 | 4.7449 | 4.8686 |
| Warmup / team / QR | 11 | 4.7979 | 20.5124 | 5.0596 | 5.0438 |
| Warmup / team / QR | 29 | 5.0881 | 24.6799 | 5.3323 | 5.1782 |
| Warmup / team / QR-CVaR | 11 | 4.9700 | 23.1838 | 5.2051 | 4.9762 |
| Warmup / team / QR-CVaR | 29 | 4.9466 | 21.8116 | 4.9434 | 5.0073 |
| AlwaysMEC | n/a | 5.7898 | 37.1140 | 5.7898 | 5.7898 |
| AlwaysV2V | n/a | 4.5466 | 21.1133 | 4.5034 | 4.5071 |
| MinimumEstimatedCombinedCost | n/a | 4.2261 | 18.7401 | 4.1612 | 4.1803 |
| MinimumEstimatedLatency | n/a | 5.4048 | 19.6899 | 5.3873 | 5.4283 |
| UniformRandom | n/a | 4.8305 | 18.8856 | 4.7801 | 4.4760 |

Warmup had lower observed mean in **12 of 32** paired scope/algorithm/seed/
profile comparisons, and lower empirical CVaR in **13 of 32**. These dependent
counts are descriptive, not significance tests or independent win rates. With
four cases and alpha=.9, empirical CVaR equals the largest observed episode
cost. Two trained seeds cannot establish reliable population tail quality.

The cached combined-cost greedy policy had lower observed mean in **all 64**
learned model/profile cells. This two-episode behavior prefix therefore did
not close the measured mean-cost gap. No budget, seed, cost weight or teacher
duration was changed after inspecting the result. Other teacher schedules are
not ruled out by this experiment, but no favorable schedule is selected here.

## Artifacts and reproducibility

The ignored local directory `reports/behavior-warmup-pilot-v1-20261120/`
contains `report.json` (22,483,613 bytes), `summary.md` and 16 final checkpoints
under variant/scope subdirectories. The JSON includes all episodes, actual
accounting, action sources, scheduled-but-unused teacher epsilon, teacher
parameters/boundary/action counts, baselines and paired summaries. Existing
reports, checkpoint files and historical configuration JSON were preserved.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `daabc297d718166fc8a739c33decc05344b9b61c9eccf9cf21570cde64b68cc5` |
| Behavior warmup protocol | `cb02c56df37cc3ab7ea9cefff1053f6224482df6b7300b681f35f41afcbca2e4` |
| Nested paired protocol | `876e36d9904f283406e5ab7f60a4837570a93a3da61d704ab14911092b8cc8fe` |
| Shared learning budget | `64236c2e679444dfa87b8cb73c051f82aa187b9bf6b90868671319fc3ed6c263` |
| Physical scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Exact report bytes | `577d91b0980a18d3511118c30f4df7d59d6028a8a07909310583841eae6f483d` |

Execution provenance records Git revision
`983a46b359ff29123f9e4f575ca745a2cfba7e99` and `git_dirty=true`, identifying
the uncommitted experimental implementation used for this run.

## Method decision

The main method retains QR-DQN + CVaR with independent local actors and the
approved physical cost and deployment selector. The optional collection
variant is retained as a transparent development experiment, without automatic
adoption into the final method. Further development should preserve that core
and use declared learning-budget or calibration studies rather than changing
the objective or weakening greedy to obtain a win. No convergence or exact
global/static-CVaR claim follows from this partially observed simulation.

No held-out test trajectory or main training was run. Main training remains
planned for Colab with notice before it starts. This stage is ready for the
requested commit/push reminder before advancing.
