# Multi-Seed Neural Development Audit v1

Status: implemented bounded stochastic training, fixed-final-model validation,
matched analytical comparisons, and explicit training-seed diagnostics.
**This is an uncalibrated pipeline pilot, not evidence of convergence, safety,
tail-risk improvement, or paper readiness.** Physical assumptions and their
limitations remain those of [development_protocol.md](development_protocol.md).

## Versioned contract and development budget

The checked-in `experiments/configs/learning_pilot_v1.json` binds to the exact
canonical SHA256 of `development_v1.json`'s expanded scenario plan. A changed
physical plan, undeclared agent seed/profile, excessive split prefix, or
inconsistent CVaR confidence is rejected before scene construction or output.
The learning plan has its own canonical hash. Reports preserve both expanded
plans, encoder/scales, scenario manifests, seeds, actual source hash, Git
revision/dirty status, package versions, and CPU/runtime settings.

| Pilot choice | Value | Meaning/status |
| --- | --- | --- |
| Algorithms | DQN, Double DQN, QR-DQN, QR-DQN + CVaR | existing tested cost-minimizing implementations |
| Agent seeds | 11, 29 | declared subset of development-plan seeds, not selected by outcomes |
| Training | nominal, train indices 0–3, once each | same ordered offered traces and channel seeds for every run |
| Validation | indices 0–1 of each of four profiles | nominal, high_load, fast_servers, harsh_channel |
| Selection | final training episode only | no validation early stopping, best checkpoint, or best seed |
| Hidden layers / quantiles | 64/64 ReLU; QR N=51 | synthetic pilot architecture, not tuned norms |
| Adam learning rate / clip | .001 / gradient norm 10 | shared across algorithms |
| Batch / warm-up / replay | 16 / 32 transitions / 2048 | uniform replay, no prioritized sampling |
| Target update | every 50 gradient updates | common hard-copy schedule |
| Huber threshold / slot gamma | 1 / 1 | undiscounted finite full-drain learning costs |
| Exploration | 1 to .05 over 1000 training slots | same physical-slot rule, including empty slots and drain |
| QR-CVaR control | alpha=.9, risk weight=.5 | neutral QR uses effective risk weight zero |

These are bounded engineering settings for exercising the training pipeline,
not literature-derived optimal hyperparameters. Four training episodes are
not intended to produce a converged agent. The parent plan's 20 training
indices and five agent seeds are available capacities, not evidence that all
have already been trained. A different budget requires a new explicit learning
configuration/hash; the command has no hidden per-method budget overrides.

Only the algorithm selector changes in the shared agent template. Scalar
DQN heads naturally have fewer outputs/parameters than N-atom QR heads;
this is **not** a parameter-count-matched comparison. The neutral QR/CVaR QR
pair has the same architecture and initialization stream for a matching seed,
making it the direct risk-control ablation. Alpha/risk_weight template fields
are inactive for scalar methods and risk_weight is inactive for neutral QR.
The existing QR cost-Bellman/quantile update is unchanged; see
[neural_agents.md](neural_agents.md) and [Dabney et al.](https://arxiv.org/abs/1710.10044).

## Execution, pairing, and information boundaries

There is one fresh shared-user agent per `(algorithm, agent_seed)` pair.
Replay, network initialization, exploration streams, and physical training
clock reset at that boundary. Training uses only `train` episodes. Every
episode starts from a fresh deep copy of the same pristine scenario prototype.
Scenario/channel streams do not depend on algorithm, agent seed, or iteration
order. Agent seeds vary initialization, exploration, and replay sampling;
the offered training/validation traces are intentionally fixed across seeds.

The common budget is **the same finite arrival traces/episodes**, not equal
wall time or necessarily equal drained slot counts. One update is requested
per newly eligible transition, exactly as in the existing smoke audit.
All admitted tasks must drain; no guard failure or difficult seed is discarded,
replaced, retried with a new seed, or silently excluded from a summary. Since
the current kernel emits one transition per offloading decision/task, the
same finite admitted tasks also yield matching transition/update counts under
these settings. The epsilon schedule depends on each run's actual elapsed
training slots, so identical schedule parameters do not imply identical
epsilon at every later task when drain durations differ.

After the last training episode, save the final checkpoint (if output is
enabled) **before** any validation. Evaluation is greedy (epsilon=0), does
not insert into replay, update either network/optimizer, consume exploration/
replay RNG, or advance the training clock. Runtime guards verify unchanged
online weights and replay/counter sizes; tests additionally check target,
optimizer, replay contents, and exploration states. The policy receives only
the existing encoded cached observations/masks. Scenario manifests, future
arrivals, and terminal truth metrics are report-only diagnostics.

AlwaysMEC, AlwaysV2V, UniformRandom, and MinimumEstimatedLatency are evaluated
once per validation case/profile, not duplicated and counted as independent
across agent seeds. Their scene fingerprints and channel seed keys match the
learned methods. They still use the common mandatory query/cache protocol.
MinimumEstimatedLatency remains latency-only, not a combined-cost greedy
baseline. Query-free MEC, combined-cost greedy, and the adapted paper
best-response baseline are not implemented by this stage.

The shifted profiles reuse the same validation seed keys for controlled
development comparisons; do not pool them as independent replications. Models
are trained only on nominal traffic here. These are known development shifts,
not an untouched out-of-distribution test benchmark. There is no `--test` or
`--allow-test` option in this runner: no test scenario is constructed or
evaluated. Loading the parent plan still validates the **seed identifiers**
of all declared splits; that does not inspect held-out trajectories/results.
Validation results may inform later development, but are not held-out claims.

## Statistical units: keep the two axes distinct

For each **fixed trained model and profile**, descriptive risk is computed
over validation episode-total combined costs (full drain, including query
prefix and fallback waste). Every episode has equal weight, including empty
episodes. The pilot has two episodes per model/profile: at alpha=.9, nominal
tail mass is only .2 episode and empirical CVaR is the observed maximum.
This is not a credible rare-event estimate. Agent conditional mean-CVaR
action scores remain distinct from empirical episode risk; the controller
does not claim exact static/nested-CVaR optimality.

Let C[a,s,e,p] denote episode cost for algorithm a, training seed s,
validation episode e, and profile p. First form

`m[a,s,p] = sum_e C[a,s,e,p] / E`.

Across S agent seeds report `mean_s m` and the sample standard deviation
`sqrt(sum_s (m - mean_s m)^2 / (S-1))`, undefined/null if S=1. Do not treat
S*E cells, users, or tasks as independent runs. Per-model episode mean/VaR/
CVaR remains available separately; **no CVaR pooled across trained policies**
and no average-of-conditional-CVaRs-as-mixture-CVaR is reported.

For a mean-cost contrast (algorithm minus AlwaysMEC, or QR-CVaR minus neutral
QR), resample matching **training-seed** keys jointly. Each key contributes
one difference of validation means. The existing paired percentile bootstrap
is reused; baseline validation means are fixed constants across seed rows.
Negative differences favor the first method. The report labels these
intervals as conditional on the particular training and validation traces.
They capture initialization/exploration/replay training variation, **not**
uncertainty over new scenario populations, shared finite training sets,
measurement/calibration uncertainty, CVaR, or multiple-comparison error.
S=1 yields no interval; S=2 gives a very coarse descriptive pilot interval,
not statistical evidence of superiority. Profiles are always separate.

Reporting run variation is motivated by
[Agarwal et al.](https://arxiv.org/abs/2108.13264). This limited conditional
seed analysis is not presented as the full rliable evaluation procedure.
A calibrated final experiment needs a separately specified scenario/run
uncertainty design, larger independent budgets, and tail-sample justification.

## Run and preserve artifacts

Install the optional learning dependencies, then from the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m experiments.run_learning_audit --output-dir reports/learning-pilot-v1
```

This runs 8 trained models (4 algorithms x 2 seeds), 32 training episodes,
64 learned validation episodes, and 32 fixed-policy validation episodes.
The output directory must not already exist, even if empty. It receives
`report.json` and 8 `checkpoints/<algorithm>-seed-<seed>-final.pt` files.
Omitting `--output-dir` prints a report without saving checkpoints. Runtime
artifacts remain ignored by Git; source/configuration/tests/docs are tracked.
Failures propagate rather than producing a success report. Completed final
checkpoints from a partially failed sweep remain on disk for diagnosis;
there is no automatic retry, partial-grid aggregation, or sweep-resume driver.

Final checkpoints use the existing complete agent-state format and record
source/scenario/learning hashes, selection rule, training clock, completed
episode count, training-scene fingerprints, and actual online-weight hash.
The core loader supports trusted local checkpoints, but this audit CLI does
not implement `--resume`; the old deterministic smoke resume contract cannot
be substituted. Changing source/configuration/encoder invalidates exact
continuation provenance; existing checkpoints/reports are never overwritten.

The common CPU loop is now `experiments/learning_runtime.py`; both smoke and
stochastic entry points use it. Neural algorithm, physical, environment, and
analytical-baseline code are unchanged. Tests cover final-checkpoint metadata,
all four algorithms, seed/profile order invariance, matched scenes, full cost
conservation, evaluation immutability, fixed budgets, no test execution,
empty episodes, conditional-seed formulas, failure propagation, protocol
rejection, global RNG isolation, and exclusive artifacts.

## Local verification snapshot (2026-10-22)

The checked-in pilot configuration completed 8 models and all 128 planned
training/validation executions. Each model ingested 77 transitions and made
46 gradient updates. All admitted tasks completed in every execution; this is
an observed pilot check, not a universal deadline/completion theorem. No test
scenario was constructed/evaluated. All 8 saved checkpoints were reloaded and
their weight and metadata hashes matched the saved report. Full local tests:
234 passed with learning dependencies; the standard-library runtime also
passed, explicitly skipping 48 optional neural tests.

Run provenance: Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.5.4, one CPU thread;
source SHA256 `a45395a10facbe96785986ecb425f231b8c568dd927df70ac29835291f1baee5`;
learning-plan SHA256 `ae2c93afe3da38dc59678b58b90da43ce4fe0bdbdaa3e96aabe74a6669742526`.
The 1,912,386-byte report is local at `reports/learning-pilot-v1/report.json`
with its final checkpoints, all ignored by Git. No convergence, comparative
improvement, or reliable CVaR estimate is inferred from these counts.

Next: inspect pilot diagnostics for implementation errors without tuning to
make baselines lose; calibrate physical/workload assumptions with defensible
sources/traces; add remaining scientific baselines; then freeze a larger
multi-seed protocol and its validation/held-out-test/uncertainty rules before
expensive training or final test access.
