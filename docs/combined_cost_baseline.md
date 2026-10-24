# Cached Plug-In Combined-Cost Baseline and Input Schema v2

Status: `MinimumEstimatedCombinedCost` is implemented as a deterministic
**myopic point-estimate heuristic**. It is project-defined from the approved
objective, not a reproduction of a published expected-cost optimizer or the
paper best-response baseline. It does not estimate a conditional cost
distribution, compute CVaR, or anticipate all future fallback/queue effects.

## Inputs and version boundary

Raw observation schema 2 adds `deadline_s`: the requesting task's original
relative deadline D in seconds. This is already known by its owner, not a
new observation of server truth. The vector encoder also exposes it as
`task.deadline / FeatureScales.time_s`, increasing dimension from `38+15*M`
to `39+15*M`. Both raw and encoded schemas are now version 2. No existing
feature was removed, although later vector indices shift by one.

D must not be reconstructed as `slack/(1-elapsed_fraction)`: at the exact
deadline both numerator and denominator are zero. Using remaining slack as
the latency denominator would silently change the approved objective.
Baselines and neural methods now both receive D explicitly. Replay rejects
encoded schema 1; checkpoint loading rejects the old encoder signature.
Old checkpoint/report/configuration files are preserved, never reshaped,
overwritten, or silently treated as current models. Training new models is
necessary for the new feature schema; no prior performance result transfers.

The physical scenario plan `development_v1.json` is unchanged. The new
`learning_pilot_v2.json` explicitly records learning schema 2, raw schema 2,
encoder schema 2, and baseline-suite version 2. The previous
`learning_pilot_v1.json` is retained as historical provenance and rejected
by the current learning-plan loader. Version 2 keeps the same finite pilot
training budget; it is not a new calibrated paper protocol.

## Cached service projection (seconds)

For input/output sizes B_in/B_out in bits, required computation C in cycles,
cached rates R_up/R_down in bit/s, and cached available CPU f in cycle/s:

`T_hat = B_in/R_up + W_hat + C/f + B_out/R_down`.

For MEC, W_hat is its supplied cached queue-delay estimate, used exactly
once. For an SV, `W_hat = cached_queue_cycles/f`. Thus the SV projection is
conservative relative to criticality-first EDF reordering but does not add
the queue twice. This is the same service estimator used by the latency-only
rule. CPU free at snapshot time is only a proxy for eventual task allocation;
it is not a prediction of future shared CPU frequency or a stability theorem.

Zero service rate/CPU with positive work gives an infinite unavailable
estimate. Zero output needs zero downlink time even if its rate is zero.
The continuous estimate omits slot-boundary phase gates, future arrivals,
contention, and fading. An infinite estimate is not an actual elapsed-time
state or a scientific cost sample.

## Estimated future mobile energy (joules)

With `P_u_tx = public_user_max_tx_W * observed_tx_power_fraction`:

`E_hat_MEC = P_u_tx * T_up + P_u_rx * T_down`,

`E_hat_SV = E_hat_MEC + P_sv_rx * T_up + P_sv_tx * T_down + kappa * C * f**2`.

Only the task's C cycles enter estimated CPU energy; queued work does not.
MEC CPU/radio energy is excluded, as in the objective. User/server radio and
SV dynamic CPU energy are included. There is no arbitrary waiting-energy
term: the current kernel charges radio busy time and query processing idle,
not user idle energy throughout remote compute/queue waiting.

The suite factory freezes public **static** max-TX/RX powers and SV kappa
from scenario prototypes, and validates a homogeneous radio/kappa cohort.
It reads no battery state, queue, position, mobility, future trace, current
allocation, or actual completion outcome for the policy's action. Objective
weights come directly from the common environment configuration, including
sensitivity changes; they cannot silently differ for this baseline. These
physical-model constants are declared experiment assumptions, not fitted
on validation outcomes. In the nominal development factory they are 1 W TX,
.5 W RX, and kappa=1e-28 J/(cycle Hz^2); the old smoke scene uses kappa=1e-27.
Their realism is still uncalibrated. Heterogeneous fixed radio/kappa values
raise an error rather than silently using one node's constants for others.

## Remaining, action-dependent score

For own current deadline slack d, task energy budget E_budget, criticality
weight w, and shared weights `(w_T,w_E,w_M,p_D)`:

`new_miss = 1 if 0 <= d < T_hat else 0`,

`score(a) = w * [w_T*T_hat/D + w_E*E_hat/E_budget + w_M*1(a==MEC) + p_D*new_miss]`.

Completion at exactly d is not a predicted miss. If d<0 the actual one-time
deadline penalty has already been charged by the kernel and is not charged
again in this **remaining increment**. At d=0, positive future service does
predict a new miss. Criticality scales the per-task score but does not alter
physical energy. It alone does not reorder different users' future actions;
the common simulator still applies criticality-first EDF.

Already incurred admission/query latency and energy are sunk and independent
of the current action, so they are omitted from its ranking. They remain in
full episode metrics/prefix accounting. No actual query cost is erased from
the simulation, and no extra surrogate fallback penalty is introduced.
For an SV candidate, the estimated MEC-use term is zero, but eventual MEC
recovery remains possible and is charged by the actual simulator.

The v2 suite enables a cached-contact guard: for a server record age A and
its predicted contact duration H at generation, `H_remaining=max(0,H-A)`.
Consider an SV only if `T_hat <= H_remaining`. This is a **heuristic
constraint**, not a success probability, ground-truth mask, or guaranteed
contact/deadline check. It uses no true trajectories after the snapshot.
The shared action mask is unchanged. The class also supports an explicit
`require_cached_contact=False` diagnostic variant; the comparison suite's
chosen setting is recorded in every report, not changed after seeing results.

Unavailable service or a failed contact guard receives score infinity,
regardless of zero objective weights; this avoids `0*inf -> NaN`. Ties use
the smallest observed action index. MEC is retained even when its current
free-CPU estimate is zero, so all-infinite projections resolve to MEC. The
kernel's eventual MEC recovery/completion assumptions are unchanged. The
baseline has no fallback probability model, multi-user game/best response,
or long-term queue control, so its score must not be called an exact expected
future cost, one-slot Bellman value, or optimal risk-aware policy.

## Comparison integration and reproducibility

Version 2 of the fixed-policy suite contains AlwaysMEC, AlwaysV2V,
UniformRandom, MinimumEstimatedLatency, and MinimumEstimatedCombinedCost.
The standard-library analytical smoke/scenario audits and optional neural
audit use the same factory. Reports include suite version, frozen parameters,
query protocol, encoder, source/configuration fingerprints, and seeds.
Report schemas for those comparison runners are now version 2.

The learning audit additionally reports paired training-seed mean-cost
contrasts against the combined-cost rule, not only AlwaysMEC, while keeping
neutral QR versus CVaR QR. Intervals retain their conditional fixed-trace
scope; they are not evidence of tail-risk reduction or generalization.

All five policies retain the same mandatory query/cache protocol and common
fallback. **Query-free MEC is still missing**, so current AlwaysMEC is not
the strongest possible end-to-end MEC-only protocol. The paper best-response
baseline also remains missing; it requires source-specific equations and a
documented action-space adaptation. No method in this stage is labeled as
that paper baseline. Empirical physical/workload calibration, both remaining
comparisons, larger training budgets, and final held-out protocols remain
necessary before publication-level conclusions.

Example development runs (unused output destinations only):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m experiments.run_scenario_audit --profile nominal --cases 2 --output reports/baseline-suite-v2-nominal.json
.\.venv\Scripts\python.exe -m experiments.run_learning_audit --output-dir reports/learning-pilot-v2
```

The latter runs 32 training, 64 learned-validation, and 40 fixed-validation
episodes (136 total); it never constructs/evaluates test scenarios. Existing
`reports/learning-pilot-v1` artifacts remain untouched. Reports/checkpoints
stay ignored by Git; actual outcomes, not heuristic score infinities, are
serialized as finite JSON. Per-policy risk remains episode-based, not pooled
over correlated tasks or training seeds. These are still tiny synthetic
development budgets, not a calibrated performance benchmark.

## Local v2 verification (2026-10-24)

The unchanged declared four-episode/two-seed pilot completed all 136 planned
executions, with no admitted task left incomplete and no test scenarios
constructed/evaluated. Each model ingested 77 transitions and made 46 gradient
updates. All 8 v2 final checkpoints reloaded with matching weight/metadata
hashes; all 8 historical v1 checkpoints were retained and correctly rejected
by the new encoder contract. Full optional-learning tests: 254 passed. The
standard-library runtime also passed, explicitly skipping 50 neural tests.

Actual source SHA256:
`e1e43a2934ecf7c3c88e8cef81a868755422695a26e4fe9fc38eef681ffdddf2`.
Learning-plan SHA256:
`5e8138419d16b3306dc79eb8d216e1e4a4a86c0af9d12927a495b78f917d8002`.
The 1,944,518-byte local report is
`reports/learning-pilot-v2/report.json`; its encoder has 99 features for M=4.
These artifacts remain ignored by Git. Both original physical/v1 learning
configuration files were preserved without modification.

There is **no superiority result**: in the two nominal validation episodes,
the combined-cost rule's mean episode-total cost was about 6.6945, while
the mean across the two QR-CVaR training seeds' validation means was about
13.9420 (dimensionless). These few-run point estimates are diagnostic, not
statistical evidence or a tail-risk comparison. All algorithms' seeds and
the other profiles remain in the report; no failed/poor run was filtered,
no weights changed to make the comparator lose, and no additional training
or test-based tuning was performed. Calibrated assumptions and a justified
larger training/evaluation budget are needed before drawing paper conclusions.
