# Stochastic Development Scenarios and Evaluation Protocol v1

Status: implemented synthetic scenario generation, split/seed planning,
terminal metrics, descriptive risk, and paired analytical-policy audits.
**These are uncalibrated development scenarios, not final paper parameters or
a frozen scientific benchmark.** No learned-policy superiority is established.

The checked-in configuration is
[`experiments/configs/development_v1.json`](../experiments/configs/development_v1.json).
It has a revision and canonical expanded-configuration hash; every report also
records actual source provenance. Changing the configuration changes its hash.
This prevents silently mixing different development protocols, but is not a
claim that final paper hypotheses/budgets have already been preregistered.

## Traffic, workload, and criticality

Per user and physical slot, independently draw a Bernoulli arrival with
`p = arrival_rate_per_user_hz * slot_duration_s`, requiring p in [0,1]. There
is at most one arrival per user/slot. This is **not** a Poisson batch process.
Without envelope violation, expected arrivals/user are `horizon_slots * p`.
The probability and effective rate are explicitly stored in the manifest.

Every candidate arrival becomes a task. If a configured admission envelope is
too small, generation raises an explicit error. It never truncates the offered
trace, retries a seed, or removes difficult episodes. The default envelope
equals the number of arrival slots, so every possible Bernoulli trace fits.
Runtime energy escrow still uses the configured worst-case envelope, not the
realized future trace. Tasks continue past deadlines and recover through MEC
under the existing kernel assumptions; full drain is required for evaluation.

Input/output sizes are discrete uniform over their inclusive bit intervals.
CPU cycles/bit are independently continuous uniform. Criticality is sampled
from an explicit categorical distribution and determines the relative deadline
and energy normalization budget. All draws use separate per-user/per-arrival
named streams, independent of channel and agent/exploration streams.

| Development input | Nominal value/range | Provenance/status |
| --- | --- | --- |
| Users / mobile servers / action capacity | 3 / 4 / 4 | synthetic cohort |
| Arrival horizon / slot | 2 s / 10 ms | finite development workload |
| Arrivals per user | 3 task/s; high-load profile 12 task/s | Bernoulli modeling choice |
| Input / output | 32,000–96,000 / 512–4,096 bit | synthetic workload |
| Computation density | 300–700 cycle/bit | synthetic workload |
| Normal/high/critical probabilities | .7/.2/.1 | synthetic labels, not measured event frequencies |
| Normal/high/critical task deadlines | .6/.3/.15 s | synthetic arrival-to-completion deadlines |
| Task energy budget | .1 J | objective normalization, not a drop threshold |
| User initial speed / braking magnitude | 8–15 m/s / 1.5–2.5 m/s² | bounded brake-to-stop motion |
| Server speed | 8–25 m/s; fast profile 20–35 m/s | synthetic straight-road cohort |
| Server CPU / self-work | .3–.8 GHz / .1–.3 of CPU | static per-episode draws |
| MEC CPU / background | 1.2 GHz / .1 of CPU | development capacity |
| Cache TTL / V2V range | .1 s / 100 m | sensitivity inputs, not calibrated limits |

All numeric workload/resource ranges above are engineering choices for a
development test, **not numbers extracted from a paper or standards table**.
Criticality is independent of input size and channel conditions here; labels
do not represent an implemented obstacle detector or semantic classifier.
The braking trajectory is likewise not triggered by the sampled task label.

For independent input size D and computation density eta,
`E[cycles] = E[D]*E[eta] = 64,000*500 = 32e6`. The manifest reports the nominal
MEC-only offered CPU-work ratio `users*rate*E[cycles]/MEC_service_CPU`: 4/15
nominally and 16/15 in the high-load profile. This is a mean-work diagnostic,
not an M/M/1 queue formula or a stability/deadline guarantee for the finite,
preemptive, queried, radio-constrained simulator.

## Mobility and single-cell completion assumptions

Vehicles occupy explicit straight-road lane-center y coordinates. A lane is
sampled uniformly; even lane indices move toward +x and odd indices toward -x.
This avoids assigning opposite traffic directions to the same lane. There is
no car-following, collision process, turn, lane change, or SUMO trace claim.

Users brake to a stop under the existing constant-acceleration primitive.
Validation requires the conservative envelope

`hypot(max_abs_initial_x + max_speed**2/(2*min_braking), max_abs_lane_y) < MEC_radius`.

Thus every possible user trajectory stays in the same MEC domain throughout
drain, regardless of which policy terminates first. The generator does not
wrap, freeze at a coverage boundary, or reject positions after seeing results.
This is specifically a bounded approach/braking scenario, not unconstrained
continuous driving. Supporting continuous traffic through multiple cells
requires a separate handover model and revised completion assumptions.

Servers start inside MEC coverage, move at constant speed, and may later leave.
The kernel refreshes registration, serves only cached candidates visible to
the user, and performs common MEC recovery for stale/invalid V2V choices.
There is no replacement-server inflow. The fixed action capacity is separate
from the current cohort size; absent servers remain padded/masked. All protocol
profiles must share that capacity. Old two-server smoke checkpoints have a
different encoder signature from this four-slot plan and are not silently
reused as trained models for the new workload.

## Radio/resource constants and calibration limits

The generator's explicit environment/entity manifests include the additional
v1 constants: 5.9 GHz reference carrier, deterministic V2I log-distance exponent
2.2, 2 MHz V2I UL/DL and aggregate V2V bandwidth, effective noise density
3.2e-20 W/Hz, 1 W user/server TX, .5 W user/server RX, .1 W user query idle,
10 W aggregate MEC DL, query request/header/record sizes 256/256/512 bit,
2 ms separate MEC query processing, and 1e-6 J query-build energy.

Nominal V2V uses exponent 2.7, 2 dB shadowing, and Rayleigh fading; the harsh
profile changes shadowing standard deviation to 6 dB. Large/small-scale draws
are currently independent per link/direction/slot, **not spatially/temporally
correlated 3GPP channels**. SNR-outage persistence is two slots. Deterministic
positive V2I service, orthogonal sharing, finite task envelopes, and no user
handover remain the conditional-completion assumptions of the kernel.

Server CPU reserves an additional fixed 5% separately from self-work, uses one
execution slot, coefficient 1e-28 in `kappa*C*f**2`, a 100–200 J service-energy
budget, and 10% energy reserve. The user budget is 1000 J. These are modeled
service budgets, not a calibrated EV traction battery. MEC has two concurrent
CPU slots. User/server waiting capacity is the declared task envelope. The
episode guard is 100 s by default; hitting it raises an error, never a success
sample or a dropped task. No failed run is silently removed from an aggregate.

Communication requirements such as
[ETSI/3GPP TS 22.186](https://www.etsi.org/deliver/etsi_ts/122100_122199/122186/18.00.01_60/ts_122186v180001p.pdf)
must not be equated to our query + radio + queue + compute task deadlines.
This analytical development model is not evidence of standards compliance.
Empirical/trace-based calibration, time correlation, and hardware measurement
remain open work before a paper benchmark can be fixed.

## Split and seed protocol

Development v1 declares 20 training, 6 validation, and 30 test episode indices,
plus future learned-agent seeds 11/29/47/71/97. Those are development budgets,
not a power analysis or final scientific sample size. The current audit trains
no neural policy and does not measure variation across learned-agent seeds.

SHA-derived named streams separately determine scenario, physical-channel,
and random-policy seeds for each `(split, index)`. The finite plan rejects
role/split collisions. Access order and number of generated profiles do not
change these assignments. Profiles intentionally reuse the same episode seed
keys for controlled changes; do not pool them as independent replications.

Every comparison policy in an episode gets the same pristine prototypes,
arrival/workload trace, and channel seed. The kernel draws all link streams
at every boundary independently of query/actions; policy-dependent episode
lengths share their common physical-time prefix. Admission/service outcomes
can naturally differ. This is common exogenous randomness, not an oracle
giving every policy identical future queues or observations.

The audit uses a declared prefix of **validation** indices by default. Test
access requires `--allow-test`; software tests exercise a separate tiny fixture,
not the checked-in held-out test scenarios. An explicit flag is an access guard,
not cryptographic blinding. Hyperparameter tuning and checkpoint selection must
not use test results. Before learned comparisons, freeze training budgets,
validation selection rules, final scenario distributions, and statistical units.

## Metrics, risk random variable, and uncertainty

Terminal metrics require full drain. They include total combined cost/mobile
energy, final MEC workload share, direct/fallback fractions, on-time fraction,
critical deadline-miss fraction, latency mean/p95/p99, and query round-trip mean.
Undefined per-task/per-critical-task averages return JSON null, not a favorable
zero. An empty episode still contributes its actual zero episode-total cost.
Task latency percentiles are descriptive pooled-within-episode diagnostics;
they are not independent-observation confidence intervals.

The risk random variable for this audit is **undiscounted total combined cost
of one fixed-arrival-horizon, full-drain episode**. Episode variation includes
offered traffic and criticality. Every episode has equal weight. CVaR is not
computed by pooling correlated task costs or averaging per-episode CVaRs.
Mean per-task cost is a separate diagnostic, not the same random variable.
Agent conditional discounted risk scores remain a distinct learning quantity.

Summaries report empirical mean/VaR/CVaR and nominal upper-tail mass
`episodes*(1-alpha)`. With two episodes at alpha=.9, empirical CVaR is just the
largest observed episode cost; it is **not** a reliable rare-event estimate.
Even the development test count of 30 has only three episodes of nominal tail
mass. No small-sample CVaR reliability or superiority claim is made.

For fixed analytical policies, the audit resamples matching episode keys
jointly to form a percentile bootstrap interval for mean cost difference
`method - AlwaysMEC`; negative favors the method on cost. Percentile endpoints
use the left-inverse bootstrap empirical CDF. It neither resamples individual
tasks nor treats users as independent trials. It is not a p-value, a CVaR
confidence interval, or an interval covering learned-agent training variation.
That later crossed agent-seed/scenario design requires its own uncertainty
analysis. The need to report run uncertainty rather than only point estimates
is discussed by [Agarwal et al.](https://arxiv.org/abs/2108.13264).

Only AlwaysMEC, AlwaysV2V, UniformRandom, and supplementary minimum estimated
latency are in this audit. All retain the mandatory cache/query protocol;
query-free MEC and adapted paper/combined-cost baselines are still needed for
the final end-to-end comparison. Do not tune parameters to make those methods
lose, or interpret this infrastructure audit as a learned-policy result.

## Run a development audit

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m experiments.run_scenario_audit --profile nominal --split validation --cases 2
.\.venv\Scripts\python.exe -m experiments.run_scenario_audit --profile high_load --cases 2
.\.venv\Scripts\python.exe -m experiments.run_scenario_audit --profile fast_servers --cases 2
.\.venv\Scripts\python.exe -m experiments.run_scenario_audit --profile harsh_channel --cases 2
```

The audit also runs without optional learning dependencies. Use `--output
reports/development-nominal-v1.json` to save complete finite JSON to an unused
filename. Existing reports are never overwritten. Reports include expanded
protocol/configuration, seeds, scenario/source fingerprints, complete scenario
manifests, per-policy terminal metrics, prefix/transition accounting, descriptive
risk, and paired intervals. Both reports/ and runtime artifacts remain ignored
by Git; only versioned code/configuration/tests/documentation enter the repo.

Next: validate/calibrate these assumptions, connect learned multi-seed training
to the declared scenario plan, freeze validation/model-selection rules, and
implement the remaining scientific baselines before long training or test use.
