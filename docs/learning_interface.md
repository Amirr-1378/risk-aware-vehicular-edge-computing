# Learning Interface, Simple Baselines, and Risk Utilities

Status: implemented and tested without third-party runtime dependencies. This
milestone does **not** train DQN/QR-DQN or establish an algorithmic improvement.
It supplies an auditable interface to the simulator described in
[`simulation_environment.md`](simulation_environment.md).

## Received information and vector schema v1

`ObservationEncoder(max_server_vehicles=M, scales=FeatureScales(...))` consumes
only immutable raw schema-v1 `OffloadingObservation` records. It never reads
the simulator, live server queues, or future arrivals/channels. Baselines
consume those same raw records; neural policies will consume their encoding.
Neither representation gives an oracle feasibility mask.

The vector has `38 + 15*M` dimensionless entries in the order exposed by
`encoder.feature_names`:

- 32 task, own-user, cache-age, and cached-MEC features;
- six query-status one-hot entries, in explicit versioned order;
- 15 features per cached server slot, including presence and record age.

Criticality uses three one-hot entries for weights 1/2/4. Heading uses sin/cos,
not a discontinuous raw angle. Absolute timestamps and task/user/server IDs
are excluded from network values. User/task IDs in `EncodedObservation` are
diagnostic labels, not neural inputs. Servers retain the raw observation's
ID-sorted slot order; an action resolves against that observation, not a later
snapshot. Server identities/slot membership may change between snapshots.

An absent server has an entirely zero block and a false action-mask entry.
A present but masked server retains its cached features and presence=1. The
boolean mask is returned separately for action selection and bootstrap
selection, not interpreted as a physical guarantee. MEC is action zero.

Reference scales are explicit positive, finite SI values:

| Quantity | Illustrative reference |
| --- | ---: |
| Input / output | 1e6 / 1e5 bit |
| CPU work / CPU rate | 1e9 cycle / 1e9 cycle/s |
| Time / energy | 1 s / 1 J |
| Speed / distance | 30 m/s / 100 m |
| Bandwidth / rate | 1e6 Hz / 1e7 bit/s |
| Linear SNR / task count / contact margin | 100 / 10 / 10 |

Fractions and one-hot values are already dimensionless. These references are
not calibrated physical parameters, sample maxima, or clipping limits. Values
are not clipped: overdue slack remains negative and extreme workload remains
distinguishable. Non-finite features are rejected. There are no running
statistics and no fitting on evaluation data. Freeze and save scales, M,
feature names, and raw/vector schema versions with each future checkpoint.

Age augmentation does not make a stale-snapshot observation fully Markov.
Feed-forward policies remain approximations in a partially observed,
multi-user setting. History/recurrent policies are a separate extension.

## Decision epochs and complete cost accounting

`DecisionAdapter(environment, encoder, gamma_per_slot=...)` owns reset/step
for the episode. Do not step/reset its environment separately. It returns
encoded observations and closes a transition separately for each user at that
user's next decision or global episode termination. Every action batch still
uses the kernel's common pre-admission observations. Empty `{}` advances the
physical simulator but creates no learned no-op action.

For an action at slot t and its next decision at t+Delta:

\[
\bar c_{u,t}=\sum_{k=0}^{\Delta-1}\gamma_{slot}^k c_{u,t+k},\qquad
d_{u,t}=\begin{cases}0&\text{terminal},\\
\gamma_{slot}^{\Delta}&\text{otherwise}.\end{cases}
\]

`DecisionTransition` carries raw interval cost, discounted interval cost,
duration in slots, start slot, action, current/next encoded observations, and
bootstrap discount d. A future distributional cost target is
`discounted_cost + bootstrap_discount * next_cost_quantiles`; it must not use
a single fixed decision-discount regardless of duration. Reward convention is
the exact negative. At a terminal transition the next observation is `None`;
there is no fake valid terminal action. Costs include all tasks belonging to
that user, not just the task named in the current observation. Other users'
concurrent actions affect service through the common simulator.

The default slot discount is 1 for undiscounted finite-episode accounting;
values in [0,1] are supported. If gamma<1, document the physical-time discount
horizon and hold it consistent in slot-size sensitivity experiments. Raw
scientific cost metrics are always reported separately from discounted returns.

Before a user's first action, there is no action transition to which query
and waiting costs can honestly be assigned. These costs are an explicit
per-user prefix, not dropped and not attached to a dummy action. A user's
prefix can still depend on other users' behavior through shared service.
After its last action, its open transition includes the entire drain phase.
For example, costs `[2,3,5,7,11]`, first decision at slot 2, next at 4, and
termination after slot 4 decompose into prefix `2+3*gamma`, interval
`5+7*gamma` with bootstrap `gamma**2`, then terminal interval `11`.

After termination, `adapter.accounting` must satisfy for each user:

\[
\sum_t c_{u,t}=P_u+\sum_j C_{u,j},\qquad
\sum_t\gamma_{slot}^t c_{u,t}
=P_u^\gamma+\sum_j\gamma_{slot}^{t_j}\bar c_{u,j}.
\]

Here prefix and closed-transition discounted accounting use absolute episode
slot zero, while each transition's discounted cost uses its own start as
origin. Kernel boundary costs are taken exactly from returned slot costs;
the adapter never peeks at ledgers to shift costs between intervals. Global
termination requires full drain. A user with no actions gets no transition;
its prefix still preserves any costs. Aborted/guard-limited runs are not
silently treated as successful terminal training samples.

## Exact finite-distribution risk scoring

For N equally weighted cost atoms z, the utilities implement

\[
\operatorname{CVaR}_{\alpha}(z)
=\frac{1}{1-\alpha}\int_{\alpha}^1 F_z^{-1}(p)\,dp,\quad
J=(1-\lambda)\operatorname{mean}(z)+\lambda\operatorname{CVaR}_{\alpha}(z).
\]

Sort a copy of the atoms descending. Let q=N*(1-alpha), k=floor(q), and
f=q-k. Upper CVaR is `(sum(top k atoms) + f * next atom) / q`, omitting the
next atom when no fractional mass remains. Thus N=15 and alpha=.9 uses the
largest atom and half the next, not an arbitrary rounded number of atoms.
For atoms 0..14 this is `(14 + .5*13)/1.5 = 41/3`.

This is a discrete-distribution tail integral, not `mean(z >= VaR)`, which
can be wrong at ties. `empirical_var` uses the left-inverse empirical CDF.
Confidence is in [0,1); alpha=0 reduces CVaR to the mean. Empty/non-finite
distributions and invalid confidence/mixture weights are rejected. Nominal
alpha=.9 and lambda=.5 match the approved design.

Cost scores are minimized. With rewards r=-z, maximize the mean/lower-tail
mixture over the smallest 1-alpha mass. `mean_cvar_reward(-z) == -J(z)`;
using the upper reward tail would favor the wrong outcomes. Masked action
rows are not evaluated; valid rows must have equal positive atom count and
finite values. Ties resolve by the smallest valid index. Scoring sorts a copy
even if predicted quantiles cross; it does not reorder training fractions.

These routines score supplied finite distributions. They do not implement a
QR-DQN update or prove that greedy CVaR distortion globally optimizes a
static episode-level CVaR objective. Static CVaR, conditional per-decision
risk distortion, and time-consistent nested risk are different formulations.
The next agent milestone must state its update/policy interpretation explicitly;
empirical tail improvement alone is not such a theorem. Likewise, few seeds
or few atoms cannot establish reliable tail estimates.
The failure of standard CVaR-greedy distributional control to guarantee either
static or dynamic CVaR optimality is analyzed by
[Lim and Malik (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract-Conference.html).

The general-distribution CVaR definition follows
[Rockafellar and Uryasev (2002)](https://doi.org/10.1016/S0378-4266(02)00271-6).
Equal-mass quantile approximation follows the representation in
[Dabney et al., QR-DQN](https://arxiv.org/abs/1710.10044); this does not imply
that their risk-neutral control results prove our risk-sensitive controller.

## Implemented comparison policies

| Policy | Observation-only rule |
| --- | --- |
| AlwaysMEC | MEC, regardless of cached V2V availability |
| AlwaysV2V | Any observed-valid V2V; minimum estimated service time within SVs; MEC if none |
| UniformRandom | Uniform over all observed-valid actions, including MEC |
| MinimumEstimatedLatency | Minimum estimated service time over MEC and SVs |

Estimated service time is cached uplink + cached backlog delay + task CPU +
cached downlink. MEC uses its supplied queue-delay estimate once; SV uses
queue-work/available-CPU. These estimates omit future arrivals, contention,
discrete phase gates, and future interruption risk. Zero rates/CPU can give
infinite estimates; deterministic action-index ties still produce an action.
Subsequent true-state rejection and MEC recovery are identical across policies.

This is an algorithm-only comparison under the same mandatory cache/query
protocol. Even AlwaysMEC waits for a valid snapshot in this kernel, although
an end-to-end MEC-only system need not query for candidate SV information.
A query-free AlwaysMEC variant is therefore an additional future protocol
baseline, not something this smoke comparison has already evaluated. The
mandatory-query results must not be presented as its best achievable behavior.

UniformRandom requires an explicit seed and separate persistent named streams
per user. It does not advance global randomness or physical channel streams.
Construct it anew with the same seed for a reproducible new evaluation run.

MinimumEstimatedLatency is supplementary, **not** the approved greedy
one-step combined-cost baseline: the cached schema alone does not expose all
power/CPU-energy calibration needed to estimate that objective. That baseline,
the adapted paper best-response baseline, DQN, Double DQN, and learned
risk-neutral/risk-aware QR-DQN remain unimplemented. No policy here is named
after the paper's method or claimed to reproduce it.

## Reproducible integration run and next milestone

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_baseline_smoke --seed 7 --policy-seed 17 --gamma-per-slot 1
python -m unittest discover -s tests -v
```

The finite-JSON report shares one scenario/configuration and environment seed
across all four rules, while recording the separate random-policy seed,
encoder schema/scales/names, action traces, decision durations, and raw and
discounted cost decompositions. It checks prefix/transition conservation after
drain. The scenario is deliberately small and synthetic; its results must not
be used as a paper benchmark or evidence of statistical superiority.

Next: implement and numerically test learning updates/replay/checkpoints for
DQN, Double DQN, risk-neutral QR-DQN, and the stated risk-distorted controller.
Then freeze training/evaluation protocols, independent seeds, calibrated
scenario ranges, confidence intervals, and ablations before performance claims.
