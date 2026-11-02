# Reference Contract: Label-Free Cost v2, QR Risk Score, and Sigmoid Preference

Approved revision: 2026-11-02. This document is the reference for the active
objective, risk-score sign, sigmoid meaning, discrete selector, and schema
boundary. It supersedes the historical task-label/deadline-penalty objective.
Old experiment reports, checkpoints and configuration JSON are preserved;
none of their outcomes is evidence for this revision.

## 1. Actual task cost (dimensionless)

\[
z_i=0.60\frac{T_i}{D_i}
    +0.25\frac{E_i^{\mathrm{mobile}}}{E_i^{\mathrm{budget}}}
    +0.15M_i.
\]

T_i is actual completion minus arrival, in seconds. It contains query wait,
admission/queue wait, upload, compute, download, slot phase gates, discarded
V2V attempt time, and MEC restart, in disjoint ledger buckets. D_i>0 is the
same task's relative deadline in seconds, with **arrival as the same origin**.
E_mobile is actual attributable UV+SV energy in joules; E_budget>0 is that
task's normalization budget. MEC CPU/radio energy remains excluded. Shared
query energy belongs only to its triggering task, not every waiting task.

M_i is exactly one if MEC ultimately executes the task, directly or following
V2V interruption, and zero after full SV completion. `_mec_event` charges .15
at most once. Restart does not charge it a second time. Both wasted attempt
energy/time and successful restart service remain in the actual cost once.

There is **no manual criticality multiplier, deadline-miss surcharge, ReLU,
hinge, clipping, sigmoid, preference multiplier, or surrogate replacement** in
this cost. T/D may exceed one without saturation. The common coefficients
(.60,.25,.15) are fixed objective weights, not task labels. Active v2 config
validation rejects different coefficients; a future objective sensitivity
study would require a separately approved/versioned contract.

The kernel accumulates this formula linearly per slot. `reward=-cost` exactly.
The decision adapter retains every interval cost and pre-action prefix; replay
stores actual costs (not sigmoid outputs), and the Bellman target is still
`discounted_interval_cost + bootstrap_discount * target_atoms`.
`final_task_cost` reconstructs the reference from final physical outcomes;
terminal metrics reject a mismatched ledger, MEC indicator, or time origin.

Deadline crossings remain one-time recorded events, and late tasks continue.
Reports include deadline-miss fraction and total/mean/max task tardiness in
seconds. `max(0, completion-absolute_deadline)` defines a **metric only** and
never enters the objective. Empty-episode undefined means return null.

## 2. Learned cost risk, not reward-tail risk

QR-DQN estimates Z_theta(s,a), the distribution of future cumulative cost,
using existing equal-mass quantile outputs, replay, Huber quantile loss, and
target updates. No output sigmoid is inserted into that distribution/loss.

\[
J_\theta(s,a)=(1-\lambda)\mathbb E[Z_\theta(s,a)]
              +\lambda\operatorname{CVaR}_{0.9}[Z_\theta(s,a)].
\]

Upper CVaR averages the largest-cost 10% of probability mass; fractional
boundary mass is retained. This is the **worst cost tail**, not the best one.
Cost selection minimizes J. Negated-reward quantiles would instead require
lower-tail CVaR and maximization; sign-equivalence remains numerically tested.

The proposed QR-CVaR controller retains the existing lambda=.5 and alpha=.9.
Lambda is a global risk-mixture parameter, unrelated to removed task weights.
Risk-neutral QR, DQN, and Double DQN retain their established effective risk
weight zero. Gamma per physical slot remains 1 in the active pilot; the
variable-duration decision discount/terminal-zero rules are unchanged.

This is still a risk-distorted greedy distributional controller in a partially
observed multi-user simulator. CVaR-greedy action choice alone does **not**
guarantee optimal full-episode/static/nested CVaR or calibrated rare-event risk.

## 3. One stable preference curve for every valid SV

For each observed-valid vehicle action j and MEC action 0:

\[
\Delta_j=J_\theta(s,j)-J_\theta(s,0),\qquad
g_{\theta,j}=\frac{1}{1+\exp(-\beta\Delta_j)},\quad\beta>0.
\]

`sigmoid_mec_preference` uses sign-dependent stable exponential branches.
Large finite gaps/products saturate correctly without exponential overflow.
Beta has inverse-score units (dimensionless with the normalized cost score).
The new explicit `AgentConfig.sigmoid_beta=1.0` is an engineering **display
default**, not fitted from old graph coordinates or validated as a scientific
hyperparameter. It has no effect on the policy/loss; lambda is not changed.

Small g means preference for that SV relative to MEC; .5 means their J values
tie; large g means preference for MEC. These are interpretations of **one
function**, not three rules/threshold regions. Center is Delta=0. Scores for
different SVs do not sum to one and are not action-selection probabilities.

`NeuralCostAgent.preference_diagnostics` records learned means, upper CVaR,
J, Delta, and g for only mask-valid actions. It reads encoded received
observations, not simulator truth, and consumes no RNG/gradient/replay state.
Greedy evaluation reports these diagnostics. There is no separate sigmoid
network, optimizer, training target, sampling policy, or semantic classifier.

Only the scalar map Delta -> g is smooth. The full simulator, ReLU network,
mask, finite-precision scores and discrete argmin need not be smooth. At
extreme gaps finite precision can make several g values identically 0 or 1;
**never rank sigmoid values**, even though the mathematical map is monotone.

## 4. Final indivisible action

\[
a^*(s)=\arg\min_{a\in\{0\}\cup\widehat{\mathcal C}(s)}J_\theta(s,a).
\]

Choose the smallest observation action index on an exact J tie (MEC=0).
With no observed-valid SV, select MEC. All valid vehicle candidates compete
in the same masked argmin; there is no pairwise lottery or task splitting.
Changing beta cannot change J ordering, chosen action, TD targets, replay
sampling, or exploration. Explicit per-user seeded epsilon exploration is
preserved for training; evaluation remains epsilon=0 and never learns.

Masks use valid received snapshots, not true current feasibility. A selected
SV may still suffer a recoverable physical interruption; existing full MEC
restart preserves all waste. The presence of MEC is **not** a deadline or
universal failure-free guarantee. Completion depends on the kernel's stated
positive-service, funded-energy, finite-workload and single-cell assumptions;
violations/guard failures still raise errors rather than fabricating success.

## 5. No manual priority through another path

Task has no criticality/priority field or multiplier. Plain EDF orders tasks
by (absolute deadline, arrival slot, stable task ID). The same key is used for
user task selection, batch admission and radio/CPU arbitration. Preemption
remains at slot boundaries with remaining work/executor preserved.

The raw/encoded schemas remove manual-label fields/one-hots. Real task size,
work, deadline/slack, rates, capacities, queues, mobility and freshness remain.
Development scenario v2 samples **unlabelled** deadline/energy profiles. Its
numeric deadline mixture retains .6/.3/.15 s at .7/.2/.1 as synthetic workload
inputs, without emergency classes or semantic labels. Independent named RNG
streams remain explicit; v2 seed namespaces/fingerprints distinguish new runs.
The network learns cost/time sensitivity from actual features and outcomes.
The sigmoid itself does not learn and cannot recognize "emergency braking"
or obstacles without corresponding semantic information.

The cached greedy baseline uses the same three-term objective; its projected
miss flag is diagnostic only. Its pre-existing cached-contact admissibility
guard remains an explicitly named heuristic constraint, never an added real
task penalty. The paper best-response baseline is still unimplemented here;
the reference repository is read-only. A future reconstruction must document
its native internal utility separately from this common evaluation objective,
not silently replace its original formulas or label it our proposed method.

## 6. Versions, artifacts and scope

| Contract | Active version |
| --- | --- |
| Actual objective / metadata cost model | 2 |
| Task specification | label-free (old criticality constructor rejected) |
| Raw / encoded observations | 3 / 3, dimension `36+15*M` (96 at M=4) |
| Replay / checkpoint containers | 2 / 2, explicit cost-model guard |
| Scenario spec / development plan | 2 / 2 |
| Learning plan / fixed-policy suite | 3 / 3 |

Active configurations are `development_v2.json` and `learning_pilot_v3.json`.
Old v1/v2 learning JSON and the old development v1 JSON remain unchanged and
are rejected by current loaders. Existing results/checkpoints are preserved,
not padded, relabeled, silently re-scored or used as corrected-model evidence.
Actual source fingerprints and expanded plan hashes remain in reports.

Tests must establish the three-term final cost (direct MEC, SV, full fallback,
query ownership and nonzero arrival origin), unbounded late-task ratio, no
label leakage, pure EDF/preemption, upper-tail/sign correctness, stable
sigmoid direction/center, masks/ties/no-SV case, beta-independent decisions
and updates, evaluation immutability, seeded reproducibility, and rejection
of wrong cost-model metadata even when vector dimensions match.

Adding this sigmoid alone establishes neither novelty nor performance
improvement. Parameter/workload calibration, sufficient independent runs,
quantile-tail calibration, query-free MEC and the faithful paper comparison
remain scientific work. No new physical failure model, local execution,
partial offloading, learned query timing or paper-baseline rewrite is added.

Implementation verification, exact run scope, source hashes and remaining
scientific limits are recorded in
[revision_validation_20261102.md](revision_validation_20261102.md).
