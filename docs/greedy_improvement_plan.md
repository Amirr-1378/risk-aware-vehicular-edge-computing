# Diagnosing the Gap to Cached Combined-Cost Greedy

The goal is a reproducible improvement under the declared scientific objective,
not a guaranteed win on every trace. Cost weights (.60/.25/.15), alpha=.9,
lambda=.5, sigmoid semantics, physical assumptions, and the test boundary remain
unchanged. No new training is performed by the action audit. Main training
remains a later Colab milestone, with notice to Amir before execution.
The [executed 2026-11-17 audit](action_audit_20261117.md) records its full scope,
descriptive results, an owner/system conflict example and source fingerprints.

## What the completed pilot establishes

The [development pilot](development_pilot_20261117.md) produced only 75
transitions/44 optimizer updates per model. Its 50-update target-refresh
interval was never reached after initialization, and its exploration decay
was incomplete. This is evidence of a small training budget, not proof that
more training alone will beat greedy. Evaluation was already greedy, epsilon=0;
lowering evaluation epsilon cannot repair the observed cost gap.

The cached heuristic has explicit radio/CPU/energy equations. A network trained
from random initialization on a few dozen transitions has to learn useful
control from very limited data. The intended advantage of learning is in
action-dependent future queues, shared contention, stale information and
recovery cost, rather than merely reproducing a one-task point projection.
Whether those advantages are learnable from the current received observations
is a hypothesis to test. Queue, age, motion and contact estimates are already
present; simply saying to add them does not identify a missing feature.

There is also a training/evaluation alignment question. `DecisionAdapter`
attributes each transition only the current user's costs until its next
decision/termination. The reported episode objective sums all users' task
costs. With shared radio and CPU service, reducing one user's cost can raise
others' costs. Parameter sharing across users does not by itself guarantee
optimization of the global sum. This is not a missing-ledger accounting bug:
the complete per-user ledgers still add up correctly. Team credit assignment
would be a separate learning-contract/design change and is not silently
introduced by this diagnostic.

QR-CVaR minimizes its local mean/upper-tail score. A risk-sensitive policy can
accept a higher mean to reduce large costs, so beating mean cost and improving
tail cost are distinct hypotheses. Distributional learning itself does not
guarantee either. In addition, ordinary distorted distributional action
selection need not solve static or dynamic CVaR exactly; see
[Lim and Malik (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract-Conference.html).
The implementation continues to make no such optimality claim.

## New offline action audit

```text
python -m experiments.run_action_audit --profiles nominal high_load --cases 2 --decisions-per-episode 8 --output reports/NEW-action-audit/report.json
```

The declared default inspects the first eight individual decisions per
validation episode on the unchanged greedy trajectory, in slot/user-ID order.
It uses two cases in each of two known development profiles. No decision is
selected because it looks favorable after rollout; fewer decisions, including
empty episodes, remain in the report. Every reference episode still drains.
The CLI has no test option and refuses to overwrite an existing output.

For each inspected user/decision, clone the complete simulator and its owned
RNG state once per observed-valid destination. Keep other simultaneous users'
actions fixed to the original greedy batch. Change only the inspected user's
current action, then recompute all later greedy actions from that branch's
received observations until full drain. Source prototypes, live reference
state, RNG streams and policies remain untouched. An unchanged-action branch
must exactly reproduce the terminal metrics/slot of the actual greedy path.
All mask-valid actions are simulated, including actions the heuristic rejects
with its additional cached-contact guard; that flag is recorded, not used to
filter the diagnostic branches. An unavailable myopic estimate is JSON null,
not infinity or an actual outcome. The simulator's ordinary recovery applies.

For task j with cost already incurred at the fork denoted by c_j^past,

`C_u(a) = sum over tasks owned by u of (terminal_cost_j(a) - c_j^past)`.

Future task arrivals have zero past cost. Subtraction is per ledger, retaining
all future query, queue, service, wasted work and recovery costs; it excludes
sunk costs already accrued at the current slot boundary. The system value is
`C_all(a) = sum_u C_u(a)`. For greedy action g, report

- `Delta_u(a) = C_u(a) - C_u(g)`;
- `Delta_all(a) = C_all(a) - C_all(g)`;
- the separate ex-post minima for the owner and system, with lowest-index ties;
- for the owner's best action, the external cost change
  `Delta_other = Delta_all - Delta_u`;
- actions whose owner/system differences have opposite strict signs.

Negative deltas favor the alternative. A dimensionless 1e-9 tolerance is used
only for counting strict gains/conflicts, never for ranking actions or altering
the objective. Means in the summary are per inspected decision, not per
independent episode and not additive episode improvements.

The clones contain hidden state and future workload needed for simulator
execution. This is an **offline, ex-post diagnostic**, not an implementable
policy, a training label, an expected action value, a CVaR estimate or an
optimal-policy bound. Selecting the best branch after seeing one future path
introduces hindsight selection. Overlapping opportunities at different forks
cannot be added into a claimed policy gain. A positive gap does not prove that
the available cached observations predict which action will be better.

## Recommended development order

1. **Establish the relevant objective and action sensitivity.** Inspect owner
   versus global cost tradeoffs before deciding whether the learning return
   should remain individual or become cooperative. A change must be explicit,
   tested and versioned, including how irregular multi-user decisions receive
   credit. Retrofitting a team cost into overlapping per-user transitions is
   not a justified global-MDP derivation by itself.
2. **Declare an informative development training budget.** Use more independent
   training traces/transitions, a fixed budget long enough to exercise repeated
   target refreshes and the exploration schedule, multiple predeclared seeds,
   and frozen greedy validation at declared checkpoints. Those mechanical
   milestones are necessary diagnostics, not sufficient convergence criteria.
   Estimate execution time and choose a separate versioned budget before
   running; do not keep extending only losing methods until they win. A later
   main training job must run in Colab.
3. **If sample efficiency remains the obstacle, test greedy initialization or
   training demonstrations as a declared variant.** Use only ordinary causal
   greedy actions/actual costs from training scenes, never the ex-post best
   validation branches above. Keep subsequent exploration possible. Compare
   with/without initialization and give risk-neutral and risk-sensitive QR
   the same treatment and total data/update budgets. Learning from
   demonstrations is supported as an approach by
   [Hester et al.](https://arxiv.org/abs/1704.03732); its results do not prove an
   improvement for this simulator. No demonstration loss, replay prefill or
   reward bonus has been added here.
4. **Test information sufficiency before adding architecture complexity.**
   Use training-only studies of whether queue history, snapshot age, contact
   estimates and outage history predict future waste/contention beyond the
   myopic estimate. If history/recurrent control or a physics-derived feature
   is introduced, version the encoder and include a matched ablation. No live
   queue, future channel, arrival trace or ex-post label may enter deployment
   observations. Do not make the physical problem harder only for the baseline.
5. **Evaluate mean and tail costs separately.** Preserve lambda=.5 in the
   approved method and lambda=0 in the neutral QR comparison. Report deadline
   misses, fallback, mobile energy and completed MEC/SV workload along with
   cost; completion alone is not on-time delivery. Use enough independent
   scenarios/seeds to quantify uncertainty before claiming improvement, in
   line with [Agarwal et al.](https://agarwl.github.io/rliable/). Two validation
   episodes cannot establish tail quality. Finish calibration, missing paper/
   query-free MEC comparisons and the frozen protocol before held-out testing.

Greedy may remain difficult to beat on light, nearly decoupled workloads.
Any eventual improvement claim must specify the supported operating region
and its uncertainty; there is no reason to promise universal dominance or
infer paper acceptance from a particular learning algorithm.
