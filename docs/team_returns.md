# Cooperative Returns With Independent User-Vehicle Decisions

Amir confirmed on 2026-11-20 that **each User Vehicle independently chooses
its destination; MEC only supplies information**. This fixes the deployment
boundary. Training may aggregate simulated outcome costs offline, but no
online MEC coordinator, shared actor observation, extra messages, or hidden
queue/future information is introduced by this layer.

`agents.team_returns.TeamReturnAdapter` is an opt-in training-data interface
over the unchanged `DecisionAdapter`. It preserves the exact local raw/vector
observations, masks, action batches, physical transitions, individual task
ledgers, query/fallback protocol and completion assumptions. Team records can
now train `TeamNeuralCostAgent` through typed `TeamReplayBuffer`, with explicit
team checkpoint/return contracts. The smoke runner exposes `--return-scope team`;
individual returns remain the default, and the stochastic v3 learning plan
still uses its declared individual-return implementation. No learning-performance
gain is claimed.
The [2026-11-20 verification record](team_return_validation_20261120.md)
documents the complete matched interface audit and automated test results.
The subsequent [learning verification](team_learning_validation_20261120.md)
records optimizer, checkpoint and resume checks.

## Objective and decision intervals

The physical task objective remains cost model 2:

`z_i = .60*T_i/D_i + .25*E_mobile_i/E_budget_i + .15*M_i`.

Let c_(u,t) be the actual cost attributed to user u in slot t. The canonical
team slot cost is `c_team(t) = sum_u c_(u,t)`, without division by the number
of users. The raw episode objective is the sum of these team slot costs. The
offline aggregator includes users without a currently eligible decision and
tasks already uploading/queued/executing/downloading; it never selects a user
subset based on outcomes. Task costs are unchanged and are still independently
reconstructed from the final ledgers by `terminal_metrics`.

For user u's action at t until its next decision t+Delta (or global termination),

`team_interval_cost = sum(k=0..Delta-1) c_team(t+k)`

`discounted_team_interval_cost = sum(k=0..Delta-1) gamma_slot**k * c_team(t+k)`.

The corresponding reward is the negative discounted team cost. The bootstrap
discount is `gamma_slot**Delta` for a continuing transition and zero at
termination. The observation/action/next observation remain exactly the same
local records as in the original individual transition. Query prefix and
variable drain costs are retained. Slots without an offloading decision do
not generate a learned no-op. Gamma zero and one are both supported.

This supplies a shared-cost signal for a cooperative independent-actor
experiment. It does not make stale local observations Markov or guarantee
global optimization: other policies change during learning and their actions
affect service. The nonstationarity of independent Q-learning in multi-agent
settings is discussed by [Lowe et al.](https://arxiv.org/abs/1706.02275).
The design does not implement actor-critic or claim the paper's results.
Centralized training and decentralized action selection are distinct concepts;
[Rashid et al.](https://proceedings.mlr.press/v80/rashid18a.html) give one
value-factorization approach, whose factorization is not implemented here.

## Overlap is an actor-view property, not a physical charge

There are two intentionally separate representations:

1. A canonical team slot history and **disjoint global cost intervals**.
   Intervals close at the next decision of any user or at termination. Global
   prefix plus these intervals reconstructs the episode cost once.
2. Each user's decision intervals view that same team history. These intervals
   can overlap across users, because decisions occur at different times.
   They are samples for separate independent actors, not disjoint accounting
   buckets. Never sum them across users as the global episode cost.

For example, team slot costs `[2,3,5,7,11]`, user 1 decisions at slots 1/3,
and user 2 decisions at slots 2/3 yield:

| Representation | Prefix | Closed costs |
| --- | ---: | --- |
| Unique global ledger | 2 | 3, 5, 18 |
| User 1 team-return view | 2 | 8, 18 |
| User 2 team-return view | 5 | 5, 18 |
| User with no actions | 28 | none |

Every row separately reconstructs the same total 28; adding rows would be
incorrect. A user who never acts has only a diagnostic prefix and no invented
transition. At nonterminal times, open interval costs are included in the
total but are not yet closed into a replay candidate. Guard failures preserve
that distinction rather than fabricating terminal success.

For discounted conservation, multiply each closed interval's relative-origin
cost by `gamma_slot**start_slot`; prefixes use the absolute slot origin. The
per-user prefix plus that user's absolutely discounted closed views equals
the one discounted team total after complete drain. Original individual
accounting remains available independently as `individual_accounting`.

## Explicit compatibility boundary

Team return schema 1 has the scope tag `all-users-sum-v1`, physical cost model
2, and a distinct `TeamDecisionTransition` type. Its `source` is the original
individual transition, retained for reproducibility; `cost` and
`discounted_cost` in the wrapper are the team values. Serialization includes
the scope/schema and original individual costs, so the targets cannot be
silently relabeled. Legacy replay rejects the distinct team record type.

Existing individual replay/checkpoint formats, encoder, hyperparameters,
physical objective weights, CVaR alpha/lambda, and sigmoid/argmin behavior are
preserved. The team learner reuses the tested network/optimizer/QR math while
validating the new record type and its explicitly declared gamma. It consumes
the wrapper's team `discounted_cost`, not `source.discounted_cost`, in the
Bellman target. Copying `source` into old replay would retain individual targets.

Team replay container schema 1 records the complete team-return contract,
physical cost model 2, wrapper/source records, ring cursor and owned RNG state.
Individual replay schema 2 and team replay schema 1 reject each other's states
and samples, including empty buffers with no dimensions to distinguish them.
Team checkpoint schema 3 records `learning_return_contract` and requires team
replay. Individual checkpoint schema 2 remains readable only by the individual
agent; both loaders reject cross-scope files even when tensors and encoder
signatures match. No implicit migration or weight warm start is performed.

## Reproducible interface audit

```text
python -m experiments.run_team_return_audit --output reports/NEW-team-return-audit/report.json
```

The default is two validation cases in nominal/high_load, with five fixed
policies. Each scene/policy is executed through both the individual and team
interfaces, from fresh state and identical channel/policy seeds. The runner
verifies identical actor-input hashes, actions, final physical metrics,
individual ledgers and original transition metadata. Team totals are checked
against both global disjoint intervals and every user's separate view. This
is an interface audit, not learned-policy evaluation. Test access is absent;
existing report paths are rejected. Empty cases remain included.

## Neural execution and next boundary

```text
python -m experiments.run_learning_smoke --return-scope team --algorithm qr_dqn_cvar --episodes 4 --checkpoint artifacts/team/NEW-first.pt
python -m experiments.run_learning_smoke --return-scope team --algorithm qr_dqn_cvar --episodes 2 --resume artifacts/team/NEW-first.pt --checkpoint artifacts/team/NEW-continued.pt
```

Each checkpoint destination must be new. The team smoke run kind is
`synthetic-team-learning-smoke-v1`; its metadata binds scenario/configuration,
seeds, schedule, team-return contract and actual source hash. Resume requires
the exact PyTorch release and matching run protocol, at complete episode
boundaries. This does not implement a resumable paper-scale sweep.

Runtime rejects mismatched agent/adapter scopes before resetting an episode.
Reports retain original individual `accounting` and a separate unique global
`team_accounting`; overlapping actor views are never summed as episode cost.
Greedy evaluation does not alter weights, replay, optimizer, exploration RNG
or training clock. All four existing algorithms support the team path.

The next milestone is an explicit stochastic individual/team-return ablation
protocol and an informative fixed development budget. The four-episode smoke
check is not a performance comparison or convergence study. Shared costs may
increase variance and do not prove global or static-CVaR optimality. The
evaluation objective remains the physical task sum. Additional independent
data, calibration, missing baselines and a frozen evaluation protocol remain
needed before main training.

Main training must run in Colab and must be announced to Amir before it starts.
At each completed verified implementation milestone, remind Amir about commit
and push before advancing. Push still requires his explicit authorization.
