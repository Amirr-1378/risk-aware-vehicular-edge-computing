# Cooperative Returns With Independent User-Vehicle Decisions

Amir confirmed on 2026-11-20 that **each User Vehicle independently chooses
its destination; MEC only supplies information**. This fixes the deployment
boundary. Training may aggregate simulated outcome costs offline, but no
online MEC coordinator, shared actor observation, extra messages, or hidden
queue/future information is introduced by this layer.

`agents.team_returns.TeamReturnAdapter` is an opt-in training-data interface
over the unchanged `DecisionAdapter`. It preserves the exact local raw/vector
observations, masks, action batches, physical transitions, individual task
ledgers, query/fallback protocol and completion assumptions. It creates team
return records for future cooperative independent learning. It is **not yet
connected to a neural learner**, and existing learning runners still use their
original individual-return contract. No learning-performance gain is claimed.
The [2026-11-20 verification record](team_return_validation_20261120.md)
documents the complete matched interface audit and automated test results.

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

No existing replay, encoder, checkpoint, hyperparameter, objective weight,
CVaR alpha/lambda, or sigmoid/argmin rule was changed. A future learner must
explicitly support the team-return contract in replay, protocol, reports and
checkpoint metadata. Matching vector dimensions alone cannot authorize
loading an individual-return model as a trained team-return model. A warm
start from such weights would be a separate declared experiment.

The new type is currently an in-memory data/validation contract. It does not
provide a team replay deserializer, optimizer update, checkpoint-resume path,
or new neural benchmark. In particular, copying `source` into old replay
would retain the old individual targets, not enable team learning.

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

## Next integration boundary

The next implementation milestone is a deliberately versioned team-compatible
learning/replay/checkpoint path, retaining independent local actors, followed
by a bounded training-mechanics check and matched individual/team-return
ablation. The shared cost may increase variance; it is not a proven cure for
the gap to greedy. The global evaluation objective remains the physical task
sum, never a sum of overlapping actor records. Additional independent data,
calibration, missing baselines and a frozen evaluation protocol remain needed.

Main training must run in Colab and must be announced to Amir before it starts.
At each completed verified implementation milestone, remind Amir about commit
and push before advancing. Push still requires his explicit authorization.
