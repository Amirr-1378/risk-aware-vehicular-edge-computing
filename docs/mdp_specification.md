# Time-Slotted Control and MDP Interface Specification

Status: approved design baseline; implementation pending.

This document defines the learning-facing interface and the exact ordering of
time-slotted state transitions. Scientific formulas and entity units are
defined in [`scientific_model.md`](scientific_model.md).

## 1. Control formulation

The simulator has a fully specified Markov environment state \(X_t\), including
true vehicle positions, true queues, true channel state, and true resource
availability. A user does not observe \(X_t\) directly. It receives an
age-augmented observation \(O_{u,t}\) built from its own current state and its
most recent timestamped MEC snapshot.

The implementation must preserve this distinction:

```text
true environment state X_t
    -> channel, mobility, service, and feasibility

cached user observation O_u,t
    -> action masking visible to the policy
    -> QR-DQN input
```

Consequently, the learning problem is partially observed from the user's
perspective. Including snapshot age, last-known motion, link estimates, and
resource estimates reduces but does not eliminate partial observability. The
paper and code must not claim that cached features are current ground truth.

The first controller is a shared task-offloading policy: every user vehicle
uses the same QR-DQN parameters. Multiple users may request decisions in one
slot. Their actions are computed from one common pre-admission environment
snapshot and applied as a batch, preventing user-order bias. A one-user
configuration is a valid special case of the same interface.

## 2. Time base and episode structure

Nominal slot duration is

\[
\Delta t=0.010\ \mathrm{s}.
\]

An episode has two phases:

1. **arrival phase:** new tasks, mobility, queries, and service are active for
   the configured horizon;
2. **drain phase:** arrivals stop, while queries, transmissions, queues, and
   computation continue until all admitted tasks complete.

The episode terminates only when

- the arrival phase has ended;
- every user task queue is empty;
- every MEC/server queue is empty;
- no task is transmitting, executing, downloading, or awaiting fallback; and
- no in-flight query is required by a waiting task.

A large implementation guard may detect a non-draining bug, but it must raise
an error rather than silently drop tasks.

## 3. Per-slot transition order

All policies use the following order. No policy may observe a value generated
later in the same slot.

### Slot start: state \(X_t\)

1. Reveal task arrivals scheduled for slot \(t\).
2. Complete query phases whose remaining work reached zero at the previous
   slot end; install their timestamped MEC snapshots.
3. For each user with an unassigned task, inspect the user cache.
4. If the cache is missing or expired, start or continue one query and keep
   affected tasks waiting.
5. If the cache is valid, select at most one unassigned task per user using
   criticality-aware EDF and construct its observation and observed action
   mask.
6. Compute every active user's offloading action from the same pre-admission
   state.
7. Apply the action batch to MEC/server queues.
8. Recompute scheduler allocations using criticality-aware preemptive EDF.

### During slot \([t,t+\Delta t)\)

9. Progress query request, MEC processing, and query response work.
10. Progress task upload, queue service, computation, and result download
    using the current slot's physical channel and resource allocations.
11. Accumulate physical elapsed time and mobile energy.

### Slot end: produce \(X_{t+1}\)

12. Detect V2V service interruption from the true physical state; mark an
    automatic MEC fallback for admission at the next slot start.
13. Mark task completions.
14. Detect first deadline crossings and charge each deadline penalty once;
    late tasks remain active.
15. Update server/user energy states.
16. Advance mobility and stochastic channel state to \(t+1\) using explicit
    seeded RNG streams.
17. Construct per-user incremental costs and the next observations.

This order prevents future position, future fading, future arrivals, and true
post-query server state from leaking into the action.

## 4. Query state machine

Each user has one of

```text
NO_CACHE
CACHE_VALID
CACHE_EXPIRED
QUERY_REQUEST_UPLINK
QUERY_MEC_PROCESSING
QUERY_RESPONSE_DOWNLINK
```

Transitions are

```text
NO_CACHE + waiting task
    -> QUERY_REQUEST_UPLINK

CACHE_VALID + age <= lifetime
    -> use cached observation

CACHE_VALID + age > lifetime
    -> CACHE_EXPIRED

CACHE_EXPIRED + waiting task
    -> QUERY_REQUEST_UPLINK

QUERY_REQUEST_UPLINK complete
    -> QUERY_MEC_PROCESSING

QUERY_MEC_PROCESSING complete
    -> build snapshot -> QUERY_RESPONSE_DOWNLINK

QUERY_RESPONSE_DOWNLINK complete
    -> install snapshot -> CACHE_VALID
```

The response is unavailable until reception completes. Once installed, cache
validity uses the snapshot's generation timestamp, not its later reception
timestamp:

\[
t-t_{\mathrm{generated}}\leq L_{\mathrm{info}}
\quad\Longrightarrow\quad\text{valid}.
\]

The exact-boundary case is therefore valid. The next representable time beyond
the boundary is expired.

Snapshot content and `generated_time_s` are fixed at response-build time.
Response transmission therefore consumes part of the snapshot lifetime before
the user can first use it. Changes after generation remain hidden until the
next query.

## 5. Task lifecycle

The task lifecycle is

```text
CREATED
    -> WAITING_FOR_INFORMATION
    -> READY_FOR_OFFLOADING
    -> UPLOADING_TO_MEC or UPLOADING_TO_SERVER_VEHICLE
    -> MEC_QUEUED or V2V_QUEUED
    -> MEC_EXECUTING or V2V_EXECUTING
    -> DOWNLOADING_FROM_MEC or DOWNLOADING_FROM_SERVER_VEHICLE
    -> COMPLETED_ON_TIME or COMPLETED_LATE
```

Any active V2V phase may transition through

```text
V2V_SERVICE_INTERRUPTED
    -> FALLBACK_PENDING
    -> UPLOADING_TO_MEC
```

Fallback restores the complete original input, CPU workload, and result size
for MEC execution. It does not erase elapsed time or energy.

## 6. Decision eligibility

A user requests an action only when

- it has at least one unassigned task;
- it has a valid MEC snapshot;
- no earlier action for that task is pending; and
- the task has not already been assigned.

At most one new task per user is assigned per slot. The user's local admission
order is the same criticality-aware EDF ordering used by compute resources.
Tasks not selected for admission remain active and accrue elapsed-time cost.

Users without an eligible decision do not choose a learned `NO_OP`; the
environment simply omits them from that slot's action batch.

## 7. Action space

Let \(K_{\max}\) be the maximum number of server-vehicle slots exposed to the
network. The fixed neural-network output space is

\[
\mathcal A=\{0,1,\ldots,K_{\max}\},
\]

where

- action 0 selects the MEC;
- action \(j\) selects the server vehicle assigned to snapshot slot \(j\).

Server slots use deterministic ordering, preferably ascending stable vehicle
identifier. Absent slots are zero-padded in the observation and masked.

### 7.1 Observed action mask

MEC action 0 is always valid. A server action is visible as valid only when the
cached snapshot says that the server

- belongs to the user's MEC region;
- is within estimated V2V range;
- meets the cached minimum link-quality threshold;
- has positive cached service CPU;
- has cached energy above its reserve; and
- is marked available in the snapshot.

The mask must use cached records only. It must not use the true current
candidate set. A cached-valid action can therefore fail physically after it is
selected, producing a V2V interruption and MEC fallback.

Masking is for impossible or cache-declared unavailable actions, not merely
risky actions. Queue congestion, short estimated contact margin, and high
uncertainty remain visible to the policy so that risk-sensitive selection can
be learned.

## 8. Observation

Every raw and normalized observation schema must be versioned. At minimum,
the task-centric observation for user \(u\) contains the following.

### 8.1 Task features

- input size in bits;
- output size in bits;
- required and remaining CPU cycles;
- absolute deadline slack in seconds;
- criticality encoding;
- energy budget;
- elapsed time normalized by deadline.

### 8.2 User features

- battery fraction;
- speed and heading;
- distance to MEC or relative MEC coordinates;
- transmit-power fraction;
- number/workload of unassigned local tasks.

### 8.3 Information-freshness features

- snapshot age in seconds;
- response age since reception in seconds, for diagnostics;
- remaining lifetime in seconds;
- normalized age \(A_u/L_{\mathrm{info}}\);
- query state; and
- query remaining time when a query is active.

An offloading action is not requested while a required query is active, but
query features remain part of per-slot diagnostic observations and future
extensions.

### 8.4 Cached MEC features

- available CPU;
- queue workload in cycles;
- estimated queue delay;
- estimated uplink and downlink rates;
- MEC utilization.

### 8.5 Cached server-vehicle features, repeated \(K_{\max}\) times

- presence bit and observed-action-mask bit;
- record age;
- estimated distance to user;
- estimated V2V SNR and rate;
- last-known relative speed;
- estimated contact time;
- contact margin (estimated contact time / estimated service time);
- available CPU;
- queue workload;
- battery fraction and energy-reserve fraction;
- cached availability.

No true current distance, true current queue, future mobility, or future
channel value may appear in the observation.

## 9. Batch action application

When several users act in one slot:

1. every user observes the same pre-admission MEC/server resource snapshot;
2. every policy action is computed before any of those actions changes a
   queue;
3. actions are then admitted as one deterministic batch;
4. batch ties at a resource are ordered by task criticality, deadline,
   arrival slot, user identifier, and task identifier.

This makes simultaneous decisions reproducible and avoids giving an arbitrary
first user fresher resource information.

## 10. Incremental cost and reward

The task-level cost in the scientific model is implemented incrementally.
For user \(u\) in slot \(t\), let \(\mathcal I_{u,t}\) be its active tasks.
An appropriate decomposition is

\[
c_{u,t}=
\sum_{i\in\mathcal I_{u,t}}
w_i^{\mathrm{crit}}
\left[
0.60\frac{\Delta T_{i,t}}{D_i}
+0.25\frac{\Delta E_{i,t}^{\mathrm{mobile}}}{E_i^{\mathrm{budget}}}
\right]
+c_{u,t}^{\mathrm{events}},
\]

where one-time event cost is

\[
c_{u,t}^{\mathrm{events}}=
\sum_i w_i^{\mathrm{crit}}
\left[
0.15\,\mathbf 1(\text{MEC becomes final executor in }t)
+2\,\mathbf 1(\text{first deadline crossing in }t)
\right].
\]

Query energy is counted once in \(\Delta E_{u,t}^{\mathrm{mobile}}\) and
attributed to the query-triggering task. Every task waiting for that query
independently accumulates its own \(\Delta T\).

The learning reward is

\[
r_{u,t}=-c_{u,t}.
\]

Scientific reporting remains in cost space to avoid ambiguity about reward
tails.

## 11. QR-DQN and CVaR action score

QR-DQN estimates a quantile approximation to the return distribution. In cost
notation,

\[
G_t^c=\sum_{k=0}^{H-t-1}\gamma^k c_{t+k}.
\]

For an ordered set of estimated cost quantiles \(q_1\le\cdots\le q_N\), the
empirical upper-tail CVaR at confidence 0.9 is the mean of quantiles in the top
10% tail, with fractional boundary weighting if necessary.

The nominal action score is

\[
J(s,a)=0.5\,\operatorname{mean}(q_1,\ldots,q_N)
+0.5\,\operatorname{CVaR}_{0.9}(q_1,\ldots,q_N),
\]

and the valid action with minimum score is selected outside exploration.

If the implementation trains reward quantiles instead, it must use the lower
10% reward tail and maximize the sign-equivalent score. Unit tests must prove
the cost-space and reward-space selectors choose the same action.

## 12. Interruption and fallback rules

A selected V2V action is not labeled a final task failure. A service
interruption is an environment event. It may be triggered by configured
physical conditions such as

- loss of same-MEC-region membership;
- loss of V2V range;
- sustained SINR below the service threshold; or
- inability to preserve the configured server energy reserve.

The exact threshold and any outage persistence window are experiment
parameters. The primary model contains no random hardware-failure coin flip.

At interruption:

1. preserve accumulated task time and mobile energy;
2. clear all partial V2V transfer and compute progress;
3. remove the task from the server queue/allocation;
4. mark MEC as the final executor;
5. admit the full task to MEC at the next slot boundary; and
6. charge the MEC-use term once.

## 13. Deadline semantics

Deadlines are relative to task arrival. Equality is on time:

\[
T_i\le D_i\quad\Longrightarrow\quad\text{on time}.
\]

The first transition from \(T_i\le D_i\) to \(T_i>D_i\) sets
`deadline_missed`, charges the deadline penalty once, and keeps the task
active. Completion later produces `COMPLETED_LATE`.

## 14. Randomness and reproducibility

Use independent RNG streams derived from one documented experiment seed for

- task arrivals and task parameters;
- mobility;
- large-scale shadowing;
- small-scale fading;
- background MEC/server workloads; and
- agent exploration/replay sampling.

Changing one subsystem's number of random draws must not silently perturb
unrelated subsystems. Tests must verify stream isolation.

## 15. Required transition-test matrix

| Scenario | Required result |
| --- | --- |
| Generated snapshot age exactly equals lifetime | Cache is valid; no query starts |
| Cache age exceeds lifetime | One query starts when a task waits |
| Long response transmission | Snapshot arrives with positive age and reduced remaining lifetime |
| Expired cache with no waiting task | No query starts |
| Two tasks wait during one query | Query energy belongs only to trigger task; both tasks accrue waiting time |
| Server absent from cached snapshot | Its action is padded/masked |
| Server true state changes but cache is fresh | Observation and mask remain cached; no truth leakage |
| Cached-valid server is physically invalid at service | Interruption then full MEC fallback |
| No cached-valid server | Only MEC action remains valid |
| Multiple users act in one slot | Actions use common pre-admission state and deterministic batch admission |
| Critical task arrives behind normal task | It preempts at the next slot boundary |
| Equal criticality, different deadlines | Earlier deadline runs first |
| Deadline reached exactly | Task remains on time |
| Deadline crossed | One penalty only; task continues |
| V2V interruption after partial compute | Partial cycles discarded; MEC receives full original cycles |
| V2V interruption after energy use | Prior energy remains charged |
| Direct MEC completion | MEC-use cost charged exactly once |
| Fallback MEC completion | MEC-use cost charged exactly once, not twice |
| Arrival horizon ends with active work | Drain phase continues |
| Final task completes late | Episode terminates only after queues and active phases empty |
| All server actions masked | MEC still valid; action set is never empty |
| User nears its fallback-energy reserve | No action may consume the protected MEC-fallback reserve |
| Same master seed | Identical trajectory and costs |
| Different fading seed only | Arrival and mobility traces remain identical |

## 16. Experiment-facing configuration defaults

These are defaults, not hidden constants. Every experiment stores them in a
versioned configuration.

| Parameter | Nominal value | Sensitivity |
| --- | --- | --- |
| Slot duration | 10 ms | 5, 10, 20 ms |
| Information lifetime | 100 ms | 50, 100, 200 ms |
| CVaR confidence | 0.9 | report at least 0.9; optional 0.95 |
| Risk-mixture weight | 0.5 | 0, 0.25, 0.5, 0.75, 1 |
| Criticality weights | 1, 2, 4 | fixed primary; ablate if needed |
| Deadline penalty | 2 | 1, 2, 4 |
| MEC-use weight | 0.15 | 0, 0.10, 0.15, 0.20 |

The maximum number of exposed server-vehicle slots, task distributions,
channel scenario, mobility traces, outage persistence window, and episode
arrival horizon must be selected in experiment configurations before agent
training. They are not entity defaults.
