# Scientific Model Specification

Status: approved design baseline; entities and physical-model primitives
implemented. Environment, objective integration, and learning remain pending.
See [`physical_models.md`](physical_models.md) for implementation conventions.

This document defines the scientific assumptions, units, entities, and physical
models for the independent risk-aware vehicular edge-computing implementation.
It is the source of truth for scientific formulas. The companion
[`mdp_specification.md`](mdp_specification.md) defines observations, actions,
slot ordering, costs, and state transitions.

## 1. Scope and research question

The system offloads each indivisible computation task to exactly one of:

1. one infrastructure MEC server; or
2. one of several mobile server vehicles.

There is no local-execution action. The MEC is always available and is the
eventual fallback executor, so every admitted task completes. The learning
objective is to increase the workload completed by server vehicles, thereby
reducing MEC load, without sacrificing deadline performance, mobile energy,
or upper-tail risk.

The proposed controller is CVaR-aware QR-DQN. The reconstructed method from
the base paper is named the **paper best-response baseline** and remains an
independent comparison policy.

## 2. Approved system assumptions

| Topic | Approved assumption |
| --- | --- |
| Infrastructure | One stationary, always-available MEC server |
| Mobile compute | Multiple server vehicles inside the MEC coverage region |
| Requesters | One or more user vehicles using a shared offloading policy |
| Execution choices | MEC or one individual server vehicle |
| Local execution | Not available |
| Task splitting | Not allowed |
| Task migration | Not allowed |
| V2V recovery | Interrupted V2V work is discarded and the complete task is restarted at the MEC |
| Final task failure | Not allowed; tasks complete either on time or late |
| Hardware faults | Not modeled in the primary study |
| Time model | Time-slotted; nominal slot duration is 10 ms |
| Queue discipline | Criticality-aware, preemptive earliest-deadline-first at slot boundaries |
| Information | MEC metadata is queried, cached, timestamped, and subject to a lifetime |
| Nominal metadata lifetime | 100 ms |
| MEC energy | Excluded from the primary objective |
| Mobile energy | User-vehicle and server-vehicle energy are included |
| Risk | Upper-tail CVaR of cumulative combined cost at confidence 0.9 |

The primary adverse conditions are mobility, short V2V contact time, fading,
stale metadata, bursty arrivals, queue congestion, limited server-vehicle CPU,
and limited server-vehicle energy. Random hardware or Byzantine failures are
outside the primary scope.

## 3. Units and numerical conventions

All internal calculations use SI units.

| Quantity | Internal unit |
| --- | --- |
| Time | second (`s`) |
| Distance | metre (`m`) |
| Speed | metre per second (`m/s`) |
| Acceleration | metre per second squared (`m/s^2`) |
| Angle | radian (`rad`) |
| Data size | bit (`bit`) |
| Data rate | bit per second (`bit/s`) |
| CPU capacity/allocation | cycle per second (`cycle/s`, numerically equivalent to Hz) |
| Workload | CPU cycle (`cycle`) |
| Bandwidth | hertz (`Hz`) |
| Power | watt (`W`) |
| Energy | joule (`J`) |
| Linear channel gain | dimensionless linear ratio |
| SNR/SINR | dimensionless linear ratio internally; dB only for reporting |

Every public field carrying a physical quantity must expose the unit in its
name or docstring. Numerical inputs must be finite; silent mixing of bytes and
bits, dBm and watts, or dB and linear ratios is forbidden.

## 4. Topology and candidate server vehicles

Let the MEC have position \(p_m\) and coverage radius \(R_m\). A user vehicle
\(u\) and server vehicle \(j\) belong to the same MEC region when

\[
\lVert p_u(t)-p_m\rVert \leq R_m,
\qquad
\lVert p_j(t)-p_m\rVert \leq R_m.
\]

Membership in the same MEC region is necessary but is not sufficient for a
direct V2V link. A server is physically usable only if the V2V link and the
server's resources also satisfy the configured feasibility rules. The true
environment-side candidate set is

\[
\mathcal C_u^{\mathrm{true}}(t)=
\left\{j:
\begin{array}{l}
u\text{ and }j\text{ are inside the MEC region},\\
d_{u,j}(t)\leq R_{\mathrm{V2V}},\\
\mathrm{SINR}_{u,j}(t)\geq \gamma_{\min},\\
f_j^{\mathrm{available}}(t)>0,\\
E_j(t)>E_j^{\mathrm{reserve}}
\end{array}
\right\}.
\]

The agent does not observe this true set directly. It observes the candidate
set contained in the most recent MEC snapshot. This distinction permits stale
metadata to cause a recoverable V2V interruption.

## 5. Entity specifications

The implementation may use immutable specification records plus mutable
runtime-state records, but the following concepts must remain explicit.

### 5.1 Task

#### Immutable properties

| Field | Type | Unit | Constraint |
| --- | --- | --- | --- |
| `task_id` | `int` | none | Unique and non-negative |
| `owner_vehicle_id` | `int` | none | Must identify an existing user vehicle |
| `arrival_slot` | `int` | slot | Non-negative |
| `input_size_bits` | `int` | bit | Positive |
| `output_size_bits` | `int` | bit | Non-negative |
| `cpu_cycles_per_bit` | `float` | cycle/bit | Positive and finite |
| `deadline_s` | `float` | s | Positive and finite |
| `criticality` | enum | none | `NORMAL`, `HIGH`, or `CRITICAL` |
| `energy_budget_j` | `float` | J | Positive normalization scale |

The required computation is derived, not independently configured:

\[
C_i = S_i^{\mathrm{in}}\,c_i,
\]

where \(S_i^{\mathrm{in}}\) is input size in bits and \(c_i\) is cycles per
input bit.

#### Mutable state

| Field | Unit | Meaning |
| --- | --- | --- |
| `status` | none | Lifecycle state |
| `initial_target_id` | none | First selected MEC or server-vehicle identifier |
| `final_executor_id` | none | Executor that ultimately completes the task |
| `remaining_upload_bits` | bit | Input still to transmit |
| `remaining_cpu_cycles` | cycle | Computation still to execute |
| `remaining_download_bits` | bit | Result still to return |
| `query_wait_time_s` | s | Time spent waiting for metadata refresh |
| `query_energy_j` | J | User energy attributable to query processing |
| `v2v_attempt_delay_s` | s | Time consumed before an interrupted V2V attempt |
| `v2v_attempt_energy_j` | J | Mobile energy consumed before interruption |
| `fallback_used` | none | Whether MEC fallback occurred |
| `deadline_missed` | none | One-time deadline-crossing flag |
| `completion_time_s` | s or `None` | Actual completion time |

Permitted terminal states are `COMPLETED_ON_TIME` and `COMPLETED_LATE`.
There is no `DROPPED` or terminal `FAILED` state.

### 5.2 UserVehicle

#### Immutable properties

| Field | Unit |
| --- | --- |
| `vehicle_id` | none |
| `max_tx_power_w` | W |
| `rx_power_w` | W |
| `idle_power_w` | W |
| `battery_capacity_j` | J |
| `radio_antenna_gain_linear` | linear ratio |
| `max_task_queue_size` | task |
| `mobility_profile_id` | none |

#### Mutable state

| Field | Unit |
| --- | --- |
| `position_m` | m, two-dimensional |
| `speed_mps` | m/s |
| `heading_rad` | rad |
| `acceleration_mps2` | m/s^2 |
| `battery_energy_j` | J |
| `tx_power_w` | W |
| `task_queue` | task identifiers |
| `cached_mec_snapshot` | metadata snapshot or `None` |
| `snapshot_generated_time_s` | s or `None` |
| `snapshot_received_time_s` | s or `None` |
| `query_status` | none |
| `query_remaining_time_s` | s |
| `query_owner_task_id` | task identifier or `None` |

The user vehicle does not execute tasks locally. Its energy consumption comes
from query construction, radio transmission, waiting/idle operation, and radio
reception.

### 5.3 ServerVehicle

#### Immutable properties

| Field | Unit |
| --- | --- |
| `vehicle_id` | none |
| `cpu_capacity_hz` | cycle/s |
| `min_self_reserved_cpu_hz` | cycle/s |
| `max_concurrent_tasks` | task |
| `max_queue_size` | task |
| `max_tx_power_w` | W |
| `rx_power_w` | W |
| `cpu_energy_coefficient` | J/(cycle Hz^2), per documented CPU model |
| `battery_capacity_j` | J |
| `min_reserve_energy_j` | J |
| `radio_antenna_gain_linear` | linear ratio |
| `mobility_profile_id` | none |

#### Mutable state

| Field | Unit |
| --- | --- |
| `position_m` | m, two-dimensional |
| `speed_mps` | m/s |
| `heading_rad` | rad |
| `acceleration_mps2` | m/s^2 |
| `battery_energy_j` | J |
| `self_workload_cpu_hz` | cycle/s |
| `service_queue` | task identifiers |
| `active_task_ids` | task identifiers |
| `cpu_allocations_hz` | task identifier to cycle/s |
| `availability` | none |

Available service CPU is

\[
f_j^{\mathrm{available}} =
f_j^{\mathrm{capacity}}
-f_j^{\mathrm{self-reserved}}
-f_j^{\mathrm{self-workload}}
-\sum_i f_{i,j}^{\mathrm{allocated}}.
\]

Price, fixed service-quality scores, certificates, consensus leadership, and
fault flags are not part of the core server-vehicle model.

### 5.4 MECServer

#### Immutable properties

| Field | Unit |
| --- | --- |
| `server_id` | none |
| `position_m` | m, two-dimensional |
| `coverage_radius_m` | m |
| `cpu_capacity_hz` | cycle/s |
| `max_concurrent_tasks` | task |
| `total_uplink_bandwidth_hz` | Hz |
| `total_downlink_bandwidth_hz` | Hz |
| `max_downlink_power_w` | W |
| `query_processing_time_s` | s |

#### Mutable state

| Field | Unit |
| --- | --- |
| `service_queue` | task identifiers |
| `active_task_ids` | task identifiers |
| `cpu_allocations_hz` | task identifier to cycle/s |
| `uplink_allocations_hz` | user identifier to Hz |
| `downlink_allocations_hz` | user identifier to Hz |
| `background_cpu_load_hz` | cycle/s |
| `registered_server_vehicle_ids` | identifiers |
| `latest_server_information` | timestamped records |

The MEC has finite CPU and radio resources, so its queue delay and utilization
can grow. It does not reject tasks and does not fail in the primary model.
`MECServer` is a compute/information service and must not be conflated with a
blockchain `MECNode` or consensus peer.

## 6. MEC information snapshots and lifetime

The MEC is assumed to maintain a sufficiently fresh server-vehicle registry
from periodic control-plane advertisements. The cost and staleness of the
server-vehicle-to-MEC advertisement plane are outside the first model. The
modeled freshness boundary is the MEC-to-user snapshot cache.

At response-build time \(t_g\), the MEC produces an immutable snapshot

\[
M_u(t_g)=\left(M^{\mathrm{MEC}}(t_g),
\{M_j^{\mathrm{SV}}(t_g)\}_{j\in\mathcal C_m(t_g)}\right).
\]

The response becomes usable only after it is received at time \(t_r\), where
\(t_r\geq t_g\). Information age is measured from generation, not reception:

\[
A_u(t)=t-t_g.
\]

With information lifetime \(L_{\mathrm{info}}\), the cache is valid when

\[
A_u(t)\leq L_{\mathrm{info}}.
\]

A query is event-driven:

- no query is sent when the user has no task requiring an offloading decision;
- a task uses the cached snapshot while it is valid;
- a missing or expired snapshot starts one query;
- tasks awaiting the same in-flight query share the response and wait for its
  completion;
- at most one query per user may be in flight.

The response contains MEC capacity and queue information plus one record per
registered server vehicle. A server record contains at least its identifier,
record timestamp, available CPU, queue workload, battery fraction, estimated
V2V SNR/rate/distance, and estimated contact time.

If \(S_q\) is request size and \(S_r\) is response size, query latency is

\[
T_{\mathrm{query}}=
\frac{S_q}{R_{u,m}^{\mathrm{UL}}}
+T_{\mathrm{process}}^{\mathrm{MEC}}
+\frac{S_r}{R_{m,u}^{\mathrm{DL}}}.
\]

Response size scales with the number of records:

\[
S_r=S_{\mathrm{header}}+N_{\mathrm{records}}S_{\mathrm{record}}.
\]

User-side query energy is

\[
E_{\mathrm{query}}=
E_{\mathrm{build}}
+P_u^{\mathrm{tx}}T_{\mathrm{request}}
+P_u^{\mathrm{idle}}T_{\mathrm{process}}^{\mathrm{MEC}}
+P_u^{\mathrm{rx}}T_{\mathrm{response}}.
\]

Query latency contributes to every waiting task's elapsed time. Query energy is
charged once to the user and attributed to the task that triggered the query
(`query_owner_task_id`), not once per waiting task. All tasks that arrive while
the query is in flight still accumulate their own waiting-time latency.

## 7. Mobility, channel, and distance estimation

### 7.1 Mobility

For slot duration \(\Delta t\), a basic kinematic update is

\[
p(t+\Delta t)=p(t)+v(t)\Delta t
+\tfrac{1}{2}a(t)\Delta t^2.
\]

The configured mobility model is responsible for road/lane constraints and
for all random draws. Randomness must be obtained from an explicitly injected,
seeded random-number generator.

### 7.2 Channel rate

For allocated bandwidth \(B\) and linear SINR \(\gamma\), the idealized rate
is

\[
R=B\log_2(1+\gamma).
\]

V2V and V2I links may have different path-loss and fading parameters. Uplink
and downlink are not assumed identical unless an experiment configuration
explicitly makes them so.

### 7.3 True and observed distance

The environment computes true geometric distance

\[
d_{u,j}^{\mathrm{true}}(t)=\lVert p_u(t)-p_j(t)\rVert
\]

for mobility and channel evolution. This value is hidden from the agent.

Under a large-scale channel model,

\[
\overline\gamma(d)=
\frac{P_tG_tG_r\beta_0d^{-\alpha}}{N+I}.
\]

The MEC estimates distance from a filtered large-scale SNR:

\[
\widehat d=
\left(
\frac{P_tG_tG_r\beta_0}
{\overline\gamma(N+I)}
\right)^{1/\alpha}.
\]

For free-space path loss, \(\alpha=2\) and
\(\beta_0=(\lambda/(4\pi))^2\). Instantaneous fast-fading SNR must not be
inverted directly. The estimator uses filtered/averaged SNR and therefore
remains noisy under shadowing and model mismatch.

Rate, SNR, and estimated distance must originate from one consistent channel
realization; they are not sampled as independent features.

## 8. Queueing and execution

Scheduling is lexicographic by

1. criticality (`CRITICAL`, `HIGH`, `NORMAL`);
2. earliest absolute deadline;
3. earliest arrival slot; and
4. task identifier as a deterministic final tie-breaker.

Execution may be preempted only at slot boundaries. Preemption preserves
remaining CPU cycles and does not change the selected executor. This is
compatible with indivisible offloading: execution is time-sliced, but the task
is not split across executors.

Queue workload is tracked in CPU cycles. For a resource serving
\(C^{\mathrm{served}}_t\) cycles and receiving \(C^{\mathrm{arrived}}_t\)
cycles,

\[
Q_{t+1}=\max(0,Q_t-C^{\mathrm{served}}_t)+C^{\mathrm{arrived}}_t.
\]

An analytical M/M/1 delay may be implemented only as a separately named
baseline model; it must not replace the explicit time-slotted queue.

## 9. Latency model

For task \(i\) executed by remote resource \(r\), total latency is

\[
T_i=
T_i^{\mathrm{query}}
+T_i^{\mathrm{queue}}
+\frac{S_i^{\mathrm{in}}}{R_{u,r}^{\mathrm{UL}}}
+\frac{C_i}{f_{i,r}^{\mathrm{allocated}}}
+\frac{S_i^{\mathrm{out}}}{R_{r,u}^{\mathrm{DL}}}
+T_i^{\mathrm{fallback}}.
\]

To avoid double counting, the queue/upload/compute/download terms refer to the
final successful attempt. The fallback term contains only the discarded V2V
attempt's elapsed time and any gap before MEC admission; subsequent MEC queue
and service time already appear in the final-attempt terms. Any initial local
admission wait must also be included once. Latency is measured from task
arrival, not from offloading admission. The implementation uses disjoint
`LatencyBreakdown` buckets detailed in `physical_models.md`.

## 10. Energy model

The primary mobile-energy objective includes user-vehicle query/radio energy
and server-vehicle receive/compute/transmit energy. MEC computation and MEC
radio energy are excluded from the primary cost but may be reported
separately.

Transmission and reception energy are

\[
E^{\mathrm{tx}}=P^{\mathrm{tx}}T^{\mathrm{tx}},
\qquad
E^{\mathrm{rx}}=P^{\mathrm{rx}}T^{\mathrm{rx}}.
\]

Server-vehicle CPU energy uses

\[
E^{\mathrm{cpu}}=\kappa C f^2,
\]

where \(C\) is executed cycles, \(f\) is allocated CPU frequency, and
\(\kappa\) has the documented units required to produce joules.

Thus,

\[
E_i^{\mathrm{mobile,MEC}}=
E_u^{\mathrm{query}}+E_u^{\mathrm{upload}}+E_u^{\mathrm{receive}},
\]

and

\[
E_i^{\mathrm{mobile,V2V}}=
E_u^{\mathrm{query}}+E_u^{\mathrm{transmit}}+E_u^{\mathrm{receive}}
+E_j^{\mathrm{receive}}+E_j^{\mathrm{compute}}+E_j^{\mathrm{transmit}}.
\]

All energy consumed before a V2V interruption remains charged when the task is
restarted at the MEC.

To preserve the invariant that every admitted task eventually completes, user
vehicles reserve enough energy for a worst-case MEC fallback under the active
experiment bounds. The primary model does not terminate a task because its
requesting user's battery reached zero. Server vehicles may become unavailable
when their configured reserve would be violated.

## 11. V2V interruption and MEC fallback

A V2V service interruption can occur when the true link or resource state no
longer supports the cached decision, for example because the vehicles leave
contact range or the link remains below its service threshold.

The recovery transition is

```text
V2V_IN_PROGRESS
    -> V2V_SERVICE_INTERRUPTED
    -> FALLBACK_TO_MEC
    -> MEC_QUEUED / MEC_IN_PROGRESS
    -> COMPLETED_ON_TIME or COMPLETED_LATE
```

Partial upload, computation, or result data from the interrupted V2V attempt
is discarded. The MEC re-executes the complete task. The physical time and
mobile energy already consumed are retained in cost accounting.

No arbitrary fallback penalty is used in the primary model. Wasted time,
wasted energy, eventual MEC usage, and any deadline miss provide the learning
signal. A separate fallback-penalty experiment may be added only as an
explicit ablation.

## 12. Combined cost and risk

For task \(i\), define normalized latency and mobile energy

\[
\widetilde T_i=\frac{T_i}{D_i},
\qquad
\widetilde E_i=\frac{E_i^{\mathrm{mobile}}}{E_i^{\mathrm{budget}}}.
\]

Let \(M_i=1\) when the MEC ultimately executes the task, including fallback,
and zero when a server vehicle completes it. Let criticality weight be 1, 2,
or 4 for normal, high, or critical tasks. The primary task cost is

\[
z_i=w_i^{\mathrm{criticality}}
\left[
0.60\widetilde T_i
+0.25\widetilde E_i
+0.15M_i
+2\,\mathbf 1(T_i>D_i)
\right].
\]

Weights are fixed before test-set evaluation and shared by every policy. The
pre-registered sensitivity grid is

- MEC-use weight: 0.00, 0.10, 0.15, 0.20;
- latency/energy alternatives: 0.50/0.50, 0.70/0.30, 0.90/0.10 when the
  MEC-use term is disabled or reported as a separate experiment;
- deadline penalty: 1, 2, 4.

For cumulative discounted cost \(G_t^c\), upper-tail CVaR at confidence
\(\alpha=0.9\) is

\[
\operatorname{CVaR}_{0.9}(G_t^c)=
\operatorname{VaR}_{0.9}(G_t^c)
+\frac{1}{0.1}
\mathbb E[(G_t^c-\operatorname{VaR}_{0.9}(G_t^c))_+].
\]

The action score is the mean-CVaR mixture

\[
J(s,a)=(1-\lambda_r)\mathbb E[G_t^c\mid s,a]
+\lambda_r\operatorname{CVaR}_{0.9}(G_t^c\mid s,a),
\]

with nominal \(\lambda_r=0.5\). The controller minimizes \(J\). The risk
sensitivity grid is 0, 0.25, 0.5, 0.75, and 1. A value of zero is the
risk-neutral QR-DQN ablation.

## 13. Evaluation metrics

Primary metrics are

- workload-weighted MEC share (MEC-executed cycles / total cycles);
- V2V-completed workload share;
- direct-MEC task ratio;
- MEC-fallback task ratio;
- V2V interruption ratio;
- on-time completion ratio;
- critical-task deadline-miss ratio;
- mean, p95, and p99 latency;
- mean mobile energy;
- mean cost and empirical CVaR at confidence 0.9;
- mean and peak MEC queue workload/utilization;
- mean information age, query rate, query latency, and query energy.

Task completion rate is expected to be 100% by construction and therefore is
an invariant, not the principal performance claim.

## 14. Primary baselines and ablations

1. Always MEC.
2. Always V2V with MEC fallback.
3. Uniform random over observed feasible actions.
4. Greedy minimum one-step expected cost.
5. Paper best-response baseline with an explicitly documented adaptation for
   selecting an individual server vehicle.
6. DQN.
7. Double DQN.
8. Risk-neutral QR-DQN.
9. CVaR-aware QR-DQN (the proposed controller).

An adapted L-DDQL comparison is optional for the final experimental study. If
implemented, it must be labeled as adapted because its original local/MEC
action space differs from this project's MEC/multi-V2V action space.

## 15. Non-goals for the first implementation

- local execution;
- partial offloading;
- task dropping;
- learned query timing;
- stale MEC-side server registry;
- separate lifetime for every metadata field;
- hardware, Byzantine, or consensus failure;
- blockchain overhead in the primary experiments;
- pricing/payment in the proposed controller;
- semantic-content inference or a semantic classifier;
- advanced localization, triangulation, or multi-MEC handover.

Task labels are called **criticality**, not semantic awareness, unless a real
semantic inference component is added later.

## 16. Required scientific tests

Every formula and state-changing rule above requires a deterministic test.

| Area | Minimum required checks |
| --- | --- |
| Units | Known conversions; reject non-finite and dimensionally invalid inputs |
| Task workload | Exact `input_bits * cycles_per_bit`; no negative remaining work |
| Geometry | Known Euclidean distances and MEC/V2V boundary cases |
| Channel | Monotonic path loss; Shannon-rate fixtures; dB/linear round trips |
| Distance estimate | Exact inversion without noise; bounded/noisy behavior with shadowing |
| Query | Request/processing/response latency; energy sum; response-size scaling |
| Cache | Fresh-at-boundary, expired-after-boundary, one in-flight query |
| Queue | Criticality order, EDF order, deterministic tie breaking, preemption |
| Latency | Upload, queue, compute, download, query, and fallback components |
| Energy | User query/radio and server receive/CPU/transmit components |
| Fallback | V2V work discarded; full MEC workload restored; prior cost retained |
| Deadline | Penalty applied once; late task continues to completion |
| Completion | No admitted task is dropped; episode drains all queues |
| Risk | VaR/CVaR fixtures for hand-computed discrete distributions |
| Reproducibility | Equal seeds reproduce trajectories; independent RNG streams stay isolated |

## 17. References used to define the model

- W. Dabney et al., "Distributional Reinforcement Learning with Quantile
  Regression," AAAI 2018, <https://arxiv.org/abs/1710.10044>.
- K. Wang, N. Kallus, and W. Sun, "Near-Minimax-Optimal Risk-Sensitive
  Reinforcement Learning with CVaR," ICML 2023,
  <https://proceedings.mlr.press/v202/wang23m.html>.
- 3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to
  100 GHz," ETSI publication.
- 3GPP TS 22.186, "Service requirements for enhanced V2X scenarios," ETSI
  publication.
- S. Park et al., "Age-of-Information Aware Contents Caching and Distribution
  for Connected Vehicles," <https://arxiv.org/abs/2210.01536>.
