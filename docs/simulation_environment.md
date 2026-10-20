# Slotted Environment: Implementation Contract

The first environment implementation joins the approved entities and physical
primitives. It is a simulator kernel with raw observations and slot costs;
it does not itself implement QR-DQN, a Gymnasium wrapper, or a calibrated
paper benchmark. Learning adapters, simple policies, and standalone risk
scoring are described in [`learning_interface.md`](learning_interface.md).
The scientific design remains in
[`scientific_model.md`](scientific_model.md) and
[`mdp_specification.md`](mdp_specification.md).

## Interface and ownership

`SlottedEnvironment` accepts an `EnvironmentConfig`, users, server vehicles,
one MEC, and a finite scheduled task trace. It validates identifiers, owners,
arrival bounds, initial empty resources, and positive MEC service capacity.
Input entities are deep-copied: simulation does not mutate scenario prototypes.

`reset(seed=...)` restores prototypes and named RNG streams and returns a
mapping from eligible user IDs to immutable `OffloadingObservation` records.
`step(actions)` requires exactly those user IDs; an empty mapping advances a
slot without decisions. All actions are validated before any assignment.

Action zero selects MEC. Positive actions resolve through that user's cached,
ID-sorted server slots. Absent slots are masked. All users decide before any
of their actions changes a resource. Admission ties are deterministic.

`StepResult` contains the elapsed slot index, next decision observations,
per-user costs, negative costs as rewards, termination, and timestamped events.
Query build/event costs incurred at a slot boundary belong to the following
slot's returned cost. A caller must therefore step through the complete drain
phase before summing episode returns. Returned decision mappings are copies;
observation records and nested snapshots are frozen.

`env.tasks`, `users`, `server_vehicles`, `mec`, `ledgers`, `wireless`, `events`,
and `metrics()` are diagnostic simulator state. They must never be passed to
a deployed policy. The first observation schema is **raw version 1**; absent
server records use `None`. The implemented `ObservationEncoder` normalizes
features and zero-pads slots without reading diagnostic truth. `DecisionAdapter`
retains costs between irregular decision times and separately records costs
before the first action. Optional neural training is implemented separately;
see [`neural_agents.md`](neural_agents.md).

## Slot ordering and discretization

At each boundary the kernel reveals that slot's arrivals, refreshes physical
links, admits pending MEC fallbacks, completes finished query phases, and
builds decisions from received, valid snapshots. Link sampling uses stable
per-link RNG names and never depends on whether a query is requested.

After the action batch, criticality-aware EDF recomputes CPU allocations at
each resource. Radio operations and CPU operations are planned from that
state. Query phases and task operations then consume actual work and energy.
No task performs more than one upload/compute/download phase in a slot.

An upload or CPU phase finishing early waits for the next slot boundary
before its next phase can start; the extra time is recorded explicitly as
`slot_boundary_wait_s`. Unselected radio/CPU tasks accrue queue time.
Transfers finishing early consume only active radio time. Final download
completion is timestamped at its actual finish inside the slot, including
zero-output completion at the next boundary. This discretization is common
to every policy and requires the planned 5/10/20 ms sensitivity study.

V2V interruptions, completions, and first deadline crossings are accounted
before advancing mobility. A detected interruption is charged in that slot;
full MEC upload starts at the next boundary. Mobility then updates the true
state for the next observation epoch. Events within an elapsed slot are
ordered by their timestamps.

The clock uses integer slot indices and decimal construction of boundary
timestamps. Cache age and absolute deadlines avoid accumulated binary-float
boundary errors. Equality is valid/on-time; the next representable timestamp
after the decimal boundary is expired/late. There is no permissive epsilon
for TTL or deadline decisions.

## Query and snapshot service

One query per user has request uplink, MEC processing, and response downlink
phases. Finished phases advance at slot boundaries; zero-size/zero-duration
phases can advance immediately at a boundary. Nonzero sizes are used in the
public smoke scenario; zero values exist for controlled test fixtures.

The snapshot is frozen when processing finishes and the response is built.
Its lifetime starts at generation, so reception consumes part of its TTL.
The response contains only currently registered in-cell servers, with actual
serialized size `header + record_count * record_bits`. All waiting tasks
share it, while query energy belongs only to the trigger task. A later task
arriving mid-query waits only from its own arrival.

The fixed query-processing duration represents a separate control-plane
service, independent of task CPU allocation; its cost is user waiting energy.
Server-to-MEC advertisements are still assumed fresh and outside the cost
model. MEC control-plane bottlenecks and advertisement staleness are future
extensions, not modeled hidden overhead.

Cached records now explicitly carry MEC nominal uplink/downlink rates,
server reserve-energy fraction, last-known relative speed, same-cell status,
and nominal downlink rate. Rates and distance estimates come from the same
large-scale link realization; fast fading is not inverted. Contact prediction
uses channel-estimated distance, reported bearing, and current registry motion
at generation. It is frozen with the other metadata.

The observed mask uses cached records only, even when live server state has
changed. Same-cell registration does not imply V2V range. Queue workload and
nominal workload/service-CPU delay are estimates of present backlog, not
M/M/1 delays or guarantees about future arrivals.

## Radio and CPU resource conventions

Each user has one radio operation per slot. An in-flight query reserves that
user's radio across its phases. Otherwise, the user's criticality-aware EDF
order selects one pending task transfer. Each server vehicle supports one
half-duplex V2V transfer per slot; contention is resolved by the same ordering.
Remote CPU execution can overlap other tasks' radio service.

V2I uplink and downlink use separate orthogonal pools with equal bandwidth
shares among active flows. MEC downlink power is shared equally too. Selected
V2V pairs share a separate aggregate orthogonal bandwidth pool. There is no
inter-link interference in this first environment. Noise is recomputed for
the actual bandwidth share rather than copying a full-band rate blindly.

CPU retains the documented equal-share allocation among EDF-selected tasks.
Completed work releases CPU at the boundary. Full server queues can exchange
waiting/active tasks atomically during preemption. If an uploaded task cannot
fit in waiting plus execution capacity, it falls back to MEC; it is not lost.
Unfinished work never migrates between server vehicles.

V2V hard interruption conditions are same-cell/range loss, unavailable
resources, no service CPU, or inability to preserve user/server reserves.
An SNR outage stalls work; the configured consecutive-outage window controls
fallback. Energy is checked before each work operation, in deterministic EDF
order when tasks compete for a server battery. An unaffordable operation is
not executed; earlier physical work and costs remain charged.

## Conditional completion guarantee and causal energy reserve

The primary kernel assumes deterministic positive V2I service, fixed positive
MEC service CPU, and user trajectories staying in one MEC region through the
drain phase. V2V may use shadowing and fading. Stochastic V2I, handover, and
user exit are explicitly rejected rather than silently granting ideal rates.
There is no artificial MEC rate floor. These assumptions must be stated when
reporting results; completion is not guaranteed for arbitrary networks.

Experiments specify `max_tasks_per_user`, `max_task_input_bits`, and
`max_task_output_bits`. Every user initially escrows worst-case MEC radio
energy for that envelope plus one worst-case query per allowed admission.
Bounds use cell-edge V2I rates, at most one flow per user, and the maximum
response size for the known server cohort. Reset rejects unfunded scenarios
and traces outside that envelope.

Crucially, runtime reserves do **not** depend on the realized future trace.
Each arrival replaces one generic escrow with that task's now-revealed bit
counts. Unused future-admission escrow is released only when the known arrival
horizon ends. Queries and MEC transfers consume their protected budgets;
V2V can consume only surplus. Query/task completion releases the unused part
of its query/fallback budget. The same configured envelope must be used for
all comparison policies and its conservatism merits sensitivity analysis.

The TTL must accommodate the worst permitted response reception delay. Each
received response can therefore support at least one admission, bounding the
number of queries by the admission envelope. Exhausted user arrival queues or
a non-draining episode guard raise explicit scenario/implementation errors;
no task is silently dropped or converted to a terminal failure.

## Accounting and validation

Per-task ledgers keep disjoint latency and mobile-energy buckets. Interrupted
V2V components move into retained-attempt buckets without being charged again.
MEC use and the first deadline miss are one-time events. Deadlines do not
cancel tasks. A completed task's accumulated cost must equal

\[
w_i\left[0.60\,T_i/D_i+0.25\,E_i/E_i^{budget}
+0.15\,1_{MEC}+2\,1_{T_i>D_i}\right]
\]

under default weights. Summed returned costs must equal summed task costs.
Total user/server battery depletion must equal recorded mobile energy.
MEC energy, server self-workload/leakage energy, and non-query user idle energy
are excluded in this implementation. Query-processing idle energy is counted
once, so simultaneous waiting tasks do not duplicate physical idle power.

Integration tests cover direct MEC and V2V completion, query sharing, aged
reception, TTL boundaries, stale masks, future-information isolation,
preemption, full queues, partial-work interruption, reserve exhaustion,
outage persistence, exact/late deadlines, drain guards, reset reproducibility,
input-order invariance, cost/energy conservation, and a seeded multi-user run.

`experiments/run_environment_smoke.py` provides a finite-JSON manifest of the
configuration, seed, Python version, initial entities/trace, terminal task
ledgers, metrics, and events. Its two fixed policies demonstrate execution;
they are not evidence of algorithmic improvement or a final baseline study.
