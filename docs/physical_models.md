# Physical Models: Implementation Contracts

This milestone implements the primitives in `src/risk_aware_v2x/models/`.
The environment and learning agent are subsequent milestones. All quantities
use SI units, and these primitives introduce no calibrated traffic or radio
scenario. Experiments must select those parameters explicitly.

## Modules and numerical conventions

| Module | Implemented behavior |
| --- | --- |
| `units` | Byte/bit, dB/linear power-ratio, and dBm/watt conversions |
| `mobility` | 2D distance, straight-road motion, braking to rest, finite-horizon contact prediction |
| `channel` | Reference-distance path loss, SINR, Shannon rate, thermal noise, optional shadowing and Rayleigh fading, filtered distance inversion |
| `latency` | Transmission/compute/query time, query response size, bounded slot progress, non-overlapping elapsed-time accounting |
| `energy` | Radio/idle energy, task CPU energy, query energy, mobile-energy accounting |
| `queueing` | Criticality-aware EDF, queue workload, slot-boundary preemption and CPU allocation |
| `randomness` | Independent named `random.Random` streams from one master seed |

Inputs reject booleans where a physical number is required, negative sizes or
rates, and non-finite values. Predicted service time is the sole intentional
exception for outputs: positive work at zero service rate returns infinity.
Actual elapsed-time and energy records remain finite. Zero work at zero rate
needs zero time. dB conversion requires positive power; zero power has no
finite dB representation.

## Mobility and contact prediction

`MotionState` stores position, scalar speed, heading, and signed tangential
acceleration. Heading is counterclockwise from the positive x axis.
`advance_straight_line` applies the documented constant-acceleration equation
while holding heading fixed. During braking it integrates only until speed
reaches zero, then holds position for the rest of the step. Negative speed or
unintended reverse motion is therefore impossible. `advance_vehicle` validates
the complete result before installing it in a user or server vehicle.

Contact prediction holds the current relative velocity constant and solves

\[
\lVert r+vt\rVert^2=R_{\mathrm{V2V}}^2
\]

for the first non-negative exit time from current contact. It returns zero for
a pair already outside the range and caps predictions at the caller's finite
`horizon_s`; stationary pairs in range return that horizon. It predicts only
V2V range exit, not channel outage or MEC-region exit. Acceleration, turns, and
future draws are not used. Policy-visible predictions must use cached motion
records. Road layout, intersections, and turning traces remain experiment and
environment responsibilities.

## Channel and information consistency

The analytical path gain is

\[
g(d,X)=\left(\frac{c}{4\pi f_c d_0}\right)^2
\left(\frac{\max(d,d_0)}{d_0}\right)^{-\alpha}10^{-X/10}.
\]

The reference distance `d0` prevents a singularity at zero separation. The
carrier, exponent, and optional shadowing deviation are explicit parameters.
Positive `X` is additional attenuation. With `alpha=2` and no shadowing this
matches the [free-space loss equation](https://www.mathworks.com/help/phased/ref/fspl.html)
for distances at least `d0`. The reference distance is a numerical modeling
boundary, not an antenna near-field model.

`realize` combines supplied shadowing and fading values deterministically;
`sample` draws shadowing as a zero-mean Gaussian in dB and optional fading as
an exponential power gain of mean one, corresponding to a normalized
Rayleigh-amplitude model. See the [Rayleigh distribution](https://www.mathworks.com/help/stats/rayleigh-distribution.html)
and [fading-channel normalization](https://www.mathworks.com/help/comm/ug/fading-channels.html)
references. These are optional analytical mechanisms, not a claim of 3GPP
38.901 compliance. No temporal or spatial correlation is imposed: future
experiments must specify refresh cadence/correlation and LOS/NLOS scenarios.
`realize` also allows externally generated correlated channel states.

For one realization, instantaneous SINR uses the fading power gain and
instantaneous rate is computed from that SINR. Large-scale SINR/rate retain
shadowing but remove fast fading. Distance inversion uses a supplied filtered
SINR and the nominal path-loss model, with unknown shadowing, so shadowing can
bias the result. The current large-scale SINR provides an ideal-filter
reference; a finite-window measurement filter is not implemented yet. Never
invert instantaneous fading SINR to create a distance feature. Interference
power is explicit; setting it to zero makes SINR equal SNR. Inversion requires
positive filtered SINR; no fabricated finite estimate is returned for a dead
link. The environment must mark such a link unavailable.

Uplink, downlink, V2V, and V2I can use separate channel configurations and RNG
streams. Noise is `density_w_per_hz * bandwidth_hz * noise_figure_linear`,
where the noise figure is at least one. Interference allocation, radio sharing,
outage persistence, and snapshot installation belong to the environment.

## Time and energy accounting

`progress_work` accepts remaining bits/cycles, constant service rate, and
interval duration. It caps service at the remaining work and returns actual
active time. Finishing early therefore costs only the corresponding active
radio/CPU energy. The primitive does not automatically start another phase;
the environment will control within-slot transitions and any leftover time.

`LatencyBreakdown` uses disjoint buckets: query wait, local admission wait,
final-attempt queue/upload/compute/download, and the entire discarded V2V
attempt. MEC restart service belongs to the final-attempt buckets, never to
the interrupted-attempt bucket. Any fallback admission gap belongs to
`admission_wait_s`. Measured total latency must match completion minus arrival.
The continuous-time service formulas are estimates, not a replacement for the
eventual simulator's physical clock and actual allocations.

Radio and idle energy use actual `P*T`. Server CPU dynamic energy is
`kappa*C*f^2` for actual executed cycles, summed across slots when `f` changes.
No positive work can execute at zero CPU frequency. Server self-workload and
leakage energy are excluded from task-attributable energy in this model.

`MobileEnergyBreakdown` includes user query/TX/RX/idle and server-vehicle
RX/CPU/TX energy; there is no infrastructure MEC-energy bucket. Idle energy is
for explicitly attributed, non-overlapping user waiting intervals. A user's
physical idle consumption must not be charged independently to every task
waiting concurrently. Query energy already contains query-phase idle time
and must not also enter another bucket. It belongs once to the query-owner
task while all waiting tasks accrue their own delay.

For fallback, old-attempt energy goes into `interrupted_attempt_j`; final MEC
radio and any additional idle energy remain in the final-attempt buckets.
Query energy is not charged again merely because execution restarts. Battery
spending and reserve enforcement use the existing vehicle entities; the
environment must budget energy before executing a step.

## Queue and CPU scheduling

Priority is criticality descending, absolute deadline ascending, arrival
slot ascending, then task identifier ascending. Queue workload is in CPU
cycles; the recurrence serves existing workload before adding new arrivals.

The initial CPU allocation convention chooses up to `max_concurrent_tasks`
by that order and shares the total service CPU equally among selected tasks.
This convention is common to every comparison policy, and is not a learned
allocation decision. Service CPU means capacity minus background/self loads,
before old task allocations are subtracted. Repeated scheduling must reclaim
previous allocations rather than repeatedly consume the residual capacity.

`reschedule_resource` must run only at slot boundaries. It validates all
candidate tasks and applies active/waiting membership, allocation, and task
status together. This permits a critical task to exchange places with an
active normal task even when the server waiting queue is full, while
preserving partial CPU work and the selected executor. Resource queue entries
at this stage contain computation-ready tasks; uploads/downloads belong to
separate environment phases. Finish/release zero-work tasks before scheduling.

The server's `max_queue_size` limits waiting tasks, with active tasks counted
separately under `max_concurrent_tasks`. Inability to accommodate a schedule
or an unavailable server raises an explicit error before mutation; the
environment must resolve interruption/MEC fallback first. MEC admission has
no finite queue limit. Energy and radio feasibility are not decided by EDF.

## Reproducibility and verification

Named streams use protocol `risk-aware-v2x:rng-v1`, SHA-256 of master seed and
stream name, then a persistent `random.Random` instance for that name. Creation
order and extra draws in another stream do not affect its sequence. There is
no global RNG mutation. Store the master seed, stream names, runtime version,
and model configuration with experiment results.

All tests can run without third-party packages in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The same tests are compatible with pytest after installing the declared dev
dependency. Tests include numeric radio/energy fixtures, unit conversions,
braking and contact boundaries, shadowing-biased inversion, early completion,
full-queue preemption, CPU conservation, and independent RNG trajectories.
