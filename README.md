# Risk-Aware Vehicular Edge Computing

Independent research implementation of dynamic, risk-aware computation
offloading in vehicular edge networks.

The project will study a learning-based offloading policy built around
QR-DQN and CVaR. The reconstructed game-theoretic method from the base
paper is treated as a comparison baseline, not as the implementation
foundation of this repository.

## Current status

Validated `Task`, `UserVehicle`, `ServerVehicle`, and `MECServer` entities and
immutable MEC snapshots are implemented. The physical-model layer now provides
mobility, channels, latency, energy, EDF scheduling, explicit unit conversions,
and independent seeded RNG streams with numeric and state-transition tests.
The time-slotted multi-user environment now joins those components, with
cached observations, batched actions, queries, MEC recovery, slot costs, and
drain-to-completion tests. Fixed-size observation encoding, complete
decision-epoch transitions, five analytical comparison policies, and exact
finite-distribution CVaR scoring are implemented and tested. CPU DQN,
Double DQN, QR-DQN, and mean-CVaR-distorted QR agents now include replay,
training updates, greedy evaluation, and resumable episode-boundary checkpoints.
Seeded stochastic development scenarios, disjoint split/seed plans, terminal
metrics, and paired analytical-policy audits are also implemented. Their
parameters are explicitly synthetic. A bounded multi-seed neural audit now
connects those scenarios to fixed-final-model training/validation and matched
analytical baselines, with separate per-model risk and conditional seed
variation. Calibrated paper-scale experiments remain a subsequent milestone.
The fixed-policy suite now includes cached combined-cost greedy control with
shared objective/hardware assumptions. Own-task deadlines are explicit in
raw/vector input schema v3 removes manual priority labels; old checkpoints
are preserved but incompatible with cost model v2.

The approved 2026-11-02 revision uses exactly
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, without a manual task multiplier or
deadline surcharge. Queues use plain EDF. QR-CVaR retains alpha=.9 and
lambda=.5; sigmoid(beta*(J_SV-J_MEC)), beta=1 by default, reports preference
only. Greedy destination selection remains masked argmin J, independent of
beta. Deadline misses/tardiness remain metrics, not added costs.

The approved design baseline is documented in:

- [`docs/objective_and_preferences.md`](docs/objective_and_preferences.md):
  authoritative current objective, risk score, sigmoid meaning, selector and version contract;
- [`docs/scientific_model.md`](docs/scientific_model.md): scientific
  assumptions, entities, units, formulas, information lifetime, and metrics;
- [`docs/mdp_specification.md`](docs/mdp_specification.md): observations,
  actions, slot ordering, query lifecycle, CVaR control, fallback, and the
  transition-test matrix.
- [`docs/physical_models.md`](docs/physical_models.md): implemented formulas,
  allocation conventions, accounting boundaries, and modeling limitations.
- [`docs/simulation_environment.md`](docs/simulation_environment.md): simulator
  API, timing, resource sharing, causal energy escrow, and completion assumptions.
- [`docs/learning_interface.md`](docs/learning_interface.md): vector schema,
  decision-gap/prefix accounting, simple baselines, and cost/reward CVaR conventions.
- [`docs/neural_agents.md`](docs/neural_agents.md): exact learning updates,
  optional CPU dependencies, exploration, checkpoint/resume, and scientific limits.
- [`docs/development_protocol.md`](docs/development_protocol.md): stochastic
  scenarios, parameter provenance/gaps, split/seed plan, metrics, and paired audits.
- [`docs/learning_development_protocol.md`](docs/learning_development_protocol.md):
  bounded multi-seed training, final-model validation, pairing, and statistical units.
- [`docs/learning_diagnostics.md`](docs/learning_diagnostics.md): read-only
  per-model development summaries, aggregation units, and the Colab main-training boundary;
- [`docs/greedy_improvement_plan.md`](docs/greedy_improvement_plan.md): offline
  action comparisons, owner/system credit alignment, and evidence-based development priorities;
- [`docs/team_returns.md`](docs/team_returns.md): independent User Vehicle
  actors, opt-in cooperative-return records, and global/per-user accounting rules;
- [`docs/return_scope_development.md`](docs/return_scope_development.md): matched
  stochastic individual/team-return training, paired episode costs and fixed budgets;
- [`docs/behavior_warmup_development.md`](docs/behavior_warmup_development.md):
  optional causal training behavior warmup, core-method boundaries and matched budgets;
- [`docs/combined_cost_baseline.md`](docs/combined_cost_baseline.md): combined-cost
  heuristic formulas, public calibration assumptions, and input/baseline version 2.

The implementation milestones are:

1. **complete:** task, user-vehicle, server-vehicle, and MEC entities;
2. **complete:** latency, energy, queue, mobility, and channel primitives;
3. **complete:** the time-slotted dynamic environment kernel;
4. **complete:** raw observations, batched actions, slot costs/rewards, and drain;
5. **complete:** learning vector/transition adapters, simple baselines, and risk utilities;
6. **complete:** DQN/Double DQN/QR-DQN updates, replay, checkpoints, and a
   reproducible CPU training/evaluation audit;
7. **complete:** synthetic stochastic development scenarios, versioned split/seed
   plans, terminal metrics, and paired fixed-policy evaluation infrastructure;
8. **complete:** bounded multi-seed stochastic neural training/validation pilot
   infrastructure, final-model artifacts, and conditional training-seed diagnostics;
9. **complete:** cached combined-cost greedy baseline, shared fixed-policy suite,
   explicit own-deadline inputs, and versioned comparison integration;
10. **complete:** approved label-free three-term cost revision, plain EDF,
   sigmoid preference diagnostics, schema/checkpoint guards, and regression tests;
11. **complete:** read-only development diagnostics, per-model validation cost
   risk, task/workload aggregation, and training-clock/update consistency checks;
   the [complete cost-model-2 pilot](docs/development_pilot_20261117.md) is recorded
   as development evidence, with no main-training/convergence claim;
12. **complete:** offline action-audit infrastructure with cloned RNG/state,
   greedy continuation, and separate owner/system cost comparisons;
13. **complete:** opt-in team-return interface and matched accounting tests
   for independent local-observation actors;
14. **complete:** team-compatible neural learning, typed replay and checkpoint
   contracts, frozen evaluation, and exact episode-boundary resume verification;
15. **complete:** versioned stochastic return-scope ablation infrastructure,
   matched model/scene budgets, frozen-state checks and descriptive comparisons;
16. **complete:** opt-in greedy training-behavior warmup infrastructure with
   unchanged core QR/CVaR math, teacher-free evaluation and matched ordinary-training controls;
17. **complete:** Colab preparation and independently verified small execution/
   checkpoint continuation; main training remains unstarted;
18. **complete:** training-only input/target diagnostics without model, replay
   or RNG mutation; the [executed audit](docs/training_input_validation_20261121.md)
   identifies SNR scale imbalance, with no encoder or objective change;
19. **next:** empirical calibration, informative learning studies, frozen paper
   protocols, adapted paper/query-free MEC baselines, and final evaluation.

Main training is planned for Google Colab. Notify Amir before starting that
milestone; local training runs are bounded development checks.
Each User Vehicle selects its own destination; MEC supplies information only.
Remind Amir about commit/push at each completed verified milestone before
advancing; pushing still requires explicit user authorization.

## Repository layout

```text
src/risk_aware_v2x/
  agents/        Learning agents such as DQN and QR-DQN
  baselines/     Paper best response and simple comparison policies
  blockchain/    Reserved for optional, non-core comparison extensions
  entities/      Tasks, vehicles, and MEC entities
  environment/   Time-slotted simulation environment
  models/        Channel, latency, energy, mobility, and queue models
  risk/          CVaR and distributional-risk utilities
  scenarios/     Seeded synthetic workloads and versioned development plans
experiments/     Reproducible experiment entry points and configurations
tests/           Unit and integration tests
```

## Baseline separation

The base-paper reconstruction remains in its own repository. This project
may reimplement documented equations and baseline behavior with explicit
provenance, but it must not silently depend on or modify that repository.
Its native internal objective must not be silently replaced by our common
evaluation cost. Faithful reconstruction/action-space adaptation is still pending.

## Development

Colab preparation now has an output-empty, opt-in
[`notebook`](notebooks/colab_development.ipynb) and an episode-boundary resumable
single-model development runner. Source/runtime/protocol guards and immutable
checkpoint reports preserve the existing method. This does not start main
training or enable greedy warmup. See [`docs/colab_development.md`](docs/colab_development.md)
for private-source upload, fixed budgets and explicit interruption recovery.

The small Colab plan completed through one episode followed by explicit
checkpoint continuation: four training episodes, eight frozen validation cases,
75 transitions and 44 updates. The received archive, all five checkpoints and
episode records passed independent verification; main training remains unstarted. See
[`docs/colab_execution_20261121.md`](docs/colab_execution_20261121.md).

The package uses a `src` layout and will target Python 3.11 or newer.

Install the package with development dependencies and run the tests with:

```text
python -m pip install -e ".[dev]"
python -m pytest
```

The current tests also run with the Python standard library (PowerShell):

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Neural training requires the optional `learning` dependencies; without them,
the core tests run and the neural tests explicitly skip. For a full CPU setup,
see [`docs/neural_agents.md`](docs/neural_agents.md). In the configured local
virtual environment, run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --algorithm qr_dqn_cvar --episodes 4
.\.venv\Scripts\python.exe -m experiments.run_learning_audit --output-dir reports/learning-pilot-v3
```

Run a synthetic environment integration demonstration (not a paper benchmark):

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
python -m experiments.run_environment_smoke --seed 7 --policy mec
python -m experiments.run_baseline_smoke --seed 7 --policy-seed 17
python -m experiments.run_scenario_audit --profile nominal --cases 2
```

The JSON report includes configuration, seed, initial scenario, per-task
accounting, and timestamped events. Read the environment contract before
using this simulator for comparisons: completion depends on the explicit
single-cell service and energy-envelope assumptions.
The baseline smoke report also checks raw/discounted prefix and transition
accounting. Neither smoke entry point is a scientific performance benchmark.
The neural smoke runner likewise checks training/evaluation mechanics rather
than providing learning-performance or static-CVaR-optimality evidence.
The stochastic audit uses validation scenarios by default and keeps the test
split closed. Its small-sample risk/interval summaries are development
diagnostics, not paper-level evidence of improvement.
The multi-seed neural audit likewise keeps test closed and uses a tiny explicit
development budget; it does not establish convergence or risk reduction.
