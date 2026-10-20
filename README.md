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
decision-epoch transitions, four simple comparison policies, and exact
finite-distribution CVaR scoring are implemented and tested. CPU DQN,
Double DQN, QR-DQN, and mean-CVaR-distorted QR agents now include replay,
training updates, greedy evaluation, and resumable episode-boundary checkpoints.
Calibrated paper-scale experiments remain a subsequent milestone.

The approved design baseline is documented in:

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

The implementation milestones are:

1. **complete:** task, user-vehicle, server-vehicle, and MEC entities;
2. **complete:** latency, energy, queue, mobility, and channel primitives;
3. **complete:** the time-slotted dynamic environment kernel;
4. **complete:** raw observations, batched actions, slot costs/rewards, and drain;
5. **complete:** learning vector/transition adapters, simple baselines, and risk utilities;
6. **complete:** DQN/Double DQN/QR-DQN updates, replay, checkpoints, and a
   reproducible CPU training/evaluation audit;
7. **next:** calibrated stochastic scenarios, frozen experimental protocols,
   adapted paper/combined-cost baselines, and statistically defensible evaluation.

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
experiments/     Reproducible experiment entry points and configurations
tests/           Unit and integration tests
```

## Baseline separation

The base-paper reconstruction remains in its own repository. This project
may reimplement documented equations and baseline behavior with explicit
provenance, but it must not silently depend on or modify that repository.

## Development

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
```

Run a synthetic environment integration demonstration (not a paper benchmark):

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
python -m experiments.run_environment_smoke --seed 7 --policy mec
python -m experiments.run_baseline_smoke --seed 7 --policy-seed 17
```

The JSON report includes configuration, seed, initial scenario, per-task
accounting, and timestamped events. Read the environment contract before
using this simulator for comparisons: completion depends on the explicit
single-cell service and energy-envelope assumptions.
The baseline smoke report also checks raw/discounted prefix and transition
accounting. Neither smoke entry point is a scientific performance benchmark.
The neural smoke runner likewise checks training/evaluation mechanics rather
than providing learning-performance or static-CVaR-optimality evidence.
