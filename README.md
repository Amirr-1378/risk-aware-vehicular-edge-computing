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
drain-to-completion tests. Learning agents and paper-scale experiments remain
subsequent milestones.

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

The implementation milestones are:

1. **complete:** task, user-vehicle, server-vehicle, and MEC entities;
2. **complete:** latency, energy, queue, mobility, and channel primitives;
3. **complete:** the time-slotted dynamic environment kernel;
4. **complete:** raw observations, batched actions, slot costs/rewards, and drain;
5. **next:** learning-facing vector/transition adapters, baselines, risk
   utilities, and reproducible learning experiments.

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

Run a synthetic environment integration demonstration (not a paper benchmark):

```powershell
$env:PYTHONPATH = "src"
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
python -m experiments.run_environment_smoke --seed 7 --policy mec
```

The JSON report includes configuration, seed, initial scenario, per-task
accounting, and timestamped events. Read the environment contract before
using this simulator for comparisons: completion depends on the explicit
single-cell service and energy-envelope assumptions.
