# Risk-Aware Vehicular Edge Computing

Independent research implementation of dynamic, risk-aware computation
offloading in vehicular edge networks.

The project will study a learning-based offloading policy built around
QR-DQN and CVaR. The reconstructed game-theoretic method from the base
paper is treated as a comparison baseline, not as the implementation
foundation of this repository.

## Current status

The first implementation milestone now provides validated `Task`,
`UserVehicle`, `ServerVehicle`, and `MECServer` entities, together with
immutable MEC metadata snapshots and unit tests for their scientific
invariants. The time-slotted environment, physical models, and learning
agents have not been implemented yet.

The approved design baseline is documented in:

- [`docs/scientific_model.md`](docs/scientific_model.md): scientific
  assumptions, entities, units, formulas, information lifetime, and metrics;
- [`docs/mdp_specification.md`](docs/mdp_specification.md): observations,
  actions, slot ordering, query lifecycle, CVaR control, fallback, and the
  transition-test matrix.

The implementation milestones are:

1. **complete:** task, user-vehicle, server-vehicle, and MEC entities;
2. **next:** latency, energy, queue, mobility, and channel models;
3. the time-slotted dynamic environment;
4. state, action, reward, and terminal conditions;
5. reproducible baseline and learning experiments.

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
