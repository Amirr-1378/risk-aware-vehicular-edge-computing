# Experiments

Experiment entry points and versioned configurations will live here.
Generated figures, checkpoints, and reports are intentionally ignored by Git.

`run_environment_smoke.py` is a synthetic integration demonstration with two
fixed rules (`mec` and `first_cached_server`). It prints a reproducibility
manifest and complete task ledgers as JSON. It is not a calibrated benchmark
or an algorithm-performance claim. Run it from the repository root with the
package installed, or set PowerShell's `$env:PYTHONPATH = "src"` first:

```text
python -m experiments.run_environment_smoke --seed 7 --policy first_cached_server
```

`run_baseline_smoke.py` runs AlwaysMEC, AlwaysV2V, UniformRandom, and the
supplementary MinimumEstimatedLatency rule on the same synthetic scenario.
It audits the learning adapter's raw and slot-discounted prefix/transition
decomposition and records explicit encoder configuration and independent seeds:

```text
python -m experiments.run_baseline_smoke --seed 7 --policy-seed 17 --gamma-per-slot 1
```

No learned policy or statistical performance claim is included. See
[`../docs/learning_interface.md`](../docs/learning_interface.md) for limitations
and the distinction between minimum latency and combined-cost control.

`run_learning_smoke.py` trains/evaluates an optional CPU neural agent and can
save/resume complete episode boundaries. It is also an integration audit, not
a calibrated benchmark. Install the `learning` dependencies first:

```text
python -m experiments.run_learning_smoke --algorithm qr_dqn_cvar --episodes 4
python -m experiments.run_learning_smoke --episodes 2 --checkpoint artifacts/learning/first.pt
python -m experiments.run_learning_smoke --episodes 2 --resume artifacts/learning/first.pt --checkpoint artifacts/learning/continued.pt
```

Choose `dqn`, `double_dqn`, `qr_dqn`, or `qr_dqn_cvar`. Reports distinguish
training updates from frozen greedy evaluation and preserve complete costs,
including pre-decision prefixes and variable drain. Separate seed labels in
this deterministic smoke scene are not independent stochastic evaluation
trials. See [`../docs/neural_agents.md`](../docs/neural_agents.md).
