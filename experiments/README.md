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
