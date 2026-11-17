# Development Diagnostics Before Main Training

The next verification step after the cost-model-2 revision is the complete
**bounded development pilot**, not main training: four algorithms, two agent
seeds, four nominal training episodes each, and two validation episodes in
each of four profiles. Configuration remains `learning_pilot_v3.json` bound
to `development_v2.json`. No hyperparameters, objective weights, risk settings,
physical inputs, or model-selection rules are changed by diagnostics.

Main training will run in **Google Colab**, as requested by the supervisor.
Notify Amir before starting that milestone. Local runs remain bounded
implementation/development checks. This pilot does not freeze a paper
protocol or establish that the project is ready for main training.
The complete execution record and descriptive outcomes are in
[development_pilot_20261117.md](development_pilot_20261117.md).

## Reproduction and artifacts

From the repository root, with the package installed or `PYTHONPATH=src`:

```text
python -m experiments.run_learning_audit --output-dir reports/NEW-pilot
python -m experiments.summarize_learning_audit reports/NEW-pilot/report.json --output-dir reports/NEW-diagnostics
```

Both output directories must be new. The summary command reads a completed
JSON report; it neither imports Torch nor loads checkpoints, runs a simulator,
trains/selects a model, opens the test split, or consumes randomness. It can
run on the standard-library-only installation. Outputs are `diagnostics.json`
and `diagnostics.md`; the JSON records the SHA256 of the exact input report,
the original source provenance, and both protocol fingerprints. The source
fingerprint identifies the implementation at execution time, not a later
postprocessing session. Reports and checkpoints stay ignored by Git.

The summary checks current report/cost contracts and protocol hashes, the
complete algorithm/seed grid, required profiles/fixed policies, episode IDs,
matching scenario fingerprints/channel seeds, full task drain, finite
nonnegative metrics, frozen greedy validation, and training-clock/update
counter consistency. Missing or corrupt records cause an error; they are not
filtered out. This is a report consistency check, not an independent rerun of
the physical ledger or a proof of correctness for an arbitrary edited report.
The producer already reconciles actual task ledgers and adapter accounting.

## Statistical units and denominators

For one fixed final model and one profile, descriptive cost risk uses full
episode totals, with equal episode weights, including empty episodes:

`mean_cost = sum(z_episode) / episodes`, with upper CVaR at the declared alpha.

Each trained model/seed has its own row. There is no pooled-policy CVaR,
best-seed selection, or pooling across the four known validation profiles.
Fixed-policy rows appear once per profile, not once per agent seed. Task
instances repeat shared traces across methods and are not independent trials.

Latency is `sum(n_episode * mean_latency_episode) / sum(n_episode)` seconds.
Mobile energy is `sum(E_mobile_episode) / sum(n_episode)` joules per task.
Deadline misses are `sum(misses_episode) / sum(n_episode)`. Tardiness uses the
same task denominator and the recorded total tardiness. Direct-MEC/fallback
task fractions use task weights. Zero task denominators produce JSON null
and Markdown `n/a`; no success rate is invented for an empty workload.

SV and MEC completed-workload fractions use equal weights over **nonempty
episodes**. This report contains fractions, not each episode's total CPU
cycles, so it cannot reconstruct a pooled fraction of CPU cycles. The output
names explicitly identify the episode mean. No arithmetic average of episode
p95/p99 latencies is presented as a pooled latency quantile.

At two validation episodes and alpha=.9 the nominal tail mass is **0.2
episodes**: empirical CVaR equals the larger episode cost. It is a tiny-sample
diagnostic, not evidence of reliable risk reduction. No new significance or
confidence claim is derived from this report.

Training traces preserve per-episode total cost, actual physical training
slots, transitions, update count, mean update loss, maximum gradient norm
before clipping, and epsilon at the episode boundary. Epsilon follows the
existing physical-slot schedule. Missing optimizer updates produce null
loss/gradient summaries. Loss scales differ between scalar and QR heads;
loss magnitudes and four training episodes do not establish policy quality
or convergence. Training-episode costs also reflect different arrival traces
and exploration levels, not a fixed-policy learning curve.

## Remaining work before main training

- Calibrate workload/resource/channel assumptions and define the supported
  scenario population; current numerical ranges remain synthetic.
- Complete the comparison design, including the faithful adapted paper
  best-response baseline and the query-free MEC comparison where appropriate.
- Declare a separate development/tuning budget and then freeze the final
  training, model-selection, evaluation, seed, and ablation protocol. Do not
  use the held-out test traces to make these decisions.
- Prepare and verify the Colab execution workflow, dependencies, artifact
  persistence, interruption/restart behavior, and compute budget. The existing
  audit is a CPU driver, not a resumable paper-scale sweep.

These steps precede the main Colab training milestone. This document does not
authorize an automatic large training job or treat a short pilot as main
training.
