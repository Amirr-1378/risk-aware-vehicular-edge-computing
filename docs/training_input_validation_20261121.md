# Training input audit 2026 11 21

The training-only checkpoint audit found a strong scale imbalance in the
existing SNR inputs. In the 75 retained training observations, four SNR columns
account for 99.999999732% of squared input magnitude. The largest encoded SNR
is 2,216,760.814; the largest magnitude among all other features is 32.497.
This warrants a controlled representation study before increasing the
training budget. It does not establish that SNR scaling explains the entire
performance gap to greedy.

The audit used the independently verified small Colab session recorded in
[`colab_execution_20261121.md`](colab_execution_20261121.md). No new environment
episode, gradient update, replay sample, scale fitting or validation-outcome
analysis was performed by the diagnostic. QR-DQN + CVaR, alpha 0.9, risk weight
0.5, independent local vehicle decisions, the actual cost and the physical
models remain unchanged. Main training remains unstarted.

## Training data and representation

The final replay retains all 75 training transitions, with 96 encoded features
per current observation. There are 63 bootstrap-active next-observation
occurrences and 12 terminal records. Next observations overlap with current
records; 75 + 63 is not a count of independent samples. The initial and final
networks were both inspected on this same final retained training cohort.

| Current training input | Absolute median | Absolute p95 | Maximum magnitude |
| --- | --- | --- | --- |
| server[0].snr | 204.037 | 108456.082 | 2216760.814 |
| server[1].snr | 45.939 | 834.778 | 68468.593 |
| server[2].snr | 185.108 | 534399.320 | 1647271.082 |
| server[3].snr | 115.284 | 3497.841 | 6602.612 |
| Largest non-SNR column, server[1].contact_margin | 2.784 | 18.925 | 32.497 |

Absolute percentiles use nearest rank, not interpolation. Values are
dimensionless, under the original fixed feature scales. Present-but-masked
server blocks remain in the statistics because they remain network inputs.

The initial online first layer assigns 99.926703% of its absolute weighted
input-term magnitude to SNR columns; the final online layer assigns
99.923815%. Bias is excluded. This describes term magnitudes, not signed
preactivation totals, causal feature importance or decision quality.
The exact formulas and limitations are in
[`training_input_diagnostics.md`](training_input_diagnostics.md).

The source path is consistent: the wireless snapshot supplies large-scale
linear SINR as the cached `estimated_snr_linear` value, and the encoder divides
it by `FeatureScales.snr_linear=100`. The peak encoded value therefore
corresponds to linear SINR 221,676,081.42, approximately 83.457 dB. No logarithmic
compression is applied by the current encoder. This investigation did not
demonstrate a channel-equation or unit-conversion error; physical parameter
calibration remains a separate research task.

## Predictions and targets on fixed training replay

| Retrospective diagnostic | Initial network | Final network |
| --- | --- | --- |
| Maximum absolute valid-action quantile | 87856.430 | 36638.984 |
| Maximum absolute bootstrap-active target atom | 58040.789 | 58040.789 |
| Fixed-cohort quantile Huber loss | 47385.004 | 18816.330 |

Recorded discounted decision-interval costs range from 0.031404 to 1.469984,
with median 0.189276. Interval costs and predicted cumulative cost distributions
have different horizons: the interval maximum is not an upper bound on a true
future return. These magnitudes nevertheless expose a numerical diagnostic
that must be checked before assuming that more training is the only obstacle.

All 612 terminal target atoms (12 records x 51 atoms) are just the recorded
discounted interval costs, with maximum 0.702673. Large target magnitudes occur
in the 3213 bootstrap-active atoms. Target parameters were never refreshed in
the small run because 44 updates are below the interval of 50; initial and
final target distributions on this cohort are therefore equal. Recomputed
float32 values may differ slightly with batch shape/runtime and are not
required to equal a single large exploratory forward pass bit for bit.

The fixed-cohort losses above are not historical sampled minibatch losses or
a learning curve. Historical recorded losses span 3518.815 to 81565.328. Their
means over episodes with updates are 47128.961, 36773.530 and 24485.942. The QR
loss sums 51 predicted quantile terms, which also affects its numerical scale.
Every recorded pre-clipping gradient norm exceeds the configured clip limit
of 10; the largest is 275263.156. Those are pre-clipping norms, not evidence
that clipping failed. Neither the loss changes nor this tiny cohort establish
convergence or improved evaluation performance.

## Reproducibility and checks

```text
python -m experiments.run_training_input_audit --session-dir reports/colab-user-execution-510b407/import-772ce9d3770349309c9a7262af8426ab/session --output reports/training-input-audit-v1-20261121/report.json
```

The exclusive output is 234,904 bytes. It contains named feature statistics,
geometry/contribution shares, valid-action outputs, targets, historical update
summaries, replay coverage, archived and auditor identities, and mutation guards.

| Fingerprint | SHA256 |
| --- | --- |
| Archived training source | `2bf447e4695f19aa4fb25cdee8eae15ecd6185558f65a24b966e122f38fe3b3f` |
| Archived session contract | `95a9b9fdaaa756468f796f8cbc7af86ec4e3c40f75eca38a7555b79e6713ea87` |
| Auditor source | `54b9e0189a464d19eae291a531059271d35c753202d04864e185c4461a08407a` |
| Diagnostic report | `6c8a0a8a1eb2bf2b3c4bdf2ea5b0a7c1951412f581abb043924b5bffa9ea9b68` |

Auditor provenance truthfully records parent commit
`34d39763e2a91c10bd77ce604a7e613c4d31ee25` and `git_dirty=true`. Archived
checkpoints are not relabeled as products of the new diagnostic code. Numerical
inspection ran on the local CPU runtime (Python 3.12.14, Torch 2.14.1+cpu,
NumPy 2.5.4); exact continuation of the Linux training session was not attempted.
All loaded learner states, owned/global RNGs, execution settings and source
journal hashes remained unchanged.

Tests cover hand-computed statistics, nearest-rank boundaries, overflow-safe
RMS, squared-input shares, first-layer contributions, partial-batch loss
weighting, masked server inputs, terminal/zero-discount exclusion, empty replay,
ring retention, corrupt/identity-mismatched checkpoints and exclusive output.
Forbidden-operation tests reject action selection, learning, replay sampling,
optimizer steps and environment simulation. A temporary fixture with altered
validation outcomes confirms that training analytics are unchanged.

The full learning-runtime suite discovered 474 tests: 473 passed and one
Windows symlink-privilege test was skipped. The standard-library runtime
discovered the same 474: 332 passed and 142 were skipped (141 optional learning
tests plus the symlink restriction). `git diff --check` passed. No scientific
source under `src/` or existing experiment configuration changed; new code is
confined to the diagnostic layer and its statistics/tests.

## Next study

Keep physical linear SINR, the actual objective and the greedy baseline intact.
Define a separately versioned, monotone SNR representation or a defensible
fixed-reference alternative. Preserve the old encoder/checkpoints and compare
variants through matched retraining, including risk-neutral QR and QR-CVaR
under the same predeclared trace/update budgets. A new transform applied to
old weights is not an improvement experiment. Validation/test outcomes must
not set the transform's reference values.

Record numerical input/output behavior first, then evaluate mean cost, tail
cost, deadline metrics and MEC/SV workload on an unchanged validation cohort.
Scenario calibration and sufficient training remain open; the diagnosis does
not guarantee a lower cost than greedy. Commit and push this diagnostic
milestone before advancing. Notify Amir before any main training in Colab.
