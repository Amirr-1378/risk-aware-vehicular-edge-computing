# Training input diagnostics

The training input audit checks the numerical representation used by the neural
controller before increasing the development training budget. It reads a
previously saved session and computes descriptive statistics on its retained
training replay. It does not fit scales, transform observations, train a model,
generate scenarios or choose parameters from validation/test data.

QR-DQN + CVaR, alpha 0.9, risk weight 0.5, independent User Vehicle decisions,
the three-term cost and all physical assumptions remain unchanged. The audit
does not introduce a teacher, an MEC controller or a new deployment feature.

## Input and provenance

```text
python -m experiments.run_training_input_audit --session-dir reports/RECEIVED/session --output reports/NEW-training-input-audit/report.json
```

Use a complete, unmodified session journal or a valid committed prefix. The
tool validates journal hashes, phase/clock/update counters, model/encoder/scope
contracts and checkpoint metadata before creating an output file. Checkpoints
use the existing restricted weights-only loader and exact Torch version guard.
The output path must be new and outside the source session; old evidence is
never overwritten. An initialized session with empty replay is valid.

The archived training source/runtime and the auditor's own source/runtime are
recorded separately. New diagnostic code changes the auditor source hash; it
does not relabel old checkpoints as produced by the new code. Read-only loading
of a Linux checkpoint on Windows is not exact cross-platform continuation.
Production resume guards remain unchanged.

The journal integrity scan reads committed files, including validation JSON,
to check hashes and structure. Numerical analysis uses only training reports
and the final checkpoint's retained replay; validation outcomes are excluded.
No held-out scenario is constructed. Replay is traversed in storage order,
without sampling or RNG draws. With a wrapped ring, retained records are a
window of training experience, not the entire trace or necessarily chronological.
The report states retained count, total seen count and whether coverage is complete.

## Feature statistics

Encoded features are dimensionless. For each named column, the report gives
count, minimum, maximum, mean, root mean square, maximum absolute value and
absolute empirical p50/p95/p99. Percentiles use exact nearest rank, the order
statistic at `ceil(p*n)` with one-based indexing. These are sample descriptions,
not confidence intervals, clipping limits or fitted reference scales.

Current observations and bootstrap-active next observations are separate.
Terminal synthetic zero rows and next observations with zero bootstrap discount
are excluded from next-input statistics. Next records can repeat current
observations; their counts must not be added into independent sample counts.

Present-but-masked Server Vehicle blocks remain in the input analysis because
the existing fully connected network still receives them. Per-slot presence
and action-valid counts are reported separately. Input values must remain finite
after conversion to the actual float32 network dtype.

For feature f, the squared-input share is

```text
sum_observations(x[f]^2) / sum_observations_and_features(x[f]^2).
```

Computation rescales all values by a common maximum magnitude before squaring.
The scale cancels in the ratio and prevents avoidable overflow. An all-zero
matrix has undefined shares, represented by null. SNR-column shares are also
summed explicitly; no feature is removed from the network.

The online network's first linear layer has affine preactivation `y = x W^T + b`, as defined by
[PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
The audit reports the fraction of absolute weighted input terms associated
with each feature, excluding bias:

```text
A[f] = sum_observations(abs(x[f])) * sum_first_layer_units(abs(W[unit,f]))
share[f] = A[f] / sum_features(A[f]).
```

This is a numerical magnitude diagnostic. Absolute terms do not preserve signed
cancellation, nonlinear interactions or downstream decisions; the share is not
causal feature importance or proof of an improvement available to the policy.
The weighted diagnostic uses online weights for both input cohorts; it is not
an attribution to the target network used by QR bootstrap selection.

## Network and update diagnostics

Initial and final networks are both evaluated on the same final retained
training replay. The report separates valid-action quantile outputs, mean/
upper-CVaR/risk scores, recorded-action quantiles, actual interval costs and
bootstrap targets. Masked destinations are excluded from output statistics.

Targets use the existing `agent.targets` implementation. QR-CVaR selects the
bootstrap action with target-network risk scores; Double DQN uses online
selection and target evaluation. Terminal/zero-discount targets contain only
recorded discounted interval cost. A replay interval may include several
tasks' costs for one user; it is not necessarily the final cost of one task.

The fixed-cohort loss recomputation uses the same Huber/quantile loss as the
agent, without gradients, on ordered bounded batches. Batch losses are weighted
by sample count, including the final shorter batch. QR loss sums predicted
quantiles and averages target atoms and samples; its scale depends on atom
count and cannot be compared directly with a scalar task cost.

Historical recorded update losses and pre-clipping gradient norms are reported
separately. The final replay is not a reconstruction of the historical random
minibatches. Initial/final fixed-cohort losses are retrospective descriptions,
not an unbiased learning curve, a generalization result or an evaluation win.
PyTorch's
[gradient clipping API](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
returns the total norm used before in-place clipping; a recorded norm above
the configured limit does not show that clipping failed.

No `act`, `observe`, `learn`, replay sampling, backward pass or optimizer step
is performed. Models, target, optimizer, replay/exploration RNGs, counters and
modes are checked unchanged. Python, NumPy and Torch global RNGs and CPU
execution settings are also checked, and the immutable journal is reverified.

## Decisions after diagnosis

Keep the original linear physical SNR, channel formulas, actual cost and strong
greedy baseline intact while investigating numerical scale imbalance. A future
monotone SNR representation or different fixed reference must be declared and
versioned, with old checkpoints preserved and matched retraining comparisons.
Applying a new transform to old weights is not an improvement experiment.
Do not estimate transforms from validation/test outcomes or increase a losing
method's budget repeatedly until it wins. Main training remains a later Colab
milestone with prior notice to Amir.

The executed training-only check is recorded in
[`training_input_validation_20261121.md`](training_input_validation_20261121.md).
The separately versioned SNR representation comparison is defined in
[`snr_representation_development.md`](snr_representation_development.md). The
numerical helper recognizes both `.snr` and `.snr_log1p` feature names; existing
resumable-session CLI contracts/defaults remain schema 3.
