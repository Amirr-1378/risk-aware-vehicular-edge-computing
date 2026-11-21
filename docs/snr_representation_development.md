# Versioned SNR representation study

The input audit identified severe numerical dominance by four linear SNR
features. This study adds an explicit normalized logarithmic representation
and compares it with the unchanged linear encoder. It changes the numerical
coordinates received by the neural network; physical channels, cached
information, actions, actual costs and QR/CVaR learning equations are retained.
Main training remains a later Colab milestone with prior notice to Amir.

## Fixed transformation

Let s be the received cached large-scale linear SINR, a dimensionless power
ratio. The legacy schema-3 feature is `s / r`. The new schema-4 feature is:

```text
f(s; r) = log1p(s) / log1p(r), with fixed r = 100 in this comparison.
```

The reference 100 is the pre-existing `FeatureScales.snr_linear` value, not a
maximum, quantile, clipping boundary or parameter fitted to these results.
The implementation uses
[`math.log1p`](https://docs.python.org/3/library/math.html#math.log1p), which
computes the logarithm accurately near zero. No empirical normalization or
validation/test statistics are used.

Both ratios are dimensionless. Zero maps to zero, the reference maps to one,
and the derivative is `1 / ((1+s)*log1p(r)) > 0`. Thus the real-valued map
preserves ordering and has no upper clipping threshold. Finite precision can
round very small differences. Finite nonnegative inputs and a finite positive
reference are required; unrepresentable/nonfinite outputs are rejected.

For the previously observed raw SINR 221,676,081.4195, the old encoded value
2,216,760.814 becomes approximately **4.163863**. This is a numerical mapping,
not an observed improvement in computation offloading. It is not the ordinary
`10*log10(s)` dB value, and it does not replace the SINR used in rate/energy
models or cached baseline calculations.

Only per-SV SNR coordinates change. Task/deadline, queue, capacity, freshness,
distance/contact, mobile energy and MEC-rate inputs remain the same at a common
raw state. Presence bits distinguish an absent zero-padded block from a present
record with zero channel quality. Present but action-masked records remain
inputs, as in the existing encoder. No future or simulator-truth input is added.

## Compatibility

| Component | Legacy control | Logarithmic candidate |
| --- | --- | --- |
| Encoder class | ObservationEncoder | LogSNRObservationEncoder |
| Encoded schema | 3 | 4 |
| Server feature name | snr | snr_log1p |
| Raw observation schema | 3 | 3 |
| Dimension at four SV slots | 96 | 96 |
| Reference source | fixed FeatureScales | same fixed FeatureScales |

The original encoder constant/default remains schema 3. Its numeric vectors,
names and full checkpoint signature retain their old values. Schema 4 adds an
explicit representation contract to its signature. Identical dimensions or
even identical vectors at the reference value do not make the schemas compatible.

Current/next schema mismatches and mixed-schema replay containers are rejected.
Agents reject observations with a schema different from their encoder before
exploration or learning. Checkpoint signatures reject cross-encoder/reference
loads, including empty replay. Existing individual replay/checkpoint container
versions remain 2/2; team containers remain 1/3. Their existing payload keys
are preserved, while per-record schemas and the encoder signature bind meaning.
Direct individual/team replay guards are both tested, but this representation
comparison fixes learning returns to individual scope.

Old artifacts are not migrated, overwritten or re-scored. The new representation
is trained from matching fresh initialization, never injected into old trained
weights as an improvement claim. Existing learning/session/Colab entry points
continue using their schema-3 configurations. This study does not silently
switch their defaults or provide sweep interruption/resume support.

## Declared paired development plan

`experiments/configs/snr_representation_pilot_v1.json` is a new outer protocol.
Its nested learning-v3 object is exactly the previous paired study's shared
hyperparameter/budget template. The nested encoder-v3 field labels that control
template; actual per-run schema 3 or 4 is recorded separately in the outer
variant contract, report and checkpoint signature.

The fixed grid is two representations x QR-DQN/QR-DQN+CVaR x seeds 11/29:
eight models. Each uses the same 20 nominal training scenes and four validation
cases in each known profile (nominal, high load, fast servers, harsh channel).
Gamma 1, alpha .9, risk weight .5, reference 100 and all other settings are
shared. Risk-neutral QR still has effective risk weight zero.

Matched seed/algorithm pairs must have equal initial weights. Offered tasks,
transition counts and update budgets are checked equal; a fresh environment
clone is used for every episode. Actions can change queues, episode durations,
query timing and therefore realized exploration under the shared physical-slot
epsilon schedule. Matching exogenous traces does not require equal endogenous
states or action histories.

The existing `_fit_model` and learning runtime perform training and frozen
final-model validation. There is no teacher, imitation loss, reward shaping,
central MEC decision or best-checkpoint selection. Each User Vehicle still
chooses its own indivisible destination from its local received observation.
The actual objective (.60 delay/.25 mobile energy/.15 MEC use), EDF scheduling,
query/cache lifecycle, recovery and physical models are unchanged.

Five fixed policies, including the strong cached combined-cost greedy baseline,
are evaluated once on the same validation scenes. Their raw information,
physical parameters, objective and selection formulas are unchanged. Planned
execution is 160 training + 128 learned validation + 80 fixed-policy episodes,
368 physical executions in total. These are bounded synthetic development
experiments, not main training or a calibrated paper benchmark.

## Reproduction and interpretation

```text
python -m experiments.run_snr_representation_audit --output-dir reports/NEW-snr-representation-pilot
```

The output directory must be new. All eight final checkpoints, a full JSON
report and a Markdown comparison are saved exclusively. The loader checks
variant, schema/reference, complete protocol, algorithm/seed, source (when
provided), actual weights and update counters. A partial failed sweep is not
silently retried or summarized as a complete grid.

Each model includes training-only input/target diagnostics. Initial/final
networks are evaluated on that model's same retained final replay. Across
representations those replay observations are policy-dependent, so diagnostic
loss differences are not a common-replay causal effect. No diagnostic fits a
transform. The session diagnostic CLI remains scoped to its legacy journals;
its numerical helper recognizes both SNR feature names for this study.

Validation costs retain equal episode weights and separate models/seeds.
Log-minus-linear differences pair complete exogenous scenes. CVaR differences
subtract each model's CVaR; they are not CVaR of paired differences. With four
episodes and alpha .9, empirical CVaR equals the largest observed episode cost.
Seed summaries are conditional on the fixed traces; there is no pooling into
a policy tail, significance test, best-seed filtering or universal dominance
claim. Numerical conditioning, mean cost, tail cost and workload/deadline
metrics must be interpreted separately.

The completed pilot and all outcomes are recorded in
[`snr_representation_validation_20261121.md`](snr_representation_validation_20261121.md).
