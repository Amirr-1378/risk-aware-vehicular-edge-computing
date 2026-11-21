# SNR representation validation 2026 11 21

The versioned logarithmic representation reduced the extreme numerical input/
initial-output scale in the declared development pilot. It did not consistently
reduce mean episode cost: log-SNR had lower mean in 3 of 16 paired model/profile
cells and lower empirical CVaR in 10 of 16. The unchanged combined-cost greedy
baseline retained a lower observed mean in all 32 learned model/profile cells.
The new representation remains an explicit candidate; legacy defaults were
not silently switched and main training was not started.

## Method and execution

The new feature is `log1p(s)/log1p(100)` for cached large-scale linear SINR s.
The reference 100 is the pre-existing fixed feature reference, with no fitting
or clipping. Raw observations/rates, channel formulas, actual three-term cost,
plain EDF, query/cache/recovery, independent User Vehicle decisions and QR/CVaR
updates remain unchanged. The outer plan and schema boundaries are defined in
[`snr_representation_development.md`](snr_representation_development.md).

```text
python -m experiments.run_snr_representation_audit --output-dir reports/snr-representation-pilot-v1-20261121
```

The predeclared grid was linear-v3/log-SNR-v4 x QR/QR-CVaR x seeds 11/29:
eight models, each with the same 20 nominal training scenes and four validation
cases per known profile. The shared budget/configuration is exactly the earlier
paired study's learning template; no budget/reference/seed was changed after
viewing the outcomes.

- 160 training, 128 learned validation and 80 fixed-policy episodes completed:
  **368 physical executions** and **9731 completed task instances**. Task
  instances across models/profiles are correlated and not independent samples.
- Each model received **347 transitions**, made **316 optimizer updates** and
  **six target refreshes**, and ended at epsilon .05. Physical training clocks
  differed naturally with actions: 4133–4209 slots.
- Initial network weights matched across representations for each algorithm/
  seed. The old linear controls' four checkpoint payloads exactly matched the
  corresponding historical individual-return controls except `run_metadata_json`,
  including online/target weights, optimizer, replay/RNGs, counters and encoder.
- All eight final checkpoints reloaded with matching representation/protocol/
  source identities and actual tensor fingerprints. Empty cross-encoder loads,
  mixed replay/current-next schemas, wrong references and altered metadata/
  weights are rejected by tests.
- Final validation used epsilon zero, no teacher/updates and frozen learner
  state. Fixed policies used the same exogenous validation fingerprints and
  were evaluated once, not duplicated across representations or agent seeds.

## Numerical diagnostics

Diagnostics use each model's own final retained training replay. Initial/final
networks share that cohort within a model; replay states can differ across
representations because policies differ. The figures are retrospective scale
descriptions, not common-replay causal loss effects.

| Diagnostic over the four models in each arm | Linear v3 | Log-SNR v4 |
| --- | --- | --- |
| Largest encoded SNR coordinate | 7052889.893 | 4.414645 |
| Initial maximum absolute valid-action quantile range | 185529.75–206015.41 | 1.83–3.57 |
| Initial fixed-cohort QR loss range | 68311.91–79385.01 | 2.17–3.50 |
| Final fixed-cohort QR loss range | 1.31–6092.11 | 0.94–1.39 |
| SNR fraction of squared current-input magnitude | above 99.9999999% | 2.90–2.94% |

The larger raw maximum than in the earlier four-episode Colab check reflects
the declared 20-scene cohort, not a change in channel parameters. A finite,
well-scaled input/output does not establish an effective policy or calibrated
risk estimates. Loss scale also depends on the 51-quantile reduction.

## All observed episode costs

Each cell is **mean / empirical CVaR .9** of four full-drain episode total costs
for one fixed model. At this sample size, empirical CVaR is the largest observed
episode cost. Seeds are separate; no policies or seeds are pooled into a tail.

| Representation / algorithm | Seed | Nominal | High load | Fast servers | Harsh channel |
| --- | --- | --- | --- | --- | --- |
| Linear / QR | 11 | 4.756417 / 7.247399 | 24.614363 / 46.861051 | 4.989737 / 8.510566 | 4.918597 / 7.290068 |
| Linear / QR | 29 | 4.755951 / 7.517928 | 22.243812 / 36.891525 | 4.618647 / 7.256257 | 4.595682 / 7.080566 |
| Linear / QR-CVaR | 11 | 4.905523 / 7.985079 | 21.112154 / 33.992825 | 4.839178 / 7.799225 | 4.688978 / 7.563597 |
| Linear / QR-CVaR | 29 | 5.138194 / 7.824351 | 22.403448 / 39.015151 | 5.103992 / 7.792757 | 5.148209 / 8.426194 |
| Log-SNR / QR | 11 | 5.039457 / 7.107205 | 27.316518 / 40.864322 | 5.153744 / 7.279358 | 5.201133 / 7.341778 |
| Log-SNR / QR | 29 | 5.044022 / 8.106203 | 21.587082 / 35.186021 | 5.132185 / 8.430866 | 4.839679 / 7.267754 |
| Log-SNR / QR-CVaR | 11 | 5.114058 / 7.239278 | 25.033348 / 44.340468 | 5.322227 / 7.526942 | 5.290230 / 7.730568 |
| Log-SNR / QR-CVaR | 29 | 5.152120 / 6.985467 | 22.293060 / 34.139202 | 5.177502 / 7.029777 | 5.053807 / 7.096281 |
| AlwaysMEC | n/a | 5.789754 / 7.029909 | 37.114040 / 60.712403 | 5.789754 / 7.029909 | 5.789754 / 7.029909 |
| AlwaysV2V | n/a | 4.546555 / 7.485865 | 21.113280 / 36.298902 | 4.503396 / 7.367386 | 4.507069 / 7.365775 |
| MinimumEstimatedLatency | n/a | 5.404850 / 6.752721 | 19.689928 / 29.474545 | 5.387252 / 6.708338 | 5.428336 / 6.832544 |
| UniformRandom | n/a | 4.830544 / 7.127920 | 18.885574 / 29.016709 | 4.780093 / 6.944051 | 4.476009 / 6.252932 |
| MinimumEstimatedCombinedCost | n/a | 4.226148 / 6.136148 | 18.740129 / 29.058878 | 4.161238 / 5.991601 | 4.180286 / 6.130503 |

Log-SNR had fewer deadline misses in 10 paired cells, equal counts in two, and
more in four. Its mean completed SV CPU-work fraction was lower in **all 16**
paired cells. Thus improved numerical conditioning did not meet the combined
goals of lower mean cost and greater SV use in this pilot. Mean SV workload
fractions span 30.29–82.20% for log-SNR and 75.21–97.94% for linear inputs;
these are unweighted means of per-episode fractions, not pooled task shares.

The paired counts are dependent descriptive observations, not independent win
rates, significance tests or population tail estimates. Two trained seeds and
four known validation cases/profile cannot establish reliable generalization.
No additional seed, reference, risk weight or training budget was selected to
obtain a favorable cell.

## Artifacts and verification

The ignored directory `reports/snr-representation-pilot-v1-20261121/` contains
`report.json` (14,523,000 bytes), `summary.md`, eight final checkpoints,
`analysis.json` and `additional_metrics.json`. Reports retain full training/
validation traces, input diagnostics, all baseline cells and paired differences.
Historical checkpoints, reports and JSON configurations were preserved.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `1c46e7e29488b3d851b81f6c68eb99cda8ec76f62fa48696d7a7950d318f67c0` |
| SNR representation protocol | `d5ca5a2381f99d589bfb911b54aa7a16d5ce7e45a4e4305cf411faf5ca2996f7` |
| Shared unchanged learning budget | `64236c2e679444dfa87b8cb73c051f82aa187b9bf6b90868671319fc3ed6c263` |
| Physical scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Report JSON bytes | `6af9fe3389579ccf47a948518bda87754184ed2981f8212f6fe63e4876ed0105` |
| Reload/control analysis | `b61b3a26aebcee404d410d37571ad3afc215049e0dfc4393ccd9c6cf58674187` |

Provenance records parent commit
`b6bb3f407f901db2f6ee63eb66404ee3563845af` and `git_dirty=true`, identifying
the uncommitted implementation actually executed. Runtime was Python 3.12.14,
Torch 2.14.1+cpu, NumPy 2.5.4, one CPU thread and deterministic algorithms.

The full learning-runtime suite discovered **525 tests: 524 passed, one skipped**
for Windows symlink privileges. The standard-library run discovered the same
525: **358 passed, 167 skipped** (166 optional learning tests and the symlink
restriction). Tests cover the transform/edge cases, exact legacy encoding,
schema/replay/checkpoint guards, same-schema next-update resume, matching
initialization/budgets, unchanged linear control, frozen validation, output
exclusivity and complete paired statistics. `git diff --check` passed.

## Next boundary

The implementation now supports a numerically balanced representation, but
this pilot does not justify a claim of lower cost or automatic replacement of
the existing configuration. Preserve both arms and the strong greedy baseline.
Next declare a learning-budget/information-sufficiency study and address
physical calibration before main training; do not change objective weights or
weaken the baseline to obtain a win. Commit and push this verified milestone
before advancing. Main training remains in Colab, with notice to Amir first.
