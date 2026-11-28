# Twelfth Colab learning budget chain verification 2026 11 28

The received `log_snr_v4 / qr_dqn / seed 47` chain passed archive,
scientific-contract, recorded-history and all-51-checkpoint checks. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500 boundaries. All twelve declared learning chains and 612
checkpoints are now verified. The study remains **awaiting shared fixed
references**; no complete-study performance comparison, main training or
held-out paper evaluation has been performed.

## Scientific identity and matched controls

This is the risk-neutral QR-DQN control with log-SNR encoder schema 4,
51 quantiles and effective risk weight zero. The stored common risk-weight
parameter remains .5 and alpha .9; the existing algorithm configuration
sets the effective weight. Recorded validation minimizes mean predicted cost.
The primary method remains QR-DQN + CVaR with effective risk weight .5.

Independent User Vehicle decisions, information-only MEC, individual returns,
ordinary learned-epsilon collection, plain EDF and indivisible offloading
remain in effect. Scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M` without a deadline-miss penalty.
No source, objective, physical model, encoder, scenario or budget was retuned.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `132cf979ab77ef007826f8385b5dfb9304e97e7d5799e84ea5b370284de35365` |
| Final journal commit | `506bbc648eb77678c624d9faf92a6c7821fadf00ba1c44a2abe7cfb05607c2f9` |

Full contracts match pinned local plans. Compared with the
[linear neutral QR/seed-47 control](colab_budget_chain_11_20261128.md), only
`encoder`, `representation_contract` and `representation_variant` differ.
Compared with the [log-SNR CVaR/seed-47 chain](colab_budget_chain_9_20261128.md),
only `algorithm` and `agent_configuration` differ, with algorithm as the sole
nested configuration difference. Compared with the
[linear CVaR/seed-47 chain](colab_budget_chain_10_20261128.md), exactly the
union of these representation and algorithm fields differs.

All 500 training episode/seed/fingerprint/manifest records and 240
budget/profile/validation seed/fingerprint records match across the four
seed-47 arms. Initial online/target tensors, optimizer, empty replay and RNG,
exploration state and seed are exactly equal between the actual Colab
checkpoints. Configurations differ only by algorithm when applicable;
encoder signatures match within each representation. Shared initial online
fingerprint:

```text
e8e258c4bb4df6f3d4e2c1b6f564a793a28a3664c112cfe77c38cbaadab216ab
```

Against the [log-SNR neutral seed-11 control](colab_budget_chain_4_20261122.md)
and [seed-29 control](colab_budget_chain_8_20261128.md), only `agent_seed`
differs in each full contract, and exogenous input records match. Equal
initial weights are not required across learner seeds. Matched inputs and
same-seed initialization do not imply matched later queues, actions,
training clocks or outcomes.

Producer preflight, export and session runtime identities agree with the
other seed-47 arms and prior chains, with fingerprint
`f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`:
Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread
and deterministic execution. The auditor used Windows AMD64, CPython 3.12.14,
Torch 2.14.1+cpu and NumPy 2.5.4. No fresh Windows initialization comparison,
exact cross-platform continuation or runtime-guard relaxation was attempted.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103497 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

Update count is `8930 - 32 + 1 = 8899`. Indices are contiguous, with target
refresh flags every 50 updates through 8850. All checkpoints passed
restricted `weights_only=True` loading and the existing core validator.
Actual online fingerprints exactly match journal metadata. Configuration,
encoder schema 4, seed 47, scope and session metadata agree. Network and Adam
tensors are finite; Adam shapes/steps, replay values/cursors and restored
RNG states passed checks. Final replay size is 2048, cursor 738 and Adam
steps 8899. Target flags and saved endpoints were checked; intermediate
per-update target tensors are absent.

Final checkpoint: `steps/000052/agent.pt`. Actual online fingerprint:

```text
e232f2c32b120883e79c65bf0c484b72189e5795a4ed8c7c891ba1974e990c9d
```

All 740 reconstructed static scenario manifests match the history: 500
training keys and 80 fixed validation profile/case keys revisited at three
correlated checkpoints. Task ownership/order, epsilon schedule, recorded
mean-cost preferences and minimum-cost action selection passed checks.
Validation records have epsilon zero, no updates and unchanged training
clocks. Full before/after learner/RNG equality is a producer guard; the
audit verifies its persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 176 | 260 |
| Validation over all budgets | 240 | 7647 | 7647 | 166 | 211 |

All 16577 offered task instances eventually completed. Deadline misses and
fallback are distinct from incomplete tasks. Validation totals repeat a fixed
cohort and are accounting diagnostics, not independent samples or a
performance comparison. Prefix/closed and discounted costs reconcile with
per-user totals and episode metrics. Maximum absolute cost conservation
residual is `1.4210854715202004e-14` using
`rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are absent, so
physical objective components cannot be independently recomputed from
aggregate metrics. No simulator episode, optimizer update, policy evaluation,
public resume or held-out trajectory was executed by this audit. Original
source and session bytes were preserved.

## Archive evidence and complete learning grid

Received file: `colab-budget-log_snr_v4-qr_dqn-seed-47-bfde210de7e341659e2c53ca2bed0234.zip`.
Its 58,803,316 compressed bytes expand to
205,340,329 bytes across 161 members: 160 session files
and the export summary. Safe paths, unique names, regular types, CRCs,
artifact hashes and the full journal chain passed checks. The original
download remains untouched. ZIP SHA256:

```text
e2378336b101e6d6303b485edf33b9ef869a83a45affce78511ae750ac539994
```

Exclusive extraction and audit artifacts remain under the ignored directory
`reports/colab-budget-chain-12-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `4d5a692fc46e993fdcf1e23517c7f6ea0c14d8fe7cce10cc9ed073daa9a881ec` |
| identity_pairing_audit.json | `a1790e8747a35c02f917ca09d2dc1ecf53a5e096be722a5a8216e18b7bf5017b` |
| history_audit.json | `8c6ed0b5ce35d56f05bd9f2c6946b2482ec08ab1c0e50d38541e2933994f20e9` |
| checkpoint_audit.json | `8a15576540837037a91a1779814acc43ae2eb82e0f49805b0e39cb29aa8fba35` |
| study_progress.json | `d346e1456d76a7d5007301672dbf2286ba6276061c34bd31bdcdc59dc817aed5` |

The merged progress record verifies twelve complete chains under common
source/runtime identity, with no missing learning chains and original archives
kept separate. Prior audit-report digests were rechecked before merging.
All three seed factor grids have matched inputs and same-seed initial
producer states. This is a verified-progress record, not a run of the
complete-grid performance collector: curves, representation comparisons and
reference summaries remain null pending shared references and full analysis.

The verified learning runs account for 6000 training episodes and 2880
validation executions, or 8880 physical episode executions. Validation
reuses fixed cases across budgets and chains. The remaining five fixed
policies on 80 validation profile/case combinations require 400 executions,
bringing the planned study total to 9280. These references have not been run
or verified here. Completion of the learning grid does not complete the study
or establish convergence, a best representation, superiority over greedy,
statistical significance or a paper benchmark.

Only documentation and ignored audit code/results changed. Both READMEs
reflect this milestone. The earlier 573-test preparation suite was not rerun
for this data/documentation audit; see
[preparation verification](learning_budget_preparation_20261121.md).
Commit and push this milestone with Amir's authorization before preparing the
shared fixed-policy validation references under the same pinned source,
producer runtime, physical model and cohort. Do not rerun or select among the
completed learning chains. Main training remains a later, explicitly
announced Colab milestone after calibration and a frozen paper protocol.
See the [fixed study instructions](learning_budget_development.md).
