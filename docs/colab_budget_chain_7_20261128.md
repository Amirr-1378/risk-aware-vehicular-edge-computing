# Seventh Colab learning budget chain verification 2026 11 28

The received `linear_v3 / qr_dqn / seed 29` chain passed archive, full-contract,
recorded-history and all-51-checkpoint checks. It completed 500 training
episodes and 240 validation executions at the fixed 20/100/500 boundaries.
Seven of twelve development chains and 357 checkpoints are now verified.
The log-SNR neutral QR/seed-29 chain, four seed-47 chains and shared fixed
references remain outstanding. Complete-grid performance comparisons remain
withheld; main training and held-out evaluation remain unstarted.

## Scientific identity and controls

This chain is the risk-neutral QR-DQN control. It retains 51 quantiles,
alpha .9 and the stored common risk-weight parameter .5, while the existing
algorithm configuration makes the **effective risk weight zero**. Recorded
validation selection minimizes mean predicted cost. The primary method
remains QR-DQN + CVaR, with effective risk weight .5. No objective, reward,
source, budget, encoder or physical model was changed for this result.

Independent User Vehicle decisions, information-only MEC, individual returns,
ordinary learned-epsilon collection, linear SNR encoder schema 3, plain EDF
and indivisible offloading remain in effect. Scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M` without a deadline-miss penalty.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `f99d53a7388236f657d207ed768573580d8caf7d30d4d041200d85a809df8a57` |
| Final journal commit | `d0e5407c0eec991fb38a36bcb706097faa9ff82bb4cd8f54b8cbc3f696f7fcb5` |

The full contract matches the pinned local plans. Compared with the
[linear CVaR/seed-29 chain](colab_budget_chain_6_20261122.md), only `algorithm`
and `agent_configuration` differ; the nested configuration differs only in
`algorithm`. All 500 training episode/seed/fingerprint/manifest records and
240 budget/profile/validation seed/fingerprint records match. Initial online
and target tensors, optimizer, empty replay and its RNG, exploration state,
encoder and seed exactly match between the actual producer checkpoints.
The initial online fingerprint is `c08cba4ec8a01a8dbe8813038574c25121cb82388b1533d26f32d1411d7f754a`.
Initial configurations differ only in algorithm. Against the
[linear neutral seed-11 control](colab_budget_chain_3_20261122.md), only
`agent_seed` differs in the full contract, and exogenous input records match.
Matched inputs and initialization do not imply matched later actions or costs.

Preflight, export and session producer-runtime identities agree and match
earlier chains, with fingerprint
`f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`:
Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread
and deterministic execution. The auditor used Windows AMD64, CPython 3.12.14,
Torch 2.14.1+cpu and NumPy 2.5.4. No new Windows initialization comparison,
cross-platform exact-continuation claim or runtime-guard relaxation is made.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 104206 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update count is `8930 - 32 + 1 = 8899`; update indices are contiguous,
with target refresh flags every 50 updates through 8850. All checkpoints
passed restricted `weights_only=True` loading and the existing core validator.
Actual online fingerprints exactly match journal metadata. Configuration,
encoder, seed, scope and session metadata agree. Online, target and Adam
tensors are finite; Adam shapes/steps, replay values/cursors and restored RNG
states passed checks. Final replay size is 2048, cursor 738 and Adam steps
8899. Target flags and saved synchronization endpoints were checked;
intermediate per-update target tensors are absent.

Final checkpoint: `steps/000052/agent.pt`. Actual online fingerprint:

```text
0d610b07d24d89db44171c75c4a12ae9b0803164e937a7d68b37765df9ddddb4
```

All 740 static scenario-manifest reconstructions match the history: 500
training keys plus 80 fixed validation profile/case keys revisited at three
correlated checkpoints. Task ownership/order, epsilon schedule, recorded
mean-cost preferences and minimum-cost action selection passed checks.
Recorded validation has epsilon zero, no updates and unchanged training
clocks. Full before/after learner/RNG equality is a producer guard; the
audit verifies the persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 224 | 372 |
| Validation over all budgets | 240 | 7647 | 7647 | 385 | 328 |

All 16577 offered task instances eventually completed. Deadline misses and
fallback are distinct from incomplete tasks. These totals are accounting
diagnostics; validation repeats the same cohort, so they are not independent
sample counts or a partial-grid performance comparison. Prefix/closed and
discounted costs reconcile with per-user totals and episode metrics. Maximum
absolute conservation residual is `2.1316282072803006e-14`,
using `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are
absent, so the physical objective components cannot be independently
recomputed from the aggregate metrics. No simulator episode, optimizer
update, policy evaluation, public resume or held-out trajectory was executed
by this audit. Source and session bytes were preserved.

## Archive evidence and next step

Received file: `colab-budget-linear_v3-qr_dqn-seed-29-916ab3672f2244028e311e8c852aa72a.zip`.
It contains 56,604,538 compressed bytes and
205,039,286 uncompressed bytes across 161 members:
160 session files and the export summary. Safe paths, regular file types,
unique names, CRCs, artifact hashes and the journal chain passed checks.
The original download remains untouched. ZIP SHA256:

```text
107554bdcb4d8604d3193f69320410495fd02b7e88d65a329eb085a4f1e3f9bd
```

Exclusive extraction and local evidence are retained in the ignored directory
`reports/colab-budget-chain-7-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `c481ca1bc08a40a75517bbefd7d18fbc88d5438ac09166b018d2adcef52895ab` |
| identity_pairing_audit.json | `69fef78bed2509b42aab395d3797839c3615fd6b66945ba9243fd123d6b16153` |
| history_audit.json | `24dbf02a54def5a6cbcf44bab17b6475771f3f6b98e066b904914ed34ecadbfe` |
| checkpoint_audit.json | `8f6aa3415782c31d8d04e1680bc0dc5ce255ed19b347edc22bb2e52ed127c7a2` |
| study_progress.json | `d7c11090523e0b2e51478c3aadf147c966c0e00241edfc2be5eee7f91e9cbfa0` |

Progress is merged by representation/algorithm/seed after separate archive
checks under common source and runtime identity; original archives remain
separate. Seven chains are verified and five are missing. Performance curves,
representation comparisons and fixed-reference summaries remain null.
No best-seed/checkpoint selection, representation adoption, convergence or
greedy-improvement claim follows from these checks.

Both READMEs reflect this milestone. Only documentation and ignored audit
code/results changed. The earlier 573-test preparation suite was not rerun
for this data/documentation audit; see
[preparation verification](learning_budget_preparation_20261121.md).
Commit and push the record with Amir's authorization before advancing to
`log_snr_v4 / qr_dqn / seed 29` under the same source/runtime/protocol.
Main training remains a later, explicitly announced Colab milestone after
calibration and a frozen paper protocol. See the
[fixed study instructions](learning_budget_development.md).
