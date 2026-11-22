# Sixth Colab learning budget chain verification 2026 11 22

The received `linear_v3 / qr_dqn_cvar / seed 29` chain passed archive,
scientific-contract, recorded-history and all-checkpoint checks. It completed
500 training episodes and 240 validation executions at the fixed 20/100/500
boundaries. Six of twelve development chains are verified: all four seed-11
combinations and both CVaR representations at seed 29. The two neutral QR
controls at seed 29, four seed-47 chains and shared fixed references remain
outstanding. Complete-grid performance comparisons remain withheld; main
training and held-out evaluation remain unstarted.

## Scientific identity and paired producer states

The run retains independent User Vehicle decisions, individual returns,
ordinary learned-epsilon collection, QR-DQN + CVaR with alpha .9 and effective
risk weight .5, and the linear SNR encoder, schema 3. The scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, with plain EDF and indivisible
offloading. No objective, physical model, budget or source was retuned.

The full contract matches the trusted source and expanded plans from pinned
scientific commit `5d8845bb0605def86541fd172a961be5211a7ba0`. The export's
declared commit is provenance information; source and full-contract checks
establish the identities used here.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `33d17c140bf77cc2b68ab55d6486d0c05efa5dfccc20da546481f46a667772b9` |
| Final journal commit | `bd5da07c43e9716855624cff4ce15109ab5ce5e355ecb0c2a6bfadaee77c07d6` |

Compared with the [log-SNR CVaR/seed-29 chain](colab_budget_chain_5_20261122.md),
only `encoder`, `representation_contract` and `representation_variant` differ
in the full contract. Agent configuration and seed 29 agree. All 500 training
seed/fingerprint/manifest records and all 240 budget/profile/validation
seed/fingerprint records match. Initial online and target tensors, optimizer,
empty replay and RNG, exploration state, configuration and seed are exactly
equal between the two received producer checkpoints. Their initial online SHA is:

```text
c08cba4ec8a01a8dbe8813038574c25121cb82388b1533d26f32d1411d7f754a
```

This comparison uses the two archived Colab models. It does not use the
newly generated Windows reference discussed in the fifth-chain diagnostic,
and no approximate tolerance replaced producer-state equality or archive
fingerprints. Later actions, queues, clocks and outcomes can differ.

Producer preflight, export and session runtime identities agree and match
earlier chains: Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6,
one CPU thread and the deterministic execution contract. The local auditor
used Windows AMD64, CPython 3.12.14, Torch 2.14.1+cpu and NumPy 2.5.4.
Read-only inspection establishes the recorded-state checks below; it does
not establish bitwise agreement of independently repeated training across
platforms. No experiment resume or runtime-guard relaxation occurred.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 104209 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update count follows `8930 - 32 + 1 = 8899`. All update indices are
contiguous and refresh flags occur every 50 updates through 8850. Every
checkpoint passed restricted `weights_only=True` loading and the existing
core validator. Actual online tensor fingerprints exactly match all journal
fingerprints. Configuration, encoder schema 3, seed 29, scope and source/
session metadata agree. Online, target and Adam tensors are finite; optimizer
shapes/steps, full replay values/cursors and restored RNG states passed checks.
Final replay size is 2048, cursor 738 and Adam steps 8899. Saved target
synchronization endpoints were checked; individual-update intermediate target
tensors are absent.

The final checkpoint is `steps/000052/agent.pt`; its actual online model SHA is:

```text
28a3047f98f1f45965c10388fd1412a1278f700596184d68e00888a79e881248
```

All 740 reconstructed episode manifests match the recorded history: 500
training keys and 80 fixed profile/validation-case keys revisited at three
correlated checkpoints. Task ownership/order, epsilon schedules and CVaR
preference/minimum-cost selection passed checks. Recorded validation has
epsilon zero, no teacher/updates and unchanged training clocks. Full before/
after learner/RNG equality is a producer guard; the audit verifies its
persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 235 | 307 |
| Validation over all three budgets | 240 | 7647 | 7647 | 294 | 322 |

All 16577 offered task instances completed. Deadline misses and fallback are
separate outcomes from eventual completion. Validation totals repeat the
fixed cohort and are execution/accounting diagnostics, not independent sample
counts or partial-grid performance comparisons. Prefix/closed and discounted
costs reconcile with user totals and episode metrics. Maximum absolute
conservation residual is `1.4210854715202004e-14`, using `rel_tol=1e-10` and
`abs_tol=1e-10`. Raw final per-task ledgers are absent for independent physical
objective-component recomputation. The audit executed no environment episode,
optimizer update, policy evaluation or held-out trajectory. Original session
bytes were preserved.

## Archive evidence and study status

The received file is
`colab-budget-linear_v3-qr_dqn_cvar-seed-29-78e2b92a24fc4339a27eef55828033fc.zip`,
57,644,111 bytes, expanding to 205,164,970 bytes. Its 161 members contain 160
session files and an export summary. Safe paths, regular types, unique names,
CRCs, all artifact hashes and the final journal commit passed checks. Its SHA is:

```text
80ad7f0aa7beb6c0215bdddcea2e63d901fbacc290d33e60695dac2f40247605
```

The original download remains untouched. Exclusive extraction and audit
artifacts are under the ignored directory
`reports/colab-budget-chain-6-5d8845b-20261122/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `d0eaebdae4438cac5ef4a575333267d5a0cdfee9ae63fe9133f1146b6607c980` |
| identity_pairing_audit.json | `70ec765256e1cdd17877f2054b87264771dde9d296294ae56bfeca1f8540a85d` |
| history_audit.json | `b72421c6715688f9c3c408791e11baca2ce1c2ecd0a0d809612d2aa6ec3620da` |
| checkpoint_audit.json | `406e6b66bc90aedd0b868fa267500b7aea3ce5fa187b978023321ebddb73d596` |
| study_progress.json | `e08ef6a81b1d43e78973dd9b0ada6e887a3b483a0ddacee6f23d13860aaf9f63` |

Each archive was checked separately against the full plan before progress
was merged by representation/algorithm/seed under common source and producer
runtime identity. The combined record has six verified and six missing chains,
with performance curves, representation comparisons and fixed-reference
summaries null. Shared references remain outstanding. No best seed,
checkpoint, representation adoption, convergence or greedy-improvement claim
follows from these identity/integrity checks.

Only documentation and ignored audit code/results changed. The 573-test suite
belongs to the earlier [preparation verification](learning_budget_preparation_20261121.md);
it was not rerun for this received-data audit. Keep scientific source, fixed
budgets/cohort and exact producer runtime identity for remaining chains.
Commit and push the record and updated READMEs with Amir's authorization before
advancing. Main training, calibration and held-out paper evaluation remain
later work. See [the fixed study instructions](learning_budget_development.md).
