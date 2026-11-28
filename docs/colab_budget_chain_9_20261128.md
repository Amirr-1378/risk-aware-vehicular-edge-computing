# Ninth Colab learning budget chain verification 2026 11 28

The received `log_snr_v4 / qr_dqn_cvar / seed 47` chain passed archive,
scientific-contract, recorded-history and all-51-checkpoint checks. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500 boundaries. Nine of twelve development chains and 459 checkpoints
are verified. The linear CVaR and both neutral QR controls at seed 47,
plus shared fixed references, remain outstanding. Complete-grid performance
comparisons remain withheld; main training and held-out evaluation remain
unstarted.

## Scientific identity and learner-seed controls

This chain uses QR-DQN + CVaR with 51 quantiles, alpha .9 and effective risk
weight .5, using log-SNR encoder schema 4. Independent User Vehicle decisions,
information-only MEC, individual returns, ordinary learned-epsilon collection,
plain EDF and indivisible offloading remain in effect. Scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M` without a deadline-miss penalty.
No source, objective, encoder, physical model, scenario or budget was retuned.
Learner seed 47 is the third seed already declared in the fixed plan.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `b880ac2acf1ab18a1a6a606a0140bee8a74abb060a1a5c1106e5074004f72d40` |
| Final journal commit | `7f6343ae638fac1093158d929369bc778c672609b7aefe294b1a31a50afdccb5` |

The full contract matches pinned local plans. Compared separately with the
[log-SNR CVaR/seed-11 chain](colab_budget_chain_1_20261122.md) and
[log-SNR CVaR/seed-29 chain](colab_budget_chain_5_20261122.md), only `agent_seed`
differs in the full contract. All 500 training episode/seed/fingerprint/
manifest records and all 240 budget/profile/validation seed/fingerprint
records match. The learner seed changes initialization and sampling while
exogenous scenario inputs remain shared.

Actual producer initialization checkpoints have equal configurations,
encoder signatures, empty optimizer state and exploration state across these
three seeds. Seed-47 online/target tensors and replay RNG differ from both
seed 11 and seed 29, with empty replay in all cases. This establishes recorded
learner-seed separation, not statistical independence of outcomes or scenario
samples. The seed-47 initial online fingerprint is:

```text
e8e258c4bb4df6f3d4e2c1b6f564a793a28a3664c112cfe77c38cbaadab216ab
```

No fresh Windows reference model was constructed, and no cross-platform
initialization equality was required. Producer preflight, export and session
runtime identities agree with both prior controls, with fingerprint
`f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`:
Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread
and deterministic execution. The auditor used Windows AMD64, CPython 3.12.14,
Torch 2.14.1+cpu and NumPy 2.5.4. Read-only inspection does not establish
cross-platform exact continuation. Runtime guards remained unchanged.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103341 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

Update count follows `8930 - 32 + 1 = 8899`. Indices are contiguous, with
target refresh flags every 50 updates through 8850. All checkpoints passed
restricted `weights_only=True` loading and the existing core validator.
Actual online fingerprints exactly match journal metadata. Configuration,
encoder schema 4, seed 47, scope and session metadata agree. Network and Adam
tensors are finite; Adam shapes/steps, full replay values/cursors and restored
RNG states passed checks. Final replay size is 2048, cursor 738 and Adam
steps 8899. Target flags and saved endpoints were checked; intermediate
per-update target tensors are absent.

Final checkpoint: `steps/000052/agent.pt`. Actual online fingerprint:

```text
e2badfc2fc26d75a1180bd8b8ec13d36acc473c35c8b3dc1c3fa72d8febdfdc6
```

All 740 reconstructed static scenario manifests match recorded history:
500 training keys and 80 fixed profile/validation-case keys revisited at
three correlated checkpoints. Task ownership/order, epsilon schedule,
recorded CVaR preference formula and minimum-risk-cost action selection
passed checks. Validation records have epsilon zero, no updates and unchanged
training clocks. Full before/after learner/RNG equality is a producer guard;
the audit verifies the persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 146 | 296 |
| Validation over all budgets | 240 | 7647 | 7647 | 129 | 126 |

All 16577 offered task instances eventually completed. Deadline misses and
fallback remain distinct from incomplete tasks. Validation totals repeat a
fixed cohort and are accounting diagnostics, not independent samples or
partial-grid performance comparisons. Prefix/closed and discounted costs
reconcile with per-user totals and episode metrics. Maximum absolute cost
conservation residual is `1.4210854715202004e-14`
with `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are absent,
so physical objective components cannot be independently recomputed from
aggregate metrics. No simulator episode, optimizer update, policy evaluation,
public resume or held-out trajectory was executed during the audit.
Original source and session bytes were preserved.

## Archive evidence and next step

Received file: `colab-budget-log_snr_v4-qr_dqn_cvar-seed-47-3d6b1844695247a1b957aa32af41d952.zip`.
Its 58,881,168 compressed bytes expand to
205,479,776 bytes across 161 members: 160 session files
and the export summary. Safe paths, unique names, regular file types, CRCs,
artifact hashes and journal chain passed checks. Original download remains
untouched. ZIP SHA256:

```text
e9dd2d4d69d880e2c3e30644d4bd75bf2bec5e0d5a5b902e1452e9c8a05a0e33
```

Exclusive extraction and audit artifacts remain under the ignored directory
`reports/colab-budget-chain-9-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `1f7f4e37bdf4ccb8e8b64bcae34771fa8cd15e4b61e2b2b2ed3d22157bfe458a` |
| identity_seed_audit.json | `989d7bdf79fbd73e280c6241362adcbdd4a113839b43111c15517814b21c406a` |
| history_audit.json | `740db40814cc634e95172b8d1bb666c078e0e0e32dbae0b055b2f9ac035e1bcc` |
| checkpoint_audit.json | `8cd960d1bab06cd8463e9d1ad0dec852898aed0d705c9e43b51850758dce86e3` |
| study_progress.json | `9d3525fd1903b1094b926d71415a6e9cf42e32f3b5d2475dcd37b8ded1cf7562` |

Combined progress contains nine verified and three missing chains under
common scientific source and producer runtime identity, with original archives
kept separate. Performance curves, representation comparisons and fixed
reference summaries remain null. No best-seed/checkpoint selection,
representation adoption, convergence or greedy-improvement claim follows
from these identity and integrity checks.

Only documentation and ignored audit code/results changed. Both READMEs
reflect this milestone. The earlier 573-test preparation suite was not rerun
for this data/documentation audit; see
[preparation verification](learning_budget_preparation_20261121.md).
Commit and push this milestone with Amir's authorization before advancing to
`linear_v3 / qr_dqn_cvar / seed 47` under the same source/runtime/protocol.
Main training remains a later, explicitly announced Colab milestone after
calibration and a frozen paper protocol. See the
[fixed study instructions](learning_budget_development.md).
