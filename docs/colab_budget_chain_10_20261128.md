# Tenth Colab learning budget chain verification 2026 11 28

The received `linear_v3 / qr_dqn_cvar / seed 47` chain passed archive,
scientific-contract, recorded-history and all-51-checkpoint checks. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500 boundaries. Ten of twelve development chains and 510 checkpoints
are verified. Both neutral QR controls at seed 47,
plus shared fixed references, remain outstanding. Complete-grid performance
comparisons remain withheld; main training and held-out evaluation remain
unstarted.

## Scientific identity and learner-seed controls

This chain uses QR-DQN + CVaR with 51 quantiles, alpha .9 and effective risk
weight .5, using linear SNR encoder schema 3. Independent User Vehicle decisions,
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
| Session contract | `927d37b1c2659e3e0627a2b2bdf8b62184764d916e7800c740607709570aa9b2` |
| Final journal commit | `4096bac9531cc9274eb0c9465b147d124085e74cc5ccb51f16e7789cdfeba7c7` |

The full contract matches pinned local plans. Compared separately with the
[linear CVaR/seed-11 chain](colab_budget_chain_2_20261122.md) and
[linear CVaR/seed-29 chain](colab_budget_chain_6_20261122.md), only `agent_seed`
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

Compared with the [log-SNR CVaR/seed-47 chain](colab_budget_chain_9_20261128.md),
only `encoder`, `representation_contract` and `representation_variant` differ
in the full contract. All 500 training and 240 validation input records
match. Initial online/target tensors, optimizer, empty replay and RNG,
exploration state, configuration and seed are exactly equal between these
two archived Colab models. Their initial online fingerprint shown above is
shared; no tolerance replaced producer-state equality or archive hashes.
Matched inputs and initialization do not imply matched later actions,
training clocks, queues or outcomes.

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
| Physical training slots | 104128 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

Update count follows `8930 - 32 + 1 = 8899`. Indices are contiguous, with
target refresh flags every 50 updates through 8850. All checkpoints passed
restricted `weights_only=True` loading and the existing core validator.
Actual online fingerprints exactly match journal metadata. Configuration,
encoder schema 3, seed 47, scope and session metadata agree. Network and Adam
tensors are finite; Adam shapes/steps, full replay values/cursors and restored
RNG states passed checks. Final replay size is 2048, cursor 738 and Adam
steps 8899. Target flags and saved endpoints were checked; intermediate
per-update target tensors are absent.

Final checkpoint: `steps/000052/agent.pt`. Actual online fingerprint:

```text
e7c33dbd1bdb5506b7cce1707431b383b290d3669adf428917b342162aa7840d
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
| Training | 500 | 8930 | 8930 | 229 | 288 |
| Validation over all budgets | 240 | 7647 | 7647 | 299 | 295 |

All 16577 offered task instances eventually completed. Deadline misses and
fallback remain distinct from incomplete tasks. Validation totals repeat a
fixed cohort and are accounting diagnostics, not independent samples or
partial-grid performance comparisons. Prefix/closed and discounted costs
reconcile with per-user totals and episode metrics. Maximum absolute cost
conservation residual is `2.1316282072803006e-14`
with `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are absent,
so physical objective components cannot be independently recomputed from
aggregate metrics. No simulator episode, optimizer update, policy evaluation,
public resume or held-out trajectory was executed during the audit.
Original source and session bytes were preserved.

## Archive evidence and next step

Received file: `colab-budget-linear_v3-qr_dqn_cvar-seed-47-3bbb1863b743410696539dd9131cc2a3.zip`.
Its 57,006,932 compressed bytes expand to
205,367,431 bytes across 161 members: 160 session files
and the export summary. Safe paths, unique names, regular file types, CRCs,
artifact hashes and journal chain passed checks. Original download remains
untouched. ZIP SHA256:

```text
f7fda9ff398c3d342af29c9ebd6b53cd2ac9bb57e7ef900e48a382e32f57d2cd
```

Exclusive extraction and audit artifacts remain under the ignored directory
`reports/colab-budget-chain-10-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `650dceafe2526f38497ab8a1456650bdfdfc88ab358c1af7377723524e8ca2f6` |
| identity_pairing_audit.json | `b906b142d2d005435d2baa99a8c20c3cd07c580318066cc46a4a9359b26ac1d6` |
| history_audit.json | `f35b5c49586d243bc203841cc56a983589c12bdae9e9e74f284b202e0d6d2a69` |
| checkpoint_audit.json | `50d24c389bd0be6ec8f7a6c206e7d6371c4273b9d929658cab24c6af699f30a7` |
| study_progress.json | `5d21546ade4fa22ce006e7255e7ffc4d62925216b22f4b5d60e5565fd0b546ec` |

Combined progress contains ten verified and two missing chains under
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
`linear_v3 / qr_dqn / seed 47` under the same source/runtime/protocol.
Main training remains a later, explicitly announced Colab milestone after
calibration and a frozen paper protocol. See the
[fixed study instructions](learning_budget_development.md).
