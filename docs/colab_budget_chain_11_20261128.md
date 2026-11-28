# Eleventh Colab learning budget chain verification 2026 11 28

The received `linear_v3 / qr_dqn / seed 47` chain passed archive,
scientific-contract, recorded-history and all-51-checkpoint checks. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500 boundaries. Eleven of twelve development chains and 561 checkpoints
are verified. The log-SNR neutral QR/seed-47 control and shared fixed
references remain outstanding. Complete-grid performance
comparisons remain withheld; main training and held-out evaluation remain
unstarted.

## Scientific identity and learner-seed controls

This chain is the risk-neutral QR-DQN control with 51 quantiles and linear
SNR encoder schema 3. It retains alpha .9 and the stored common risk-weight
parameter .5, while the existing QR configuration sets **effective risk weight
zero**. Recorded validation minimizes mean predicted cost. The primary method
remains QR-DQN + CVaR with effective risk weight .5.

Independent User Vehicle decisions, information-only MEC, individual returns,
ordinary learned-epsilon collection, plain EDF and indivisible offloading
remain in effect. Scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M` without a deadline-miss penalty.
No objective, source, physical model, encoder, scenario or budget was retuned.
Learner seed 47 is the third seed already declared in the fixed plan.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `7db9306c530c70279d7e34aec0c5212f5d7fbec723f24d563350972ad0e69296` |
| Final journal commit | `b8806219eefdb8d1b1c2eca99376a2a79ddd2a52cc07c3104551f1999e2f1376` |

The full contract matches pinned local plans. Compared separately with the
[linear neutral QR/seed-11 chain](colab_budget_chain_3_20261122.md) and
[linear neutral QR/seed-29 chain](colab_budget_chain_7_20261128.md), only `agent_seed`
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

Compared with the [linear CVaR/seed-47 chain](colab_budget_chain_10_20261128.md),
only `algorithm` and `agent_configuration` differ in the full contract;
algorithm is the sole nested configuration difference. All 500 training
and 240 validation input records match. Initial online/target tensors,
optimizer, empty replay and RNG, exploration state, encoder and seed are
exactly equal between these two actual Colab checkpoints. Initial checkpoint
configurations differ only by algorithm. The initial online fingerprint
shown above is shared. No tolerance replaced producer-state equality or
archive hashes. Matched inputs and initialization do not imply matched later
actions, training clocks, queues or outcomes.

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
| Physical training slots | 104170 |
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
db76dc634faabea59887d79422a6663946d606c8c84236773b0a38797025c9d6
```

All 740 reconstructed static scenario manifests match recorded history:
500 training keys and 80 fixed profile/validation-case keys revisited at
three correlated checkpoints. Task ownership/order, epsilon schedule,
recorded mean-cost preferences and minimum-cost action selection
passed checks. Validation records have epsilon zero, no updates and unchanged
training clocks. Full before/after learner/RNG equality is a producer guard;
the audit verifies the persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 248 | 346 |
| Validation over all budgets | 240 | 7647 | 7647 | 286 | 247 |

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

Received file: `colab-budget-linear_v3-qr_dqn-seed-47-14ab287bde6d46eaadc3df7e3499af56.zip`.
Its 57,026,676 compressed bytes expand to
205,295,465 bytes across 161 members: 160 session files
and the export summary. Safe paths, unique names, regular file types, CRCs,
artifact hashes and journal chain passed checks. Original download remains
untouched. ZIP SHA256:

```text
bd401748bd3f6c2bb1cb83e95d134a20ea07502d8e07dcf3427b5dcee4d28bd4
```

Exclusive extraction and audit artifacts remain under the ignored directory
`reports/colab-budget-chain-11-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `04785d85c871f70bd487fb296ec7d15b198bec100bd43f7f7b0aaacafdc19c09` |
| identity_pairing_audit.json | `1a8c14fcc3ad20cabf8b612815fe50e0976be838d3a80e431cb85212d246d018` |
| history_audit.json | `9493292b58bdcd2a51b1d8ce95c9de72c165fe6363926e4ec47753a4fae24fef` |
| checkpoint_audit.json | `9ddde8ea711d0a63d5289ef796b60e38265c3582cc4912222c198584e3cd74c2` |
| study_progress.json | `58c3e3d0737d530c6391b3c5d0a41a71f9d3ab9426da960ab5e3863553cedab7` |

Combined progress contains eleven verified and one missing chain under
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
`log_snr_v4 / qr_dqn / seed 47` under the same source/runtime/protocol.
Main training remains a later, explicitly announced Colab milestone after
calibration and a frozen paper protocol. See the
[fixed study instructions](learning_budget_development.md).
