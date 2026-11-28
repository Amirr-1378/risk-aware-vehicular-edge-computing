# Eighth Colab learning budget chain verification 2026 11 28

The received `log_snr_v4 / qr_dqn / seed 29` chain passed archive,
scientific-contract, recorded-history and all-51-checkpoint checks. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500 boundaries. All four representation/risk combinations at seeds
11 and 29 are verified: eight of twelve chains and 408 checkpoints.
Four seed-47 chains and shared fixed references remain outstanding.
Complete-grid performance comparisons remain withheld; main training and
held-out evaluation remain unstarted.

## Scientific identity and matched controls

This chain is the risk-neutral QR-DQN control with log-SNR encoder schema 4.
It keeps 51 quantiles, alpha .9 and the stored common risk-weight parameter
.5; the existing QR algorithm configuration sets **effective risk weight zero**.
Recorded validation minimizes mean predicted cost. The primary method remains
QR-DQN + CVaR with effective risk weight .5. No source, objective, encoder,
physical model, scenario, seed or budget was retuned for this result.

Independent User Vehicle decisions, information-only MEC, individual returns,
ordinary learned-epsilon collection, plain EDF and indivisible offloading
remain in effect. Scalar cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M` without a deadline-miss penalty.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `36e75db32c46677a28b574721fbf5cb9a03eeef71329d10beab4a05e367fe584` |
| Final journal commit | `dda048d96e0f5d2676bf7267a5d2e2618d9ed972e77ef3b30fb850d8cb3e10a2` |

Full contracts match the pinned local plans. Compared with the
[linear neutral QR/seed-29 control](colab_budget_chain_7_20261128.md), only
`encoder`, `representation_contract` and `representation_variant` differ.
Compared with the [log-SNR CVaR/seed-29 chain](colab_budget_chain_5_20261122.md),
only `algorithm` and `agent_configuration` differ, with algorithm as the sole
nested configuration difference. Compared with the
[linear CVaR/seed-29 chain](colab_budget_chain_6_20261122.md), exactly the union
of these representation and algorithm fields differs.

All 500 training episode/seed/fingerprint/manifest records and all 240
budget/profile/validation seed/fingerprint records match across the four
seed-29 arms. Initial online and target tensors, optimizer, empty replay
and sampling RNG, exploration state and seed are exactly equal between the
actual Colab checkpoints. Configurations differ only by algorithm when
applicable; the encoder signatures match within each representation.
Their shared initial online fingerprint is:

```text
c08cba4ec8a01a8dbe8813038574c25121cb82388b1533d26f32d1411d7f754a
```

Against the [log-SNR neutral seed-11 control](colab_budget_chain_4_20261122.md),
only `agent_seed` differs in the full contract, and exogenous input records
match. Initial weights are not required to match across learner seeds.
Matched inputs and same-seed initialization do not imply matched later
queues, actions, training clocks or outcomes.

Producer preflight, export and session runtime identities agree with all
four seed-29 arms and the recorded seed-11 identity. Runtime fingerprint:
`f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`.
The producer used Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6,
one CPU thread and deterministic execution. The auditor used Windows AMD64,
CPython 3.12.14, Torch 2.14.1+cpu and NumPy 2.5.4. No fresh Windows reference
comparison, cross-platform exact-continuation claim or runtime-guard
relaxation is made.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103623 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

Update count follows `8930 - 32 + 1 = 8899`. Update indices are contiguous;
target refresh flags occur every 50 updates through 8850. Every checkpoint
passed restricted `weights_only=True` loading and the existing core validator.
Actual online fingerprints exactly match journal metadata. Configuration,
encoder schema 4, seed 29, scope and session metadata agree. Network and Adam
tensors are finite; Adam steps/shapes, full replay values/cursors, encoder
schemas and restored RNG states passed checks. Final replay size is 2048,
cursor 738 and Adam steps 8899. Target flags and saved endpoints were checked;
intermediate per-update target tensors are absent.

Final checkpoint: `steps/000052/agent.pt`. Actual online fingerprint:

```text
fa10b23373c42c22a122d3be40ec15096ee14f3893d2b44b8de6c9f68228378e
```

All 740 reconstructed static scenario manifests match recorded history:
500 training keys plus 80 fixed profile/validation-case keys revisited at
three correlated checkpoints. Task ownership/order, epsilon schedule,
recorded mean-cost preferences and minimum-cost action selection passed
checks. Recorded validation has epsilon zero, no updates and unchanged
training clocks. Full before/after learner/RNG equality is a producer guard;
the audit verifies the persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 198 | 252 |
| Validation over all budgets | 240 | 7647 | 7647 | 259 | 325 |

All 16577 offered task instances eventually completed. Deadline misses and
fallback are distinct from incomplete tasks. Validation totals repeat the
fixed cohort and are accounting diagnostics, not independent samples or a
partial-grid performance comparison. Prefix/closed and discounted costs
reconcile with per-user totals and episode metrics. Maximum absolute cost
conservation residual is `1.4210854715202004e-14`
with `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are absent,
so the physical objective components cannot be independently recomputed
from aggregate metrics. No simulator episode, optimizer update, policy
evaluation, public resume or held-out trajectory was executed by the audit.
Original source and session bytes were preserved.

## Archive evidence and next step

Received file: `colab-budget-log_snr_v4-qr_dqn-seed-29-09785cfa85f84737a6aa12ff845d540f.zip`.
Its 58,506,695 compressed bytes expand to
205,010,861 bytes across 161 members: 160 session files
plus the export summary. Safe paths, unique names, regular types, CRCs,
artifact hashes and the complete journal chain passed checks. Original
download remains untouched. ZIP SHA256:

```text
2ac275aca790263a3c484454fdee6819130d086e8e4f18b8bbe38def23041c0b
```

Exclusive extraction and audit artifacts remain under the ignored directory
`reports/colab-budget-chain-8-5d8845b-20261128/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `6ec43f0c8f24adb8335a099f5bc6800995e3ef937377e01b732e62a4d2b6b5ae` |
| identity_pairing_audit.json | `f965f41c78090edd073161e664369d4b9c5473ca1e0bd7b69df22b2d68ef20eb` |
| history_audit.json | `ef0c9efa25aea91bb9f5d917f4f4a3747e16162094242d35edd12dbee9aa8c08` |
| checkpoint_audit.json | `c821d178cc4e2108d73afe37dcabaa0363978228792d0295495d857f648072a3` |
| study_progress.json | `31236796b20856372fdfeac4ac85e497449a4387f3fa15d01a5b94839da4b51f` |

Combined progress retains eight verified and four missing chains under
common source and producer runtime identity, with original archives separate.
Performance curves, representation comparisons and fixed-reference summaries
remain null. No best-seed/checkpoint selection, representation adoption,
convergence or greedy-improvement claim follows from these integrity checks.

Only documentation and ignored audit code/results changed. Both READMEs
reflect this milestone. The earlier 573-test preparation suite was not rerun
for this data/documentation audit; see
[preparation verification](learning_budget_preparation_20261121.md).
Commit and push this milestone with Amir's authorization before advancing to
`log_snr_v4 / qr_dqn_cvar / seed 47` under the same scientific source,
runtime and protocol. Main training remains a later, explicitly announced
Colab milestone after calibration and a frozen paper protocol. See the
[fixed study instructions](learning_budget_development.md).
