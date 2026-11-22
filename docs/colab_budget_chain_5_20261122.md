# Fifth Colab learning budget chain verification 2026 11 22

The received `log_snr_v4 / qr_dqn_cvar / seed 29` chain passed archive,
scientific-contract, recorded-history and all-checkpoint verification. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500-episode boundaries. Five of twelve declared development chains
are verified; seven chains and the shared fixed references remain outstanding.
Performance curves and comparisons remain withheld until the complete grid
is available. Main training and held-out evaluation remain unstarted.

## Scientific identity and learner-seed separation

The chain uses independent User Vehicle decisions, individual returns,
ordinary learned-epsilon collection, QR-DQN + CVaR with alpha .9 and effective
risk weight .5, and explicit log-SNR encoder schema 4. Cost remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, with plain EDF and indivisible
offloading. No physical model, objective, budget or source was retuned.

The full contract matches trusted Python source and expanded plans from
scientific commit `5d8845bb0605def86541fd172a961be5211a7ba0`. Its declared
commit label is provenance information; source and full-contract checks
establish the identities used by this audit.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `63d6d4c41bd284f6042053be854b1daeb7f122f7cd4f70b317e0c29662e9d461` |
| Final journal commit | `bf55f41492b53cf93dec645fcfc61dfb99201bb57c31476f5f78ba3a8b21be15` |

Against the [same method at seed 11](colab_budget_chain_1_20261122.md), only
`agent_seed` differs in the full contract. All 500 training seed/fingerprint/
manifest records and 240 budget/profile/validation seed/fingerprint records
match. Agent configuration and encoder agree. Initial online/target weights
and replay RNG state differ between seeds 11 and 29, while initial replay
items and exploration dictionaries are empty and optimizer state is unchanged.
This verifies explicit learner-seed separation on shared exogenous inputs;
it does not establish statistical independence of outcomes or scenarios.

The producer runtime matches prior chains exactly: Linux x86_64, CPython
3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread and the deterministic
execution contract. Archived preflight, export and session identities agree.
The local auditor used Windows AMD64, CPython 3.12.14, Torch 2.14.1+cpu and
NumPy 2.5.4. No cross-platform experiment resume or runtime-guard relaxation
was attempted.

## Auxiliary initialization comparison across runtimes

An additional local seed-29 initialization was constructed without training.
The first audit attempt incorrectly required it to be bitwise equal to the
Colab initial checkpoint. That auxiliary assertion failed. The failed script
was preserved, and the comparison was diagnosed rather than changing source,
stored checkpoints, seed, training budget or continuation guards.

| Seed | Colab initial online SHA | Local Windows initial online SHA |
| --- | --- | --- |
| 11 | `db55bca0f4312f30b5a27bec92c53123c7a72d4a4b3e48c1e4fda59c23ba7a5f` | `e9be1f58fec2b81d6e98c54293e385ce048afa73d92f4de78b28fb33b3c6fe93` |
| 29 | `c08cba4ec8a01a8dbe8813038574c25121cb82388b1533d26f32d1411d7f754a` | `644cd07bfdfb86f14cfa12897dc64c1f38460b20a1b565634a9fdc402953280e` |

For both seeds, differences are limited to first-layer weights/biases, with
maximum absolute difference `7.450580596923828e-09`. All later-layer tensors
are exactly equal. Seed-29 local optimizer, replay values/RNG and empty
exploration state match its initial checkpoint. The separate diagnostic
records numerical closeness using `rtol=1e-6`, `atol=1e-8`; this is an auxiliary
observation, not an archive-validity or resume tolerance. The pattern is
consistent with cross-build numerical differences, but the lower-level cause
was not established.

Exact archive tensor fingerprints and exact producer-runtime guards remain
unchanged. All previous producer-to-producer initial equality checks remain
valid; this discrepancy concerns a newly generated Windows reference. Neither
this diagnostic nor successful read-only loading establishes bitwise repeated
training agreement across runtimes. The actual Colab initial/final hashes
remain the identities of the received model.

## Checkpoints and recorded history

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103499 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update rule gives `8930 - 32 + 1 = 8899`. All update indices are contiguous
and target refresh flags occur every 50 updates through 8850. Every checkpoint
passed restricted `weights_only=True` loading and the existing core validator.
Actual online tensor fingerprints exactly match journal metadata, including
the actual Colab initial hash above. Configuration, encoder schema 4, seed 29,
scope and source/session metadata agree. Online, target and Adam tensors are
finite. Optimizer shapes/steps, full replay values/cursors and restored RNG
states passed checks. Final replay size is 2048, cursor 738 and Adam steps
8899. Saved target synchronization endpoints were checked; individual-update
intermediate target tensors are not archived.

The final checkpoint is `steps/000052/agent.pt`; its actual online model SHA is:

```text
c92e3d8a9646a608501a6fc4f33e03edff0b740ff90fda732417ce01798ed742
```

All 740 reconstructed episode manifests match the history: 500 training keys
and 80 fixed profile/validation-case keys revisited at three correlated
checkpoints. Recorded task ownership/order, epsilon schedules and CVaR
preference/minimum-cost selection passed checks. Validation has epsilon zero,
no teacher/updates and an unchanged training clock. Full before/after
learner/RNG equality is a producer guard; this audit checks its persisted
frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 165 | 254 |
| Validation over all three budgets | 240 | 7647 | 7647 | 221 | 264 |

All 16577 offered task instances completed. Deadline misses and fallback are
separate outcomes from eventual completion. Repeated validation counts are
execution/accounting diagnostics, not independent samples or partial-grid
performance comparisons. Prefix/closed and discounted costs reconcile with
user totals and episode metrics. Maximum absolute conservation residual is
`1.4210854715202004e-14`, using `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final
per-task ledgers are absent for independent physical objective-component
recomputation. The audit initialized reference models but executed no optimizer
update, environment episode, policy evaluation or held-out trajectory.
Original session bytes were preserved.

## Archive evidence and study status

The received file is
`colab-budget-log_snr_v4-qr_dqn_cvar-seed-29-ba09e05887a54bec8a869385c78061c5.zip`,
59,025,602 bytes, expanding to 205,266,416 bytes. Its 161 members contain 160
session files and an export summary. Safe paths, regular types, unique names,
CRCs, all artifact hashes and the final journal commit passed checks. Its SHA is:

```text
2b2def013b7459a79aaef5bb50432280a9a611593d879ba1970e0e24acf449ea
```

The original download remains untouched. Exclusive extraction, the preserved
failed auxiliary script, initialization diagnostic and audit results are under
the ignored directory `reports/colab-budget-chain-5-5d8845b-20261122/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `867dada550f1de64f7110f2ff2f934453a7fbeb7f6af471f683502e96d6e261f` |
| identity_seed_audit.json | `bf7af2747087e86bc33dfc662ed0e958ea0998f0f249832b059714ad667d66c5` |
| history_audit.json | `a4be1b92344db7872e5f154c76121cbb6df399d15b43a594ca1f747ea0ff4d18` |
| checkpoint_audit.json | `eae1533debe7bed4ad9ed7195572d521694d5ae63679eeca775a8974ac8698be` |
| initialization_diagnostic.json | `055d5ece0432c3f86cffb1dc33c6e8c13880f5f1faba3750e1907207292c8fca` |
| study_progress.json | `8a19ea980310e1a8705bcb89e0fc2f6dce75d72fee74371d283097b9169848ef` |
| audit_identity_seed_exact_reference_attempt.py | `73c084249233bea38f59430935dd33c5a7e614c43b5ff283ea6ba195c8755671` |

Each archive was checked separately against the full plan before progress was
merged by representation/algorithm/seed under common source and producer
runtime identity. The combined record has five verified and seven missing
chains; performance curves, representation comparisons and fixed-reference
summaries remain null. Shared fixed references are outstanding. No best-seed,
checkpoint, representation-adoption, convergence or greedy-improvement claim
follows from these checks.

Only documentation and ignored audit code/results changed. The 573-test suite
belongs to the earlier [preparation verification](learning_budget_preparation_20261121.md);
it was not rerun for this received-data audit. Keep the scientific source,
fixed budgets/cohort and exact producer runtime identity for remaining chains.
Commit and push the record and updated READMEs with Amir's authorization before
advancing. Main training, calibration and held-out paper evaluation remain
later milestones. See [the fixed study instructions](learning_budget_development.md).
