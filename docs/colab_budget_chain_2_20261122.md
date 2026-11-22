# Second Colab learning budget chain verification 2026 11 22

The received `linear_v3 / qr_dqn_cvar / seed 11` chain passed independent
archive, protocol, recorded-history and all-checkpoint verification. It
completed 500 training episodes and 240 validation executions at the fixed
20/100/500-episode boundaries. Together with the
[first log-SNR chain](colab_budget_chain_1_20261122.md), two of the twelve
declared development chains are verified. Ten chains and the shared fixed
references remain outstanding; complete-grid performance results are withheld.
Main training and held-out evaluation remain unstarted.

## Scientific identity and paired inputs

This control uses the existing linear SNR encoder, schema 3, with independent
User Vehicle decisions, individual returns and standard learned-epsilon
collection. QR-DQN + CVaR retains alpha .9, risk weight .5 and the original
agent settings. The objective remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, plain EDF and indivisible offloading.
The physical, query/cache, mobility and recovery models are unchanged.

The full contract matches the predeclared plans and trusted Python source
from commit `5d8845bb0605def86541fd172a961be5211a7ba0`. The archive's declared
commit is provenance information; the source fingerprint and expanded
scientific-contract checks establish the identities used by this audit.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `17a2735573a47cc9dc8234a5d6eda19687928d184808725d98d4af8a194b41a2` |
| Final journal commit | `2cee09e58adfdb97c982e4004c342a967709e51ec0480a5908709877da22c3d4` |

Across the two received contracts, only `encoder`, `representation_contract`
and `representation_variant` differ. All 500 training records match in seed,
scenario fingerprint and scenario manifest. All 240 validation records match
in budget, profile, episode ID, seed and scenario fingerprint. Initial online
and target tensors, optimizer state, empty replay and its RNG, exploration
state, agent configuration and seed are equal under restricted checkpoint
inspection. Their initial online model SHA is
`db55bca0f4312f30b5a27bec92c53123c7a72d4a4b3e48c1e4fda59c23ba7a5f`.
Subsequent decisions, queues, training clocks and outcomes can differ.

Producer runtime identity matches the first chain exactly: Linux x86_64,
CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread and the
deterministic-algorithm contract. Archived preflight, export and session
identities agree. The local audit used Windows AMD64, CPython 3.12.14,
Torch 2.14.1+cpu and NumPy 2.5.4. Read-only loading establishes the recorded
state checks below; it does not establish equality of independently repeated
training across platforms. No resume or runtime-guard relaxation occurred.

## Verified execution and checkpoints

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 104110 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Saved checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update count follows `8930 - 32 + 1 = 8899`. Every update index and
target-refresh flag agrees with the 50-update interval, through update 8850.
All 51 checkpoints loaded with `weights_only=True` and the existing core
validator. Their actual online tensor fingerprints match journal metadata;
configuration, encoder schema 3, seed, scope and session metadata agree.
Online, target and Adam tensors are finite. Adam state shapes/steps, full
replay values, schema-3 samples, replay cursors and restored RNG states passed
checks without sampling or learning. The final replay has 2048 transitions,
cursor 738, and Adam steps 8899.

The final checkpoint is `steps/000052/agent.pt`; its actual online model SHA is:

```text
474f60a2da0d5e129150f91fd09207eca251913af26491427396c11b24681bc7
```

Target/online equality was checked at saved endpoints coinciding with target
synchronization. Per-update intermediate target tensors are not archived;
trace and endpoint checks do not independently reproduce every learning step.

## Recorded episodes and accounting

All 740 scenario-manifest reconstructions match the archived history: 500
training keys and 80 fixed profile/validation-case keys, with validation
revisited at three correlated checkpoints. Recorded task ownership, action
order, the training epsilon schedule and CVaR preference/minimum-cost selection
passed checks. Validation records have epsilon zero, no teacher or updates,
and unchanged training clocks. Full before/after validation learner/RNG
equality is a producer guard; the audit verifies the persisted boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 219 | 334 |
| Validation over all three budgets | 240 | 7647 | 7647 | 311 | 260 |

All 16577 offered task instances completed. Deadline misses and fallback are
separate recorded outcomes from eventual completion. These counts are audit
diagnostics; validation tasks repeat the fixed cohort and do not constitute
7647 independent validation samples or a partial-grid performance comparison.

Per-user prefix and closed costs conserve total costs, including discounted
accounting under gamma 1. Episode metrics reconcile with these totals; maximum
absolute cost-conservation residual is `2.1316282072803006e-14`, using audit
comparisons with `rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers
are absent, so the physical objective components cannot be independently
recomputed from aggregate metrics. No environment episode, optimizer update,
frozen-policy evaluation or test trajectory was executed by this audit.
Original session bytes were preserved.

## Archive and audit evidence

The received archive is
`colab-budget-linear_v3-qr_dqn_cvar-seed-11-616889e16a4d421d94ac167a67de7259.zip`,
56,502,653 bytes. Its 161 members expand to 205,336,594 bytes: 160 session
files and the export summary. Safe paths, regular types, unique member names,
CRCs, all artifact hashes and the final journal commit passed checks. Its SHA is:

```text
c1d11f512c9003930486dc3f6b66c55127c0efc5fa348eef6f6f7be1566f7216
```

The original download remains untouched. Exclusive extraction and read-only
audit artifacts are retained under the ignored directory
`reports/colab-budget-chain-2-5d8845b-20261122/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `15d20b411d87ce427f329e38c352153c5c2af7029e6c4ea9cc20e7fa7a94cbd9` |
| identity_pairing_audit.json | `f051cc158f55d3f345726478a06fa79d8c9d6ebaeb4da73cab3543abee1d3642` |
| history_audit.json | `fefdf0f037eb704b520f8e6c6ddb492ea64f8707a6ab33f617d3ba8c78f9291a` |
| checkpoint_audit.json | `9377ec7c4b7d43d507855ba9dd5ce1f1c1913260c35bb128a8a597b2bc40642d` |
| study_progress.json | `fa750ce64869b131eadfac9c6c2b1f7d5d003524ca285c1fb91161ad537ebb3a` |

The collector checked each archive separately against the full plan. After
both chains' checkpoint/history checks passed, their progress records were
merged by declared representation/algorithm/seed identity, with common source
and runtime verified. Archives were kept separate and unchanged. The combined
record contains two verified chains and ten missing chains, with performance
curves, representation comparisons and fixed-reference summaries null.

## Study boundary and next action

The complete twelve-chain study and unchanged shared references are still
required for performance comparisons. No best seed, checkpoint, risk weight,
budget extension, representation adoption, greedy improvement or convergence
claim follows from these checks. Keep the same pinned scientific source,
fixed cohort and producer runtime for remaining chains.

Only documentation and ignored audit artifacts changed. The 573-test suite
belongs to the earlier [preparation verification](learning_budget_preparation_20261121.md);
it was not rerun for this received-data audit. Main training, calibration and
held-out paper evaluation remain later work. Commit and push the verification
record and updated READMEs with Amir's authorization before advancing.
See [the fixed study instructions](learning_budget_development.md).
