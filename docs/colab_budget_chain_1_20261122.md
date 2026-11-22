# First Colab learning budget chain verification 2026 11 22

The received export completes the first chain in the declared development study:
`log_snr_v4 / qr_dqn_cvar / seed 11`, with 500 training episodes and 240 frozen
validation executions at the fixed 20/100/500-episode boundaries. Its archive,
immutable journal, all 51 checkpoints, recorded episode history, source and
complete scientific contract passed independent verification. One of the 12
declared chains is complete; the study remains incomplete and main training
and held-out evaluation remain unstarted.

## Scientific contract

The run uses independent User Vehicle decisions, individual returns, ordinary
learned-epsilon collection, QR-DQN + CVaR with alpha .9 and risk weight .5, and
the explicit log-SNR schema-4 candidate. MEC supplies information only. The
objective remains `z=.60*T/D+.25*E_mobile/E_budget+.15*M`, with plain EDF,
indivisible offloading and the existing channel/query/cache/recovery models.
No teacher, warmup, centralized destination controller or objective change
was introduced. The legacy schema-3 encoder remains the default elsewhere.

The full expanded learning/scenario plans, encoder signature, agent settings,
training and validation seed sequences, return scope and execution controls
match the committed preparation at
`5d8845bb0605def86541fd172a961be5211a7ba0`. The source hash matches the trusted
local checkout. The archive's declared commit is provenance information;
Python-source and full scientific-contract checks establish the identities
used by this audit.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `06928baf7e3aaaac64db948ba632aa56d73b8e88f1f89457cc48393312361e19` |
| Final journal commit | `276f39750bf417235cb551b114420f5a0190aeddd42c768ed6a524774cfa60c1` |

The archived preflight reports `ok=true` with no missing or incompatible
libraries. Its source/model contract and exact runtime identity agree with
the export and session contract. The producer used CPython 3.13.18 on Linux
x86_64, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread and the runner's
deterministic-algorithm setting. Its runtime identity SHA is
`f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`.

The local auditor used CPython 3.12.14 on Windows AMD64, Torch 2.14.1+cpu and
NumPy 2.5.4. Read-only inspection across those runtimes does not establish
bitwise equality of independently repeated training. No experiment resume
was attempted and the exact-runtime continuation guard was retained.

## Completed execution

The final journal and exported summary agree on the following counters:

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103492 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Saved checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The fixed warmup/update rule gives `8930 - 32 + 1 = 8899` updates. Each
evaluation point contains the same 20 validation cases in each of the four
declared profiles. Thus 240 validation executions revisit 80 fixed
profile/case pairs at three checkpoints; they are not 240 independent
validation scenarios. Intermediate checkpoints remain correlated observations
of one continuing model and cannot select an early stopping point.

The final checkpoint is `steps/000052/agent.pt`; the final online model SHA is:

```text
81ec1ced8d1aad532bbc492616ba4fa085d9f6001ed45db14116eb40680951f5
```

## Checkpoints and recorded history

All 51 checkpoints loaded with restricted `weights_only=True` deserialization
and the existing core checkpoint validator. Actual online tensor fingerprints
match every saved journal fingerprint. Configuration, encoder schema 4,
individual-return scope, seed, source/session metadata, replay samples and
counters, optimizer shapes/steps and restorable RNG states passed checks.
Online, target and optimizer tensors are finite. The final replay holds 2048
transitions with cursor 738, and all final Adam state steps equal 8899.

All 177 recorded target refreshes occur at the declared 50-update interval,
through update 8850. Saved checkpoint endpoints agree with target equality
where synchronization coincides with a boundary. Individual-update target
tensors are not archived; the audit checks the trace and saved endpoints,
without independently replaying the learning updates.

Reconstructed scenario manifests, seeds, fingerprints and order match all
500 training and 240 validation records. These 740 reconstructions cover 580
distinct training/profile/validation keys; no environment episode was run.
The same validation cohort is paired across all three budgets. Recorded
validation uses epsilon zero, no updates, no teacher and an unchanged training
clock. Recorded actor ownership/order, the training epsilon schedule and the
CVaR preference formula/minimum-cost selection also passed checks. The full
before/after learner/RNG equality check is a producer guard; the archive audit
verifies its persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 156 | 221 |
| Validation over all three budgets | 240 | 7647 | 7647 | 154 | 161 |

All 16577 offered task instances completed. Deadline misses and fallback
remain separate outcomes from eventual completion. Validation task totals
revisit the fixed cohort across correlated checkpoints and are not counts
of independent samples. These are drainage/accounting diagnostics, not
partial-grid cost or tail-performance comparisons.

Per-user prefix plus closed costs reconcile with total costs, and their sum
reconciles with episode metrics. The maximum absolute conservation residual
is `1.4210854715202004e-14`; audit comparisons used `rel_tol=1e-10` and
`abs_tol=1e-10`. The export lacks
raw final per-task ledgers, so the three physical objective components cannot
be independently recomputed from the aggregate metrics. The audit performed
no simulation, optimizer update, frozen-policy evaluation or held-out access;
source-session bytes remained unchanged.

## Archive evidence

The received file is
`colab-budget-log_snr_v4-qr_dqn_cvar-seed-11-53d0b523dd0941569f019b5c09524add.zip`,
59,304,685 bytes, with SHA256:

```text
aa74b0b7f9a3b847fa25c4b55b3304d6e450a4cd76e9d9dca0e584bfc121b65a
```

Its 161 members expand to 205,500,593 bytes: 160 session files and one export
summary. Safe member paths, regular file types, unique names, CRCs, the full
SHA256 artifact chain and the exported final commit passed verification.
The session has one initialization, 50 training chunks and three validation
entries. The original download was preserved; files were exclusively
extracted under the ignored directory
`reports/colab-budget-chain-1-5d8845b-20261122/received/`.

The independent audit records remain outside Git in
`reports/colab-budget-chain-1-5d8845b-20261122/`. The identity checks are in
`identity_audit.json`, and `study_progress.json` records one complete and 11
missing chains. The full-grid collector emits `status=incomplete` and null
performance curves, representation comparisons and fixed-reference summaries.
Shared fixed-policy references were not included in this export.

| Audit artifact | SHA256 |
| --- | --- |
| identity_audit.json | `43ce49967c529b75ea473678cd563758ac3bfb89ffd4c65b26f9b7f22a9f73d5` |
| checkpoint_audit.json | `f04de9cf826d2eabb892606621f2dcf407000ba02c9f200288e1fa2957409808` |
| history_audit.json | `447eb9b2ba24092553962f5f311d187ec646671cdc2c43ada604f81e5abdc440` |
| study_progress.json | `0c6e96a4bc7369333a7116e70420f2181a44f62758c0fcd41c5aea3285afb501` |

## Study boundary and next action

The full predeclared grid remains both SNR representations x QR/QR-CVaR x
seeds 11/29/47. Completion of this chain does not adopt the log representation,
choose a seed/checkpoint, or establish a gain over greedy. The remaining 11
chains and unchanged shared references must retain the same fixed cohort,
budgets, source and producer runtime before complete-grid comparisons.

This milestone records received-data verification. It changes no scientific
source, configuration or protocol. The 573-test suite belongs to the earlier
[preparation verification](learning_budget_preparation_20261121.md); it was
not rerun for these documentation/data checks. Main training, physical
calibration and the held-out paper protocol remain separate later work.

Commit and push this verified milestone record with Amir's authorization
before advancing. Continue the remaining development chains using the same
pinned source; do not extend or overwrite the completed chain. See
[the fixed study instructions](learning_budget_development.md).
