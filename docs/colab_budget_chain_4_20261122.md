# Fourth Colab learning budget chain verification 2026 11 22

The received `log_snr_v4 / qr_dqn / seed 11` control passed independent
archive, scientific-contract, recorded-history and all-checkpoint checks.
It completed 500 training episodes and 240 validation executions at the fixed
20/100/500-episode boundaries. All four representation/algorithm combinations
for seed 11 are now verified: four of twelve declared development chains.
Eight chains for seeds 29/47 and the shared fixed references remain outstanding.
Complete-grid performance comparisons remain withheld. Main training and
held-out evaluation remain unstarted.

## Scientific identity and fresh runtime

This is the predeclared mean-cost QR control, with effective risk weight zero,
and the explicit log-SNR encoder, schema 4. QR-DQN + CVaR remains the primary
method. No objective, source, physical model or protocol parameter was retuned.
The scalar cost remains `z=.60*T/D+.25*E_mobile/E_budget+.15*M`, independent
User Vehicle decisions, individual returns, plain EDF and indivisible offloading.
Ordinary learned-epsilon collection, 51 quantiles, gamma 1 and the existing
channel/query/cache/mobility/recovery models remain unchanged.

The full contract matches trusted source and expanded plans from scientific
commit `5d8845bb0605def86541fd172a961be5211a7ba0`. The export's commit label is
provenance information; source and full-contract checks establish the
identities used here.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `6c646858a44ee0ce7baac967d3c8304a4e35dd6e364536a371c7cac67c8f63d3` |
| Final journal commit | `e884942e996fc24151c8728cc5341a455ae6788d39ad1915d39636ad04d29f80` |

Amir reported preparing a fresh runtime from the same source ZIP before this
chain. Archived preflight, export and session runtime identities agree, and
match previous chains exactly: Linux x86_64, CPython 3.13.18, Torch 2.14.1+cpu,
NumPy 2.4.6, one CPU thread and the deterministic execution contract. The runtime
identity SHA is `f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`.
This compares software/build/driver identity; it does not identify a VM instance
or demonstrate checkpoint-based recovery. This chain was a fresh model run.

Local inspection used Windows AMD64, CPython 3.12.14, Torch 2.14.1+cpu and
NumPy 2.5.4. No experiment resume or runtime-guard relaxation occurred. A
read-only audit across those environments does not establish equality of
independently repeated training or exact cross-platform continuation.

## Paired inputs and the seed-11 grid

Against the [linear QR control](colab_budget_chain_3_20261122.md), only `encoder`,
`representation_contract` and `representation_variant` differ in the full
contract. Agent configuration is identical. All 500 training seed/fingerprint/
manifest records and all 240 budget/profile/validation seed/fingerprint records
match. Initial online and target tensors, optimizer, empty replay and RNG,
exploration state, configuration and seed match under restricted inspection.

An additional four-model identity audit verified the complete seed-11 factor
grid: `linear_v3`/`log_snr_v4` x `qr_dqn`/`qr_dqn_cvar`. Nonfactor contract fields,
source, runtime and agent hyperparameters agree. Initial online/target tensors,
optimizer, replay/RNG and exploration states match across all four. The shared
initial online SHA is
`db55bca0f4312f30b5a27bec92c53123c7a72d4a4b3e48c1e4fda59c23ba7a5f`.
The three recorded pairing edges connect all four models with the same 500
training and 240 validation record identities. Subsequent actions, queues,
clocks and outcomes can differ. This is an identity/input audit without cost
curves or a partial-grid performance comparison.

## Checkpoints and recorded episodes

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 103344 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update rule gives `8930 - 32 + 1 = 8899`. All update indices and refresh
flags agree with the 50-update interval through update 8850. All 51 checkpoints
passed restricted `weights_only=True` loading and the core checkpoint validator.
Actual online tensor fingerprints match every journal fingerprint. Agent
configuration, seed, scope, encoder schema 4 and source/session metadata agree.
Online, target and Adam tensors are finite. Optimizer state shapes/steps, full
replay values/cursors and restored RNG states passed checks. Replay observations
use schema 4; replay/checkpoint containers remain version 2. Final replay size
is 2048, cursor 738 and Adam steps 8899.

The final checkpoint is `steps/000052/agent.pt`; its actual online model SHA is:

```text
cb4aa8920e81fa742ed450b9630a0e32bcbe6be83071f074336faf8af9254aab
```

Saved target synchronization endpoints were checked; individual-update target
tensors are not archived. All 740 reconstructed episode manifests match the
history: 500 training keys and 80 fixed profile/validation-case keys revisited
at three correlated checkpoints. Task ownership/order, epsilon schedules and
mean-cost selection with effective risk weight zero passed checks. Recorded
validation uses epsilon zero, no teacher/updates and an unchanged training
clock. Full before/after learner/RNG equality is a producer guard; the audit
checks its persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 182 | 244 |
| Validation over all three budgets | 240 | 7647 | 7647 | 188 | 187 |

All 16577 offered task instances completed. Deadline misses and fallback are
separate outcomes from eventual completion. Validation counts repeat the
fixed cohort; these are execution/accounting diagnostics, not independent
sample counts or performance comparisons. Prefix/closed and discounted costs
reconcile with user totals and episode metrics. Maximum absolute conservation
residual is `1.4210854715202004e-14`, using `rel_tol=1e-10` and `abs_tol=1e-10`.
The export lacks raw final per-task ledgers for independent physical objective
component recomputation. The audit executed no environment episode, optimizer
update, policy evaluation or held-out trajectory. Original session bytes were
preserved.

## Archive evidence and study status

The received file is
`colab-budget-log_snr_v4-qr_dqn-seed-11-9f2659f01eb243609d2537f9dc6228ff.zip`,
59,065,419 bytes, expanding to 205,399,198 bytes. Its 161 members contain 160
session files and the export summary. Safe paths, regular types, unique names,
CRCs, artifact hashes and the final journal commit passed checks. Its SHA is:

```text
b827eb7fb620449f46c51de17e0a1e19d4b4e8ac29b00d59b877cb08ad285b54
```

The original download remains untouched. Exclusive extraction and audit
artifacts are retained under the ignored directory
`reports/colab-budget-chain-4-5d8845b-20261122/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `574336463164a7fe225aca06782245036d63638e19335a302200f50a73b68398` |
| identity_pairing_audit.json | `f9542bd8d1856d293a52eccaac8efcba6117d502a168265403e150de31cd7155` |
| history_audit.json | `1f7b1b7b2e54071bc6b613096dd85d597e64eb15c37908cc883e63ea83f19d1e` |
| checkpoint_audit.json | `0c6d78393b42a2fa18822fb7c9c1455812e3ec22e89120a925aba44522e15320` |
| seed11_matrix_audit.json | `bf8572378e82ebf228a744768045868590edeed5ef8aee56ee48c050ba747f5c` |
| study_progress.json | `31ab0e954050eb1ac96a3f6599557ca48887259958c98d8960c49bd1d0a31ad8` |

Each archive was checked separately against the full plan before verified
progress was merged by representation/algorithm/seed under the common source
and runtime identity. The combined record contains four verified and eight
missing chains, with performance curves, representation comparisons and
fixed-reference summaries null. Shared fixed references remain outstanding.
No seed/checkpoint selection, representation adoption, convergence or greedy
improvement follows from these identity and integrity checks.

Only documentation and ignored audit artifacts changed. The 573-test suite
belongs to the earlier [preparation verification](learning_budget_preparation_20261121.md);
it was not rerun for this received-data audit. Keep the scientific source,
fixed budgets/cohort and exact runtime identity for remaining chains. Commit
and push the verification record and both updated READMEs with Amir's
authorization before advancing. Main training, calibration and held-out paper
evaluation remain later work. See [the fixed study instructions](learning_budget_development.md).
