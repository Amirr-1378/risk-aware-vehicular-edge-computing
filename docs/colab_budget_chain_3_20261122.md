# Third Colab learning budget chain verification 2026 11 22

The received `linear_v3 / qr_dqn / seed 11` control passed independent
archive, scientific-contract, recorded-history and all-checkpoint checks.
It completed 500 training episodes and 240 validation executions at the
predeclared 20/100/500-episode boundaries. Three of twelve development chains
are now verified; nine chains and shared fixed references remain outstanding.
Complete-grid performance comparisons remain withheld. Main training and
held-out evaluation have not started.

## Neutral control and paired inputs

This chain is the predeclared risk-neutral QR control: action and bootstrap
selection use mean cost, with effective risk weight zero. The common stored
risk parameter remains .5 in the shared template; the `qr_dqn` algorithm
selects its effective weight through the existing implementation. The primary
method remains QR-DQN + CVaR. No objective, physical model or shared protocol
parameter was retuned for this control.

The scalar objective remains `z=.60*T/D+.25*E_mobile/E_budget+.15*M`, with
independent User Vehicle choices, individual returns, plain EDF and indivisible
offloading. Linear SNR encoding uses schema 3. Ordinary learned-epsilon
collection, 51 quantiles, gamma 1, query/cache, mobility and recovery remain
as declared in the fixed development plan.

The full contract matches the trusted source and expanded plans from pinned
scientific commit `5d8845bb0605def86541fd172a961be5211a7ba0`. The export's commit
label is provenance information; source and complete-contract checks establish
the identities used here.

| Identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Session contract | `33b1da3ca6f665d62715223c858c37b85b16318ef20f5f754dea08dc179f7016` |
| Final journal commit | `0e7eb90cc8494183274f4e220e4b82f710aefcf12ff97aaafa6ae71a758b825b` |

Compared with the [matched linear CVaR chain](colab_budget_chain_2_20261122.md),
only the outer `algorithm` and `agent_configuration` contract fields differ;
the nested agent settings differ only in their algorithm identifier. All 500
training seeds/fingerprints/manifests and all 240 budget/profile/validation
seed/fingerprint records match. Initial online/target tensors, optimizer,
empty replay and RNG, exploration state, seed and encoder signature match.
The shared initial online SHA is
`db55bca0f4312f30b5a27bec92c53123c7a72d4a4b3e48c1e4fda59c23ba7a5f`.
Paired inputs and initial states allow subsequent actions, queues, clocks and
outcomes to differ.

The producer runtime matches previous chains: Linux x86_64, CPython 3.13.18,
Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread and the deterministic execution
contract. Preflight, export and session identities agree. The local audit used
Windows AMD64, CPython 3.12.14, Torch 2.14.1+cpu and NumPy 2.5.4. Read-only
inspection does not establish exact cross-platform training continuation;
no runtime guard was relaxed and no experiment resume was attempted.

## Checkpoints, history and accounting

| Quantity | Verified value |
| --- | ---: |
| Training episodes | 500 / 500 |
| Completed evaluation budgets | 20, 100, 500 |
| Validation executions | 240 / 240 |
| Physical training slots | 104032 |
| Seen transitions | 8930 |
| Optimizer updates | 8899 |
| Target refresh flags | 177 |
| Checkpoints including initialization | 51 |
| Committed journal entries | 54 |

The update rule gives `8930 - 32 + 1 = 8899`. All update indices are contiguous,
and target refresh flags occur at every 50th update through 8850. Every
checkpoint loaded with restricted `weights_only=True` and the existing core
validator. Actual online tensor fingerprints match all journal fingerprints.
Configuration, encoder schema 3, scope, seed, source/session metadata, full
replay values/cursors and restored RNG states passed checks. Online, target
and Adam tensors are finite, and optimizer shapes/steps agree. Final replay
size is 2048, cursor 738 and Adam steps 8899. Saved target synchronization
endpoints were checked; intermediate per-update target tensors are absent.

The final checkpoint is `steps/000052/agent.pt`; its actual online model SHA is:

```text
3ab874248df8cbd7e913f0cf30347709d1fc3c14decca830e6329c69a887dff9
```

All 740 recorded episode manifests match reconstruction: 500 training keys
and 80 fixed validation profile/case keys revisited at three correlated
checkpoints. Task ownership/order and epsilon schedules agree. Recorded
validation selects by mean cost, with effective risk weight zero, epsilon
zero, no updates/teacher and an unchanged training clock. Full before/after
validation learner/RNG equality is a producer guard; this audit checks the
persisted frozen-validation boundary.

| Recorded executions | Episodes | Offered tasks | Completed tasks | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 500 | 8930 | 8930 | 230 | 314 |
| Validation over all three budgets | 240 | 7647 | 7647 | 349 | 386 |

All 16577 offered task instances completed. Deadline misses and fallback are
separate recorded outcomes from eventual completion. Validation counts repeat
the fixed cohort; these are execution/accounting diagnostics, not independent
sample counts or partial-grid performance comparisons. Per-user prefix/closed
and discounted costs reconcile with totals and episode metrics. Maximum
absolute conservation residual is `2.1316282072803006e-14`, using audit
`rel_tol=1e-10` and `abs_tol=1e-10`. Raw final per-task ledgers are absent, so
the physical objective components cannot be independently recomputed from
aggregate metrics. The audit executed no environment episode, optimizer
update, policy evaluation or held-out trajectory. Session bytes are unchanged.

## Reported browser disconnection

Amir reported that the browser connection was lost during execution and that
the complete output appeared on reconnect without rerunning the training cell.
The export records this as a user-reported browser event, with no verified
runtime interruption or checkpoint-based recovery. The complete journal has
one initialization, 50 training chunks and three validation entries, with no
missing committed step or inconsistent artifact. This establishes preserved
history integrity. The journal format does not log browser connections,
runtime-process lifecycle or explicit resume-invocation events, so it cannot
establish whether a runtime stopped or recovered during that observation.

## Archive evidence and study status

The received file is
`colab-budget-linear_v3-qr_dqn-seed-11-603978f12f8e4e0c98b048e106325dbb.zip`,
57,622,997 bytes, expanding to 205,023,550 bytes. Its 161 members contain 160
session files and an export summary. Safe paths, regular types, unique names,
CRCs, every artifact hash and the final journal commit passed checks. Its SHA is:

```text
5d79dde500805cc34612ec932ec3d7a424f098128f7674ec2942976b3d835af4
```

The original download remains untouched. Exclusive extraction and audit
artifacts are retained under the ignored directory
`reports/colab-budget-chain-3-5d8845b-20261122/`.

| Audit artifact | SHA256 |
| --- | --- |
| archive_receipt.json | `550b51f2adbcd666f34ffd2f83f87dc80fb32e198268bf679f33427ea49235c5` |
| identity_pairing_audit.json | `2b2e32193e1e6043fdbdc50ac0cb4e6d84c1a0bc7518b51253dd9845236f5a56` |
| history_audit.json | `041ca5a7814c6fe989fa62d0a7d78c0be38bae4efa0c662008f6870adb971c0a` |
| checkpoint_audit.json | `0c1450cbe29be34c2a6cf88fef40aaaf4cdf520c49bc67e8bd1f0e881c641463` |
| study_progress.json | `4c8a69840d1375023c0881e8dfd7e7ca76f6ac1672233c1a03e833458c891f84` |

Each archive was checked separately against the full plan. Verified progress
was merged by representation/algorithm/seed under the same source and producer
runtime. The combined record contains three verified and nine missing chains;
performance curves, representation comparisons and fixed-reference summaries
remain null. Shared fixed references are still outstanding. No best seed,
checkpoint, representation adoption, convergence or greedy-improvement claim
follows from these checks.

Only documentation and ignored audit artifacts changed. The 573-test suite
belongs to the earlier [preparation verification](learning_budget_preparation_20261121.md);
it was not rerun for this received-data audit. Keep the scientific source,
fixed budgets/cohort and runtime for remaining chains. Commit and push this
verification record and both updated READMEs with Amir's authorization before
advancing. Main training, calibration and held-out paper evaluation remain
later milestones. See [the fixed study instructions](learning_budget_development.md).
