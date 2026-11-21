# Colab development execution 2026 11 21

The received Colab export passed independent archive, checkpoint and episode
audits. It contains the small development plan: one training episode, followed
by an explicit resume for three more episodes and eight frozen validation
cases. All 320 offered tasks completed across these 12 physical executions.
Main training and held-out test evaluation remain unstarted.

## Source and runtime

The prepared archive came from commit
`510b4072c1cca1ebf716266ddc2796b3fd8517cf`. The reported source SHA matches the
trusted local checkout:

```text
2bf447e4695f19aa4fb25cdee8eae15ecd6185558f65a24b966e122f38fe3b3f
```

The archived preflight has `ok=true`, `ready_for_learning=true`, and no missing
or incompatible libraries. It matches the initialization report and the source/
runtime fields in the session contract. Git commit/dirty metadata are null
because the source was extracted from ZIP without a `.git` directory.

| Runtime field | Reported value |
| --- | --- |
| Python | CPython 3.13.18, GCC 13.3.0 |
| OS and architecture | Linux, x86_64 |
| Torch | 2.14.1+cpu |
| NumPy | 2.4.6 |
| Device and threads | CPU, one Torch thread |
| Deterministic algorithms | enabled by the runner contract |

The reported exact runtime identity SHA is:

```text
f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463
```

This is a new Linux session. Its model fingerprints are not required to equal
those in the earlier Windows check, whose Python/NumPy/platform also differ.
An exact continuation must retain this session's own source, runtime and
scientific contract.

## Verified execution

The selected method remains QR-DQN + CVaR, alpha 0.9 and risk weight 0.5,
seed 11, individual returns and standard learned-epsilon collection. The
existing learning pilot v3 and scenario plan v2 were retained. Each User
Vehicle decides independently; MEC supplies information. No greedy teacher,
objective change or centralized destination controller was introduced.

| Invocation | Status | Training episodes | Validation episodes | Physical training slots | Seen transitions | Gradient updates |
| --- | --- | --- | --- | --- | --- | --- |
| Fresh, cap 1 | incomplete | 1 / 4 | 0 / 8 | 200 | 20 | 0 |
| Resume, cap 3 | complete | 4 / 4 | 8 / 8 | 840 | 75 | 44 |

With learning starting at 32 transitions, `75 - 32 + 1 = 44` is the expected
update count. Zero updates at 20 transitions in the first invocation is
therefore expected. Both summaries explicitly report
`main_training_started=false` and `test_split_accessed=false`.

The first checkpoint was `steps/000001/agent.pt`, with online model SHA:

```text
db55bca0f4312f30b5a27bec92c53123c7a72d4a4b3e48c1e4fda59c23ba7a5f
```

The final checkpoint was `steps/000004/agent.pt`, with online model SHA:

```text
8aea526dae68200665f9415aaa326f6efa8778e6a7d995967f20bd5935740020
```

All five checkpoints loaded using restricted `weights_only=True` deserialization
and the core checkpoint validator. Their schema, encoder, configuration, return
scope, source/session identity, replay size and optimizer counters match the
journal. Online model hashes recomputed from actual tensors match the saved
fingerprints. Network and optimizer tensors are finite. Loading for inspection
did not invoke the experiment runner or bypass its exact-runtime resume guard.

| Saved boundary | Seen transitions | Optimizer updates | Physical training slots |
| --- | --- | --- | --- |
| Initialization | 0 | 0 | 0 |
| Episode 1 | 20 | 0 | 200 |
| Episode 2 | 35 | 4 | 414 |
| Episode 3 | 57 | 26 | 619 |
| Episode 4 | 75 | 44 | 840 |

All declared train/validation seed keys, scene fingerprints and training scene
manifests match regenerated scenarios. No environment episode, new training or
held-out trajectory was run during verification. Per-user prefix plus closed
costs reconcile with reported total costs, and their sum reconciles with global
metrics within `1e-12`. The exported reports lack raw final per-task outcomes,
so this audit does not independently reconstruct each task's three-term cost.

Every validation report has epsilon zero, no updates and no teacher actions.
Recorded choices match the per-user masked risk-score argmin and tie rule;
float32 risk diagnostics reconcile within `1e-6`. The exported full summary's
validation records exactly match the committed final validation report.

## Descriptive validation results

Each row contains two declared validation cases for one fixed final model.
Mean cost is the arithmetic mean of full-drain episode total costs. Workload
share is the unweighted mean of the two episodes' completed CPU-work fractions;
it is not the fraction of tasks or a pooled workload estimate.

| Profile | Mean episode cost | Completed tasks | Deadline misses | Mean SV workload share |
| --- | --- | --- | --- | --- |
| nominal | 3.819199 | 39 / 39 | 0 | 72.65% |
| high_load | 14.753659 | 128 / 128 | 0 | 65.45% |
| fast_servers | 3.783145 | 39 / 39 | 0 | 67.88% |
| harsh_channel | 3.872536 | 39 / 39 | 1 | 75.66% |

All 245 validation tasks completed; one missed its deadline. Training completed
all 75 tasks with two deadline misses. Completion and on-time service remain
distinct metrics. These small descriptive results do not establish superiority
over greedy, calibrated tail quality or performance in a broader traffic model.

## Archive evidence

The received values are preserved in the ignored local artifact
`reports/colab-user-execution-510b407/user_reported_evidence.json`. Its SHA is:

```text
836f09e7eff35071c06d4cd392922e4e57546a9cddce33272595778527a6e3b1
```

The received file is `colab-development-772ce9d3770349309c9a7262af8426ab.zip`,
1,815,446 bytes. Its 20 members include the contract, six committed journal
generations, five checkpoints and two metadata files. Safe member paths,
regular file types, CRCs, artifact SHA256 values and the complete commit chain
passed validation. The archive was copied and extracted exclusively under
`reports/colab-user-execution-510b407/import-772ce9d3770349309c9a7262af8426ab/`.
The original download and historical reports were preserved.

```text
Archive SHA256: 605063dfc8290bb56f2dbee302cc73e02b61c23a30b4f3d356437e96d93aaca9
Session contract: 95a9b9fdaaa756468f796f8cbc7af86ec4e3c40f75eca38a7555b79e6713ea87
Final journal commit: 2df3a941bfe15cbad6f1b8b1ca5e58854056feaf668c23410bfbe5d7ce937196
```

Audit artifacts remain ignored by Git in `reports/colab-user-execution-510b407/`:

| Audit file | SHA256 |
| --- | --- |
| archive_audit.json | `f55c0d79547fc170f11917f8c8b26942a0829858570d2d9231c621ee8166539e` |
| checkpoint_audit.json | `103fb1adbb3f3e7a4cbd8385a5aa79374ad717b0b9af4418cea496f38293c507` |
| episode_audit.json | `0a360238bd822eb8198c0b698dd995dee3281738c46c461b475c355a8e7ec278` |

`verification_summary.json` combines the three passed audits. The earlier
`user_reported_evidence.json` remains unchanged as the record received before
archive verification. No scientific source, configuration or protocol changed.

## Interpretation and next boundary

This run checks execution and checkpoint continuation within a live Colab
runtime. It does not test an actual runtime disconnection, remote Drive
durability, or equality with an uninterrupted Linux execution. Its 44 updates
remain below the target-refresh interval of 50; all target weights correctly
remain equal to their initialization. Logged losses are finite, ranging from
3518.8154 to 81565.3281. The largest logged gradient norm, 275263.15625, is the
norm before clipping at 10. These values warrant review in the subsequent
training-diagnostics study and do not establish learning stability/convergence.

Commit and push this verified milestone record before advancing to scenario
calibration and a declared learning-budget study.
Keep the original completion logs and runtime identity; do not extend this
small plan after completion or reinterpret it as main training.
