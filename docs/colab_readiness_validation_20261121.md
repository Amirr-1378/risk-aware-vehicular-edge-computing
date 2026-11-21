# Colab preparation validation, 2026-11-21

This is a local CPU infrastructure check. Neither Colab nor main training was
started. The method remains standard QR-DQN + CVaR with independent User Vehicle
decisions, individual returns by default, alpha 0.9 and risk weight 0.5. No greedy
teacher or warmup was enabled. Physical/learning formulas and existing protocol
files were not changed.

## Verification

The full optional-learning suite discovered 448 tests: **447 passed, 1 skipped**.
The skip is the journal symlink check because this Windows account
cannot create symlinks. The dependency-free runtime discovered the same 448:
320 passed and 128 explicitly skipped (127 optional-learning tests and the
same Windows symlink privilege restriction). `git diff --check` passed.

New tests cover metadata-only preflight, safe credential-free source ZIP
extraction, exact embedded notebook bootstrap, empty notebook outputs, default
execution gates, immutable artifact hashes, corrupt/missing/extra files,
commit-publication interruption and verified-prefix recovery. No Colab runtime,
Drive mount or external account was accessed by these tests.

Both individual and team scopes were checked against uninterrupted training:
split-and-resumed weights, target, optimizer, replay including sampling RNG,
exploration RNG, modes and counters match exactly within the same runtime.
Validation outputs and model hashes also match the existing return-scope
runner on the same fixtures. Teacher actions, contradictory episode phases/
channel seeds, altered runtime/source/protocol/budget/scope and inconsistent
checkpoint counters are rejected. Reopening a complete session performs no
episode and writes no artifact. Frozen validation, no test-trajectory access,
same-seed retry and preservation of interrupted source evidence are tested.

## Bounded default-plan integration

Artifacts were exclusively created under the ignored directory
`reports/colab-readiness-v1-20261121/`; no historical report was overwritten.
The initial report truthfully records parent commit
`071e4415118afa1529cc8fc25a49b4177e9ec900` and `git_dirty=true`, since this
implementation had not been committed at execution time. Its source SHA is:

```text
2bf447e4695f19aa4fb25cdee8eae15ecd6185558f65a24b966e122f38fe3b3f
```

The session contract SHA is:

```text
ee0a9669ca20a307b9bdf4a0a14ec38ea42846bab9a39e4a62843aa32d00b559
```

Runtime: CPython 3.12.14 (Windows AMD64), Torch 2.14.1+cpu, NumPy 2.5.4,
one Torch thread, deterministic algorithms. The preflight records the exact
Python compiler/build string as well. This records the local execution; it
does not prescribe these exact wheel builds for a **new** Colab session.

One default-plan model used `learning_pilot_v3.json`, `development_v2.json`,
algorithm `qr_dqn_cvar`, seed 11, individual returns and standard learned-epsilon
behavior. The fixed total plan was four nominal training episodes followed by
eight frozen validation episodes across the four declared profiles.

| Invocation | Status | Completed train | Completed validation | Training slots | Seen transitions | Gradient updates |
| --- | --- | --- | --- | --- | --- | --- |
| Fresh, cap 1 | incomplete | 1 / 4 | 0 / 8 | 200 | 20 | 0 |
| Explicit resume, cap 3 | complete | 4 / 4 | 8 / 8 | 840 | 75 | 44 |

All 75 offered training tasks became transitions; `75 - 32 + 1 = 44` updates
matches the unchanged warmup/update rule. Final replay size is 75. Training
paused without validation after the first episode, then resumed without
extending the total budget. The final checkpoint is
`session/steps/000004/agent.pt`; its online model SHA is:

```text
786fdae140a3360376be09949f94ef25878b2cb50113dbbeee1e9606ad5fb945
```

The journal has one initialization, four training generations and one frozen
validation generation. Every published report/checkpoint is hashed. Summaries
remain in `partial.json` and `complete.json`, with the preflight in
`preflight.json`. They explicitly mark main training and test-split access false.
These files are development artifacts, excluded from Git along with checkpoints.

Integrity fingerprints:

```text
complete.json SHA256: cc6dd3ff33e273bdb56529302994412cdb6f9d3a9272004b5895cb32bbad18bb
last journal commit: 5d70522774398c6d023b594ce17b594c9ceedff53776aa871b04c3cbf89620a1
```

## Interpretation and next boundary

This validates local mechanics and a notebook's offline structure. Actual
private-source upload, Linux learning-library imports, Colab execution, Drive
permissions/synchronization and recovery on real disconnections remain to be
checked in Colab. Filesystem hashes/rename are not a cloud durability guarantee.
There is no new comparison against greedy, convergence result, calibrated
physical protocol or paper-level performance claim.

Commit and push this milestone before using its commit-pinned archive in Colab.
Then perform a small explicitly enabled development check there. Before main
training, notify Amir and freeze the scientific plan separately. See
[`colab_development.md`](colab_development.md) for exact setup and recovery.
