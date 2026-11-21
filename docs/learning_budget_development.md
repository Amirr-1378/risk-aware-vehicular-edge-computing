# Fixed development learning-budget study

This study asks whether additional training changes the observed cost, tail
cost and Server Vehicle workload under the existing synthetic development
model. It prepares a larger, resumable experiment after the small SNR pilot;
it does not start main training, establish convergence, or select a new final
method. The default encoder remains the legacy schema 3. Normalized-log1p SNR
schema 4 remains an explicit candidate with a matched legacy control.

The complete declared study has not been executed during this preparation
milestone. Main training remains a separately announced Colab milestone after
physical calibration and agreement on a frozen paper protocol. The learning
budget, source and comparison rules must be fixed before starting this study;
validation results must not decide when to stop, extend a budget, drop a seed,
or select a best intermediate checkpoint.

## Scientific boundary

Each User Vehicle independently chooses one indivisible destination from one
MEC and the eligible Server Vehicles using its own received observation. MEC
supplies information only. The objective remains
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, with plain EDF, no manual task labels
and no separate deadline penalty. Deadline misses, tardiness, fallback and SV
workload remain reported outcomes. QR-CVaR retains alpha .9 and risk weight .5;
the risk-neutral QR control has effective risk weight zero.

The study uses individual returns and ordinary learned-epsilon behavior. No
team return, teacher, greedy warmup, imitation loss, reward shaping, central
controller or extra actor input is introduced. The actual channel, latency,
energy, mobility, query/cache and recovery models are unchanged. The existing
strong cached combined-cost greedy policy keeps its information, objective,
hardware assumptions and selection rule. It is not weakened or retuned in
response to neural results.

## Declared traces, budgets and controls

[`../experiments/configs/development_budget_v1.json`](../experiments/configs/development_budget_v1.json)
expands the existing development split capacities to 500 train, 20 validation
and 30 reserved test seed keys. It retains the master seed, physical profiles
and role-separated seed derivation. The first 20 training and first six
validation seed keys therefore match the earlier development plan; these are
not newly sampled replacements for difficult episodes. Test keys are reserved
metadata only: no held-out trajectory is constructed or evaluated here.

The training profile is nominal. Validation uses the same 20 independently
seeded cases in each known profile: nominal, high load, fast servers and harsh
channel. All models and budget points use that same fixed validation cohort.
Scenario seeds and fingerprints bind pairing; each execution starts from a
pristine environment. Different actions can still produce different queues,
query timings, episode lengths and realized exploration schedules.

[`../experiments/configs/learning_budget_v1.json`](../experiments/configs/learning_budget_v1.json)
declares the following grid before execution:

| Dimension | Fixed values |
| --- | --- |
| Representations | `linear_v3`, `log_snr_v4` |
| Algorithms | `qr_dqn`, `qr_dqn_cvar` |
| Agent seeds | 11, 29, 47 |
| Training budget points | 20, 100, 500 complete episodes |
| Validation per point and model | 20 cases x four profiles = 80 episodes |
| Return scope | Individual |
| Maximum checkpoint chunk | Ten complete training episodes |

There are **12 continuing model chains**. Each trains once to 500 episodes,
with mandatory frozen evaluations after episodes 20, 100 and 500. The budget
points are checkpoints of the same chain, not independent fresh training
runs. The optimizer, target network, replay and physical-slot epsilon clock
continue across those points. Checkpoint rows are consequently correlated.

The shared controls remain two hidden layers of 64 units, 51 quantiles,
learning rate .001, batch size 16, replay capacity 2048, learning starts at 32
transitions, target update every 50 updates, gamma per slot 1, and epsilon
1 to .05 over 1000 training slots. The SNR reference remains the fixed value
100 in both representations. No fitted normalization or new exploration
schedule is added as part of increasing the episode budget.

Five unchanged fixed policies run once on the shared validation cohort:
AlwaysMEC, AlwaysV2V, MinimumEstimatedLatency, UniformRandom and
MinimumEstimatedCombinedCost. They are not rerun for every model or budget.

| Execution type | Planned physical episodes |
| --- | ---: |
| Training: 12 chains x 500 | 6000 |
| Learned validation: 12 x three points x 80 | 2880 |
| Shared fixed references: five policies x 80 | 400 |
| Total | **9280** |

These counts describe the complete planned study, not completed work, wall
time or storage demand. The physical parameters remain explicitly synthetic
and uncalibrated. Increasing trace counts does not resolve their provenance
or turn this study into a paper benchmark. With 20 episodes per profile and
alpha .9, empirical CVaR uses only the upper two episode weights; tail
precision remains limited.

The old pilot used four validation cases per profile. Its recorded results
remain intact. Even at the new 20-training-episode point, this study evaluates
the newly fixed 20-case cohort; the old four-case summaries must not be
spliced into the new learning curve.

## Plan inspection and one-chain execution

Inspect counts without constructing a scenario, learner or output directory:

```text
python -m experiments.run_learning_budget_session --plan
```

The new runner is separate from the earlier episode-boundary development
runner. It calls the existing episode/agent implementation and advances one
explicitly selected chain. The following commands show the interface; the
full study is intended for Colab rather than an unattended local sweep:

```text
python -m experiments.runtime_preflight --require-learning
python -m experiments.run_learning_budget_session --variant log_snr_v4 --algorithm qr_dqn_cvar --agent-seed 11 --output-dir reports/NEW-budget-study/log_snr_v4/qr_dqn_cvar/seed-11 --max-additional-episodes 10
python -m experiments.run_learning_budget_session --variant log_snr_v4 --algorithm qr_dqn_cvar --agent-seed 11 --output-dir reports/NEW-budget-study/log_snr_v4/qr_dqn_cvar/seed-11 --resume --max-additional-episodes 10
```

The cap counts **training episodes only**. A call that reaches 20, 100 or 500
also evaluates all 80 frozen validation episodes before publishing that
milestone or continuing training. Such a call can take substantially longer
than an ordinary ten-training-episode chunk. A failed unpublished validation
is retried from that saved checkpoint before any further training; the fixed
cohort is not shortened to fit the invocation cap.

Every new chain needs its own fresh directory. Changing representation,
algorithm or seed requires choosing another declared chain, never resuming
the previous directory with different settings. Reopening a completed chain
verifies it and returns progress without further training or validation.
The full grid must retain both representations, both algorithms and all three
seeds; the initially selected log/QR-CVaR/seed-11 chain is only an execution
convenience, not adoption of that representation.

## Colab handoff and persistence

Use [`../notebooks/colab_learning_budget.ipynb`](../notebooks/colab_learning_budget.ipynb).
It contains no saved outputs and all action gates default to false. First
commit and push the verified preparation milestone with Amir's authorization.
Then obtain the full commit and Python-source SHA from that same clean trusted
checkout, download the commit-pinned private GitHub ZIP in the signed-in
browser, and fill both identities in the notebook. No GitHub credential is
placed in Colab code or outputs. The notebook reuses the offline archive
verifier and metadata preflight described in
[`colab_development.md`](colab_development.md).

Source setup and plan inspection do not execute the study. After reviewing
them, explicitly mount Drive and enable the selected-chain execution gate.
The notebook selects one chain manually and has no whole-grid loop or
main-training launcher. Its default study root is
`/content/drive/MyDrive/risk-aware-v2x/learning-budget-v1`, with paths
`<variant>/<algorithm>/seed-<seed>`. Keep one writer per chain. Do not rerun the
initial configuration while continuing: it clears readiness flags. Do not
extract source over an existing destination.

Each committed training chunk contains reports for complete episodes and
one full learner checkpoint. Chunk boundaries stop at the next budget point,
the invocation cap, or ten additional training episodes, whichever comes
first. Frozen validation is a separate immutable report with no learner
update or training-clock advance. Full state guards include optimizer,
target network, replay, RNGs and exploration state.

The immutable contract binds exact source/runtime, expanded protocols,
encoder/reference, algorithm, seed, return scope and training behavior.
Resume verifies the complete journal and checkpoint before writing. Merely
compatible Python/Torch/NumPy ranges do not satisfy exact continuation.
The driver uses one-thread deterministic CPU execution; selecting a GPU
does not accelerate it or waive runtime guards.

There is no mid-episode or mid-chunk simulator checkpoint. If a chunk was not
committed, its same predeclared scenes run again from the preceding committed
learner state. A difficult scene is not skipped or replaced. An incomplete
final generation makes ordinary resume fail closed. Explicit
`copy_verified_prefix` recovery copies verified history to a **fresh** directory
and leaves original evidence intact; use the procedure in
[`colab_development.md`](colab_development.md). Corrupt published artifacts,
gaps or incomplete middle generations cannot be silently omitted.

The collector expects the canonical `<variant>/<algorithm>/seed-<seed>`
layout. To retain a broken original and still collect a recovered chain,
assemble a fresh study root with the same layout. Stop every writer to the
old root first. The following copies verified history for every existing
declared chain, including complete chains, and copies the shared reference
report byte-for-byte if it exists. Missing chains stay missing; no training,
validation, resampling or checkpoint selection occurs:

```python
import hashlib
import json
import shutil
from pathlib import Path
from experiments.learning_budget_protocol import (
    DEFAULT_LEARNING_BUDGET_PROTOCOL, LearningBudgetProtocol,
)
from experiments.session_journal import copy_verified_prefix

old_root = Path("/content/drive/MyDrive/risk-aware-v2x/learning-budget-v1")
new_root = Path("/content/drive/MyDrive/risk-aware-v2x/learning-budget-v1-recovered")
if not old_root.is_dir() or old_root.is_symlink():
    raise ValueError("old_root must be the original regular study directory")
if new_root.exists() or new_root.is_symlink():
    raise FileExistsError("choose a fresh new_root")
old_root, new_root = old_root.resolve(), new_root.resolve()
if (old_root == new_root or new_root.is_relative_to(old_root)
        or old_root.is_relative_to(new_root)):
    raise ValueError("study roots must be distinct and outside one another")
plan = LearningBudgetProtocol.load(DEFAULT_LEARNING_BUDGET_PROTOCOL)
new_root.mkdir(parents=True, exist_ok=False)
for variant in plan.variants:
    for algorithm in plan.learning.algorithms:
        for seed in plan.learning.agent_seeds:
            relative = Path(variant) / algorithm / f"seed-{seed}"
            source = old_root / relative
            if not source.exists() and not source.is_symlink():
                continue
            contract = json.loads((source / "contract.json").read_text(encoding="utf-8"))
            copy_verified_prefix(source, new_root / relative, contract)

reference = old_root / "references" / "report.json"
if reference.exists() or reference.is_symlink():
    if (reference.parent.is_symlink() or reference.is_symlink()
            or not reference.is_file()):
        raise ValueError("reference must be a regular report file")
    def sha256(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    expected = sha256(reference)
    destination = new_root / "references" / "report.json"
    destination.parent.mkdir(exist_ok=False)
    with reference.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if sha256(destination) != expected or sha256(reference) != expected:
        raise RuntimeError("reference copy changed; retain evidence and inspect")
print(new_root)
```

If verification fails, preserve both roots for inspection; this procedure
does not delete or move original evidence. It can omit only an uncommitted
final generation through the existing recovery helper. Earlier committed
artifacts remain exact copies. It does not repair an invalid initialization
or corrupt published history.

Point the notebook's `STUDY_OUTPUT_ROOT` at the new root. If rerunning the
configuration cell to set it, remember that configuration clears readiness:
repeat successful preflight and plan inspection, and restore the mounted
Drive checks before enabling execution. Keep the same source/runtime and
select the interrupted chain's same variant, algorithm and seed with
`RESUME=True`. Use `RESUME=False` only for a declared chain that was missing
and has no directory in the new root. Existing complete chains need no new
execution. Collect all subsequent results with `--study-dir` pointing to the
new root; do not combine the original and recovered roots as extra runs.

At aligned ten-episode invocation boundaries, each chain normally has an
initial checkpoint plus 50 training checkpoints: about 612 across 12 chains.
Smaller invocation caps create additional checkpoints. Replay contents,
serialization and pause cadence affect bytes, so this is not a storage quota
or runtime estimate. Preserve all committed artifacts and a separate backup;
hashes and filesystem publication do not promise distributed locking or
remote Drive transactional durability.

## Fixed references and complete-grid collection

Run the shared references explicitly once under the same source/runtime and
protocols, using a fresh `references` directory:

```text
python -m experiments.learning_budget_results --study-dir reports/NEW-budget-study --references
```

This command executes only the five fixed policies on validation scenes. It
does not train a model. Then collect the grid into a new summary file:

```text
python -m experiments.learning_budget_results --study-dir reports/NEW-budget-study --output reports/NEW-budget-study/summary.json
```

The collector verifies immutable artifact hashes, declared model identities,
budget order, frozen validation and the shared validation cohort. It is
read-only with respect to sessions and does not load learner checkpoints,
generate trajectories or perform training. Its record-level verification
does not independently reconstruct every physical transition or model tensor.
Missing or unfinished chains yield progress only, with no partial-grid
performance curves. Completed model chains without the shared references
are marked as awaiting references rather than a complete study.

Final summaries retain every model, budget and profile. Episode costs have
equal episode weights; task latency/energy/deadline metrics and SV workload
retain their documented aggregation units. CVaR remains separate per model,
budget and profile, with no pooled-policy or pooled-seed tail. Paired
log-minus-linear differences use the same scenario keys; a CVaR difference
subtracts the two model CVaRs, not the CVaR of episode differences.

All three budget points remain visible regardless of outcome. No best seed,
checkpoint, early stopping, significance, convergence, or guaranteed gain
over greedy follows from these descriptive development results. Calibration,
adapted paper/query-free MEC baselines and a frozen held-out paper evaluation
remain later work.
