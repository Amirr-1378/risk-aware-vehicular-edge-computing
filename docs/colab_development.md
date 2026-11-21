# Colab development preparation

This stage prepares a bounded, resumable **development** session. It does not
start main training, freeze a paper protocol, or establish improvement over
greedy. Main training must run in Colab after Amir is notified and its scientific
protocol is agreed. There is no main-training entry point in this notebook.

## Method boundary

The default remains QR-DQN + CVaR, alpha 0.9, risk weight 0.5, individual
returns, and independent decisions at each User Vehicle. MEC supplies information.
There is one MEC and several Server Vehicles, with whole-task remote offloading.
The cost, queue, mobility, channel, query/cache and recovery models are unchanged.
The runner calls the existing `_episode` / learning runtime and agent classes.
It has no teacher-policy argument, greedy warmup, imitation loss, centralized
controller, extra actor input, or held-out evaluation option. The separately
identified `team` return scope remains opt-in; checkpoint scopes cannot mix.

The default scientific plan is the existing `learning_pilot_v3.json` and
`development_v2.json`. A session chooses **one** declared algorithm and agent
seed from that plan: by default QR-CVaR, seed 11, four nominal training episodes,
then two validation cases in each of four known profiles. This is neither a
complete multi-algorithm comparison nor a new paper experiment. A pause after
one episode is not a new training budget or a validation-based stopping rule.

## Source and runtime setup

Use [`../notebooks/colab_development.ipynb`](../notebooks/colab_development.ipynb).
All execution gates initially equal `False`; setup with filled identities can
check source/runtime, but Run all cannot mount Drive or start training by default.
The checked-in notebook has no saved outputs or credentials.

1. Commit and push this milestone before transferring it to Colab. On that same
   trusted, clean checkout, obtain the full commit with `git rev-parse HEAD` and
   the source digest with
   `python -c "from experiments.provenance import source_provenance; print(source_provenance()['source_sha256'])"`.
2. In a browser signed into GitHub, download
   `https://github.com/Amirr-1378/risk-aware-vehicular-edge-computing/archive/<FULL_COMMIT>.zip`.
   Upload it through Colab's Files sidebar. Copy the independently obtained
   commit and digest into the notebook. GitHub credentials stay in the browser.
3. Choose a CPU runtime. The notebook's embedded standard-library verifier is
   exactly `experiments/colab_source.py`, checked by an offline test. It verifies
   every archive path, required files, regular member types, size limit and
   normalized Python-source digest **before** creating an exclusive source root.
   Existing source destinations are not overwritten. The archive root declares
   a commit; it is not cryptographic proof of Git history. The Python digest
   excludes JSON/configuration, notebook, documentation and `pyproject.toml`,
   so the entire archive must come from the trusted browser download.
4. Metadata-only preflight checks Python >=3.11, stable Torch >=2.6,<3 and stable
   NumPy >=1.26,<3. It does not import learning libraries, install packages,
   train or evaluate. The notebook uses the source root and `src` on `sys.path`
   and `PYTHONPATH`; no package installation is necessary. If preflight fails,
   prepare compatible dependencies explicitly before proceeding. Passing
   metadata checks does not prove that a library can import successfully.
5. Optionally enable only the offline setup tests. They use standard-library
   unittest and do not train. Enable Drive mounting separately and select a
   new output subfolder under MyDrive. Only then opt into a development call.

The agent runs on CPU with one Torch thread and deterministic algorithms. A
selected Colab GPU does not accelerate this driver. Exact Python build,
implementation, OS family, machine architecture, Torch wheel version including
local build tag, NumPy version and execution settings form the resume identity.
A new session records whichever compatible runtime is actually used; a resume
requires the exact recorded identity. A Windows checkpoint is therefore not
resumed as a Linux session. These guards do not promise bitwise reproducibility
between different CPU hardware, kernels or library builds. PyTorch explicitly
limits cross-release/platform/device reproducibility in its
[reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Session lifecycle

The CLI equivalent of the notebook's opt-in development call is:

```text
python -m experiments.runtime_preflight --require-learning
python -m experiments.run_resumable_development --output-dir reports/NEW-session --max-additional-episodes 1
python -m experiments.run_resumable_development --output-dir reports/NEW-session --resume --max-additional-episodes 1
```

`--learning-protocol`, `--scenario-protocol`, `--algorithm`, `--agent-seed` and
`--return-scope` must preserve the existing session's contract on resume. The
per-call episode cap can change; the planned total budget, split order and
model settings cannot. A new or repeated invocation never silently extends the
plan. Reopening a complete session verifies it and returns the same results,
without training, evaluation or new artifacts.

The immutable contract includes the full expanded scientific plans, their
fingerprints, exact source SHA, encoder, model/return contracts, runtime,
algorithm/seed and standard learned-epsilon behavior. Git revision/dirty state
are audit information in the initial report; the actual normalized source hash
binds the implementation, including uncommitted Python changes. JSON object keys
are normalized to strings at persistence boundaries, including vehicle IDs.

Each session contains `contract.json` and a contiguous `steps/` journal:

| Step | Artifacts | Meaning |
| --- | --- | --- |
| 000000 | report, untrained checkpoint, commit | Exact reproducible initial state |
| 000001 ... N | episode report, checkpoint, commit | One fully drained training episode per step |
| N+1 | frozen validation report, commit | All declared final-model validation cases completed |

Reports/checkpoints are written exclusively, followed by a temporary
`commit.pending` and publication as `commit.json`. Every committed generation
hashes its actual artifacts and previous commit. There is no mutable HEAD or
automatic selection of a newer uncommitted checkpoint. Resume verifies the
whole chain, scenes, seed/order/phase, full-drain counts, training-slot clock,
transition/update/replay counters, checkpoint metadata and exact runtime before
writing anything. Artifact hashes detect byte corruption; scientific
reproducibility is assessed through state equality. Torch files can have
different serialization/storage sharing after load while holding equal state.

The simulator is not checkpointed mid-episode. If interrupted before a new
journal step starts, the same pristine scene/channel seed is rerun from the
last committed learner state. No failed scene is skipped or rerolled. If an
uncommitted final step remains, ordinary resume stops and retains all evidence.

Final validation runs only after every planned training episode. It uses the
fixed final checkpoint, epsilon zero, no updates and no teacher. The complete
learner state, including target, optimizer, replay and RNGs, is checked before
and after evaluation. Interrupted unpublished validation repeats the same
validation set from the saved final learner; it does not train again. Results
are complete only after every declared validation case has been published.

## Explicit recovery and storage limits

Keep one writer per output directory; never use two Colab tabs/runtimes to write
the same session. Exclusive creation and hashes do not provide distributed
locking. Journal publication uses filesystem rename; this is not a guarantee
of remote Google Drive transactional durability. Colab virtual machines are
temporary and mounted Drive operations can fail or time out; see the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).
Wait for synchronization and preserve a separate copy of complete artifacts.

For a final generation without a published commit (including a partial
`commit.pending`), inspect the evidence and explicitly copy the verified prefix
to a **new** directory. The original remains untouched:

```python
import json
from pathlib import Path
from experiments.session_journal import copy_verified_prefix

source = Path("reports/interrupted-session")
destination = Path("reports/NEW-recovered-session")
contract = json.loads((source / "contract.json").read_text(encoding="utf-8"))
copied = copy_verified_prefix(source, destination, contract)
```

Then invoke the normal runner with `resume=True` on the recovered directory and
the original scientific/runtime inputs. Recovery does not waive their guards.
Only the incomplete **final** generation may be omitted. Corrupt published
commits/artifacts, gaps, an incomplete middle generation, extra files, symbolic
paths or an incompatible contract are rejected. Do not delete journal files,
overwrite old checkpoints, weaken identity checks or discard a difficult seed
to force continuation. Corrupt committed evidence may require a completely new
development run, with the interruption documented. An initialization interrupted
before a valid contract/steps directory exists is also not automatically repaired.

Incomplete sessions explicitly report completed/planned train/validation counts
and `status=incomplete`; they provide no final comparison against greedy.
No performance conclusion, change to the final paper method, or main-training
approval follows from these infrastructure checks.

The local verification record is
[`colab_readiness_validation_20261121.md`](colab_readiness_validation_20261121.md).
The subsequent Colab execution and independently verified archive, checkpoints
and episode records are documented in
[`colab_execution_20261121.md`](colab_execution_20261121.md). This completes the
small execution/continuation check; main training remains a separate milestone.
