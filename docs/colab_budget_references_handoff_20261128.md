# Shared fixed-reference Colab handoff 2026 11 28

All twelve learning chains and 612 checkpoints passed the
[received-archive audits](colab_budget_chain_12_20261128.md). The next step is
one shared fixed-policy reference run: five unchanged policies, 20 validation
cases in each of four profiles, or 400 physical episode executions. This
handoff prepares that run; it does not execute real evaluation or training.

The policies are `always_mec`, `always_v2v`, `minimum_estimated_latency`,
`uniform_random` and `minimum_estimated_combined_cost`. They run once on the
same fixed cohort, rather than once per model, seed or budget. The strong
combined-cost greedy policy and all physical/information assumptions remain
unchanged. No learned checkpoint is selected, loaded, resumed or retrained.

The existing `experiments.learning_budget_results.run_learning_budget_references`
implementation is reused without modification. The launcher is an ignored
handoff artifact, outside the scientific Python-source fingerprint:
`artifacts/colab-learning-budget-5d8845b-20261122/run_fixed_budget_references.py`.

| Required identity | SHA256 |
| --- | --- |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Producer runtime | `f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463` |
| Corrected launcher bytes | `7e5e891f1997630c3f45c2b885c7837b35c7ba7ee0845a22f23dc671aff72713` |

## Execution in the existing notebook

Keep the original pinned source ZIP, prepared plan and mounted Drive. A fresh
runtime must first restore those prerequisites and the exact producer
runtime identity. Upload the launcher to `/content`, then execute a new cell:

```python
RUN_STUDY = False
RUN_FIXED_REFERENCES = True
%run -i /content/run_fixed_budget_references.py
```

Do not rerun the initial configuration or ZIP extraction in an already
prepared runtime. Do not rerun any completed learning chain. The launcher
requires plan/Drive readiness, verifies actual source/runtime and freshly
loaded protocol fingerprints, and checks presence and source/runtime/selection
identities of the twelve chain contracts. It relies on the completed local
archive audits for journal/checkpoint verification; it does not rehash all
learning checkpoints, generate scenes during readiness checks or calculate
learning curves.

The only real execution entry point is the unchanged fixed-reference runner.
Its destination is `<STUDY_OUTPUT_ROOT>/references/report.json`, normally
`/content/drive/MyDrive/risk-aware-v2x/learning-budget-v1/references/report.json`.
The `references` directory must be fresh. Existing directories, partial
outputs and errors are preserved; no overwrite, automatic retry, deletion
or recovery is performed. The existing runner saves its single report after
all 400 executions and has no per-profile resume checkpoint. If interrupted,
preserve the directory and diagnose before any retry.

The compact result prints count 400, all five policy names, four profiles,
20 cases per profile, exact identities, report path/size/SHA256, and false
training/main-training/test flags. Full output is retained in notebook
variables `BUDGET_REFERENCE_RESULT`, `BUDGET_REFERENCE_SUMMARY` and
`BUDGET_REFERENCE_PATH`. Send the compact JSON first, then transfer the saved
reference report for local input/cohort/accounting verification. The launcher
does not execute the full-grid performance collector.

## Local preparation verification and remaining work

Launcher checks used twelve copied contract files and a mocked evaluator,
with dummy output explicitly labeled synthetic. The successful fixture called
the mocked evaluator once, verified saved JSON bytes/digest and unchanged
contract bytes. Eleven negative cases rejected execution before that evaluator:
disabled reference gate, enabled learning gate, missing plan/Drive/mount
readiness, different source, different exact runtime, existing reference
directory, missing chain, changed chain identity and Drive root as study path.
No real environment episode, learner initialization or training update ran.
Evidence is ignored under `reports/fixed-budget-references-handoff-20261128/`.
The earlier full test suite was not rerun for this external launcher and
documentation change; the scientific implementation remains unchanged.

At handoff preparation the study was awaiting references. Completion and audit of those 400
executions precede complete-study collection and interpretation across every
declared model, budget and profile. No performance win, representation adoption,
checkpoint selection or paper result follows from launch preparation. Main
training remains a separately announced Colab milestone after physical
calibration and a frozen paper protocol; held-out trajectories remain closed.

## First reported attempt: serialization guard correction

The user's screenshot shows the first invocation returned from the unchanged
reference evaluator, passed reference identity/cohort-count checks and reached
the final saved-report comparison, which raised
`ValueError: Persisted reference bytes differ from the returned report.`
The report had already been saved by the existing runner. This is a launcher
error after evaluation, not evidence that the physical run must be repeated.
At diagnosis time the real reference report had not yet been received for
local verification. Its later successful receipt is documented below.

The launcher incorrectly compared `json.loads(payload)` directly with the
in-memory report. `asdict` retains tuples in the scenario protocol, including
`split_counts`, whereas JSON decodes those arrays as lists. The old equality
therefore rejects a correctly serialized report. The initial mocked evaluator
used JSON-native data and missed this case; its success did not establish
compatibility with the real tuple-bearing protocol.

The corrected launcher compares bytes against the exact unchanged
`save_audit_report` serialization (`json.dumps(..., indent=2, allow_nan=False)`
plus a newline, encoded as UTF-8). No tolerance or scientific/runtime guard
was relaxed. The scientific runner, source fingerprint, protocols and saved
reference report are unchanged. The original launcher, SHA256
`d5a9ff878f62bdd98a51591a30650d14892cab7a0a67c6b200a6cff2fabe1147`,
is preserved under the ignored correction-evidence directory.

Regression verification used the actual pinned scenario protocol inside a
synthetic report, the real saving function, and a mocked evaluator. The old
object-equality failure was reproduced, corrected exact-byte comparison
passed, eleven pre-execution guard cases still failed closed, and altered
saved bytes were rejected even when they remained valid JSON. No real episode,
training or experiment rerun occurred locally. Evidence is under
`reports/fixed-budget-references-json-fix-20261128/`, including
`serialization_diagnostic.json` and `verification_final.json`.

Preserve the existing `references` directory and do not rerun the reference
launcher. Retrieve the original report in a new Colab cell:

```python
from pathlib import Path
import hashlib
from google.colab import files

RUN_STUDY = False
RUN_FIXED_REFERENCES = False
reference_path = Path(STUDY_OUTPUT_ROOT) / "references" / "report.json"
reference_payload = reference_path.read_bytes()
print("bytes:", len(reference_payload))
print("sha256:", hashlib.sha256(reference_payload).hexdigest())
files.download(str(reference_path))
```

This reads and downloads the existing report without changing Drive evidence
or invoking evaluation. The new launcher is not needed to retrieve this
already saved report.

## Subsequent receipt and verification

The original saved report was subsequently received without repeating the
400 evaluations. Its
[verification report](colab_budget_references_verification_20261128.md)
records exact source/runtime/protocol agreement, all 400 fixed-policy
records, 80 static manifests, cost accounting and pairings with all twelve
learning chains. Actual report bytes reproduce the tuple/list comparison
diagnosis while matching the unchanged serializer. The full plan's 9280
physical executions are verified; complete-grid collection and scientific
interpretation remain the next step. Main training and held-out evaluation
remain unstarted.
