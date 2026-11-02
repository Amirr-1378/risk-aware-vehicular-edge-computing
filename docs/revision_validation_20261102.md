# Approved Cost/Preference Revision: Verification Record (2026-11-02)

This is a new implementation audit, not a rewrite or re-score of historical
results. The reference is [objective_and_preferences.md](objective_and_preferences.md).
No convergence, comparative advantage, novelty, global CVaR optimality or
unconditional deadline/failure guarantee is inferred.

## Actual dependency trace

1. Radio/CPU/query progress records actual UV+SV energy; all active tasks
   accrue elapsed time. Discarded V2V work retains its own disjoint buckets.
2. `environment.accounting.incremental_cost` adds only .60*delta_T/D,
   .25*delta_E/E_budget, and the one-time .15 MEC-use event. Miss events record
   metrics, not charges. Task/queue/observation have no manual class field.
3. `DecisionAdapter` retains slot cost, irregular decision intervals, full
   drain, and explicit pre-action prefix. Reward is exactly negative cost.
4. Replay records explicit cost model 2; targets use actual interval cost
   plus discounted target atoms. Upper-cost CVaR action selection is unchanged.
5. Greedy behavior/target selection never reads beta or sigmoid outputs.
   Read-only evaluation diagnostics expose the same J values and the stable
   sigmoid of each valid SV's J-minus-MEC difference.
6. All experiment runners use terminal metrics which independently reconstruct
   final cost from completion-minus-arrival, actual mobile energy and final
   executor. Incorrect time origin, duplicate MEC charge or a hidden .2/2
   penalty causes a reconciliation error rather than a misleading report.

## Automated tests

- Full CPU learning environment: **273 tests passed**, no skips.
- Standard-library-only runtime: **273 discovered, 216 passed, 57 optional
  neural tests explicitly skipped**.
- Existing episode-boundary checkpoint/resume, seeded global-RNG isolation,
  action masking/ties, sign equivalence and fractional upper tails still pass.
- New tests cover late ratios above one without saturation, absence of manual
  fields/penalty arguments, pure EDF/preemption, direct MEC/SV/fallback cost
  reconstruction including nonzero arrival origin, wasted mobile energy,
  tardiness, sigmoid sign/center/monotonicity/extremes, and beta-independent
  discrete selection, exploration, TD targets and exact optimizer updates.
- Version checks reject historical replay/transitions/checkpoints even with
  matching vector dimensions; old configuration loaders fail closed.

Commands from the repository root (PowerShell):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

## Fresh bounded stochastic verification

A declared subset of the current learning plan was run, preserving the same
agent hyperparameters, gamma=1, alpha=.9, lambda=.5 and beta=1:

```python
from dataclasses import replace
from experiments.learning_protocol import LearningProtocol
from experiments.run_learning_audit import DEFAULT_LEARNING_PROTOCOL, run_learning_audit

plan = replace(
    LearningProtocol.load(DEFAULT_LEARNING_PROTOCOL),
    revision="approved-revision-verification-20261102",
    algorithms=("qr_dqn_cvar",), agent_seeds=(11,),
    validation_profiles=("nominal",), validation_cases=1,
)
# Choose a NEW unused directory for a repeated run; never replace this one.
report = run_learning_audit(learning=plan, output_directory="reports/approved-revision-20261102")
```

This completed 4 training episodes, 1 learned validation episode and 5
fixed-policy validation episodes: **10 executions**, all drained. It ingested
75 training transitions and performed **44 gradient updates**. There were no
incomplete admitted tasks in these executions. This is an observed bounded
audit result, not an all-scenarios safety proof. No test scene was instantiated
or evaluated; no poor run or offered task was silently removed/replaced.

The local report is `reports/approved-revision-20261102/report.json` (313,594
bytes) with one final checkpoint. The checkpoint reloaded with matching
weights/metadata; all 19 learned validation decisions had valid preference
diagnostics at alpha=.9, lambda=.5, beta=1. Raw/encoded input schema is 3,
dimension 96 for four server slots. Model/source/config hashes were verified.

- Actual source SHA256:
  `933a8a927e209877f137be5c818595cb194fb3a943bf58bf371410ae3223bbe6`.
- Scenario-plan SHA256:
  `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30`.
- Expanded short-learning-plan SHA256:
  `1f1ff3e9f4080a11008aa40b08d2e71401663d3bbb9449795d8e50903870b9a2`.

Historical `development_v1.json`, `learning_pilot_v1.json`, and
`learning_pilot_v2.json` are unchanged. All 16 historical v1/v2 pilot
checkpoints remain present and their loading under the new contract was
correctly rejected. Historical verification-record bodies in the active
combined-cost/learning documents were compared to HEAD and remain unchanged
after newline normalization; they are clearly marked as old-objective records.
Runtime reports/checkpoints remain ignored by Git. No commit/push was made.

## Scientific limits still open

Beta=1 is an explicit untrained display default, not an empirically calibrated
preference scale; changing it must never change selection. Real workload,
radio/mobility/time-correlation, budget and deadline calibration remain open.
The retained numeric deadline mixture is unlabelled synthetic data, not an
emergency-braking/obstacle detector. Larger independent training/scenario
budgets and quantile-tail calibration are still needed. The paper best-response
baseline and query-free MEC remain separate incomplete comparisons; neither
was fabricated or silently redefined in this revision.
