# Cost-Model-2 Development Pilot: 2026-11-17

This is a fresh bounded development audit under the corrected objective and
label-free observations, not main training or held-out paper evidence.
Main training has not started; it is planned for Colab, with prior notice to
Amir. See [learning_diagnostics.md](learning_diagnostics.md).

## Execution and verification

The complete checked-in `learning_pilot_v3.json` was executed once, without
changing budgets/hyperparameters after inspecting outcomes:

- Four algorithms x two seeds (11, 29): eight final models.
- Four nominal training episodes per model: 32 training executions.
- Two validation episodes per profile/model: 64 learned validation executions.
- Five fixed policies x four profiles x two cases: 40 fixed validation executions.
- **136 full-drain executions**, no unfinished admitted tasks, no discarded runs.
- Every model ingested 75 transitions and performed 44 gradient updates;
  352 updates across the eight models. Task traces are intentionally shared.
- All eight final checkpoints reloaded with matching online-weight hashes,
  protocol metadata, and source fingerprints. Test trajectories remained closed.
- The read-only summary produced 52 per-model/per-profile validation rows.
  Repeating postprocessing in the standard-library-only runtime produced the
  same JSON content, without Torch.
- **286 tests passed** in the full CPU learning environment. The bundled
  standard-library runtime discovered 286: 221 passed, 65 optional learning
  tests explicitly skipped. Diff whitespace checks passed.

The 3,785 completed task instances count repeated traces across model/policy
executions; they are not 3,785 independent trials or distinct traffic traces.

Local ignored artifacts, relative to the repository root:

- `reports/learning-pilot-v3-20261117/report.json` (5,525,940 bytes).
- `reports/learning-pilot-v3-20261117/checkpoints/` (eight final checkpoints).
- `reports/learning-pilot-v3-diagnostics-20261117/diagnostics.json` and `.md`.
- `reports/learning-pilot-v3-diagnostics-stdlib-20261117/` (postprocessing check).

The run records Git revision `a457326f146e1e3b1d657c5dafaec72b49b00c4a` and
`git_dirty=true`, correctly identifying the uncommitted diagnostic additions.
Historical reports/checkpoints/configurations were preserved.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `700c23f5048b6912ad5fa74a8714d15c67f031b30d50afe84644992ef4d61b2c` |
| Expanded scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Expanded full learning plan | `24e0d2a68be76c8bcdaf6f4262260785a9f1ad4ae2d144aa13a5d441593d5a1a` |
| Exact input report bytes | `517593776efa4ae0f4da869b4aec22809e95ada778a03e67b9b175c17b167836` |

## Descriptive outcomes

Every cell below is the **mean total cost of two validation episodes** for
one fixed model/policy. Lower is preferable under the approved objective.
Seeds remain separate; profiles are known development shifts, not independent
held-out populations. The complete diagnostics also record per-row CVaR,
latency, mobile energy, deadline misses/tardiness, workload shares and fallback.

| Policy | Agent seed | Nominal | High load | Fast servers | Harsh channel |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlwaysMEC | n/a | 5.3856 | 26.2162 | 5.3856 | 5.3856 |
| AlwaysV2V | n/a | 3.3126 | 13.6091 | 3.2940 | 3.3332 |
| Double DQN | 11 | 4.4539 | 18.0953 | 4.4324 | 4.2333 |
| Double DQN | 29 | 3.7266 | 14.7269 | 3.5586 | 3.7342 |
| DQN | 11 | 4.7279 | 18.1633 | 4.4145 | 4.6425 |
| DQN | 29 | 3.8144 | 15.8976 | 3.7456 | 3.9024 |
| MinimumEstimatedCombinedCost | n/a | 3.3467 | 13.4842 | 3.2975 | 3.3123 |
| MinimumEstimatedLatency | n/a | 5.0474 | 15.2737 | 5.0354 | 5.0503 |
| QR-DQN | 11 | 4.6629 | 19.8061 | 4.8369 | 4.5162 |
| QR-DQN | 29 | 3.8590 | 14.9776 | 4.0619 | 3.9510 |
| QR-DQN + CVaR | 11 | 3.8192 | 14.7537 | 3.7831 | 3.8725 |
| QR-DQN + CVaR | 29 | 3.6322 | 15.1847 | 3.7976 | 3.6462 |
| UniformRandom | n/a | 3.6654 | 13.2153 | 3.6840 | 3.7076 |

The cached combined-cost heuristic had lower observed mean cost than every
learned model in these four small profile samples. It was not the best fixed
policy in every profile. QR-CVaR did not consistently beat its matching-seed
neutral QR ablation: for seed 29/high load its mean cost was 15.1847 versus
14.9776. No algorithm, training seed or profile has been omitted on that basis.

QR-CVaR used SVs substantially, but deadline misses remained: seed 29/nominal
missed one of 39 task instances, while its matching neutral QR model missed
none. All tasks still completed. Seed 29/high load missed two of 128 tasks.
The risk score concerns the combined return, not a hard per-task deadline
constraint. MEC recovery guarantees neither on-time delivery nor zero cost.

For alpha=.9 with just two validation episodes, nominal tail mass is .2
episodes and empirical CVaR equals the maximum observed episode cost. These
values cannot establish a reliable tail-risk advantage or failure probability.
No old-cost-model outcome has been re-scored or treated as comparable evidence.

## Implications for the next stage

This budget exercises the pipeline rather than training a mature policy.
Each model completed **44 updates**, below the 50-update hard target-refresh
interval: the update logs contain zero scheduled target synchronizations
after initialization. The final exploration epsilon ranged from .19345 to
.22575, above the .05 endpoint. Evaluation still used epsilon=0. These facts
must accompany the small-sample outcome table; finishing the pilot does not
mean learning has converged.

Do not change objective weights, lambda, seeds, task deadlines or the final
test set to manufacture an improvement. First resolve physical/workload
calibration and the missing comparisons, declare a separate development budget
with enough updates and independent scenarios, then freeze the paper protocol
and prepare Colab persistence/restart. Main training remains a later milestone.
