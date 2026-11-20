# Team-Return Neural Integration: Verification, 2026-11-20

The opt-in team-return path now connects independent local-observation actors
to typed replay, the existing DQN/QR optimizer math, frozen evaluation, and
complete episode-boundary checkpoints. See [team_returns.md](team_returns.md)
for scientific meaning and [neural_agents.md](neural_agents.md) for updates.
This is a mechanics verification, not main training or evidence of lower
cost than greedy. MEC remains an information provider, not an action coordinator.

## Implemented contracts

- `TeamNeuralCostAgent` inherits the tested network, masks, action selection,
  risk score, quantile loss and optimizer behavior. It consumes team wrapper
  costs in TD targets; it never substitutes `source` individual costs.
- Physical cost model 2 and raw/vector observation schema 3 remain unchanged.
  Alpha=.9, lambda=.5, sigmoid beta and discrete argmin semantics are preserved.
- Team transition schema 1 and replay container schema 1 record
  `all-users-sum-v1`, the complete return contract, actual physical cost model,
  individual source records, ring cursor and replay RNG. Team and individual
  buffers reject cross-scope records and headers, even when empty.
- Team checkpoint schema 3 requires the return contract and team replay.
  Individual checkpoint schema 2 remains supported by the individual loader.
  Matching dimensions do not permit cross-scope resume; no weight migration
  or demonstration prefill is implemented.
- `run_learning_smoke --return-scope team` uses run kind
  `synthetic-team-learning-smoke-v1` and report schema 3. Run metadata binds
  source, scenario/configuration, seeds, schedule and return contract. The
  default remains `individual`; the stochastic v3 learning protocol still
  explicitly uses individual returns.
- Runtime rejects agent/adapter scope mismatches before resetting or updating
  an episode. Reports retain per-user physical `accounting` plus unique global
  `team_accounting`. Overlapping actor-view samples are not added as cost.

## Automated verification

**329 tests passed** in the full CPU learning runtime. The standard-library
runtime discovered 329: **251 passed**, 78 optional neural tests explicitly
skipped. No learning dependency was introduced into core package imports.

New hand-computed tests check DQN/Double-Q targets from discounted team cost
7.5 and bootstrap .81: target 9.12 for target value 2, and 13.98 for the
Double-Q-selected target value 8. QR tests check fractional upper-tail action
selection and terminal team targets. With team costs set equal to individual
costs, shared initialization, exploration samples, optimizer statistics and
updated weights exactly match the original learner for all four algorithms.

Additional tests cover strict JSON roundtrips, cost-scope preservation,
uniform ring/RNG resume, invalid cursors/contracts, gamma mismatch, empty
cross-scope checkpoints, optimizer continuation, frozen target/online/replay/
exploration states in evaluation, zero-episode runs, existing-file protection,
and complete split/uninterrupted learning equivalence. Existing individual
learning/checkpoint tests also pass.

## Fresh bounded execution

The actual verification used the smoke runner's existing engineering settings:
hidden layers 32/32, QR N=15, batch/warm-up 4, replay 256, target update every
5 optimizer updates, physical-slot gamma=1, alpha=.9/lambda=.5, epsilon decay
over 200 training slots. These are smoke settings, not main-training choices.
The default deterministic six-task/two-user/two-SV fixture, agent seed 17 and
the predeclared physical/evaluation seeds were used without outcome-driven
budget or seed selection.

For each of DQN, Double DQN, QR-DQN and QR-DQN + CVaR:

1. Train four episodes with `return_scope="team"` and save a fresh final checkpoint.
2. Verify **24 transitions, 21 gradient updates and four target refreshes**.
3. Run two frozen greedy evaluation episodes; check complete task drain and
   actual physical/global cost conservation.

For QR-CVaR, separately train two episodes, save, reload and train two more.
Its final checkpoint was recursively compared with the uninterrupted four-
episode checkpoint: **every payload field matched**, including online/target
weights, optimizer moments, replay contents/cursor/RNG, exploration RNGs,
counters, encoder/configuration and run metadata. The final evaluation reports
also matched exactly. All six resulting checkpoints reloaded through the team
loader with matching source/return metadata.

Across these verification executions there were **20 training episodes,
12 evaluation episodes and 105 actual optimizer updates**. The split-run
executions repeat the QR-CVaR experiment; they are not additional independent
trained seeds. All 32 physical executions completed all six admitted tasks.
The deterministic fixture/evaluation seeds are not an untouched stochastic
benchmark, and no test-split trajectory was constructed or evaluated.

All **eight historical cost-model-2 individual checkpoints** from the earlier
v3 pilot still load through `NeuralCostAgent`, with identical online-weight
fingerprints and unchanged files. This demonstrates continued individual-loader
compatibility; it does not waive the smoke runner's exact-source/protocol
requirements for reproducing an old training continuation.

## Local artifacts and provenance

Fresh ignored artifacts are in `reports/team-learning-verification-20261120/`:
four algorithm reports/final checkpoints, QR-CVaR first-two/resumed reports and
checkpoints, plus `verification.json`. The seven JSON files total 336,182 bytes;
`verification.json` contains per-report byte hashes and resume verification.
Historical reports/checkpoints/configurations were preserved.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `219d7b098eeee2473723b1eb5a0aef4998e0614c8f9ae14951a8bc4c48f8dc25` |
| Exact verification index bytes | `92f29e32fec10ca376b2a643c1e63ae3901a69af10244417c7937148fdfb3374` |

Run provenance records Git revision
`4963b6849e3a00a7c1639f128cea17565d55ebd5` and `git_dirty=true`, correctly
identifying the new implementation before commit.

## Remaining work

Team-cost independent learning remains partially observed and potentially
nonstationary, with correlated replay samples and possible higher variance.
No convergence, global/static-CVaR optimality, safety or performance improvement
is established. The next stage needs an explicit stochastic return-scope
ablation plan with matched budgets/seeds and an informative fixed development
budget. Calibration, missing baselines and the final evaluation protocol remain
open. Main training is still reserved for Colab and must be announced first.
This completed stage is ready for the requested commit/push reminder before
proceeding; no push is authorized by the implementation request alone.
