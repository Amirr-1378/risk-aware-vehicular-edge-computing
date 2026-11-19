# Independent-Actor Team Return Interface: Verification, 2026-11-20

This implements and verifies the data/accounting stage of
[team_returns.md](team_returns.md). It follows Amir's explicit decision:
each User Vehicle independently chooses its destination; MEC only provides
information. Team costs are aggregated offline for potential training use.
There is no online MEC coordinator, additional actor information or new
physical communication protocol in this change.

## Executed checks

```text
python -m experiments.run_team_return_audit --output reports/team-return-audit-20261120/report.json
```

The default configuration was executed without changing the declared cost,
physical/scenario parameters or seeds:

- Profiles: nominal and high_load; validation indices 0 and 1.
- Policies: AlwaysMEC, AlwaysV2V, UniformRandom, MinimumEstimatedLatency,
  MinimumEstimatedCombinedCost.
- **20 matched pairs, 40 complete physical episode executions**: one original
  individual interface and one team-return interface for each scene/policy.
- Every pair had identical hashes of local actor inputs, action traces,
  elapsed slots, final physical metrics, individual accounting and original
  individual transition metadata. Policy RNGs were independently reset to the
  same declared seed for each interface.
- The unique global ledger reconstructed actual full-drain task costs through
  its prefix and disjoint intervals. Each user's separate overlapping team
  view also reconstructed the same total with its own prefix; views were not
  added across users. Discount-origin arithmetic was additionally checked at
  gamma=.5/.9/0 in automated fixtures.
- All admitted tasks completed. The offered task counts were 19, 20, 62 and
  66 per respective scenario execution; repeated policies/interfaces do not
  create additional independent traffic samples.
- No training, checkpoint creation, test-split trajectory access or historical
  artifact replacement occurred. All neural learning/checkpoint paths still
  use their previous individual-return contract.

**311 tests passed** with the full CPU learning runtime. The standard-library
runtime discovered 311: **246 passed**, 65 optional learning tests explicitly
skipped. New fixtures cover hand-computed global/individual-view overlap,
undiscounted/discounted prefix and drain conservation, users with no actions,
empty episodes, zero discount, local observation/action equivalence, invalid
batch atomicity, reset reproducibility, guard failures and legacy replay
rejection of the new team record type. Whitespace checks passed.

## Artifacts and compatibility

The ignored local report is
`reports/team-return-audit-20261120/report.json` (1,109,383 bytes). It records
the full configuration, local-input hashes, transition return summaries,
individual/global accounting, source provenance and the new team-return scope.
Physical cost remains model 2; team-return schema 1 is tagged
`all-users-sum-v1` and uses a distinct `TeamDecisionTransition` type.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `a6caff4f7d084067cabe57ef96c14a586c56ef95cc24ad7e87fbae58727d5f43` |
| Expanded scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Exact report bytes | `ed81f37e2ddefe62d4d7b72fdcdce9f8809e7c80dfd7b2f0bfb5c175d371ab09` |

Git provenance was `e3d82d84db14f8708136f1073d26be573f969673`, with
`git_dirty=true` accurately recording the new uncommitted interface.

## Limits and next milestone

This verifies accounting and information boundaries, not improved learning.
No team-compatible neural learner, replay deserializer, checkpoint resume
path or main training has been added. Connecting the new records requires an
explicit return-scope compatibility contract and a controlled learning audit.
Independent actors still have partial observations and interact through shared
resources; team rewards do not prove global or static-CVaR optimality.

Main training remains a later Colab milestone, to be announced before it
starts. This verified implementation is ready for a commit/push reminder
before advancing to neural integration; pushing requires user authorization.
