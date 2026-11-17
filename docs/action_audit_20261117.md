# Offline Action Audit: 2026-11-17

This audit diagnoses possible gaps to cached combined-cost greedy, without
training a model, changing the objective, or opening held-out test trajectories.
The [method and development recommendations](greedy_improvement_plan.md)
define the information boundary and the interpretation of ex-post branches.
Main training has not started and remains planned for Colab with prior notice.

## Declared execution

```text
python -m experiments.run_action_audit --output reports/action-audit-20261117/report.json
```

Defaults were used: nominal/high_load, validation indices 0/1, the first eight
individual decisions of each unchanged greedy path in slot/user-ID order.
The run inspected **32 decisions** over **four reference episodes**, with
**139 full-drain action branches**. Every observed-valid destination was
retained, including actions rejected by the heuristic's contact constraint.
The four reference episodes also drained completely. Unchanged-action branches
exactly reproduced reference terminal metrics and termination slots.

| Episode | Reference decisions | Inspected | Owner gain opportunities | System gain opportunities | Owner/system sign conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal:validation:0 | 19 | 8 | 4 | 4 | 1 |
| nominal:validation:1 | 20 | 8 | 3 | 5 | 2 |
| high_load:validation:0 | 62 | 8 | 7 | 7 | 5 |
| high_load:validation:1 | 66 | 8 | 7 | 8 | 2 |
| Total | 167 | 32 | 21 | 24 | 10 |

An opportunity means at least one branch had strictly smaller remaining cost
than the unchanged greedy branch, with tolerance 1e-9. A conflict means at
least one alternative's owner/system cost differences had opposite signs;
it need not be either objective's best action. Counts are descriptive and
dependent. The mean ex-post owner gain per inspected decision was .07094196;
the separate mean ex-post system gain was .11553659. **These are not additive
episode improvements or measured gains of a deployable learned policy.**

## Concrete credit-alignment example

At slot 23 for user 2 in `high_load:validation:0`, greedy chose action 3.
The owner's ex-post best branch instead chose action 2:

- Owner future cost changed by **-.05062374**.
- Other users' future cost changed by **+.13478983**.
- Total future cost changed by **+.08416609**.

Thus the owner improved while the system became more expensive on that
realized continuation. The system's ex-post best action at this fork was
MEC (action 0). This is one concrete illustration of the distinction between
per-user learning returns and the reported global objective, not evidence
that a particular online policy should select MEC at an indistinguishable
received observation. Other simultaneous actions were fixed across branches;
all later actions were greedy on their own received observations.

The current per-user transition design therefore merits an explicit objective/
credit-assignment decision before substantially increasing training. Simply
copying global slot costs into overlapping per-user transitions does not
derive a centralized team MDP. No learning-return, encoder, QR target, CVaR
weight or checkpoint compatibility change was made by this audit.

## Interpretation and next priorities

Greedy was not ex-post optimal for every inspected single-action continuation.
That supports investigating delayed costs and shared-resource externalities;
it does **not** establish that future improvements are predictable from cached
observations. Branch minima know a realized future, use a fixed greedy
continuation, and are neither expected Q values nor CVaR estimates. They are
not an optimal-policy bound and must never become validation-derived training
labels. Broader independent-scenario evidence and information-sufficiency
checks are still needed.

Priority is to align the intended learning objective with the global metric,
then declare an informative training-only development budget with repeated
target updates and sufficient distinct traces. Greedy demonstrations or
initialization can be a separate controlled variant if sample efficiency
remains a problem. Changing cost weights, selecting favorable seeds or
weakening the baseline is not an improvement strategy. Lower mean and lower
upper-tail cost remain separate hypotheses for QR-DQN + CVaR.

## Verification and provenance

- **296 tests passed** with the full CPU learning environment.
- Standard-library runtime: 296 discovered, **231 passed**, 65 optional
  learning tests explicitly skipped. The new action audit has no Torch dependency.
- Tests cover hand-computed owner/system/external-cost differences, action
  ties, invalid costs/missing branches, complete batch semantics, source/RNG
  preservation, reversed branch order, sunk-cost subtraction, full drain,
  explicit failures, fixed-prefix planning and validation-only seed access.
- No historical report/checkpoint/configuration was replaced. The new local
  report is ignored by Git and is 523,009 bytes.

| Fingerprint | SHA256 |
| --- | --- |
| Actual implementation source | `69990f9d0704fb6f19394341c3c52d4593508276e594e718700be208eaac8a81` |
| Expanded scenario plan | `646cd5580e7b36cf2a4be1713c6f08313301d64853c98fd66660f70c2d59cd30` |
| Exact report bytes | `ebc7bcfa1c7f9caecd457d832109e32681bba6e9a3ffb3fc8309dcbec7387c6a` |

Git revision at execution was `a457326f146e1e3b1d657c5dafaec72b49b00c4a`,
with `git_dirty=true` recording the new uncommitted diagnostic implementation.
