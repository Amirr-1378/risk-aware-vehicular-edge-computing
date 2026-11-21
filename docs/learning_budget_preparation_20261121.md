# Learning-budget preparation verification, 2026-11-21

The larger development study is prepared and locally verified. Its declared
12-chain experiment has **not** been executed. No Colab session, Drive mount,
main training or held-out trajectory was started during this milestone.
Local learning occurred only in the small deterministic test fixtures below.
The previous cost and SV-workload gaps against the strong greedy baseline
remain unresolved; infrastructure readiness is not a performance result.

The design and execution/recovery instructions are in
[`learning_budget_development.md`](learning_budget_development.md). The new
runner reuses the existing independent User Vehicle actors, individual
returns, cost/physical models and QR/QR-CVaR update implementation. Legacy
schema 3 remains the default elsewhere; log-SNR schema 4 is an explicit
paired candidate. The five fixed policies remain unchanged.

## Fixed plan and metadata-only check

The new scenario plan expands capacities to 500 training, 20 validation and
30 reserved test keys. Physical profiles and the existing seed prefixes are
unchanged. The new learning plan declares both encoders, both QR algorithms
and seeds 11/29/47. Each model continues to 500 episodes with mandatory
frozen evaluations after 20, 100 and 500 episodes. Validation contains the
same 20 cases per known profile at every point. No validation-selected
checkpoint, stopping rule, extra seed or budget extension is introduced.

```text
python -m experiments.run_learning_budget_session --plan
```

This command completed without a learner, scenario trajectory or output
directory. It reported 6000 training, 2880 learned validation and 400 shared
fixed-policy episodes: **9280 planned physical executions**, not completed
executions. The invocation cap counts training episodes only; reaching a
budget also runs the full 80-episode validation cohort before returning.

A separate exclusive metadata record is retained in the ignored file
`reports/learning-budget-preparation-v1-20261121/plan.json`. Its preflight
passed on CPython 3.12.14, Windows AMD64, Torch 2.14.1+cpu and NumPy 2.5.4.
It truthfully records parent commit
`012c3cec2b6f1d38989611bcab5ff4c3c80c1a11` and `git_dirty=true`.
These local builds describe this verification; a new Colab chain must record
its own exact identity and preserve it throughout continuation and comparison.

| Artifact identity | SHA256 |
| --- | --- |
| Actual Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Expanded scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Metadata record bytes | `9a1b9f36dda3d228a81bb16a757861387a3c296bfeba029ef0c8a702870c24dc` |

## Tests and state invariants

The optional-learning suite discovered **573 tests: 572 passed, one skipped**.
The skip is the existing Windows symlink-privilege check. The dependency-free
runtime discovered the same 573: **368 passed, 205 skipped**, comprising 204
optional-learning tests and the same symlink check. Both suites passed and
`git diff --check` passed. The milestone adds 48 tests: 11 protocol, 13 session,
14 result-collection and ten offline notebook checks.

```text
python -m unittest discover -s tests -q
```

Session fixtures use six training episodes, one nominal validation case per
point, budgets 2/4/6 and a maximum chunk of three episodes. For both encoders,
one-call and one-episode-per-call execution reached exactly equal online and
target networks, optimizer, replay/sampling RNG, exploration state, counters
and validation reports. Each model saw 36 transitions and 35 updates. A
matched training-only control also reached equal learner state, establishing
that mandatory intermediate evaluations did not alter continued learning.

Other session tests verified chunk and milestone boundaries, training-only
caps, complete-session reopening without execution or writes, strict source/
runtime/protocol/encoder identity, same-seed retry after mid-chunk interruption,
and frozen validation retry before any further training. Injected checkpoint
write interruption made ordinary resume fail closed; copying the verified
prefix to a fresh directory recovered equal final state while preserving
the original evidence. Injected validation mutation prevented publication.

Result fixtures contain four tiny chains, two training episodes per chain,
budgets 1/2, one validation case and five shared reference executions. They
verified complete-grid reporting, progress-only incomplete grids, exact
cohort pairing across budgets/models/references, separate per-model risk
summaries and unchanged fixed-policy membership. Changed source/runtime,
encoder/seed substitution, corrupt artifacts and modified validation flags,
clocks, seeds or fingerprints are rejected. Reference source/runtime changes
during execution cannot publish a report.

The collector was checked with model construction/deserialization,
environment stepping and trajectory generation forbidden. It preserved
source-session bytes and Python/Torch RNG states. Its verification covers
journal hashes and recorded identities; it does not independently reconstruct
physical transitions or checkpoint tensors. Output is exclusive and cannot
be written inside input sessions or reference directories.

Offline notebook tests compiled every code cell, checked the exact embedded
archive verifier, empty outputs and false default action gates. They exercised
preflight/plan/Drive guards with mocks and verified one explicitly selected
chain call, without a whole-grid loop. These tests did not access Colab,
Drive, GitHub credentials or external services.

## Handoff

Commit and push this verified milestone with Amir's authorization before
preparing its commit-pinned source ZIP. Then use
[`../notebooks/colab_learning_budget.ipynb`](../notebooks/colab_learning_budget.ipynb)
to inspect the plan and explicitly advance one development chain. Preserve
all declared arms and seeds; the full grid is required for final curves.
Calibration, paper/query-free MEC baselines, a frozen paper protocol and main
Colab training remain later milestones. More training alone does not promise
lower mean cost, reliable tail estimates or publication readiness.
