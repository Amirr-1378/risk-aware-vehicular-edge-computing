# Shared fixed-reference verification 2026 11 28

The original Colab `references/report.json` passed source/runtime/protocol,
cohort, static-manifest, recorded-metric and accounting checks for all five
fixed policies across four profiles and 20 validation cases per profile.
All 400 fixed-reference executions are verified, together with all twelve
learning chains and 612 checkpoints. The declared execution plan is complete:
6000 training episodes + 2880 learned validation executions + 400 shared
fixed-reference executions = **9280 physical episode executions**.

Complete-grid performance collection and scientific interpretation remain
pending. This milestone verifies the received records; it does not establish
lower cost than greedy, convergence, statistical significance, a final
representation choice or a paper benchmark. Main training and held-out paper
evaluation remain unstarted.

## Received evidence and scientific identity

The received file was `C:/Users/Amir/Downloads/report.json`,
4,916,972 bytes, copied exclusively without modifying the
original. Its SHA256 is:

```text
70b55ea7f2167c46cab9d2524f968a41375b2f0cf42043cd4b8f67090451d274
```

The local evidence directory is ignored:
`reports/colab-budget-fixed-references-5d8845b-20261128/`.
It contains the original bytes under `received/references/report.json`.

| Identity | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Producer runtime | `f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463` |

The producer identity matches all learning chains: Linux x86_64,
CPython 3.13.18, Torch 2.14.1+cpu, NumPy 2.4.6, one CPU thread and deterministic
execution. Read-only audit used Windows AMD64, CPython 3.12.14, the same Torch
release and NumPy 2.5.4. No exact continuation across those runtimes is claimed.

The fixed policies are `always_mec`, `always_v2v`,
`minimum_estimated_latency`, `uniform_random` and
`minimum_estimated_combined_cost`. Every profile has all five, each with
20 records. The public hardware/objective configuration was independently
reconstructed from each static scenario and matches the recorded suite v3
configuration. Strong combined-cost greedy was not weakened or retuned.
All policies retain the mandatory shared query/cache protocol; AlwaysMEC
here is not a query-free MEC baseline or the paper best-response baseline.

Independent User Vehicle destination decisions, information-only MEC,
indivisible offloading, plain EDF and objective v2 remain unchanged:
`z=.60*T/D+.25*E_mobile/E_budget+.15*M`, without manual labels or a separate
deadline-miss penalty. The proposed method remains QR-DQN + CVaR. No objective,
physical model, learning source, seed or budget changed during this step.

## Cohort, metric and accounting verification

All 80 recorded validation scenarios have the expected episode IDs,
role-separated seeds, fingerprints and full static manifests. The audit
reconstructed those manifests without stepping environments. All twelve
learning journals were rechecked against their prior evidence digests and
contract identities. Artifact hashes, mandatory budgets, frozen validation
and the same profile/case cohort passed checks for all 2880 learned
validation executions, including every model and budget, without filtering
seeds or selecting checkpoints.

All 400 reference records retain complete task drain, finite metrics,
per-user prefix/closed cost conservation and discounted-cost agreement at
gamma 1. Episode totals reconcile with task means, energy, tardiness and
miss counts. AlwaysMEC has direct-MEC fraction one and no V2V workload,
interruption or fallback on nonempty episodes. All 20 stored mean/VaR/CVaR
summaries were recomputed; at alpha .9 and 20 equal-weight episodes, CVaR
matches the upper two episode costs. This small tail remains imprecise.
All 16 stored fixed-policy-versus-AlwaysMEC bootstrap summaries were also
recomputed using the declared explicit bootstrap seeds. Those intervals
concern fixed-policy episode differences; they are not learned-agent seed
uncertainty or significance evidence.

| Recorded reference executions | Episodes | Offered task instances | Completed task instances | Deadline misses | Fallback tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 100 | 1800 | 1800 | 9 | 35 |
| high_load | 100 | 7345 | 7345 | 362 | 220 |
| fast_servers | 100 | 1800 | 1800 | 8 | 35 |
| harsh_channel | 100 | 1800 | 1800 | 10 | 69 |
| Total | 400 | 12745 | 12745 | 389 | 359 |

All 12745 offered task instances eventually completed. Deadline misses and
fallback are distinct from incomplete tasks. Each profile/case is executed
once per fixed policy; repeated tasks and validation cases across policies,
models and budgets are not independent samples. These counts are integrity
and accounting diagnostics, not a learned-versus-greedy performance result.
Maximum absolute episode cost conservation residual is
`4.263256414560601e-14` with
`rel_tol=1e-10` and `abs_tol=1e-10`.

Reference action histories and raw final per-task ledgers are not archived,
so the auditor cannot independently replay policy actions or recover the
three physical objective components from aggregate metrics. No real policy
evaluation, environment step, learner initialization, optimizer update or
held-out trajectory was executed locally. Static scenario reconstruction
and seeded statistical recomputation are distinct from trajectory replay.
Original source, reports and journals remain unchanged.

## Launcher error resolution and remaining work

The [handoff record](colab_budget_references_handoff_20261128.md) documents the
external launcher's false post-save rejection. The received report is exactly
the existing runner's JSON serialization. Restoring the actual pinned
protocol's tuples reproduces Python-object inequality while yielding the
same serialized bytes. Thus the real file confirms the diagnosed tuple/list
comparison error; it does not require rerunning the 400 evaluations. Neither
the scientific runner nor saved evidence was modified to bypass a check.

| Local audit artifact | SHA256 |
| --- | --- |
| receipt.json | `4032369a26d4940f758fa7c48e57d72c503768566f87978c79fea6448431413d` |
| reference_audit.json | `8f2bf45a6dc518acfae19974512bd034eb7e27946f9ca0a79b2bcd2d40ad9cd0` |
| study_progress.json | `f71d8b56573cc97444af32e5866eb2a1df7328a7327ebfb780c8f54614fecbaf` |

The new progress record is `executions_verified_awaiting_analysis`.
All twelve chains and references are verified, with total physical executions
9280. This merged progress file is not the complete-grid performance summary;
its cost curves and comparisons remain null. Historical per-chain evidence
and progress records are preserved as milestone snapshots.

Only documentation and ignored handoff/audit artifacts changed. The earlier
573-test preparation suite was not rerun for this received-data audit.
The external launcher correction has its separate tuple-bearing regression
and guard checks; no scientific-source change occurred. Both READMEs reflect
the verified receipt and next step. Commit and push these records with Amir's
authorization before complete-grid collection and interpretation across all
budgets, representations, algorithms, seeds and profiles. Calibration,
adapted paper/query-free MEC baselines, a frozen paper protocol and explicitly
announced main training remain later work.
