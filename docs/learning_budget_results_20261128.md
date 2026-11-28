# Complete development learning-budget results 2026 11 28

The completed synthetic study does **not** show an advantage over cached combined-cost greedy. At the fixed final budget of 500 episodes, all 48 model/profile rows have higher mean episode-total cost and higher empirical CVaR .9 than the matched greedy reference. The same direction holds for all 144 model/budget/profile rows across budgets 20, 100 and 500. These are correlated descriptive rows, not 144 independent trials.

Log-SNR reduces the gap relative to the linear representation for QR-CVaR in the across-seed means for every profile, but neither representation beats greedy. CVaR changes the QR results in a profile-dependent way: in log-SNR/high-load it lowers both mean and tail cost relative to neutral QR for all three seeds, while log-SNR/nominal slightly increases mean model CVaR and log-SNR/fast-servers increases mean cost. This is not evidence of consistent risk improvement.

All 9280 planned physical executions are retained: 6000 training, 2880 learned validation and 400 shared fixed-policy references. Collection is complete, but the physical parameters remain synthetic and uncalibrated. Main training, held-out paper evaluation and any final representation decision remain unstarted.

## Metric and comparison contract

Each risk row uses 20 equally weighted full-drain episode-total costs for one model, budget and profile. CVaR .9 consequently averages the upper two episode costs. Cost is dimensionless and includes every task in the episode; it is not per-task cost. Profiles differ in workload and are compared to their own matched baseline, without pooling profile costs.

Tables marked mean ± SD average the three separate model statistics at learner seeds 11, 29 and 47; SD is sample standard deviation across those seeds conditional on the same traces. In particular, the displayed mean CVaR is the arithmetic mean of three model CVaRs, **not** CVaR of pooled policy/seed episodes. It is not a confidence interval or significance result. A positive percentage gap is worse than greedy: `100 * (model statistic / greedy statistic - 1)`.

Latency and mobile energy are task-weighted, in ms and mJ/task. Miss, direct-MEC and fallback percentages are task fractions. SV workload is an average of nonempty-episode CPU-workload fractions, not a pooled CPU-cycle fraction. Workload and destination counts are different quantities. The [complete tables](learning_budget_results_tables_20261128.md) retain all 144 learned rows and all 20 fixed-policy/profile rows.

Independent User Vehicle decisions, information-only MEC, individual returns, no local/partial offloading, ordinary learned-epsilon behavior and plain EDF remain in effect. Cost is `z=.60*T/D+.25*E_mobile/E_budget+.15*M` with no separate deadline-miss penalty. Primary QR-CVaR retains alpha .9 and effective risk weight .5; the QR control has effective weight zero. Strong greedy, model controls, budgets and scenarios were not changed to obtain these results.

## QR-CVaR at the predeclared final budget

Both representations are shown; neither a seed nor a checkpoint is selected. The default remains `linear_v3`, and log-SNR remains a candidate.

| Profile | Representation | Greedy mean | QR-CVaR mean ± SD | Mean gap % | Greedy CVaR | Mean model CVaR ± SD | CVaR gap % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | linear_v3 | 3.6237 | 4.3485 ± 0.2028 | +20.00 | 5.7895 | 7.1400 ± 0.6423 | +23.33 |
| nominal | log_snr_v4 | 3.6237 | 4.0500 ± 0.1269 | +11.76 | 5.7895 | 6.5363 ± 0.2783 | +12.90 |
| high_load | linear_v3 | 18.8802 | 23.8548 ± 1.6770 | +26.35 | 27.8467 | 36.3799 ± 4.3417 | +30.64 |
| high_load | log_snr_v4 | 18.8802 | 20.0701 ± 0.7923 | +6.30 | 27.8467 | 30.8280 ± 2.2024 | +10.71 |
| fast_servers | linear_v3 | 3.6645 | 4.3691 ± 0.1542 | +19.23 | 5.8294 | 6.9722 ± 0.6743 | +19.60 |
| fast_servers | log_snr_v4 | 3.6645 | 4.1687 ± 0.1038 | +13.76 | 5.8294 | 6.7337 ± 0.1255 | +15.51 |
| harsh_channel | linear_v3 | 3.6741 | 4.3555 ± 0.1478 | +18.55 | 5.8728 | 7.1297 ± 0.4778 | +21.40 |
| harsh_channel | log_snr_v4 | 3.6741 | 4.0928 ± 0.1335 | +11.40 | 5.8728 | 6.5528 ± 0.1826 | +11.58 |

For linear QR-CVaR, the mean-cost gap ranges from +18.55% to +26.35%, and the mean-of-model-CVaRs gap from +19.60% to +30.64% across profiles. For log-SNR QR-CVaR those ranges are +6.30% to +13.76% and +10.71% to +15.51%. These are ranges of profile-specific seed averages, not uncertainty intervals.

## All fixed training-budget points

These are across-seed averages, with the full per-seed values preserved in the linked tables. Checkpoints are continuing states of the same models and are correlated. Training is not extended or stopped based on these observations.

| Profile | Representation | Algorithm | Mean 20 | Mean 100 | Mean 500 | Mean model CVaR 20 | CVaR 100 | CVaR 500 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | linear_v3 | qr_dqn | 4.3589 | 4.4809 | 4.3609 | 7.5646 | 7.7716 | 7.3130 |
| nominal | linear_v3 | qr_dqn_cvar | 4.3718 | 4.4417 | 4.3485 | 7.6148 | 6.9979 | 7.1400 |
| nominal | log_snr_v4 | qr_dqn | 4.2942 | 4.1484 | 4.0666 | 6.8349 | 6.7596 | 6.5159 |
| nominal | log_snr_v4 | qr_dqn_cvar | 4.3381 | 4.2433 | 4.0500 | 6.8048 | 6.7034 | 6.5363 |
| high_load | linear_v3 | qr_dqn | 23.1752 | 23.3139 | 23.5852 | 37.1476 | 35.4991 | 37.6070 |
| high_load | linear_v3 | qr_dqn_cvar | 23.6343 | 24.1759 | 23.8548 | 35.3631 | 37.0299 | 36.3799 |
| high_load | log_snr_v4 | qr_dqn | 23.7501 | 20.9269 | 21.1544 | 35.2315 | 29.4263 | 31.9124 |
| high_load | log_snr_v4 | qr_dqn_cvar | 22.2203 | 21.3072 | 20.0701 | 34.1302 | 30.0468 | 30.8280 |
| fast_servers | linear_v3 | qr_dqn | 4.3875 | 4.4406 | 4.3810 | 7.6281 | 7.4133 | 7.2945 |
| fast_servers | linear_v3 | qr_dqn_cvar | 4.3748 | 4.4471 | 4.3691 | 7.4823 | 6.9330 | 6.9722 |
| fast_servers | log_snr_v4 | qr_dqn | 4.4676 | 4.2209 | 4.1141 | 7.0791 | 6.7424 | 6.8758 |
| fast_servers | log_snr_v4 | qr_dqn_cvar | 4.4770 | 4.3376 | 4.1687 | 6.9995 | 6.8248 | 6.7337 |
| harsh_channel | linear_v3 | qr_dqn | 4.3307 | 4.4953 | 4.3071 | 7.4028 | 7.6290 | 7.0934 |
| harsh_channel | linear_v3 | qr_dqn_cvar | 4.3411 | 4.4227 | 4.3555 | 7.5056 | 6.7980 | 7.1297 |
| harsh_channel | log_snr_v4 | qr_dqn | 4.3508 | 4.1479 | 4.0676 | 6.8144 | 6.8996 | 6.7080 |
| harsh_channel | log_snr_v4 | qr_dqn_cvar | 4.3497 | 4.2725 | 4.0928 | 6.8757 | 6.7435 | 6.5528 |

Log-SNR QR-CVaR reduces mean cost from budget 20 to 500 in all twelve seed/profile combinations. Tail changes are less uniform: high-load CVaR decreases for only one of its three seeds even though the across-seed average falls. Linear QR-CVaR has small or inconsistent mean changes; high-load mean and mean model CVaR are higher at 500 than 20 in the across-seed average. Three fixed budget points do not establish convergence, and neither earlier checkpoints nor favorable profiles are substituted for the fixed final result.

## Latency, mobile energy and MEC offload

Learned rows below are three-seed means at budget 500. The existing greedy reference appears once per profile.

| Profile | Method | Latency ms | Mobile mJ/task | Deadline miss % | SV workload % | Direct MEC tasks % | Fallback tasks % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | combined-cost greedy | 122.569 | 4.957 | 0.000 | 87.481 | 9.167 | 1.944 |
| nominal | linear_v3 QR-CVaR | 133.979 | 4.443 | 2.407 | 76.983 | 19.537 | 3.426 |
| nominal | log_snr_v4 QR-CVaR | 124.961 | 4.546 | 1.481 | 79.597 | 17.407 | 1.759 |
| high_load | combined-cost greedy | 164.519 | 5.101 | 1.225 | 76.737 | 16.746 | 4.357 |
| high_load | linear_v3 QR-CVaR | 188.233 | 4.149 | 4.947 | 56.430 | 40.526 | 3.358 |
| high_load | log_snr_v4 QR-CVaR | 150.854 | 4.261 | 2.519 | 60.213 | 35.625 | 2.700 |
| fast_servers | combined-cost greedy | 123.233 | 4.865 | 0.000 | 85.397 | 9.722 | 2.500 |
| fast_servers | linear_v3 QR-CVaR | 133.811 | 4.430 | 2.037 | 77.393 | 21.111 | 2.500 |
| fast_servers | log_snr_v4 QR-CVaR | 119.761 | 4.118 | 1.111 | 68.850 | 27.685 | 2.037 |
| harsh_channel | combined-cost greedy | 120.263 | 4.886 | 0.000 | 82.615 | 10.278 | 5.000 |
| harsh_channel | linear_v3 QR-CVaR | 129.755 | 4.297 | 1.574 | 72.546 | 21.667 | 5.370 |
| harsh_channel | log_snr_v4 QR-CVaR | 122.249 | 4.541 | 1.574 | 76.392 | 18.889 | 3.241 |

QR-CVaR mobile energy is lower than greedy in these across-seed averages, and log-SNR/high-load and fast-servers also have lower task-weighted latency. Those facts do not imply a lower objective: the objective uses per-task deadline/energy normalization and the MEC term, rather than one fixed weight times the displayed average latency or energy. Final task ledgers are absent, so exact objective-component attribution is not established by this summary.

Both QR-CVaR representations have a smaller mean SV-workload fraction than greedy in every profile. Thus the present results do not meet the intended relative goal of shifting more work to Server Vehicles than this baseline. All recorded episodes drain, but eventual completion does not imply meeting deadlines; deadline misses and fallback remain explicit outcomes.

## Matched CVaR-versus-neutral-QR ablation at 500

Deltas are QR-CVaR minus neutral QR within the same representation, seed and profile; negative favors QR-CVaR. SD refers to the three paired seed differences. The counts describe observed seeds and are not significance tests.

| Profile | Representation | Mean-cost delta ± SD | Seeds with lower mean | Model-CVaR delta ± SD | Seeds with lower CVaR |
| --- | --- | ---: | ---: | ---: | ---: |
| nominal | linear_v3 | -0.0124 ± 0.1579 | 2/3 | -0.1730 ± 0.6788 | 1/3 |
| nominal | log_snr_v4 | -0.0167 ± 0.2276 | 2/3 | +0.0203 ± 0.0431 | 1/3 |
| high_load | linear_v3 | +0.2696 ± 1.2814 | 1/3 | -1.2271 ± 9.9584 | 2/3 |
| high_load | log_snr_v4 | -1.0843 ± 1.0858 | 3/3 | -1.0844 ± 0.6633 | 3/3 |
| fast_servers | linear_v3 | -0.0118 ± 0.1328 | 1/3 | -0.3223 ± 0.6759 | 2/3 |
| fast_servers | log_snr_v4 | +0.0546 ± 0.2686 | 2/3 | -0.1422 ± 0.3511 | 2/3 |
| harsh_channel | linear_v3 | +0.0484 ± 0.0979 | 1/3 | +0.0363 ± 0.5343 | 1/3 |
| harsh_channel | log_snr_v4 | +0.0252 ± 0.1905 | 1/3 | -0.1553 ± 0.2034 | 3/3 |

The log-SNR/high-load result is a useful conditional observation: QR-CVaR improves both statistics relative to neutral QR in all three seeds, yet remains above greedy. Log-SNR/harsh-channel lowers CVaR in all three seeds while mean cost is slightly higher on average. These outcomes support examining the tradeoff, not claiming uniform superiority or silently changing the approved risk weight.

## What to do next

1. Preserve this completed study and its negative greedy comparison. Do not append episodes to its 500-episode budget, choose the best intermediate checkpoint, drop a seed, weaken greedy or adjust objective weights to produce a win.
2. Predeclare a training-only checkpoint diagnostic across all twelve final models and the same fixed training replay/episode indices. Check observation scales, finite quantiles and action masks, predicted versus bootstrap-target magnitude, quantile calibration/crossing, update/gradient behavior and whether risk scoring changes actions. The existing [training-input audit](training_input_validation_20261121.md) provides the earlier numerical finding; this study has not proved it explains the whole remaining gap. Keep all deployment information local and causal.
3. Treat the individual-return versus global episode-cost issue and physical-model calibration as explicit research questions. Shared resources can create external costs, but these summaries do not establish that as the cause. Independent decisions and the approved individual-return method remain unchanged; no team return, centralized MEC assignment, teacher, imitation or reward shaping is introduced by this analysis.
4. Use diagnostic evidence to predeclare a separate matched development hypothesis and, where justified, a new versioned experiment. Calibrate physical parameters with provenance and retain adapted paper/query-free MEC baselines before freezing the paper protocol. Notify Amir before any main-training Colab run. No new training or configuration change has been launched here.

## Provenance and verification

The unchanged `experiments.learning_budget_results.summarize_learning_budget` collector returned `status=complete`, with 144 learned model/budget/profile rows, 20 fixed-policy/profile rows and all three representation comparisons. The declared grid, cohort, contracts, frozen validation and artifact hashes passed its existing checks.

Original archives remain separate. A fresh canonical-layout directory under ignored `reports/learning-budget-analysis-5d8845b-20261128/` uses 1920 NTFS hard links to the verified learning journal files to avoid duplicating checkpoints, plus an exclusive byte copy of the reference report. After assembly the layout was only read. No file bytes or attributes were edited through these aliases; all original journals were rehashed again after collection. Link counts changed by construction. Do not use this collection layout as a training destination.

The independent analysis rechecked all 144 learned and 20 reference means and upper-two CVaRs from raw episode records, task-weighted latency/energy/miss rates, workload fractions and paired greedy mean differences. Seed-level statistics were computed only after retaining all declared rows. The audit and analysis loaded no learner tensors, generated no trajectories, performed no optimizer updates and accessed no held-out test trajectory. No hypothesis test, new bootstrap interval or pooled-policy risk was added.

| Identity / evidence | SHA256 |
| --- | --- |
| Pinned scientific commit | `5d8845bb0605def86541fd172a961be5211a7ba0` |
| Python source | `2abe684318cd33323bb2e94a5db41a2bef7e974028fc56355a607b8bde8a50a6` |
| Learning-budget protocol | `bbcf8f210094b25494dbe4be637a935ac697d17a225653ba66420b67c549aefb` |
| Scenario protocol | `0ddfa2cb6e9fd6c08362d356f089d4f3af611cb423b8cdb5b9b0d59b45bbac22` |
| Reference report | `70b55ea7f2167c46cab9d2524f968a41375b2f0cf42043cd4b8f67090451d274` |
| complete_grid_summary.json | `8578182aa95dcbe7526d4c3ac0312869b7f625a32d30a20dee7385da8b5b6315` |
| collection_evidence.json | `d0bc07b371fb4c28167ebf1a1d1b543888bd1de835ae1f43bb9ac7b5d45c6080` |
| descriptive_analysis.json | `2c89b048c8d23d5bf69db34888041cbdea0af9a0384a4196f8bbab0b8d18b591` |

Producer execution used the exact recorded Linux CPU identity `f42fedce041862a177f3cd7fb86fefbd53557342af970b898cfd86e95f5b8463`; read-only collection used the local Windows auditor. This does not claim cross-platform bitwise training agreement. The earlier 573-test preparation suite was not rerun for data collection/reporting; current verification consists of journal/record/metric/accounting and aggregation checks described above.

Only documentation and ignored collection/analysis artifacts changed. Keep [the fixed study protocol](learning_budget_development.md), [chain verification](colab_budget_chain_12_20261128.md) and [reference verification](colab_budget_references_verification_20261128.md) as the evidence trail. Commit and push this milestone with Amir's authorization before implementing the next declared diagnostic.
