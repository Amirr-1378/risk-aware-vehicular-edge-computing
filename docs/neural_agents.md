# Neural Cost Agents and Reproducible Training Audit

Status: CPU DQN, Double DQN, risk-neutral QR-DQN, and a mean-CVaR-distorted
QR controller are implemented with numerical tests, replay, greedy evaluation,
and complete episode-boundary checkpoints. This is an implementation milestone,
not a calibrated performance study or evidence of Q1 publication readiness.

## Dependency boundary and execution

The simulator, encoders, replay, schedules, and analytical baselines remain
standard-library-only. PyTorch is imported explicitly from
`risk_aware_v2x.agents.neural`, not when importing `risk_aware_v2x.agents`.
Install the optional `learning` extra for neural training. The tested Windows
Python 3.12.14 environment uses PyTorch 2.14.1+cpu and NumPy 2.5.4.

Example PowerShell setup from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
.\.venv\Scripts\python.exe -m pip install -e ".[learning]"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The dependency ranges in pyproject are compatibility ranges, not a complete
paper-release lockfile. Reports capture all installed package versions,
Python/platform/machine, CPU/thread settings, seeds, and scientific
configuration. Source provenance records Git revision/dirty status and a hash
of actual Python sources, including uncommitted changes. Cross-release,
cross-platform, and CPU/GPU bitwise agreement
is not promised; see [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
No GPU execution is implemented in this milestone.

## Common architecture and information

`NeuralCostAgent(config, encoder, seed=...)` uses a fully connected ReLU network
with output `[batch, actions, atoms]`. There is no softmax over costs and no
output clipping. DQN/Double DQN have one scalar per action; QR agents have N
equal-mass atoms per action. Default hidden widths are 128/128, N=51, Adam
learning rate 1e-3, and Huber threshold 1. These are implementation defaults,
not tuned experimental recommendations.

All variants use the same encoded received observations and cached masks.
The network does not receive simulator truth, future traces, identifiers, or
per-task ledgers. A single parameter-shared policy serves all users. Each
slot's full user action batch is chosen before service or learning updates.
The observation is still partially observed; the fully observed MDP theory
of the source algorithms is not automatically a theorem for this simulator.

Default replay capacity is 10,000, batch size 32, and warm-up 128 stored
transitions. Sampling is uniform without replacement from a fixed-capacity
ring. Records preserve their original current/next masks and variable slot
durations. Current action validity, finite immutable features, terminal
consistency, and dimensions are validated. Agent ingestion also checks that
the bootstrap discount matches its configured physical-slot gamma.

Initialization uses a seeded isolated Torch CPU RNG context; the caller's
Torch RNG state is restored. Replay and per-user exploration have independent
seeded Python streams. No global Python/NumPy seeding is used. There is no
dropout or stochastic forward operation. Greedy evaluation consumes neither
exploration nor replay RNG and does not learn.

## Cost Bellman targets and updates

The sign convention is **cost minimization**, consistently for behavior and
targets. Let c-bar be the accumulated slot-discounted interval cost and d be
gamma-slot**duration, or zero at a terminal transition. They come from
[`DecisionAdapter`](learning_interface.md), not a fixed one-step decision
discount. Prefix costs before the first action remain in episode metrics;
they are not fabricated into action transitions.

| Variant | Next-action selector | Target evaluator |
| --- | --- | --- |
| DQN | minimum masked target-network scalar cost | target network |
| Double DQN | minimum masked online-network scalar cost | target network |
| QR-DQN | minimum masked target-network mean cost | target network atoms |
| QR-DQN + CVaR | minimum masked target-network mean-CVaR score | target network atoms |

The scalar or atomwise target is `c_bar + d * target(next_state, next_action)`.
The Double DQN row separates selection and evaluation, as adapted to cost
minimization from [van Hasselt et al.](https://arxiv.org/abs/1509.06461).
QR variants here use target-network selection, not a hidden Double-QR
modification. Behavior always selects from the online network. Terminal and
zero-discount targets do not select/read a next action; zero-discount samples
are still nonterminal when the adapter says so.

DQN variants minimize batch-mean Huber TD loss. QR uses all predicted/target
atom pairs with midpoint fractions tau_i=(i+.5)/N:

\[
\delta_{bij}=y_{bj}-\theta_{bi},\qquad
L=\frac1B\sum_b\sum_i\frac1M\sum_j
|\tau_i-1_{\delta_{bij}<0}| H_\kappa(\delta_{bij}).
\]

`H_kappa(delta)` is .5*delta**2 inside +/-kappa, and
`kappa*(abs(delta)-.5*kappa)` outside. There is no division by kappa. The loss
sums over predicted quantiles, averages target atoms and batches, and does
not silently average over all three axes. This scaling must be retained when
comparing quantile counts/learning rates. Targets and sign indicators are
detached. Predicted atoms are **not** sorted during training: their fixed
quantile-fraction indices matter. See [Dabney et al.](https://arxiv.org/abs/1710.10044).

Adam uses betas .9/.999, epsilon 1e-8, no weight decay, no AMSGrad, and
non-fused/non-foreach updates. Gradient norm is clipped at 10 by default.
Non-finite loss/gradients/parameters cause explicit errors. The frozen target
is hard-copied every 100 **gradient updates**, not actions or simulator slots;
both networks start identical. Update statistics report pre-clipping norm,
loss, gradient-update count, and synchronization events.

## Meaning of the risk-aware variant

Only `qr_dqn_cvar` uses the risk-mixture weight; the other variants have
effective weight zero. Both behavior and bootstrap action selection minimize
`(1-lambda)*mean + lambda*upper_CVaR_alpha`, nominally alpha=.9, lambda=.5.
Scoring uses the exact fractional tail integral from
[`learning_interface.md`](learning_interface.md). Inference sorts a copy for
tail integration; the QR loss retains original fraction indices. Masked
NaN/optimistic values cannot select a padded action.

This is an explicitly **risk-distorted greedy distributional controller**.
It is not claimed to solve static full-episode CVaR exactly, nor to implement
a nested time-consistent CVaR Bellman equation. That distinction follows the
issue analyzed by [Lim and Malik](https://papers.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract-Conference.html).
Using lambda=0 gives the exact same updates as risk-neutral QR under the same
data/seeds. Learned tail scores are distribution estimates, not safety or
deadline guarantees. Quantile calibration/crossing and tail-estimation quality
still need evaluation in the scientific experiment stage.

## Exploration clock and training/evaluation protocol

`EpsilonSchedule` linearly decays from initial to final epsilon over a
configured number of elapsed **training slots**, saturating exactly at the
final value. Empty slots and final drain advance the clock; evaluation does
not. All users in a slot use the same epsilon. Exploratory actions are uniform
over that user's observed-valid actions, including MEC.

The demonstration runner first stores all transitions closed in a slot, then
requests one gradient update per newly closed transition. Before warm-up it
does no update; in a slot with no new transition it does not repeatedly train
on unchanged replay. Warm-up is counted by total transitions, including the
threshold transition exactly once, independent of simultaneous user closures.
Thus after K stored transitions the demonstration has max(0,K-warmup+1)
updates. These choices and target-sync units are explicit. The
future benchmark must freeze training budgets and exploration schedules rather
than changing them per method after observing results.

Evaluation uses epsilon=0, no replay insertion, no optimizer step, and no
schedule advancement. Training/evaluation seed overlap is rejected. However,
the current smoke scene has deterministic arrivals/mobility/channels. Separate
seed numbers alone do **not** make it a generalization benchmark or independent
tail-risk study. Full costs, including query prefixes and drain, are reported.
Episode lengths are not fixed to the arrival horizon: slow service/fallback
can extend the drain.

## Checkpoints and exact local continuation

Checkpoints contain primitive metadata/tensors, not serialized simulator or
agent objects: configuration, encoder signature/scales/order, online/target
weights, Adam state, replay contents/cursor/RNG, per-user exploration RNGs,
seen-transition and gradient-update counters, runtime versions, and caller
run metadata. The loader uses CPU mapping and `weights_only=True`; load only
trusted local files, as advised by [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.load.html).

Encoder/schema, exact Torch version, replay dimensions, discount contract,
optimizer settings, and finite parameters/moments are checked. Existing files
are never overwritten. The runner additionally validates scenario fingerprint,
physical configuration, seeds, epsilon schedule, and update protocol before
resuming, including the actual Python-source fingerprint. Its metadata includes
the next episode index and physical training
clock. Checkpoints are episode-boundary snapshots; in-flight simulator tasks,
queries, and unfinished decision intervals are **not** resumable mid-episode.

Tests verify the next exploratory action and optimizer update after loading,
as well as 2+2 resumed episodes versus 4 uninterrupted episodes: identical
local weights, replay/RNG state, and final greedy evaluation. This is a tested
property in the fixed CPU runtime, not a cross-platform guarantee.

## Short integration run

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --algorithm qr_dqn_cvar --episodes 4
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --algorithm dqn --episodes 4
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --algorithm double_dqn --episodes 4
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --algorithm qr_dqn --episodes 4
```

Optional save/continue, choosing unused filenames:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --episodes 2 --checkpoint artifacts/learning/first.pt
.\.venv\Scripts\python.exe -m experiments.run_learning_smoke --episodes 2 --resume artifacts/learning/first.pt --checkpoint artifacts/learning/continued.pt
```

`--episodes` means additional episodes after resume. The small runner uses
32/32 hidden widths, N=15 (a fractional 1.5-atom upper tail at alpha=.9),
batch/warm-up 4, replay 256, target synchronization every 5 gradient updates,
and epsilon decay over 200 training slots. Its JSON includes actions, losses,
settings, seeds, complete accounting, and separate evaluation. Checkpoints
belong under ignored `artifacts/` or `checkpoints/`, never in source Git.

Next scientific work: calibrated stochastic scenarios and training/validation/
test partitions, a frozen comparison protocol with independent scenario and
agent seeds, adapted paper and combined-cost baselines, query-free MEC protocol
comparison, uncertainty intervals, tail calibration, and predeclared ablations.
Do not report the short audit as convergence, superiority, or a safety result.

The next infrastructure milestone is now available in
[`development_protocol.md`](development_protocol.md): independent stochastic
scenes, split/seed planning, terminal diagnostics, and paired analytical-policy
audits. A bounded learned multi-seed training/validation pilot on that plan is
now available in [learning_development_protocol.md](learning_development_protocol.md).
Both neural entry points share `experiments/learning_runtime.py`; the neural
updates are unchanged. Empirical calibration and larger final experiments
remain pending. Code changes alter source fingerprints; an old development
checkpoint is not automatically a reproducible training-resume point for a new
source/configuration/encoder contract. Existing files are preserved.
