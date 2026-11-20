# Optional greedy behavior warmup for QR-DQN and CVaR

This experiment changes early training data collection while preserving the
QR-DQN + CVaR learner and its deployed decision rule. In the optional
`greedy_warmup` variant, the existing cached combined-cost greedy rule chooses
actions during the first two of the same 20 training episodes. Normal learned
epsilon behavior then continues for the remaining 18 episodes. Evaluation
always uses the learned masked argmin J on each User Vehicle's local input.
The ordinary `standard` variant remains available and remains the default of
the existing runners. MEC continues to provide information only.

The distinction matters for describing the method: this is an optional
**training-behavior variant**, not the identical original training recipe,
weight initialization, behavior cloning, or a greedy/learned deployment switch.
It does not become part of the final paper method automatically.
The [executed development comparison](behavior_warmup_validation_20261120.md)
preserved the standard learner states exactly and did not support adopting
this warmup setting into the main method.

## Method boundaries

| Component | Standard | Optional greedy warmup |
| --- | --- | --- |
| Training episodes 0 and 1 | Existing learned epsilon behavior | Causal cached combined-cost greedy behavior |
| Training episodes 2 through 19 | Existing learned epsilon behavior | Existing learned epsilon behavior |
| Network and quantile outputs | Existing QR network | Same network |
| Actual cost and reward | Cost model 2; reward is negative cost | Same actual costs and reward |
| Replay and optimizer updates | Uniform replay, existing TD/quantile loss | Same rules and total update budget |
| Risk settings | alpha=.9, QR-CVaR lambda=.5 | Same settings |
| Evaluation/deployment | Learned masked argmin J, epsilon=0 | Learned masked argmin J, epsilon=0 |
| Actor information | Own received snapshot/task/user state | Same local actor information |

Physical task cost remains `.60*T/D + .25*E_mobile/E_budget + .15*M`.
There is no imitation loss, expert action label, margin, priority replay,
reward bonus, altered deadline penalty, query action, local execution, task
splitting, or additional optimizer phase. Sigmoid remains a read-only
preference diagnostic. The teacher's contact guard never replaces the
received mask used by the learner or target selector.

QR-DQN learns quantile return estimates through distributional TD targets;
see [Dabney et al.](https://arxiv.org/abs/1710.10044). Here, actual transitions
collected by a different training behavior feed the same existing cost target.
This is an implementation choice for an empirical study, not a convergence or
CVaR-optimality theorem for stale local observations. Demonstration-based deep
Q-learning has precedents such as [Hester et al.](https://arxiv.org/abs/1704.03732),
whose DQfD also uses supervised classification and prioritized replay. Those
additional algorithmic changes are not implemented by this experiment.

## Causal information and physical accounting

The teacher is exactly the existing `MinimumEstimatedCombinedCost` class.
It receives one current user's immutable received raw observation at a time
and declared static objective/radio/energy constants matching the common
baseline. It cannot inspect live remote queues, future channel draws, future
arrivals, ex-post branch winners or validation outcomes. The ordinary query,
snapshot lifetime, action validity, transmission, CPU, energy and recovery
models continue to determine every actual outcome.

The adapter records actual elapsed time and mobile energy, including query
prefix, queueing, discarded V2V work, MEC restart and drain. The teacher's
projected score is used only to choose its training action; it never becomes
a cost target or a fabricated expert label. Individual/team return scopes
remain separately typed. Physical episode cost remains the unique task sum,
not a sum of overlapping team actor views.

Runtime rejects a teacher passed to evaluation before resetting the episode
or mutating model/RNG state. Validation calls never receive a teacher. The
standard route still calls `agent.act`; the optional route is used only for
the declared training prefix and logs its action source explicitly.

## Equal budgets and exploration clocks

The immutable `BehaviorWarmupProtocol` schema 1 wraps the unchanged paired
return-scope plan. The grid is two behavior variants x two return scopes x
QR/QR-CVaR x two agent seeds: 16 models. Every model receives the same 20
nominal training scenes once each and four validation cases in each of four
profiles. Five fixed policies run once per validation scene, not per variant.
The default totals are 320 training, 256 learned validation and 80 fixed
validation episodes: 656 physical executions.

Teacher episodes **replace** the corresponding standard episodes; there are
zero extra teacher episodes, offline replay passes or optimizer updates.
Offered-task, transition and update counts must match across every variant.
All methods use the same initial weight streams, shared hyperparameters,
physical inputs and exogenous scene/channel seeds. Output preserves failures
as errors rather than excluding difficult tasks or seeds.

Physical training time advances during teacher behavior, including empty
slots and drain. Epsilon decay is not restarted after the teacher phase.
Teacher actions are deterministic under their received information, so action
records identify `scheduled_epsilon_applied=false` separately from the
scheduled epsilon value. They do not pretend the usual epsilon lottery ran.

The teacher route skips `agent.act`, so it does not consume that learner's
exploration RNG draws. Later exploration therefore begins at a different draw
position from the standard arm; policy-dependent drain can also change the
physical clock. These are consequences of this training-behavior intervention,
not perfectly aligned later exploration trajectories. Common seeds do not
imply identical actions, replay contents, queues or observations.

## Execution and checkpoint identity

```text
python -m experiments.run_behavior_warmup_audit --output-dir reports/NEW-behavior-warmup-pilot
```

The CLI accepts explicit protocol/scenario files and requires a new output
directory. It has no test access, adaptive stopping, main-training launch or
sweep-resume option. The original learning and return-scope configurations
remain unchanged. A strict nested fingerprint binds the optional protocol,
the paired/shared learning budgets and the physical scenario plan.

Final checkpoints are saved before frozen validation under
`checkpoints/<variant>/<scope>/<algorithm>-seed-<seed>-final.pt`. Metadata
records variant, exact teacher parameters, teacher episode boundary, actual
teacher action count, completion clock, post-teacher phase, source/protocol
hashes and initial/final model hashes. The experiment loader rejects a wrong
variant, phase, scope or protocol. Individual/team checkpoint containers retain
their existing separate versions. The deployment model does not call a teacher.

`report.json` contains complete training/validation traces and baseline metrics.
`summary.md` retains per-model cost risk and matched `greedy_warmup - standard`
episode-cost differences, with negative differences favoring warmup. Seeds and
known shifted profiles remain separate. Four cases at alpha=.9 imply a tail
mass of .4 episodes: empirical CVaR is the largest observed episode cost.
Two seeds describe variation conditional on one fixed cohort, not reliable
population uncertainty or significance.

Main training remains reserved for Colab with notice to Amir before execution.
Any later adoption of this optional collection protocol needs an explicit
method description and supporting comparisons; a favorable development cell
does not replace the original method or justify selecting a preferred seed.
