# PitStopRL

An original reinforcement-learning mini-project for **variable-length F1-style pit strategy**. A contextual **Double DQN** learns whether to stay out or pit for soft, medium, or hard tyres. Its neural network predicts four action values from the current race and its settings. Training uses simulator interaction only: no expert labels, imitation, search planner, baseline demonstrations, forced pit decisions, or pit-advantage override.

The simulator is a controlled teaching model, not a real F1 digital twin.

## Install

Python 3.10+ is required. The published run used Python 3.12, PyTorch 2.9.0, and NumPy 2.3.4 on CPU.

```sh
git clone https://github.com/harshitdeora/PitStopRL.git
cd PitStopRL
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```sh
python -m pip install -e ".[dev]"
```

Runtime dependencies: NumPy, Matplotlib, and PyTorch. Install with `pip install -e .` if tests/lint are not needed. No Gymnasium, RL framework, telemetry service, GPU, or background workers are required.

## Run the trained agent

```sh
python -m pitstoprl.demo --seed 2000000
python -m pitstoprl.demo --laps 66 --wear-multiplier 1.2 --pit-loss 29 --sc-probability 0.06 --seed 2000000
python -m pitstoprl.demo --laps 90 --seed 2000001
```

The default model is `artifacts/dqn.pt`. The demo prints total time, pit/safety-car events, and whether the two-compound rule was met. Starting fuel scales with race length. The simulator accepts other positive horizons, but this is **not a promise of generalization outside the tested domain**.

## RL formulation

One step completes one lap. Every race starts on medium tyres. The four actions are `0 = stay out`, `1 = pit soft`, `2 = pit medium`, `3 = pit hard`.

### State: 27 scalar features

The full ordered schema and scaling factors are `RaceEnv.OBSERVATION_NAMES` in `pitstoprl/env.py`. Features are supplied to the network without a discrete state lookup:

| Indices | Features |
| --- | --- |
| 0–7 | Lap progress, compound, tyre age, wear, fuel fraction, safety-car flag, remaining safety-car laps, compound-change flag |
| 8–14 | Actual laps remaining, total race length, pit cost, wear multiplier, safety-car probability, safety-car pit discount, base lap time |
| 15–16 | Wear squared and cubed, representing nonlinear degradation |
| 17–26 | Starting fuel, fuel burn, fuel-mass penalty, safety-car lap delta, lap-noise level, rule penalty, reward scale, minimum/maximum safety-car duration, reward reference |

Lap counts are scaled by a constant 100, not inferred by multiplying progress by 57. Wear is not clipped: distinct degradation states remain distinguishable. Scenario parameters are explicitly observed; this is not hidden-parameter adaptation. The environment exposes safety-car time remaining, a simplifying assumption unavailable exactly in real races.

### Dynamics and reward

Lap time combines base pace, deterministic fuel-mass cost, compound grip, nonlinear wear cost `1.8*w + 11*w**3`, random lap noise, safety-car slowdown, and pit loss. Tyres wear at compound-specific rates, multiplied by the scenario's wear multiplier and lap noise in wear. A pit resets tyre age/wear. A safety car reduces pit loss and tyre wear. Failing to use a second compound adds a terminal 30-second cost; it is a **soft penalty, not a guaranteed constraint**.

For DQN training, define controllable time cost:

```text
c_t = compound_grip + 1.8*w + 11*w^3 + pit_loss + terminal_rule_penalty
r_t = -c_t / 10
gamma = 1
```

The remainder of race time (base pace, fuel mass, safety-car slowdown and lap-time noise) is independent of the policy under this simulator's paired exogenous randomness. For a fixed configuration and seed:

```text
race_time = sum(uncontrollable_time) - 10 * sum(training_reward)
```

Thus maximizing expected training return minimizes expected total race time. Removing the action-independent offset reduces target variance without adding expert rules or baseline-derived rewards. A test verifies this equality across different strategies. `RaceEnv.step()` also returns the raw-time reward `(reward_reference - lap_time - terminal_penalty)/reward_scale`; the DQN loop explicitly uses `info["controllable_cost"]` instead. Training traces store the centered DQN reward; evaluation reports actual elapsed time.

### Algorithm

A 27 → 128 → 128 → 4 ReLU network estimates Q-values. Double DQN uses the online network to select the bootstrap action and a frozen target network to evaluate it. Training uses epsilon-greedy exploration, a 100,000-transition replay buffer, random batches of 256, Huber loss, Adam at 0.0003, gradient clipping at 10, and a hard target update every 250 optimization steps. Gamma is 1 with a finite, observed horizon. Sixteen in-process environment instances share batched inference; there is one CPU thread and no worker processes.

## Training domain and held-out tests

Each training episode samples a new configuration:

| Setting | Training range |
| --- | --- |
| Race length | Integers 20–90, **excluding 53, 66, and 80** |
| Tyre-wear multiplier | 0.70–1.30 |
| Normal pit loss | 12–32 seconds |
| Safety-car onset probability | 0–0.07 per eligible lap |
| Safety-car pit-loss factor | 0.45–0.70 |
| Base lap time | 75–100 seconds |
| Lap-noise standard deviation | 0.10–0.40 seconds |
| Fuel | 1.8 kg per configured lap |

Training environment seeds are 7–30006. Fixed validation races use seeds 1000000–1000063, with a separately seeded configuration generator. The saved checkpoint is selected only by the lowest mean seconds per lap over those fixed validation races. The training trace includes every validation score and identifies the selected episode; the last training iterate is not automatically the best checkpoint.

Final evaluation uses seeds 2000000–2000999 and 13 predeclared cases: the original 57-lap setting; 30/53/66/80/90 laps; lower/higher wear; cheaper/more expensive pits; no/more safety cars; and a held-out combination of length and conditions. **53, 66, and 80 laps occur in neither training nor validation.** No final-test results are used to choose the model. CLI evaluation rejects reserved/overlapping seed ranges.

Every policy receives the same lap noise, wear-noise draws and safety-car schedule for a given seed/configuration. Pairing is checked during evaluation. Comparisons use an evaluation-only fixed medium-to-hard stint and a wear-threshold rule. Both rule baselines use actual race length. They never generate training targets or override DQN actions.

## Reproduce

```sh
python -m pitstoprl.train --episodes 30000 --seed 7 --env-count 16 --validate-every 2000
python -m pitstoprl.evaluate --checkpoint artifacts/dqn.pt --episodes 1000 --seed 2000000 --output-dir results/dqn
pytest
ruff check .
```

Training replaces `artifacts/dqn.pt`, `dqn_training.npz`, and `dqn_training.json`. Use `--output` to preserve the published checkpoint. Checkpoints contain inference weights and provenance, not a resumable optimizer/replay state. CPU reproducibility is best effort across different PyTorch/platform versions; exact package versions and the checkpoint hash are recorded with the results.

## Results and limitations

Current DQN results live in `results/dqn/`: per-race CSV, scenario/strategy summary, protocol and configuration metadata, reference strategy comparison, generalization plot, sample stint timeline, and training curve. See `results/dqn/RESULTS.md` for the published comparison. Positive `delta_vs_agent_seconds` means the comparator is slower than DQN. Paired 95% intervals use the normal approximation and are not adjusted for multiple comparisons.

The published benchmark contains **39,000 races**: 1,000 paired seeds × 13 scenarios × 3 policies. DQN was faster than both baselines on average in every published scenario, and all 13,000 sampled DQN races met the compound rule. These observations are limited to the tested settings and seeds. No submission report is generated by this repository.

This model represents one car on a dry track. It omits opponents, traffic, rain, tyre temperatures, stint-dependent compound allocation and circuit calibration. Fuel sufficiency is checked when constructing a configuration. Simulator support for arbitrary positive lap counts does not imply the trained network is reliable at arbitrary lengths or on real circuits. Published results are synthetic generalization tests, not real F1 performance claims.

## Repository layout

```text
pitstoprl/env.py          contextual race environment and configuration
pitstoprl/dqn.py          network, replay buffer, Double DQN and safe loading
pitstoprl/domain.py       training domain and predeclared held-out cases
pitstoprl/training.py     randomized training and validation-only selection
pitstoprl/policies.py     evaluation-only baselines
pitstoprl/train.py        training CLI
pitstoprl/evaluate.py     paired benchmark and plots
pitstoprl/demo.py         configurable demo CLI
artifacts/               trained models, training traces and provenance
tests/                   environment, learner and integration tests
results/dqn/             current generalization results
```

## References

- Course lectures 20–24: MDPs, Q-learning, approximate Q-learning, DQN, replay and target networks.
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.
- van Hasselt, Guez and Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI, 2016.

MIT license. Standard RL algorithms are implemented for this project; no external RL project or tutorial implementation is vendored.
