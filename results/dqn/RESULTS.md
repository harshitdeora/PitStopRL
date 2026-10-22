# Double DQN benchmark with a hard tyre-wear limit

The model was retrained from scratch for environment `wear_limit_v1`. A checkpoint was selected at episode 30,000 from 30,000 training episodes using fixed validation races only. Training used 1,633,334 transitions. No final-test races select the checkpoint.

**39,000 evaluation races:** 1,000 paired seeds (2,000,000–2,000,999) × 13 scenarios × 3 independent policies. All policies receive identical lap noise, wear-noise draws, and safety-car schedules. No model weights change during evaluation.

## Wear safety

- Hard action constraint: worst-case next-lap wear must be at most `1 - 0.000001`.
- Highest training wear: **99.7904%** across all training transitions.
- Highest benchmark wear across all three policies: **99.9755%**.
- Training and benchmark wear-limit violations: **0**. No wear value is clipped.
- Tests also exercise maximum possible wear noise over 300-lap races, unsafe-action rejection, constrained exploration, legal-action Bellman targets, and Python/browser equivalence.

The planned baseline now refreshes medium tyres when necessary before its planned hard-tyre switch at 40% distance, then refreshes hard tyres as needed. The wear-based baseline keeps its 80% trigger and also respects the same physical action constraint. These replace the earlier unbounded-wear benchmark; changes in times reflect a changed environment, baseline and trained model.

## Total race times

Lower mean time is better. Full confidence intervals and paired time differences are in [summary.json](summary.json); intervals use a marginal 95% normal approximation and are not corrected for multiple comparisons.

| Scenario | DQN time (s) | Planned + safety time (s) | Wear-rule time (s) | DQN stops | DQN peak wear | DQN compound compliance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference_57 | 5347.69 | 5433.58 | 5403.65 | 3.324 | 83.195% | 100.0% |
| laps_30 | 2781.76 | 2782.64 | 2785.78 | 1.165 | 68.228% | 100.0% |
| laps_53 | 4963.76 | 5019.31 | 5017.28 | 3.070 | 82.367% | 100.0% |
| laps_66 | 6209.73 | 6347.91 | 6261.49 | 4.115 | 83.518% | 100.0% |
| laps_80 | 7563.56 | 7706.39 | 7637.85 | 5.244 | 88.689% | 97.1% |
| laps_90 | 8536.58 | 8670.95 | 8606.19 | 6.094 | 88.094% | 94.5% |
| wear_low | 5321.59 | 5345.82 | 5357.73 | 2.878 | 87.113% | 100.0% |
| wear_high | 5371.74 | 5469.19 | 5414.43 | 4.309 | 90.128% | 100.0% |
| pit_cheap | 5319.57 | 5421.07 | 5384.81 | 6.114 | 71.482% | 100.0% |
| pit_expensive | 5372.61 | 5446.09 | 5422.50 | 2.524 | 97.120% | 100.0% |
| no_safety_cars | 5235.10 | 5330.42 | 5265.08 | 3.000 | 81.901% | 100.0% |
| more_safety_cars | 5449.70 | 5524.91 | 5512.32 | 3.469 | 99.336% | 100.0% |
| held_out_combination | 5938.01 | 6056.53 | 5992.64 | 4.047 | 99.367% | 100.0% |

DQN had a lower average race time than both baselines in **13 of 13** scenarios. This is an empirical comparison, not a guarantee of optimality or real F1 transfer.

**The two-compound rule is separate from the hard wear constraint.** It still has a 30-second terminal penalty and is not guaranteed. DQN compound compliance was **97.1% at 80 laps** and **94.5% at 90 laps**; the other published scenarios achieved 100% in this sample. Total race times include penalties. This limitation must not be described as full rule compliance.

53, 66 and 80 laps were excluded from both training and validation. Other scenarios test familiar horizons with fresh seeds, parameter boundaries, and a held-out combination. Behaviour outside the 20–90-lap training domain is extrapolation and can be weaker, including compound-rule failures.

## Reproduce and inspect

- Training: `python -m pitstoprl.train --episodes 30000 --seed 7 --env-count 16 --validate-every 2000`
- Evaluation: `python -m pitstoprl.evaluate --episodes 1000 --seed 2000000 --output-dir results/dqn`
- Checkpoint format: 2; earlier unbounded-wear checkpoints are rejected.
- Checkpoint SHA-256: `8dfe999315cf2b57b33ca2543ac9592cc37e38156fae76d120592d9e6a73f27a`
- Individual race times, stops, compliance and peak wear: [evaluation.csv](evaluation.csv)
- Aggregate metrics and paired intervals: [summary.json](summary.json)
- Environment, scenarios and seed provenance: [protocol.json](protocol.json)
- Training and validation history: [dqn_training.json](../../artifacts/dqn_training.json)

The 27-input network learns from simulator rewards and legal-action bootstrapping; no demonstrations or baseline targets are used. The safety guarantee comes from the shared physical action constraint. Historical F1 records do not train or validate this synthetic simulator.

## Figures

![Generalization gaps](generalization.png)

![Reference comparison](strategy_comparison.png)

![Reference strategy timeline](sample_strategy.png)

![Training reward per lap](training_curve.png)
