# MiscalibrationClassifier.fit() — two retrains, one live result

**Date:** 2026-09-01
**Branch:** `experiment/miscalibration-classifier-fit`
**Status:** Simulation-domain classifier wired in and evaluated against the full
ablation suite. Real-hardware classifier kept in the codebase but not wired into
the offline simulation (does not transfer — see below).

---

## Why this was never run before

`MiscalibrationClassifier` (`edge.py`) ships with a `fit()` method and a
`_fallback_heuristic()` static method. Every existing evaluation run — including
every number already in the report — used the untrained classifier, which falls
straight through to the heuristic (`cpu_thermal > 0.85 or error_rate > 0.10` →
DEGRADING, etc.). `fit()` had never actually been called: no training data had
been collected for it, real or synthetic.

## Attempt 1: real hardware

`src/devmind/misc_classifier_fit_experiment.py` collects training data by
genuinely stressing the machine (`multiprocessing`-based CPU burn) while running
real DistilBERT inference against the Jigsaw test set, sampling real `psutil`
resource readings. Labels come only from real observed accuracy relative to an
idle baseline (never from the heuristic itself, to avoid circularity) via
`label_from_accuracy()`: rolling error rate vs. baseline, thresholded at
delta ≤ 0.10 → NOMINAL, ≤ 0.25 → STRESSED, else DEGRADING.

80,139 samples, `LogisticRegression(class_weight="balanced")`, DEGRADING gated
behind a tuned probability threshold (0.905, found via `precision_recall_curve`
on a held-out slice) rather than plain argmax. 5-fold cross-validated:

| Metric | Value |
|---|---|
| Balanced accuracy | 68.0% ± 2.6% |
| DEGRADING precision / recall | 64.4% ± 2.8% / 66.4% ± 7.2% |
| NOMINAL precision / recall | 94.5% / 68.6% |
| STRESSED precision / recall | 22.6% / 69.1% |

STRESSED precision (22.6%) is a disclosed limitation — six further techniques
(RandomForest, GradientBoosting, HistGradientBoosting with fair class weighting,
polynomial features, a custom 3x DEGRADING class weight, a C-parameter grid
search) were tried and none beat the threshold-tuned logistic regression above.

**This model does not transfer into the Gymnasium simulation.** Wired in via
`DEVMIND_MISC_CLASSIFIER_PATH` and checked directly (instrumented run, counting
predicted `operational_state` across 2000 simulated steps): it predicts NOMINAL
on effectively 100% of inputs. Its decision boundary was fit on real `psutil`
value ranges; the simulation's synthetic stress generator
(`InferenceGatewayEnv._build_bronze()`) draws from a different distribution
entirely (e.g. `rng.uniform(0.6, 0.95)` for "stressed" CPU vs. whatever a real
machine under `multiprocessing` load actually reports). Any earlier-looking
ablation improvement from wiring this model in was an artifact of that collapse
to NOMINAL, not a real effect, and was not reported as one.

Artifacts: `evaluation/misc-classifier-final-2026-09-01_22-57-02.joblib` (+
`-meta.json`), raw data cached in
`evaluation/misc-classifier-raw-data-2026-09-01_21-21-09.npz`.

## Attempt 2: retrained for the simulation's own distribution

Real hardware contention doesn't matter if the deployment target is a
simulation with its own synthetic stress formula. `src/devmind/
misc_classifier_fit_simulation.py` mirrors the experiment script exactly, but
draws stress from `draw_simulated_stress()` — the identical formula
`environment.py` uses — instead of real `psutil` reads, across
`edge_stress_prob` levels spanning every scenario actually used in evaluation
(0.0 idle, 0.1 steady/bursty/degraded_network, 0.4 held_out, plus 0.7 and 1.0 to
guarantee real DEGRADING coverage). Labels: same non-circular
`label_from_accuracy()`, unchanged.

27,000 samples in 377s (vs. 37.5 minutes for the real-hardware collection —
driving a synthetic RNG is far cheaper than actually stressing a CPU). Same
fitting methodology (class-weighted logistic regression, threshold-tuned
DEGRADING gate). 5-fold cross-validated:

| Metric | Value |
|---|---|
| Balanced accuracy | 54.6% ± 2.0% |
| DEGRADING precision / recall | 10.3% ± 5.6% / 22.0% ± 6.1% |
| NOMINAL precision / recall | 93.5% / 72.9% |
| STRESSED precision / recall | 33.2% / 68.9% |

Weaker than the real-hardware fit across the board — synthetic stress values
carry less real signal about actual accuracy than genuine hardware contention
did. Still meaningfully above a 3-class random floor (~33%).

**Transfer check** (the same instrumented-run test that caught Attempt 1's
failure, run against both `steady` and `held_out`): produces a genuinely
non-degenerate distribution, with DEGRADING share scaling with the scenario's
`edge_stress_prob` as expected —

| Scenario | NOMINAL | STRESSED | DEGRADING | UNREACHABLE |
|---|---|---|---|---|
| steady (`edge_stress_prob=0.1`) | 59.3% | 37.0% | 2.75% | 0.9% |
| held_out (`edge_stress_prob=0.4`) | 57.5% | 36.7% | 5.15% | 0.65% |

Artifacts: `misc_classifier_fit_simulation.py`,
`evaluation/misc-classifier-sim-final-2026-09-01_23-24-56.joblib` (+
`-meta.json`, threshold 0.869), raw data in
`evaluation/misc-classifier-sim-raw-data-2026-09-01_23-20-28.npz`.

## Ablation re-run: heuristic vs. this classifier

Wired in via `DEVMIND_MISC_CLASSIFIER_PATH`/`DEVMIND_MISC_CLASSIFIER_THRESHOLD`
and run through the exact same `run_ablation()`/`run_holdout_ablation()` calls
`evaluation.py`'s `main()` uses (`max_samples=1000`), across all three core
scenarios plus the held-out scenario. `run1_full` and `run4_no_reflect` are the
two conditions that actually route through `EdgeDevice`/`operational_state`;
`run2_confidence_only` and `run3_calibration_delta` are fixed-rule baselines and
change only marginally, as expected.

| Scenario | Condition | Escalation rate | p95 latency | Energy | Accuracy |
|---|---|---|---|---|---|
| steady | run1_full, heuristic | 0.235 | 164.5ms | 185.1 | 0.785 |
| steady | run1_full, classifier | **0.035** | **82.6ms** | **127.9** | 0.785 |
| bursty | run1_full, heuristic | 0.195 | 146.9ms | 194.3 | 0.780 |
| bursty | run1_full, classifier | **0.030** | **78.8ms** | **140.3** | 0.800 |
| degraded_network | run1_full, heuristic | 0.245 | 889.2ms | 193.0 | 0.805 |
| degraded_network | run1_full, classifier | **0.045** | **486.6ms** | **126.7** | 0.790 |
| held_out | run6, heuristic | 0.265 | 3894.1ms | 326.5 | 0.805 |
| held_out | run6, classifier | **0.040** | **680.8ms** | **151.9** | 0.790 |

Consistent across all four conditions: escalation rate drops 82-85% relative,
p95 latency drops 44-82%, energy drops 22-53%, and accuracy is essentially flat
(largest change ±0.015). `run4_no_reflect` shows the identical pattern in every
scenario, confirming this is driven by `operational_state` quality, not the
bidirectional loop.

**Interpretation:** the heuristic's `cpu_thermal > 0.65 or mem_disk > 0.80` →
STRESSED rule fires on a wide swath of the simulation's own synthetic stress
distribution (its "stressed" branch draws from `rng.uniform(0.6, 0.95)`, well
above the heuristic's 0.65 threshold), so the heuristic reports STRESSED far
more often than the trained classifier does — and every STRESSED/DEGRADING
report reveals `calibration_delta`/`error_rate` to the PPO policy, which was
trained to treat their presence as a signal worth escalating on. Whether the
heuristic or the trained classifier is closer to "correct" is a separate
question from whether it changes routing behavior — it clearly does, sharply
and consistently, in the direction of fewer escalations at equivalent accuracy.

A quick 60-episode pass at `max_samples=300` (run before this full pass, for a
first read) showed much smaller and noisier differences — underscoring that the
full 1000-sample run was necessary before treating this as a real effect rather
than sampling noise from an easy slice of the dataset.

## What this does and doesn't establish

- **Non-circular fit() genuinely works**, twice, on two different domains with
  two different feature distributions.
- **Domain transfer matters and was checked directly, not assumed** — the
  real-hardware model's failure and the simulation-domain model's success were
  both confirmed by running the actual system and counting outputs, not by
  inspecting cross-validation numbers alone.
- **The classifier's own accuracy is honestly modest** (54.6% balanced
  accuracy, weak DEGRADING precision) — it is a noisier signal than the
  real-hardware version, not a stronger one, even though it produces a larger
  behavioral effect once wired into the full system.
- **The behavioral effect is real, not an artifact** — unlike Attempt 1, this
  model's output distribution is non-degenerate, and the ablation table above
  was generated at the same sample size as the report's other ablation numbers.
- **The effect is a consequence of Silver's static threshold, not of the
  classifier's superior judgement per se** — the heuristic and the classifier
  disagree mostly because the heuristic's fixed thresholds and the simulation's
  synthetic stress ranges happen to overlap heavily, not because the classifier
  demonstrably tracks ground truth better in the simulation domain (its own
  cross-validated precision/recall says it doesn't, particularly for STRESSED).
