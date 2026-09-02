# MiscalibrationClassifier.fit() — two retrains, one live result

**Date:** 2026-09-01
**Branch:** `experiment/miscalibration-classifier-fit`

---

## Why this was never run before

## Attempt 1: real hardware

| Metric | Value |
|---|---|
| Balanced accuracy | 68.0% ± 2.6% |
| DEGRADING precision / recall | 64.4% ± 2.8% / 66.4% ± 7.2% |
| NOMINAL precision / recall | 94.5% / 68.6% |
| STRESSED precision / recall | 22.6% / 69.1% |

Artifacts: `evaluation/misc-classifier-final-2026-09-01_22-57-02.joblib` (+
`-meta.json`), raw data cached in
`evaluation/misc-classifier-raw-data-2026-09-01_21-21-09.npz`.

## Attempt 2: retrained for the simulation's own distribution

27,000 samples in 377s.

| Metric | Value |
|---|---|
| Balanced accuracy | 54.6% ± 2.0% |
| DEGRADING precision / recall | 10.3% ± 5.6% / 22.0% ± 6.1% |
| NOMINAL precision / recall | 93.5% / 72.9% |
| STRESSED precision / recall | 33.2% / 68.9% |

| Scenario | NOMINAL | STRESSED | DEGRADING | UNREACHABLE |
|---|---|---|---|---|
| steady (`edge_stress_prob=0.1`) | 59.3% | 37.0% | 2.75% | 0.9% |
| held_out (`edge_stress_prob=0.4`) | 57.5% | 36.7% | 5.15% | 0.65% |

Artifacts: `misc_classifier_fit_simulation.py`,
`evaluation/misc-classifier-sim-final-2026-09-01_23-24-56.joblib` (+
`-meta.json`, threshold 0.869), raw data in
`evaluation/misc-classifier-sim-raw-data-2026-09-01_23-20-28.npz`.

## Ablation re-run: heuristic vs. this classifier

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

## What this does and doesn't establish

## Full picture: classifier x policy-retrain, 2x2

| Policy | Classifier | Steady esc | Bursty esc | Degraded esc | Held-out esc |
|---|---|---|---|---|---|
| pre-retrain (frozen) | heuristic | 0.235 | 0.195 | 0.245 | 0.265 |
| pre-retrain (frozen) | trained | 0.035 | 0.030 | 0.045 | 0.040 |
| retrained | heuristic | 0.010 | 0.015 | 0.035 | 0.010 |
| retrained | trained | 0.010 | 0.005 | 0.015 | 0.005 |

Raw data: `evaluation/heuristic-classifier-on-retrained-policy-2026-09-02.txt`
(the new row), `evaluation/misc-classifier-ablation-comparison-2026-09-01.txt`
(the frozen-policy rows), `evaluation/2026-09-02_00-47-07.json` (retrained+trained,
via `run1_full`).
