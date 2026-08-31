# Option B (learned orchestrator meta-policy) — failure note

**Date:** 2026-08-24
**Branch:** `meta-orchestrator-rl`

---

## What was attempted

## Design (what was built)

## What happened — real numbers, not estimated

Timing probe: 188.2s total → 23.5s/step (8 real steps).

**Run 1** (`total_steps=320`, `rollout_size=16`): killed after 32/320 steps.

```
meta step=16/320 loss=2.4716
meta step=32/320 loss=2.7610
```

**Run 2** (`total_steps=160`, `rollout_size=16`): completed in full.

```
meta step=16/160  loss=1.5711
meta step=32/160  loss=2.5148
meta step=48/160  loss=1.8441
meta step=64/160  loss=2.0410
meta step=80/160  loss=1.6692
meta step=96/160  loss=1.7206
meta step=112/160 loss=2.6144
meta step=128/160 loss=2.4243
meta step=144/160 loss=1.7478
meta step=160/160 loss=1.9471
```

Wall clock: launched 00:39, saved `meta_policy.pt`/`meta_state_stats.json` at 02:45.

### Real-world test: `_meta_decide()` on the actual seed policy, actual client scenarios

```
client_streamforge: decision=reuse      fallback=True  candidates={'seed': (acc=0.906, sla_viol=0.000, esc=0.022)}
client_nhs:          decision=reuse      fallback=True  candidates={'seed': (acc=0.911, sla_viol=0.000, esc=0.017)}
client_babcock:      decision=fine_tune  fallback=True  candidates={'seed': (acc=0.911, sla_viol=1.000, esc=0.017)}
client_newco:        decision=reuse      fallback=True  candidates={'seed': (acc=0.922, sla_viol=0.000, esc=0.000)}
```

## Root cause

Measured cost: 47-75s/step. 100,000 × ~50s ≈ 58 days.

## What this does *not* mean

## Paths not taken, for the record

## What replaced it: Option C

```
agrees: False
recommended_decision: train_new
justification: The candidate policy 'seed' fails the sla_violation_rate requirement
  significantly (1.00 vs <= 0.15), necessitating a more cautious approach than
  fine-tuning.
parsed_ok: True
latency_ms: 2502.99
```
