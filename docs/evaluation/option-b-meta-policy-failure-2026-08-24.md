# Option B (learned orchestrator meta-policy) — failure note

**Date:** 2026-08-24
**Branch:** `meta-orchestrator-rl`
**Status:** Built, integrated, self-check covered — but never converged. Superseded in
production by Option C (LLM review of the deterministic rule). Kept in the codebase and
documented honestly rather than deleted, per the project's design-drift discipline.

---

## What was attempted

Three governance options were discussed for making `PolicyOrchestrator.select_decision()`
(the REUSE / FINE_TUNE / TRAIN_NEW onboarding decision) genuinely learned instead of a fixed
threshold rule:

- **Option A** — an LLM makes the call directly. Rejected: hands real compute spend
  (a TRAIN_NEW can be 30-90+ min) to an unchecked model.
- **Option B** — a second PPO-style learned meta-policy, same architecture as the
  per-request agent (Component 4), trained via reinforcement learning on the REUSE/
  FINE_TUNE/TRAIN_NEW action space. Chosen first as the "purest" answer.
- **Option C** — the deterministic rule stays the default; an LLM reviews it and may only
  escalate to a more cautious action, never downgrade. Built after B failed to converge —
  see [Option C](#what-replaced-it-option-c) below.

## Design (what was built)

New file `src/devmind/orchestrator_trainer.py`:

- `OrchestrationDecisionEnv` (Gym-style): one episode = one synthetic onboarding decision,
  `done=True` after a single step (contextual bandit dressed as an episode-length-1 MDP,
  so `PPOTrainer`/`PPONetwork` from `trainer.py`/`agent.py` are reused completely
  unmodified — no bandit-specific trainer was written).
  - **Observation**: 13-dim (matches `PPONetwork`'s default `input_dim=13`, deliberately
    the same width as the per-request Gold vector) — best candidate's
    `[accuracy, sla_violation_rate, escalation_rate, fallback_rate]`, three signed gaps
    against the sampled client's `ToleranceThresholds`, five normalized scenario
    descriptors, and a library-empty flag.
  - **Action**: `Discrete(3)` = REUSE / FINE_TUNE / TRAIN_NEW.
  - **Reward**: always computed from a *real* evaluation of whatever the chosen action
    actually produced — REUSE evaluates the real seed policy; FINE_TUNE/TRAIN_NEW really
    invoke `PolicyOrchestrator._train()` (at a reduced step budget, to keep rollout
    collection tractable) and then really evaluate the result. Nothing in the reward is
    fabricated. `quality(metrics, thresholds) - action_cost[action]`, where `quality` is
    `+1.0` if every threshold is met, else the negative sum of how far each is missed.
- `PolicyOrchestrator` gained `meta_policy_path`/`meta_state_stats_path` constructor args
  and a `_meta_decide()` method that mirrors `AgenticOrchestrator.decide()`'s exact safety
  pattern (`agent.py`): one forward pass through the loaded meta-policy, and if
  `entropy > 0.9` **or** the observation is OOD vs. `meta_state_stats.json`, fall back to
  the deterministic `select_decision()` rule rather than trust the network's action.

This safety-first fallback is the same pattern already used at the per-request layer — and
it is exactly what caught the failure below cleanly, rather than letting a bad governance
decision through silently.

## What happened — real numbers, not estimated

**Timing probe** (8 real steps, full budget: `meta_fine_tune_steps=500`,
`meta_train_new_steps=800`, `max_samples=200`): **188.2s total → 23.5s/step**, giving a
~125-minute estimate for a 320-step run. This estimate did not hold under real training.

**Run 1** (`total_steps=320`, `rollout_size=16`, full budget): killed after 32/320 steps.

```
meta step=16/320 loss=2.4716
meta step=32/320 loss=2.7610
```

32 steps took ~40 real minutes (~75s/step) — roughly 3x the probe's rate, because the
real mix of sampled actions in that stretch skewed toward more TRAIN_NEW draws (800 real
training steps each) than the 8-step probe happened to sample. Projected total at this
rate: **5-6 hours**, not the ~125 minutes estimated. Killed and relaunched at a reduced
budget rather than let an underestimated multi-hour job run unannounced.

**Run 2** (`total_steps=160`, `rollout_size=16`, reduced budget:
`meta_fine_tune_steps=250`, `meta_train_new_steps=400`, `max_samples=150`): completed in
full.

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

Wall clock: launched 00:39, saved `meta_policy.pt`/`meta_state_stats.json` at 02:45 —
**126 minutes for 160 steps (~47s/step)**. Loss shows no real convergence trend across
the run (fluctuating 1.5-2.7 throughout), consistent with a policy that never
meaningfully differentiated its action distribution.

### Real-world test: `_meta_decide()` on the actual seed policy, actual client scenarios

```
client_streamforge: decision=reuse      fallback=True  candidates={'seed': (acc=0.906, sla_viol=0.000, esc=0.022)}
client_nhs:          decision=reuse      fallback=True  candidates={'seed': (acc=0.911, sla_viol=0.000, esc=0.017)}
client_babcock:      decision=fine_tune  fallback=True  candidates={'seed': (acc=0.911, sla_viol=1.000, esc=0.017)}
client_newco:        decision=reuse      fallback=True  candidates={'seed': (acc=0.922, sla_viol=0.000, esc=0.000)}
```

**`fallback=True` on every single one of the 4 real `CLIENT_SCENARIOS`.** The trained
meta-policy's own action was never trusted — every decision shown above is actually the
deterministic `select_decision()` rule's output, reached via the entropy-fallback path,
not the learned policy. This is why the subsequent `run_ablation_7()` comparison
(deterministic rule vs. meta-policy-configured orchestrator) produced **identical
decisions** on all 4 clients — the "meta-policy" path never actually engaged.

## Root cause

Entropy stayed above the 0.9 fallback threshold after 160 real training steps. This is
the *same* failure mode diagnosed earlier this session for the per-request PPO
(`ppo_policy.pt`), which needed **100,000 steps** to reliably cross that same threshold
on its own 13-dim/3-action problem (see the `2026-08-20` retrain history in the project documentation's
Key Risks table).

The difference is cost. The per-request PPO's environment (`InferenceGatewayEnv`) is a
fast simulator — synthetic queue/RTT dynamics plus one real (but cheap, single
forward-pass) HF inference call per step, on the order of milliseconds. Reaching 100,000
steps took ~42-83 minutes across this session's retrains.

`OrchestrationDecisionEnv`'s steps are not cheap, *by design* — the reward is never
fabricated, so a FINE_TUNE or TRAIN_NEW action means really invoking
`PolicyOrchestrator._train()`, even at a reduced budget. Measured cost: 47-75s/step.
Matching the per-request PPO's 100,000-step convergence point at this rate would take
**100,000 × ~50s ≈ 58 days** of continuous training — not tractable as currently
designed. "Just train longer," the fix that worked for the per-request PPO, does not
transfer here.

## What this does *not* mean

- The integration code is correct and is kept in the codebase: `_meta_decide()`,
  `PolicyOrchestrator(meta_policy_path=..., meta_state_stats_path=...)`, and
  `orchestrator_trainer.py`'s `OrchestrationDecisionEnv`/`train_meta_policy()` all pass
  their self-checks (`python -m devmind.orchestrator`,
  `python -m devmind.orchestrator_trainer --selfcheck`).
- The safety fallback worked exactly as intended — an undertrained model was correctly
  never trusted, rather than silently making bad governance calls. That the failure was
  *caught* rather than *hidden* is itself evidence the architectural pattern (entropy/OOD
  gating a learned action, same shape as Component 4) is sound; only this particular
  learned component's training economics didn't work out.
- `meta_policy.pt`/`meta_state_stats.json` remain on disk and `PolicyOrchestrator` will
  use them (with the safety fallback) if pointed at them — this isn't disabled, just not
  the production path.

## Paths not taken, for the record

1. **Accept and document as-is** (what happened) — cheapest, and the honest outcome given
   the "never fabricate the reward" design constraint that made the cost structure what
   it is.
2. **Build a cheap surrogate reward model** — train a small regressor on a modest number
   of real `(state → outcome)` samples, then run RL rollouts against the surrogate
   instead of real training every step. Legitimate technique, but weakens the "always
   real, never fabricated" framing and is real new scope.
3. **Run a genuinely long background job** (days) accepting the cost. Not attempted.

## What replaced it: Option C

Built the same session, after this failure: the deterministic rule stays the default
decision-maker; an LLM (`OllamaDiagnosisProvider.review_decision()`, `diagnosis.py`)
reviews it and may only **escalate** to a more cautious action (REUSE → FINE_TUNE →
TRAIN_NEW), never downgrade — so a hallucinated review can waste compute at worst, never
silently authorize an under-provisioned policy. `PolicyOrchestrator._governance_review()`
wires this in, reusing the exact off-the-fast-path Ollama plumbing already built for
`_diagnose_unreachable_threshold()`.

**Live-tested against real Ollama** with `client_babcock`'s actual numbers from the
comparison above (seed policy: `sla_violation_rate=1.00` vs. a `0.15` requirement, rule
chose `fine_tune`):

```
agrees: False
recommended_decision: train_new
justification: The candidate policy 'seed' fails the sla_violation_rate requirement
  significantly (1.00 vs <= 0.15), necessitating a more cautious approach than
  fine-tuning.
parsed_ok: True
latency_ms: 2502.99
```

Correctly identified that FINE_TUNE was too weak a response to a 6.7x SLA-violation gap
and escalated to TRAIN_NEW with a coherent justification, in ~2.5 seconds (after
Ollama's own cold-start unload/reload — a first attempt right after the model had been
idle for hours timed out at 40s across both retry attempts; a warm follow-up call
succeeded normally. Worth keeping in mind operationally, not a code bug).
