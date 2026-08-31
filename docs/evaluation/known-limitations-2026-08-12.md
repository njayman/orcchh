# Known Limitations and Fixes — 2026-08-12

Notes for the methodology/limitations chapter, from a code-reviewer + AI-engineer

- critique review of `orchestrator.py`, `agent.py`, `environment.py`, `trainer.py`
  (commits `0a79add`, `aa6244b`, and the fixes below).

## For the Limitations section

**`query_extended_context()` was architecturally unable to reveal new information
to the policy, now fixed by not pretending it does.** `SilverEnricher.enrich()`
computes `calibration_delta`/`error_rate` purely from the edge's real
`operational_state` (STRESSED/DEGRADING), before `decide()` ever runs — the Gold
vector the agent sees is identical whether or not it queries. The original
`decide()` implementation called `reason()` a second time on that identical
vector, so any change in the second decision was PPO's stochastic sampling noise,
not "the agent learning when extra perception is worth paying for" as originally
framed. Confirmed independently by a background code-reviewer agent, which also
flagged that this **silently confounded Ablation Run 5** (agent-driven perception
value) — that ablation was measuring resample noise, not real perception value.

Fix (2026-08-12): `AgenticOrchestrator.decide()` no longer resamples the policy on
`QUERY_EXTENDED_CONTEXT`. It consults the extended MCP skills
(`get_edge_calibration_delta`, `get_edge_error_rate`) directly and applies a
deterministic risk rule (escalate if either exceeds a threshold, else route to
edge). This is honest about what the mechanism can do given the current
Silver/Gold masking design — a structural extension where masking becomes
_query-conditional_ rather than state-conditional is future work, not attempted
here because it would require touching the raw PPO training loop (which samples
`policy.forward()` directly, bypassing `decide()` entirely) to keep train/serve
consistent.

**Domain-randomized fine-tune/train-new policies vs. the shared seed policy
(Ablation Run 7).** The seed PPO policy (`trainer.py::train_agent`) has trained
under domain randomization since 2026-08-12. `PolicyOrchestrator._fine_tune()`
and `._train_new()` (used by the Policy Orchestration Layer, Component 7) called
their own rollout loop in `_train()` that never randomized the scenario, so any
per-client policy the orchestrator produced was overfit to one fixed
`ScenarioConfig` while the shared baseline wasn't. This would have confounded
Ablation Run 7's comparison (single shared policy vs. orchestrated multi-policy
library) with a training-recipe mismatch, not just the fleet-governance effect
being isolated. Fixed by routing `_train()` through the same
`randomize_scenario()` helper.

**Reward weight sensitivity sweeps could silently saturate.** `ScenarioConfig.reward_weights`
was added for a planned λ-sensitivity analysis, but `_inference_step()` still hard-clips
the final reward to `[-1.0, 1.0]`. A weight tuple not summing to ~1.0 (any sweep that
isn't the default) could push the unclipped value above 1.0 and get truncated at the
clip boundary, flattening the intended sensitivity signal. Fixed by normalizing weights
by their sum inside `_compute_reward()` before the fixed clip is applied, so the clip
constrains scale, not the ratio the sweep is actually testing.

## Still not run (unchanged from prior review)

- No baseline comparison (`evaluate_baselines()` / `evaluate_literature_baselines()`)
  executed against the sentiment/spam/topic onboarding tasks yet, despite the
  machinery supporting it via `ScenarioConfig.task`.
- `run_ablation_7()` has not been re-run since the domain-randomization fix above —
  the confound it would have introduced was caught before any reported numbers
  were generated from it, but the run itself is still pending.
- λ-sensitivity sweep script does not exist yet; only the parameterization does.
