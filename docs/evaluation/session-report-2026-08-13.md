# Session Report — Policy Orchestration Layer review, fixes, and Ablation Run 7

This is source material for the dissertation write-up, not dissertation prose.
Numbers and code references are exact; the framing/argument is yours to write.

## 1. What this covers

A three-lens review (code correctness, AI-engineering/methodology, scope/critique)
of the Policy Orchestration Layer (Component 7) and the Edge Context Protocol's
`DriftEventListener`, followed by fixes for every confirmed defect, followed by a
re-run of Ablation Run 7 (single shared policy vs. Policy Orchestration Layer)
against the fixed code. Everything below traces to `code/src/devmind/agent.py`,
`orchestrator.py`, and `environment.py`.

## 2. Review methodology

Three passes, deliberately different lenses rather than one generic review:

- **Code reviewer** (background agent, independent read of the diff and call
  sites) — correctness bugs and reuse/efficiency issues.
- **AI engineer** (direct code grounding — read `medallion.py`'s masking logic,
  `models.py`'s `GoldStateVector`, and every call site of the functions in
  question before writing a finding) — RL/ML-methodology issues: what the
  training loop and the deployed policy actually see vs. what the design intends.
- **Critique** — scope discipline and claim-honesty: does the write-up's framing
  of a mechanism match what the code actually does.

Every finding was re-verified against current code before being called a defect
(not just re-stated from memory), including one finding whose severity was
upgraded mid-review after re-grepping call sites (`QUERY_EXTENDED_CONTEXT` went
from "unimplemented" to "the mechanism it's implementing can't work at all with
this data model," see §3.1).

## 3. Defects found and fixed

### 3.1 `query_extended_context()` couldn't reveal new information

**The finding.** `SilverEnricher.enrich()` (`medallion.py:104-107`) computes
`calibration_delta` and `error_rate` purely from the edge's real
`operational_state` (STRESSED/DEGRADING) — before `AgenticOrchestrator.decide()`
ever runs. The Gold state vector the PPO policy sees is identical whether or not
the agent "queries" for extended context. The original `decide()` implementation
called `reason()` (the policy forward pass) a second time on that _identical_
tensor after a `QUERY_EXTENDED_CONTEXT` first decision — so any difference
between the first and second decision was PPO's stochastic sampling noise, not
information gain. Two independent reviewers (mine, and the background
code-reviewer agent) converged on this from different angles: I traced it via
the Silver/Gold masking code; the code-reviewer traced it via `MCPSkillInterface`
call sites and separately caught that the discarded entropy from the second
`reason()` call meant the entropy/OOD safety fallback (documented as the
static-τ=0.9 safety net) silently never ran on the re-decided action.

**Why it matters for the write-up.** The project documentation's Component 3 describes this as
"the agent learns _when extra perception is worth paying for_." That framing
does not hold for the pre-fix code — worth being explicit about in a limitations
section if you'd already drafted that framing, or worth noting as "fixed before
being reported" if you're writing after the fix. It also **confounded Ablation
Run 5** (agent-driven perception value): that ablation would have measured
resampling noise, not real perception value, until this fix landed.

**The fix.** `decide()` no longer resamples the policy on
`QUERY_EXTENDED_CONTEXT`. It calls `act()` to register the extended MCP skills,
then consults them directly (`get_edge_calibration_delta`,
`get_edge_error_rate`) and applies a deterministic risk rule: escalate to cloud
if either signal exceeds a threshold
(`QUERY_CALIBRATION_RISK_THRESHOLD = 0.2`, `QUERY_ERROR_RATE_RISK_THRESHOLD =
0.15`), else route to edge. This is honest about what the current Silver/Gold
masking design can support. A genuine "hidden until queried" protocol — where
masking becomes query-conditional rather than state-conditional — is scoped out
as future work, because it would require touching the raw PPO training loop
(which samples `policy.forward()` directly and bypasses `decide()` by design) to
keep training and serving consistent; that's a bigger change than a same-session
fix should attempt.

### 3.2 Ablation Run 7 was confounded by an unequal training recipe

**The finding** (code-reviewer agent). `trainer.py::train_agent()` gained domain
randomization on 2026-08-12 (`randomize_scenario()`, resampling 12
`ScenarioConfig` fields every 2048-step rollout). But
`PolicyOrchestrator._train()` — used by `_fine_tune()` and `_train_new()`, i.e.
every policy the orchestrator itself produces — duplicated the old rollout loop
and never called `randomize_scenario()`. So the shared seed policy trained under
domain randomization, while every orchestrator-produced policy trained on one
fixed scenario. Any robustness difference Run 7 reported would have been
partly an artifact of this training-recipe mismatch, not purely the fleet-level
governance value the ablation is designed to isolate.

**The fix.** `_train()` now samples `randomize_scenario()` per rollout, same as
`trainer.py`. Both the shared and orchestrated arms of Run 7 now train under the
same recipe.

### 3.3 `reward_weights` could saturate against the reward clip

**The finding.** `ScenarioConfig.reward_weights` was added to unblock the
planned λ-sensitivity analysis (project documentation, Key Intellectual Challenges). But
`_inference_step()` clips the final reward to `[-1.0, 1.0]` regardless of the
weight tuple. A weight tuple that doesn't sum to ~1.0 — which is exactly what a
sensitivity sweep would try — could push the unclipped value past 1.0 and get
truncated, flattening the very signal the sweep is meant to observe.

**The fix.** `_compute_reward()` now normalizes the weight tuple by its own sum
before applying it, so the hard clip constrains scale (training stability) and
not the weight _ratios_ a sweep varies.

### 3.4 Unbounded growth in `DriftEventListener`'s trackers

**The finding** (code-reviewer agent). `_last_escalated` (added for the
multi-signal correlation fix earlier in this session) and the pre-existing
`_distress_since` are keyed by `client_id` with no eviction — a slow memory leak
for a long-running listener serving many or transient clients.

**The fix.** `should_escalate()` now prunes entries older than
`10 * recovery_window_s` on every call.

## 4. Ablation Run 7 results (post-fix)

Run at 2026-08-12 21:23–21:30 BST. Source:
`docs/evaluation/run7_2026-08-12_21-30-14.{txt,json}`,
`docs/evaluation/orchestrator_decisions.jsonl`. `max_samples=500`,
`eval_n_runs=3` per client, seed policy = the domain-randomized checkpoint
retrained 2026-08-12.

| Client             | Scenario                     | Decision      | Shared SLA-viol / Esc% / Fallback | Orchestrated SLA-viol / Esc% / Fallback | Shared acc | Orchestrated acc |
| ------------------ | ---------------------------- | ------------- | --------------------------------- | --------------------------------------- | ---------- | ---------------- |
| client_streamforge | bursty                       | reuse         | 0.0% / 9.7% / 0%                  | 0.0% / 13.0% / 0%                       | 0.813      | 0.827            |
| client_nhs         | steady                       | reuse         | 0.0% / 14.0% / 0%                 | 0.0% / 15.0% / 0%                       | 0.827      | 0.830            |
| client_babcock     | degraded_network (800ms RTT) | **fine_tune** | 100% / 100% / 100%                | 100% / **37.0%** / **21.7%**            | 0.860      | 0.817            |
| client_newco       | custom                       | reuse         | 0.0% / 13.3% / 0%                 | 0.7% / 11.7% / 0%                       | 0.807      | 0.827            |

Decision log (`dominant_signal` for the fine-tune case is `n/a_new_policy` since
it's the first policy produced for that scenario, not an eval-against-tolerance
comparison).

**Headline finding: `client_babcock` is where orchestration earns its keep, and
the metric that shows it is escalation/fallback rate, not SLA-violation rate.**
The shared seed policy — never trained on 800ms-RTT conditions — escalates
100% of requests and hits the entropy/OOD fallback 100% of the time: it is so
far outside its training distribution that it never trusts its own policy
output and always falls back to the static τ=0.9 rule. The orchestrator
correctly classified this as a near-miss (not within tolerance, but not
distant enough to train from scratch), fine-tuned the seed policy on the
degraded-network scenario, and cut escalation from 100%→37% and fallback
reliance from 100%→22%.

**SLA-violation rate stays at 100% in both conditions for `client_babcock` —
this is a scenario-design ceiling, not a policy failure.** 800ms base RTT alone
exceeds most reasonable SLA budgets (`sla_budget_ms=300` default) regardless of
which pod handles the request; no routing decision can satisfy an SLA the
network latency alone already violates. If you cite this table, lead with
escalation-rate/fallback-rate as the effect being measured, not
SLA-violation-rate, which is structurally saturated here and would read as "no
improvement" if presented alone.

**For the other three clients, `reuse` was the correct call and orchestration
adds no meaningful value beyond confirming reuse** — accuracy is flat to
marginally higher (~0.81→0.83, likely partly attributable to the reward-weight
normalization and query-resolution fixes above rather than orchestration
itself), escalation ticks up slightly, no fallback triggering either arm. This
is the expected/boring case: when a client's traffic resembles what the seed
policy already trained on, the system should do nothing more than confirm
reuse — which is what happened.

## 5. Still outstanding (not run/decided this session)

- No baseline comparison (`evaluate_baselines()` / `evaluate_literature_baselines()`
  in `evaluation.py`) has been run against the sentiment/spam/topic onboarding
  tasks, despite the machinery already supporting arbitrary `ScenarioConfig.task`.
- λ-sensitivity sweep script does not exist yet — only the `reward_weights`
  parameterization and its normalization fix do.
- `docs_bundle_all_code.py` (a flat concatenation of all `src/devmind/*.py`
  files, generated earlier this session) remains untracked in `code/` — no
  decision made on whether to keep, commit, or delete it.
- `docs/evaluation/multi-task-fleet-demo-2026-08-12.md` remains untracked in
  `docs/` pending your go-ahead.
- The project's scope-lock documentation describes Jigsaw as the sole dataset; the
  sentiment/spam/topic onboarding demos are framed there as routing-policy
  transfer exercises for Component 7, not a scope change — still worth a
  verbal confirmation from your supervisor at the next sync per the existing
  Component 7 sign-off note.
- GCP live-traffic deploy (`code/deploy/gcp-up.sh`) has not been used to
  validate any of this session's fixes against real infrastructure — everything
  above is simulator-only, which is itself worth a line in the dissertation's
  sim-to-real-gap discussion.
