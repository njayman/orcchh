# Multi-Region Live Traffic Test — 2026-08-05

**Status:** completed run, real (not simulated) deployment on GCP across 3 regions.
**Purpose:** first live, non-Gymnasium exercise of the full stack — Policy Orchestration Layer onboarding real client scenarios, and the Cascade Controller/PPO fast loop serving real HTTP traffic across genuine cross-region network paths, not localhost/minikube.

## Setup

| Component | Region | Role |
|---|---|---|
| `devmind-cloud` | europe-west2 (London) | Cloud pod (BERT-large/TorchServe proxy) + orchestrator dashboard |
| `devmind-edge-near` | europe-west1 (Belgium) | Edge gateways for `client_nhs`, `client_streamforge`, `client_newco` |
| `devmind-edge-far` | australia-southeast1 (Sydney) | Edge gateway for `client_babcock` |

Real cross-region RTT: ~5-10ms London↔Belgium, ~250-270ms London↔Sydney — genuine network distance, not the synthetic `rtt_base`/`rtt_degraded` values used in the offline Gymnasium simulation.

## What happened

### 1. Client onboarding (Policy Orchestration Layer, live)

All 4 clients onboarded via the orchestrator dashboard's `/clients` endpoint, each running a real evaluation episode against the frozen seed policy (`ppo_policy.pt`) before `select_decision()` picked reuse/fine-tune/train-new:

| Client | Scenario | Decision | Policy assigned | Seed accuracy | Seed SLA violation | Seed trust |
|---|---|---|---|---|---|---|
| `client_nhs` | steady | **fine_tune** | `seed_ft_client_nhs` | 0.775 | 35.0% | 0.760 |
| `client_streamforge` | bursty | **reuse** | `seed_ft_client_nhs` | 0.775 (seed) / 0.800 (ft policy) | 7.5% (seed) / 10.0% (ft policy) | 0.817 / 0.832 |
| `client_babcock` | degraded_network | **fine_tune** | `seed_ft_client_babcock` | 0.775 | **100%** | **0.025** |
| `client_newco` | custom (stress 0.35, degrade 0.10) | **fine_tune** | `seed_ft_client_nhs_ft_client_newco` | 0.800 (seed) | 27.5% (seed) | 0.719 (seed) |

`client_babcock` is the standout: the seed policy hit a 100% SLA-violation rate and trust_score collapsed to 0.025 under `degraded_network` — correctly triggering fine-tune rather than reuse. `client_streamforge` is the only reuse: the policy already fine-tuned for `nhs` happened to also clear tolerance for `streamforge`'s bursty profile, so no new training was needed. `client_newco` fine-tuned *from* the `nhs`-tuned policy (the closest candidate at eval time), not from the original seed — a chained fine-tune, not something the offline ablation script's fixed candidate set typically produces.

Full entries: `evaluation/orchestrator_decisions.jsonl` (also viewable live in the dashboard's new Decision Log table).

### 2. Live traffic test (Cascade Controller fast loop, live)

A 6-block, 60-minute schedule (10 min/block) mixing which clients were active and with what request pattern, run simultaneously on `devmind-edge-near` and `devmind-edge-far`:

| Block | Time (UTC) | Active clients | Requests | Errors | Avg latency (ms) |
|---|---|---|---|---|---|
| 1 — all 4, light | 20:39-20:49 | nhs, streamforge, newco, babcock | 966 | 5 | nhs 75.5 / streamforge 73.2 / newco 110.2 / babcock 114.5 |
| 2 — streamforge isolated, bursty | 20:49-20:59 | streamforge only | 884 | 1 | streamforge 67.7 |
| 3 — nhs + babcock pair | 20:59-21:09 | nhs, babcock | 404 | 0 | nhs 73.3 / babcock 106.2 |
| 4 — streamforge + newco pair | 21:09-21:19 | streamforge, newco | 1163 | 0 | streamforge 71.9 / newco 81.6 |
| 5 — all 4, native patterns | 21:19-21:29 | nhs, streamforge, newco, babcock | 1579 | 1 | nhs 91.0 / streamforge 69.9 / newco 72.5 / babcock 103.0 |
| 6 — babcock isolated | 21:29-21:39 | babcock only | 118 | 0 | babcock 96.1 |

**Totals: 5,116 requests, 7 errors (0.14%).**

## Takeaways

1. **The full stack works live, end to end, across real geography** — not just in the Gymnasium simulation. This is the first evidence that Bronze/Silver/Gold, the Cascade Controller, the frozen PPO policy, and the Policy Orchestration Layer all function correctly outside localhost/minikube, over genuine cross-region network paths.
2. **The Policy Orchestration Layer's decisions are sensible and evidence-backed live, not just offline.** `client_babcock`'s 100%-SLA-violation/0.025-trust seed evaluation is exactly the kind of clear-cut signal the reuse/fine-tune/train-new logic is designed to act on, and it did, live, without any manual intervention.
3. **Real cross-region RTT didn't dominate `babcock`'s observed latency as much as expected.** Average latency for `client_babcock` stayed in the 96-115ms range across all three blocks it appeared in — nowhere near the ~250-270ms genuine Sydney↔London RTT. That means the PPO policy is keeping most `babcock` traffic at the edge rather than escalating to cloud, even under a scenario onboarded specifically because the seed policy failed badly on `degraded_network`. Worth checking directly (not yet done) whether the *fine-tuned* `seed_ft_client_babcock` policy was actually the one serving live traffic during the test, or whether the live gateway was still running the original seed policy the whole time — the dashboard's onboarding and the live gateway's loaded policy are two separate processes/state (`devmind-orchestrator` vs `gw-babcock`), and nothing in this test wired one to update the other.
4. **A real, reproducible robustness gap was found and root-caused**, not just observed: `CloudClient` (`model_clients.py:52-56`) uses a single shared `httpx.AsyncClient(timeout=5.0)` with no retry, and `CascadeController._dispatch()` has no try/except around the cloud-escalation call — asymmetric with the edge path, which does have this protection (feeding the deadman's switch). All 7 errors trace to this one gap: 6 were ~5000-5100ms timeout failures (mostly at container startup, one recurring later), 1 was a 56ms `httpcore.RemoteProtocolError` from a stale pooled connection under block 5's heavier concurrent load — confirmed via `devmind-cloud`'s own logs showing zero errors/restarts, so the failure was entirely client-side. Full detail saved to project memory (`cascade-cloud-error-handling-gap`).
5. **An operational mistake during this session, worth recording so it isn't repeated**: rebuilding the orchestrator container to ship the dashboard update (`docker rm -f` + fresh `docker run`) destroyed the container's in-memory policy library and on-disk decision log, since neither was on a persistent volume. The decision log was recoverable only because it had been manually captured beforehand; the 3 fine-tuned policy checkpoints (`seed_ft_client_nhs`, `seed_ft_client_babcock`, `seed_ft_client_nhs_ft_client_newco`) were not recoverable and would need re-running to regenerate. `orchestrator_app.py` should mount `policy_library/` and `evaluation/` as persistent volumes before any future live use.

## Improvements

1. **Close the cloud-dispatch error-handling gap** (finding #4): wrap `_dispatch()`'s cloud-escalation call in try/except with a bounded retry, mirroring the edge path's philosophy. Consider also tuning `httpx.AsyncClient`'s connection pool/keep-alive expiry to be shorter than uvicorn's server-side keep-alive, to stop stale-connection reuse at the source.
2. **Persist orchestrator state across container recreations.** Mount `policy_library/` and `evaluation/` as Docker volumes (or a small persistent disk) in `gcp-up.sh`/`gcp-update.sh`'s container-run commands, so a code update doesn't silently discard fine-tuned checkpoints and the decision log.
3. **Wire the orchestrator's fine-tuned policy back to the live gateway.** Right now `PolicyOrchestrator` (governance layer, `devmind-orchestrator` container) and `AgenticOrchestrator` (fast loop, the `gw-*` containers) never talk to each other live — onboarding `client_babcock` produced `seed_ft_client_babcock`, but `gw-babcock` kept serving from whatever `ppo_policy.pt` it was launched with. Without this wiring, live onboarding decisions are informative but don't actually change live routing behavior, which limits how much finding #3 can be trusted as evidence of the fine-tuned policy's real effect.
4. **Add per-request action logging to the live gateway** (which of ROUTE_TO_EDGE / ESCALATE_TO_CLOUD / QUERY_EXTENDED_CONTEXT was actually taken), not just latency — the offline simulation's `EvalMetrics` already tracks escalation_rate, but the live path currently has no equivalent, which is why finding #3 had to be inferred indirectly from latency rather than confirmed directly.
5. **Formalize this as a sim-to-real comparison for the dissertation.** This live run is structurally the same experiment as Ablation Run 7, just live instead of offline. Comparing this run's onboarding metrics (accuracy/SLA-violation/escalation per client) against the equivalent offline `run_ablation_7()` numbers would give a genuine sim-to-real gap data point — exactly the evaluation question already flagged as a core risk to validate, and currently unaddressed.
6. **The idle-shutdown automation discussion was paused mid-decision** (Instance Schedule vs. Cloud Scheduler+Function) and never resumed — worth returning to separately, since VMs were left running for the ~80 minutes this whole exercise took.
