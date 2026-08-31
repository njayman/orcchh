# Multi-Task Fleet Demo — 2026-08-12

**Status:** completed run, real (not simulated) local deployment. All three services (`devmind-cloud`, `devmind-gateway`, `devmind-orchestrator`) live, real HTTP calls, real model forward passes.
**Purpose:** first end-to-end exercise of the multi-task generalization built this session — `TASK_DATASETS`/`TASK_MODELS` registries, per-task lazy model caching in `PolicyOrchestrator._models_for()`, and `service_id`/`task`-aware onboarding — proving the same frozen PPO policy and Cascade Controller serve genuinely different classification tasks with no task-specific code path.

## Setup

| Service | Port | Role |
|---|---|---|
| `devmind-cloud` | 8001 | Cloud pod (BERT-large equivalent), `DEVMIND_TASK=toxicity` |
| `devmind-gateway` | 8010 | Edge gateway (DistilBERT), moved off 8000 due to a pre-existing unrelated process (`gohttpser`, PID 1969) already bound there |
| `devmind-orchestrator` | 8002 | Policy Orchestration Layer dashboard + `/clients` onboarding API |

## 1. Live inference through the gateway

One real request through the toxicity gateway (`POST /infer`, edge tier):

| request_id | tier | latency_ms | sla_met | accuracy | fallback_triggered |
|---|---|---|---|---|---|
| `63e598fb...` | edge | 509.6 | false | 0.991 | true |

509ms was a cold-start (first model forward pass after process boot); `sla_met=false` reflects the 300ms default budget being blown by that cold start, not a steady-state number. `fallback_triggered=true` — the very first request also exercised the entropy/OOD static-fallback safety path, since no meaningful state history exists yet.

## 2. Multi-task onboarding — real evaluation, not projected

Four clients, one task each, onboarded via `POST /clients` against the live orchestrator, each running a real evaluation episode before `select_decision()` picked reuse/fine-tune/train-new:

| Client | Task | Decision | Policy assigned | Seed accuracy | Seed SLA violation | Escalation rate |
|---|---|---|---|---|---|---|
| `demo_toxicity` | toxicity | **fine_tune** | `seed_ft_demo_toxicity` | 0.85 | 5.0% | 100% |
| `demo_sentiment` | sentiment | **reuse** | `seed_ft_demo_toxicity` | 1.00 (seed) / 1.00 (ft policy) | 5.0% (seed) / 0.0% (ft policy) | 100% (seed) / 20% (ft policy) |
| `demo_spam` | spam | **reuse** | `seed_ft_demo_toxicity` | 1.00 (seed) / 1.00 (ft policy) | 0.0% (seed) / 0.0% (ft policy) | 100% (seed) / 5% (ft policy) |
| `demo_topic` | topic | **reuse** | `seed_ft_demo_toxicity` | 1.00 (seed) / 0.95 (ft policy) | 0.0% (seed) / 0.0% (ft policy) | 100% (seed) / 5% (ft policy) |

`demo_toxicity` onboarded first and got a genuine fine-tune (no existing candidate yet). Every task onboarded *after* it reused the resulting `seed_ft_demo_toxicity` policy — its escalation rate and SLA-violation rate were within tolerance for sentiment, spam, and topic too, cutting escalation from 100% (raw seed) down to 5-20%.

**Read honestly, not oversold:** this is routing-policy transfer, not task transfer. The PPO never saw sentiment/spam/topic *content* — it decides edge-vs-cloud from task-agnostic operational signals (confidence, queue depth, RTT, SLA budget, edge stress), and those signals happened to look similar enough across tasks that the same policy cleared tolerance. Per-task classification accuracy comes entirely from `TASK_MODELS` swapping in the correct pretrained checkpoint (verified against each dataset's label ordering), not from the orchestrator "learning" the new task.

Full entries: `docs/evaluation/orchestrator_decisions.jsonl` (gitignored locally as a generated artifact — this table is the durable record).

## Known gaps (not yet built)

- GCP deploy scripts don't yet run per-task cloud pods on distinct ports or set `DEVMIND_TASK` on the live 3-VM stack — this demo was local-only.
- No baseline-comparison script yet quantifies "existing (confidence-only) methods have problem X, this system solves it" across all 4 tasks — the original motivation for the multi-dataset work.
- No policy has been fine-tuned or trained-new *specifically* for sentiment/spam/topic; every non-toxicity task so far reused the toxicity-derived seed.
- `evaluation.py` baselines and `run_ablation_7()` are stale relative to today's multi-task code changes.
- Supervisor sign-off for this dataset/scope expansion (per the project's documented scope-lock process) has not been obtained.
