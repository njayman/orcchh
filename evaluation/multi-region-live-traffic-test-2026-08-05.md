# Multi-Region Live Traffic Test — 2026-08-05

## Setup

| Component | Region | Role |
|---|---|---|
| `devmind-cloud` | europe-west2 (London) | Cloud pod (BERT-large/TorchServe proxy) + orchestrator dashboard |
| `devmind-edge-near` | europe-west1 (Belgium) | Edge gateways for `client_nhs`, `client_streamforge`, `client_newco` |
| `devmind-edge-far` | australia-southeast1 (Sydney) | Edge gateway for `client_babcock` |

## What happened

### 1. Client onboarding (Policy Orchestration Layer, live)

| Client | Scenario | Decision | Policy assigned | Seed accuracy | Seed SLA violation | Seed trust |
|---|---|---|---|---|---|---|
| `client_nhs` | steady | **fine_tune** | `seed_ft_client_nhs` | 0.775 | 35.0% | 0.760 |
| `client_streamforge` | bursty | **reuse** | `seed_ft_client_nhs` | 0.775 (seed) / 0.800 (ft policy) | 7.5% (seed) / 10.0% (ft policy) | 0.817 / 0.832 |
| `client_babcock` | degraded_network | **fine_tune** | `seed_ft_client_babcock` | 0.775 | **100%** | **0.025** |
| `client_newco` | custom (stress 0.35, degrade 0.10) | **fine_tune** | `seed_ft_client_nhs_ft_client_newco` | 0.800 (seed) | 27.5% (seed) | 0.719 (seed) |

Full entries: `evaluation/orchestrator_decisions.jsonl`.

### 2. Live traffic test (Cascade Controller fast loop, live)

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

## Improvements
