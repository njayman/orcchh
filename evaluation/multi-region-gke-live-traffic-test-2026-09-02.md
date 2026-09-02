# Multi-Region GKE Live Traffic Test — 2026-09-02

## Setup

| Component | Cluster / Region | Role |
|---|---|---|
| `devmind-cloud-gke` | europe-west2-a (London) | Cloud pod (BERT-large) + orchestrator dashboard + Prometheus + Grafana + Jaeger |
| `devmind-edge-near-gke` | europe-west4-a (Netherlands) | Edge gateways for `client_nhs`, `client_streamforge`, `client_newco` |
| `devmind-edge-far-gke` | australia-southeast1-a (Sydney) | Edge gateway for `client_babcock` |

Smoke test: 972 requests, 0 errors.

## What happened

### Live traffic test (Cascade Controller fast loop)

| Block | Active clients |
|---|---|
| 1 | nhs, streamforge, newco, babcock |
| 2 | streamforge only |
| 3 | nhs, babcock |
| 4 | streamforge, newco |
| 5 | nhs, streamforge, newco, babcock |
| 6 | babcock only |

**Totals: 9,835 requests, 0 errors.**

| Client | Region | Requests | Escalation rate | SLA violation rate | p95 latency |
|---|---|---|---|---|---|
| nhs | near | 2,403 | 1.7% | 1.7% | 118ms |
| streamforge | near | 3,202 | 1.7% | 1.7% | 115ms |
| newco | near | 2,400 | 2.0% | 2.0% | 117ms |
| babcock | far | 1,830 | 2.5% | 2.5% | 673ms |

Fallback rate: 0.000%.

## Comparison against the 2026-08-31 run

| Metric | 2026-08-31 (pre-classifier-retrain policy) | 2026-09-02 (this run) |
|---|---|---|
| nhs escalation | 3.2% | 1.7% |
| streamforge escalation | 2.5% | 1.7% |
| newco escalation | 2.6% | 2.0% |
| babcock escalation | 6.0% | 2.5% |
| Overall SLA violation rate | 3.4% | ~1.9% (weighted) |
| Fallback rate | 0.066% | 0.000% |
| Total requests / errors | 13,729 / 0 | 9,835 / 0 |

## Takeaways
