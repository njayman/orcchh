# Multi-Region GKE Live Traffic Test — 2026-08-31

## Setup

| Component | Cluster / Region | Role |
|---|---|---|
| `devmind-cloud-gke` | europe-west2-a (London) | Cloud pod (BERT-large) + orchestrator dashboard + Prometheus + Grafana |
| `devmind-edge-near-gke` | europe-west4-a (Netherlands) | Edge gateways for `client_nhs`, `client_streamforge`, `client_newco` |
| `devmind-edge-far-gke` | australia-southeast1-a (Sydney) | Edge gateway for `client_babcock` |

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

**Totals: 13,729 requests, 0 errors.**

| Client | Region | Requests | Escalation rate | SLA violation rate | p95 latency |
|---|---|---|---|---|---|
| nhs | near | 3,352 | 3.2% | 3.3% | 117.1ms |
| streamforge | near | 4,500 | 2.5% | 2.5% | 96.1ms |
| newco | near | 3,326 | 2.6% | 2.6% | 99.9ms |
| babcock | far | 2,551 | 6.0% | 6.0% | 756.3ms |

Overall SLA violation rate 3.4%, fallback rate 0.066%.

## Takeaways

## Improvements
