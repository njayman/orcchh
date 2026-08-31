# Multi-Region GKE Live Traffic Test — 2026-08-31

**Status:** completed run, real (not simulated) Kubernetes deployment on GCP across 3 regions.
**Purpose:** live-test the system as originally proposed — deployed via Kubernetes, not just Docker — using the current, post-reward-gradient-fix policy (`ppo_policy.pt`, committed 2026-08-23), the same policy behind this report's offline headline numbers. Also exercises the observability stack (Prometheus + Grafana) named in the original project proposal alongside Kubernetes/Minikube.

## Setup

A single Kubernetes cluster cannot span these 3 regions — etcd consensus isn't built for the ~250ms RTT to Sydney — so each region runs its own small GKE cluster, with cross-region calls staying at the HTTP layer exactly as they already do in the Docker/GCP path (`CloudClient` → `DEVMIND_CLOUD_URL`).

| Component | Cluster / Region | Role |
|---|---|---|
| `devmind-cloud-gke` | europe-west2-a (London) | Cloud pod (BERT-large) + orchestrator dashboard + Prometheus + Grafana |
| `devmind-edge-near-gke` | europe-west4-a (Netherlands) | Edge gateways for `client_nhs`, `client_streamforge`, `client_newco` |
| `devmind-edge-far-gke` | australia-southeast1-a (Sydney) | Edge gateway for `client_babcock` |

Real cross-region RTT: comparable London↔Netherlands distance to the earlier Docker test's London↔Belgium leg, ~250ms+ London↔Sydney — same genuine network distance as the original multi-region test, not synthetic.

Each region's images are built once and pushed to a Google Artifact Registry repository, then deployed via `kubectl apply` against real `Deployment`/`Service` manifests (`code/k8s/cloud.yaml`, `orchestrator.yaml`, `gateway.yaml.tmpl` rendered per client, `prometheus.yaml.tmpl`, `grafana.yaml`) — not `docker run` on a bare VM. Every service is reachable over its own `LoadBalancer` Service external IP.

Prometheus runs in the cloud-region cluster and scrapes all six workloads: the in-cluster cloud pod and orchestrator, plus the three near-cluster gateways and the one far-cluster gateway over their external IPs. Grafana runs alongside it with a provisioned dashboard (request rate by tier, p95 latency by client, SLA violation rate, fallback rate, cloud pod inference latency, onboarding decisions), all sourced from `prometheus_client` metrics exposed at `/metrics` on each service.

The orchestrator's live-monitoring view (`/ws`, request-log tail) is fed by gateway pods forwarding each request's action-log entry to the orchestrator over HTTP (`DEVMIND_ORCHESTRATOR_URL` → `POST /log-action`) rather than relying on a shared local file — necessary once the gateway and the orchestrator are genuinely separate pods in separate regions, which a single-host deployment never has to contend with.

## What happened

### Live traffic test (Cascade Controller fast loop)

The same 6-block/60-minute schedule as the 2026-08-05 Docker test (10 min/block, varying which clients are active), run against the GKE stack:

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

Overall SLA violation rate 3.4%, fallback rate 0.066% (the entropy/OOD safety net rarely needed to override the trained policy's own action). Every pod across all three clusters ran the entire session — including two earlier shorter validation passes before this full run — with zero restarts.

## Takeaways

1. **The full stack works live on real Kubernetes, across real geography, with the current policy.** This closes the gap the original Docker/GCP test left open: that test used plain `docker run`, not Kubernetes, and predated both the undertraining fix and the reward-gradient fix, so it exercised the deterministic fallback rule rather than the trained policy. This run does neither — real `kubectl`-managed Deployments and Services, current `ppo_policy.pt`.
2. **Babcock's latency correctly reflects genuine cross-region physics.** p95 756.3ms for the Sydney-based client against a London cloud pod, versus 96-117ms for the three Netherlands-based clients — this is real network RTT showing up in the numbers, not a synthetic scenario parameter.
3. **The fallback rate (0.066%) confirms the trained policy is governing routing almost entirely on its own**, consistent with the offline evaluation's `fallback_rate=0.000` for this policy — a small non-zero rate under genuine live conditions (rather than the offline simulation's exact domain-randomized distribution) is the expected, honest difference between sim and real deployment, not a regression.
4. **Prometheus histograms need explicit millisecond-scale buckets.** `prometheus_client.Histogram`'s default buckets are calibrated for second-scale durations (topping out at 10.0), which silently degrades quantile estimates for millisecond-scale latencies — everything collapses into the tail bucket. Both gateway and cloud pod latency histograms now declare explicit millisecond buckets (`gateway/app.py`, `gateway/cloud_app.py`); verified by comparing `histogram_quantile` output before and after against the same live traffic.
5. **Live monitoring across regions needs an explicit forwarding path, not just a shared log file.** The orchestrator's `/ws` live-monitoring tail loop reads a local `request_log.jsonl`; in a genuinely multi-region deployment the gateway pods writing that log and the orchestrator pod reading it are on different hosts entirely, so nothing arrives without an explicit channel. `CascadeController` now optionally forwards each action-log entry to the orchestrator over HTTP (`action_log_url`, `POST /log-action`) alongside its local write, verified end-to-end with real cross-region requests from both the near and far clusters.

## Improvements

1. **Formalize this as the primary live-deployment evidence going forward.** This run supersedes the 2026-08-05 Docker test for questions about the *current* policy's live routing behavior; the earlier run remains valid evidence for the deployment path working end-to-end and for the cloud-dispatch error-handling gap it found (already fixed, Section 6 of the report).
2. **Cross-region Prometheus federation is currently single-hop (central scrape of external IPs), not a federated hierarchy.** Fine at this scale (6 targets); would need revisiting for a fleet larger than a handful of regions.
3. **OpenTelemetry tracing (named in the original proposal alongside Prometheus/Grafana) was not part of this run.** Prometheus/Grafana cover metrics; distributed tracing across the edge→cloud request path remains open.
