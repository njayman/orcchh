#!/usr/bin/env bash
# Build+push images and (re)deploy everything onto the 3 GKE clusters created
# by gke-up.sh: cloud pod + orchestrator dashboard + Prometheus/Grafana in the
# cloud-region cluster, per-client gateway pods in the two edge-region
# clusters. Safe to re-run repeatedly while iterating (rebuilds images,
# re-applies manifests, restarts deployments) -- this is the script to run
# during the smoke-test / fix-errors loop before the full traffic run.
# Run from the `code/` directory. Requires gke-up.sh to have run first.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGISTRY="europe-west2-docker.pkg.dev/${PROJECT}/devmind"

CLOUD_ZONE="europe-west2-a"
NEAR_ZONE="europe-west4-a"
FAR_ZONE="australia-southeast1-a"

CLOUD_CLUSTER="devmind-cloud-gke"
NEAR_CLUSTER="devmind-edge-near-gke"
FAR_CLUSTER="devmind-edge-far-gke"

echo "== Building and pushing images to $REGISTRY =="
docker build -q -f Dockerfile.cloud -t "${REGISTRY}/devmind-cloud:test" .
docker build -q -f Dockerfile.gateway -t "${REGISTRY}/devmind-gateway:test" .
docker build -q -f Dockerfile.orchestrator -t "${REGISTRY}/devmind-orchestrator:test" .
docker push -q "${REGISTRY}/devmind-cloud:test"
docker push -q "${REGISTRY}/devmind-gateway:test"
docker push -q "${REGISTRY}/devmind-orchestrator:test"

wait_for_lb_ip() {
  local svc="$1"
  for _ in $(seq 1 30); do
    ip="$(kubectl get svc "$svc" -n devmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
    if [[ -n "$ip" ]]; then echo "$ip"; return 0; fi
    sleep 10
  done
  echo "Timed out waiting for $svc's external IP" >&2
  exit 1
}

echo "== Deploying Jaeger (tracing) to $CLOUD_CLUSTER =="
gcloud container clusters get-credentials "$CLOUD_CLUSTER" --zone="$CLOUD_ZONE"
kubectl apply -f k8s/jaeger.yaml
echo "== Waiting for Jaeger's external IP =="
JAEGER_IP="$(wait_for_lb_ip jaeger)"
OTEL_ENDPOINT="http://${JAEGER_IP}:4317"
echo "   UI: http://${JAEGER_IP}:16686   OTLP: $OTEL_ENDPOINT"

echo "== Deploying cloud pod + orchestrator to $CLOUD_CLUSTER =="
sed -e "s|__REGISTRY__|${REGISTRY}|g" -e "s|__OTEL_ENDPOINT__|${OTEL_ENDPOINT}|g" k8s/cloud.yaml | kubectl apply -f -
sed -e "s|__REGISTRY__|${REGISTRY}|g" -e "s|__OTEL_ENDPOINT__|${OTEL_ENDPOINT}|g" k8s/orchestrator.yaml | kubectl apply -f -
kubectl rollout restart deployment/devmind-cloud deployment/devmind-orchestrator -n devmind

echo "== Waiting for cloud pod's external IP =="
CLOUD_IP="$(wait_for_lb_ip devmind-cloud)"
CLOUD_URL="http://${CLOUD_IP}:8001"
echo "   $CLOUD_URL"

echo "== Waiting for orchestrator's external IP =="
# Each K8s LoadBalancer Service gets its own external IP -- unlike the Docker/VM
# path, where the cloud pod and orchestrator share one VM's IP on different ports.
ORCH_IP="$(wait_for_lb_ip devmind-orchestrator)"
ORCH_URL="http://${ORCH_IP}:8002"
echo "   $ORCH_URL"

render_gateway() {
  local name="$1" client="$2" port="$3"
  sed -e "s|__REGISTRY__|${REGISTRY}|g" \
      -e "s|__NAME__|${name}|g" \
      -e "s|__CLIENT__|${client}|g" \
      -e "s|__PORT__|${port}|g" \
      -e "s|__CLOUD_URL__|${CLOUD_URL}|g" \
      -e "s|__ORCH_URL__|${ORCH_URL}|g" \
      -e "s|__OTEL_ENDPOINT__|${OTEL_ENDPOINT}|g" \
      k8s/gateway.yaml.tmpl
}

echo "== Deploying gateway pods (nhs/streamforge/newco) to $NEAR_CLUSTER =="
gcloud container clusters get-credentials "$NEAR_CLUSTER" --zone="$NEAR_ZONE"
render_gateway nhs client_nhs 8000 | kubectl apply -f -
render_gateway streamforge client_streamforge 8010 | kubectl apply -f -
render_gateway newco client_newco 8020 | kubectl apply -f -
kubectl rollout restart deployment/devmind-gateway-nhs deployment/devmind-gateway-streamforge deployment/devmind-gateway-newco -n devmind

echo "== Deploying gateway pod (babcock) to $FAR_CLUSTER =="
gcloud container clusters get-credentials "$FAR_CLUSTER" --zone="$FAR_ZONE"
render_gateway babcock client_babcock 8000 | kubectl apply -f -
kubectl rollout restart deployment/devmind-gateway-babcock -n devmind

echo "== Waiting for gateway external IPs =="
gcloud container clusters get-credentials "$NEAR_CLUSTER" --zone="$NEAR_ZONE"
NHS_IP="$(wait_for_lb_ip devmind-gateway-nhs)"
SF_IP="$(wait_for_lb_ip devmind-gateway-streamforge)"
NEWCO_IP="$(wait_for_lb_ip devmind-gateway-newco)"
gcloud container clusters get-credentials "$FAR_CLUSTER" --zone="$FAR_ZONE"
BABCOCK_IP="$(wait_for_lb_ip devmind-gateway-babcock)"

echo "== Deploying Prometheus + Grafana to $CLOUD_CLUSTER =="
gcloud container clusters get-credentials "$CLOUD_CLUSTER" --zone="$CLOUD_ZONE"
NEAR_TARGETS="\"${NHS_IP}:8000\", \"${SF_IP}:8010\", \"${NEWCO_IP}:8020\""
FAR_TARGETS="\"${BABCOCK_IP}:8000\""
sed -e "s|__NEAR_TARGETS__|${NEAR_TARGETS}|g" -e "s|__FAR_TARGETS__|${FAR_TARGETS}|g" \
  k8s/prometheus.yaml.tmpl | kubectl apply -f -
kubectl apply -f k8s/grafana.yaml
kubectl rollout restart deployment/prometheus deployment/grafana -n devmind 2>/dev/null || true

echo "== Waiting for Prometheus/Grafana external IPs =="
PROM_IP="$(wait_for_lb_ip prometheus)"
GRAFANA_IP="$(wait_for_lb_ip grafana)"

cat <<SUMMARY

== Deployed. ==
Orchestrator dashboard: http://${ORCH_IP}:8002
Cloud pod:              ${CLOUD_URL}
Gateway (nhs):           http://${NHS_IP}:8000
Gateway (streamforge):   http://${SF_IP}:8010
Gateway (newco):         http://${NEWCO_IP}:8020
Gateway (babcock):       http://${BABCOCK_IP}:8000
Prometheus:              http://${PROM_IP}:9090
Grafana:                 http://${GRAFANA_IP}:3000
Jaeger:                  http://${JAEGER_IP}:16686
SUMMARY
