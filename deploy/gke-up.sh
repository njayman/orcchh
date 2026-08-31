#!/usr/bin/env bash
# One-command provision of the 3-region GKE eval stack, mirroring gcp-up.sh's
# topology but real Kubernetes: one small zonal GKE cluster per region instead
# of a bare VM. A single K8s cluster can't span these 3 regions (etcd isn't
# built for ~250ms cross-region RTT), so each region gets its own cluster;
# cross-region calls stay at the HTTP layer exactly as they already do in the
# Docker path (CloudClient -> DEVMIND_CLOUD_URL).
# Run from the `code/` directory. Safe to re-run: skips clusters that already exist.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGISTRY="europe-west2-docker.pkg.dev/${PROJECT}/devmind"

CLOUD_ZONE="europe-west2-a"
NEAR_ZONE="europe-west4-a"
FAR_ZONE="australia-southeast1-a"

CLOUD_CLUSTER="devmind-cloud-gke"
NEAR_CLUSTER="devmind-edge-near-gke"
FAR_CLUSTER="devmind-edge-far-gke"

create_cluster() {
  local name="$1" zone="$2" machine="$3"
  if gcloud container clusters describe "$name" --zone="$zone" &>/dev/null; then
    echo "== $name already exists, skipping create =="
    return
  fi
  echo "== Creating GKE cluster $name in $zone ($machine, 1 node) =="
  gcloud container clusters create "$name" \
    --zone="$zone" --machine-type="$machine" --num-nodes=1 \
    --disk-size=50
}

create_cluster "$CLOUD_CLUSTER" "$CLOUD_ZONE" e2-standard-2
create_cluster "$NEAR_CLUSTER" "$NEAR_ZONE" e2-standard-2
create_cluster "$FAR_CLUSTER" "$FAR_ZONE" e2-standard-2

apply_base() {
  local name="$1" zone="$2"
  echo "== Applying namespace/configmap to $name =="
  gcloud container clusters get-credentials "$name" --zone="$zone"
  kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml
}

apply_base "$CLOUD_CLUSTER" "$CLOUD_ZONE"
apply_base "$NEAR_CLUSTER" "$NEAR_ZONE"
apply_base "$FAR_CLUSTER" "$FAR_ZONE"

echo "== Base infra up. Run deploy/gke-update.sh to build, push and deploy the app. =="
