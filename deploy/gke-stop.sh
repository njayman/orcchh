#!/usr/bin/env bash
# Pause the GKE stack from gke-up.sh without losing it: scale each cluster's
# node pool to 0, which stops node compute billing while keeping the cluster
# control plane and every Deployment/Service/ConfigMap intact. Mirrors
# gcp-stop.sh's role for the Docker/VM path. Resume with gke-start.sh.
set -euo pipefail

gcloud container clusters resize devmind-cloud-gke --zone=europe-west2-a --num-nodes=0 --quiet
gcloud container clusters resize devmind-edge-near-gke --zone=europe-west4-a --num-nodes=0 --quiet
gcloud container clusters resize devmind-edge-far-gke --zone=australia-southeast1-a --num-nodes=0 --quiet

echo "== Stopped. Resume with deploy/gke-start.sh =="
