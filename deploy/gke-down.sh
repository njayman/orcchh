#!/usr/bin/env bash
# Tear down all 3 GKE clusters created by gke-up.sh. Mirrors gcp-down.sh.
set -euo pipefail

gcloud container clusters delete devmind-cloud-gke --zone=europe-west2-a --quiet || true
gcloud container clusters delete devmind-edge-near-gke --zone=europe-west1-b --quiet || true
gcloud container clusters delete devmind-edge-far-gke --zone=australia-southeast1-a --quiet || true
