#!/usr/bin/env bash
# Resume a stack paused with gke-stop.sh: scale each cluster's node pool back
# to its original size. Pods reschedule automatically once nodes are back;
# no rebuild or re-apply needed, since nothing was deleted, only scaled down.
set -euo pipefail

gcloud container clusters resize devmind-cloud-gke --zone=europe-west2-a --num-nodes=1 --quiet
gcloud container clusters resize devmind-edge-near-gke --zone=europe-west4-a --num-nodes=1 --quiet
gcloud container clusters resize devmind-edge-far-gke --zone=australia-southeast1-a --num-nodes=1 --quiet

echo "== Resumed. Pods will reschedule shortly; check with 'kubectl get pods -n devmind' per cluster. =="
