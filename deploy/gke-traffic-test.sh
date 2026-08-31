#!/usr/bin/env bash
# Replay the same 6-block/60-min live traffic schedule as gcp-traffic-test.sh
# (see evaluation/multi-region-live-traffic-test-2026-08-05.md) against
# the GKE stack from gke-up.sh/gke-update.sh. Unlike the Docker path, the
# gateway pods are reachable directly over their LoadBalancer external IPs,
# so this runs locally (no SSH-and-remote-curl needed) and logs straight to
# /tmp/traffic-<client>.csv.
#
# BLOCK_SECONDS controls block length -- override for the smoke-test /
# validation passes before committing to the full 600s (10 min) blocks:
#   BLOCK_SECONDS=60  ./deploy/gke-traffic-test.sh   # ~1 min viability run
#   BLOCK_SECONDS=1200 ./deploy/gke-traffic-test.sh  # scaled-up full run
set -euo pipefail

CLOUD_ZONE="europe-west2-a"; NEAR_ZONE="europe-west4-a"; FAR_ZONE="australia-southeast1-a"
NEAR_CLUSTER="devmind-edge-near-gke"; FAR_CLUSTER="devmind-edge-far-gke"

BLOCK_SECONDS="${BLOCK_SECONDS:-600}"
NEAR_BLOCKS=(8000,8010,8020 8010 8000 8010,8020 8000,8010,8020 "")
FAR_BLOCKS=(8000 "" 8000 "" 8000 8000)

echo "== Looking up gateway external IPs =="
gcloud container clusters get-credentials "$NEAR_CLUSTER" --zone="$NEAR_ZONE"
NHS_IP="$(kubectl get svc devmind-gateway-nhs -n devmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
SF_IP="$(kubectl get svc devmind-gateway-streamforge -n devmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
NEWCO_IP="$(kubectl get svc devmind-gateway-newco -n devmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
gcloud container clusters get-credentials "$FAR_CLUSTER" --zone="$FAR_ZONE"
BABCOCK_IP="$(kubectl get svc devmind-gateway-babcock -n devmind -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

declare -A HOST=( [8000_near]="$NHS_IP" [8010_near]="$SF_IP" [8020_near]="$NEWCO_IP" [8000_far]="$BABCOCK_IP" )
declare -A CLIENT=( [8000_near]="client_nhs" [8010_near]="client_streamforge" [8020_near]="client_newco" [8000_far]="client_babcock" )

TEXTS=("this is a great update" "why is this so slow today" "I disagree strongly with this decision"
       "thanks for the quick response" "this behavior seems broken" "can we escalate this ticket")

# Truncate each client's log once, up front -- clients (nhs/streamforge/newco)
# run across multiple non-adjacent blocks, and truncating inside fire() would
# wipe out earlier blocks' data every time that client reactivates.
for client in client_nhs client_streamforge client_newco client_babcock; do
  : > "/tmp/traffic-${client}.csv"
done

fire() {
  local region="$1" port="$2"
  local host="${HOST[${port}_${region}]}" client="${CLIENT[${port}_${region}]}"
  local log="/tmp/traffic-${client}.csv"
  local end=$(( $(date +%s) + BLOCK_SECONDS ))
  while [ "$(date +%s)" -lt "$end" ]; do
    txt="${TEXTS[$((RANDOM % ${#TEXTS[@]}))]}"
    t0=$(date +%s%3N)
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -X POST "http://${host}:${port}/infer" \
      -H 'Content-Type: application/json' \
      -d "{\"text\": \"${txt}\"}" || echo "000")
    t1=$(date +%s%3N)
    echo "$(date -u +%FT%TZ),${code},$((t1 - t0))" >> "$log"
    sleep "0.$((RANDOM % 700 + 300))"
  done
}

run_block() {
  local region="$1" ports_csv="$2"
  [ -z "$ports_csv" ] && return 0
  IFS=',' read -ra ports <<< "$ports_csv"
  for p in "${ports[@]}"; do
    fire "$region" "$p" &
  done
}

for i in "${!NEAR_BLOCKS[@]}"; do
  echo "== Block $((i + 1))/6 (${BLOCK_SECONDS}s): near=[${NEAR_BLOCKS[$i]}] far=[${FAR_BLOCKS[$i]}] =="
  run_block near "${NEAR_BLOCKS[$i]}"
  run_block far "${FAR_BLOCKS[$i]}"
  wait
done

echo "== Summary =="
for client in client_nhs client_streamforge client_newco client_babcock; do
  log="/tmp/traffic-${client}.csv"
  [ -s "$log" ] || continue
  awk -F, -v c="$client" '
    { n++; if ($2 != "200") errs++; sum += $3 }
    END { printf "%-22s requests=%-6d errors=%-4d avg_latency_ms=%.1f\n", c, n, errs+0, (n ? sum/n : 0) }
  ' "$log"
done
echo "== Done. Per-request logs in /tmp/traffic-<client>.csv =="
