#!/usr/bin/env bash
# Replay the 6-block/60-min live traffic schedule from the 2026-08-05 run
# (docs/evaluation/multi-region-live-traffic-test-2026-08-05.md) against an
# already-running stack. Does not create instances -- run gcp-up.sh first.
#
# Each block is 10 min. Per active client, a background curl loop on its
# own VM hits POST /infer at ~1-3 req/s. Results are logged remotely to
# ~/devmind-code/traffic-<client>.csv and pulled back + summarized at the end.
set -euo pipefail

NEAR_ZONE="europe-west1-b"
FAR_ZONE="australia-southeast1-a"
NEAR_NAME="devmind-edge-near"
FAR_NAME="devmind-edge-far"

BLOCK_SECONDS="${BLOCK_SECONDS:-600}"  # override with a small number for a smoke run

# duration is BLOCK_SECONDS for every block; each entry is the comma-separated
# ports active on that VM for that block (empty = idle this block).
NEAR_BLOCKS=(8000,8010,8020 8010 8000 8010,8020 8000,8010,8020 "")
FAR_BLOCKS=(8000 "" 8000 "" 8000 8000)

remote_runner() {
  # $1 = space-separated "block:ports" list for this host
  cat <<'RUNNER'
set -e
cd ~/devmind-code
TEXTS=("this is a great update" "why is this so slow today" "I disagree strongly with this decision"
       "thanks for the quick response" "this behavior seems broken" "can we escalate this ticket")
fire() {
  port="$1"; client="$2"
  log="traffic-${client}.csv"
  : > "$log"
  end=$(( $(date +%s) + BLOCK_SECONDS ))
  while [ "$(date +%s)" -lt "$end" ]; do
    txt="${TEXTS[$((RANDOM % ${#TEXTS[@]}))]}"
    t0=$(date +%s%3N)
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -X POST "http://localhost:${port}/infer" \
      -H 'Content-Type: application/json' \
      -d "{\"text\": \"${txt}\"}" || echo "000")
    t1=$(date +%s%3N)
    echo "$(date -u +%FT%TZ),${code},$((t1 - t0))" >> "$log"
    sleep "0.$((RANDOM % 700 + 300))"
  done
}
RUNNER
}

run_block_on_host() {
  local name="$1" zone="$2" ports_csv="$3" client_map="$4"
  [ -z "$ports_csv" ] && return 0
  local cmds=""
  IFS=',' read -ra ports <<< "$ports_csv"
  for p in "${ports[@]}"; do
    client=$(echo "$client_map" | tr ' ' '\n' | grep ":${p}$" | cut -d: -f1)
    cmds+="fire ${p} ${client} & "
  done
  cmds+="wait"
  gcloud compute ssh "$name" --zone="$zone" --command="
    export BLOCK_SECONDS=${BLOCK_SECONDS}
    $(remote_runner)
    ${cmds}
  " &
}

NEAR_CLIENTS="client_nhs:8000 client_streamforge:8010 client_newco:8020"
FAR_CLIENTS="client_babcock:8000"

for i in "${!NEAR_BLOCKS[@]}"; do
  echo "== Block $((i + 1))/6 (${BLOCK_SECONDS}s): near=[${NEAR_BLOCKS[$i]}] far=[${FAR_BLOCKS[$i]}] =="
  run_block_on_host "$NEAR_NAME" "$NEAR_ZONE" "${NEAR_BLOCKS[$i]}" "$NEAR_CLIENTS"
  run_block_on_host "$FAR_NAME" "$FAR_ZONE" "${FAR_BLOCKS[$i]}" "$FAR_CLIENTS"
  wait
done

echo "== Pulling logs and summarizing =="
summarize() {
  local name="$1" zone="$2" clients="$3"
  for pair in $clients; do
    client="${pair%%:*}"
    gcloud compute ssh "$name" --zone="$zone" --command="cat ~/devmind-code/traffic-${client}.csv 2>/dev/null || true" \
      > "/tmp/traffic-${client}.csv" 2>/dev/null || true
    if [ -s "/tmp/traffic-${client}.csv" ]; then
      awk -F, -v c="$client" '
        { n++; if ($2 != "200") errs++; sum += $3 }
        END { printf "%-22s requests=%-6d errors=%-4d avg_latency_ms=%.1f\n", c, n, errs+0, (n ? sum/n : 0) }
      ' "/tmp/traffic-${client}.csv"
    fi
  done
}
summarize "$NEAR_NAME" "$NEAR_ZONE" "$NEAR_CLIENTS"
summarize "$FAR_NAME" "$FAR_ZONE" "$FAR_CLIENTS"

echo "== Done. Per-request logs in /tmp/traffic-<client>.csv =="
