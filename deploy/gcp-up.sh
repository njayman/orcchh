#!/usr/bin/env bash
# One-command provision + deploy for the multi-region devmind eval stack:
# devmind-cloud (BERT-large pod + orchestrator dashboard), devmind-edge-near
# (client_nhs, client_streamforge, client_newco), devmind-edge-far (client_babcock).
# Run from the `code/` directory. Safe to re-run: skips instances that already exist.
set -euo pipefail

CLOUD_ZONE="europe-west2-a"
NEAR_ZONE="europe-west1-b"
FAR_ZONE="australia-southeast1-a"

CLOUD_NAME="devmind-cloud"
NEAR_NAME="devmind-edge-near"
FAR_NAME="devmind-edge-far"

MY_IP_RAW="${MY_IP:-$(curl -4 -s --max-time 5 https://ifconfig.me || true)}"
if [[ ! "$MY_IP_RAW" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
  echo "Could not determine your public IP (got: '$MY_IP_RAW')." >&2
  echo "Set it explicitly and re-run, e.g.:  MY_IP=1.2.3.4 ./deploy/gcp-up.sh" >&2
  exit 1
fi
MY_IP="${MY_IP_RAW}/32"

region_of() { echo "${1%-*}"; }

ensure_static_ip() {
  local addr_name="$1" region="$2"
  gcloud compute addresses describe "$addr_name" --region="$region" &>/dev/null || \
    gcloud compute addresses create "$addr_name" --region="$region"
  gcloud compute addresses describe "$addr_name" --region="$region" --format='get(address)'
}

create_instance() {
  local name="$1" zone="$2" machine="$3"; shift 3
  if gcloud compute instances describe "$name" --zone="$zone" &>/dev/null; then
    echo "== $name already exists, skipping create =="
    return
  fi
  local addr_ip
  addr_ip="$(ensure_static_ip "${name}-ip" "$(region_of "$zone")")"
  echo "== Creating $name in $zone (static IP $addr_ip) =="
  gcloud compute instances create "$name" \
    --zone="$zone" --machine-type="$machine" \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB --address="$addr_ip" "$@"
}

create_instance "$CLOUD_NAME" "$CLOUD_ZONE" e2-standard-2 --tags=devmind-cloud
create_instance "$NEAR_NAME" "$NEAR_ZONE" e2-standard-4
create_instance "$FAR_NAME" "$FAR_ZONE" e2-standard-2

echo "== Waiting for SSH to come up =="
wait_for_ssh() {
  local name="$1" zone="$2"
  for _ in $(seq 1 30); do
    if gcloud compute ssh "$name" --zone="$zone" --command="true" &>/dev/null; then
      return 0
    fi
    sleep 10
  done
  echo "Timed out waiting for SSH on $name" >&2
  exit 1
}
wait_for_ssh "$CLOUD_NAME" "$CLOUD_ZONE"
wait_for_ssh "$NEAR_NAME" "$NEAR_ZONE"
wait_for_ssh "$FAR_NAME" "$FAR_ZONE"

CLOUD_IP="$(gcloud compute instances describe "$CLOUD_NAME" --zone="$CLOUD_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
NEAR_IP="$(gcloud compute instances describe "$NEAR_NAME" --zone="$NEAR_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
FAR_IP="$(gcloud compute instances describe "$FAR_NAME" --zone="$FAR_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo "== Locking down firewall to just these hosts =="
create_or_update_firewall() {
  local name="$1" port="$2" ranges="$3"
  if gcloud compute firewall-rules describe "$name" &>/dev/null; then
    gcloud compute firewall-rules update "$name" --source-ranges="$ranges"
  else
    gcloud compute firewall-rules create "$name" \
      --allow="tcp:$port" --target-tags=devmind-cloud --source-ranges="$ranges"
  fi
}
create_or_update_firewall devmind-cloud-access 8001 "${NEAR_IP}/32,${FAR_IP}/32"
create_or_update_firewall devmind-dashboard-access 8002 "$MY_IP"

echo "== Shipping code (tracked files + ppo_policy.pt) =="
ship_code() {
  local name="$1" zone="$2"
  tar -czf - -T <(git ls-files) ppo_policy.pt | \
    gcloud compute ssh "$name" --zone="$zone" --command="mkdir -p ~/devmind-code && tar xzf - -C ~/devmind-code"
}
ship_code "$CLOUD_NAME" "$CLOUD_ZONE"
ship_code "$NEAR_NAME" "$NEAR_ZONE"
ship_code "$FAR_NAME" "$FAR_ZONE"

echo "== Building + running containers on $CLOUD_NAME =="
gcloud compute ssh "$CLOUD_NAME" --zone="$CLOUD_ZONE" --command="
  set -e
  command -v docker &>/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq docker.io)
  cd ~/devmind-code
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.cloud -t devmind-cloud .
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.orchestrator -t devmind-orchestrator .
  sudo docker rm -f devmind-cloud devmind-orchestrator 2>/dev/null || true
  sudo docker run -d --restart unless-stopped -p 8001:8001 --name devmind-cloud devmind-cloud
  mkdir -p ~/devmind-state/policy_library ~/devmind-state/evaluation && sudo docker run -d --restart unless-stopped -p 8002:8002 -v ~/devmind-state/policy_library:/app/policy_library -v ~/devmind-state/evaluation:/app/evaluation --name devmind-orchestrator devmind-orchestrator
"

echo "== Building + running containers on $NEAR_NAME (nhs, streamforge, newco) =="
gcloud compute ssh "$NEAR_NAME" --zone="$NEAR_ZONE" --command="
  set -e
  command -v docker &>/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq docker.io)
  cd ~/devmind-code
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.gateway -t devmind-gateway .
  for spec in 'gw-nhs:8000:client_nhs' 'gw-streamforge:8010:client_streamforge' 'gw-newco:8020:client_newco'; do
    IFS=: read -r cname port client <<< \"\$spec\"
    sudo docker rm -f \"\$cname\" 2>/dev/null || true
    sudo docker run -d --restart unless-stopped -p \"\$port:\$port\" \
      -e DEVMIND_CLOUD_URL=http://${CLOUD_IP}:8001 -e DEVMIND_CLIENT_ID=\"\$client\" -e DEVMIND_PORT=\"\$port\" \
      --name \"\$cname\" devmind-gateway
  done
"

echo "== Building + running container on $FAR_NAME (babcock) =="
gcloud compute ssh "$FAR_NAME" --zone="$FAR_ZONE" --command="
  set -e
  command -v docker &>/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq docker.io)
  cd ~/devmind-code
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.gateway -t devmind-gateway .
  sudo docker rm -f gw-babcock 2>/dev/null || true
  sudo docker run -d --restart unless-stopped -p 8000:8000 \
    -e DEVMIND_CLOUD_URL=http://${CLOUD_IP}:8001 -e DEVMIND_CLIENT_ID=client_babcock -e DEVMIND_PORT=8000 \
    --name gw-babcock devmind-gateway
"

cat <<SUMMARY

== Up. ==
Dashboard (add clients/scenarios): http://${CLOUD_IP}:8002
  Only reachable from your current IP (${MY_IP%/32}). If that IP changes later, re-run this
  script or update the devmind-dashboard-access firewall rule's source-ranges.

Gateways aren't public — tunnel to reach them:
  near (nhs :8000, streamforge :8010, newco :8020):
    gcloud compute ssh $NEAR_NAME --zone=$NEAR_ZONE -- -L 8000:localhost:8000 -L 8010:localhost:8010 -L 8020:localhost:8020
  far (babcock :8000):
    gcloud compute ssh $FAR_NAME --zone=$FAR_ZONE -- -L 8000:localhost:8000

Run ./deploy/gcp-down.sh to tear everything down and stop billing.
SUMMARY
