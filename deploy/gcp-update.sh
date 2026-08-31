#!/usr/bin/env bash
# Ship local code changes to an already-running stack and rebuild/restart
# containers in place. Does NOT create instances or touch firewall rules --
# run gcp-up.sh first if the stack isn't up yet.
set -euo pipefail

CLOUD_ZONE="europe-west2-a"
NEAR_ZONE="europe-west1-b"
FAR_ZONE="australia-southeast1-a"

CLOUD_NAME="devmind-cloud"
NEAR_NAME="devmind-edge-near"
FAR_NAME="devmind-edge-far"

for pair in "$CLOUD_NAME:$CLOUD_ZONE" "$NEAR_NAME:$NEAR_ZONE" "$FAR_NAME:$FAR_ZONE"; do
  name="${pair%%:*}"; zone="${pair##*:}"
  gcloud compute instances describe "$name" --zone="$zone" &>/dev/null || {
    echo "$name doesn't exist -- run gcp-up.sh first." >&2
    exit 1
  }
done

CLOUD_IP="$(gcloud compute instances describe "$CLOUD_NAME" --zone="$CLOUD_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo "== Shipping code (tracked files + ppo_policy.pt) =="
ship_code() {
  local name="$1" zone="$2"
  tar -czf - -T <(git ls-files) ppo_policy.pt | \
    gcloud compute ssh "$name" --zone="$zone" --command="mkdir -p ~/devmind-code && tar xzf - -C ~/devmind-code"
}
ship_code "$CLOUD_NAME" "$CLOUD_ZONE"
ship_code "$NEAR_NAME" "$NEAR_ZONE"
ship_code "$FAR_NAME" "$FAR_ZONE"

echo "== Rebuilding + restarting containers on $CLOUD_NAME =="
gcloud compute ssh "$CLOUD_NAME" --zone="$CLOUD_ZONE" --command="
  set -e
  cd ~/devmind-code
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.cloud -t devmind-cloud .
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.orchestrator -t devmind-orchestrator .
  sudo docker rm -f devmind-cloud devmind-orchestrator 2>/dev/null || true
  sudo docker run -d --restart unless-stopped -p 8001:8001 --name devmind-cloud devmind-cloud
  mkdir -p ~/devmind-state/policy_library ~/devmind-state/evaluation && sudo docker run -d --restart unless-stopped -p 8002:8002 -v ~/devmind-state/policy_library:/app/policy_library -v ~/devmind-state/evaluation:/app/evaluation --name devmind-orchestrator devmind-orchestrator
"

echo "== Rebuilding + restarting containers on $NEAR_NAME =="
gcloud compute ssh "$NEAR_NAME" --zone="$NEAR_ZONE" --command="
  set -e
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

echo "== Rebuilding + restarting container on $FAR_NAME =="
gcloud compute ssh "$FAR_NAME" --zone="$FAR_ZONE" --command="
  set -e
  cd ~/devmind-code
  DOCKER_BUILDKIT=1 sudo -E docker build -q -f Dockerfile.gateway -t devmind-gateway .
  sudo docker rm -f gw-babcock 2>/dev/null || true
  sudo docker run -d --restart unless-stopped -p 8000:8000 \
    -e DEVMIND_CLOUD_URL=http://${CLOUD_IP}:8001 -e DEVMIND_CLIENT_ID=client_babcock -e DEVMIND_PORT=8000 \
    --name gw-babcock devmind-gateway
"

echo "== Updated. Dashboard: http://${CLOUD_IP}:8002 =="
