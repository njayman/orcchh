# Deploying to AWS for testing

Target: a single EC2 instance running Minikube (matches the project's
proposed tech stack — Kubernetes/Minikube — without standing up a full EKS
cluster for a test deploy).

## 1. Launch the EC2 instance

- Ubuntu 22.04+, **t3.xlarge or bigger** (BERT-large + DistilBERT both loaded
  in-process across the two pods need headroom; t3.medium will swap).
- Security group: allow inbound SSH (22) from your IP. No other inbound ports
  needed — access goes through an SSH tunnel (see below).
- Attach an internet gateway/NAT so the pods can pull models from Hugging
  Face Hub at startup.

## 2. Copy the repo and run the bootstrap script

```bash
scp -r code/ <ec2-user>@<ec2-public-ip>:~/devmind
ssh <ec2-user>@<ec2-public-ip>
cd ~/devmind
./deploy/ec2-minikube-setup.sh
```

Re-run the script if it exits after installing Docker (needed once, for the
`docker` group membership to take effect).

## 3. Reach the gateway from your laptop

```bash
# on the EC2 instance
kubectl port-forward -n devmind svc/devmind-gateway 8000:8000 &

# on your laptop
ssh -L 8000:localhost:8000 <ec2-user>@<ec2-public-ip>
curl -X POST http://localhost:8000/infer -H 'Content-Type: application/json' \
  -d '{"text": "some comment text", "sla_budget_ms": 300, "true_label": 0}'
```

`true_label` is optional — pass it when replaying labeled Jigsaw rows to get
real accuracy instead of the confidence-based proxy.

## Test knobs

`k8s/configmap.yaml` (`devmind-test-knobs`) exposes two env vars for
replaying scenarios without real load or hardware:

- `DEVMIND_ENERGY_MJ` — fixed energy reading (no RAPL access on an EC2 VM).
- `DEVMIND_TRAFFIC_OVERRIDE` — pins `traffic_intensity` instead of the real
  rolling requests/min counter, for bursty-traffic scenario testing.

Edit the ConfigMap and `kubectl apply -f k8s/configmap.yaml && kubectl
rollout restart deployment/devmind-gateway -n devmind` to pick up changes.

## Local smoke test without Kubernetes

`docker-compose.yml` runs the same two images for a quick sanity check
before touching EC2:

```bash
docker compose up --build
curl http://localhost:8000/health
```
