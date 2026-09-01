from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.cascade import CascadeController
from devmind.edge import EdgeDevice, MiscalibrationClassifier, ResourceMonitor
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, MetricSource, SilverEnricher
from devmind.model_clients import CloudClient, DistilBERTEdge
from devmind.tracing import setup_tracing

_REQUESTS = Counter("devmind_gateway_requests_total", "Requests handled", ["tier", "client"])
_LATENCY_BUCKETS_MS = (10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, float("inf"))
_LATENCY = Histogram(
    "devmind_gateway_latency_ms", "End-to-end request latency (ms)", ["client"], buckets=_LATENCY_BUCKETS_MS
)
_SLA_VIOLATIONS = Counter("devmind_gateway_sla_violations_total", "Requests that missed their SLA budget", ["client"])
_FALLBACKS = Counter("devmind_gateway_fallbacks_total", "Requests routed by the safety fallback, not the learned policy", ["client"])

_DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ppo_policy.pt")
_TRAFFIC_WINDOW_S = 60.0
_HEARTBEAT_INTERVAL_S = 2.0


def _load_policy() -> PPONetwork:
    policy = PPONetwork()
    path = os.environ.get("DEVMIND_POLICY_PATH", _DEFAULT_POLICY_PATH)
    if os.path.exists(path):
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        raise RuntimeError(
            f"No trained policy found at {path}. Train one with `python -m devmind.trainer` "
            "or set DEVMIND_POLICY_PATH, before starting the gateway."
        )
    return policy


class InferenceRequest(BaseModel):
    text: str
    sla_budget_ms: float = 300.0
    true_label: int | None = None


class InferenceResponse(BaseModel):
    request_id: str
    tier: str
    confident: bool
    sla_met: bool
    latency_ms: float
    fallback_triggered: bool


def _traffic_intensity(app: FastAPI) -> float:
    override = os.environ.get("DEVMIND_TRAFFIC_OVERRIDE")
    if override is not None:
        return float(override)
    now = time.time()
    times = [t for t in app.state.request_times if now - t < _TRAFFIC_WINDOW_S]
    app.state.request_times = times
    return float(len(times)) * (60.0 / _TRAFFIC_WINDOW_S)


async def _heartbeat_loop(edge: EdgeDevice, monitor: ResourceMonitor) -> None:
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        edge.heartbeat(monitor.sample())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.request_times = []
    task = os.environ.get("DEVMIND_TASK", "toxicity")
    edge_model = DistilBERTEdge(task=task)
    cloud_client = CloudClient(base_url=os.environ.get("DEVMIND_CLOUD_URL", "http://localhost:8001"))
    edge = EdgeDevice(classifier=MiscalibrationClassifier.from_env())
    registry = DynamicMetricRegistry()
    registry.register(MetricSource("edge_context", "EdgeContextReport", lambda: edge.last_report or edge.emit_report(0.5)))
    registry.register(MetricSource("cloud_queue_depth", "int", lambda: cloud_client.inflight))
    registry.register(MetricSource("rtt_ms", "float", lambda: cloud_client.rtt_ms))
    registry.register(MetricSource("energy_mj", "float", lambda: float(os.environ.get("DEVMIND_ENERGY_MJ", 0.0))))
    registry.register(MetricSource("traffic_intensity", "float", lambda: _traffic_intensity(app)))

    silver = SilverEnricher()
    gold = GoldNormalizer()
    stats_path = os.environ.get(
        "DEVMIND_STATE_STATS_PATH", os.path.join(os.path.dirname(_DEFAULT_POLICY_PATH), "state_stats.json")
    )
    agent = AgenticOrchestrator(_load_policy(), state_stats_path=stats_path)
    client_id = os.environ.get("DEVMIND_CLIENT_ID", "default")
    action_log_path = os.environ.get("DEVMIND_ACTION_LOG_PATH", "evaluation/request_log.jsonl")
    orchestrator_url = os.environ.get("DEVMIND_ORCHESTRATOR_URL")
    action_log_url = f"{orchestrator_url}/log-action" if orchestrator_url else None
    controller = CascadeController(
        agent, edge, registry, silver, gold, edge_model, cloud_client,
        client_id=client_id, action_log_path=action_log_path, action_log_url=action_log_url,
    )
    app.state.controller = controller
    app.state.edge = edge

    heartbeat_task = asyncio.create_task(_heartbeat_loop(edge, ResourceMonitor()))
    yield
    heartbeat_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await heartbeat_task


app = FastAPI(title="DevMind Gateway", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
setup_tracing(f"devmind-gateway-{os.environ.get('DEVMIND_CLIENT_ID', 'default')}", app)


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest) -> InferenceResponse:
    controller: CascadeController = app.state.controller
    app.state.request_times.append(time.time())
    outcome = await controller.process(str(uuid.uuid4()), req.text, req.sla_budget_ms, req.true_label)
    client = controller.client_id
    _REQUESTS.labels(tier=outcome.tier, client=client).inc()
    _LATENCY.labels(client=client).observe(outcome.latency_ms)
    if not outcome.sla_met:
        _SLA_VIOLATIONS.labels(client=client).inc()
    if outcome.fallback_triggered:
        _FALLBACKS.labels(client=client).inc()
    return InferenceResponse(
        request_id=outcome.request_id,
        tier=outcome.tier,
        confident=outcome.accuracy > 0.6,
        sla_met=outcome.sla_met,
        latency_ms=outcome.latency_ms,
        fallback_triggered=outcome.fallback_triggered,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-gateway"}


@app.get("/actions")
async def recent_actions(n: int = 50) -> list[dict]:
    controller: CascadeController = app.state.controller
    path = controller.action_log_path
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    return [json.loads(line) for line in lines[-n:]]


@app.get("/edge/status")
async def edge_status() -> dict:
    edge: EdgeDevice = app.state.edge
    report = edge.last_report
    return {
        "unreachable": edge.is_unreachable,
        "operational_state": report.operational_state.value if report else None,
        "resource_stress": vars(report.resource_stress) if report else None,
    }


def main() -> None:
    port = int(os.environ.get("DEVMIND_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
