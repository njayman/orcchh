from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, make_asgi_app
from pydantic import BaseModel

from devmind.dataset import TASK_DATASETS
from devmind.diagnosis import Diagnosis, OllamaDiagnosisProvider
from devmind.environment import ScenarioConfig
from devmind.orchestrator import EscalationDiagnosisMonitor, PolicyOrchestrator, ToleranceThresholds
from devmind.tracing import setup_tracing

_ONBOARDING_DECISIONS = Counter(
    "devmind_orchestrator_onboarding_decisions_total", "Client onboarding decisions", ["decision"]
)

_PRESETS = {
    "steady": ScenarioConfig.steady,
    "bursty": ScenarioConfig.bursty,
    "degraded_network": ScenarioConfig.degraded_network,
}

_ACTION_LOG_PATH = os.environ.get("DEVMIND_ACTION_LOG_PATH", "evaluation/request_log.jsonl")
_TAIL_POLL_INTERVAL_S = 1.5


class ClientRequest(BaseModel):
    client_id: str
    service_id: str | None = None
    task: str = "toxicity"
    scenario: str = "steady"
    base_rate: float = 4000.0
    burst_rate: float = 4000.0
    rtt_base: float = 40.0
    rtt_degraded: float = 40.0
    edge_stress_prob: float = 0.1
    edge_degrade_prob: float = 0.02
    max_samples: int = 200
    max_sla_violation_rate: float | None = None
    max_escalation_rate: float | None = None
    min_accuracy: float | None = None

    @property
    def onboarding_id(self) -> str:
        return f"{self.client_id}.{self.service_id}" if self.service_id else self.client_id


def _scenario_from_request(req: ClientRequest) -> ScenarioConfig:
    if req.task not in TASK_DATASETS:
        raise HTTPException(400, f"unknown task '{req.task}', choose from {list(TASK_DATASETS)}")
    if req.scenario in _PRESETS:
        base = _PRESETS[req.scenario](task=req.task)
        base.name = req.onboarding_id
        return base
    if req.scenario != "custom":
        raise HTTPException(400, f"unknown scenario '{req.scenario}', use steady/bursty/degraded_network/custom")
    return ScenarioConfig(
        name=req.onboarding_id,
        task=req.task,
        base_rate=req.base_rate,
        burst_rate=req.burst_rate,
        rtt_base=req.rtt_base,
        rtt_degraded=req.rtt_degraded,
        edge_stress_prob=req.edge_stress_prob,
        edge_degrade_prob=req.edge_degrade_prob,
    )


async def _broadcast(app: FastAPI, message: dict) -> None:
    dead = []
    for ws in app.state.ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        app.state.ws_clients.discard(ws)


def _log_diagnosis(orch: PolicyOrchestrator, client_id: str, diagnosis: Diagnosis, reason: str = "escalation") -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "diagnosis_generated",
        "reason": reason,
        "client": client_id,
        "summary": diagnosis.summary,
        "likely_cause": diagnosis.likely_cause,
        "resource_recommendation": diagnosis.resource_recommendation,
        "model_used": diagnosis.model_used,
        "latency_ms": diagnosis.latency_ms,
        "parsed_ok": diagnosis.parsed_ok,
    }
    os.makedirs(os.path.dirname(orch.log_path), exist_ok=True)
    with open(orch.log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _make_diagnosis_callback(app: FastAPI, orch: PolicyOrchestrator, reason: str = "escalation"):
    # onboard() (and therefore _diagnose_unreachable_threshold) runs inside
    # run_in_executor's worker thread, not the event-loop thread, so scheduling the
    # broadcast must be thread-safe. run_coroutine_threadsafe works from either thread.
    def _callback(client_id: str, diagnosis: Diagnosis) -> None:
        _log_diagnosis(orch, client_id, diagnosis, reason)
        asyncio.run_coroutine_threadsafe(
            _broadcast(
                app,
                {
                    "type": "notification",
                    "severity": "urgent",
                    "reason": reason,
                    "client": client_id,
                    "summary": diagnosis.summary,
                    "likely_cause": diagnosis.likely_cause,
                    "resource_recommendation": diagnosis.resource_recommendation,
                    "parsed_ok": diagnosis.parsed_ok,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
            app.state.loop,
        )

    return _callback


def _log_governance_review(orch: PolicyOrchestrator, client_id: str, review) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "governance_review",
        "client": client_id,
        "agrees": review.agrees,
        "recommended_decision": review.recommended_decision,
        "justification": review.justification,
        "model_used": review.model_used,
        "latency_ms": review.latency_ms,
        "parsed_ok": review.parsed_ok,
    }
    os.makedirs(os.path.dirname(orch.log_path), exist_ok=True)
    with open(orch.log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _make_governance_review_callback(app: FastAPI, orch: PolicyOrchestrator):
    # Same worker-thread caveat as _make_diagnosis_callback -- onboard() runs off
    # the event loop, so the broadcast has to be scheduled thread-safely.
    def _callback(client_id: str, review) -> None:
        _log_governance_review(orch, client_id, review)
        if review.agrees:
            return  # routine "confirmed the rule" reviews don't need a live notification
        asyncio.run_coroutine_threadsafe(
            _broadcast(
                app,
                {
                    "type": "notification",
                    "severity": "info",
                    "reason": "governance_override",
                    "client": client_id,
                    "summary": f"LLM escalated onboarding decision to '{review.recommended_decision}'",
                    "likely_cause": review.justification,
                    "resource_recommendation": "",
                    "parsed_ok": review.parsed_ok,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            ),
            app.state.loop,
        )

    return _callback


async def _tail_action_log_loop(app: FastAPI, monitor: EscalationDiagnosisMonitor, path: str) -> None:
    offset = 0
    while True:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size < offset:
                offset = 0  # file was truncated/rotated
            if size > offset:
                with open(path) as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        client_id = row.get("client", "default")
                        monitor.notify(client_id, row)
                        await _broadcast(
                            app,
                            {
                                "type": "escalation_point",
                                "client": client_id,
                                "timestamp": row.get("timestamp"),
                                "tier": row.get("tier"),
                                "action": row.get("action"),
                                "sla_met": row.get("sla_met"),
                                "latency_ms": row.get("latency_ms"),
                            },
                        )
                        if row.get("fallback_reason") == "edge_unreachable":
                            await _broadcast(
                                app,
                                {
                                    "type": "notification",
                                    "severity": "urgent",
                                    "reason": "edge_unreachable",
                                    "client": client_id,
                                    "summary": f"Edge device unreachable for client '{client_id}'; request routed to cloud",
                                    "likely_cause": "Edge inference call timed out, or the heartbeat has gone stale.",
                                    "resource_recommendation": "Check edge device connectivity/health.",
                                    "parsed_ok": True,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                    offset = f.tell()
        await asyncio.sleep(_TAIL_POLL_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ws_clients = set()
    app.state.loop = asyncio.get_running_loop()

    diagnosis_provider = OllamaDiagnosisProvider()
    orch = PolicyOrchestrator(
        library_dir=os.environ.get("DEVMIND_POLICY_LIBRARY_DIR", "policy_library"),
        log_path=os.environ.get("DEVMIND_DECISION_LOG", "evaluation/orchestrator_decisions.jsonl"),
        fine_tune_steps=int(os.environ.get("DEVMIND_FINE_TUNE_STEPS", "2000")),
        train_new_steps=int(os.environ.get("DEVMIND_TRAIN_NEW_STEPS", "8000")),
        eval_n_runs=1,
        diagnosis_provider=diagnosis_provider,
        meta_policy_path=os.environ.get("DEVMIND_META_POLICY_PATH", "meta_policy.pt"),
        meta_state_stats_path=os.environ.get("DEVMIND_META_STATE_STATS_PATH", "meta_state_stats.json"),
    )
    orch.on_threshold_diagnosis = _make_diagnosis_callback(app, orch, reason="threshold_too_tight")
    orch.on_governance_review = _make_governance_review_callback(app, orch)
    seed_path = os.environ.get("DEVMIND_POLICY_PATH", "ppo_policy.pt")
    if os.path.exists(seed_path):
        orch.register_seed_policy("seed", seed_path, validated_scenarios=["steady", "bursty", "degraded_network"])
    app.state.orchestrator = orch

    monitor = EscalationDiagnosisMonitor(diagnosis_provider, on_diagnosis=_make_diagnosis_callback(app, orch))
    monitor_task = asyncio.create_task(monitor.run_forever())
    tail_task = asyncio.create_task(_tail_action_log_loop(app, monitor, _ACTION_LOG_PATH))

    yield

    for task in (monitor_task, tail_task):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="DevMind Orchestrator Dashboard", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
setup_tracing("devmind-orchestrator", app)


@app.get("/clients")
async def list_clients() -> list[dict]:
    orch: PolicyOrchestrator = app.state.orchestrator
    return [
        {
            "policy_id": pid,
            "clients_assigned": rec.clients_assigned,
            "validated_scenarios": rec.validated_scenarios,
        }
        for pid, rec in orch.library.items()
    ]


def _fmt_pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


@app.get("/decisions", response_class=HTMLResponse)
async def list_decisions() -> str:
    orch: PolicyOrchestrator = app.state.orchestrator
    entries = []
    if os.path.exists(orch.log_path):
        with open(orch.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    rows = ""
    for row in reversed(entries):
        if "decision" not in row:
            continue
        m = (row.get("candidates_evaluated") or {}).get(row.get("policy_assigned"), {})
        rows += (
            "<tr>"
            f"<td>{row.get('timestamp')}</td><td>{row.get('client')}</td><td>{row.get('task', '-')}</td>"
            f"<td>{row.get('decision')}</td><td>{row.get('policy_assigned')}</td>"
            f"<td>{_fmt_num(m.get('accuracy'))}</td><td>{_fmt_pct(m.get('sla_violation_rate'))}</td>"
            f"<td>{_fmt_pct(m.get('escalation_rate'))}</td><td>{_fmt_num(m.get('trust_score'))}</td>"
            f"<td>{row.get('dominant_signal')}</td><td>{row.get('trigger', 'onboarding')}</td>"
            "</tr>"
        )
    return (
        '<table id="decisions-table">'
        "<thead><tr>"
        "<th>Timestamp</th><th>Client</th><th>Task</th><th>Decision</th><th>Policy assigned</th>"
        "<th>Accuracy</th><th>SLA violation</th><th>Escalation</th><th>Trust</th>"
        "<th>Dominant signal</th><th>Trigger</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


@app.get("/recalibrations", response_class=HTMLResponse)
async def list_recalibrations() -> str:
    orch: PolicyOrchestrator = app.state.orchestrator
    entries = []
    if os.path.exists(orch.log_path):
        with open(orch.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    rows = ""
    for row in reversed(entries):
        if row.get("event") != "threshold_recalibrated":
            continue
        old, new = row.get("old_thresholds", {}), row.get("new_thresholds", {})
        rows += (
            "<tr>"
            f"<td>{row.get('timestamp')}</td><td>{_fmt_pct(row.get('false_reuse_rate'))}</td>"
            f"<td>{_fmt_pct(old.get('max_sla_violation_rate'))} → {_fmt_pct(new.get('max_sla_violation_rate'))}</td>"
            f"<td>{_fmt_pct(old.get('min_accuracy'))} → {_fmt_pct(new.get('min_accuracy'))}</td>"
            f"<td>{_fmt_pct(old.get('max_escalation_rate'))} → {_fmt_pct(new.get('max_escalation_rate'))}</td>"
            f"<td>{row.get('changed')}</td>"
            "</tr>"
        )
    return (
        '<table id="recalibrations-table">'
        "<thead><tr>"
        "<th>Timestamp</th><th>False reuse rate</th><th>SLA violation cap</th>"
        "<th>Min accuracy</th><th>Escalation cap</th><th>Changed</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


@app.get("/diagnoses", response_class=HTMLResponse)
async def list_diagnoses() -> str:
    orch: PolicyOrchestrator = app.state.orchestrator
    entries = []
    if os.path.exists(orch.log_path):
        with open(orch.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    rows = ""
    for row in reversed(entries):
        if row.get("event") != "diagnosis_generated":
            continue
        ok = "yes" if row.get("parsed_ok") else "no (unparseable)"
        rows += (
            "<tr>"
            f"<td>{row.get('timestamp')}</td><td>{row.get('client')}</td>"
            f"<td>{row.get('summary')}</td><td>{row.get('likely_cause')}</td>"
            f"<td>{row.get('resource_recommendation')}</td>"
            f"<td>{row.get('model_used')}</td><td>{_fmt_num(row.get('latency_ms'))}</td><td>{ok}</td>"
            "</tr>"
        )
    return (
        '<table id="diagnoses-table">'
        "<thead><tr>"
        "<th>Timestamp</th><th>Client</th><th>Summary</th><th>Likely cause</th>"
        "<th>Resource recommendation</th><th>Model</th><th>Latency (ms)</th><th>Parsed OK</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


@app.post("/clients")
async def add_client(req: ClientRequest) -> dict:
    orch: PolicyOrchestrator = app.state.orchestrator
    scenario = _scenario_from_request(req)
    oid = req.onboarding_id
    if req.max_sla_violation_rate is not None or req.max_escalation_rate is not None or req.min_accuracy is not None:
        defaults = ToleranceThresholds()
        orch.set_client_thresholds(
            oid,
            ToleranceThresholds(
                max_sla_violation_rate=req.max_sla_violation_rate or defaults.max_sla_violation_rate,
                max_escalation_rate=req.max_escalation_rate or defaults.max_escalation_rate,
                min_accuracy=req.min_accuracy or defaults.min_accuracy,
            ),
        )
    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(None, orch.onboard, oid, scenario, req.max_samples)
    _ONBOARDING_DECISIONS.labels(decision=decision.value).inc()
    assigned = next(pid for pid, rec in orch.library.items() if oid in rec.clients_assigned)
    return {"client_id": oid, "task": req.task, "decision": decision.value, "policy_assigned": assigned}


@app.post("/log-action")
async def log_action(entry: dict) -> dict:
    # Receiving end of CascadeController's action_log_url forwarding
    # (cascade.py): lets a gateway pod in a different region/host from the
    # orchestrator still feed the live-monitoring tail loop below, which
    # otherwise only sees entries written to its own local filesystem.
    loop = asyncio.get_running_loop()

    def _append() -> None:
        os.makedirs(os.path.dirname(_ACTION_LOG_PATH) or ".", exist_ok=True)
        with open(_ACTION_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    await loop.run_in_executor(None, _append)
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    # Live monitoring depends on request_log.jsonl being visible on this
    # process's filesystem. In a co-located deployment (same host/pod) that's
    # automatic; across separate pods/VMs (e.g. the GKE multi-region path)
    # gateways must be configured with DEVMIND_ORCHESTRATOR_LOG_URL to forward
    # entries here via POST /log-action instead. Silent either way if
    # misconfigured: the tail loop just sees a file that never grows, no error
    # anywhere. Surface it here so an operator can tell "no traffic yet" apart
    # from "wrong path".
    exists = os.path.exists(_ACTION_LOG_PATH)
    return {
        "status": "ok",
        "service": "devmind-orchestrator",
        "action_log_path": _ACTION_LOG_PATH,
        "action_log_exists": exists,
        "action_log_size_bytes": os.path.getsize(_ACTION_LOG_PATH) if exists else 0,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    websocket.app.state.ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        websocket.app.state.ws_clients.discard(websocket)


_DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DevMind Policy Orchestrator</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.3rem; }
label { display: block; margin-top: 0.75rem; font-size: 0.9rem; }
input, select { width: 100%; padding: 0.4rem; box-sizing: border-box; }
button { margin-top: 1rem; padding: 0.5rem 1rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
th, td { text-align: left; border-bottom: 1px solid #ccc; padding: 0.4rem; font-size: 0.85rem; }
#result { margin-top: 1rem; font-size: 0.9rem; white-space: pre-wrap; }
#custom-fields { display: none; }
#escalation-chart { width: 100%; height: 160px; border: 1px solid #ccc; margin-top: 0.5rem; }
#ws-status { font-size: 0.8rem; color: #888; }
#notifications { list-style: none; padding: 0; margin-top: 0.5rem; }
#notifications li { border-left: 4px solid #c0392b; background: #fdecea; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
#notifications li .meta { color: #666; font-size: 0.75rem; }
</style>
</head>
<body>
<h1>DevMind Policy Orchestrator</h1>

<form id="add-form">
  <label>Client ID <input name="client_id" required></label>
  <label>Service ID (optional -- multiple services per client)<input name="service_id" placeholder="e.g. triage, sentiment-feed"></label>
  <label>Task
    <select name="task">
      <option value="toxicity">toxicity (Jigsaw)</option>
      <option value="sentiment">sentiment (SST-2)</option>
      <option value="spam">spam (SMS Spam)</option>
      <option value="topic">topic (AG News)</option>
    </select>
  </label>
  <label>Scenario
    <select name="scenario" id="scenario-select">
      <option value="steady">steady</option>
      <option value="bursty">bursty</option>
      <option value="degraded_network">degraded_network</option>
      <option value="custom">custom</option>
    </select>
  </label>
  <div id="custom-fields">
    <label>Base rate <input name="base_rate" type="number" value="4000"></label>
    <label>Burst rate <input name="burst_rate" type="number" value="4000"></label>
    <label>RTT base (ms) <input name="rtt_base" type="number" value="40"></label>
    <label>RTT degraded (ms) <input name="rtt_degraded" type="number" value="40"></label>
    <label>Edge stress prob <input name="edge_stress_prob" type="number" step="0.01" value="0.1"></label>
    <label>Edge degrade prob <input name="edge_degrade_prob" type="number" step="0.01" value="0.02"></label>
  </div>
  <button type="submit">Onboard client</button>
</form>

<div id="result"></div>

<table id="clients-table">
  <thead><tr><th>Policy</th><th>Clients</th><th>Validated scenarios</th></tr></thead>
  <tbody></tbody>
</table>

<h1>Live Monitoring <span id="ws-status">connecting...</span></h1>
<p style="font-size:0.85rem;color:#666">Rolling escalation rate per client, computed client-side from a live websocket feed. Urgent notifications fire when a client's escalation rate stays &ge;95% for a sustained window, diagnosed by a local LLM off the request path.</p>
<canvas id="escalation-chart"></canvas>
<h2 style="font-size:1rem;margin-top:1rem;">Urgent Attention</h2>
<ul id="notifications"></ul>

<h1>Decision Log</h1>
<div id="decisions-container" style="overflow-x:auto"></div>

<h1>Threshold Recalibrations</h1>
<p style="font-size:0.85rem;color:#666">Governance-layer self-improvement: tolerance thresholds adjusted from the orchestrator's own false-reuse track record, every 5 onboarding calls.</p>
<div id="recalibrations-container" style="overflow-x:auto"></div>

<h1>Diagnosis History</h1>
<p style="font-size:0.85rem;color:#666">Past LLM diagnoses (local Ollama), one row per sustained-escalation event that triggered the diagnosis monitor.</p>
<div id="diagnoses-container" style="overflow-x:auto"></div>

<script>
const scenarioSelect = document.getElementById("scenario-select");
const customFields = document.getElementById("custom-fields");
scenarioSelect.addEventListener("change", () => {
  customFields.style.display = scenarioSelect.value === "custom" ? "block" : "none";
});

async function refreshClients() {
  const res = await fetch("/clients");
  const rows = await res.json();
  const tbody = document.querySelector("#clients-table tbody");
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.policy_id}</td><td>${row.clients_assigned.join(", ")}</td><td>${row.validated_scenarios.join(", ")}</td>`;
    tbody.appendChild(tr);
  }
}

async function refreshDecisions() {
  const res = await fetch("/decisions");
  document.getElementById("decisions-container").innerHTML = await res.text();
}

async function refreshRecalibrations() {
  const res = await fetch("/recalibrations");
  document.getElementById("recalibrations-container").innerHTML = await res.text();
}

async function refreshDiagnoses() {
  const res = await fetch("/diagnoses");
  document.getElementById("diagnoses-container").innerHTML = await res.text();
}

document.getElementById("add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const body = Object.fromEntries(form.entries());
  const resultEl = document.getElementById("result");
  resultEl.textContent = "Onboarding (this can take a while for fine-tune/train-new decisions)...";
  const res = await fetch("/clients", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  resultEl.textContent = res.ok ? JSON.stringify(data, null, 2) : `Error: ${data.detail}`;
  await refreshClients();
  await refreshDecisions();
  await refreshRecalibrations();
  await refreshDiagnoses();
});

refreshClients();
refreshDecisions();
refreshRecalibrations();
refreshDiagnoses();

// --- Live monitoring: websocket-fed rolling escalation chart + urgent notifications ---
const ROLLING_WINDOW = 20;
const MAX_POINTS = 100;
const clientSeries = {};   // client -> { events: [bool escalated,...], points: [rate,...] }
const chartColors = ["#2563eb", "#c0392b", "#16a34a", "#d97706", "#7c3aed", "#0891b2"];
const clientColor = {};
let colorIdx = 0;

function colorFor(client) {
  if (!(client in clientColor)) {
    clientColor[client] = chartColors[colorIdx % chartColors.length];
    colorIdx++;
  }
  return clientColor[client];
}

function drawChart() {
  const canvas = document.getElementById("escalation-chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#eee";
  for (let i = 0; i <= 4; i++) {
    const y = h - (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const clients = Object.keys(clientSeries);
  if (clients.length === 0) {
    ctx.fillStyle = "#999"; ctx.font = "12px system-ui";
    ctx.fillText("waiting for live traffic...", 8, 20);
    return;
  }

  for (const client of clients) {
    const pts = clientSeries[client].points;
    if (pts.length < 2) continue;
    ctx.strokeStyle = colorFor(client);
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((rate, i) => {
      const x = (i / (MAX_POINTS - 1)) * w;
      const y = h - rate * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  let lx = 8, ly = 14;
  ctx.font = "11px system-ui";
  for (const client of clients) {
    ctx.fillStyle = colorFor(client);
    ctx.fillRect(lx, ly - 8, 8, 8);
    ctx.fillStyle = "#333";
    const label = `${client} (${(clientSeries[client].points.slice(-1)[0] * 100).toFixed(0)}%)`;
    ctx.fillText(label, lx + 12, ly);
    lx += ctx.measureText(label).width + 30;
  }
}

function addNotification(msg) {
  const ul = document.getElementById("notifications");
  const li = document.createElement("li");
  const ok = msg.parsed_ok === false ? " (diagnosis unparseable)" : "";
  li.innerHTML = `<strong>${msg.client}</strong>${ok}: ${msg.summary}<br>` +
    `<em>Likely cause:</em> ${msg.likely_cause}<br>` +
    `<em>Needs:</em> ${msg.resource_recommendation}` +
    `<div class="meta">${msg.timestamp}</div>`;
  ul.prepend(li);
  while (ul.children.length > 20) ul.removeChild(ul.lastChild);
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  const statusEl = document.getElementById("ws-status");

  ws.onopen = () => { statusEl.textContent = "live"; statusEl.style.color = "#16a34a"; };
  ws.onclose = () => {
    statusEl.textContent = "disconnected, retrying...";
    statusEl.style.color = "#c0392b";
    setTimeout(connectWs, 3000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "escalation_point") {
      const series = clientSeries[msg.client] || (clientSeries[msg.client] = { events: [], points: [] });
      series.events.push(msg.action === "ESCALATE_TO_CLOUD");
      if (series.events.length > ROLLING_WINDOW) series.events.shift();
      const rate = series.events.filter(Boolean).length / series.events.length;
      series.points.push(rate);
      if (series.points.length > MAX_POINTS) series.points.shift();
      drawChart();
    } else if (msg.type === "notification") {
      addNotification(msg);
      refreshDiagnoses();
    }
  };
}

connectWs();
window.addEventListener("resize", drawChart);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


def main() -> None:
    port = int(os.environ.get("DEVMIND_ORCH_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)


def demo() -> None:
    import tempfile
    from types import SimpleNamespace

    async def _run() -> None:
        class FakeWS:
            def __init__(self) -> None:
                self.received: list[dict] = []

            async def send_json(self, msg: dict) -> None:
                self.received.append(msg)

        class DeadWS:
            async def send_json(self, msg: dict) -> None:
                raise RuntimeError("connection closed")

        ws = FakeWS()
        fake_app = SimpleNamespace(state=SimpleNamespace(ws_clients={ws}))
        await _broadcast(fake_app, {"type": "ping"})
        assert ws.received == [{"type": "ping"}]

        dead = DeadWS()
        fake_app2 = SimpleNamespace(state=SimpleNamespace(ws_clients={dead}))
        await _broadcast(fake_app2, {"type": "ping"})
        assert dead not in fake_app2.state.ws_clients, "a socket that raises on send must be dropped"

        with tempfile.TemporaryDirectory() as tmp:
            log_path = f"{tmp}/decisions.jsonl"
            orch = SimpleNamespace(log_path=log_path)
            diagnosis = Diagnosis(
                summary="s", likely_cause="c", resource_recommendation="r",
                model_used="m", latency_ms=42.0, raw_response="{}",
            )
            _log_diagnosis(orch, "client_babcock", diagnosis)
            with open(log_path) as f:
                row = json.loads(f.readline())
            assert row["event"] == "diagnosis_generated"
            assert row["client"] == "client_babcock"
            assert row["summary"] == "s"

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/request_log.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps({"client": "c1", "action": "ESCALATE_TO_CLOUD", "sla_met": False}) + "\n")

            notified: list[tuple[str, dict]] = []

            class FakeMonitor:
                def notify(self, client_id: str, row: dict) -> None:
                    notified.append((client_id, row))

            ws3 = FakeWS()
            fake_app3 = SimpleNamespace(state=SimpleNamespace(ws_clients={ws3}))
            task = asyncio.create_task(_tail_action_log_loop(fake_app3, FakeMonitor(), path))
            for _ in range(50):
                if notified:
                    break
                await asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert notified == [("c1", {"client": "c1", "action": "ESCALATE_TO_CLOUD", "sla_met": False})]
            assert any(m.get("type") == "escalation_point" for m in ws3.received)

        # Deadman's-switch row (cascade.py's fallback_reason="edge_unreachable") must
        # fire an urgent notification straight away, not wait on the sustained-window
        # escalation monitor.
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/request_log.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps({
                    "client": "c2", "action": "ESCALATE_TO_CLOUD", "sla_met": False,
                    "operational_state": "UNREACHABLE", "fallback_reason": "edge_unreachable",
                }) + "\n")

            ws4 = FakeWS()
            fake_app4 = SimpleNamespace(state=SimpleNamespace(ws_clients={ws4}))
            task = asyncio.create_task(_tail_action_log_loop(fake_app4, FakeMonitor(), path))
            for _ in range(50):
                if any(m.get("type") == "notification" for m in ws4.received):
                    break
                await asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            urgent = [m for m in ws4.received if m.get("type") == "notification"]
            assert len(urgent) == 1, "edge-unreachable row must fire exactly one urgent notification"
            assert urgent[0]["severity"] == "urgent"
            assert urgent[0]["reason"] == "edge_unreachable"
            assert urgent[0]["client"] == "c2"

    asyncio.run(_run())
    print("orchestrator_app self-check passed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
