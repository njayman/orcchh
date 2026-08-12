from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from devmind.dataset import TASK_DATASETS
from devmind.environment import ScenarioConfig
from devmind.orchestrator import PolicyOrchestrator

_PRESETS = {
    "steady": ScenarioConfig.steady,
    "bursty": ScenarioConfig.bursty,
    "degraded_network": ScenarioConfig.degraded_network,
}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = PolicyOrchestrator(
        library_dir=os.environ.get("DEVMIND_POLICY_LIBRARY_DIR", "policy_library"),
        log_path=os.environ.get("DEVMIND_DECISION_LOG", "docs/evaluation/orchestrator_decisions.jsonl"),
        fine_tune_steps=int(os.environ.get("DEVMIND_FINE_TUNE_STEPS", "2000")),
        train_new_steps=int(os.environ.get("DEVMIND_TRAIN_NEW_STEPS", "8000")),
        eval_n_runs=1,
    )
    seed_path = os.environ.get("DEVMIND_POLICY_PATH", "ppo_policy.pt")
    if os.path.exists(seed_path):
        orch.register_seed_policy("seed", seed_path, validated_scenarios=["steady", "bursty", "degraded_network"])
    app.state.orchestrator = orch
    yield


app = FastAPI(title="DevMind Orchestrator Dashboard", version="0.1.0", lifespan=lifespan)


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


@app.post("/clients")
async def add_client(req: ClientRequest) -> dict:
    orch: PolicyOrchestrator = app.state.orchestrator
    scenario = _scenario_from_request(req)
    oid = req.onboarding_id
    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(None, orch.onboard, oid, scenario, req.max_samples)
    assigned = next(pid for pid, rec in orch.library.items() if oid in rec.clients_assigned)
    return {"client_id": oid, "task": req.task, "decision": decision.value, "policy_assigned": assigned}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-orchestrator"}


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

<h1>Decision Log</h1>
<div id="decisions-container" style="overflow-x:auto"></div>

<h1>Threshold Recalibrations</h1>
<p style="font-size:0.85rem;color:#666">Governance-layer self-improvement: tolerance thresholds adjusted from the orchestrator's own false-reuse track record, every 5 onboarding calls.</p>
<div id="recalibrations-container" style="overflow-x:auto"></div>

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
});

refreshClients();
refreshDecisions();
refreshRecalibrations();
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


if __name__ == "__main__":
    main()
