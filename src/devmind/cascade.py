from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from devmind.agent import AgenticOrchestrator
from devmind.edge import EdgeDevice
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, SilverEnricher
from devmind.model_clients import CloudClient, DistilBERTEdge, InferenceResult
from devmind.models import Action, BronzeMetricSnapshot, EdgeContextReport
from devmind.orchestrator import DriftEventListener


@dataclass
class RequestOutcome:
    request_id: str
    tier: str
    latency_ms: float
    sla_met: bool
    accuracy: float
    fallback_triggered: bool
    action: Action
    confidence_raw: float
    sla_margin_ms: float = 0.0
    trust_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier": self.tier,
            "latency_ms": self.latency_ms,
            "sla_met": self.sla_met,
            "accuracy": self.accuracy,
            "fallback_triggered": self.fallback_triggered,
            "action": self.action.name,
            "confidence_raw": self.confidence_raw,
            "sla_margin_ms": self.sla_margin_ms,
            "trust_score": self.trust_score,
        }


class CascadeController:

    def __init__(
        self,
        agent: AgenticOrchestrator,
        edge: EdgeDevice,
        registry: DynamicMetricRegistry,
        silver: SilverEnricher,
        gold: GoldNormalizer,
        edge_model: DistilBERTEdge,
        cloud_client: CloudClient,
        edge_timeout_s: float = 2.0,
        client_id: str = "default",
        drift_listener: DriftEventListener | None = None,
        action_log_path: str | None = None,
        action_log_url: str | None = None,
    ):
        self.agent = agent
        self.edge = edge
        self.registry = registry
        self.silver = silver
        self.gold = gold
        self.edge_model = edge_model
        self.cloud_client = cloud_client
        self.edge_timeout_s = edge_timeout_s
        self.client_id = client_id
        self.drift_listener = drift_listener
        self.action_log_path = action_log_path
        # action_log_path alone only works when the gateway and the
        # orchestrator share a filesystem (same host/VM). In a real multi-region
        # deployment they're separate pods, so the orchestrator's live-monitoring
        # tail loop never sees anything. action_log_url forwards the same entry
        # over HTTP when the two aren't co-located; harmless to set both.
        self.action_log_url = action_log_url

    async def process(
        self,
        request_id: str,
        text: str,
        sla_budget_ms: float = 300.0,
        true_label: int | None = None,
    ) -> RequestOutcome:
        loop = asyncio.get_running_loop()
        try:
            edge_result = await asyncio.wait_for(
                loop.run_in_executor(None, self.edge_model.predict, text, true_label),
                timeout=self.edge_timeout_s,
            )
        except Exception:
            self.edge.mark_unreachable()
            report = self.edge.last_report
            if self.drift_listener is not None and report is not None:
                self.drift_listener.notify(self.client_id, report)
            return await self._unreachable_fallback(request_id, text, true_label, sla_budget_ms, report)

        report = self.edge.emit_report(
            edge_result.confidence,
            edge_result.is_correct if true_label is not None else None,
            sla_budget_ms,
        )
        if self.drift_listener is not None:
            self.drift_listener.notify(self.client_id, report)
        bronze = self.registry.snapshot()
        bronze.sla_budget_ms = sla_budget_ms
        bronze.sla_remaining_ms = sla_budget_ms
        silver = self.silver.enrich(bronze)
        gold = self.gold.normalize(silver)
        action, fallback = self.agent.decide(gold)
        fallback_reason = self.agent.last_fallback_reason

        tier, latency, accuracy = await self._dispatch(action, text, true_label, edge_result, bronze)
        sla_met = latency <= bronze.sla_budget_ms
        self.agent.reflect(latency, sla_met, accuracy, tier, fallback)
        self.edge.update_from_outcome(latency, sla_met, accuracy)
        if self.action_log_path:
            loop.run_in_executor(
                None,
                self._log_action,
                request_id,
                action,
                tier,
                latency,
                sla_met,
                accuracy,
                fallback,
                report,
                fallback_reason,
            )
        return RequestOutcome(
            request_id=request_id,
            tier=tier,
            latency_ms=latency,
            sla_met=sla_met,
            accuracy=accuracy,
            fallback_triggered=fallback,
            action=action,
            confidence_raw=edge_result.confidence,
            sla_margin_ms=report.sla_margin_ms,
            trust_score=report.trust_score,
        )

    def _log_action(
        self,
        request_id: str,
        action: Action,
        tier: str,
        latency_ms: float,
        sla_met: bool,
        accuracy: float,
        fallback: bool,
        report: EdgeContextReport | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        if not self.action_log_path and not self.action_log_url:
            return
        entry = {
            "timestamp": time.time(),
            "client": self.client_id,
            "request_id": request_id,
            "action": action.name,
            "tier": tier,
            "latency_ms": latency_ms,
            "sla_met": sla_met,
            "accuracy": accuracy,
            "fallback_triggered": fallback,
        }
        if report is not None:
            entry["operational_state"] = report.operational_state.value
            entry["resource_stress"] = vars(report.resource_stress)
            entry["calibration_delta"] = report.calibration_delta
            entry["error_rate"] = report.error_rate
            entry["trust_score"] = report.trust_score
        if fallback_reason is not None:
            entry["fallback_reason"] = fallback_reason
        if self.action_log_path:
            os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
            with open(self.action_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        if self.action_log_url:
            try:
                httpx.post(self.action_log_url, json=entry, timeout=2.0)
            except httpx.HTTPError:
                pass

    async def _unreachable_fallback(
        self,
        request_id: str,
        text: str,
        true_label: int | None,
        sla_budget_ms: float,
        report: EdgeContextReport | None = None,
    ) -> RequestOutcome:
        cloud_result = await self.cloud_client.predict(text, true_label)
        accuracy = float(cloud_result.is_correct) if true_label is not None else (
            0.5 + 0.5 * cloud_result.confidence
        )
        sla_met = cloud_result.latency_ms <= sla_budget_ms
        self.agent.reflect(cloud_result.latency_ms, sla_met, accuracy, "cloud", fallback=True)
        if self.action_log_path or self.action_log_url:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                self._log_action,
                request_id,
                Action.ESCALATE_TO_CLOUD,
                "cloud",
                cloud_result.latency_ms,
                sla_met,
                accuracy,
                True,
                report,
                "edge_unreachable",
            )
        return RequestOutcome(
            request_id=request_id,
            tier="cloud",
            latency_ms=cloud_result.latency_ms,
            sla_met=sla_met,
            accuracy=accuracy,
            fallback_triggered=True,
            action=Action.ESCALATE_TO_CLOUD,
            confidence_raw=0.0,
            sla_margin_ms=0.0,
            trust_score=0.0,
        )

    async def _dispatch(
        self,
        action: Action,
        text: str,
        true_label: int | None,
        edge_result: InferenceResult,
        bronze: BronzeMetricSnapshot,
    ) -> tuple[str, float, float]:
        if action == Action.ESCALATE_TO_CLOUD:
            for attempt in range(2):
                try:
                    cloud_result = await self.cloud_client.predict(text, true_label)
                    accuracy = float(cloud_result.is_correct) if true_label is not None else (
                        0.5 + 0.5 * cloud_result.confidence
                    )
                    return "cloud", cloud_result.latency_ms, accuracy
                except Exception:
                    if attempt == 1:
                        break
            accuracy = float(edge_result.is_correct) if true_label is not None else (
                0.5 + 0.5 * bronze.edge_context.confidence_calibrated
            )
            return "edge_fallback_cloud_unreachable", edge_result.latency_ms, accuracy
        accuracy = float(edge_result.is_correct) if true_label is not None else (
            0.5 + 0.5 * bronze.edge_context.confidence_calibrated
        )
        return "edge", edge_result.latency_ms, accuracy


def demo() -> None:
    import tempfile

    from devmind.models import OperationalState, ResourceStress

    with tempfile.TemporaryDirectory() as tmp:
        log_path = f"{tmp}/action_log.jsonl"
        controller = CascadeController.__new__(CascadeController)
        controller.client_id = "demo_client"
        controller.action_log_path = log_path
        controller.action_log_url = None

        controller._log_action("r1", Action.ROUTE_TO_EDGE, "edge", 120.0, True, 1.0, False, report=None)
        with open(log_path) as f:
            row = json.loads(f.readline())
        assert "resource_stress" not in row, "no report given must not fabricate resource_stress"
        assert "fallback_reason" not in row, "no reason given must not fabricate one"

        report = EdgeContextReport(
            resource_stress=ResourceStress(cpu=0.8, thermal=0.6),
            operational_state=OperationalState.DEGRADING,
            calibration_delta=0.3,
            error_rate=0.2,
            trust_score=0.4,
        )
        controller._log_action(
            "r2", Action.ESCALATE_TO_CLOUD, "cloud", 900.0, False, 0.0, False,
            report=report, fallback_reason="low_confidence",
        )
        with open(log_path) as f:
            rows = [json.loads(line) for line in f]
        row2 = rows[1]
        assert row2["operational_state"] == "DEGRADING"
        assert row2["resource_stress"]["cpu"] == 0.8 and row2["resource_stress"]["thermal"] == 0.6
        assert row2["calibration_delta"] == 0.3
        assert row2["error_rate"] == 0.2
        assert row2["trust_score"] == 0.4
        assert row2["fallback_reason"] == "low_confidence"

        # action_log_url forwarding: an unreachable URL must not raise (fire-and-forget,
        # off the request's critical path) -- and it must still write the local copy.
        broken_controller = CascadeController.__new__(CascadeController)
        broken_controller.client_id = "demo_client"
        broken_controller.action_log_path = log_path
        broken_controller.action_log_url = "http://127.0.0.1:1/log-action"
        broken_controller._log_action("r3", Action.ROUTE_TO_EDGE, "edge", 50.0, True, 1.0, False, report=None)
        with open(log_path) as f:
            assert len(f.readlines()) == 3, "local write must still happen even if forwarding fails"

        # Deadman's switch: edge_model.predict raising must mark the edge unreachable,
        # notify the drift listener with an UNREACHABLE-flavored report, and log the
        # fallback with fallback_reason="edge_unreachable" -- so the orchestrator
        # dashboard's tail loop has something to key a live notification off.
        class _FailingEdgeModel:
            def predict(self, text: str, true_label: int | None) -> None:
                raise RuntimeError("edge down")

        class _StubCloudResult:
            latency_ms = 40.0
            is_correct = True
            confidence = 0.9

        class _StubCloudClient:
            async def predict(self, text: str, true_label: int | None) -> _StubCloudResult:
                return _StubCloudResult()

        class _StubAgent:
            def reflect(self, *args: Any, **kwargs: Any) -> None:
                pass

        unreachable_log = f"{tmp}/unreachable_log.jsonl"
        edge = EdgeDevice()
        edge.emit_report(0.9, is_correct=True, sla_budget_ms=300.0)  # seed a last_report to promote
        listener = DriftEventListener(orchestrator=None, trust_floor=2.0, recovery_window_s=1.0)

        deadman_controller = CascadeController.__new__(CascadeController)
        deadman_controller.agent = _StubAgent()
        deadman_controller.edge = edge
        deadman_controller.edge_model = _FailingEdgeModel()
        deadman_controller.cloud_client = _StubCloudClient()
        deadman_controller.edge_timeout_s = 0.5
        deadman_controller.client_id = "demo_client"
        deadman_controller.drift_listener = listener
        deadman_controller.action_log_path = unreachable_log
        deadman_controller.action_log_url = None

        async def _run() -> RequestOutcome:
            outcome = await deadman_controller.process("r4", "some text", sla_budget_ms=300.0, true_label=1)
            await asyncio.sleep(0.1)  # let the fire-and-forget log write land
            return outcome

        outcome = asyncio.run(_run())
        assert outcome.fallback_triggered and outcome.tier == "cloud"

        assert not listener._queue.empty(), "drift listener must be notified on edge-unreachable"
        notified_client, notified_report, _ts = listener._queue.get_nowait()
        assert notified_client == "demo_client"
        assert notified_report.operational_state == OperationalState.UNREACHABLE

        with open(unreachable_log) as f:
            row = json.loads(f.readline())
        assert row["fallback_reason"] == "edge_unreachable"
        assert row["operational_state"] == "UNREACHABLE"

    print("cascade self-check passed")


if __name__ == "__main__":
    demo()
