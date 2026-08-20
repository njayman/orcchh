from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class DiagnosisContext:
    client_id: str
    window_requests: int
    escalation_rate: float
    sla_violation_rate: float
    dominant_operational_state: str
    avg_resource_stress: dict[str, float]
    avg_calibration_delta: float | None = None
    avg_error_rate: float | None = None
    avg_trust_score: float | None = None
    dominant_fallback_reason: str | None = None

    def to_prompt(self) -> str:
        stress = ", ".join(f"{k}={v:.2f}" for k, v in self.avg_resource_stress.items())
        extra = ""
        if self.avg_calibration_delta is not None:
            extra += f" avg_calibration_delta={self.avg_calibration_delta:.2f}"
        if self.avg_error_rate is not None:
            extra += f" avg_error_rate={self.avg_error_rate:.2f}"
        if self.avg_trust_score is not None:
            extra += f" avg_trust_score={self.avg_trust_score:.2f}"
        if self.dominant_fallback_reason is not None:
            extra += f" dominant_fallback_reason={self.dominant_fallback_reason}"
        return (
            f"Client '{self.client_id}' edge inference gateway telemetry over the last "
            f"{self.window_requests} requests: escalation_rate={self.escalation_rate:.2f}, "
            f"sla_violation_rate={self.sla_violation_rate:.2f}, "
            f"dominant_operational_state={self.dominant_operational_state}, "
            f"resource_stress=({stress}).{extra}\n"
            "Diagnose the likely root cause of this sustained distress and what resource "
            "or configuration change would fix it. Respond with ONLY this exact JSON shape, "
            'nothing else: {"summary": "one sentence", "likely_cause": "one sentence", '
            '"resource_recommendation": "one sentence"}'
        )


@dataclass
class Diagnosis:
    summary: str
    likely_cause: str
    resource_recommendation: str
    model_used: str
    latency_ms: float
    raw_response: str
    parsed_ok: bool = True


class DiagnosisProvider(ABC):
    @abstractmethod
    async def diagnose(self, context: DiagnosisContext) -> Diagnosis: ...


class OllamaDiagnosisProvider(DiagnosisProvider):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 1,
        num_predict: int = 350,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma4:12b-it-qat")
        self.max_retries = max_retries
        self.num_predict = num_predict
        self._client = httpx.AsyncClient(timeout=timeout)

    async def diagnose(self, context: DiagnosisContext) -> Diagnosis:
        prompt = context.to_prompt()
        start = time.perf_counter()
        last_exc: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"num_predict": self.num_predict},
                    },
                )
                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start) * 1000
                raw = resp.json()["response"]
                return self._parse(raw, latency_ms)
            except Exception as exc:
                last_exc = exc
        latency_ms = (time.perf_counter() - start) * 1000
        return Diagnosis(
            summary=f"diagnosis unavailable: {last_exc}",
            likely_cause="unknown",
            resource_recommendation="unknown",
            model_used=self.model,
            latency_ms=latency_ms,
            raw_response="",
            parsed_ok=False,
        )

    def _parse(self, raw: str, latency_ms: float) -> Diagnosis:
        try:
            data = json.loads(raw)
            return Diagnosis(
                summary=data.get("summary", ""),
                likely_cause=data.get("likely_cause", ""),
                resource_recommendation=data.get("resource_recommendation", ""),
                model_used=self.model,
                latency_ms=latency_ms,
                raw_response=raw,
                parsed_ok=True,
            )
        except (json.JSONDecodeError, AttributeError):
            return Diagnosis(
                summary=raw[:300],
                likely_cause="unparseable",
                resource_recommendation="unparseable",
                model_used=self.model,
                latency_ms=latency_ms,
                raw_response=raw,
                parsed_ok=False,
            )


def demo() -> None:
    provider = OllamaDiagnosisProvider.__new__(OllamaDiagnosisProvider)
    provider.model = "test-model"

    good = provider._parse('{"summary": "s", "likely_cause": "c", "resource_recommendation": "r"}', 123.0)
    assert good.parsed_ok and good.summary == "s" and good.likely_cause == "c" and good.resource_recommendation == "r"

    bad = provider._parse("not json at all", 45.0)
    assert not bad.parsed_ok, "malformed model output must not crash, must fall back gracefully"
    assert bad.summary == "not json at all"
    assert bad.likely_cause == "unparseable"

    ctx = DiagnosisContext(
        client_id="c1",
        window_requests=50,
        escalation_rate=1.0,
        sla_violation_rate=0.95,
        dominant_operational_state="DEGRADING",
        avg_resource_stress={"cpu": 0.87, "thermal": 0.71},
        avg_trust_score=0.2,
    )
    prompt = ctx.to_prompt()
    assert "c1" in prompt and "escalation_rate=1.00" in prompt and "cpu=0.87" in prompt
    assert "avg_trust_score=0.20" in prompt
    assert "avg_calibration_delta" not in prompt, "omitted optional fields must not appear in the prompt"
    assert "dominant_fallback_reason" not in prompt, "omitted optional fields must not appear in the prompt"
    assert '"summary"' in prompt

    ctx_with_reason = DiagnosisContext(
        client_id="c1", window_requests=50, escalation_rate=1.0, sla_violation_rate=0.95,
        dominant_operational_state="DEGRADING", avg_resource_stress={"cpu": 0.87},
        dominant_fallback_reason="queue_backed_up",
    )
    assert "dominant_fallback_reason=queue_backed_up" in ctx_with_reason.to_prompt()

    print("diagnosis self-check passed")


if __name__ == "__main__":
    demo()
