from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from devmind.model_clients import BERTLargeCloud
from devmind.tracing import setup_tracing

_inflight = 0
_REQUESTS = Counter("devmind_cloud_requests_total", "Cloud pod predictions served")
_LATENCY_BUCKETS_MS = (5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, float("inf"))
_LATENCY = Histogram(
    "devmind_cloud_latency_ms", "Cloud pod inference latency (ms)", buckets=_LATENCY_BUCKETS_MS
)


class PredictRequest(BaseModel):
    text: str
    true_label: int | None = None


class PredictResponse(BaseModel):
    confidence: float
    is_correct: bool
    inflight: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = BERTLargeCloud(task=os.environ.get("DEVMIND_TASK", "toxicity"))
    yield


app = FastAPI(title="DevMind Cloud Pod", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
setup_tracing("devmind-cloud", app)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    global _inflight
    _inflight += 1
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, app.state.model.predict, req.text, req.true_label)
    finally:
        _inflight -= 1
    _REQUESTS.inc()
    _LATENCY.observe((time.perf_counter() - t0) * 1000.0)
    return PredictResponse(confidence=result.confidence, is_correct=result.is_correct, inflight=_inflight)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-cloud", "inflight": _inflight}


def main() -> None:
    port = int(os.environ.get("DEVMIND_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
