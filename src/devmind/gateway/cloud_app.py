from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from devmind.model_clients import BERTLargeCloud

_inflight = 0


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


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    global _inflight
    _inflight += 1
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, app.state.model.predict, req.text, req.true_label)
    finally:
        _inflight -= 1
    return PredictResponse(confidence=result.confidence, is_correct=result.is_correct, inflight=_inflight)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "devmind-cloud", "inflight": _inflight}


def main() -> None:
    port = int(os.environ.get("DEVMIND_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
