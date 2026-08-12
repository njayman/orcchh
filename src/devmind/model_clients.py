from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from devmind.medallion import EWMA

# Each task = one edge (small/fast) + cloud (larger/accurate) checkpoint pair,
# independently fine-tuned on the same dataset -- mirrors the original
# DistilBERT/BERT-large toxicity split. Verified loadable at time of writing.
TASK_MODELS: dict[str, tuple[str, str]] = {
    "toxicity": ("newsmediabias/DistilBert_Toxicity_Classification", "AnonymousCS/bert-large-uncased-Twitter-toxicity"),
    "sentiment": ("distilbert-base-uncased-finetuned-sst-2-english", "yoshitomo-matsubara/bert-large-uncased-sst2"),
    "spam": ("mrm8488/bert-tiny-finetuned-sms-spam-detection", "wesleyacheng/sms-spam-classification-with-bert"),
    "topic": ("textattack/distilbert-base-uncased-ag-news", "textattack/bert-base-uncased-ag-news"),
}


@dataclass
class InferenceResult:
    confidence: float
    latency_ms: float
    is_correct: bool


class ModelClient:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, true_label: int | None = None) -> InferenceResult:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        latency = (time.perf_counter() - start) * 1000
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, predicted].item())
        is_correct = true_label is not None and predicted == true_label
        return InferenceResult(confidence=confidence, latency_ms=latency, is_correct=is_correct)


class DistilBERTEdge(ModelClient):
    def __init__(self, task: str = "toxicity", device: str = "cpu"):
        super().__init__(TASK_MODELS[task][0], device)


class BERTLargeCloud(ModelClient):
    def __init__(self, task: str = "toxicity", device: str = "cpu"):
        super().__init__(TASK_MODELS[task][1], device)


class CloudClient:

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._rtt = EWMA(alpha=0.2)
        self._last_inflight = 0

    async def predict(self, text: str, true_label: int | None = None) -> InferenceResult:
        start = time.perf_counter()
        resp = await self._client.post(
            f"{self.base_url}/predict", json={"text": text, "true_label": true_label}
        )
        resp.raise_for_status()
        latency = (time.perf_counter() - start) * 1000
        self._rtt.update(latency)
        data = resp.json()
        self._last_inflight = data["inflight"]
        return InferenceResult(confidence=data["confidence"], latency_ms=latency, is_correct=data["is_correct"])

    @property
    def rtt_ms(self) -> float:
        return self._rtt.value or 40.0

    @property
    def inflight(self) -> int:
        return self._last_inflight
