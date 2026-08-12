from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset


@dataclass
class Sample:
    text: str
    label: int
    split: str


TASK_DATASETS: dict[str, dict[str, str | None]] = {
    "toxicity": dict(hf_id="tasksource/jigsaw_toxicity", config=None, text_field="comment_text", label_field="toxic"),
    "sentiment": dict(hf_id="nyu-mll/glue", config="sst2", text_field="sentence", label_field="label"),
    "spam": dict(hf_id="ucirvine/sms_spam", config=None, text_field="sms", label_field="label"),
    "topic": dict(hf_id="fancyzhx/ag_news", config=None, text_field="text", label_field="label"),
}


class TextClassificationDataset:
    """Generic HF text-classification loader. One (text_field, label_field)
    pair on a given dataset id/config -- covers any binary or multi-class
    classification task, not just Jigsaw toxicity."""

    def __init__(
        self,
        hf_id: str,
        text_field: str,
        label_field: str,
        config: str | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        ds = load_dataset(hf_id, config, split="train", streaming=False) if config else load_dataset(
            hf_id, split="train", streaming=False
        )
        texts = ds[text_field]
        labels = np.array([int(v) for v in ds[label_field]], dtype=np.int64)

        if max_samples and max_samples < len(texts):
            idx = np.random.default_rng(seed).permutation(len(texts))[:max_samples]
            texts = [texts[i] for i in idx]
            labels = labels[idx]

        n = len(texts)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        self._samples: list[Sample] = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            split = "train" if i < train_end else ("val" if i < val_end else "test")
            self._samples.append(Sample(text=text, label=int(label), split=split))

        self._rng = np.random.default_rng(seed)

    @property
    def train(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "train"]

    @property
    def val(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "val"]

    @property
    def test(self) -> list[Sample]:
        return [s for s in self._samples if s.split == "test"]

    def samples(self, split: str = "train") -> list[Sample]:
        return [s for s in self._samples if s.split == split]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def positive_ratio(self) -> float:
        labels = [s.label for s in self._samples]
        return sum(1 for l in labels if l > 0) / len(labels) if labels else 0.0

    def shuffle_train(self) -> None:
        train = self.train
        self._rng.shuffle(train)


def load_task_dataset(task: str, max_samples: int | None = None) -> TextClassificationDataset:
    if task not in TASK_DATASETS:
        raise ValueError(f"unknown task '{task}', choose from {list(TASK_DATASETS)}")
    spec = TASK_DATASETS[task]
    return TextClassificationDataset(
        spec["hf_id"], spec["text_field"], spec["label_field"], config=spec["config"], max_samples=max_samples
    )


class JigsawDataset(TextClassificationDataset):
    def __init__(self, max_samples: int | None = None):
        spec = TASK_DATASETS["toxicity"]
        super().__init__(spec["hf_id"], spec["text_field"], spec["label_field"], max_samples=max_samples)

    @property
    def toxic_ratio(self) -> float:
        return self.positive_ratio
