from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from devmind.dataset import load_task_dataset
from devmind.edge import EdgeDevice
from devmind.model_clients import DistilBERTEdge
from devmind.models import OperationalState, ResourceStress

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")


@dataclass
class Sample:
    cpu: float
    gpu: float
    memory: float
    disk_io: float
    thermal: float
    error_rate: float
    correct: bool


def draw_simulated_stress(rng: np.random.Generator, edge_stress_prob: float, load_pct: float = 1.0) -> ResourceStress:
    return ResourceStress(
        cpu=rng.uniform(0, 0.3) * load_pct if rng.uniform() > edge_stress_prob else rng.uniform(0.6, 0.95),
        thermal=rng.uniform(0, 0.2) + 0.1 * load_pct
        if rng.uniform() > edge_stress_prob * 0.7
        else rng.uniform(0.6, 0.9),
        memory=rng.uniform(0.1, 0.4) if rng.uniform() > edge_stress_prob * 0.5 else rng.uniform(0.7, 0.95),
        disk_io=rng.uniform(0.1, 0.2) if rng.uniform() > edge_stress_prob * 0.5 else rng.uniform(0.7, 0.9),
        gpu=rng.uniform(0.0, 0.3),
    )


def collect(
    phases: list[tuple[str, float, int]], edge_model: DistilBERTEdge, samples: list
) -> tuple[list[Sample], float]:
    """phases: list of (label, edge_stress_prob, n_samples)."""
    rng = np.random.default_rng(0)
    device = EdgeDevice()
    collected: list[Sample] = []
    baseline_error_rate: float | None = None
    idx = 0

    for phase_name, edge_stress_prob, n in phases:
        print(f"== phase={phase_name} edge_stress_prob={edge_stress_prob} n={n} ==")
        phase_start = len(collected)
        for _ in range(n):
            sample = samples[idx % len(samples)]
            idx += 1
            stress = draw_simulated_stress(rng, edge_stress_prob)
            device.apply_stress(**vars(stress))
            result = edge_model.predict(sample.text, sample.label)
            report = device.emit_report(result.confidence, result.is_correct)
            collected.append(
                Sample(
                    cpu=stress.cpu, gpu=stress.gpu, memory=stress.memory, disk_io=stress.disk_io,
                    thermal=stress.thermal, error_rate=report.error_rate, correct=result.is_correct,
                )
            )
        print(f"   collected {len(collected) - phase_start} samples this phase")
        if phase_name == "idle" and baseline_error_rate is None:
            baseline_error_rate = float(np.mean([not s.correct for s in collected[phase_start:]]))
            print(f"   baseline (idle) error rate = {baseline_error_rate:.3f}")

    return collected, (baseline_error_rate or 0.0)


def label_from_accuracy(collected: list[Sample], baseline_error_rate: float, window: int = 20) -> list[str]:
    labels: list[str] = []
    for i in range(len(collected)):
        lo = max(0, i - window + 1)
        rolling_error_rate = float(np.mean([not s.correct for s in collected[lo : i + 1]]))
        delta = rolling_error_rate - baseline_error_rate
        if delta <= 0.10:
            labels.append(OperationalState.NOMINAL.value)
        elif delta <= 0.25:
            labels.append(OperationalState.STRESSED.value)
        else:
            labels.append(OperationalState.DEGRADING.value)
    return labels


def main() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print("Loading real edge model and dataset...")
    edge_model = DistilBERTEdge()
    dataset = load_task_dataset("toxicity", max_samples=30000)
    samples = dataset.test if len(dataset.test) > 200 else (dataset.test * 50)

    phases = [
        ("idle", 0.0, 3000),
        ("low (steady/bursty/degraded_network)", 0.1, 6000),
        ("elevated (held_out)", 0.4, 6000),
        ("high", 0.7, 6000),
        ("saturated", 1.0, 6000),
    ]
    t0 = time.perf_counter()
    collected, baseline = collect(phases, edge_model, samples)
    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {len(collected)} samples in {elapsed:.0f}s, baseline error rate={baseline:.3f}")

    labels = label_from_accuracy(collected, baseline)
    from collections import Counter
    print("Label distribution:", Counter(labels))

    X = np.array([[s.cpu, s.gpu, s.memory, s.disk_io, s.thermal, s.error_rate] for s in collected])
    y = np.array(labels)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    np.savez(os.path.join(_OUTPUT_DIR, f"misc-classifier-sim-raw-data-{timestamp}.npz"), X=X, y=y)

    with open(os.path.join(_OUTPUT_DIR, f"misc-classifier-sim-collection-{timestamp}.json"), "w") as f:
        json.dump(
            {"n_samples": len(collected), "baseline_error_rate": baseline, "label_distribution": dict(Counter(labels)), "elapsed_s": elapsed},
            f, indent=2,
        )
    print(f"\nRaw data saved: misc-classifier-sim-raw-data-{timestamp}.npz")


if __name__ == "__main__":
    main()
