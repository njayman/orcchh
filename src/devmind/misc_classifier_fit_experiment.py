from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass

import numpy as np

from devmind.dataset import load_task_dataset
from devmind.edge import EdgeDevice, ResourceMonitor
from devmind.model_clients import DistilBERTEdge
from devmind.models import OperationalState

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")  # code/evaluation/


def _burn_cpu(stop_flag) -> None:
    x = 1.0000001
    while not stop_flag.value:
        x = (x * 1.0000001 + 1.0) % 1e6


@dataclass
class Sample:
    cpu: float
    gpu: float
    memory: float
    disk_io: float
    thermal: float
    error_rate: float
    correct: bool


def _spawn_stress(n_workers: int) -> tuple[list[multiprocessing.Process], "multiprocessing.sharedctypes.Synchronized"]:
    stop_flag = multiprocessing.Value("b", False)
    procs = [multiprocessing.Process(target=_burn_cpu, args=(stop_flag,), daemon=True) for _ in range(n_workers)]
    for p in procs:
        p.start()
    return procs, stop_flag


def _stop_stress(procs: list[multiprocessing.Process], stop_flag) -> None:
    stop_flag.value = True
    for p in procs:
        p.join(timeout=5)


def collect(
    phases: list[tuple[str, int, int]],
    edge_model: DistilBERTEdge,
    samples: list,
) -> tuple[list[Sample], float]:
    """phases: list of (label, n_stress_workers, duration_s). Returns collected
    samples and the idle-phase baseline error rate."""
    monitor = ResourceMonitor()
    device = EdgeDevice()
    collected: list[Sample] = []
    baseline_error_rate: float | None = None
    idx = 0

    for phase_name, n_workers, duration_s in phases:
        print(f"== phase={phase_name} stress_workers={n_workers} duration={duration_s}s ==")
        procs, stop_flag = (_spawn_stress(n_workers) if n_workers > 0 else ([], None))
        t_end = time.perf_counter() + duration_s
        phase_samples = 0
        try:
            while time.perf_counter() < t_end:
                sample = samples[idx % len(samples)]
                idx += 1
                stress = monitor.sample()
                result = edge_model.predict(sample.text, sample.label)
                report = device.emit_report(result.confidence, result.is_correct)
                collected.append(
                    Sample(
                        cpu=stress.cpu, gpu=stress.gpu, memory=stress.memory,
                        disk_io=stress.disk_io, thermal=stress.thermal,
                        error_rate=report.error_rate, correct=result.is_correct,
                    )
                )
                phase_samples += 1
        finally:
            if procs:
                _stop_stress(procs, stop_flag)
        print(f"   collected {phase_samples} samples this phase")
        if phase_name == "idle" and baseline_error_rate is None and collected:
            baseline_error_rate = float(np.mean([not s.correct for s in collected[-phase_samples:]]))
            print(f"   baseline (idle) error rate = {baseline_error_rate:.3f}")

    return collected, (baseline_error_rate or 0.0)


def label_from_accuracy(collected: list[Sample], baseline_error_rate: float, window: int = 20) -> list[str]:
    labels: list[str] = []
    for i in range(len(collected)):
        lo = max(0, i - window + 1)
        window_errors = [not s.correct for s in collected[lo : i + 1]]
        rolling_error_rate = float(np.mean(window_errors))
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
    dataset = load_task_dataset("toxicity", max_samples=5000)
    samples = dataset.test if len(dataset.test) > 200 else (dataset.test * 50)

    n_cores = os.cpu_count() or 8
    phases = [
        ("idle", 0, 60),
        ("moderate", max(1, n_cores // 2), 90),
        ("heavy", n_cores, 2100),
    ]
    t0 = time.perf_counter()
    collected, baseline = collect(phases, edge_model, samples)
    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {len(collected)} real samples in {elapsed:.0f}s, baseline error rate={baseline:.3f}")

    labels = label_from_accuracy(collected, baseline)
    from collections import Counter
    print("Label distribution:", Counter(labels))

    X = np.array([[s.cpu, s.gpu, s.memory, s.disk_io, s.thermal, s.error_rate] for s in collected])
    y = np.array(labels)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    np.savez(os.path.join(_OUTPUT_DIR, f"misc-classifier-raw-data-{timestamp}.npz"), X=X, y=y)
    print(f"Raw data saved (X, y) for future re-analysis without re-collecting.")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

    if len(set(y)) < 2:
        print("DEGENERATE: only one class observed, cannot fit or validate meaningfully. Not viable.")
        result = {"viable": False, "reason": "degenerate_single_class", "label_distribution": dict(Counter(labels))}
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        labels_order = sorted(set(y))
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        per_class_recall = {
            k: float(v) for k, v in zip(labels_order, recall_score(y_test, y_pred, labels=labels_order, average=None))
        }

        from devmind.edge import MiscalibrationClassifier
        from devmind.models import ResourceStress
        heuristic = MiscalibrationClassifier()
        heuristic_pred = [
            heuristic._fallback_heuristic(
                ResourceStress(cpu=row[0], gpu=row[1], memory=row[2], disk_io=row[3], thermal=row[4]), row[5]
            ).value
            for row in X_test
        ]
        heuristic_acc = accuracy_score(y_test, heuristic_pred)
        heuristic_bal_acc = balanced_accuracy_score(y_test, heuristic_pred)

        print(f"held-out accuracy: fitted={acc:.3f} heuristic={heuristic_acc:.3f}")
        print(f"held-out BALANCED accuracy: fitted={bal_acc:.3f} heuristic={heuristic_bal_acc:.3f}")
        print(f"per-class recall (fitted): {per_class_recall}")
        print(f"labels: {labels_order}")
        print(f"confusion matrix:\n{cm}")

        result = {
            "viable": bool(bal_acc > (1.0 / len(labels_order)) + 0.1 and per_class_recall.get("DEGRADING", 0) > 0),
            "n_samples": len(collected),
            "baseline_error_rate": float(baseline),
            "held_out_balanced_accuracy_fitted": float(bal_acc),
            "held_out_balanced_accuracy_heuristic": float(heuristic_bal_acc),
            "per_class_recall": per_class_recall,
            "label_distribution": dict(Counter(labels)),
            "held_out_accuracy_fitted": float(acc),
            "held_out_accuracy_heuristic": float(heuristic_acc),
            "labels_order": labels_order,
            "confusion_matrix": cm.tolist(),
            "elapsed_s": float(elapsed),
        }

        import joblib
        joblib.dump(clf, os.path.join(_OUTPUT_DIR, f"misc-classifier-experiment-{timestamp}.joblib"))

    with open(os.path.join(_OUTPUT_DIR, f"misc-classifier-fit-experiment-{timestamp}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved. viable={result['viable']}")


if __name__ == "__main__":
    main()
