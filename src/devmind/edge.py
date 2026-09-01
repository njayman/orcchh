from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression

from devmind.medallion import EWMA
from devmind.models import EdgeContextReport, OperationalState, ResourceStress

_THERMAL_CRITICAL_C = 85.0


@dataclass
class ResourceMonitor:

    disk_path: str = "/"

    def __post_init__(self) -> None:
        psutil.cpu_percent(interval=None)

    def sample(self) -> ResourceStress:
        cpu = psutil.cpu_percent(interval=None) / 100.0
        memory = psutil.virtual_memory().percent / 100.0
        disk = psutil.disk_usage(self.disk_path).percent / 100.0
        return ResourceStress(cpu=cpu, gpu=0.0, memory=memory, disk_io=disk, thermal=self._read_thermal())

    @staticmethod
    def _read_thermal() -> float:
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return 0.0
        readings = [t.current for entries in temps.values() for t in entries if t.current]
        if not readings:
            return 0.0
        return float(min(max(readings) / _THERMAL_CRITICAL_C, 1.0))


class MiscalibrationClassifier:
    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        # Gates DEGRADING behind its own tuned probability threshold instead of
        # plain argmax -- validated against real, non-circular labels (real
        # accuracy under genuine induced hardware stress, not the heuristic's own
        # output): argmax alone gave DEGRADING 7% precision (mostly false
        # alarms); a tuned threshold (~0.90) took that to 64% while barely
        # touching recall. None preserves plain argmax for any caller that
        # fits without one. See misc_classifier_fit_experiment.py.
        self._degrading_threshold: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, degrading_threshold: float | None = None) -> None:
        self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._model.fit(X, y)
        self._degrading_threshold = degrading_threshold

    def predict(self, resource_stress: ResourceStress, error_rate: float) -> OperationalState:
        if self._model is None:
            return self._fallback_heuristic(resource_stress, error_rate)
        x = np.concatenate([resource_stress.as_array(), [error_rate]]).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        labels = list(self._model.classes_)
        if self._degrading_threshold is not None and OperationalState.DEGRADING.value in labels:
            deg_idx = labels.index(OperationalState.DEGRADING.value)
            if proba[deg_idx] >= self._degrading_threshold:
                return OperationalState.DEGRADING
            other = [(p, l) for p, l in zip(proba, labels) if l != OperationalState.DEGRADING.value]
            return OperationalState(max(other)[1])
        return OperationalState(labels[int(proba.argmax())])

    @classmethod
    def load_pretrained(cls, model_path: str, degrading_threshold: float | None = None) -> MiscalibrationClassifier:
        import joblib

        clf = cls()
        clf._model = joblib.load(model_path)
        clf._degrading_threshold = degrading_threshold
        return clf

    @staticmethod
    def from_env() -> MiscalibrationClassifier | None:
        # Opt-in: DEVMIND_MISC_CLASSIFIER_PATH unset means every existing caller
        # (live gateway, offline simulation, self-checks) keeps using the fresh,
        # unfitted default (-> _fallback_heuristic), unchanged. Returns None so
        # callers can fall back to their own EdgeDevice() default in that case.
        import os

        path = os.environ.get("DEVMIND_MISC_CLASSIFIER_PATH")
        if not path:
            return None
        threshold_str = os.environ.get("DEVMIND_MISC_CLASSIFIER_THRESHOLD")
        threshold = float(threshold_str) if threshold_str else None
        return MiscalibrationClassifier.load_pretrained(path, degrading_threshold=threshold)

    @staticmethod
    def _fallback_heuristic(resource_stress: ResourceStress, error_rate: float) -> OperationalState:
        cpu_thermal = max(resource_stress.cpu, resource_stress.thermal)
        mem_disk = max(resource_stress.memory, resource_stress.disk_io)
        if cpu_thermal > 0.85 or error_rate > 0.10:
            return OperationalState.DEGRADING
        if cpu_thermal > 0.65 or mem_disk > 0.80:
            return OperationalState.STRESSED
        return OperationalState.NOMINAL

    @property
    def fitted(self) -> bool:
        return self._model is not None


def compute_calibration_delta(confidence_raw: float, cpu: float, thermal: float) -> tuple[float, float]:
    temp = 1.0 + 0.5 * cpu + 0.3 * thermal
    calibrated = 1.0 / (1.0 + np.exp(-(np.log(confidence_raw / (1.0 - confidence_raw + 1e-8)) / temp)))
    return float(calibrated), float(abs(confidence_raw - calibrated))


class EdgeDevice:
    def __init__(self, classifier: MiscalibrationClassifier | None = None, stale_timeout_s: float = 5.0):
        self.classifier = classifier or MiscalibrationClassifier()
        self._resource_stress = ResourceStress()
        self._error_buffer: list[bool] = []
        self._last_report: EdgeContextReport | None = None
        self._trust_ewma = EWMA(alpha=0.05)
        self.stale_timeout_s = stale_timeout_s
        self._last_seen: float | None = None

    @property
    def last_report(self) -> EdgeContextReport | None:
        if self._last_report is None:
            return None
        if self.is_unreachable:
            return replace(self._last_report, operational_state=OperationalState.UNREACHABLE)
        return self._last_report

    @property
    def is_unreachable(self) -> bool:
        if self._last_seen is None:
            return True
        return (time.monotonic() - self._last_seen) > self.stale_timeout_s

    def mark_unreachable(self) -> None:
        self._last_seen = None

    def heartbeat(self, resource_stress: ResourceStress, sla_budget_ms: float = 300.0) -> EdgeContextReport:
        self._resource_stress = resource_stress
        last_confidence = self._last_report.confidence_raw if self._last_report else 0.5
        return self.emit_report(last_confidence, is_correct=None, sla_budget_ms=sla_budget_ms)

    def emit_report(
        self,
        confidence_raw: float,
        is_correct: bool | None = None,
        sla_budget_ms: float = 300.0,
    ) -> EdgeContextReport:
        cpu = self._resource_stress.cpu
        thermal = self._resource_stress.thermal
        calibrated, delta = compute_calibration_delta(confidence_raw, cpu, thermal)

        if is_correct is not None:
            self._error_buffer.append(not is_correct)
            if len(self._error_buffer) > 60:
                self._error_buffer.pop(0)
        error_rate = sum(self._error_buffer) / max(len(self._error_buffer), 1)

        state = self.classifier.predict(self._resource_stress, error_rate)

        predicted_local_latency_ms = 50.0 + cpu * 30.0 + thermal * 10.0
        sla_margin_ms = sla_budget_ms - predicted_local_latency_ms

        self._last_report = EdgeContextReport(
            resource_stress=ResourceStress(
                **{
                    k: getattr(self._resource_stress, k)
                    for k in ["cpu", "gpu", "memory", "disk_io", "thermal"]
                }
            ),
            operational_state=state,
            confidence_raw=confidence_raw,
            confidence_calibrated=calibrated,
            calibration_delta=delta,
            error_rate=error_rate,
            sla_margin_ms=sla_margin_ms,
            trust_score=(
                self._trust_ewma.value if self._trust_ewma._value is not None else 1.0
            ),
        )
        self._last_seen = time.monotonic()
        return self._last_report

    def apply_stress(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self._resource_stress, k):
                setattr(self._resource_stress, k, v)

    def update_from_outcome(self, latency_ms: float, sla_met: bool, accuracy: float) -> None:
        self._error_buffer.append(accuracy < 0.5)
        if len(self._error_buffer) > 60:
            self._error_buffer.pop(0)
        self._trust_ewma.update(1.0 if (sla_met and accuracy >= 0.5) else 0.0)


def demo() -> None:
    calibrated_lo, delta_lo = compute_calibration_delta(0.3, cpu=0.8, thermal=0.5)
    assert calibrated_lo > 0.3, "temperature scaling must push low confidence toward 0.5, not always down"
    assert abs(delta_lo - abs(0.3 - calibrated_lo)) < 1e-9

    calibrated_hi, _ = compute_calibration_delta(0.9, cpu=0.8, thermal=0.5)
    assert calibrated_hi < 0.9, "temperature scaling must pull high confidence down toward 0.5"

    device = EdgeDevice()
    device.apply_stress(cpu=0.8, thermal=0.5)
    report = device.emit_report(0.3)
    assert abs(report.confidence_calibrated - calibrated_lo) < 1e-9, (
        "emit_report must use the signed calibrated value, not raw-minus-unsigned-delta"
    )

    # MiscalibrationClassifier: fit + threshold-gated DEGRADING + load_pretrained round-trip.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 6))
    y = np.array(["NOMINAL"] * 150 + ["STRESSED"] * 40 + ["DEGRADING"] * 10)
    clf = MiscalibrationClassifier()
    clf.fit(X, y, degrading_threshold=0.9)
    assert clf.fitted
    # a threshold this high means DEGRADING should be rare in practice, never a KeyError/crash
    for _ in range(20):
        state = clf.predict(ResourceStress(cpu=rng.uniform(), thermal=rng.uniform()), rng.uniform())
        assert state in (OperationalState.NOMINAL, OperationalState.STRESSED, OperationalState.DEGRADING)

    clf_no_threshold = MiscalibrationClassifier()
    clf_no_threshold.fit(X, y)  # degrading_threshold=None must still behave as plain argmax
    assert clf_no_threshold._degrading_threshold is None

    import tempfile

    import joblib

    with tempfile.TemporaryDirectory() as tmp:
        model_path = f"{tmp}/model.joblib"
        joblib.dump(clf._model, model_path)
        loaded = MiscalibrationClassifier.load_pretrained(model_path, degrading_threshold=0.9)
        assert loaded.fitted
        same_stress, same_error_rate = ResourceStress(cpu=0.5, thermal=0.5), 0.1
        assert loaded.predict(same_stress, same_error_rate) == clf.predict(same_stress, same_error_rate), (
            "a loaded pretrained model must predict identically to the original fitted instance"
        )

    print("edge self-check passed")


if __name__ == "__main__":
    demo()
