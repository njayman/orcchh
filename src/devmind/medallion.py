from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from devmind.models import (
    BronzeMetricSnapshot,
    EdgeContextReport,
    GoldStateVector,
    OperationalState,
    SilverFeatureVector,
)


@dataclass
class MetricSource:
    name: str
    schema_type: str
    read_fn: Callable[[], Any]

    def read(self) -> Any:
        return self.read_fn()


class DynamicMetricRegistry:
    def __init__(self):
        self._sources: dict[str, MetricSource] = {}

    def register(self, source: MetricSource) -> None:
        self._sources[source.name] = source

    def snapshot(self) -> BronzeMetricSnapshot:
        snap = BronzeMetricSnapshot()
        for name, source in self._sources.items():
            val = source.read()
            match name:
                case "cloud_queue_depth":
                    snap.cloud_queue_depth = int(val)
                case "rtt_ms":
                    snap.rtt_ms = float(val)
                case "sla_remaining_ms":
                    snap.sla_remaining_ms = float(val)
                case "edge_context":
                    snap.edge_context = val
                case "energy_mj":
                    snap.energy_mj = float(val)
                case "traffic_intensity":
                    snap.traffic_intensity = float(val)
        return snap


class EWMA:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, sample: float) -> float:
        if self._value is None:
            self._value = sample
        else:
            self._value = self.alpha * sample + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value or 0.0


class SilverEnricher:
    def __init__(self):
        self._queue_ewma = EWMA(alpha=0.3)
        self._rtt_ewma = EWMA(alpha=0.2)

    def enrich(self, bronze: BronzeMetricSnapshot) -> SilverFeatureVector:
        ctx = bronze.edge_context
        is_degrading = ctx.operational_state == OperationalState.DEGRADING
        is_stressed = ctx.operational_state == OperationalState.STRESSED
        is_unreachable = ctx.operational_state == OperationalState.UNREACHABLE

        drain_rate = bronze.cloud_queue_depth / max(bronze.sla_remaining_ms, 1)
        predicted_wait = self._queue_ewma.update(drain_rate) * bronze.sla_remaining_ms
        predicted_rtt = self._rtt_ewma.update(bronze.rtt_ms)

        sla_pred = (predicted_wait + predicted_rtt) > bronze.sla_remaining_ms

        features = SilverFeatureVector(
            confidence=ctx.confidence_raw,
            predicted_queue_wait_ms=predicted_wait,
            predicted_rtt_ms=predicted_rtt,
            sla_remaining_ms=bronze.sla_remaining_ms,
            edge_cpu_load=ctx.resource_stress.cpu,
            resource_stress_cpu=ctx.resource_stress.cpu,
            resource_stress_gpu=ctx.resource_stress.gpu,
            resource_stress_memory=ctx.resource_stress.memory,
            resource_stress_disk_io=ctx.resource_stress.disk_io,
            resource_stress_thermal=ctx.resource_stress.thermal,
            sla_violation_predicted=sla_pred,
            operational_state=ctx.operational_state,
            stale=is_unreachable,
        )

        if is_stressed or is_degrading:
            features.calibration_delta = ctx.calibration_delta
        if is_degrading:
            features.error_rate = ctx.error_rate

        return features


class GoldNormalizer:
    @staticmethod
    def normalize(silver: SilverFeatureVector) -> GoldStateVector:
        return GoldStateVector.from_silver(silver)
