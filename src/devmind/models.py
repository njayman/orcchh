from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    ROUTE_TO_EDGE = 0
    ESCALATE_TO_CLOUD = 1
    QUERY_EXTENDED_CONTEXT = 2


class OperationalState(str, Enum):
    NOMINAL = "NOMINAL"
    STRESSED = "STRESSED"
    DEGRADING = "DEGRADING"
    UNREACHABLE = "UNREACHABLE"


@dataclass
class ResourceStress:
    cpu: float = 0.0
    gpu: float = 0.0
    memory: float = 0.0
    disk_io: float = 0.0
    thermal: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.cpu, self.gpu, self.memory, self.disk_io, self.thermal])


@dataclass
class EdgeContextReport:
    resource_stress: ResourceStress = field(default_factory=ResourceStress)
    operational_state: OperationalState = OperationalState.NOMINAL
    confidence_raw: float = 0.0
    confidence_calibrated: float = 0.0
    calibration_delta: float = 0.0
    error_rate: float = 0.0
    sla_margin_ms: float = 0.0
    trust_score: float = 1.0


@dataclass
class BronzeMetricSnapshot:
    cloud_queue_depth: int = 0
    rtt_ms: float = 0.0
    sla_remaining_ms: float = 0.0
    sla_budget_ms: float = 300.0
    edge_context: EdgeContextReport = field(default_factory=EdgeContextReport)
    energy_mj: float = 0.0
    traffic_intensity: float = 0.0

    request_id: str = ""


@dataclass
class SilverFeatureVector:
    confidence: float = 0.0
    predicted_queue_wait_ms: float = 0.0
    predicted_rtt_ms: float = 0.0
    sla_remaining_ms: float = 0.0
    edge_cpu_load: float = 0.0

    resource_stress_cpu: float = 0.0
    resource_stress_gpu: float = 0.0
    resource_stress_memory: float = 0.0
    resource_stress_disk_io: float = 0.0
    resource_stress_thermal: float = 0.0

    calibration_delta: Optional[float] = None
    error_rate: Optional[float] = None
    sla_violation_predicted: bool = False
    operational_state: OperationalState = OperationalState.NOMINAL
    stale: bool = False


@dataclass
class GoldStateVector:
    NUM_SLOTS = 13
    SLOT_NAMES = [
        "confidence",
        "predicted_queue_wait",
        "predicted_rtt",
        "sla_remaining",
        "edge_cpu_load",
        "resource_cpu",
        "resource_gpu",
        "resource_memory",
        "resource_disk_io",
        "resource_thermal",
        "calibration_delta",
        "error_rate",
        "operational_state",
    ]

    slots: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))
    mask: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))

    @classmethod
    def from_silver(cls, silver: SilverFeatureVector) -> GoldStateVector:
        op_map = {
            OperationalState.NOMINAL: 0.0,
            OperationalState.STRESSED: 0.5,
            OperationalState.DEGRADING: 1.0,
            OperationalState.UNREACHABLE: 1.0,
        }
        slots = np.zeros(13, dtype=np.float32)
        mask = np.ones(13, dtype=np.float32)

        slots[0] = silver.confidence
        slots[1] = silver.predicted_queue_wait_ms / 1000.0
        slots[2] = silver.predicted_rtt_ms / 1000.0
        slots[3] = silver.sla_remaining_ms / 1000.0
        slots[4] = silver.edge_cpu_load
        slots[5] = silver.resource_stress_cpu
        slots[6] = silver.resource_stress_gpu
        slots[7] = silver.resource_stress_memory
        slots[8] = silver.resource_stress_disk_io
        slots[9] = silver.resource_stress_thermal
        slots[10] = silver.calibration_delta if silver.calibration_delta is not None else 0.0
        slots[11] = silver.error_rate if silver.error_rate is not None else 0.0
        slots[12] = op_map[silver.operational_state]

        if silver.calibration_delta is None:
            mask[10] = 0.0
        if silver.error_rate is None:
            mask[11] = 0.0
        if silver.stale:
            slots[4:10] = 0.0
            mask[4:10] = 0.0

        return cls(slots=slots, mask=mask)

    def __getitem__(self, idx: int) -> float:
        return float(self.slots[idx])
