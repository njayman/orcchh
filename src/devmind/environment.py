from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from devmind.dataset import load_task_dataset
from devmind.edge import EdgeDevice
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, MetricSource, SilverEnricher
from devmind.model_clients import BERTLargeCloud, DistilBERTEdge, InferenceResult
from devmind.models import (
    Action,
    BronzeMetricSnapshot,
    EdgeContextReport,
    GoldStateVector,
    OperationalState,
    ResourceStress,
)


@dataclass
class ScenarioConfig:
    name: str = "steady"
    task: str = "toxicity"
    base_rate: float = 4000.0
    burst_rate: float = 4000.0
    burst_start_frac: float = 1.1
    burst_duration_frac: float = 0.0
    degraded_start_frac: float = 1.1
    degraded_duration_frac: float = 0.0
    rtt_base: float = 40.0
    rtt_degraded: float = 40.0
    edge_stress_prob: float = 0.1
    edge_degrade_prob: float = 0.02
    edge_unreachable_prob: float = 0.01
    cloud_service_time_ms: float = 8.0
    sla_budget_ms: float = 300.0
    reward_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)

    @classmethod
    def steady(cls, task: str = "toxicity") -> ScenarioConfig:
        return cls(name="steady", task=task, base_rate=4000, burst_rate=4000)

    @classmethod
    def bursty(cls, task: str = "toxicity") -> ScenarioConfig:
        return cls(
            name="bursty",
            task=task,
            base_rate=4000,
            burst_rate=22000,
            burst_start_frac=0.2,
            burst_duration_frac=0.5,
        )

    @classmethod
    def degraded_network(cls, task: str = "toxicity") -> ScenarioConfig:
        return cls(
            name="degraded_network",
            task=task,
            base_rate=4000,
            burst_rate=4000,
            rtt_degraded=800.0,
            degraded_start_frac=0.0,
            degraded_duration_frac=1.0,
        )

    @classmethod
    def held_out(cls, task: str = "toxicity") -> ScenarioConfig:
        return cls(
            name="held_out",
            task=task,
            base_rate=2000,
            burst_rate=30000,
            burst_start_frac=0.15,
            burst_duration_frac=0.6,
            rtt_base=100.0,
            rtt_degraded=1200.0,
            degraded_start_frac=0.0,
            degraded_duration_frac=1.0,
            edge_stress_prob=0.4,
            edge_degrade_prob=0.15,
            cloud_service_time_ms=80.0,
        )


@dataclass
class SimulationState:
    t: float = 0.0
    cloud_queue: int = 0
    total_requests: int = 0
    sla_violations: int = 0
    escalations: int = 0
    edge_routes: int = 0
    edge_errors: int = 0
    cloud_correct: int = 0
    edge_correct: int = 0
    edge_total: int = 0
    cloud_total: int = 0
    edge_unreachable_events: int = 0
    energy_mj: float = 0.0
    _drain_acc: float = 0.0
    _last_latency: float = 0.0
    _last_accuracy: float = 0.0
    _last_sla_met: bool = True
    _sample_idx: int = 0


class InferenceGatewayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        seed: int | None = None,
        max_samples: int = 5000,
        edge_model: DistilBERTEdge | None = None,
        cloud_model: BERTLargeCloud | None = None,
        use_reflect: bool = True,
    ):
        super().__init__()
        self.scenario = scenario or ScenarioConfig.steady()
        self.use_reflect = use_reflect
        self._rng = np.random.default_rng(seed)
        self._edge_device = EdgeDevice()
        self._silver = SilverEnricher()
        self._gold = GoldNormalizer()
        self._state = SimulationState()
        self._bronze_registry = DynamicMetricRegistry()
        self._setup_registry()

        self._dataset = load_task_dataset(self.scenario.task, max_samples=max_samples)
        self._episode_length = max(len(self._dataset.test), 1)
        self._edge_model = edge_model or DistilBERTEdge(task=self.scenario.task)
        self._cloud_model = cloud_model or BERTLargeCloud(task=self.scenario.task)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _setup_registry(self) -> None:
        def _read_edge() -> EdgeContextReport:
            report = self._edge_device.last_report
            if report is None:
                return EdgeContextReport(operational_state=OperationalState.UNREACHABLE)
            return report

        def _read_queue() -> int:
            return self._state.cloud_queue

        def _read_rtt() -> float:
            return self._rtt

        def _read_sla() -> float:
            return self.scenario.sla_budget_ms

        self._bronze_registry.register(MetricSource("edge_context", "EdgeContextReport", _read_edge))
        self._bronze_registry.register(MetricSource("cloud_queue_depth", "int", _read_queue))
        self._bronze_registry.register(MetricSource("rtt_ms", "float", _read_rtt))
        self._bronze_registry.register(MetricSource("sla_remaining_ms", "float", _read_sla))
        self._bronze_registry.register(MetricSource("energy_mj", "float", lambda: 0.0))
        self._bronze_registry.register(MetricSource("traffic_intensity", "float", lambda: 0.0))

    def _progress(self) -> float:
        return self._state.total_requests / self._episode_length

    def _get_current_rate(self) -> float:
        progress = self._progress()
        s = self.scenario
        if s.burst_start_frac <= progress < s.burst_start_frac + s.burst_duration_frac:
            return s.burst_rate
        return s.base_rate

    def _get_current_rtt(self) -> float:
        progress = self._progress()
        s = self.scenario
        if s.degraded_start_frac <= progress < s.degraded_start_frac + s.degraded_duration_frac:
            return s.rtt_degraded
        return s.rtt_base

    def _build_bronze(self) -> BronzeMetricSnapshot:
        rate = self._get_current_rate()
        load_pct = min(rate / self.scenario.base_rate, 5.0)
        self._edge_device.apply_stress(
            cpu=self._rng.uniform(0, 0.3) * load_pct
            if self._rng.uniform() > self.scenario.edge_stress_prob
            else self._rng.uniform(0.6, 0.95),
            thermal=self._rng.uniform(0, 0.2) + 0.1 * load_pct
            if self._rng.uniform() > self.scenario.edge_stress_prob * 0.7
            else self._rng.uniform(0.6, 0.9),
            memory=self._rng.uniform(0.1, 0.4)
            if self._rng.uniform() > self.scenario.edge_stress_prob * 0.5
            else self._rng.uniform(0.7, 0.95),
            disk_io=self._rng.uniform(0.1, 0.2)
            if self._rng.uniform() > self.scenario.edge_stress_prob * 0.5
            else self._rng.uniform(0.7, 0.9),
            gpu=self._rng.uniform(0.0, 0.3),
        )
        self._rtt = self._get_current_rtt()
        return self._bronze_registry.snapshot()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._state = SimulationState()
        self._edge_device = EdgeDevice()
        self._silver = SilverEnricher()
        self._rng = np.random.default_rng(seed)
        self._rtt = self.scenario.rtt_base
        self._dataset.shuffle_train()
        bronze = self._build_bronze()
        silver = self._silver.enrich(bronze)
        gold = self._gold.normalize(silver)
        return gold.slots.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_results = self._inference_step(action)
        bronze = self._build_bronze()
        silver = self._silver.enrich(bronze)
        gold = self._gold.normalize(silver)
        rate = self._get_current_rate()
        dt = 60.0 / rate if rate > 0 else 1.0

        self._state.total_requests += 1
        reward, latency, sla_met, accuracy = step_results
        self._state._last_latency = latency
        self._state._last_accuracy = accuracy
        self._state._last_sla_met = sla_met
        self._advance_clock(dt)

        done = self._state.total_requests >= len(self._dataset.test)
        return gold.slots.copy(), reward, done, False, {}

    def _inference_step(self, action: int) -> tuple[float, float, bool, float]:
        samples = self._dataset.test
        if not samples:
            return 0.0, 0.0, True, 0.0

        idx = self._state._sample_idx % len(samples)
        sample = samples[idx]
        self._state._sample_idx += 1
        text = sample.text
        true_label = sample.label
        sla = self.scenario.sla_budget_ms

        if action in (Action.ROUTE_TO_EDGE, Action.QUERY_EXTENDED_CONTEXT):
            if self._rng.uniform() < self.scenario.edge_unreachable_prob:
                self._edge_device.mark_unreachable()
                self._state.edge_unreachable_events += 1
                action = Action.ESCALATE_TO_CLOUD

        perception_cost = 0.0
        if action in (Action.ROUTE_TO_EDGE, Action.QUERY_EXTENDED_CONTEXT):
            if action == Action.QUERY_EXTENDED_CONTEXT:
                perception_cost = 0.02
            self._state.edge_routes += 1
            self._state.edge_total += 1
            result = self._edge_model.predict(text, true_label)
            confidence = result.confidence
            report = self._edge_device.emit_report(confidence, result.is_correct if self.use_reflect else None)
            latency = (
                result.latency_ms
                + self._rtt * 0.5
                + report.resource_stress.memory * 40.0
                + report.resource_stress.disk_io * 40.0
            )
            sla_met = latency <= sla
            energy = 0.5 + 0.3 * report.resource_stress.cpu
            accuracy = 1.0 if result.is_correct else 0.0
            if result.is_correct:
                self._state.edge_correct += 1
            else:
                self._state.edge_errors += 1

        elif action == Action.ESCALATE_TO_CLOUD:
            self._state.escalations += 1
            self._state.cloud_total += 1
            queue_wait = self._state.cloud_queue * self.scenario.cloud_service_time_ms
            self._state.cloud_queue += 1
            result = self._cloud_model.predict(text, true_label)
            latency = queue_wait + self._rtt + result.latency_ms
            sla_met = latency <= sla
            energy = 2.0 + 0.1 * self._state.cloud_queue
            accuracy = 1.0 if result.is_correct else 0.0
            if result.is_correct:
                self._state.cloud_correct += 1
        else:
            return -0.05, 0.0, True, 0.0

        self._edge_device.update_from_outcome(latency, sla_met, accuracy)

        reward = self._compute_reward(accuracy, latency, sla, sla_met, energy) - perception_cost
        self._state.energy_mj += energy
        if not sla_met:
            self._state.sla_violations += 1
        return float(np.clip(reward, -1.0, 1.0)), latency, sla_met, accuracy

    def _compute_reward(
        self, accuracy: float, latency: float, sla_budget: float, sla_met: bool, energy: float
    ) -> float:
        w_accuracy, w_sla, w_energy, w_latency = self.scenario.reward_weights
        return (
            w_accuracy * accuracy
            + w_sla * (1.0 if sla_met else -0.5)
            + w_energy * (1.0 - energy / 5.0)
            + w_latency * max(0.0, 1.0 - latency / sla_budget)
        )

    def _advance_clock(self, dt: float) -> None:
        self._state.t += dt
        service_per_sec = 1000.0 / self.scenario.cloud_service_time_ms
        self._state._drain_acc += service_per_sec * dt
        drain = int(self._state._drain_acc)
        if drain > 0:
            self._state._drain_acc -= drain
            self._state.cloud_queue = max(0, self._state.cloud_queue - drain)

    def render(self) -> None:
        print(
            f"req={self._state.total_requests} t={self._state.t:.1f}s "
            f"queue={self._state.cloud_queue} violations={self._state.sla_violations}"
        )


def demo() -> None:
    from types import SimpleNamespace

    default_weights = SimpleNamespace(scenario=SimpleNamespace(reward_weights=(0.4, 0.3, 0.2, 0.1)))
    accuracy_only = SimpleNamespace(scenario=SimpleNamespace(reward_weights=(1.0, 0.0, 0.0, 0.0)))

    r_default = InferenceGatewayEnv._compute_reward(default_weights, accuracy=1.0, latency=100.0, sla_budget=300.0, sla_met=True, energy=1.0)
    r_accuracy_only = InferenceGatewayEnv._compute_reward(accuracy_only, accuracy=1.0, latency=100.0, sla_budget=300.0, sla_met=True, energy=1.0)

    assert abs(r_accuracy_only - 1.0) < 1e-9, "accuracy-only weights should return raw accuracy"
    assert r_default != r_accuracy_only, "reward_weights must actually change the computed reward"
    print("environment self-check passed")


if __name__ == "__main__":
    demo()
