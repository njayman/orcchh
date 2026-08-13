from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from devmind.models import Action, GoldStateVector


class PPONetwork(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, n_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value

    def get_action_and_entropy(self, state: torch.Tensor) -> tuple[int, float]:
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = int(dist.sample().item())
        entropy = float(dist.entropy().item())
        return action, entropy


@dataclass
class PerceptionCache:
    confidence: float | None = None
    cloud_queue_depth: int | None = None
    edge_rtt: float | None = None
    sla_remaining: float | None = None
    edge_cpu_load: float | None = None
    calibration_delta: float | None = None
    error_rate: float | None = None
    operational_state: str | None = None


class MCPSkillInterface:
    def __init__(self, gold_vector: GoldStateVector):
        self._gold = gold_vector
        self._skills: dict[str, bool] = {
            "get_model_confidence": True,
            "get_cloud_queue_depth": True,
            "get_edge_rtt": True,
            "get_sla_remaining": True,
            "get_edge_cpu_load": True,
            "get_edge_calibration_delta": False,
            "get_edge_error_rate": False,
            "get_edge_operational_state": False,
        }

    def list_tools(self) -> list[str]:
        return [name for name, active in self._skills.items() if active]

    def register_skill(self, name: str) -> None:
        if name in self._skills:
            self._skills[name] = True

    def call(self, skill_name: str) -> Any:
        g = self._gold
        match skill_name:
            case "get_model_confidence":
                return g[0]
            case "get_cloud_queue_depth":
                return g[1]
            case "get_edge_rtt":
                return g[2]
            case "get_sla_remaining":
                return g[3]
            case "get_edge_cpu_load":
                return g[4]
            case "get_edge_calibration_delta":
                return g[10] if g.mask[10] else None
            case "get_edge_error_rate":
                return g[11] if g.mask[11] else None
            case "get_edge_operational_state":
                return g[12]
        return None

    def perceive(self) -> PerceptionCache:
        return PerceptionCache(
            confidence=self.call("get_model_confidence"),
            cloud_queue_depth=self.call("get_cloud_queue_depth"),
            edge_rtt=self.call("get_edge_rtt"),
            sla_remaining=self.call("get_sla_remaining"),
            edge_cpu_load=self.call("get_edge_cpu_load"),
            calibration_delta=self.call("get_edge_calibration_delta"),
            error_rate=self.call("get_edge_error_rate"),
            operational_state=self.call("get_edge_operational_state"),
        )


class AgenticOrchestrator:
    ENTROPY_FALLBACK_THRESHOLD = 0.9
    FALLBACK_THRESHOLD = 0.9
    OOD_ZSCORE_THRESHOLD = 4.0
    OOD_STD_FLOOR = 0.02
    OOD_MIN_ANOMALOUS_SLOTS = 2
    QUERY_CALIBRATION_RISK_THRESHOLD = 0.2
    QUERY_ERROR_RATE_RISK_THRESHOLD = 0.15
    FALLBACK_QUEUE_WAIT_CEILING = 0.4

    def __init__(self, policy: PPONetwork | None = None, device: str = "cpu", state_stats_path: str | None = None):
        self.policy = policy or PPONetwork()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self.log_buffer: list[dict[str, Any]] = []
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None
        if state_stats_path and os.path.exists(state_stats_path):
            with open(state_stats_path) as f:
                stats = json.load(f)
            self.state_mean = np.array(stats["mean"])
            self.state_std = np.array(stats["std"])

        self.fallback_queue_wait_ceiling = self.FALLBACK_QUEUE_WAIT_CEILING
        self.fallback_guards: list[tuple[str, Callable[[GoldStateVector], bool], Action]] = [
            ("queue_backed_up", lambda gold: gold[1] > self.fallback_queue_wait_ceiling, Action.ROUTE_TO_EDGE),
            ("low_confidence", lambda gold: gold[0] < self.FALLBACK_THRESHOLD, Action.ESCALATE_TO_CLOUD),
        ]

    def perceive(self, gold: GoldStateVector) -> PerceptionCache:
        mcp = MCPSkillInterface(gold)
        return mcp.perceive()

    def reason(self, gold: GoldStateVector) -> tuple[int, float]:
        state = torch.from_numpy(gold.slots).float().unsqueeze(0).to(self.device)
        return self.policy.get_action_and_entropy(state)

    def is_ood(self, gold: GoldStateVector) -> bool:
        if self.state_mean is None:
            return False
        std = np.maximum(self.state_std, self.OOD_STD_FLOOR)
        z = np.abs((gold.slots - self.state_mean) / std)
        return bool((z > self.OOD_ZSCORE_THRESHOLD).sum() >= self.OOD_MIN_ANOMALOUS_SLOTS)

    def act(self, action: int, mcp: MCPSkillInterface) -> Action:
        if action == Action.QUERY_EXTENDED_CONTEXT:
            mcp.register_skill("get_edge_calibration_delta")
            mcp.register_skill("get_edge_error_rate")
            mcp.register_skill("get_edge_operational_state")
            return Action.QUERY_EXTENDED_CONTEXT
        return Action(action)

    def _resolve_fallback(self, gold: GoldStateVector) -> Action:
        for _name, guard, guard_action in self.fallback_guards:
            if guard(gold):
                return guard_action
        return Action.ROUTE_TO_EDGE

    def decide(self, gold: GoldStateVector) -> tuple[Action, bool]:
        action, entropy = self.reason(gold)
        fallback = entropy > self.ENTROPY_FALLBACK_THRESHOLD or self.is_ood(gold)
        if fallback:
            return self._resolve_fallback(gold), fallback
        if action == Action.QUERY_EXTENDED_CONTEXT:
            mcp = MCPSkillInterface(gold)
            self.act(action, mcp)
            calibration_delta = mcp.call("get_edge_calibration_delta")
            error_rate = mcp.call("get_edge_error_rate")
            high_risk = (
                calibration_delta is not None and calibration_delta > self.QUERY_CALIBRATION_RISK_THRESHOLD
            ) or (error_rate is not None and error_rate > self.QUERY_ERROR_RATE_RISK_THRESHOLD)
            return (Action.ESCALATE_TO_CLOUD if high_risk else Action.ROUTE_TO_EDGE), False
        return Action(action), False

    def reflect(self, latency_ms: float, sla_met: bool, accuracy: float, tier: str, fallback: bool = False) -> None:
        self.log_buffer.append(
            {"latency_ms": latency_ms, "sla_met": sla_met, "accuracy": accuracy, "tier": tier, "fallback": fallback}
        )


class _ScriptedPolicy:
    def __init__(self, actions: list[tuple[int, float]]):
        self._actions = list(actions)
        self._calls = 0

    def to(self, device: Any) -> _ScriptedPolicy:
        return self

    def eval(self) -> None:
        pass

    def get_action_and_entropy(self, state: torch.Tensor) -> tuple[int, float]:
        result = self._actions[min(self._calls, len(self._actions) - 1)]
        self._calls += 1
        return result


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = f"{tmp}/state_stats.json"
        with open(stats_path, "w") as f:
            json.dump({"mean": [0.5] * 13, "std": [0.1] * 13}, f)

        orch = AgenticOrchestrator(state_stats_path=stats_path)
        in_distribution = GoldStateVector(slots=np.full(13, 0.5, dtype=np.float32))
        far_out = GoldStateVector(slots=np.full(13, 5.0, dtype=np.float32))

        assert not orch.is_ood(in_distribution)
        assert orch.is_ood(far_out)

        no_stats_orch = AgenticOrchestrator()
        assert not no_stats_orch.is_ood(far_out)

        one_slot_spike = np.full(13, 0.5, dtype=np.float32)
        one_slot_spike[0] = 5.0
        assert not orch.is_ood(GoldStateVector(slots=one_slot_spike)), (
            "a single anomalous slot (e.g. low confidence under sustained degradation) "
            "must not alone distrust the whole state"
        )

        two_slot_spike = np.full(13, 0.5, dtype=np.float32)
        two_slot_spike[0] = 5.0
        two_slot_spike[1] = 5.0
        assert orch.is_ood(GoldStateVector(slots=two_slot_spike)), (
            "correlated anomalies across multiple slots should still trigger OOD"
        )

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = f"{tmp}/near_constant_stats.json"
        near_constant_std = [0.1] * 13
        near_constant_std[3] = 1e-6
        with open(stats_path, "w") as f:
            json.dump({"mean": [0.5] * 13, "std": near_constant_std}, f)
        orch2 = AgenticOrchestrator(state_stats_path=stats_path)
        tiny_real_deviation = np.full(13, 0.5, dtype=np.float32)
        tiny_real_deviation[3] = 0.5013
        assert not orch2.is_ood(GoldStateVector(slots=tiny_real_deviation)), (
            "a near-zero-std slot must not dominate the OOD check on a tiny real deviation"
        )

    querying = AgenticOrchestrator(policy=_ScriptedPolicy([(Action.QUERY_EXTENDED_CONTEXT, 0.1)]))
    revealed_mask = np.ones(13, dtype=np.float32)

    low_risk_slots = np.full(13, 0.5, dtype=np.float32)
    low_risk_slots[10] = 0.05
    low_risk_slots[11] = 0.05
    action, fallback = querying.decide(GoldStateVector(slots=low_risk_slots, mask=revealed_mask))
    assert action == Action.ROUTE_TO_EDGE and not fallback, "QUERY with low calibration/error risk should route to edge"

    high_risk_slots = np.full(13, 0.5, dtype=np.float32)
    high_risk_slots[10] = 0.9
    action, fallback = querying.decide(GoldStateVector(slots=high_risk_slots, mask=revealed_mask))
    assert action == Action.ESCALATE_TO_CLOUD and not fallback, "QUERY revealing high calibration_delta must escalate"

    masked_slots = np.full(13, 0.5, dtype=np.float32)
    masked_slots[10] = 0.9
    action, fallback = querying.decide(GoldStateVector(slots=masked_slots))
    assert action == Action.ROUTE_TO_EDGE, "QUERY with masked-out extended slots has no signal to escalate on"

    uncertain = AgenticOrchestrator(policy=_ScriptedPolicy([(Action.ROUTE_TO_EDGE, 0.99)]))
    queue_backed_up_slots = np.full(13, 0.5, dtype=np.float32)
    queue_backed_up_slots[0] = 0.5
    queue_backed_up_slots[1] = 0.9
    action, fallback = uncertain.decide(GoldStateVector(slots=queue_backed_up_slots))
    assert fallback and action == Action.ROUTE_TO_EDGE, (
        "a backed-up queue must not be escalated into further, even under low confidence"
    )

    normal_queue_slots = np.full(13, 0.5, dtype=np.float32)
    normal_queue_slots[0] = 0.2
    normal_queue_slots[1] = 0.05
    action, fallback = uncertain.decide(GoldStateVector(slots=normal_queue_slots))
    assert fallback and action == Action.ESCALATE_TO_CLOUD, (
        "with a healthy queue, low confidence should still fall through to the confidence guard"
    )

    uncertain.fallback_queue_wait_ceiling = 0.95
    action, fallback = uncertain.decide(GoldStateVector(slots=queue_backed_up_slots))
    assert fallback and action == Action.ESCALATE_TO_CLOUD, (
        "raising the calibrated ceiling must change behavior without editing fallback_guards"
    )

    print("agent self-check passed")


if __name__ == "__main__":
    demo()
