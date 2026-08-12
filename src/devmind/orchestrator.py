
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import torch

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.environment import InferenceGatewayEnv, ScenarioConfig
from devmind.evaluation import EvalMetrics, average_metrics, make_env, ppo_policy, run_episode
from devmind.models import EdgeContextReport, OperationalState
from devmind.trainer import PPOTrainer


class PolicyDecision(str, Enum):
    REUSE = "reuse"
    FINE_TUNE = "fine_tune"
    TRAIN_NEW = "train_new"


@dataclass
class PolicyRecord:
    policy_id: str
    checkpoint_path: str
    validated_scenarios: list[str] = field(default_factory=list)
    clients_assigned: list[str] = field(default_factory=list)


@dataclass
class ToleranceThresholds:

    max_sla_violation_rate: float = 0.15
    min_accuracy: float = 0.80
    max_escalation_rate: float = 0.60


THRESHOLD_BOUNDS = {
    "max_sla_violation_rate": (0.05, 0.30),
    "min_accuracy": (0.65, 0.90),
    "max_escalation_rate": (0.40, 0.80),
}


def _meets_tolerance(m: EvalMetrics, t: ToleranceThresholds) -> bool:
    return (
        m.sla_violation_rate <= t.max_sla_violation_rate
        and m.accuracy >= t.min_accuracy
        and m.escalation_rate <= t.max_escalation_rate
    )


def select_decision(
    candidates: dict[str, EvalMetrics], thresholds: ToleranceThresholds
) -> tuple[PolicyDecision, str | None]:
    fits = {pid: m for pid, m in candidates.items() if _meets_tolerance(m, thresholds)}
    if fits:
        best = max(fits, key=lambda pid: fits[pid].accuracy - fits[pid].sla_violation_rate)
        return PolicyDecision.REUSE, best
    if candidates:
        closest = max(candidates, key=lambda pid: candidates[pid].accuracy - candidates[pid].sla_violation_rate)
        return PolicyDecision.FINE_TUNE, closest
    return PolicyDecision.TRAIN_NEW, None


def dominant_signal(m: EvalMetrics, thresholds: ToleranceThresholds) -> str:
    gaps = {
        "sla_violation_rate": m.sla_violation_rate - thresholds.max_sla_violation_rate,
        "accuracy": thresholds.min_accuracy - m.accuracy,
        "escalation_rate": m.escalation_rate - thresholds.max_escalation_rate,
    }
    worst = max(gaps, key=gaps.get)
    return worst if gaps[worst] > 0 else "within_tolerance"


class PolicyOrchestrator:
    def __init__(
        self,
        library_dir: str = "policy_library",
        thresholds: ToleranceThresholds | None = None,
        edge_model: Any = None,
        cloud_model: Any = None,
        log_path: str | None = None,
        fine_tune_steps: int = 5_000,
        train_new_steps: int = 50_000,
        eval_n_runs: int = 3,
    ):
        self.library_dir = library_dir
        self.thresholds = thresholds or ToleranceThresholds()
        self.library: dict[str, PolicyRecord] = {}
        self._edge_model_override = edge_model
        self._cloud_model_override = cloud_model
        self._edge_models: dict[str, Any] = {}
        self._cloud_models: dict[str, Any] = {}
        self.fine_tune_steps = fine_tune_steps
        self.train_new_steps = train_new_steps
        self.eval_n_runs = eval_n_runs
        self.log_path = log_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation", "orchestrator_decisions.jsonl"
        )
        os.makedirs(self.library_dir, exist_ok=True)
        self._onboard_count = 0
        self.recalibrate_every = 5

    def _models_for(self, task: str) -> tuple[Any, Any]:
        if self._edge_model_override is not None and self._cloud_model_override is not None:
            return self._edge_model_override, self._cloud_model_override
        if task not in self._edge_models:
            from devmind.model_clients import BERTLargeCloud, DistilBERTEdge

            self._edge_models[task] = DistilBERTEdge(task=task)
            self._cloud_models[task] = BERTLargeCloud(task=task)
        return self._edge_models[task], self._cloud_models[task]

    def register_seed_policy(self, policy_id: str, checkpoint_path: str, validated_scenarios: list[str]) -> None:
        self.library[policy_id] = PolicyRecord(policy_id, checkpoint_path, validated_scenarios)

    def onboard(
        self, client: str, scenario: ScenarioConfig, max_samples: int = 500, trigger: str = "onboarding"
    ) -> PolicyDecision:
        candidates = {
            pid: self._evaluate(rec.checkpoint_path, scenario, max_samples)
            for pid, rec in self.library.items()
        }
        decision, chosen = select_decision(candidates, self.thresholds)

        if decision == PolicyDecision.FINE_TUNE:
            chosen = self._fine_tune(chosen, scenario)
        elif decision == PolicyDecision.TRAIN_NEW:
            chosen = self._train_new(client, scenario)

        rec = self.library[chosen]
        if scenario.name not in rec.validated_scenarios:
            rec.validated_scenarios.append(scenario.name)
        if client not in rec.clients_assigned:
            rec.clients_assigned.append(client)

        self._log_decision(client, scenario, candidates, decision, chosen, trigger)
        self._onboard_count += 1
        if self._onboard_count % self.recalibrate_every == 0:
            self.recalibrate_thresholds()
        return decision

    def _evaluate(self, checkpoint_path: str, scenario: ScenarioConfig, max_samples: int) -> EvalMetrics:
        edge_model, cloud_model = self._models_for(scenario.task)
        ppo = PPONetwork()
        ppo.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        agent = AgenticOrchestrator(ppo)
        metrics_list = [
            run_episode(
                make_env(scenario, max_samples, edge_model, cloud_model),
                ppo_policy(agent),
                desc=f"orchestrator/{scenario.name}",
            )
            for _ in range(self.eval_n_runs)
        ]
        return average_metrics(metrics_list)

    def _train(self, scenario: ScenarioConfig, total_steps: int, init_state_dict: dict | None = None) -> PPONetwork:
        edge_model, cloud_model = self._models_for(scenario.task)
        env = InferenceGatewayEnv(scenario, edge_model=edge_model, cloud_model=cloud_model)
        trainer = PPOTrainer(env)
        if init_state_dict is not None:
            trainer.policy.load_state_dict(init_state_dict)
        step = 0
        while step < total_steps:
            trainer.collect_rollout(2048)
            trainer.train()
            step += 2048
        return trainer.policy

    def _fine_tune(self, base_policy_id: str, scenario: ScenarioConfig, steps: int | None = None) -> str:
        base = self.library[base_policy_id]
        state_dict = torch.load(base.checkpoint_path, map_location="cpu", weights_only=True)
        policy = self._train(scenario, steps or self.fine_tune_steps, init_state_dict=state_dict)
        new_id = f"{base_policy_id}_ft_{scenario.name}"
        path = os.path.join(self.library_dir, f"{new_id}.pt")
        torch.save(policy.state_dict(), path)
        self.library[new_id] = PolicyRecord(new_id, path)
        return new_id

    def _train_new(self, client: str, scenario: ScenarioConfig, steps: int | None = None) -> str:
        policy = self._train(scenario, steps or self.train_new_steps)
        new_id = f"{client}_{scenario.name}"
        path = os.path.join(self.library_dir, f"{new_id}.pt")
        torch.save(policy.state_dict(), path)
        self.library[new_id] = PolicyRecord(new_id, path)
        return new_id

    def _log_decision(
        self,
        client: str,
        scenario: ScenarioConfig,
        candidates: dict[str, EvalMetrics],
        decision: PolicyDecision,
        chosen: str,
        trigger: str = "onboarding",
    ) -> None:
        signal = dominant_signal(candidates[chosen], self.thresholds) if chosen in candidates else "n/a_new_policy"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client": client,
            "scenario": scenario.name,
            "task": scenario.task,
            "candidates_evaluated": {pid: vars(m) for pid, m in candidates.items()},
            "decision": decision.value,
            "policy_assigned": chosen,
            "dominant_signal": signal,
            "trigger": trigger,
        }
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _false_reuse_rate(self, window: int = 20) -> float | None:
        if not os.path.exists(self.log_path):
            return None
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        reuse_events = [e for e in entries if e["decision"] == "reuse"]
        if not reuse_events:
            return None
        reuse_events = reuse_events[-window:]
        false_reuses = 0
        for e in reuse_events:
            later = [
                x for x in entries
                if x["client"] == e["client"] and x["timestamp"] > e["timestamp"]
                and x["trigger"] == "drift_detected"
            ]
            if later:
                false_reuses += 1
        return false_reuses / len(reuse_events)

    def recalibrate_thresholds(self, window: int = 20, step: float = 0.02) -> dict[str, Any] | None:
        """Governance-time self-improvement: tighten/loosen tolerance from the
        orchestrator's own decision-log track record. Never called from the
        per-request fast loop."""
        rate = self._false_reuse_rate(window)
        if rate is None:
            return None
        old = {f.name: getattr(self.thresholds, f.name) for f in self.thresholds.__dataclass_fields__.values()}
        direction = 1 if rate > 0.15 else (-1 if rate < 0.03 else 0)
        if direction != 0:
            for name in ("max_sla_violation_rate", "max_escalation_rate"):
                lo, hi = THRESHOLD_BOUNDS[name]
                new_val = getattr(self.thresholds, name) - direction * step
                setattr(self.thresholds, name, max(lo, min(hi, new_val)))
            lo, hi = THRESHOLD_BOUNDS["min_accuracy"]
            new_val = self.thresholds.min_accuracy + direction * step
            self.thresholds.min_accuracy = max(lo, min(hi, new_val))

        new = {f.name: getattr(self.thresholds, f.name) for f in self.thresholds.__dataclass_fields__.values()}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "threshold_recalibrated",
            "false_reuse_rate": rate,
            "old_thresholds": old,
            "new_thresholds": new,
            "changed": old != new,
        }
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


class DriftEventListener:
    def __init__(
        self,
        orchestrator: PolicyOrchestrator,
        trust_floor: float = 0.5,
        recovery_window_s: float = 30.0,
        error_rate_ceiling: float = 0.15,
    ):
        self.orchestrator = orchestrator
        self.trust_floor = trust_floor
        self.recovery_window_s = recovery_window_s
        self.error_rate_ceiling = error_rate_ceiling
        self._distress_since: dict[str, float] = {}
        self._queue: asyncio.Queue[tuple[str, EdgeContextReport, float]] = asyncio.Queue()

    def should_escalate(self, client_id: str, report: EdgeContextReport, now: float) -> bool:
        state_distressed = report.operational_state in (OperationalState.DEGRADING, OperationalState.UNREACHABLE)
        low_trust = report.trust_score < self.trust_floor
        high_error = report.error_rate_1m > self.error_rate_ceiling
        # multi-signal correlation: state distress alone must sustain trust_floor breach
        # over recovery_window_s, but a simultaneous error-rate spike escalates immediately
        if state_distressed and low_trust and high_error:
            self._distress_since.pop(client_id, None)
            return True
        if not state_distressed or not low_trust:
            self._distress_since.pop(client_id, None)
            return False
        first_seen = self._distress_since.setdefault(client_id, now)
        if now - first_seen < self.recovery_window_s:
            return False
        self._distress_since.pop(client_id, None)
        return True

    def notify(self, client_id: str, report: EdgeContextReport) -> None:
        self._queue.put_nowait((client_id, report, time.monotonic()))

    async def run_forever(self, scenario_lookup: dict[str, ScenarioConfig], max_samples: int = 500) -> None:
        loop = asyncio.get_running_loop()
        while True:
            client_id, report, now = await self._queue.get()
            scenario = scenario_lookup.get(client_id)
            if scenario is not None and self.should_escalate(client_id, report, now):
                await loop.run_in_executor(
                    None, self.orchestrator.onboard, client_id, scenario, max_samples, "drift_detected"
                )


CLIENT_SCENARIOS: dict[str, ScenarioConfig] = {
    "client_streamforge": ScenarioConfig.bursty(),
    "client_nhs": ScenarioConfig.steady(),
    "client_babcock": ScenarioConfig.degraded_network(),
    "client_newco": ScenarioConfig(
        name="client_newco",
        base_rate=4000,
        burst_rate=4000,
        edge_stress_prob=0.35,
        edge_degrade_prob=0.10,
    ),
}


def run_ablation_7(
    seed_policy_path: str = "ppo_policy.pt",
    max_samples: int = 500,
    fine_tune_steps: int = 5_000,
    train_new_steps: int = 50_000,
    edge_model: Any = None,
    cloud_model: Any = None,
    eval_n_runs: int = 3,
) -> dict[str, Any]:
    shared_ppo = PPONetwork()
    shared_ppo.load_state_dict(torch.load(seed_policy_path, map_location="cpu", weights_only=True))
    shared_agent = AgenticOrchestrator(shared_ppo)

    shared_results: dict[str, EvalMetrics] = {}
    for client, scenario in CLIENT_SCENARIOS.items():
        metrics_list = [
            run_episode(
                make_env(scenario, max_samples, edge_model, cloud_model),
                ppo_policy(shared_agent),
                desc=f"run7_shared/{client}",
            )
            for _ in range(eval_n_runs)
        ]
        shared_results[client] = average_metrics(metrics_list)

    orch = PolicyOrchestrator(
        edge_model=edge_model,
        cloud_model=cloud_model,
        fine_tune_steps=fine_tune_steps,
        train_new_steps=train_new_steps,
        eval_n_runs=eval_n_runs,
    )
    orch.register_seed_policy(
        "seed", seed_policy_path, validated_scenarios=["steady", "bursty", "degraded_network"]
    )

    decisions: dict[str, PolicyDecision] = {}
    orchestrated_results: dict[str, EvalMetrics] = {}
    for client, scenario in CLIENT_SCENARIOS.items():
        decisions[client] = orch.onboard(client, scenario, max_samples=max_samples)
        assigned = next(pid for pid, rec in orch.library.items() if client in rec.clients_assigned)
        orchestrated_results[client] = orch._evaluate(orch.library[assigned].checkpoint_path, scenario, max_samples)

    return {
        "shared": shared_results,
        "orchestrated": orchestrated_results,
        "decisions": {k: v.value for k, v in decisions.items()},
    }


def main_ablation_7() -> None:
    import datetime
    import time

    from devmind.evaluation import print_results, save_results
    from devmind.model_clients import BERTLargeCloud, DistilBERTEdge

    print("Loading models (one-time)...")
    t0 = time.perf_counter()
    edge_model = DistilBERTEdge()
    cloud_model = BERTLargeCloud()
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    result = run_ablation_7(edge_model=edge_model, cloud_model=cloud_model)

    text_buffer: list[str] = []
    json_buffer: list[dict] = []
    save_results(result["shared"], "RUN 7: Single Shared Policy", text_buffer, json_buffer)
    save_results(result["orchestrated"], "RUN 7: Policy Orchestration Layer", text_buffer, json_buffer)
    print_results(result["shared"], title="RUN 7: Single Shared Policy")
    print_results(result["orchestrated"], title="RUN 7: Policy Orchestration Layer")
    print("\nPer-client decisions:", result["decisions"])
    text_buffer.append(f"\nPer-client decisions: {result['decisions']}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(out_dir, f"run7_{timestamp}.txt"), "w") as f:
        f.write("\n".join(text_buffer))
    with open(os.path.join(out_dir, f"run7_{timestamp}.json"), "w") as f:
        json.dump(json_buffer, f, indent=2)
    print(f"\nResults saved to {out_dir}/run7_{timestamp}.*")


def demo() -> None:
    t = ToleranceThresholds()

    good = EvalMetrics(accuracy=0.9, sla_violation_rate=0.05, escalation_rate=0.3)
    close_miss = EvalMetrics(accuracy=0.7, sla_violation_rate=0.2, escalation_rate=0.5)

    decision, chosen = select_decision({"p1": good}, t)
    assert decision == PolicyDecision.REUSE and chosen == "p1"

    decision, chosen = select_decision({"p1": close_miss}, t)
    assert decision == PolicyDecision.FINE_TUNE and chosen == "p1"

    decision, chosen = select_decision({}, t)
    assert decision == PolicyDecision.TRAIN_NEW and chosen is None

    assert dominant_signal(close_miss, t) == "accuracy"
    assert dominant_signal(good, t) == "within_tolerance"

    listener = DriftEventListener(orchestrator=None, trust_floor=0.5, recovery_window_s=10.0)
    nominal = EdgeContextReport(operational_state=OperationalState.NOMINAL, trust_score=0.9)
    degrading_low_trust = EdgeContextReport(operational_state=OperationalState.DEGRADING, trust_score=0.2)
    degrading_recovered = EdgeContextReport(operational_state=OperationalState.DEGRADING, trust_score=0.8)

    assert not listener.should_escalate("c1", nominal, now=0.0)
    assert not listener.should_escalate("c1", degrading_low_trust, now=100.0)
    assert not listener.should_escalate("c1", degrading_low_trust, now=105.0)
    assert listener.should_escalate("c1", degrading_low_trust, now=111.0)
    assert not listener.should_escalate("c2", degrading_recovered, now=200.0)

    degrading_high_error = EdgeContextReport(
        operational_state=OperationalState.DEGRADING, trust_score=0.2, error_rate_1m=0.3
    )
    assert listener.should_escalate("c3", degrading_high_error, now=300.0)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "decisions.jsonl")
        orch = PolicyOrchestrator(library_dir=os.path.join(tmp, "lib"), log_path=log_path)
        base = {"accuracy": 0.9, "sla_violation_rate": 0.05, "escalation_rate": 0.3, "trust_score": 0.9}
        for i in range(6):
            orch._log_decision(
                f"c{i}", ScenarioConfig(name="steady"), {"seed": EvalMetrics(**base)},
                PolicyDecision.REUSE, "seed", trigger="onboarding",
            )
        for i in range(6):
            orch._log_decision(
                f"c{i}", ScenarioConfig(name="steady"), {"seed": EvalMetrics(**base)},
                PolicyDecision.FINE_TUNE, "seed", trigger="drift_detected",
            )
        rate = orch._false_reuse_rate()
        assert rate == 1.0
        result = orch.recalibrate_thresholds()
        assert result is not None and result["changed"]
        assert orch.thresholds.max_sla_violation_rate < ToleranceThresholds().max_sla_violation_rate

    print("orchestrator self-check passed")


if __name__ == "__main__":
    demo()
