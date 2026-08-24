
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.diagnosis import (
    DecisionReview,
    Diagnosis,
    DiagnosisContext,
    DiagnosisProvider,
    GovernanceReviewContext,
    ThresholdDiagnosisContext,
)
from devmind.environment import InferenceGatewayEnv, ScenarioConfig
from devmind.evaluation import EvalMetrics, average_metrics, make_env, ppo_policy, run_episode
from devmind.models import EdgeContextReport, OperationalState
from devmind.trainer import PPOTrainer, randomize_scenario


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

TRAINING_STEP_BOUNDS = {
    "train_new_steps": (10_000, 100_000),
    "fine_tune_steps": (1_000, 20_000),
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
        diagnosis_provider: DiagnosisProvider | None = None,
        on_threshold_diagnosis: Callable[[str, Diagnosis], None] | None = None,
        meta_policy_path: str | None = None,
        meta_state_stats_path: str | None = None,
        on_governance_review: Callable[[str, DecisionReview], None] | None = None,
    ):
        self.library_dir = library_dir
        self.thresholds = thresholds or ToleranceThresholds()
        self.client_thresholds: dict[str, ToleranceThresholds] = {}
        self.diagnosis_provider = diagnosis_provider
        self.on_threshold_diagnosis = on_threshold_diagnosis
        self.on_governance_review = on_governance_review
        self.library: dict[str, PolicyRecord] = {}
        self.meta_policy = None
        self.meta_state_mean: np.ndarray | None = None
        self.meta_state_std: np.ndarray | None = None
        if meta_policy_path and os.path.exists(meta_policy_path):
            self.meta_policy = PPONetwork()
            self.meta_policy.load_state_dict(torch.load(meta_policy_path, map_location="cpu", weights_only=True))
            self.meta_policy.eval()
            if meta_state_stats_path and os.path.exists(meta_state_stats_path):
                with open(meta_state_stats_path) as f:
                    stats = json.load(f)
                self.meta_state_mean = np.array(stats["mean"])
                self.meta_state_std = np.array(stats["std"])
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
        self.fallback_ceilings: dict[str, float] = {}

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

    def set_client_thresholds(self, client: str, thresholds: ToleranceThresholds) -> None:
        """Let a company set its own onboarding bar (max_sla_violation_rate,
        max_escalation_rate, min_accuracy) instead of the shared default. Once set,
        recalibrate_thresholds() (which only ever adjusts self.thresholds, the
        default) no longer drifts this client's bar."""
        self.client_thresholds[client] = thresholds

    def _thresholds_for(self, client: str) -> ToleranceThresholds:
        return self.client_thresholds.get(client, self.thresholds)

    def _diagnose_unreachable_threshold(
        self, client: str, scenario: ScenarioConfig, thresholds: ToleranceThresholds, metrics: EvalMetrics
    ) -> None:
        """Governance-time: a policy freshly trained/fine-tuned specifically for this
        scenario still misses the client's own tolerance thresholds. That's evidence
        the requirement itself may be unreachable for this scenario (e.g. RTT alone
        exceeds the SLA budget), not that training needs another attempt. Ask the LLM
        diagnosis provider to explain it, reusing the same off-the-fast-path pattern
        as EscalationDiagnosisMonitor. No-op if no provider is configured."""
        if self.diagnosis_provider is None:
            return
        context = ThresholdDiagnosisContext(
            client_id=client,
            scenario=scenario.name,
            max_sla_violation_rate=thresholds.max_sla_violation_rate,
            max_escalation_rate=thresholds.max_escalation_rate,
            min_accuracy=thresholds.min_accuracy,
            achieved_sla_violation_rate=metrics.sla_violation_rate,
            achieved_escalation_rate=metrics.escalation_rate,
            achieved_accuracy=metrics.accuracy,
        )
        diagnosis = asyncio.run(self.diagnosis_provider.diagnose(context))
        if self.on_threshold_diagnosis is not None:
            self.on_threshold_diagnosis(client, diagnosis)

    def _meta_decide(
        self, candidates: dict[str, EvalMetrics], thresholds: ToleranceThresholds, scenario: ScenarioConfig
    ) -> tuple[PolicyDecision, str | None, bool]:
        """Learned governance decision (orchestrator_trainer.py's meta-policy) in place
        of select_decision()'s threshold rule. Mirrors AgenticOrchestrator.decide()'s
        exact safety pattern (agent.py): entropy > 0.9 or OOD vs. meta_state_stats.json
        -> fall back to the deterministic rule, never a hard failure. Returns
        (decision, chosen_candidate_id, fallback_used)."""
        from devmind.orchestrator_trainer import build_observation

        best_id = (
            max(candidates, key=lambda pid: candidates[pid].accuracy - candidates[pid].sla_violation_rate)
            if candidates else None
        )
        best_metrics = candidates[best_id] if best_id is not None else EvalMetrics(
            accuracy=0.0, sla_violation_rate=1.0, escalation_rate=1.0, fallback_rate=1.0
        )
        obs = build_observation(best_metrics, thresholds, scenario, library_empty=not candidates)

        state = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.meta_policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = int(dist.sample().item())
        entropy = float(dist.entropy().item())

        ood = False
        if self.meta_state_mean is not None:
            std = np.maximum(self.meta_state_std, 0.02)
            z = np.abs((obs - self.meta_state_mean) / std)
            ood = bool((z > 4.0).sum() >= 2)

        decision = [PolicyDecision.REUSE, PolicyDecision.FINE_TUNE, PolicyDecision.TRAIN_NEW][action]
        invalid = decision in (PolicyDecision.REUSE, PolicyDecision.FINE_TUNE) and not candidates
        if entropy > 0.9 or ood or invalid:
            decision, chosen = select_decision(candidates, thresholds)
            return decision, chosen, True

        chosen = best_id if decision != PolicyDecision.TRAIN_NEW else None
        return decision, chosen, False

    _DECISION_ORDER = {PolicyDecision.REUSE: 0, PolicyDecision.FINE_TUNE: 1, PolicyDecision.TRAIN_NEW: 2}

    def _governance_review(
        self,
        client: str,
        scenario: ScenarioConfig,
        thresholds: ToleranceThresholds,
        candidates: dict[str, EvalMetrics],
        decision: PolicyDecision,
        chosen: str | None,
    ) -> tuple[PolicyDecision, str | None]:
        """Option C: an LLM reviews the rule/meta-policy's decision and may only
        escalate it to something MORE cautious (reuse -> fine_tune -> train_new),
        never downgrade -- so a hallucinated review can waste compute at worst,
        never silently authorize an under-provisioned policy. Same off-the-fast-path
        Ollama plumbing as _diagnose_unreachable_threshold. No-op if no
        diagnosis_provider is configured, or if already at TRAIN_NEW (nothing to
        escalate to)."""
        if self.diagnosis_provider is None or decision == PolicyDecision.TRAIN_NEW:
            return decision, chosen
        context = GovernanceReviewContext(
            client_id=client,
            scenario=scenario.name,
            rule_decision=decision.value,
            candidate_metrics={pid: vars(m) for pid, m in candidates.items()},
            max_sla_violation_rate=thresholds.max_sla_violation_rate,
            max_escalation_rate=thresholds.max_escalation_rate,
            min_accuracy=thresholds.min_accuracy,
        )
        review = asyncio.run(self.diagnosis_provider.review_decision(context))
        if self.on_governance_review is not None:
            self.on_governance_review(client, review)
        if not review.parsed_ok:
            return decision, chosen
        try:
            recommended = PolicyDecision(review.recommended_decision)
        except ValueError:
            return decision, chosen
        if self._DECISION_ORDER[recommended] > self._DECISION_ORDER[decision]:
            new_chosen = None if recommended == PolicyDecision.TRAIN_NEW else chosen
            return recommended, new_chosen
        return decision, chosen

    def onboard(
        self, client: str, scenario: ScenarioConfig, max_samples: int = 500, trigger: str = "onboarding"
    ) -> PolicyDecision:
        thresholds = self._thresholds_for(client)
        candidates = {
            pid: self._evaluate(rec.checkpoint_path, scenario, max_samples)
            for pid, rec in self.library.items()
        }
        if self.meta_policy is not None:
            decision, chosen, _meta_fallback = self._meta_decide(candidates, thresholds, scenario)
        else:
            decision, chosen = select_decision(candidates, thresholds)
        decision, chosen = self._governance_review(client, scenario, thresholds, candidates, decision, chosen)
        decision_metrics = candidates.get(chosen)

        if decision == PolicyDecision.FINE_TUNE:
            chosen = self._fine_tune(chosen, scenario)
            post_metrics = self._evaluate(self.library[chosen].checkpoint_path, scenario, max_samples)
            self.calibrate_training_steps(decision, post_metrics)
            if not _meets_tolerance(post_metrics, thresholds):
                self._diagnose_unreachable_threshold(client, scenario, thresholds, post_metrics)
        elif decision == PolicyDecision.TRAIN_NEW:
            chosen = self._train_new(client, scenario)
            post_metrics = self._evaluate(self.library[chosen].checkpoint_path, scenario, max_samples)
            self.calibrate_training_steps(decision, post_metrics)
            if not _meets_tolerance(post_metrics, thresholds):
                self._diagnose_unreachable_threshold(client, scenario, thresholds, post_metrics)

        rec = self.library[chosen]
        if scenario.name not in rec.validated_scenarios:
            rec.validated_scenarios.append(scenario.name)
        if client not in rec.clients_assigned:
            rec.clients_assigned.append(client)

        if decision_metrics is not None:
            self.calibrate_fallback_ceiling(client, decision_metrics)

        self._log_decision(client, scenario, candidates, decision, chosen, trigger, thresholds)
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

    def _train(
        self, scenario: ScenarioConfig, total_steps: int, init_state_dict: dict | None = None, seed: int | None = None
    ) -> PPONetwork:
        edge_model, cloud_model = self._models_for(scenario.task)
        env = InferenceGatewayEnv(scenario, edge_model=edge_model, cloud_model=cloud_model)
        trainer = PPOTrainer(env)
        if init_state_dict is not None:
            trainer.policy.load_state_dict(init_state_dict)
        rng = np.random.default_rng(seed)
        step = 0
        while step < total_steps:
            env.scenario = randomize_scenario(scenario, rng)
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
        thresholds: ToleranceThresholds | None = None,
    ) -> None:
        thresholds = thresholds or self.thresholds
        signal = dominant_signal(candidates[chosen], thresholds) if chosen in candidates else "n/a_new_policy"
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
            "fallback_queue_wait_ceiling": self.fallback_ceilings.get(client),
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

    def calibrate_fallback_ceiling(self, client: str, metrics: EvalMetrics, step: float = 0.05) -> float:
        """Governance-time: tune this client's AgenticOrchestrator.fallback_queue_wait_ceiling
        from its own evaluation history, the same evidence-driven pattern as
        recalibrate_thresholds(). A client that hits the entropy/OOD fallback often
        should have that fallback err toward not escalating into an already-backed-up
        queue (tighter ceiling); a client that rarely hits it can afford a looser one.
        Never called from the per-request fast loop — the live CascadeController reads
        fallback_ceilings[client] when constructing that client's AgenticOrchestrator."""
        baseline = AgenticOrchestrator.FALLBACK_QUEUE_WAIT_CEILING
        current = self.fallback_ceilings.get(client, baseline)
        if metrics.fallback_rate > 0.3:
            new_val = current - step
        elif metrics.fallback_rate < 0.05:
            new_val = current + step
        else:
            new_val = current
        new_val = max(0.1, min(1.0, new_val))
        self.fallback_ceilings[client] = new_val
        return new_val

    def calibrate_training_steps(
        self, decision: PolicyDecision, metrics: EvalMetrics, step_increment: int = 5_000
    ) -> int | None:
        """Governance-time: nudge train_new_steps/fine_tune_steps toward whatever the
        policy just trained actually needed, the same evidence-driven pattern as
        calibrate_fallback_ceiling()/recalibrate_thresholds(). If it meets tolerance,
        next TRAIN_NEW/FINE_TUNE needs less compute; if it doesn't, it needs more.
        Never called from the per-request fast loop."""
        if decision == PolicyDecision.TRAIN_NEW:
            attr = "train_new_steps"
        elif decision == PolicyDecision.FINE_TUNE:
            attr = "fine_tune_steps"
        else:
            return None
        lo, hi = TRAINING_STEP_BOUNDS[attr]
        old = getattr(self, attr)
        met = _meets_tolerance(metrics, self.thresholds)
        new_val = max(lo, min(hi, old + (-step_increment if met else step_increment)))
        setattr(self, attr, new_val)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "training_steps_calibrated",
            "decision": decision.value,
            "attr": attr,
            "old_value": old,
            "new_value": new_val,
            "met_tolerance": met,
        }
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return new_val


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
        self._last_escalated: dict[str, float] = {}
        self._queue: asyncio.Queue[tuple[str, EdgeContextReport, float]] = asyncio.Queue()

    def _prune_stale(self, now: float) -> None:
        stale_before = now - 10 * self.recovery_window_s
        for tracker in (self._distress_since, self._last_escalated):
            stale = [cid for cid, t in tracker.items() if t < stale_before]
            for cid in stale:
                del tracker[cid]

    def should_escalate(self, client_id: str, report: EdgeContextReport, now: float) -> bool:
        self._prune_stale(now)
        state_distressed = report.operational_state in (OperationalState.DEGRADING, OperationalState.UNREACHABLE)
        low_trust = report.trust_score < self.trust_floor
        high_error = report.error_rate > self.error_rate_ceiling
        if not state_distressed or not low_trust:
            self._distress_since.pop(client_id, None)
            return False
        last = self._last_escalated.get(client_id)
        cooled_down = last is None or now - last >= self.recovery_window_s
        if high_error and cooled_down:
            self._distress_since.pop(client_id, None)
            self._last_escalated[client_id] = now
            return True
        first_seen = self._distress_since.setdefault(client_id, now)
        if now - first_seen < self.recovery_window_s:
            return False
        self._distress_since.pop(client_id, None)
        self._last_escalated[client_id] = now
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


class EscalationDiagnosisMonitor:
    """Governance-time: watches per-client action-log rows for sustained near-100%
    escalation and calls a DiagnosisProvider (e.g. local Ollama) to explain why, off
    the request path entirely. Mirrors DriftEventListener's queue + sustained-window +
    cooldown shape. Never called from the per-request fast loop."""

    WINDOW = 50
    ESCALATION_THRESHOLD = 0.95
    COOLDOWN_S = 300.0

    def __init__(
        self,
        diagnosis_provider: DiagnosisProvider,
        on_diagnosis: Callable[[str, Diagnosis], None] | None = None,
        window: int = WINDOW,
        escalation_threshold: float = ESCALATION_THRESHOLD,
        cooldown_s: float = COOLDOWN_S,
    ):
        self.diagnosis_provider = diagnosis_provider
        self.on_diagnosis = on_diagnosis
        self.window = window
        self.escalation_threshold = escalation_threshold
        self.cooldown_s = cooldown_s
        self._rows: dict[str, deque[dict]] = {}
        self._last_diagnosed: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._queue: asyncio.Queue[tuple[str, dict, float]] = asyncio.Queue()

    def _prune_stale(self, now: float) -> None:
        stale_before = now - 10 * self.cooldown_s
        stale = [cid for cid, t in self._last_seen.items() if t < stale_before]
        for cid in stale:
            self._rows.pop(cid, None)
            self._last_diagnosed.pop(cid, None)
            self._last_seen.pop(cid, None)

    def _should_diagnose(self, client_id: str, now: float) -> bool:
        rows = self._rows.get(client_id)
        if rows is None or len(rows) < self.window:
            return False
        escalated = sum(1 for r in rows if r.get("action") == "ESCALATE_TO_CLOUD")
        if escalated / len(rows) < self.escalation_threshold:
            return False
        last = self._last_diagnosed.get(client_id)
        return last is None or now - last >= self.cooldown_s

    def _build_context(self, client_id: str) -> DiagnosisContext:
        rows = list(self._rows[client_id])
        n = len(rows)
        escalated = sum(1 for r in rows if r.get("action") == "ESCALATE_TO_CLOUD")
        violated = sum(1 for r in rows if not r.get("sla_met", True))
        states = [r["operational_state"] for r in rows if r.get("operational_state")]
        dominant_state = Counter(states).most_common(1)[0][0] if states else "UNKNOWN"

        stress_keys = ["cpu", "gpu", "memory", "disk_io", "thermal"]
        stress_rows = [r["resource_stress"] for r in rows if r.get("resource_stress")]
        avg_stress = {
            k: (sum(s.get(k, 0.0) for s in stress_rows) / len(stress_rows)) if stress_rows else 0.0
            for k in stress_keys
        }

        cal_vals = [r["calibration_delta"] for r in rows if r.get("calibration_delta") is not None]
        err_vals = [r["error_rate"] for r in rows if r.get("error_rate") is not None]
        trust_vals = [r["trust_score"] for r in rows if r.get("trust_score") is not None]
        reasons = [r["fallback_reason"] for r in rows if r.get("fallback_reason")]
        dominant_reason = Counter(reasons).most_common(1)[0][0] if reasons else None

        return DiagnosisContext(
            client_id=client_id,
            window_requests=n,
            escalation_rate=escalated / n,
            sla_violation_rate=violated / n,
            dominant_operational_state=dominant_state,
            avg_resource_stress=avg_stress,
            avg_calibration_delta=(sum(cal_vals) / len(cal_vals)) if cal_vals else None,
            avg_error_rate=(sum(err_vals) / len(err_vals)) if err_vals else None,
            dominant_fallback_reason=dominant_reason,
            avg_trust_score=(sum(trust_vals) / len(trust_vals)) if trust_vals else None,
        )

    def notify(self, client_id: str, row: dict) -> None:
        self._queue.put_nowait((client_id, row, time.monotonic()))

    async def run_forever(self) -> None:
        while True:
            client_id, row, now = await self._queue.get()
            self._prune_stale(now)
            self._last_seen[client_id] = now
            buf = self._rows.setdefault(client_id, deque(maxlen=self.window))
            buf.append(row)
            if self._should_diagnose(client_id, now):
                self._last_diagnosed[client_id] = now
                context = self._build_context(client_id)
                # ponytail: awaited inline, so one client's diagnosis call serializes
                # behind another's on this single queue consumer. Ollama can serve
                # concurrent requests fine; upgrade path if this matters is
                # asyncio.create_task(self.diagnosis_provider.diagnose(context)) per
                # client with an in-flight guard (skip if that client already has one
                # running) so distressed clients don't queue behind each other.
                diagnosis = await self.diagnosis_provider.diagnose(context)
                if self.on_diagnosis is not None:
                    self.on_diagnosis(client_id, diagnosis)


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
    meta_policy_path: str | None = None,
    meta_state_stats_path: str | None = None,
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
        meta_policy_path=meta_policy_path,
        meta_state_stats_path=meta_state_stats_path,
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


class _ScriptedDiagnosisProvider:
    def __init__(self, review_recommendation: str | None = None):
        self.review_recommendation = review_recommendation

    async def diagnose(self, context: DiagnosisContext) -> Diagnosis:
        return Diagnosis(
            summary="s", likely_cause="c", resource_recommendation="r",
            model_used="test", latency_ms=1.0, raw_response="{}",
        )

    async def review_decision(self, context: GovernanceReviewContext) -> DecisionReview:
        recommended = self.review_recommendation or context.rule_decision
        return DecisionReview(
            agrees=recommended == context.rule_decision,
            recommended_decision=recommended,
            justification="scripted", model_used="test", latency_ms=1.0, raw_response="{}",
        )


def demo() -> None:
    t = ToleranceThresholds()

    good = EvalMetrics(accuracy=0.9, sla_violation_rate=0.05, escalation_rate=0.3)
    close_miss = EvalMetrics(accuracy=0.7, sla_violation_rate=0.2, escalation_rate=0.5)

    calib_orch = PolicyOrchestrator(library_dir="/tmp/devmind_calib_demo")
    baseline = AgenticOrchestrator.FALLBACK_QUEUE_WAIT_CEILING
    high_fallback = EvalMetrics(fallback_rate=0.5)
    tightened = calib_orch.calibrate_fallback_ceiling("c_noisy", high_fallback)
    assert tightened < baseline, "a client that hits the safety net often should get a tighter queue-wait ceiling"

    low_fallback = EvalMetrics(fallback_rate=0.0)
    loosened = calib_orch.calibrate_fallback_ceiling("c_quiet", low_fallback)
    assert loosened > baseline, "a client that rarely hits the safety net should get a looser ceiling"
    assert calib_orch.fallback_ceilings["c_noisy"] != calib_orch.fallback_ceilings["c_quiet"], (
        "calibration must be per-client, not global"
    )

    step_orch = PolicyOrchestrator(library_dir="/tmp/devmind_step_demo")
    base_train_new = step_orch.train_new_steps
    step_orch.calibrate_training_steps(PolicyDecision.TRAIN_NEW, good)
    assert step_orch.train_new_steps < base_train_new, "meeting tolerance should shrink train_new_steps"

    base_fine_tune = step_orch.fine_tune_steps
    step_orch.calibrate_training_steps(PolicyDecision.FINE_TUNE, close_miss)
    assert step_orch.fine_tune_steps > base_fine_tune, "missing tolerance should grow fine_tune_steps"

    for _ in range(50):
        step_orch.calibrate_training_steps(PolicyDecision.TRAIN_NEW, good)
    lo, _ = TRAINING_STEP_BOUNDS["train_new_steps"]
    assert step_orch.train_new_steps == lo, "must clamp at the lower bound, never spiral to zero"

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
    assert "c1" in listener._last_escalated
    assert not listener.should_escalate("c2", degrading_recovered, now=200.0)

    degrading_high_error = EdgeContextReport(
        operational_state=OperationalState.DEGRADING, trust_score=0.2, error_rate=0.3
    )
    assert listener.should_escalate("c3", degrading_high_error, now=300.0)
    assert not listener.should_escalate("c3", degrading_high_error, now=301.0)
    assert not listener.should_escalate("c3", degrading_high_error, now=309.9)
    assert listener.should_escalate("c3", degrading_high_error, now=310.0)

    listener.should_escalate("c4", nominal, now=310.0 + 10 * listener.recovery_window_s + 1)
    assert "c1" not in listener._last_escalated, "stale trackers must be pruned"

    monitor = EscalationDiagnosisMonitor(
        _ScriptedDiagnosisProvider(), window=5, escalation_threshold=0.8, cooldown_s=100.0
    )
    escalated_row = {
        "action": "ESCALATE_TO_CLOUD", "sla_met": False, "operational_state": "DEGRADING",
        "resource_stress": {"cpu": 0.9, "gpu": 0.0, "memory": 0.3, "disk_io": 0.2, "thermal": 0.7},
        "calibration_delta": 0.4, "error_rate": 0.2, "trust_score": 0.1,
        "fallback_reason": "low_confidence",
    }
    for _ in range(4):
        monitor._rows.setdefault("c1", deque(maxlen=5)).append(escalated_row)
    assert not monitor._should_diagnose("c1", now=0.0), "must not fire before the window is full"

    monitor._rows["c1"].append(escalated_row)
    assert monitor._should_diagnose("c1", now=0.0), "full window at 100% escalation must fire"

    monitor._last_diagnosed["c1"] = 0.0
    assert not monitor._should_diagnose("c1", now=50.0), "must respect cooldown"
    assert monitor._should_diagnose("c1", now=150.0), "must fire again once cooldown elapses"

    healthy_rows: deque = deque(maxlen=5)
    for _ in range(5):
        healthy_rows.append({"action": "ROUTE_TO_EDGE", "sla_met": True})
    monitor._rows["c2"] = healthy_rows
    assert not monitor._should_diagnose("c2", now=0.0), "low escalation rate must not fire"

    context = monitor._build_context("c1")
    assert context.client_id == "c1" and context.window_requests == 5
    assert context.escalation_rate == 1.0 and context.sla_violation_rate == 1.0
    assert context.dominant_operational_state == "DEGRADING"
    assert context.avg_resource_stress["cpu"] == 0.9 and context.avg_resource_stress["thermal"] == 0.7
    assert context.avg_trust_score == 0.1
    assert context.dominant_fallback_reason == "low_confidence"

    healthy_context = monitor._build_context("c2")
    assert healthy_context.dominant_fallback_reason is None, (
        "rows with no fallback_reason recorded must not fabricate one"
    )

    monitor._last_seen = {"c1": 0.0, "c2": 0.0}
    monitor._prune_stale(now=20 * monitor.cooldown_s)
    assert "c1" not in monitor._rows, "stale clients must be pruned"
    assert "c1" not in monitor._last_diagnosed

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

    with tempfile.TemporaryDirectory() as tmp:
        thresh_orch = PolicyOrchestrator(
            library_dir=os.path.join(tmp, "lib"),
            log_path=os.path.join(tmp, "decisions.jsonl"),
            diagnosis_provider=_ScriptedDiagnosisProvider(),
        )
        assert thresh_orch._thresholds_for("unset_client").max_sla_violation_rate == thresh_orch.thresholds.max_sla_violation_rate

        strict = ToleranceThresholds(max_sla_violation_rate=0.01, max_escalation_rate=0.05, min_accuracy=0.99)
        thresh_orch.set_client_thresholds("strict_client", strict)
        assert thresh_orch._thresholds_for("strict_client") is strict
        assert thresh_orch._thresholds_for("other_client").max_sla_violation_rate != strict.max_sla_violation_rate

        received: list[tuple[str, Diagnosis]] = []
        thresh_orch.on_threshold_diagnosis = lambda client, diag: received.append((client, diag))
        unmet = EvalMetrics(accuracy=0.7, sla_violation_rate=0.5, escalation_rate=0.9)
        thresh_orch._diagnose_unreachable_threshold("strict_client", ScenarioConfig(name="degraded"), strict, unmet)
        assert len(received) == 1 and received[0][0] == "strict_client", (
            "a requirement no fresh policy can meet must trigger the LLM diagnosis callback"
        )

        no_provider_orch = PolicyOrchestrator(library_dir=os.path.join(tmp, "lib2"), log_path=os.path.join(tmp, "d2.jsonl"))
        no_provider_orch._diagnose_unreachable_threshold("c", ScenarioConfig(name="degraded"), strict, unmet)

    meta_path = "meta_policy.pt"
    if os.path.exists(meta_path):
        meta_orch = PolicyOrchestrator(
            library_dir=tempfile.mkdtemp(), meta_policy_path=meta_path,
            meta_state_stats_path="meta_state_stats.json" if os.path.exists("meta_state_stats.json") else None,
        )
        assert meta_orch.meta_policy is not None
        empty_decision, empty_chosen, empty_fallback = meta_orch._meta_decide(
            {}, ToleranceThresholds(), ScenarioConfig.steady()
        )
        assert empty_decision == PolicyDecision.TRAIN_NEW, "empty library must never resolve to REUSE/FINE_TUNE"
        assert empty_chosen is None

        no_meta_orch = PolicyOrchestrator(library_dir=tempfile.mkdtemp())
        assert no_meta_orch.meta_policy is None, "no meta_policy_path given -> unchanged deterministic behavior"

    escalate_orch = PolicyOrchestrator(
        library_dir=tempfile.mkdtemp(), diagnosis_provider=_ScriptedDiagnosisProvider(review_recommendation="train_new")
    )
    reviews: list[tuple[str, DecisionReview]] = []
    escalate_orch.on_governance_review = lambda client, review: reviews.append((client, review))
    fake_candidates = {"seed": EvalMetrics(accuracy=0.9, sla_violation_rate=0.05, escalation_rate=0.1)}
    d, c = escalate_orch._governance_review(
        "c1", ScenarioConfig(name="steady"), ToleranceThresholds(), fake_candidates, PolicyDecision.REUSE, "seed"
    )
    assert d == PolicyDecision.TRAIN_NEW and c is None, "LLM must be able to escalate REUSE -> TRAIN_NEW"
    assert len(reviews) == 1 and reviews[0][0] == "c1"

    no_downgrade_orch = PolicyOrchestrator(
        library_dir=tempfile.mkdtemp(), diagnosis_provider=_ScriptedDiagnosisProvider(review_recommendation="reuse")
    )
    d2, c2 = no_downgrade_orch._governance_review(
        "c2", ScenarioConfig(name="steady"), ToleranceThresholds(), fake_candidates, PolicyDecision.FINE_TUNE, "seed"
    )
    assert d2 == PolicyDecision.FINE_TUNE, "LLM recommending something cheaper than the rule must NOT downgrade it"

    skip_orch = PolicyOrchestrator(
        library_dir=tempfile.mkdtemp(), diagnosis_provider=_ScriptedDiagnosisProvider(review_recommendation="reuse")
    )
    d2b, c2b = skip_orch._governance_review(
        "c2b", ScenarioConfig(name="steady"), ToleranceThresholds(), fake_candidates, PolicyDecision.TRAIN_NEW, None
    )
    assert d2b == PolicyDecision.TRAIN_NEW, "already TRAIN_NEW -> nothing to escalate to, review skipped"

    no_review_orch = PolicyOrchestrator(library_dir=tempfile.mkdtemp())
    d3, c3 = no_review_orch._governance_review(
        "c3", ScenarioConfig(name="steady"), ToleranceThresholds(), fake_candidates, PolicyDecision.REUSE, "seed"
    )
    assert d3 == PolicyDecision.REUSE and c3 == "seed", "no diagnosis_provider -> unchanged deterministic decision"

    print("orchestrator self-check passed")


if __name__ == "__main__":
    demo()
