from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.environment import InferenceGatewayEnv, ScenarioConfig
from devmind.model_clients import BERTLargeCloud, DistilBERTEdge
from devmind.models import Action, GoldStateVector


@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    p95_latency: float = 0.0
    sla_violation_rate: float = 0.0
    escalation_rate: float = 0.0
    energy_mj: float = 0.0
    fallback_rate: float = 0.0
    sla_margin_ms: float = 0.0
    trust_score: float = 1.0


def run_episode(
    env: InferenceGatewayEnv,
    policy_fn: Callable[[GoldStateVector], tuple[int, bool]],
    desc: str | None = None,
) -> EvalMetrics:
    state, _ = env.reset()
    done = False
    accuracies: list[float] = []
    latencies: list[float] = []
    sla_margins: list[float] = []
    trust_scores: list[float] = []
    violations = 0
    escalations = 0
    total = 0
    fallbacks = 0
    energy = 0.0

    total_steps = len(env._dataset.test) if hasattr(env, "_dataset") else None
    pbar = tqdm(total=total_steps, desc=desc, leave=False)
    while not done:
        gold = GoldStateVector(slots=state.copy(), mask=np.ones(13))
        action, fallback = policy_fn(gold)
        state, reward, done, _, info = env.step(action)
        total += 1
        pbar.update(1)
        if fallback:
            fallbacks += 1
        if hasattr(env, "_state") and hasattr(env._state, "_last_latency"):
            latencies.append(env._state._last_latency)
            accuracies.append(env._state._last_accuracy)
            if not env._state._last_sla_met:
                violations += 1
        if action == 1:
            escalations += 1
        if hasattr(env, "_edge_device") and env._edge_device.last_report is not None:
            report = env._edge_device.last_report
            sla_margins.append(report.sla_margin_ms)
            trust_scores.append(report.trust_score)
    pbar.close()

    latencies = np.array(latencies)
    return EvalMetrics(
        accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        p95_latency=float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0,
        sla_violation_rate=violations / max(total, 1),
        escalation_rate=escalations / max(total, 1),
        energy_mj=env._state.energy_mj if hasattr(env._state, "energy_mj") else 0.0,
        fallback_rate=fallbacks / max(total, 1),
        sla_margin_ms=float(np.mean(sla_margins)) if sla_margins else 0.0,
        trust_score=float(np.mean(trust_scores)) if trust_scores else 1.0,
    )


def always_edge(gold: GoldStateVector) -> tuple[int, bool]:
    return 0, False


def always_cloud(gold: GoldStateVector) -> tuple[int, bool]:
    return 1, False


def static_threshold(tau: float) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        return (1 if gold[0] < tau else 0), False
    return policy


def random_policy(gold: GoldStateVector) -> tuple[int, bool]:
    return int(np.random.default_rng().integers(0, 2)), False


def branchynet_entropy(tau: float = 0.6) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        p = min(max(gold[0], 1e-6), 1.0 - 1e-6)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return (1 if entropy >= tau else 0), False
    return policy


def msdnet_budget(tau: float = 0.6, budget_ms: float = 200.0) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        predicted_rtt_ms = gold[2] * 1000.0
        can_afford = predicted_rtt_ms <= budget_ms
        return (1 if (gold[0] < tau and can_afford) else 0), False
    return policy


def edgebert_dvfs(tau: float = 0.6, cpu_limit: float = 0.7) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        escalate = gold[0] < tau and gold[4] < cpu_limit
        return (1 if escalate else 0), False
    return policy


def spinn_connectivity(tau: float = 0.6, rtt_limit_ms: float = 500.0) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        predicted_rtt_ms = gold[2] * 1000.0
        connected = predicted_rtt_ms <= rtt_limit_ms
        return (1 if (gold[0] < tau and connected) else 0), False
    return policy


def proteus_accuracy_floor(target_acc: float = 0.85) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        return (1 if gold[0] < target_acc else 0), False
    return policy


def splitwise_queue_energy(
    queue_wait_limit_s: float = 0.3, energy_stress_limit: float = 0.6
) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        queue_ok = gold[1] <= queue_wait_limit_s
        resource_stress_avg = float(np.mean([gold[5], gold[6], gold[7], gold[8], gold[9]]))
        energy_ok = resource_stress_avg <= energy_stress_limit
        return (1 if (queue_ok and energy_ok) else 0), False
    return policy


def _fit_routellm_learned_router(
    scenario: ScenarioConfig, max_samples: int = 300, edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None
) -> Callable[[GoldStateVector], tuple[int, bool]]:
    from sklearn.linear_model import LogisticRegression

    env = make_env(scenario, max_samples, edge_model, cloud_model)
    state, _ = env.reset()
    X, y = [], []
    done = False
    while not done:
        confidence = float(state[0])
        state, _, done, _, _ = env.step(0)
        X.append([confidence])
        y.append(1 if env._state._last_accuracy >= 0.5 else 0)

    model = LogisticRegression()
    model.fit(np.array(X), np.array(y))

    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        proba_correct = model.predict_proba([[gold[0]]])[0, 1]
        return (1 if proba_correct < 0.5 else 0), False
    return policy


def _fit_nsga2_pareto_rule(
    scenario: ScenarioConfig,
    max_samples: int = 200,
    edge_model: DistilBERTEdge | None = None,
    cloud_model: BERTLargeCloud | None = None,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> Callable[[GoldStateVector], tuple[int, bool]]:
    taus = np.linspace(0.1, 0.95, 18)
    best_tau, best_score = 0.6, -1e9
    for tau in taus:
        env = make_env(scenario, max_samples, edge_model, cloud_model)
        m = run_episode(env, static_threshold(float(tau)), desc=None)
        score = (
            weights[0] * m.accuracy
            - weights[1] * m.sla_violation_rate
            - weights[2] * (m.energy_mj / max_samples)
        )
        if score > best_score:
            best_score, best_tau = score, float(tau)
    return static_threshold(best_tau)


def evaluate_literature_baselines(
    scenario: ScenarioConfig, max_samples: int = 1000, edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None
) -> dict[str, EvalMetrics]:
    literature_baselines = {
        "branchynet_entropy": branchynet_entropy(),
        "msdnet_budget": msdnet_budget(),
        "edgebert_dvfs": edgebert_dvfs(),
        "spinn_connectivity": spinn_connectivity(),
        "proteus_accuracy_floor": proteus_accuracy_floor(),
        "splitwise_queue_energy": splitwise_queue_energy(),
        "routellm_learned_router": _fit_routellm_learned_router(scenario, edge_model=edge_model, cloud_model=cloud_model),
        "nsga2_pareto_rule": _fit_nsga2_pareto_rule(scenario, edge_model=edge_model, cloud_model=cloud_model),
    }
    results: dict[str, EvalMetrics] = {}
    for name, policy_fn in literature_baselines.items():
        results[name] = run_episode(
            make_env(scenario, max_samples, edge_model, cloud_model),
            policy_fn,
            desc=f"{scenario.name}/{name}",
        )
    return results


def ppo_policy(agent: AgenticOrchestrator) -> Callable[[GoldStateVector], tuple[int, bool]]:
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        return agent.decide(gold)
    return policy


def make_env(scenario: ScenarioConfig, max_samples: int = 1000, edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None) -> InferenceGatewayEnv:
    return InferenceGatewayEnv(scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model)


def average_metrics(metrics_list: list[EvalMetrics]) -> EvalMetrics:
    return EvalMetrics(
        accuracy=float(np.mean([m.accuracy for m in metrics_list])),
        p95_latency=float(np.mean([m.p95_latency for m in metrics_list])),
        sla_violation_rate=float(np.mean([m.sla_violation_rate for m in metrics_list])),
        escalation_rate=float(np.mean([m.escalation_rate for m in metrics_list])),
        energy_mj=float(np.mean([m.energy_mj for m in metrics_list])),
        fallback_rate=float(np.mean([m.fallback_rate for m in metrics_list])),
        sla_margin_ms=float(np.mean([m.sla_margin_ms for m in metrics_list])),
        trust_score=float(np.mean([m.trust_score for m in metrics_list])),
    )


def evaluate_baselines(scenario: ScenarioConfig, n_runs: int = 1, max_samples: int = 1000, edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None) -> dict[str, EvalMetrics]:
    baselines = {
        "always_edge": always_edge,
        "always_cloud": always_cloud,
        "static_tau_0.6": static_threshold(0.6),
        "static_tau_0.95": static_threshold(0.95),
        "random": random_policy,
    }
    results: dict[str, EvalMetrics] = {}
    for name, policy_fn in baselines.items():
        metrics_list = [
            run_episode(
                make_env(scenario, max_samples, edge_model, cloud_model),
                policy_fn,
                desc=f"{scenario.name}/{name}",
            )
            for _ in range(n_runs)
        ]
        results[name] = average_metrics(metrics_list)
    return results


def run_ablation(scenario: ScenarioConfig, policy_path: str = "ppo_policy.pt", max_samples: int = 1000, edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None) -> dict[str, EvalMetrics]:
    import torch
    results: dict[str, EvalMetrics] = {}

    def load_agent() -> AgenticOrchestrator:
        ppo = PPONetwork()
        ppo.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
        return AgenticOrchestrator(ppo)

    env = make_env(scenario, max_samples, edge_model, cloud_model)
    results["run1_full"] = run_episode(env, ppo_policy(load_agent()), desc=f"{scenario.name}/run1_full")

    env2 = make_env(scenario, max_samples, edge_model, cloud_model)
    def confidence_only(gold: GoldStateVector) -> tuple[int, bool]:
        return (1 if gold[0] < 0.6 else 0), False
    results["run2_confidence_only"] = run_episode(env2, confidence_only, desc=f"{scenario.name}/run2_confidence_only")

    env3 = make_env(scenario, max_samples, edge_model, cloud_model)
    def confidence_plus_calibration(gold: GoldStateVector) -> tuple[int, bool]:
        calibration_drift = gold[10] if gold.mask[10] else 0.0
        if calibration_drift > 0.15:
            return 1, False
        return (1 if gold[0] < 0.6 else 0), False
    results["run3_calibration_delta"] = run_episode(
        env3, confidence_plus_calibration, desc=f"{scenario.name}/run3_calibration_delta"
    )

    env4 = make_env(scenario, max_samples, edge_model, cloud_model)
    env4.use_reflect = False
    results["run4_no_reflect"] = run_episode(env4, ppo_policy(load_agent()), desc=f"{scenario.name}/run4_no_reflect")

    return results


def run_holdout_ablation(max_samples: int = 1000, policy_path: str = "ppo_policy.pt", edge_model: DistilBERTEdge | None = None, cloud_model: BERTLargeCloud | None = None) -> EvalMetrics:
    import torch
    env = make_env(ScenarioConfig.held_out(), max_samples, edge_model, cloud_model)
    ppo = PPONetwork()
    ppo.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    agent = AgenticOrchestrator(ppo)
    return run_episode(env, ppo_policy(agent), desc="held_out/run6")


def agent_driven_silver_policy(
    agent: AgenticOrchestrator, env: InferenceGatewayEnv
) -> Callable[[GoldStateVector], tuple[int, bool]]:
    # Mirrors AgenticOrchestrator.decide()'s own logic (same thresholds, same
    # fallback guards) but, unlike decide(), can actually act on
    # QUERY_EXTENDED_CONTEXT: it pays the perception cost by asking the env for a
    # freshly-revealed Silver pass over this request's Bronze snapshot, rather
    # than reading a slot that was already zeroed or already revealed upstream by
    # a fixed operational_state rule. This is eval-only scaffolding for Ablation
    # Run 5 — agent.py's decide() stays untouched (single network call, as
    # documented in Section 5) since that's the production code path.
    def policy(gold: GoldStateVector) -> tuple[int, bool]:
        action, entropy = agent.reason(gold)
        fallback = entropy > agent.ENTROPY_FALLBACK_THRESHOLD or agent.is_ood(gold)
        if fallback:
            return agent._resolve_fallback(gold), True
        agent.last_fallback_reason = None
        if action == Action.QUERY_EXTENDED_CONTEXT:
            revealed = env.peek_extended_silver()
            calibration_delta, error_rate = revealed[10], revealed[11]
            high_risk = (
                calibration_delta > agent.QUERY_CALIBRATION_RISK_THRESHOLD
                or error_rate > agent.QUERY_ERROR_RATE_RISK_THRESHOLD
            )
            return (Action.ESCALATE_TO_CLOUD if high_risk else Action.ROUTE_TO_EDGE), False
        return Action(action), False

    return policy


def run_ablation_5_silver_modes(
    scenario: ScenarioConfig,
    policy_path: str = "ppo_policy.pt",
    max_samples: int = 1000,
    edge_model: DistilBERTEdge | None = None,
    cloud_model: BERTLargeCloud | None = None,
) -> dict[str, EvalMetrics]:
    """Ablation Run 5: static vs conditional (current) vs agent-driven Silver
    enrichment, isolating the dynamic Medallion pipeline's contribution the same
    way Run 4 isolates the bidirectional loop's. Same frozen policy as run1_full
    in all three conditions; only what Silver reveals to it changes."""
    import torch

    def load_agent() -> AgenticOrchestrator:
        ppo = PPONetwork()
        ppo.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
        return AgenticOrchestrator(ppo)

    results: dict[str, EvalMetrics] = {}

    env_static = InferenceGatewayEnv(
        scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model, silver_mode="static"
    )
    results["run5a_static_silver"] = run_episode(
        env_static, ppo_policy(load_agent()), desc=f"{scenario.name}/run5a_static_silver"
    )

    env_conditional = InferenceGatewayEnv(
        scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model, silver_mode="conditional"
    )
    results["run5b_conditional_silver"] = run_episode(
        env_conditional, ppo_policy(load_agent()), desc=f"{scenario.name}/run5b_conditional_silver"
    )

    env_agent_driven = InferenceGatewayEnv(
        scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model, silver_mode="masked_off"
    )
    agent_for_5c = load_agent()
    results["run5c_agent_driven_silver"] = run_episode(
        env_agent_driven, agent_driven_silver_policy(agent_for_5c, env_agent_driven), desc=f"{scenario.name}/run5c_agent_driven_silver"
    )

    return results


def print_results(results: dict[str, EvalMetrics], title: str = "Results") -> None:
    print(f"\n=== {title} ===")
    print(f"{'Policy':<20} {'Acc':<8} {'P95 Lat':<10} {'SLA Viol':<10} {'Esc%':<8} {'Energy':<8} {'Fallback':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<20} {m.accuracy:<8.3f} {m.p95_latency:<10.1f} "
            f"{m.sla_violation_rate:<10.3f} {m.escalation_rate:<8.3f} "
            f"{m.energy_mj:<8.1f} {m.fallback_rate:<8.3f}"
        )


_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation")


def _metrics_to_dict(name: str, m: EvalMetrics) -> dict:
    return {
        "policy": name,
        "accuracy": m.accuracy,
        "p95_latency_ms": m.p95_latency,
        "sla_violation_rate": m.sla_violation_rate,
        "escalation_rate": m.escalation_rate,
        "energy_mj": m.energy_mj,
        "fallback_rate": m.fallback_rate,
    }


def save_results(results: dict[str, EvalMetrics], title: str, text_buffer: list[str], json_buffer: list[dict]) -> None:
    text_buffer.append(f"\n=== {title} ===")
    text_buffer.append(f"{'Policy':<20} {'Acc':<8} {'P95 Lat':<10} {'SLA Viol':<10} {'Esc%':<8} {'Energy':<8} {'Fallback':<8}")
    text_buffer.append("-" * 80)
    for name, m in results.items():
        text_buffer.append(
            f"{name:<20} {m.accuracy:<8.3f} {m.p95_latency:<10.1f} "
            f"{m.sla_violation_rate:<10.3f} {m.escalation_rate:<8.3f} "
            f"{m.energy_mj:<8.1f} {m.fallback_rate:<8.3f}"
        )
        json_buffer.append(_metrics_to_dict(name, m))


def main() -> None:
    import time
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_path = os.path.join(_OUTPUT_DIR, f"{timestamp}.txt")
    json_path = os.path.join(_OUTPUT_DIR, f"{timestamp}.json")

    print("Loading models (one-time)...")
    t0 = time.perf_counter()
    edge_model = DistilBERTEdge()
    cloud_model = BERTLargeCloud()
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    text_buffer: list[str] = []
    json_buffer: list[dict] = []

    max_samples = 1000
    print(f"Running evaluation (max_samples={max_samples}, 1 run per baseline)...")
    for scenario in [ScenarioConfig.steady(), ScenarioConfig.bursty(), ScenarioConfig.degraded_network()]:
        results = evaluate_baselines(scenario, n_runs=1, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model)
        print_results(results, title=scenario.name.upper())
        save_results(results, f"{scenario.name.upper()} Baselines", text_buffer, json_buffer)

        literature = evaluate_literature_baselines(scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model)
        print_results(literature, title=f"{scenario.name.upper()} Literature Baselines")
        save_results(literature, f"{scenario.name.upper()} Literature Baselines", text_buffer, json_buffer)

        ablation = run_ablation(scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model)
        print_results(ablation, title=f"{scenario.name.upper()} Ablation")
        save_results(ablation, f"{scenario.name.upper()} Ablation", text_buffer, json_buffer)

    holdout = {"run6_held_out": run_holdout_ablation(max_samples, edge_model=edge_model, cloud_model=cloud_model)}
    print_results(holdout, title="HELD-OUT SCENARIO (Run 6)")
    save_results(holdout, "HELD-OUT SCENARIO (Run 6)", text_buffer, json_buffer)

    with open(txt_path, "w") as f:
        f.write("\n".join(text_buffer))
    with open(json_path, "w") as f:
        json.dump(json_buffer, f, indent=2)
    print(f"\nResults saved:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
