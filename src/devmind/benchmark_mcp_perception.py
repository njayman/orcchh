"""Benchmarks the MCP-pattern skill interface's per-request perception cost
(Bronze snapshot -> Silver enrich -> Gold normalize -> AgenticOrchestrator.decide(),
which exercises MCPSkillInterface.list_tools()/call() internally) against real
DistilBERT edge-inference latency, using the exact same component wiring as the
live gateway's CascadeController (gateway/app.py). Closes the gap flagged in the
report's Conclusion: this cost was claimed to fit "well within the per-request
perception budget" but was never separately measured.
"""
import statistics
import time

import torch

from devmind.agent import AgenticOrchestrator, PPONetwork
from devmind.edge import EdgeDevice
from devmind.dataset import load_task_dataset
from devmind.medallion import DynamicMetricRegistry, GoldNormalizer, MetricSource, SilverEnricher
from devmind.model_clients import DistilBERTEdge


def main(n_requests: int = 300) -> None:
    edge_model = DistilBERTEdge()
    edge = EdgeDevice()

    registry = DynamicMetricRegistry()
    registry.register(MetricSource("edge_context", "EdgeContextReport", lambda: edge.last_report or edge.emit_report(0.5)))
    registry.register(MetricSource("cloud_queue_depth", "int", lambda: 3))
    registry.register(MetricSource("rtt_ms", "float", lambda: 50.0))
    registry.register(MetricSource("energy_mj", "float", lambda: 0.0))
    registry.register(MetricSource("traffic_intensity", "float", lambda: 0.0))

    silver = SilverEnricher()
    gold = GoldNormalizer()
    ppo = PPONetwork()
    ppo.load_state_dict(torch.load("ppo_policy.pt", map_location="cpu", weights_only=True))
    agent = AgenticOrchestrator(ppo, state_stats_path="state_stats.json")

    dataset = load_task_dataset("toxicity", max_samples=n_requests + 50)
    samples = dataset.test[:n_requests] if len(dataset.test) >= n_requests else (dataset.test * n_requests)[:n_requests]

    edge_latencies_ms: list[float] = []
    perception_latencies_ms: list[float] = []

    for sample in samples:
        t0 = time.perf_counter()
        result = edge_model.predict(sample.text, sample.label)
        t1 = time.perf_counter()
        edge_latencies_ms.append((t1 - t0) * 1000.0)

        report = edge.emit_report(result.confidence, result.is_correct)

        t2 = time.perf_counter()
        bronze = registry.snapshot()
        bronze.sla_budget_ms = 300.0
        bronze.sla_remaining_ms = 300.0
        silver_features = silver.enrich(bronze)
        gold_state = gold.normalize(silver_features)
        agent.decide(gold_state)
        t3 = time.perf_counter()
        perception_latencies_ms.append((t3 - t2) * 1000.0)

    def summarize(name: str, values: list[float]) -> None:
        values_sorted = sorted(values)
        p50 = statistics.median(values_sorted)
        p95 = values_sorted[int(0.95 * len(values_sorted)) - 1]
        print(f"{name}: mean={statistics.mean(values):.3f}ms p50={p50:.3f}ms p95={p95:.3f}ms max={max(values):.3f}ms")

    print(f"n_requests={n_requests}")
    summarize("Edge inference (real DistilBERT forward pass)", edge_latencies_ms)
    summarize("Perception (Bronze->Silver->Gold->decide(), incl. MCPSkillInterface)", perception_latencies_ms)
    ratio = statistics.mean(perception_latencies_ms) / statistics.mean(edge_latencies_ms) * 100
    print(f"Perception cost is {ratio:.2f}% of edge inference latency on average")


if __name__ == "__main__":
    main()
