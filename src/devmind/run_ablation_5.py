import datetime
import json
import os
import time

from devmind.environment import ScenarioConfig
from devmind.evaluation import _metrics_to_dict, print_results, run_ablation_5_silver_modes
from devmind.model_clients import BERTLargeCloud, DistilBERTEdge

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "evaluation")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_path = os.path.join(OUTPUT_DIR, f"run5-silver-ablation-{timestamp}.txt")
    json_path = os.path.join(OUTPUT_DIR, f"run5-silver-ablation-{timestamp}.json")

    print("Loading models (one-time)...")
    t0 = time.perf_counter()
    edge_model = DistilBERTEdge()
    cloud_model = BERTLargeCloud()
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s")

    text_buffer: list[str] = []
    json_buffer: list[dict] = []
    max_samples = 1000

    for scenario in [ScenarioConfig.steady(), ScenarioConfig.bursty(), ScenarioConfig.degraded_network()]:
        t0 = time.perf_counter()
        results = run_ablation_5_silver_modes(
            scenario, max_samples=max_samples, edge_model=edge_model, cloud_model=cloud_model
        )
        print(f"{scenario.name} done in {time.perf_counter() - t0:.1f}s")
        print_results(results, title=f"{scenario.name.upper()} Run 5 (Silver modes)")
        text_buffer.append(f"\n=== {scenario.name.upper()} Run 5 (Silver modes) ===")
        text_buffer.append(f"{'Policy':<26} {'Acc':<8} {'P95 Lat':<10} {'SLA Viol':<10} {'Esc%':<8} {'Energy':<8} {'Fallback':<8}")
        text_buffer.append("-" * 90)
        for name, m in results.items():
            text_buffer.append(
                f"{name:<26} {m.accuracy:<8.3f} {m.p95_latency:<10.1f} "
                f"{m.sla_violation_rate:<10.3f} {m.escalation_rate:<8.3f} "
                f"{m.energy_mj:<8.1f} {m.fallback_rate:<8.3f}"
            )
            entry = _metrics_to_dict(name, m)
            entry["scenario"] = scenario.name
            json_buffer.append(entry)

    with open(txt_path, "w") as f:
        f.write("\n".join(text_buffer))
    with open(json_path, "w") as f:
        json.dump(json_buffer, f, indent=2)
    print(f"\nRun 5 results saved:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
