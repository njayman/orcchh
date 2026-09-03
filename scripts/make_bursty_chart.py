"""Regenerate media/bursty-results-chart.pdf from a named evaluation run.

Reads the BURSTY rows straight out of the run's .txt summary so the figure
can never drift from the table it sits next to.

    python scripts/make_bursty_chart.py evaluation/2026-09-02_00-47-07.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POLICIES = [
    ("always_edge", "always_edge"),
    ("always_cloud", "always_cloud"),
    ("static_tau_0.6", "static_tau_0.6"),
    ("static_tau_0.95", "static_tau_0.95"),
    ("routellm_learned_router", "routellm_learned\n_router"),
    ("run1_full", "run1_full\n(proposed)"),
]
SLA_COLOUR, ESC_COLOUR = "#d62728", "#1f77b4"


def read_bursty(path: Path) -> dict[str, tuple[float, float]]:
    """Return {policy: (sla_violation_rate, escalation_rate)} for BURSTY blocks."""
    rows: dict[str, tuple[float, float]] = {}
    in_bursty = False
    for line in path.read_text().splitlines():
        if line.startswith("==="):
            in_bursty = "BURSTY" in line
            continue
        parts = line.split()
        if in_bursty and len(parts) == 7:
            rows[parts[0]] = (float(parts[3]), float(parts[4]))
    return rows


def main(src: Path, dest: Path) -> None:
    rows = read_bursty(src)
    missing = [k for k, _ in POLICIES if k not in rows]
    if missing:
        raise SystemExit(f"{src}: no BURSTY row for {missing}")

    sla = [rows[k][0] for k, _ in POLICIES]
    esc = [rows[k][1] for k, _ in POLICIES]
    x = np.arange(len(POLICIES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = [
        ax.bar(x - width / 2, sla, width, label="SLA violation rate", color=SLA_COLOUR),
        ax.bar(x + width / 2, esc, width, label="Escalation rate", color=ESC_COLOUR),
    ]
    for group in bars:
        for bar in group:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )

    ax.set_title("Bursty traffic: SLA violation rate and escalation rate per policy")
    ax.set_ylabel("Rate")
    ax.set_xticks(x, [label for _, label in POLICIES], fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest, format="pdf", bbox_inches="tight")
    print(f"wrote {dest} from {src}")
    for (key, _), s, e in zip(POLICIES, sla, esc):
        print(f"  {key:24s} sla={s:.3f} esc={e:.3f}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "evaluation/2026-09-02_00-47-07.txt"
    output = root / "msc-project-report_latex/media/bursty-results-chart.pdf"
    main(source, output)
