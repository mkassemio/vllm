# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot descriptive pilot results; requires matplotlib in the analysis environment."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    colors = {"stock": "#64748b", "stock_guard": "#2563eb", "orbital": "#059669"}
    names = {"stock": "Stock", "stock_guard": "Guarded", "orbital": "Orbital"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    ax = axes[0]
    x = 0
    ticks, labels = [], []
    for scenario in ("steady", "windows"):
        for mode in ("stock", "stock_guard", "orbital"):
            row = next(
                (
                    r
                    for r in data["scheduler"]
                    if r["scenario"] == scenario and r["mode"] == mode
                ),
                None,
            )
            if row is None:
                continue
            urgent = statistics.mean(row["interactive_completed"])
            onboard = statistics.mean(row["onboard_completed"])
            ax.bar(x, urgent, color=colors[mode], width=0.72)
            ax.bar(
                x, onboard, bottom=urgent, color=colors[mode], alpha=0.35, width=0.72
            )
            ax.plot(
                [x] * len(row["completed"]),
                row["completed"],
                "_",
                color="#111827",
                markersize=10,
            )
            ticks.append(x)
            labels.append(
                names[mode] + "\n" + ("Open" if scenario == "steady" else "Windows")
            )
            x += 1
        x += 0.6
    ax.set_xticks(ticks, labels, fontsize=9)
    ax.set_ylabel("Requests completed before deadline / 80 offered")
    ax.set_ylim(0, 80)
    ax.set_title(
        "Scheduler: interactive service versus onboard throughput", fontsize=11
    )
    ax.text(
        0.02,
        0.97,
        "Solid: interactive · Pale: onboard\nBlack marks: individual runs\n"
        "Guarded: stock with a calibrated window guard",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[1]
    handoff_names = {
        "stock_replay": "Stock application replay",
        "recompute": "Orbital replay",
        "transfer": "Orbital KV transfer",
    }
    handoff_colors = {
        "stock_replay": "#2563eb",
        "recompute": "#d97706",
        "transfer": "#059669",
    }
    for mode in handoff_names:
        rows = sorted(
            [r for r in data["handoff"] if r["mode"] == mode],
            key=lambda r: r["context_tokens"],
        )
        if not rows:
            continue
        contexts = [r["context_tokens"] for r in rows]
        medians = [r["median_gap_ms"] for r in rows]
        errors = [
            [r["median_gap_ms"] - r["gap_range_ms"][0] for r in rows],
            [r["gap_range_ms"][1] - r["median_gap_ms"] for r in rows],
        ]
        ax.errorbar(
            contexts,
            medians,
            yerr=errors,
            marker="o",
            capsize=4,
            color=handoff_colors[mode],
            label=handoff_names[mode],
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([256, 2048, 8192], ["256", "2,048", "8,192"])
    ax.set_xlabel("Original prompt tokens")
    ax.set_ylabel("Gap between delivered tokens at handoff (ms)")
    ax.set_title("Handoff: warm replicas, measured local transport", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        "Orbital serving pilot · Qwen2.5-0.5B · A100 80GB · eager BF16", fontsize=14
    )
    fig.supxlabel(
        "Scheduler: independent trace runs. Handoff: request-repeat ranges, "
        "not confidence intervals. Local transport only.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
