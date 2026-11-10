# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot scheduler outcomes and measured prefill adaptation; requires matplotlib."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_xeoverse_scheduler import LABELS, read_events

COLORS = {
    "stock128": "#64748b",
    "stock512": "#2563eb",
    "orbital": "#059669",
    "orbital_flat": "#b45309",
}
NAMES = {
    "stock128": "Stock\n128",
    "stock512": "Stock\n512",
    "orbital": "Orbital\ndynamic",
    "orbital_flat": "Orbital\nflat",
}
SCENARIOS = ("sunlit", "ingress", "egress")
TITLES = {
    "sunlit": "Sunlit control",
    "ingress": "Eclipse entry",
    "egress": "Eclipse exit",
}


def outcomes(data, output):
    counts = {g["independent_launches"] for g in data["aggregates"]}
    repetition = (
        f"{next(iter(counts))} independent launches per case"
        if len(counts) == 1
        else "per-case launch counts in summary"
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for column, scenario in enumerate(SCENARIOS):
        for index, label in enumerate(LABELS):
            group = next(
                (
                    g
                    for g in data["aggregates"]
                    if g["scenario"] == scenario and g["label"] == label
                ),
                None,
            )
            if group is None:
                continue
            classes = group["by_class"]
            urgent = statistics.mean(classes["interactive"]["completed_each"])
            onboard = statistics.mean(classes["onboard"]["completed_each"])
            ax = axes[0, column]
            ax.bar(index, urgent, color=COLORS[label], width=0.65)
            ax.bar(
                index,
                onboard,
                bottom=urgent,
                color=COLORS[label],
                alpha=0.3,
                width=0.65,
            )
            ax.plot(
                [index] * len(group["completed_each"]),
                group["completed_each"],
                "_",
                color="#111827",
                markersize=10,
            )
            ax = axes[1, column]
            values = [
                r["metrics"]["by_class"]["interactive"]["ttft_p50_s"] * 1000
                for r in data["runs"]
                if r["scenario"] == scenario
                and r["label"] == label
                and r["metrics"]["by_class"]["interactive"]["ttft_p50_s"] is not None
            ]
            ax.scatter(
                [index] * len(values), values, color=COLORS[label], alpha=0.7, s=32
            )
            if values:
                ax.plot(
                    index,
                    statistics.median(values),
                    "_",
                    color="#111827",
                    markersize=16,
                )
        axes[0, column].set_title(TITLES[scenario])
        axes[0, column].set_ylim(0, 180)
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylim(20, 2000)
        axes[1, column].set_yticks([30, 100, 300, 1000], ["30", "100", "300", "1,000"])
        for row in range(2):
            axes[row, column].set_xticks(
                range(4), [NAMES[m] for m in LABELS], fontsize=9
            )
            axes[row, column].grid(axis="y", alpha=0.2)
            axes[row, column].set_axisbelow(True)
    axes[0, 0].set_ylabel("On-time requests / 180 offered per run")
    axes[1, 0].set_ylabel("Interactive median TTFT (ms, log scale)")
    fig.suptitle(
        f"XeoVerse scheduler replay · A100 · Qwen2.5-0.5B · {repetition}",
        fontsize=14,
    )
    fig.supxlabel(
        "Top: solid = interactive; pale = onboard; marks = run totals. "
        "Bottom: dots = run medians, black = median of runs.\n"
        "TTFT excludes missing first tokens; all misses remain in completion counts. "
        "Assumed compute budgets; real GPU execution.",
        fontsize=9,
    )
    fig.savefig(output / "outcomes.png", dpi=180)
    plt.close(fig)


def adaptation(root, output):
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), sharex=True, sharey=True, layout="constrained"
    )
    for ax, label in zip(axes, LABELS):
        directory = root / f"ingress-0-{label}"
        trace = json.loads((directory / "trace.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        times, tokens = [], []
        for event in read_events(directory):
            count = sum(
                w["tokens"] for w in event["scheduled"] if w["kind"] == "prefill"
            )
            if event["model_executed"] and count:
                times.append(event["start_mono"] - result["monotonic_start"])
                tokens.append(count)
        for limit, color in zip(
            trace["step_limits"], ("#e0f2fe", "#fef3c7", "#fee2e2")
        ):
            left, right = limit["start_s"], limit["end_s"]
            ax.axvspan(left, right, color=color, alpha=0.5)
            ax.text(
                (left + right) / 2,
                540,
                f"{limit['max_step_s'] * 1000:g} ms allowance",
                ha="center",
                fontsize=9,
            )
        ax.axvspan(33, 45, color="#e5e7eb")
        ax.text(
            39,
            260,
            "Eclipse\npaused",
            ha="center",
            va="center",
            color="#4b5563",
            fontsize=10,
        )
        ax.scatter(times, tokens, s=6, alpha=0.55, color=COLORS[label])
        ax.set_ylabel(
            NAMES[label].replace("\n", " ") + "\nprefill tokens / step", fontsize=10
        )
        ax.set_ylim(0, 590)
        ax.set_yticks([0, 128, 256, 512])
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Real elapsed seconds in the eclipse-entry segment")
    axes[-1].set_xlim(0, 45)
    fig.suptitle(
        "Measured prefill adaptation · eclipse entry · arrival seed 0", fontsize=14
    )
    fig.supxlabel(
        "Each point is actual prefill work in one executed step. "
        "Orbital-flat retains priority/deadline/window policy "
        "with a permissive step allowance.\n"
        "Budget tiers are mission assumptions; XeoVerse supplies the eclipse geometry. "
        "Recomputed prefill is included.",
        fontsize=9,
    )
    fig.savefig(output / "adaptation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.summary.read_text())
    if not data["complete"]:
        raise ValueError("Final plots require a complete paired campaign")
    outcomes(data, args.output)
    adaptation(args.root, args.output)


if __name__ == "__main__":
    main()
