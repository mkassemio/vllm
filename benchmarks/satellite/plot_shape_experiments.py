# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot verified SC/LC x SO/LO latencies and actual scheduler adaptation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_xeoverse_scheduler import read_events
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from shape_utils import MODES, SCENARIOS, SHAPES

COLORS = {"stock": "#2563eb", "orbital": "#059669", "orbital_flat": "#c07812"}
NAMES = {
    "stock": "Stock vLLM",
    "orbital": "Orbital dynamic",
    "orbital_flat": "Orbital permissive",
}
TITLES = {
    "sunlit": "Sunlit: 100 ms allowance",
    "recovery": "After eclipse: 30 → 50 → 100 ms",
}


def latencies(data, output, load):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    for column, scenario in enumerate(SCENARIOS):
        for row, (metric, label, multiplier) in enumerate(
            (
                ("ttft_s", "Time to first token (s, log scale)", 1),
                ("tpot_s", "Time per output token (ms)", 1000),
                ("completion_latency_s", "Completion latency (s)", 1),
            )
        ):
            ax = axes[row, column]
            for mi, mode in enumerate(MODES):
                for si, shape in enumerate(SHAPES):
                    group = next(
                        g
                        for g in data["aggregates"]
                        if (
                            g["scenario"] == scenario
                            and g["load"] == load
                            and g["shape"] == shape
                            and g["mode"] == mode
                        )
                    )
                    values = [s[metric] * multiplier for s in group["launch_medians"]]
                    middle = group["metrics"][metric]["median"] * multiplier
                    x = si + (mi - 1) * 0.23
                    ax.plot(
                        [x, x],
                        [min(values), max(values)],
                        color=COLORS[mode],
                        alpha=0.6,
                    )
                    ax.scatter(
                        [x] * len(values), values, s=23, color=COLORS[mode], alpha=0.5
                    )
                    ax.plot(
                        x,
                        middle,
                        "_",
                        markersize=14,
                        markeredgewidth=2,
                        color=COLORS[mode],
                        label=NAMES[mode] if column == row == si == 0 else None,
                    )
            ax.set_xticks(range(4), SHAPES)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_title(TITLES[scenario], fontsize=12)
            else:
                ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.08)
    legend = fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
    )
    legend.set_in_layout(False)
    title = (
        "One request at a time"
        if load == "isolated"
        else "Mixed batch: two requests of every shape"
    )
    heading = fig.suptitle(f"{title} · Qwen2.5-0.5B · A100 80GB", fontsize=15, y=0.995)
    heading.set_in_layout(False)
    fig.supxlabel(
        "SC = 1,000 input tokens; LC = 16,500. SO = 256 output tokens; LO = 2,000.\n"
        "Dots: three engine-launch measurements "
        "(within-launch shape medians for mixed batches). "
        "Bars: median; lines: observed range.\n"
        "Complete outputs, no request deadlines. "
        "Real GPU execution; assumed compute allowances. "
        "Local EngineCore receipt latency, excluding network/API.",
        fontsize=9,
    )
    fig.savefig(output / f"{load}-latency.png", dpi=180)
    plt.close(fig)


def adaptation(root, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for row, mode in enumerate(MODES):
        directory = root / f"rep0-{mode}/recovery-mixed"
        result = json.loads((directory / "result.json").read_text())
        trace = json.loads((directory / "trace.json").read_text())
        prefill, decode = [], []
        for event in read_events(directory):
            elapsed = event["start_mono"] - result["monotonic_start"]
            if elapsed > 30 or not event["model_executed"]:
                continue
            p = sum(w["tokens"] for w in event["scheduled"] if w["kind"] == "prefill")
            d = sum(w["tokens"] for w in event["scheduled"] if w["kind"] == "decode")
            if p:
                prefill.append((elapsed, p))
            decode.append((elapsed, d))
        for column, points in enumerate((prefill, decode)):
            ax = axes[row, column]
            for limit, color in zip(
                trace["step_limits"], ("#fee2e2", "#fef3c7", "#e0f2fe")
            ):
                left, right = limit["start_s"], min(limit["end_s"], 30)
                ax.axvspan(left, right, color=color, alpha=0.6)
                if row == 0:
                    ax.text(
                        (left + right) / 2,
                        0.96,
                        f"{limit['max_step_s'] * 1000:g} ms",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=9,
                    )
            if points:
                ax.scatter(*zip(*points), s=6, color=COLORS[mode], alpha=0.5)
            ax.set_ylabel(
                NAMES[mode]
                + (
                    "\nprefill tokens / step"
                    if column == 0
                    else "\ndecode tokens / step"
                )
            )
            ax.set_ylim(-5 if column == 0 else -0.2, 570 if column == 0 else 9)
            ax.set_yticks([0, 128, 256, 512] if column == 0 else [0, 2, 4, 6, 8])
            ax.grid(axis="y", alpha=0.2)
            ax.set_xlim(0, 30)
    axes[0, 0].set_title("Actual prefill work")
    axes[0, 1].set_title("Actual decode concurrency")
    for ax in axes[-1]:
        ax.set_xlabel("Elapsed seconds after common request release")
    fig.suptitle(
        "Scheduler response during recovery · mixed batch · repetition 0", fontsize=14
    )
    fig.supxlabel(
        "Dots show executed steps, including recomputation. "
        "Backgrounds show assumed 30/50/100 ms "
        "allowances tied to XeoVerse eclipse exit.\n"
        "Stock and permissive controls do not enforce these caps. "
        "All engines receive the same eight requests.",
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
    data = json.loads(args.summary.read_text())
    if not data["complete"]:
        raise ValueError("Final plots require all paired measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    for load in ("isolated", "mixed"):
        latencies(data, args.output, load)
    adaptation(args.root, args.output)


if __name__ == "__main__":
    main()
