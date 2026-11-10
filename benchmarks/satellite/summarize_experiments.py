# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize complete pilot artifacts without treating tokens as replicates."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiment_utils import write_json


def median_optional(values):
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def recomputation_evidence(result_path, result):
    events = result_path.with_name("events.jsonl")
    compressed = result_path.with_name("events.jsonl.gz")
    if not events.exists() and not compressed.exists():
        return None
    decoded, repeated = set(), set()
    prefill = defaultdict(int)
    if events.exists():
        contents = events.read_text()
    else:
        with gzip.open(compressed, "rt") as stream:
            contents = stream.read()
    for line in contents.splitlines():
        for work in json.loads(line)["scheduled"]:
            name = work["request_id"]
            if work["kind"] == "decode":
                decoded.add(name)
            else:
                prefill[name] += work["tokens"]
                if name in decoded:
                    repeated.add(name)
    return {
        "requests_prefilled_after_decoding": sorted(repeated),
        "excess_prefill_tokens_lower_bound": sum(
            max(0, prefill[r["id"]] - r["prompt_tokens"]) for r in result["requests"]
        ),
        "scope": "Observed positive computed-token deltas; reset steps can be omitted",
    }


def summarize(root):
    scheduler = defaultdict(dict)
    handoffs = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["kind"] == "scheduler":
            trace = json.loads(path.with_name("trace.json").read_text())
            result["recomputation_evidence"] = recomputation_evidence(path, result)
            scheduler[(trace["scenario"], trace["seed"])][result["mode"]] = result
        elif result["kind"] == "handoff":
            handoffs.append(
                {
                    "context_tokens": result["context_tokens"],
                    "mode": result["mode"],
                    "engine_launches": 1,
                    "request_repeats": len(result["runs"]),
                    "median_gap_ms": result["median_recovery_gap_s"] * 1000,
                    "gap_range_ms": [
                        min(r["recovery_gap_s"] for r in result["runs"]) * 1000,
                        max(r["recovery_gap_s"] for r in result["runs"]) * 1000,
                    ],
                    "kv_image_bytes": max(r["kv_image_bytes"] for r in result["runs"]),
                    "exact_tokens": all(r["exact_tokens"] for r in result["runs"]),
                    "reference_sha256": result["reference_sha256"],
                }
            )
    paired = []
    grouped = defaultdict(list)
    for (scenario, seed), modes in sorted(scheduler.items()):
        if len({r["trace_sha256"] for r in modes.values()}) != 1:
            raise ValueError("Paired scheduler modes have different traces")
        if len({r["calibration_sha256"] for r in modes.values()}) != 1:
            raise ValueError("Paired scheduler modes have different calibrations")
        if (
            len(
                {
                    json.dumps(r["reference_sha256"], sort_keys=True)
                    for r in modes.values()
                }
            )
            != 1
        ):
            raise ValueError("Scheduler modes generated different greedy references")
        for mode, result in modes.items():
            metrics = result["metrics"]
            ttft = metrics["by_class"]["interactive"]["ttft_p50_s"]
            row = {
                "scenario": scenario,
                "seed": seed,
                "mode": mode,
                "completed": metrics["completed_on_time"],
                "offered": metrics["offered"],
                "interactive_completed": metrics["by_class"]["interactive"][
                    "completed_on_time"
                ],
                "interactive_ttft_p50_ms": ttft * 1000 if ttft is not None else None,
                "onboard_completed": metrics["by_class"]["onboard"][
                    "completed_on_time"
                ],
                "energy_j": result["measured_device_energy_j"],
                "boundary_overrun_ms": result["step_time_outside_windows_s"] * 1000,
                "recomputation_evidence": result["recomputation_evidence"],
            }
            grouped[(scenario, mode)].append(row)
        if all(mode in modes for mode in ("orbital", "stock_guard")):
            paired.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "orbital_minus_stock_guard_completed": modes["orbital"]["metrics"][
                        "completed_on_time"
                    ]
                    - modes["stock_guard"]["metrics"]["completed_on_time"],
                }
            )
    aggregates = []
    for (scenario, mode), rows in sorted(grouped.items()):
        aggregates.append(
            {
                "scenario": scenario,
                "mode": mode,
                "independent_launches": len(rows),
                "completed": [r["completed"] for r in rows],
                "offered_each": [r["offered"] for r in rows],
                "interactive_completed": [r["interactive_completed"] for r in rows],
                "onboard_completed": [r["onboard_completed"] for r in rows],
                "median_interactive_ttft_ms": median_optional(
                    r["interactive_ttft_p50_ms"] for r in rows
                ),
                "ttft_runs_observed": sum(
                    r["interactive_ttft_p50_ms"] is not None for r in rows
                ),
                "median_device_energy_j": statistics.median(r["energy_j"] for r in rows)
                if all(r["energy_j"] is not None for r in rows)
                else None,
                "boundary_overrun_ms": [r["boundary_overrun_ms"] for r in rows],
                "recomputation_evidence": [r["recomputation_evidence"] for r in rows],
            }
        )
    for context in {h["context_tokens"] for h in handoffs}:
        if (
            len(
                {
                    h["reference_sha256"]
                    for h in handoffs
                    if h["context_tokens"] == context
                }
            )
            != 1
        ):
            raise ValueError("Handoff modes generated different greedy references")
    return {
        "scheduler": aggregates,
        "paired_differences": paired,
        "handoff": sorted(handoffs, key=lambda r: (r["context_tokens"], r["mode"])),
        "scope": (
            "Descriptive pilot; three seeds cannot establish a broad performance "
            "claim; handoff repeats share one engine pair"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
