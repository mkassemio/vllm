# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze paired scheduler runs, retaining misses and step-budget overruns."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .experiment_utils import digest, percentile, write_json
else:
    from experiment_utils import digest, percentile, write_json

LABELS = ("stock128", "stock512", "orbital", "orbital_flat")


def read_events(directory):
    path = directory / "events.jsonl"
    if path.exists():
        with path.open() as stream:
            yield from (json.loads(line) for line in stream)
    else:
        with gzip.open(path.with_suffix(".jsonl.gz"), "rt") as stream:
            yield from (json.loads(line) for line in stream)


def label_for(result):
    return (
        f"stock{result['engine_options']['max_num_batched_tokens']}"
        if result["mode"] == "stock_guard"
        else result["mode"]
    )


def validate_saved_requests(trace, result):
    expected = {request["id"]: request for request in trace["requests"]}
    saved = result["requests"]
    if len(saved) != len(expected) or {r["id"] for r in saved} != set(expected):
        raise ValueError("Saved requests changed the offered workload denominator")
    for request in saved:
        if any(request[key] != value for key, value in expected[request["id"]].items()):
            raise ValueError("Saved request metadata differs from the frozen trace")
        if len(request["tokens"]) > request["output_tokens"]:
            raise ValueError("Saved output exceeds the original generation allowance")


def event_metrics(events, profile=None):
    distributions = (
        "prefill_chunks",
        "prefill_step_tokens",
        "step_s",
        "predicted_s",
        "calibrated_ratio",
        "decode_tokens",
    )
    phases = defaultdict(
        lambda: {
            "steps": 0,
            "overruns": 0,
            "excess_s": 0.0,
            "prefill_chunks": [],
            "prefill_step_tokens": [],
            "step_s": [],
            "predicted_s": [],
            "calibrated_ratio": [],
            "decode_tokens": [],
        }
    )
    frontier = defaultdict(int)
    decode_batches = defaultdict(list)
    repeated = preemptions = executed = 0
    work_tokens = 0
    for event in events:
        preemptions += len(event.get("preempted", []))
        if not event["model_executed"]:
            continue
        executed += 1
        limit = event.get("step_limit")
        key = str(limit["max_step_s"]) if limit else "outside"
        phase = phases[key]
        phase["steps"] += 1
        elapsed = event["elapsed_s"]
        phase["step_s"].append(elapsed)
        if event.get("policy"):
            phase["predicted_s"].append(event["policy"]["estimated_s"])
        if limit:
            excess = max(0, elapsed - limit["max_step_s"])
            phase["overruns"] += int(excess > 0)
            phase["excess_s"] += excess
        prefill = decode = 0
        for work in event["scheduled"]:
            name = work["request_id"]
            count = work["tokens"]
            start = work["computed_start"]
            end = start + count
            repeated += max(0, min(end, frontier[name]) - start)
            frontier[name] = max(frontier[name], end)
            work_tokens += count
            if work["kind"] == "prefill":
                phase["prefill_chunks"].append(count)
                prefill += count
            else:
                decode += count
        if prefill:
            phase["prefill_step_tokens"].append(prefill)
        phase["decode_tokens"].append(decode)
        if not prefill and decode:
            decode_batches[decode].append(elapsed)
        if profile is not None:
            prediction = profile["safety_factor"] * (
                profile["fixed_step_s"]
                + prefill * profile["prefill_token_s"]
                + decode * profile["decode_token_s"]
            )
            phase["calibrated_ratio"].append(prediction / elapsed)
    summarized = {}
    for cap, phase in phases.items():
        summarized[cap] = {k: v for k, v in phase.items() if k not in distributions}
        for field in distributions:
            values = phase[field]
            summarized[cap][field] = {
                "count": len(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    return {
        "phases": summarized,
        "preemptions": preemptions,
        "recomputed_token_work": repeated,
        "computed_token_work": work_tokens,
        "model_executed_steps": executed,
        "step_budget_overruns": sum(p["overruns"] for p in phases.values()),
        "step_budget_excess_s": sum(p["excess_s"] for p in phases.values()),
        "decode_batch_costs": {
            str(count): {
                "steps": len(times),
                "actual_p50_s": percentile(times, 0.5),
                "actual_p95_s": percentile(times, 0.95),
                "profile_estimate_s": profile["safety_factor"]
                * (profile["fixed_step_s"] + count * profile["decode_token_s"])
                if profile is not None
                else None,
            }
            for count, times in sorted(decode_batches.items())
        },
    }


def summarize(root, allow_partial=False):
    plan = json.loads((root / "plan.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Frozen plan integrity failure")
    if not any(job["args"][0] == "scheduler" for job in plan["jobs"]):
        raise ValueError("Plan contains no scheduler comparisons")
    rows = []
    pairs = defaultdict(dict)
    pending = []
    for job in plan["jobs"]:
        if job["args"][0] != "scheduler":
            continue
        directory = root / job["name"]
        path = directory / "result.json"
        if not path.exists():
            pending.append(job["name"])
            continue
        result = json.loads(path.read_text())
        trace = json.loads((directory / "trace.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        label = label_for(result)
        if label not in LABELS:
            raise ValueError(f"Unknown configuration: {label}")
        key = (trace["scenario"], trace["seed"])
        if label in pairs[key]:
            raise ValueError(f"Duplicate result: {key} {label}")
        if digest(trace) != result["trace_sha256"]:
            raise ValueError("Trace integrity failure")
        if result["trace_sha256"] != plan["traces"][f"trace-{key[0]}-{key[1]}.json"]:
            raise ValueError("Run differs from frozen trace")
        if not result["all_generated_prefixes_match_reference"]:
            raise ValueError("Output correctness failure")
        validate_saved_requests(trace, result)
        for name, value in manifest["harness_sha256"].items():
            if plan["harness_sha256"][name] != value:
                raise ValueError("Harness changed during campaign")
        source = "stock" if label.startswith("stock") else "orbital"
        if result["engine"]["source_commit"] != plan["sources"][source]:
            raise ValueError("Serving source changed during campaign")
        batch = result["engine_options"]["max_num_batched_tokens"]
        calibration = json.loads(
            (root / f"calibration-{batch}" / "result.json").read_text()
        )
        if digest(calibration) != result["calibration_sha256"]:
            raise ValueError("Calibration changed during campaign")
        pairs[key][label] = result
        timing = event_metrics(read_events(directory), calibration["profile"])
        row = {
            "job": job["name"],
            "scenario": key[0],
            "seed": key[1],
            "label": label,
            "metrics": result["metrics"],
            "energy_j": result["measured_device_energy_j"],
            "boundary_overrun_s": result["step_time_outside_windows_s"],
            "useful_tokens": sum(
                len(r["tokens"])
                for r in result["requests"]
                if r["outcome"] == "completed_on_time"
            ),
            **timing,
        }
        rows.append(row)
    if pending and not allow_partial:
        raise ValueError(f"Incomplete campaign: {len(pending)} missing results")
    for key, modes in pairs.items():
        if (
            len(
                {
                    json.dumps(r["reference_sha256"], sort_keys=True)
                    for r in modes.values()
                }
            )
            != 1
        ):
            raise ValueError(f"Greedy references differ across modes: {key}")
        if not allow_partial and set(modes) != set(LABELS):
            raise ValueError(f"Missing comparison mode: {key}")
    references = {}
    results = [result for modes in pairs.values() for result in modes.values()]
    for result in results:
        for request in result["requests"]:
            expected_hash = result["reference_sha256"][str(request["prompt_tokens"])]
            if digest(request["tokens"]) == expected_hash:
                references[expected_hash] = request["tokens"]
    verified = 0
    missing_references = set()
    for result in results:
        for request in result["requests"]:
            expected_hash = result["reference_sha256"][str(request["prompt_tokens"])]
            if expected_hash not in references:
                missing_references.add(expected_hash)
                continue
            tokens = request["tokens"]
            if tokens != references[expected_hash][: len(tokens)]:
                raise ValueError("Saved output prefix differs from greedy reference")
            verified += 1
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["label"])].append(row)
    aggregates = []
    for (scenario, label), group in sorted(groups.items()):
        group.sort(key=lambda row: row["seed"])
        by_class = {}
        for name in ("interactive", "onboard"):
            classes = [r["metrics"]["by_class"][name] for r in group]
            ttft = [c["ttft_p50_s"] for c in classes if c["ttft_p50_s"] is not None]
            p95 = [c["ttft_p95_s"] for c in classes if c["ttft_p95_s"] is not None]
            by_class[name] = {
                "offered": sum(c["offered"] for c in classes),
                "completed": sum(c["completed_on_time"] for c in classes),
                "completed_each": [c["completed_on_time"] for c in classes],
                "ttft_observations": sum(c["ttft_observations"] for c in classes),
                "median_run_ttft_p50_ms": statistics.median(ttft) * 1000
                if ttft
                else None,
                "median_run_ttft_p95_ms": statistics.median(p95) * 1000
                if p95
                else None,
            }
        aggregates.append(
            {
                "scenario": scenario,
                "label": label,
                "independent_launches": len(group),
                "seeds": [r["seed"] for r in group],
                "by_class": by_class,
                "completed_each": [r["metrics"]["completed_on_time"] for r in group],
                "preemptions_each": [r["preemptions"] for r in group],
                "recomputed_token_work_each": [
                    r["recomputed_token_work"] for r in group
                ],
                "budget_overrun_steps_each": [r["step_budget_overruns"] for r in group],
                "boundary_overrun_ms_each": [
                    r["boundary_overrun_s"] * 1000 for r in group
                ],
                "device_energy_j_each": [r["energy_j"] for r in group],
            }
        )
    return {
        "complete": not pending,
        "pending": pending,
        "runs": rows,
        "aggregates": aggregates,
        "offline_output_validation": {
            "requests_checked": verified,
            "missing_complete_reference_hashes": sorted(missing_references),
            "scope": (
                "Complete observed outputs must hash to the recorded reference; "
                "all saved prefixes are then compared independently."
            ),
        },
        "scope": (
            "Paired arrival seeds; descriptive launch replicates, "
            "not confidence intervals. Step allowance is assumed, eclipse is "
            "XeoVerse geometry. TTFT is conditional; "
            "missing first tokens remain misses."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root, args.allow_partial)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "runs": len(result["runs"]),
                "pending": len(result["pending"]),
            }
        )
    )


if __name__ == "__main__":
    main()
