# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate saved shape experiments and aggregate independent engine launches."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .analyze_xeoverse_scheduler import event_metrics, read_events
    from .experiment_utils import calibrated_profile, digest, step_limit_at, write_json
    from .shape_utils import (
        MODES,
        SHAPES,
        case_plan,
        latency_metrics,
        offered_requests,
        operating_trace,
    )
else:
    from analyze_xeoverse_scheduler import event_metrics, read_events
    from experiment_utils import calibrated_profile, digest, step_limit_at, write_json
    from shape_utils import (
        MODES,
        SHAPES,
        case_plan,
        latency_metrics,
        offered_requests,
        operating_trace,
    )

METRICS = ("ttft_s", "tpot_s", "completion_latency_s", "itl_p95_s")


def validate_event_receipts(events, result, trace):
    """Cross-check saved request receipts against the unaggregated engine log."""
    observed = {
        r["id"]: {"tokens": [], "deliveries": [], "finish_s": None}
        for r in result["requests"]
    }
    calls = 0
    for event in events:
        calls += 1
        start = event["start_mono"] - result["monotonic_start"]
        end = event["end_mono"] - result["monotonic_start"]
        received = event["delivered_s"]
        if not 0 <= start <= end <= received or event["elapsed_s"] != (
            event["end_mono"] - event["start_mono"]
        ):
            raise ValueError("Invalid engine step timing")
        if event["step_limit"] != step_limit_at(trace, start):
            raise ValueError("Step allowance differs from frozen trace")
        for output in event["outputs"]:
            if output["id"] not in observed or output["error"]:
                raise ValueError("Unknown or failed output in engine log")
            row = observed[output["id"]]
            if row["finish_s"] is not None:
                raise ValueError("Engine delivered output after completion")
            row["tokens"].extend(output["tokens"])
            if output["tokens"]:
                row["deliveries"].append(
                    {
                        "received_s": received,
                        "count": len(output["tokens"]),
                        "offset": len(row["tokens"]),
                    }
                )
            if output["finish_reason"] is not None:
                if output["finish_reason"] != "length":
                    raise ValueError("Unexpected engine completion reason")
                row["finish_s"] = received
        yield event
    if calls != result["step_calls"]:
        raise ValueError("Engine log has missing or extra steps")
    for request in result["requests"]:
        if any(request[k] != v for k, v in observed[request["id"]].items()):
            raise ValueError("Saved request receipts differ from raw engine log")


def validate_case_requests(result, trace, references):
    """Recompute metrics and compare every output with the stock reference."""
    offered = trace["requests"]
    saved = result["requests"]
    if len(saved) != len(offered) or [r["id"] for r in saved] != [
        r["id"] for r in offered
    ]:
        raise ValueError("Saved request denominator or ordering changed")
    for request, expected in zip(saved, offered):
        if any(request[key] != value for key, value in expected.items()):
            raise ValueError("Saved request metadata changed")
        reference = references[str(request["prompt_tokens"])]["tokens"]
        tokens = request["tokens"]
        if tokens != reference[: len(tokens)]:
            raise ValueError("Saved output differs from stock reference")
        metrics = latency_metrics(request)
        if metrics != request["metrics"]:
            raise ValueError("Recorded latency metrics differ from actual receipts")
        if not metrics["completed"]:
            raise ValueError("Incomplete output cannot enter full-request comparison")
    if not result["all_completed"] or not result["all_outputs_match_reference"]:
        raise ValueError("Case did not pass completion and correctness checks")


def check_manifest(directory, plan):
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_names = {
        "run_shape_experiments.py",
        "shape_utils.py",
        "run_experiments.py",
        "experiment_worker.py",
        "experiment_utils.py",
        "xeoverse_trace.py",
    }
    if set(manifest["harness_sha256"]) != expected_names:
        raise ValueError("Incomplete harness identity")
    if any(
        plan["harness_sha256"][name] != value
        for name, value in manifest["harness_sha256"].items()
    ):
        raise ValueError("Harness changed during campaign")
    return manifest


def summarize(root, allow_partial=False):
    plan = json.loads((root / "plan.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Frozen plan integrity failure")
    jobs = [job for job in plan["jobs"] if job["name"] != "calibration"]
    if len(jobs) != 9 or {(j["mode"], j["repetition"]) for j in jobs} != {
        (mode, repetition) for mode in MODES for repetition in range(3)
    }:
        raise ValueError("Expected three independent launches for each mode")
    calibration = json.loads((root / "calibration/result.json").read_text())
    check_manifest(root / "calibration", plan)
    if calibration["engine"]["source_commit"] != plan["sources"]["stock"]:
        raise ValueError("Calibration must use immutable stock source")
    if calibrated_profile(calibration["samples"]) != calibration["profile"]:
        raise ValueError("Calibration profile differs from observed samples")
    references = calibration["references"]
    if set(references) != {"1000", "16500"}:
        raise ValueError("Missing reference prompt length")
    for length, reference in references.items():
        if len(reference["prompt"]) != int(length) or len(reference["tokens"]) != 2000:
            raise ValueError("Invalid calibration prompt or output length")
        if (
            digest(reference["prompt"]) != reference["prompt_sha256"]
            or digest(reference["tokens"]) != reference["output_sha256"]
        ):
            raise ValueError("Reference integrity failure")
    geometry = json.loads((root / "geometry.json").read_text())
    if digest(geometry) != plan["geometry_sha256"]:
        raise ValueError("Geometry differs from frozen plan")
    base = calibration["engine_options"]
    rows, pending = [], []
    verified = generated = 0
    for job in jobs:
        directory = root / job["name"]
        cases = case_plan(job["repetition"])
        if job["cases"] != cases:
            raise ValueError("Paired case order differs from frozen plan")
        if (directory / "manifest.json").exists():
            manifest = check_manifest(directory, plan)
            args = manifest["arguments"]
            if args["mode"] != job["mode"] or args["repetition"] != job["repetition"]:
                raise ValueError("Launch arguments differ from frozen plan")
        for case in cases:
            name = f"{case['scenario']}-{case['workload']}"
            folder = directory / name
            path = folder / "result.json"
            if not path.exists():
                pending.append(f"{job['name']}/{name}")
                continue
            if not (directory / "manifest.json").exists():
                raise ValueError("Case lacks launch manifest")
            result = json.loads(path.read_text())
            trace = json.loads((folder / "trace.json").read_text())
            workload = offered_requests(case["workload"], job["repetition"])
            workload = [r | {"id": f"{name}-{r['id']}"} for r in workload]
            expected_trace = operating_trace(geometry, case["scenario"], 300)
            if trace != expected_trace | {"requests": workload}:
                raise ValueError("Trace differs from frozen workload or geometry")
            if any(result[key] != value for key, value in case.items()) or (
                result["mode"] != job["mode"]
                or result["repetition"] != job["repetition"]
                or result["watchdog_s"] != 300
            ):
                raise ValueError("Result case identity changed")
            source = "stock" if job["mode"] == "stock" else "orbital"
            if result["engine"]["source_commit"] != plan["sources"][source]:
                raise ValueError("Serving source changed during campaign")
            if result["geometry_sha256"] != digest(geometry) or result[
                "calibration_sha256"
            ] != digest(calibration):
                raise ValueError("Run used different calibration or geometry")
            config = result["engine_options"]
            common = {
                k: v
                for k, v in config.items()
                if k not in ("scheduler_cls", "additional_config")
            }
            if common != base:
                raise ValueError("Engine options changed between comparisons")
            if source == "stock":
                if config != base:
                    raise ValueError("Stock engine contains orbital configuration")
            else:
                if (
                    config["scheduler_cls"]
                    != "vllm.satellite.scheduler.SatelliteScheduler"
                ):
                    raise ValueError("Wrong orbital scheduler")
                orbital = config["additional_config"]["orbital"]
                if (
                    orbital["profile"] != calibration["profile"]
                    or orbital["guard_s"] != 0.001
                ):
                    raise ValueError("Orbital profile or guard changed")
            validate_case_requests(result, trace, references)
            verified += len(result["requests"])
            generated += sum(len(r["tokens"]) for r in result["requests"])
            timing = event_metrics(
                validate_event_receipts(read_events(folder), result, trace),
                calibration["profile"],
            )
            if timing["model_executed_steps"] != result["executed_steps"]:
                raise ValueError("Saved step count differs from raw events")
            rows.append(
                {
                    "job": job["name"],
                    **case,
                    "mode": job["mode"],
                    "repetition": job["repetition"],
                    "requests": [
                        {"id": r["id"], "shape": r["shape"], **r["metrics"]}
                        for r in result["requests"]
                    ],
                    "energy_j": result["measured_device_energy_j"],
                    "makespan_s": max(r["finish_s"] for r in result["requests"]),
                    **timing,
                }
            )
        if (directory / "result.json").exists():
            launch = json.loads((directory / "result.json").read_text())
            if not (
                launch["kind"] == "shape_launch"
                and launch["cases"] == cases
                and launch["mode"] == job["mode"]
                and launch["repetition"] == job["repetition"]
                and launch["all_completed"]
                and launch["all_outputs_match_reference"]
            ):
                raise ValueError("Invalid engine-launch completion record")
        else:
            pending.append(job["name"])
    if pending and not allow_partial:
        raise ValueError(f"Incomplete campaign: {len(pending)} pending cases/launches")
    groups = defaultdict(list)
    for row in rows:
        load = "mixed" if row["workload"] == "mixed" else "isolated"
        for shape in SHAPES:
            requests = [r for r in row["requests"] if r["shape"] == shape]
            if not requests:
                continue
            groups[(row["scenario"], load, shape, row["mode"])].append(
                {
                    "repetition": row["repetition"],
                    "requests": len(requests),
                    **{
                        metric: statistics.median([r[metric] for r in requests])
                        if all(r[metric] is not None for r in requests)
                        else None
                        for metric in METRICS
                    },
                }
            )
    aggregates = []
    for (scenario, load, shape, mode), samples in sorted(groups.items()):
        samples.sort(key=lambda s: s["repetition"])
        metrics = {}
        for metric in METRICS:
            values = [s[metric] for s in samples if s[metric] is not None]
            metrics[metric] = {
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "launches_observed": len(values),
            }
        aggregates.append(
            {
                "scenario": scenario,
                "load": load,
                "shape": shape,
                "mode": mode,
                "independent_launches": len(samples),
                "requests": sum(s["requests"] for s in samples),
                "launch_medians": samples,
                "metrics": metrics,
            }
        )
    if not pending and (len(rows) != 90 or verified != 216 or generated != 243648):
        raise ValueError("Final workload denominator or output count differs")
    return {
        "complete": not pending,
        "pending": pending,
        "plan_sha256": plan["sha256"],
        "sources": plan["sources"],
        "runs": rows,
        "aggregates": aggregates,
        "offline_output_validation": {
            "requests_checked": verified,
            "output_tokens_checked": generated,
            "reference_source": "immutable stock calibration",
        },
        "scope": (
            "Real GPU EngineCore execution, local IPC receipt times, no HTTP or "
            "network emulation. TTFT includes queueing and prefill; TPOT excludes "
            "first-token wait. Only complete outputs enter latency comparisons. "
            "Medians across three engine launches, after within-launch medians "
            "of two requests per shape for mixed batches; descriptive ranges, "
            "not confidence intervals. XeoVerse supplies orbital timing; the "
            "30/50/100 ms compute allowances are assumptions."
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
                "cases": len(result["runs"]),
                **result["offline_output_validation"],
            }
        )
    )


if __name__ == "__main__":
    main()
