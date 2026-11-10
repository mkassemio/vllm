# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure SC/LC x SO/LO completion, including TTFT and full-output TPOT."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from experiment_utils import (
    EngineActor,
    calibrated_profile,
    digest,
    energy_difference,
    energy_snapshot,
    percentile,
    step_limit_at,
    wait_until,
    write_json,
)
from run_experiments import MODEL, REVISION, finish, forecast_file, options
from shape_utils import (
    MODES,
    SHAPES,
    case_plan,
    latency_metrics,
    offered_requests,
    operating_trace,
)

FILES = (
    "run_shape_experiments.py",
    "shape_utils.py",
    "run_experiments.py",
    "experiment_worker.py",
    "experiment_utils.py",
    "xeoverse_trace.py",
)


def calibrate(args):
    config = options(args)
    with EngineActor(args.source, args.gpu, config, args.output / "worker") as actor:
        prompts = {n: actor.call("prompt", length=n) for n in (1000, 16500)}
        for length, prompt in prompts.items():
            finish(actor, f"warmup-{length}", prompt, 16)
        references = {}
        samples = []
        for length, prompt in prompts.items():
            tokens, steps = finish(
                actor, f"reference-{length}", prompt, 2000, record_steps=True
            )
            references[str(length)] = {
                "prompt": prompt,
                "tokens": tokens,
                "prompt_sha256": digest(prompt),
                "output_sha256": digest(tokens),
            }
            samples.extend(steps)
        profile = calibrated_profile(samples)
        write_json(
            args.output / "result.json",
            {
                "kind": "shape_calibration",
                "engine": actor.ready,
                "engine_options": config,
                "profile": profile,
                "references": references,
                "samples": samples,
                "profile_minimum_decode_s": (
                    profile["fixed_step_s"] + profile["decode_token_s"]
                )
                * profile["safety_factor"],
                "observed_step_p99_s": percentile(
                    [s["elapsed_s"] for s in samples], 0.99
                ),
                "scope": (
                    "Single-request calibration at both target prompt lengths "
                    "through 2,000 output tokens; unchanged additive profile, "
                    "modeled power only."
                ),
            },
        )


def run_case(actor, args, case, index, config, calibration, geometry):
    name = f"{case['scenario']}-{case['workload']}"
    directory = args.output / name
    directory.mkdir()
    trace = operating_trace(geometry, case["scenario"], args.watchdog_s)
    requests = offered_requests(case["workload"], args.repetition)
    requests = [r | {"id": f"{name}-{r['id']}"} for r in requests]
    write_json(directory / "trace.json", trace | {"requests": requests})
    if actor.call("status")["live"]:
        raise RuntimeError("Previous case left live engine requests")
    energy_before = energy_snapshot([args.gpu])
    start = time.monotonic() + 0.2
    epoch = time.time() + start - time.monotonic()
    if args.mode != "stock":
        forecast_file(
            args.output / "forecast.json",
            trace["windows"],
            epoch,
            args,
            version=index + 2,
            horizon_s=args.watchdog_s,
            step_limits=trace["step_limits"] if args.mode == "orbital" else None,
        )
    rows = {}
    for request in requests:
        reference = calibration["references"][str(request["prompt_tokens"])]
        actor.call(
            "add",
            id=request["id"],
            prompt=reference["prompt"],
            output_tokens=request["output_tokens"],
            priority=0,
            arrival_time=epoch,
        )
        rows[request["id"]] = request | {
            "tokens": [],
            "deliveries": [],
            "finish_s": None,
            "outcome": "pending",
        }
    active = set(rows)
    calls = executed = 0
    wait_until(start)
    if time.monotonic() - start > 0.01:
        raise RuntimeError("Request submission missed the common release time")
    with (directory / "events.jsonl").open("w") as events:
        while active and time.monotonic() - start < args.watchdog_s:
            step = actor.call("step")
            received = time.monotonic() - start
            calls += 1
            executed += int(step["model_executed"])
            events.write(
                json.dumps(
                    step
                    | {
                        "delivered_s": received,
                        "step_limit": step_limit_at(trace, step["start_mono"] - start),
                    }
                )
                + "\n"
            )
            for output in step["outputs"]:
                if output["id"] not in active or output["error"]:
                    raise RuntimeError(f"Unexpected output: {output}")
                row = rows[output["id"]]
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
                    if (
                        output["finish_reason"] != "length"
                        or len(row["tokens"]) != row["output_tokens"]
                    ):
                        raise RuntimeError(
                            "Request finished with the wrong output length"
                        )
                    row["finish_s"] = received
                    row["outcome"] = "completed"
                    active.remove(output["id"])
            if not step["model_executed"]:
                time.sleep(0.001)
    if active:
        actor.call("abort", ids=sorted(active))
        for name in active:
            rows[name]["outcome"] = "watchdog_expired"
    energy_after = energy_snapshot([args.gpu])
    for row in rows.values():
        expected = calibration["references"][str(row["prompt_tokens"])]["tokens"]
        if row["tokens"] != expected[: len(row["tokens"])]:
            raise RuntimeError(
                "Output differs from immutable upstream greedy reference"
            )
        row["metrics"] = latency_metrics(row)
    result = {
        "kind": "shape_case",
        **case,
        "mode": args.mode,
        "repetition": args.repetition,
        "engine": actor.ready,
        "engine_options": config,
        "calibration_sha256": digest(calibration),
        "geometry_sha256": digest(geometry),
        "requests": list(rows.values()),
        "all_outputs_match_reference": True,
        "all_completed": not active,
        "monotonic_start": start,
        "epoch_start": epoch,
        "watchdog_s": args.watchdog_s,
        "step_calls": calls,
        "executed_steps": executed,
        "energy_samples": [energy_before, energy_after],
        "measured_device_energy_j": energy_difference(energy_before, energy_after),
    }
    write_json(directory / "result.json", result)
    if active:
        raise TimeoutError(
            "Incomplete requests retained; watchdog is not a serving deadline"
        )
    if actor.call("status")["live"]:
        raise RuntimeError("Completed case left engine requests alive")
    return result


def run(args):
    calibration = json.loads(args.calibration.read_text())
    geometry = json.loads(args.geometry.read_text())
    forecast = args.output / "forecast.json"
    config = options(
        args, forecast if args.mode != "stock" else None, calibration["profile"]
    )
    base = calibration["engine_options"]
    for name in (
        "model",
        "revision",
        "dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if config[name] != base[name]:
            raise ValueError(f"Calibration mismatch: {name}")
    cases = case_plan(args.repetition)
    completed = []
    with EngineActor(args.source, args.gpu, config, args.output / "worker") as actor:
        if args.mode != "stock":
            forecast_file(forecast, [[0, 3600]], time.time(), args)
        for length, reference in calibration["references"].items():
            actual = actor.call("prompt", length=int(length))
            if actual != reference["prompt"]:
                raise RuntimeError("Tokenizer changed the reference prompt")
            tokens, _ = finish(actor, f"warmup-{length}", actual, 16)
            if tokens != reference["tokens"][:16]:
                raise RuntimeError("Warmup differs from upstream reference")
        for index, case in enumerate(cases):
            progress = {
                "case_index": index + 1,
                "cases": len(cases),
                **case,
                "state": "running",
            }
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            result = run_case(actor, args, case, index, config, calibration, geometry)
            completed.append(case)
            print(
                json.dumps(
                    {
                        "completed": case,
                        "metrics": [
                            {"shape": r["shape"], **r["metrics"]}
                            for r in result["requests"]
                        ],
                    }
                ),
                flush=True,
            )
            write_json(args.output / "progress.json", progress | {"state": "completed"})
    write_json(
        args.output / "result.json",
        {
            "kind": "shape_launch",
            "mode": args.mode,
            "repetition": args.repetition,
            "cases": completed,
            "all_completed": True,
            "all_outputs_match_reference": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("calibrate", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="stock")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--watchdog-s", type=float, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-model-len", type=int, default=18500)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--max-seqs", type=int, default=8)
    args = parser.parse_args()
    if args.max_model_len < max(a + b for a, b in SHAPES.values()):
        parser.error("All four shapes require max_model_len >= 18500")
    if args.gpu < 0 or min(args.batch_tokens, args.max_seqs) < 1 or args.repetition < 0:
        parser.error("Invalid engine limits, GPU or repetition")
    if args.experiment == "run":
        if args.geometry is None or args.calibration is None:
            parser.error("Measured runs require geometry and calibration")
        for scenario in ("sunlit", "recovery"):
            operating_trace(
                json.loads(args.geometry.read_text()), scenario, args.watchdog_s
            )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "harness_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in FILES
            },
        },
    )
    try:
        (calibrate if args.experiment == "calibrate" else run)(args)
    except BaseException:
        import traceback

        write_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
