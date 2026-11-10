# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired XeoVerse scheduler traces serially on one real GPU."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import digest, validate_trace, write_json


def checked_job(command, log, timeout=900):
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f"Experiment exited with status {code}")
    except BaseException:
        # Only this job's process group is owned by this runner.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--orbital", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    traces = sorted(args.traces.glob("trace-*.json"))
    if not traces:
        parser.error("No exported XeoVerse traces")
    payloads = {p.name: validate_trace(json.loads(p.read_text())) for p in traces}
    if len({(p["scenario"], p["seed"]) for p in payloads.values()}) != len(traces):
        parser.error("Duplicate scenario/seed traces")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    script = Path(__file__).with_name("run_experiments.py")
    plan = []
    for batch in (128, 512):
        plan.append(
            {
                "name": f"calibration-{batch}",
                "args": [
                    "calibrate",
                    "--source",
                    str(args.stock),
                    "--batch-tokens",
                    str(batch),
                ],
            }
        )
    rng = random.Random(1771)
    rng.shuffle(traces)
    for trace in traces:
        configurations = [
            ("stock128", "stock_guard", 128),
            ("stock512", "stock_guard", 512),
            ("orbital", "orbital", 512),
            ("orbital_flat", "orbital_flat", 512),
        ]
        rng.shuffle(configurations)
        for label, mode, batch in configurations:
            plan.append(
                {
                    "name": f"{trace.stem.removeprefix('trace-')}-{label}",
                    "args": [
                        "scheduler",
                        "--source",
                        str(args.orbital if mode.startswith("orbital") else args.stock),
                        "--mode",
                        mode,
                        "--batch-tokens",
                        str(batch),
                        "--trace",
                        str(trace),
                        "--calibration",
                        str(args.output / f"calibration-{batch}" / "result.json"),
                    ],
                }
            )
    sources = {}
    for name, path in (("stock", args.stock), ("orbital", args.orbital)):
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        )
        if status.strip():
            raise ValueError(f"Serving checkout must be clean: {path}")
        sources[name] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    identity = {
        "jobs": plan,
        "sources": sources,
        "traces": {name: digest(data) for name, data in payloads.items()},
        "harness_sha256": {
            name: hashlib.sha256(script.with_name(name).read_bytes()).hexdigest()
            for name in (
                "run_xeoverse_scheduler.py",
                "run_experiments.py",
                "experiment_worker.py",
                "experiment_utils.py",
            )
        },
    }
    signature = digest(identity)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["sha256"] != signature:
        raise ValueError("Resume requires unchanged plan, harness, traces and sources")
    write_json(plan_path, identity | {"sha256": signature})
    with Path("/tmp/orbital-experiment-gpus.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(plan):
            directory = args.output / job["name"]
            if args.resume and (directory / "result.json").exists():
                continue
            if directory.exists():
                raise RuntimeError(
                    f"Preserve and inspect incomplete attempt: {directory}"
                )
            if job["args"][0] == "scheduler":
                profile = json.loads(
                    (args.output / "calibration-512" / "result.json").read_text()
                )["profile"]
                decode_floor = profile["safety_factor"] * (
                    profile["fixed_step_s"] + profile["decode_token_s"]
                )
                if any(
                    limit["max_step_s"] < decode_floor
                    for trace in payloads.values()
                    for limit in trace["step_limits"]
                ):
                    raise ValueError(
                        "Step allowance below calibrated minimum decode duration"
                    )
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if active:
                raise RuntimeError(f"GPU already in use: {active}")
            progress = {
                "index": index + 1,
                "total": len(plan),
                "job": job["name"],
                "state": "running",
                "unix_s": time.time(),
            }
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            try:
                with (args.output / f"{job['name']}.log").open("w") as log:
                    checked_job(
                        [
                            sys.executable,
                            str(script),
                            *job["args"],
                            "--output",
                            str(directory),
                        ],
                        log,
                    )
            except BaseException:
                write_json(
                    args.output / "progress.json", progress | {"state": "failed"}
                )
                raise
            write_json(args.output / "progress.json", progress | {"state": "completed"})
            print(json.dumps({"completed": job["name"]}), flush=True)
        write_json(
            args.output / "progress.json", {"state": "finished", "jobs": len(plan)}
        )


if __name__ == "__main__":
    main()
