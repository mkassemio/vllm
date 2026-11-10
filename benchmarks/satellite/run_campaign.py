# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial, randomized pilot campaign; stop on errors and preserve every run."""

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import digest, write_json


def gpu_idle():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if result:
        raise RuntimeError(f"GPU compute processes are already active: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--orbital", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("scheduler", "handoff", "both"), default="both"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--contexts", type=int, nargs="+", default=[256, 2048, 8192])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or not args.contexts or min(args.contexts) < 1:
        parser.error("Seed count and contexts must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    script = Path(__file__).with_name("run_experiments.py")
    rng = random.Random(7031)
    plan = []
    if args.stage in ("scheduler", "both"):
        for seed in range(args.seeds):
            for scenario in ("steady", "windows"):
                trace = args.output / f"trace-{scenario}-{seed}.json"
                generated = trace.with_suffix(".generated.json")
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "make-trace",
                        "--seed",
                        str(seed),
                        "--scenario",
                        scenario,
                        "--output",
                        str(generated),
                    ],
                    check=True,
                )
                if trace.exists() and trace.read_bytes() != generated.read_bytes():
                    raise ValueError(f"Existing trace differs from this plan: {trace}")
                if trace.exists():
                    generated.unlink()
                else:
                    generated.replace(trace)
                modes = ["stock", "stock_guard", "orbital"]
                rng.shuffle(modes)
                for mode in modes:
                    name = f"scheduler-{scenario}-{seed}-{mode}"
                    source = args.orbital if mode == "orbital" else args.stock
                    plan.append(
                        {
                            "name": name,
                            "args": [
                                "scheduler",
                                "--source",
                                str(source),
                                "--mode",
                                mode,
                                "--trace",
                                str(trace),
                                "--calibration",
                                str(args.calibration),
                            ],
                        }
                    )
    if args.stage in ("handoff", "both"):
        for context in args.contexts:
            modes = ["stock_replay", "recompute", "transfer"]
            if context == 2048:
                modes.extend(["wait", "restart"])
            rng.shuffle(modes)
            for mode in modes:
                source = (
                    args.orbital if mode in ("recompute", "transfer") else args.stock
                )
                plan.append(
                    {
                        "name": f"handoff-{context}-{mode}",
                        "args": [
                            "handoff",
                            "--source",
                            str(source),
                            "--mode",
                            mode,
                            "--context",
                            str(context),
                            "--repeats",
                            "3",
                        ],
                    }
                )
    identity = {
        "jobs": plan,
        "calibration_sha256": digest(json.loads(args.calibration.read_text())),
        "sources": {
            name: subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, path in (("stock", args.stock), ("orbital", args.orbital))
        },
        "harness_sha256": {
            name: hashlib.sha256(script.with_name(name).read_bytes()).hexdigest()
            for name in (
                "run_campaign.py",
                "run_experiments.py",
                "experiment_utils.py",
                "experiment_worker.py",
            )
        },
    }
    signature = digest(identity)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["sha256"] != signature:
        raise ValueError("Cannot resume a different experiment plan")
    write_json(plan_path, identity | {"sha256": signature})
    # Cooperating harnesses share this lease. Never terminate foreign workers.
    with Path("/tmp/orbital-experiment-gpus.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(plan):
            directory = args.output / job["name"]
            if args.resume and (directory / "result.json").exists():
                continue
            if directory.exists():
                raise RuntimeError(
                    f"Inspect incomplete attempt before resuming: {directory}"
                )
            gpu_idle()
            progress = {
                "index": index + 1,
                "total": len(plan),
                "job": job["name"],
                "state": "running",
                "unix_s": time.time(),
            }
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            with (args.output / f"{job['name']}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        *job["args"],
                        "--output",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
            if result.returncode:
                write_json(
                    args.output / "progress.json", progress | {"state": "failed"}
                )
                raise RuntimeError(f"Experiment failed; inspect {job['name']}.log")
            data = json.loads((directory / "result.json").read_text())
            metric = data.get(
                "metrics", {"median_recovery_gap_s": data.get("median_recovery_gap_s")}
            )
            print(json.dumps({"completed": job["name"], "metrics": metric}), flush=True)
            write_json(args.output / "progress.json", progress | {"state": "completed"})
        write_json(
            args.output / "progress.json", {"state": "finished", "jobs": len(plan)}
        )


if __name__ == "__main__":
    main()
