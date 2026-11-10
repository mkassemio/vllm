# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a frozen, paired shape campaign serially on one idle A100."""

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
from run_shape_experiments import FILES
from run_xeoverse_scheduler import checked_job
from shape_utils import MODES, SCENARIOS, SHAPES, case_plan, operating_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--orbital", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("stock", "orbital", "geometry", "output"):
        setattr(args, name, getattr(args, name).resolve())
    geometry = json.loads(args.geometry.read_text())
    for scenario in SCENARIOS:
        operating_trace(geometry, scenario, 300)
    script = Path(__file__).with_name("run_shape_experiments.py")
    plan = [{"name": "calibration", "args": ["calibrate", "--source", str(args.stock)]}]
    for repetition in range(3):
        modes = list(MODES)
        random.Random(3801 + repetition).shuffle(modes)
        for mode in modes:
            plan.append(
                {
                    "name": f"rep{repetition}-{mode}",
                    "mode": mode,
                    "repetition": repetition,
                    "cases": case_plan(repetition),
                    "args": [
                        "run",
                        "--source",
                        str(args.stock if mode == "stock" else args.orbital),
                        "--mode",
                        mode,
                        "--repetition",
                        str(repetition),
                        "--calibration",
                        str(args.output / "calibration/result.json"),
                        "--geometry",
                        str(args.geometry),
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
        "geometry_sha256": digest(geometry),
        "shapes": SHAPES,
        "harness_sha256": {
            name: hashlib.sha256(script.with_name(name).read_bytes()).hexdigest()
            for name in (*FILES, "run_shape_campaign.py", "run_xeoverse_scheduler.py")
        },
        "measurement": {
            "gpu": 0,
            "repetitions": 3,
            "watchdog_s": 300,
            "request_deadlines": False,
            "mixed_requests_per_shape": 2,
            "max_model_len": 18500,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 8,
        },
    }
    signature = digest(identity)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["sha256"] != signature:
        raise ValueError(
            "Resume requires unchanged plan, harness, geometry and sources"
        )
    write_json(plan_path, identity | {"sha256": signature})
    write_json(args.output / "geometry.json", geometry)
    with Path("/tmp/orbital-experiment-gpus.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(plan):
            directory = args.output / job["name"]
            if args.resume and (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                if job["name"] != "calibration" and not (
                    result["all_completed"] and result["all_outputs_match_reference"]
                ):
                    raise RuntimeError("Cannot resume past an unsuccessful launch")
                continue
            if directory.exists():
                raise RuntimeError(
                    f"Preserve and inspect incomplete attempt: {directory}"
                )
            if job["name"] != "calibration":
                calibration = json.loads(
                    (args.output / "calibration/result.json").read_text()
                )
                if calibration["profile_minimum_decode_s"] >= 0.1:
                    raise ValueError("Calibration cannot fit even the 100 ms allowance")
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
                        timeout=3600,
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
