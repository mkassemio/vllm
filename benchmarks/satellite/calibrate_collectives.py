# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure real collective costs with torchrun and emit a candidate group plan.

Example: torchrun --standalone --nproc-per-node=2
benchmarks/satellite/calibrate_collectives.py --backend nccl --output costs.json
"""

import argparse
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.satellite.collectives import gathered_sum


def execute(tensor, group, mode, comm):
    if mode == "native":
        if comm is not None:
            return comm.all_reduce(tensor)
        result = tensor.clone()
        dist.all_reduce(result, group=group)
        return result
    if comm is not None:
        return gathered_sum(tensor, group, mode == "direct", pynccl_comm=comm)
    return gathered_sum(tensor, group, mode == "direct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 4096, 65536])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--transport", choices=("torch", "pynccl"), default="torch")
    parser.add_argument("--max-scratch-bytes", type=int, default=268435456)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    args = parser.parse_args()
    if min(args.sizes) <= 0 or args.repeats < 2 or args.warmup < 1:
        parser.error("Positive sizes, at least two repeats and one warmup required")
    if args.transport == "pynccl" and args.backend != "nccl":
        parser.error("PyNCCL requires the NCCL backend")
    device = torch.device("cpu")
    if args.backend == "nccl":
        index = int(os.environ["LOCAL_RANK"])
        torch.accelerator.set_device_index(index)
        device = torch.device("cuda", index)
    dist.init_process_group(
        args.backend,
        timeout=timedelta(seconds=60),
        device_id=device if device.type == "cuda" else None,
    )
    comm = None
    try:
        group = dist.group.WORLD
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if args.transport == "pynccl":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            control = dist.new_group(backend="gloo")
            comm = PyNcclCommunicator(control, device)
            if comm.disabled:
                raise RuntimeError("PyNCCL is unavailable")
        dist.barrier()
        samples, costs = [], []
        for mode in ("native", "direct", "allgather"):
            observations = []
            for size in sorted(set(args.sizes)):
                tensor = torch.full(
                    (size,), rank + 1, dtype=getattr(torch, args.dtype), device=device
                )
                message_bytes = tensor.numel() * tensor.element_size()
                if (world_size + 4) * message_bytes > args.max_scratch_bytes:
                    raise ValueError("Calibration tensor exceeds scratch budget")
                elapsed = []
                for iteration in range(args.warmup + args.repeats):
                    dist.barrier()
                    if device.type != "cpu":
                        torch.accelerator.synchronize()
                    start = time.perf_counter()
                    result = execute(tensor, group, mode, comm)
                    if device.type != "cpu":
                        torch.accelerator.synchronize()
                    duration = time.perf_counter() - start
                    worst = torch.tensor(duration, dtype=torch.float64, device=device)
                    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
                    if iteration >= args.warmup:
                        elapsed.append(worst.item())
                    torch.testing.assert_close(
                        result,
                        torch.full_like(result, world_size * (world_size + 1) / 2),
                        rtol=0,
                        atol=0,
                    )
                p95 = sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1]
                samples.append(
                    {
                        "algorithm": mode,
                        "message_bytes": message_bytes,
                        "median_s": statistics.median(elapsed),
                        "p95_s": p95,
                    }
                )
                observations.append((message_bytes, p95))
                if args.profile_dir:
                    args.profile_dir.mkdir(parents=True, exist_ok=True)
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type != "cpu":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    dist.barrier()
                    with torch.profiler.profile(activities=activities) as profile:
                        result = execute(tensor, group, mode, comm)
                        if device.type != "cpu":
                            torch.accelerator.synchronize()
                    stem = args.profile_dir / f"rank{rank}-{mode}-{size}"
                    profile.export_chrome_trace(str(stem) + ".trace.json")
                    events = [
                        {
                            "name": event.key,
                            "count": event.count,
                            "self_cpu_us": event.self_cpu_time_total,
                            "self_device_us": event.self_device_time_total,
                        }
                        for event in profile.key_averages()
                    ]
                    Path(str(stem) + ".summary.json").write_text(
                        json.dumps(events, indent=2) + "\n"
                    )
            xs, ys = zip(*observations)
            mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
            denominator = sum((x - mean_x) ** 2 for x in xs)
            slope = (
                max(
                    0,
                    sum((x - mean_x) * (y - mean_y) for x, y in observations)
                    / denominator,
                )
                if denominator
                else 0
            )
            intercept = max(0, max(y - slope * x for x, y in observations))
            costs.append(
                {
                    "name": mode,
                    "latency_s": 1.2 * intercept,
                    "seconds_per_byte": 1.2 * slope,
                }
            )
        if rank == 0:
            report = {
                "backend": args.backend,
                "transport": args.transport,
                "dtype": args.dtype,
                "torch_version": str(torch.__version__),
                "repeats": args.repeats,
                "warmup": args.warmup,
                "samples": samples,
                "plan": {
                    "version": 1,
                    "world_size": world_size,
                    "max_scratch_bytes": args.max_scratch_bytes,
                    "costs": costs,
                },
                "scope": (
                    "This execution group and tested sizes only; "
                    "remeasure after route changes."
                ),
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Wrote {len(samples)} measured cases to {args.output}")
    finally:
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
