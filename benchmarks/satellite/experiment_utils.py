# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace contracts, accounting, and isolated real-engine benchmark workers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def energy_snapshot(gpus):
    """Read whole-device energy counters; unsupported readings stay null."""
    from vllm.third_party import pynvml

    pynvml.nvmlInit()
    try:
        devices = {}
        for gpu in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            try:
                joules = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000
            except pynvml.NVMLError:
                joules = None
            devices[str(gpu)] = joules
        return {"monotonic_s": time.monotonic(), "total_device_j": devices}
    finally:
        pynvml.nvmlShutdown()


def energy_difference(before, after):
    values = []
    for gpu, start in before["total_device_j"].items():
        end = after["total_device_j"][gpu]
        if start is None or end is None or end < start:
            return None
        values.append(end - start)
    return sum(values)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def validate_trace(trace):
    """Reject ambiguous/censored workloads before loading any model."""
    if trace.get("schema_version") != 1:
        raise ValueError("Expected trace schema_version=1")
    horizon = trace["duration_s"]
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("duration_s must be positive and finite")
    end = 0.0
    for window in trace["windows"]:
        start, stop = window
        if not all(math.isfinite(v) for v in window):
            raise ValueError("Nonfinite window")
        if not 0 <= end <= start < stop <= horizon:
            raise ValueError("Windows must be ordered, disjoint, and inside horizon")
        end = stop
    seen = set()
    for request in trace["requests"]:
        name = request["id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Request IDs must be unique nonempty strings")
        seen.add(name)
        arrival, deadline = request["arrival_s"], request["deadline_s"]
        if not 0 <= arrival < deadline <= horizon:
            raise ValueError("Every request deadline must be inside the horizon")
        for field in ("prompt_tokens", "output_tokens"):
            if type(request[field]) is not int or request[field] < 1:
                raise ValueError(f"Invalid {field}")
        if type(request["priority"]) is not int:
            raise ValueError("Priority must be an integer")
        if request["class"] not in ("interactive", "onboard"):
            raise ValueError("Unknown workload class")
    if not seen:
        raise ValueError("Empty workload")
    return trace


def window_at(windows, now):
    return next(((a, b) for a, b in windows if a <= now < b), None)


def outside_windows(start, end, windows):
    """Wall time of a measured step outside permitted compute windows."""
    inside = sum(max(0.0, min(end, b) - max(start, a)) for a, b in windows)
    return max(0.0, end - start - inside)


def summarize_requests(records, duration_s):
    def summarize(rows):
        good = [r for r in rows if r["outcome"] == "completed_on_time"]
        ttft = [r["first_s"] - r["arrival_s"] for r in rows if r["first_s"] is not None]
        return {
            "offered": len(rows),
            "completed_on_time": len(good),
            "missed": len(rows) - len(good),
            "completion_fraction": len(good) / len(rows) if rows else None,
            "deadline_goodput_requests_s": len(good) / duration_s,
            "ttft_p50_s": percentile(ttft, 0.5),
            "ttft_p95_s": percentile(ttft, 0.95),
            "ttft_observations": len(ttft),
            "generated_tokens": sum(len(r["tokens"]) for r in rows),
        }

    return summarize(records) | {
        "by_class": {
            name: summarize([r for r in records if r["class"] == name])
            for name in ("interactive", "onboard")
        }
    }


def calibrated_profile(samples):
    """Conservative additive envelope, derived only from calibration steps."""
    decode = [s["elapsed_s"] for s in samples if s["kind"] == "decode"]
    prefill = [s for s in samples if s["kind"] == "prefill" and s["tokens"] >= 16]
    if not decode or not prefill:
        raise ValueError("Calibration requires prefill and decode observations")
    fixed = statistics.median(decode) / 2
    return {
        "fixed_step_s": fixed,
        "prefill_token_s": max(
            max(s["elapsed_s"] - fixed, 1e-6) / s["tokens"] for s in prefill
        ),
        "decode_token_s": max(percentile(decode, 0.95) - fixed, 1e-6),
        "power_w": 100.0,
        "safety_factor": 1.2,
    }


class EngineActor:
    """One subprocess per physical GPU, with bounded trusted local IPC.

    This transport is local control IPC, not an emulated intersatellite link.
    The worker loads vLLM from `source`, allowing an immutable upstream baseline.
    """

    def __init__(self, source, gpu, config, directory, timeout=300):
        self.timeout = timeout
        self.process = None
        self.connection = None
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="orbital-actor-")
        address = str(Path(self.temporary.name) / "control.sock")
        config_path = self.directory / "config.json"
        write_json(config_path, config)
        self.log = (self.directory / "worker.log").open("w")
        listener = Listener(address, family="AF_UNIX")
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            VLLM_BATCH_INVARIANT="1",
            VLLM_NO_USAGE_STATS="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONPATH=str(Path(source).resolve()),
        )
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("experiment_worker.py")),
                    address,
                    str(config_path.resolve()),
                ],
                cwd=source,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            # Connect happens before importing CUDA; bound startup separately.
            listener._listener._socket.settimeout(30)
            self.connection = listener.accept()
            self.ready = self.receive()
        except BaseException:
            listener.close()
            self.close()
            raise
        finally:
            listener.close()

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"Worker timeout: {self.directory / 'worker.log'}")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def call(self, operation, **payload):
        self.connection.send((operation, payload))
        return self.receive()

    def control(self, command, payload):
        return self.call("control", command=command, payload=payload)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                if self.connection is not None:
                    self.connection.send(("shutdown", {}))
                self.process.wait(timeout=10)
            except (OSError, EOFError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.connection is not None:
            self.connection.close()
        self.log.close()
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def wait_until(deadline):
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
