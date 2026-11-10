# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible real-GPU scheduler and handoff experiments.

Resource windows are explicit synthetic inputs, replayed at wall-clock speed.
Handoff uses measured local IPC and GPU copies; it does not emulate an ISL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import ExitStack
from pathlib import Path

from experiment_utils import (
    EngineActor,
    calibrated_profile,
    digest,
    energy_difference,
    energy_snapshot,
    outside_windows,
    percentile,
    step_limit_at,
    summarize_requests,
    validate_trace,
    wait_until,
    window_at,
    write_json,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def options(args, forecast=None, profile=None):
    result = {
        "model": args.model,
        "revision": args.revision,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.batch_tokens,
        "max_num_seqs": args.max_seqs,
        "gpu_memory_utilization": 0.25,
        "scheduling_policy": "priority",
        "seed": 42,
        "disable_log_stats": True,
    }
    if forecast is not None:
        result.update(
            scheduler_cls="vllm.satellite.scheduler.SatelliteScheduler",
            additional_config={
                "orbital": {
                    "forecast_path": str(forecast.resolve()),
                    "guard_s": 0.001,
                    "profile": profile
                    or {
                        "fixed_step_s": 0.00001,
                        "prefill_token_s": 0.000001,
                        "decode_token_s": 0.000001,
                        "power_w": 100,
                    },
                }
            },
        )
    return result


def forecast_file(
    path, windows, epoch, args, version=1, horizon_s=3600, step_limits=None
):
    now = time.time()
    limits = (
        step_limits
        if step_limits is not None
        else [
            {"start_s": start, "end_s": end, "max_step_s": 10} for start, end in windows
        ]
    )
    write_json(
        path,
        {
            "version": version,
            "issued_s": now,
            "expires_s": epoch + horizon_s + 1,
            "windows": [
                {
                    "start_s": epoch + limit["start_s"],
                    "end_s": epoch + limit["end_s"],
                    "max_power_w": 1000,
                    "energy_j": 1_000_000,
                    "max_step_s": limit["max_step_s"],
                    "max_tokens": args.batch_tokens,
                    "max_requests": args.max_seqs,
                }
                for limit in limits
            ],
        },
    )


def finish(actor, request_id, prompt, count, *, record_steps=False):
    actor.call("add", id=request_id, prompt=prompt, output_tokens=count)
    tokens, samples = [], []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = actor.call("step")
        if record_steps and len(result["scheduled"]) == 1:
            samples.append(result["scheduled"][0] | {"elapsed_s": result["elapsed_s"]})
        for output in result["outputs"]:
            if output["id"] != request_id or output["error"]:
                raise RuntimeError(f"Unexpected output: {output}")
            tokens.extend(output["tokens"])
            if output["finish_reason"] is not None:
                if output["finish_reason"] != "length" or len(tokens) != count:
                    raise RuntimeError("Request failed or changed its output budget")
                return tokens, samples
    raise TimeoutError("Request failed to finish")


def make_trace(args):
    rng = random.Random(args.seed)
    duration = args.duration
    windows = (
        [[0, duration]]
        if args.scenario == "steady"
        else [
            [0, duration * 0.25],
            [duration * 0.4, duration * 0.65],
            [duration * 0.8, duration],
        ]
    )
    requests = []
    for index in range(args.requests):
        interactive = index % 2 == 1
        arrival = rng.uniform(0, duration * 0.7)
        requests.append(
            {
                "id": f"r{index:04d}",
                "arrival_s": arrival,
                "deadline_s": min(duration, arrival + (2.0 if interactive else 6.0)),
                "prompt_tokens": 256 if interactive else 2048,
                "output_tokens": 32 if interactive else 128,
                "priority": 0 if interactive else 1,
                "class": "interactive" if interactive else "onboard",
            }
        )
    trace = validate_trace(
        {
            "schema_version": 1,
            "seed": args.seed,
            "scenario": args.scenario,
            "provenance": "Synthetic compute availability, not eclipse/ground contact",
            "duration_s": duration,
            "windows": windows,
            "requests": sorted(requests, key=lambda r: (r["arrival_s"], r["id"])),
        }
    )
    write_json(args.output, trace)


def calibrate(args):
    with EngineActor(
        args.source, args.gpu, options(args), args.output / "worker"
    ) as actor:
        prompt = actor.call("prompt", length=2048)
        finish(actor, "warmup", prompt, 32)
        samples = []
        for repetition in range(3):
            for length in (128, 512, 2048):
                _, steps = finish(
                    actor,
                    f"c{repetition}-{length}",
                    prompt[:length],
                    32,
                    record_steps=True,
                )
                samples.extend(steps)
        write_json(
            args.output / "result.json",
            {
                "kind": "calibration",
                "engine": actor.ready,
                "engine_options": options(args),
                "profile": calibrated_profile(samples),
                "stock_guard_s": percentile([s["elapsed_s"] for s in samples], 0.99)
                * 1.2,
                "samples": samples,
                "scope": "Single-request timing envelope; power_w=100 is modeled only",
            },
        )


def scheduler(args):
    trace = validate_trace(json.loads(args.trace.read_text()))
    calibration = json.loads(args.calibration.read_text())
    profile = calibration["profile"]
    path = args.output / "forecast.json"
    orbital = args.mode in ("orbital", "orbital_flat")
    config = options(args, path if orbital else None, profile)
    for field in (
        "model",
        "revision",
        "dtype",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if config[field] != calibration["engine_options"][field]:
            raise ValueError(f"Calibration mismatch: {field}")
    if (
        max(r["prompt_tokens"] + r["output_tokens"] for r in trace["requests"])
        > args.max_model_len
    ):
        raise ValueError("Trace exceeds max_model_len")
    with EngineActor(args.source, args.gpu, config, args.output / "worker") as actor:
        if orbital:
            forecast_file(path, [[0, 3600]], time.time(), args)
        prompts = {
            n: actor.call("prompt", length=n)
            for n in sorted({r["prompt_tokens"] for r in trace["requests"]})
        }
        references = {}
        for n, prompt in prompts.items():
            count = max(
                r["output_tokens"] for r in trace["requests"] if r["prompt_tokens"] == n
            )
            references[n], _ = finish(actor, f"warmup-{n}", prompt, count)
        records = {
            r["id"]: r
            | {
                "tokens": [],
                "first_s": None,
                "finish_s": None,
                "admitted_s": None,
                "outcome": "pending",
            }
            for r in trace["requests"]
        }
        monotonic_start = time.monotonic() + 0.2
        epoch_start = time.time() + (monotonic_start - time.monotonic())
        if orbital:
            forecast_file(
                path,
                trace["windows"],
                epoch_start,
                args,
                version=2,
                horizon_s=trace["duration_s"],
                step_limits=trace.get("step_limits")
                if args.mode == "orbital"
                else None,
            )
        write_json(args.output / "trace.json", trace)
        pending = sorted(records.values(), key=lambda r: (r["arrival_s"], r["id"]))
        active = set()
        violations, executed, step_wall = 0.0, 0, 0.0
        guard = calibration["stock_guard_s"] if args.mode == "stock_guard" else 0
        with (args.output / "events.jsonl").open("w") as events:
            wait_until(monotonic_start)
            energy_before = energy_snapshot([args.gpu])
            while True:
                now = time.monotonic() - monotonic_start
                expired = [
                    name for name in active if records[name]["deadline_s"] <= now
                ]
                if expired:
                    actor.call("abort", ids=expired)
                    for name in expired:
                        records[name]["outcome"] = "deadline_expired"
                        active.remove(name)
                while pending and pending[0]["arrival_s"] <= now:
                    request = pending.pop(0)
                    if request["deadline_s"] <= now:
                        request["outcome"] = "expired_before_admission"
                        continue
                    actor.call(
                        "add",
                        id=request["id"],
                        prompt=prompts[request["prompt_tokens"]],
                        output_tokens=request["output_tokens"],
                        priority=request["priority"],
                        arrival_time=epoch_start + request["arrival_s"],
                        extra_args={
                            "orbital": {
                                "deadline_s": epoch_start + request["deadline_s"]
                            }
                        }
                        if orbital
                        else None,
                    )
                    request["admitted_s"] = time.monotonic() - monotonic_start
                    active.add(request["id"])
                now = time.monotonic() - monotonic_start
                if now >= trace["duration_s"]:
                    break
                window = window_at(trace["windows"], now)
                if not active or window is None or now + guard >= window[1]:
                    # No GPU step outside the shared availability gate. This is
                    # stock's pause/keep baseline, without discarding resident KV.
                    time.sleep(0.001)
                    continue
                result = actor.call("step")
                delivered = time.monotonic() - monotonic_start
                start = result["start_mono"] - monotonic_start
                end = result["end_mono"] - monotonic_start
                violation = (
                    outside_windows(start, end, trace["windows"])
                    if result["model_executed"]
                    else 0.0
                )
                step_wall += result["elapsed_s"]
                violations += violation
                executed += 1
                events.write(
                    json.dumps(
                        result
                        | {
                            "delivered_s": delivered,
                            "outside_s": violation,
                            "step_limit": step_limit_at(trace, start),
                        }
                    )
                    + "\n"
                )
                for output in result["outputs"]:
                    request = records[output["id"]]
                    request["tokens"].extend(output["tokens"])
                    if output["tokens"] and request["first_s"] is None:
                        request["first_s"] = delivered
                    if output["finish_reason"] is not None:
                        request["finish_s"] = delivered
                        complete = (
                            output["finish_reason"] == "length"
                            and len(request["tokens"]) == request["output_tokens"]
                        )
                        request["outcome"] = (
                            "completed_on_time"
                            if complete and delivered <= request["deadline_s"]
                            else "completed_late"
                            if complete
                            else "engine_aborted"
                        )
                        active.discard(output["id"])
            if active:
                actor.call("abort", ids=sorted(active))
                for name in active:
                    records[name]["outcome"] = "horizon_expired"
            for request in pending:
                request["outcome"] = "never_admitted"
        rows = list(records.values())
        for row in rows:
            if row["tokens"] != references[row["prompt_tokens"]][: len(row["tokens"])]:
                raise RuntimeError(
                    f"Scheduled output changed greedy tokens: {row['id']}"
                )
        energy_after = energy_snapshot([args.gpu])
        report = {
            "kind": "scheduler",
            "mode": args.mode,
            "engine": actor.ready,
            "engine_options": config,
            "trace_sha256": digest(trace),
            "calibration_sha256": digest(calibration),
            "reference_sha256": {n: digest(tokens) for n, tokens in references.items()},
            "all_generated_prefixes_match_reference": True,
            "epoch_start": epoch_start,
            "monotonic_start": monotonic_start,
            "metrics": summarize_requests(rows, trace["duration_s"]),
            "step_calls": executed,
            "step_wall_s": step_wall,
            "step_time_outside_windows_s": violations,
            "measured_device_energy_j": energy_difference(energy_before, energy_after),
            "energy_samples": [energy_before, energy_after],
            "stock_guard_s": guard,
            "requests": rows,
            "final_status": actor.call("status"),
            "scope": (
                "TP1, eager, synchronous EngineCore; trace-defined compute windows; "
                "local token recipient; no power cap or ground-contact emulation"
            ),
        }
        write_json(args.output / "result.json", report)
        print(json.dumps(report["metrics"]), flush=True)


class InjectedReplyLoss(ConnectionError):
    """A completed activation whose response was deliberately discarded."""


def timed_endpoint(actor, node, rpc_events, fault):
    def call(command, payload):
        start = time.monotonic()
        result = actor.control(command, payload)
        images = result.get("images", payload.get("images", []))
        rpc_events.append(
            {
                "node": node,
                "command": command,
                "elapsed_s": time.monotonic() - start,
                "kv_payload_bytes": sum(len(image["payload"]) for image in images),
            }
        )
        if (
            fault["enabled"]
            and not fault["injected"]
            and node == "target"
            and command == "activate"
        ):
            fault["injected"] = True
            rpc_events[-1]["reply_discarded"] = True
            raise InjectedReplyLoss("Lost the target activation reply after execution")
        return result

    return call


def token_publisher(coordinator, request_id, tokens, deliveries):
    def publish(new_tokens, owner, epoch, finished=False):
        tokens.extend(new_tokens)
        coordinator.publish(request_id, owner, epoch, tuple(tokens), finished=finished)
        if new_tokens:
            deliveries.append({"offset": len(tokens), "mono": time.monotonic()})

    return publish


def handoff(args):
    from vllm.satellite.handoff import HandoffCoordinator
    from vllm.satellite.state import StateConflict, StateJournal

    orbital = args.mode in ("recompute", "transfer")
    forecasts = [args.output / f"forecast-{index}.json" for index in range(2)]
    if args.context + args.output_tokens > args.max_model_len:
        raise ValueError("Handoff request exceeds max_model_len")
    if not 0 < args.trigger_tokens < args.output_tokens:
        raise ValueError("Trigger must be strictly inside the output budget")
    with ExitStack() as stack:
        actors = [
            stack.enter_context(
                EngineActor(
                    args.source,
                    gpu,
                    options(args, forecasts[index] if orbital else None),
                    args.output / f"worker-{index}",
                )
            )
            for index, gpu in enumerate((args.gpu, args.target_gpu))
        ]
        source, target = actors
        if orbital:
            for path in forecasts:
                forecast_file(path, [[0, 3600]], time.time(), args)
        prompt = source.call("prompt", length=args.context)
        for index, actor in enumerate(actors):
            finish(actor, f"warmup-{index}", prompt, 8)
        expected, _ = finish(source, "reference", prompt, args.output_tokens)
        runs = []
        for repetition in range(args.repeats):
            request_id = f"handoff-{repetition}"
            journal = StateJournal(args.output / f"journal-{repetition}.sqlite")
            try:
                journal.register(request_id, "source" if orbital else "gateway")
                rpc_events = []
                fault = {
                    "enabled": args.fault == "lost_activate_reply",
                    "injected": False,
                }

                coordinator = HandoffCoordinator(
                    journal,
                    {
                        "source": timed_endpoint(source, "source", rpc_events, fault),
                        "target": timed_endpoint(target, "target", rpc_events, fault),
                    },
                )
                tokens = []
                deliveries = []

                publish = token_publisher(coordinator, request_id, tokens, deliveries)

                energy_before = energy_snapshot([args.gpu, args.target_gpu])
                start = time.monotonic()
                source.call(
                    "add",
                    id=request_id,
                    prompt=prompt,
                    output_tokens=args.output_tokens,
                )
                limit = start + 120
                while len(tokens) < args.trigger_tokens:
                    if time.monotonic() > limit:
                        raise TimeoutError("Source did not reach the handoff trigger")
                    for output in source.call("step")["outputs"]:
                        if output["error"] or output["finish_reason"] is not None:
                            raise RuntimeError("Source terminated before handoff")
                        publish(output["tokens"], "source" if orbital else "gateway", 0)
                cut = len(tokens)
                last_source = deliveries[-1]["mono"]
                transition_start = time.monotonic()
                stale_fenced = None
                if orbital:
                    try:
                        owner = coordinator.start(
                            f"move-{repetition}", request_id, "target", mode=args.mode
                        )
                    except InjectedReplyLoss:
                        owner = coordinator.advance(f"move-{repetition}")
                    if fault["enabled"] and not fault["injected"]:
                        raise RuntimeError("Requested fault was not exercised")
                    if (owner.owner, owner.epoch) != ("target", 1):
                        raise RuntimeError("Handoff failed to commit ownership")
                    try:
                        coordinator.publish(request_id, "source", 0, tuple(tokens))
                    except StateConflict:
                        stale_fenced = True
                    if not stale_fenced:
                        raise RuntimeError("Stale source was not fenced")
                    receiver, receiver_owner, receiver_epoch = target, "target", 1
                elif args.mode == "wait":
                    wait_until(transition_start + args.outage_s)
                    receiver, receiver_owner, receiver_epoch = source, "gateway", 0
                else:
                    source.call("abort", ids=[request_id])
                    replay = args.mode == "stock_replay"
                    target.call(
                        "add",
                        id=request_id,
                        prompt=prompt + tokens if replay else prompt,
                        output_tokens=args.output_tokens - cut
                        if replay
                        else args.output_tokens,
                    )
                    receiver, receiver_owner, receiver_epoch = target, "gateway", 0
                transition_end = time.monotonic()
                generated = []
                finished = False
                while not finished:
                    if time.monotonic() > limit + args.outage_s:
                        raise TimeoutError("Destination failed to complete")
                    for output in receiver.call("step")["outputs"]:
                        if output["id"] != request_id or output["error"]:
                            raise RuntimeError(
                                f"Unexpected destination output: {output}"
                            )
                        new_tokens = output["tokens"]
                        if args.mode == "restart":
                            previous = len(generated)
                            generated.extend(new_tokens)
                            if (
                                generated[: min(cut, len(generated))]
                                != expected[: min(cut, len(generated))]
                            ):
                                raise RuntimeError(
                                    "Restart changed already delivered tokens"
                                )
                            new_tokens = generated[max(previous, cut) :]
                        finished = output["finish_reason"] is not None
                        if finished and output["finish_reason"] != "length":
                            raise RuntimeError("Destination aborted")
                        publish(new_tokens, receiver_owner, receiver_epoch, finished)
                end = time.monotonic()
                energy_after = energy_snapshot([args.gpu, args.target_gpu])
                if tuple(expected) != journal.tokens(request_id) or tokens != expected:
                    raise RuntimeError("Handoff changed tokens or the output limit")
                if journal.ownership(request_id).phase != "FINISHED":
                    raise RuntimeError("Completion was not durably committed")
                if (
                    orbital
                    and coordinator.advance(f"move-{repetition}").phase != "FINISHED"
                ):
                    raise RuntimeError("Completed handoff retry was not idempotent")
                next_delivery = next(d["mono"] for d in deliveries if d["offset"] > cut)
                runs.append(
                    {
                        "repetition": repetition,
                        "cut_tokens": cut,
                        "recovery_gap_s": next_delivery - last_source,
                        "transition_s": transition_end - transition_start,
                        "completion_s": end - start,
                        "measured_device_energy_j": energy_difference(
                            energy_before, energy_after
                        ),
                        "energy_samples": [energy_before, energy_after],
                        "exact_tokens": True,
                        "output_sha256": digest(tokens),
                        "output_tokens": len(tokens),
                        "stale_source_fenced": stale_fenced,
                        "lost_activation_reply_injected": fault["injected"],
                        "kv_image_bytes": max(
                            (e["kv_payload_bytes"] for e in rpc_events), default=0
                        ),
                        "rpc_events": rpc_events,
                        "deliveries": deliveries,
                    }
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in runs[-1].items()
                            if k not in ("rpc_events", "deliveries")
                        }
                    ),
                    flush=True,
                )
            finally:
                journal.close()
        write_json(
            args.output / "result.json",
            {
                "kind": "handoff",
                "mode": args.mode,
                "engines": [a.ready for a in actors],
                "engine_options": options(args),
                "context_tokens": args.context,
                "output_tokens": args.output_tokens,
                "outage_s": args.outage_s,
                "runs": runs,
                "median_recovery_gap_s": percentile(
                    [r["recovery_gap_s"] for r in runs], 0.5
                ),
                "reference_sha256": digest(expected),
                "scope": (
                    "Planned handoff, warm TP1 replicas; reachable local authority; "
                    "measured local IPC and GPU KV copies; no ISL or model reload; "
                    "or source crash"
                ),
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    trace = subparsers.add_parser("make-trace")
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--requests", type=int, default=80)
    trace.add_argument("--duration", type=float, default=16)
    trace.add_argument("--scenario", choices=("steady", "windows"), default="windows")
    trace.add_argument("--output", type=Path, required=True)
    for name in ("calibrate", "scheduler", "handoff"):
        child = subparsers.add_parser(name)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--gpu", type=int, default=0)
        child.add_argument("--model", default=MODEL)
        child.add_argument("--revision", default=REVISION)
        child.add_argument("--max-model-len", type=int, default=12288)
        child.add_argument("--batch-tokens", type=int, default=128)
        child.add_argument("--max-seqs", type=int, default=8)
        if name == "scheduler":
            child.add_argument(
                "--mode",
                choices=("stock", "stock_guard", "orbital", "orbital_flat"),
                required=True,
            )
            child.add_argument("--trace", type=Path, required=True)
            child.add_argument("--calibration", type=Path, required=True)
        elif name == "handoff":
            child.add_argument(
                "--mode",
                choices=("wait", "restart", "stock_replay", "recompute", "transfer"),
                required=True,
            )
            child.add_argument("--target-gpu", type=int, default=1)
            child.add_argument("--context", type=int, default=2048)
            child.add_argument("--output-tokens", type=int, default=64)
            child.add_argument("--trigger-tokens", type=int, default=16)
            child.add_argument("--outage-s", type=float, default=2)
            child.add_argument("--repeats", type=int, default=3)
            child.add_argument(
                "--fault", choices=("none", "lost_activate_reply"), default="none"
            )
    args = parser.parse_args()
    if args.experiment == "make-trace":
        if args.requests < 1 or not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("Request count and finite duration must be positive")
        make_trace(args)
        return
    if min(args.max_model_len, args.batch_tokens, args.max_seqs) < 1 or args.gpu < 0:
        parser.error("Engine limits must be positive and GPU index nonnegative")
    if args.experiment == "handoff":
        if args.fault != "none" and args.mode not in ("recompute", "transfer"):
            parser.error("Activation-reply injection requires an orbital handoff mode")
        if args.target_gpu == args.gpu or args.target_gpu < 0:
            parser.error("Handoff requires two distinct nonnegative GPU indices")
        if args.repeats < 1 or args.context < 1:
            parser.error("Handoff repetitions and context must be positive")
        if not math.isfinite(args.outage_s) or args.outage_s < 0:
            parser.error("Outage must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "harness_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "run_experiments.py",
                    "experiment_worker.py",
                    "experiment_utils.py",
                )
            },
            "created_unix_s": time.time(),
        },
    )
    try:
        {"calibrate": calibrate, "scheduler": scheduler, "handoff": handoff}[
            args.experiment
        ](args)
    except BaseException:
        import traceback

        write_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
