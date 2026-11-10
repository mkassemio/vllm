# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact workload shapes and receipt-time metrics for completion experiments."""

import math
import random

if __package__:
    from .experiment_utils import percentile
    from .xeoverse_trace import segment_limits
else:
    from experiment_utils import percentile
    from xeoverse_trace import segment_limits

SHAPES = {
    "SC-SO": (1000, 256),
    "SC-LO": (1000, 2000),
    "LC-SO": (16500, 256),
    "LC-LO": (16500, 2000),
}
MODES = ("stock", "orbital", "orbital_flat")
SCENARIOS = ("sunlit", "recovery")


def case_plan(repetition):
    cases = [
        {"scenario": scenario, "workload": shape}
        for scenario in SCENARIOS
        for shape in (*SHAPES, "mixed")
    ]
    random.Random(901 + repetition).shuffle(cases)
    return cases


def offered_requests(workload, repetition):
    names = [workload] if workload in SHAPES else list(SHAPES) * 2
    if workload != "mixed" and workload not in SHAPES:
        raise ValueError("Unknown workload shape")
    if workload == "mixed":
        random.Random(2401 + repetition).shuffle(names)
    return [
        {
            "id": f"r{index:02d}",
            "shape": name,
            "prompt_tokens": SHAPES[name][0],
            "output_tokens": SHAPES[name][1],
            "arrival_s": 0.0,
            "priority": 0,
        }
        for index, name in enumerate(names)
    ]


def operating_trace(geometry, scenario, horizon):
    if scenario not in SCENARIOS or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Invalid scenario or watchdog horizon")
    transitions = geometry["transitions"]
    exit_event = next(t for t in transitions if not t["eclipsed_after"])
    exit_s = exit_event["offset_s"]
    entry_s = next(
        t["offset_s"]
        for t in transitions
        if t["eclipsed_after"] and t["offset_s"] > exit_s
    )
    start = (
        geometry["segments"]["sunlit"]["start_offset_s"]
        if scenario == "sunlit"
        else exit_event["bracket_s"][1] + 0.001
    )
    windows, limits = segment_limits(start, horizon, exit_s, entry_s)
    if windows != [[0, horizon]]:
        raise ValueError(
            "Completion experiment must remain sunlit through its watchdog"
        )
    return {
        "scenario": scenario,
        "duration_s": horizon,
        "windows": windows,
        "step_limits": limits,
        "orbital_start_offset_s": start,
        "orbital_seconds_per_wall_second": 1,
        "scope": (
            "Assumed sunlight compute slots; recovery begins just after the "
            "XeoVerse eclipse-exit bracket. No forced pause or request deadline."
        ),
    }


def latency_metrics(request):
    """Compute full-request TPOT; never synthesize token gaps within chunks."""
    arrivals = request["arrival_s"]
    expected = request["output_tokens"]
    tokens = request["tokens"]
    deliveries = request["deliveries"]
    if not math.isfinite(arrivals) or type(expected) is not int or expected < 1:
        raise ValueError("Invalid arrival time or output limit")
    if len(tokens) > expected:
        raise ValueError("Output exceeded the original generation limit")
    offset = 0
    previous = arrivals
    for delivery in deliveries:
        when = delivery["received_s"]
        count = delivery["count"]
        if (
            not math.isfinite(when)
            or when < previous
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError("Invalid delivery time or chunk count")
        offset += count
        if delivery["offset"] != offset:
            raise ValueError("Delivery offsets must be contiguous")
        previous = when
    if offset != len(tokens):
        raise ValueError("Delivered token counts do not match saved output")
    complete = request["outcome"] == "completed"
    finish = request["finish_s"]
    if complete and (
        len(tokens) != expected
        or finish is None
        or not math.isfinite(finish)
        or finish < previous
    ):
        raise ValueError(
            "Completed request has incomplete output or invalid finish time"
        )
    first = deliveries[0]["received_s"] if deliveries else None
    last = deliveries[-1]["received_s"] if deliveries else None
    individual = bool(deliveries) and all(d["count"] == 1 for d in deliveries)
    gaps = [
        b["received_s"] - a["received_s"] for a, b in zip(deliveries, deliveries[1:])
    ]
    return {
        "completed": complete,
        "generated_tokens": len(tokens),
        "ttft_s": first - arrivals if first is not None else None,
        "tpot_s": (last - first) / (expected - 1)
        if complete and expected > 1
        else None,
        "completion_latency_s": finish - arrivals if complete else None,
        "itl_p95_s": percentile(gaps, 0.95) if complete and individual else None,
        "receipt_granularity": "individual_tokens"
        if individual
        else "chunks"
        if deliveries
        else "none",
    }
