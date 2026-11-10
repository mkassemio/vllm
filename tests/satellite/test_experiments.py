# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect experiment denominators, window boundaries, and calibration inputs."""

import gzip
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmarks.satellite.experiment_utils import (
    calibrated_profile,
    outside_windows,
    summarize_requests,
    validate_trace,
    window_at,
)


def trace():
    return {
        "schema_version": 1,
        "duration_s": 10,
        "windows": [[0, 3], [5, 10]],
        "requests": [
            {
                "id": "r",
                "arrival_s": 1,
                "deadline_s": 5,
                "prompt_tokens": 32,
                "output_tokens": 16,
                "priority": 0,
                "class": "interactive",
            }
        ],
    }


def test_window_end_is_closed_and_spanning_step_counts_only_outside_time():
    windows = trace()["windows"]
    assert window_at(windows, 0) == (0, 3)
    assert window_at(windows, 3) is None
    assert window_at(windows, 5) == (5, 10)
    assert window_at(windows, 10) is None
    assert outside_windows(2, 6, windows) == 2
    assert outside_windows(3, 5, windows) == 2
    assert outside_windows(0, 10, windows) == 2
    assert outside_windows(6, 7, windows) == 0


def test_failed_and_never_started_requests_remain_in_goodput_denominator():
    rows = [
        {
            "class": "interactive",
            "arrival_s": 0,
            "first_s": 1,
            "tokens": [1, 2],
            "outcome": "completed_on_time",
        },
        {
            "class": "interactive",
            "arrival_s": 0,
            "first_s": None,
            "tokens": [],
            "outcome": "expired_before_admission",
        },
        {
            "class": "onboard",
            "arrival_s": 0,
            "first_s": 2,
            "tokens": [1],
            "outcome": "horizon_expired",
        },
    ]
    result = summarize_requests(rows, 10)
    assert result["offered"] == 3
    assert result["missed"] == 2
    assert result["completion_fraction"] == 1 / 3
    assert result["deadline_goodput_requests_s"] == 0.1
    assert result["ttft_observations"] == 2
    assert result["by_class"]["onboard"]["completion_fraction"] == 0
    assert result["by_class"]["interactive"]["completion_fraction"] == 0.5


def test_trace_rejects_duplicate_ids_and_deadlines_beyond_measurement_horizon():
    original = trace()
    assert validate_trace(original) == original
    duplicate = deepcopy(original)
    duplicate["requests"].append(deepcopy(original["requests"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_trace(duplicate)
    censored = deepcopy(original)
    censored["requests"][0]["deadline_s"] = 11
    with pytest.raises(ValueError, match="horizon"):
        validate_trace(censored)
    overlapping = deepcopy(original)
    overlapping["windows"] = [[0, 5], [4, 8]]
    with pytest.raises(ValueError, match="disjoint"):
        validate_trace(overlapping)


def test_calibration_envelope_covers_observed_prefill_steps():
    samples = [
        {"kind": "prefill", "tokens": 128, "elapsed_s": 0.02},
        {"kind": "prefill", "tokens": 32, "elapsed_s": 0.01},
        {"kind": "decode", "tokens": 1, "elapsed_s": 0.006},
        {"kind": "decode", "tokens": 1, "elapsed_s": 0.007},
    ]
    profile = calibrated_profile(samples)
    for sample in samples[:2]:
        predicted = (
            profile["fixed_step_s"] + sample["tokens"] * profile["prefill_token_s"]
        )
        assert predicted >= sample["elapsed_s"]
    with pytest.raises(ValueError, match="prefill and decode"):
        calibrated_profile(samples[:2])


def analysis_artifact(root, mode):
    directory = root / mode
    directory.mkdir()
    result = {
        "kind": "scheduler",
        "mode": mode,
        "trace_sha256": "same-trace",
        "calibration_sha256": "same-calibration",
        "reference_sha256": {"32": "same"},
        "measured_device_energy_j": None,
        "step_time_outside_windows_s": 0,
        "metrics": {
            "offered": 1,
            "completed_on_time": 0,
            "by_class": {
                "interactive": {"completed_on_time": 0, "ttft_p50_s": None},
                "onboard": {"completed_on_time": 0},
            },
        },
    }
    (directory / "trace.json").write_text(
        json.dumps({"scenario": "windows", "seed": 0})
    )
    (directory / "result.json").write_text(json.dumps(result))
    return result, directory / "result.json"


def analyze(root):
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/satellite/summarize_experiments.py"
    )
    return subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--output",
            str(root / "summary.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "field", ["trace_sha256", "calibration_sha256", "reference_sha256"]
)
def test_analysis_rejects_unpaired_inputs(tmp_path, field):
    analysis_artifact(tmp_path, "stock_guard")
    result, path = analysis_artifact(tmp_path, "orbital")
    result[field] = "different"
    path.write_text(json.dumps(result))
    process = analyze(tmp_path)
    assert process.returncode != 0
    assert "different" in process.stderr
    assert not (tmp_path / "summary.json").exists()


def test_analysis_keeps_missing_latency_and_energy_as_null(tmp_path):
    analysis_artifact(tmp_path, "stock_guard")
    process = analyze(tmp_path)
    assert process.returncode == 0, process.stderr
    row = json.loads((tmp_path / "summary.json").read_text())["scheduler"][0]
    assert row["median_interactive_ttft_ms"] is None
    assert row["ttft_runs_observed"] == 0
    assert row["median_device_energy_j"] is None
    assert row["completed"] == [0]


def test_compressed_step_artifacts_preserve_recomputation_evidence(tmp_path):
    result, path = analysis_artifact(tmp_path, "orbital")
    result["requests"] = [{"id": "r", "prompt_tokens": 16}]
    path.write_text(json.dumps(result))
    with gzip.open(path.with_name("events.jsonl.gz"), "wt") as stream:
        for kind, count in (("decode", 1), ("prefill", 32)):
            stream.write(
                json.dumps(
                    {"scheduled": [{"request_id": "r", "kind": kind, "tokens": count}]}
                )
                + "\n"
            )
    process = analyze(tmp_path)
    assert process.returncode == 0, process.stderr
    row = json.loads((tmp_path / "summary.json").read_text())["scheduler"][0]
    evidence = row["recomputation_evidence"][0]
    assert evidence["requests_prefilled_after_decoding"] == ["r"]
    assert evidence["excess_prefill_tokens_lower_bound"] == 16


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--gpu", "0", "--target-gpu", "0"], "distinct"),
        (["--repeats", "0"], "positive"),
        (["--outage-s", "nan"], "finite"),
    ],
)
def test_invalid_handoff_is_rejected_before_loading_a_gpu(tmp_path, arguments, message):
    script = (
        Path(__file__).resolve().parents[2] / "benchmarks/satellite/run_experiments.py"
    )
    output = tmp_path / "run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "handoff",
            "--source",
            str(tmp_path),
            "--mode",
            "transfer",
            "--output",
            str(output),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert message in process.stderr
    assert not output.exists()


def test_resource_limits_must_cover_open_time_without_holes_or_overlap():
    item = trace()
    item["step_limits"] = [
        {"start_s": 0, "end_s": 1, "max_step_s": 0.1},
        {"start_s": 1, "end_s": 3, "max_step_s": 0.025},
        {"start_s": 5, "end_s": 10, "max_step_s": 0.05},
    ]
    assert validate_trace(item) == item
    for field, value in (("start_s", 2), ("max_step_s", float("nan"))):
        invalid = deepcopy(item)
        invalid["step_limits"][1][field] = value
        with pytest.raises(ValueError):
            validate_trace(invalid)
    invalid = deepcopy(item)
    invalid["step_limits"][1]["start_s"] = 0.5
    with pytest.raises(ValueError, match="disjoint"):
        validate_trace(invalid)


def test_xeoverse_segments_preserve_real_time_and_eclipse_gate():
    from benchmarks.satellite.xeoverse_trace import segment_limits, shadow_margin

    assert shadow_margin([7000, 0, 0], [1, 0, 0]) > 0
    assert shadow_margin([-7000, 0, 0], [1, 0, 0]) < 0
    assert shadow_margin([-7000, 7000, 0], [1, 0, 0]) > 0
    windows, limits = segment_limits(67, 45, 0, 100)
    assert windows == [[0, 33]]
    assert [(s["start_s"], s["end_s"], s["max_step_s"]) for s in limits] == [
        (0, 13, 0.1),
        (13, 23, 0.05),
        (23, 33, 0.030),
    ]
    windows, limits = segment_limits(88, 45, 100, 500)
    assert windows == [[12, 45]]
    assert [s["max_step_s"] for s in limits] == [0.030, 0.05, 0.1]


def test_step_analysis_counts_recomputed_work_and_keeps_empty_steps_separate():
    from benchmarks.satellite.analyze_xeoverse_scheduler import event_metrics

    events = []
    for start, tokens, preempted, executed, elapsed in (
        (0, 100, [], True, 0.04),
        (100, 1, [], True, 0.02),
        (0, 50, ["r"], True, 0.03),
        (50, 100, [], True, 0.02),
        (150, 0, [], False, 0.01),
    ):
        events.append(
            {
                "preempted": preempted,
                "model_executed": executed,
                "elapsed_s": elapsed,
                "step_limit": {"max_step_s": 0.03},
                "scheduled": [
                    {
                        "request_id": "r",
                        "computed_start": start,
                        "tokens": tokens,
                        "kind": "prefill",
                    }
                ],
            }
        )
    result = event_metrics(events)
    assert result["recomputed_token_work"] == 101
    assert result["computed_token_work"] == 251
    assert result["model_executed_steps"] == 4
    assert result["preemptions"] == 1
    assert result["step_budget_overruns"] == 1
    assert result["phases"]["0.03"]["prefill_chunks"]["count"] == 4


def test_xeoverse_analysis_refuses_to_call_partial_campaign_complete(tmp_path):
    from benchmarks.satellite.analyze_xeoverse_scheduler import summarize
    from benchmarks.satellite.experiment_utils import digest

    plan = {"jobs": [{"name": "missing", "args": ["scheduler"]}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan | {"sha256": digest(plan)}))
    with pytest.raises(ValueError, match="Incomplete campaign"):
        summarize(tmp_path)
    result = summarize(tmp_path, allow_partial=True)
    assert result["complete"] is False
    assert result["pending"] == ["missing"]


def test_saved_workload_cannot_drop_misses_or_change_generation_limits():
    from benchmarks.satellite.analyze_xeoverse_scheduler import validate_saved_requests

    original = trace()
    result = {"requests": [original["requests"][0] | {"tokens": [1]}]}
    validate_saved_requests(original, result)
    with pytest.raises(ValueError, match="denominator"):
        validate_saved_requests(original, {"requests": []})
    result["requests"][0]["tokens"] = list(range(17))
    with pytest.raises(ValueError, match="generation allowance"):
        validate_saved_requests(original, result)
    result["requests"][0]["output_tokens"] = 17
    with pytest.raises(ValueError, match="metadata"):
        validate_saved_requests(original, result)


def test_shape_workloads_preserve_exact_lengths_and_paired_mixed_denominator():
    from collections import Counter

    from benchmarks.satellite.shape_utils import SHAPES, case_plan, offered_requests

    assert SHAPES == {
        "SC-SO": (1000, 256),
        "SC-LO": (1000, 2000),
        "LC-SO": (16500, 256),
        "LC-LO": (16500, 2000),
    }
    assert max(a + b for a, b in SHAPES.values()) == 18500
    for repetition in range(3):
        plan = case_plan(repetition)
        assert len(plan) == 10
        assert len({(c["scenario"], c["workload"]) for c in plan}) == 10
        mixed = offered_requests("mixed", repetition)
        assert mixed == offered_requests("mixed", repetition)
        assert Counter(r["shape"] for r in mixed) == dict.fromkeys(SHAPES, 2)
        assert len({r["id"] for r in mixed}) == 8
        assert all(r["arrival_s"] == 0 and r["priority"] == 0 for r in mixed)
        assert all("deadline_s" not in r for r in mixed)
    with pytest.raises(ValueError, match="Unknown"):
        offered_requests("unknown", 0)


def token_receipts():
    return {
        "arrival_s": 0,
        "output_tokens": 4,
        "tokens": [10, 20, 30, 40],
        "deliveries": [
            {"received_s": t, "count": 1, "offset": n}
            for n, t in enumerate((2, 3, 4, 5), 1)
        ],
        "outcome": "completed",
        "finish_s": 5,
    }


def test_tpot_excludes_first_token_wait_and_never_fills_missing_tokens():
    from benchmarks.satellite.shape_utils import latency_metrics

    row = token_receipts()
    result = latency_metrics(row)
    assert result["ttft_s"] == 2
    assert result["tpot_s"] == 1
    assert result["completion_latency_s"] == 5
    assert result["itl_p95_s"] == 1
    row["outcome"] = "watchdog_expired"
    row["finish_s"] = None
    row["tokens"].pop()
    row["deliveries"].pop()
    result = latency_metrics(row)
    assert result["ttft_s"] == 2
    assert result["tpot_s"] is None
    assert result["completion_latency_s"] is None
    assert result["completed"] is False


def test_chunked_receipts_do_not_invent_individual_token_latencies():
    from benchmarks.satellite.shape_utils import latency_metrics

    row = token_receipts()
    row["deliveries"] = [
        {"received_s": 2, "count": 2, "offset": 2},
        {"received_s": 5, "count": 2, "offset": 4},
    ]
    result = latency_metrics(row)
    assert result["tpot_s"] == 1
    assert result["itl_p95_s"] is None
    assert result["receipt_granularity"] == "chunks"


@pytest.mark.parametrize(
    "field,value", [("count", 0), ("offset", 2), ("received_s", float("nan"))]
)
def test_invalid_receipt_logs_cannot_produce_latency_metrics(field, value):
    from benchmarks.satellite.shape_utils import latency_metrics

    row = token_receipts()
    row["deliveries"][0][field] = value
    with pytest.raises(ValueError):
        latency_metrics(row)
    row = token_receipts()
    row["output_tokens"] = 3
    with pytest.raises(ValueError, match="limit"):
        latency_metrics(row)


def test_recovery_trace_uses_eclipse_exit_bracket_without_forced_pause():
    from benchmarks.satellite.shape_utils import operating_trace

    geometry = {
        "transitions": [
            {"offset_s": 100, "bracket_s": [99.999, 100.001], "eclipsed_after": False},
            {"offset_s": 1000, "eclipsed_after": True},
        ],
        "segments": {"sunlit": {"start_offset_s": 220}},
    }
    result = operating_trace(geometry, "recovery", 300)
    assert result["windows"] == [[0, 300]]
    assert result["orbital_start_offset_s"] > 100.001
    assert [s["max_step_s"] for s in result["step_limits"]] == [0.03, 0.05, 0.1]
    assert result["step_limits"][0]["end_s"] == pytest.approx(9.998)
    assert operating_trace(geometry, "sunlit", 300)["step_limits"] == [
        {"start_s": 0.0, "end_s": 300.0, "max_step_s": 0.1}
    ]
    with pytest.raises(ValueError, match="remain sunlit"):
        operating_trace(geometry, "recovery", 1000)


def test_shapes_reject_insufficient_context_before_loading_gpu(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/satellite/run_shape_experiments.py"
    )
    output = tmp_path / "run"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "calibrate",
            "--source",
            str(tmp_path),
            "--output",
            str(output),
            "--max-model-len",
            "18499",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert "18500" in process.stderr
    assert not output.exists()


def test_shape_analysis_recomputes_metrics_and_rejects_changed_output():
    from benchmarks.satellite.analyze_shape_experiments import validate_case_requests
    from benchmarks.satellite.shape_utils import latency_metrics

    row = token_receipts() | {"id": "r", "prompt_tokens": 1000}
    row["metrics"] = latency_metrics(row)
    metadata = {
        k: row[k] for k in ("id", "arrival_s", "output_tokens", "prompt_tokens")
    }
    trace = {"requests": [metadata]}
    result: dict[str, Any] = {
        "requests": [row],
        "all_completed": True,
        "all_outputs_match_reference": True,
    }
    reference = {"1000": {"tokens": [10, 20, 30, 40]}}
    validate_case_requests(result, trace, reference)
    invalid = deepcopy(result)
    invalid["requests"][0]["metrics"]["tpot_s"] = 0.001
    with pytest.raises(ValueError, match="actual receipts"):
        validate_case_requests(invalid, trace, reference)
    invalid = deepcopy(result)
    invalid["requests"][0]["tokens"][1] = 123
    with pytest.raises(ValueError, match="stock reference"):
        validate_case_requests(invalid, trace, reference)
    invalid = deepcopy(result)
    invalid["requests"] = []
    with pytest.raises(ValueError, match="denominator"):
        validate_case_requests(invalid, trace, reference)
    invalid = deepcopy(result)
    invalid["requests"][0]["outcome"] = "watchdog_expired"
    invalid["requests"][0]["metrics"] = latency_metrics(invalid["requests"][0])
    with pytest.raises(ValueError, match="Incomplete"):
        validate_case_requests(invalid, trace, reference)


def test_shape_analysis_cross_checks_receipts_against_raw_engine_events():
    from benchmarks.satellite.analyze_shape_experiments import validate_event_receipts

    row = token_receipts() | {"id": "r"}
    result = {"requests": [row], "monotonic_start": 0, "step_calls": 4}
    events: list[dict[str, Any]] = [
        {
            "start_mono": t - 0.5,
            "end_mono": t,
            "elapsed_s": 0.5,
            "delivered_s": t,
            "step_limit": None,
            "outputs": [
                {
                    "id": "r",
                    "tokens": [token],
                    "error": None,
                    "finish_reason": "length" if t == 5 else None,
                }
            ],
        }
        for t, token in zip((2, 3, 4, 5), row["tokens"])
    ]
    assert list(validate_event_receipts(iter(events), result, {})) == events
    damaged = deepcopy(events)
    damaged[0]["delivered_s"] += 0.1
    with pytest.raises(ValueError, match="raw engine log"):
        list(validate_event_receipts(iter(damaged), result, {}))
    with pytest.raises(ValueError, match="missing or extra"):
        list(validate_event_receipts(iter(events[:-1]), result, {}))
