# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect experiment denominators, window boundaries, and calibration inputs."""

import gzip
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

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
