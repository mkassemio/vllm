# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policies must respect resource limits before a GPU batch is allocated."""

import json
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vllm.satellite.contracts import (
    CostProfile,
    Forecast,
    OperatingWindow,
    RequestWork,
)
from vllm.satellite.forecast import ForecastReader
from vllm.satellite.policy import PredictivePolicy


def forecast(**changes):
    window = OperatingWindow(100, 110, 100, 100, 1, 100, 4)
    return Forecast(1, 100, 110, (replace(window, **changes),))


def request(name="a", **changes):
    return replace(RequestWork(name, 100, 10, 20, True), **changes)


def policy():
    return PredictivePolicy(CostProfile(0.01, 0.01, 0.1, 50, 1), guard_s=0.01)


def test_end_of_window_limits_work_before_it_is_admitted():
    result = policy().plan([request()], forecast(), 109.95, 100)
    assert 0 < result.token_limits["a"] < 10
    assert result.estimated_s <= 110 - 109.95 - 0.01


@pytest.mark.parametrize("now", [99, 110, 111])
def test_no_execution_outside_valid_forecast(now):
    result = policy().plan([request()], forecast(), now, 100)
    assert result.tokens == ()
    assert result.estimated_energy_j == 0


def test_no_execution_when_power_is_insufficient():
    assert policy().plan([request()], forecast(max_power_w=49), 100, 100).tokens == ()


def test_energy_already_reserved_is_not_spent_twice():
    p = policy()
    f = forecast(energy_j=5)
    first = p.plan([request()], f, 100, 100)
    second = p.plan([request()], f, 100, 100, first.estimated_energy_j)
    assert first.estimated_energy_j + second.estimated_energy_j <= 5


def test_urgent_request_deadline_bounds_the_entire_batch():
    requests = [request("urgent", deadline_s=100.05), request("batch")]
    result = policy().plan(requests, forecast(), 100, 100)
    assert "urgent" in result.token_limits
    assert result.estimated_s <= 0.04 + 1e-12


def test_suspended_request_cannot_execute_but_other_work_can():
    result = policy().plan(
        [request("a", suspended=True), request("b")], forecast(), 100, 100
    )
    assert result.token_limits == {"b": 10}
    assert ("a", "suspended") in result.deferred


def test_aging_gives_old_batch_work_service():
    f = Forecast(1, 0, 110, (OperatingWindow(0, 110, 100, 100, 1, 100, 1),))
    result = policy().plan(
        [request("old", arrival_s=0, priority=2), request("new")], f, 100, 100
    )
    assert list(result.token_limits) == ["old"]


def test_invalid_forecast_revokes_admission_and_cannot_reset_version(tmp_path):
    path = tmp_path / "forecast.json"
    reader = ForecastReader(str(path))
    path.write_text(json.dumps(forecast().to_dict()))
    assert reader.read() == forecast()
    path.write_text(json.dumps(forecast(max_power_w=1000).to_dict()))
    assert reader.read() is None
    assert "version" in reader.error
    path.write_text("{")
    assert reader.read() is None
    path.write_text(json.dumps(replace(forecast(), version=2).to_dict()))
    assert reader.read().version == 2
    path.write_text(json.dumps(forecast().to_dict()))
    assert reader.read() is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_physical_quantities_are_rejected(value):
    with pytest.raises(ValueError):
        forecast(energy_j=value)


@given(
    tokens=st.integers(0, 100),
    energy=st.floats(min_value=0, max_value=100, allow_nan=False),
    duration=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_generated_plans_obey_all_batch_budgets(tokens, energy, duration):
    result = policy().plan(
        [request(str(i), pending_tokens=100) for i in range(6)],
        forecast(energy_j=energy, max_step_s=duration),
        100,
        tokens,
    )
    assert sum(result.token_limits.values()) <= tokens
    assert len(result.tokens) <= 4
    assert result.estimated_energy_j <= energy + 1e-12
    assert result.estimated_s <= duration + 1e-12
    assert all(n > 0 for n in result.token_limits.values())
