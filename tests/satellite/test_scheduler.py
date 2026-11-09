# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real scheduler/allocator integration; no GPU model is substituted or mocked."""

import json
import time

import pytest

from tests.v1.core.utils import create_requests
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput


def test_token_limits_are_applied_before_allocator_and_progress(scheduler):
    request = create_requests(1, num_tokens=10)[0]
    scheduler.add_request(request)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {request.request_id: 3}
    assert request.num_computed_tokens == 3
    scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request.request_id], sampled_token_ids=[[]])
    )
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {request.request_id: 3}
    assert request.num_computed_tokens == 6


def test_suspension_preserves_progress_and_resume_continues(scheduler):
    request = create_requests(1, num_tokens=2)[0]
    scheduler.add_request(request)
    step = scheduler.schedule()
    scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request.request_id], sampled_token_ids=[[7]])
    )
    scheduler.suspend_request(request.request_id)
    paused = scheduler.schedule()
    assert paused.total_num_scheduled_tokens == 0
    assert request.num_computed_tokens == 2
    scheduler.resume_request(request.request_id)
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {request.request_id: 1}
    assert list(request.output_token_ids) == [7]


def test_expired_deadline_returns_terminal_error_instead_of_hanging(scheduler):
    request = create_requests(1)[0]
    request.sampling_params.extra_args = {"orbital": {"deadline_s": time.time() - 1}}
    scheduler.add_request(request)
    step = scheduler.schedule()
    assert step.total_num_scheduled_tokens == 0
    outputs = scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[], sampled_token_ids=[])
    )
    output = outputs[0].outputs[0]
    assert output.finish_reason == FinishReason.ERROR
    assert output.error.code == "orbital_deadline_exceeded"
    assert not scheduler.has_request(request.request_id)


def test_cancellation_of_suspended_request_releases_memory(scheduler):
    from vllm.v1.request import RequestStatus

    request = create_requests(1, num_tokens=2)[0]
    scheduler.add_request(request)
    initial_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    step = scheduler.schedule()
    scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request.request_id], sampled_token_ids=[[7]])
    )
    scheduler.suspend_request(request.request_id)
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == initial_free


def test_forecast_refresh_preserves_energy_spent_since_issue(scheduler):
    request = create_requests(1, num_tokens=10)[0]
    scheduler.add_request(request)
    step = scheduler.schedule()
    scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request.request_id], sampled_token_ids=[[]])
    )
    first = scheduler.orbital_status()["reserved_energy_j"]
    path = scheduler.orbital_reader.path
    forecast = json.loads(path.read_text())
    forecast["version"] += 1
    path.write_text(json.dumps(forecast))
    scheduler.schedule()
    assert scheduler.orbital_status()["reserved_energy_j"] == pytest.approx(2 * first)


def test_frozen_request_does_not_expire_until_resumed(scheduler):
    request = create_requests(1)[0]
    request.sampling_params.extra_args = {"orbital": {"deadline_s": time.time() - 1}}
    scheduler.add_request(request)
    scheduler.suspend_request(request.request_id)
    step = scheduler.schedule()
    assert step.total_num_scheduled_tokens == 0
    assert scheduler.has_request(request.request_id)
    scheduler.resume_request(request.request_id)
    scheduler.schedule()
    assert not scheduler.has_request(request.request_id)


def test_frozen_request_keeps_its_kv_when_active_slots_are_full(scheduler_factory):
    scheduler = scheduler_factory(max_num_seqs=1)
    first, second = create_requests(2, num_tokens=2)
    scheduler.add_request(first)
    step = scheduler.schedule()
    scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[first.request_id], sampled_token_ids=[[7]])
    )
    scheduler.suspend_request(first.request_id)
    blocks = scheduler.kv_cache_manager.get_block_ids(first.request_id)
    scheduler.add_request(second)
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert first.num_computed_tokens == 2
    assert scheduler.kv_cache_manager.get_block_ids(first.request_id) == blocks
    scheduler.resume_request(first.request_id)
    assert scheduler.schedule().num_scheduled_tokens == {first.request_id: 1}


def test_pre_startup_forecast_cannot_restore_old_energy_allowance(scheduler):
    path = scheduler.orbital_reader.path
    forecast = json.loads(path.read_text())
    forecast["issued_s"] = scheduler.orbital_status()["engine_started_s"] - 1
    path.write_text(json.dumps(forecast))
    request = create_requests(1, num_tokens=2)[0]
    scheduler.add_request(request)
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert "predates" in scheduler.orbital_status()["forecast_error"]
    forecast["version"] += 1
    forecast["issued_s"] = time.time()
    path.write_text(json.dumps(forecast))
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 2}
