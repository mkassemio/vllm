# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real schedulers and durable journal; inject only external RPC delivery faults."""

import pytest

from tests.v1.core.utils import create_requests
from vllm.sampling_params import SamplingParams
from vllm.satellite.engine_state import EngineStateAdapter
from vllm.satellite.handoff import HandoffCoordinator
from vllm.satellite.state import StateConflict, StateJournal
from vllm.v1.outputs import ModelRunnerOutput


@pytest.fixture
def engines(scheduler_factory):
    source = EngineStateAdapter(scheduler_factory())
    target = EngineStateAdapter(scheduler_factory())
    request = create_requests(
        1, num_tokens=2, sampling_params=SamplingParams(temperature=0, max_tokens=20)
    )[0]
    source.scheduler.add_request(request)
    step = source.scheduler.schedule()
    source.scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request.request_id], sampled_token_ids=[[7]])
    )
    return source, target, request.request_id


def test_recompute_handoff_runs_real_scheduler_and_preserves_output_limit(
    tmp_path, engines
):
    source, target, request_id = engines
    journal = StateJournal(tmp_path / "journal.sqlite")
    journal.register(request_id, "source")
    coordinator = HandoffCoordinator(
        journal, {"source": source.dispatch, "target": target.dispatch}
    )
    owner = coordinator.start("move", request_id, "target")
    assert (owner.owner, owner.epoch) == ("target", 1)
    assert not source.scheduler.has_request(request_id)
    assert journal.tokens(request_id) == (7,)
    step = target.scheduler.schedule()
    assert step.num_scheduled_tokens == {request_id: 3}
    target.scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request_id], sampled_token_ids=[[8]])
    )
    assert list(target.scheduler.requests[request_id].output_token_ids) == [7, 8]
    coordinator.publish(request_id, "target", 1, (7, 8))
    assert journal.tokens(request_id, after=1) == (8,)
    with pytest.raises(StateConflict):
        coordinator.publish(request_id, "source", 0, (7, 9))
    assert coordinator.start("move", request_id, "target") == journal.ownership(
        request_id
    )
    journal.close()


@pytest.mark.parametrize(
    "lost_command", ["checkpoint", "prepare", "activate", "retire"]
)
def test_lost_rpc_reply_recovers_without_dual_execution(
    tmp_path, engines, lost_command
):
    source, target, request_id = engines
    lost = False

    def endpoint(adapter):
        def call(command, payload):
            nonlocal lost
            result = adapter.dispatch(command, payload)
            if command == lost_command and not lost:
                lost = True
                raise ConnectionError("Injected reply loss after execution")
            return result

        return call

    journal = StateJournal(tmp_path / "journal.sqlite")
    journal.register(request_id, "source")
    endpoints = {"source": endpoint(source), "target": endpoint(target)}
    coordinator = HandoffCoordinator(journal, endpoints)
    with pytest.raises(ConnectionError):
        coordinator.start("move", request_id, "target")
    # Restart the authority process with the same durable database.
    journal.close()
    journal = StateJournal(tmp_path / "journal.sqlite")
    coordinator = HandoffCoordinator(journal, endpoints)
    coordinator.start("move", request_id, "target")
    assert not source.scheduler.has_request(request_id)
    assert target.scheduler.schedule().total_num_scheduled_tokens == 3
    assert journal.tokens(request_id) == (7,)
    journal.close()


def test_abort_is_idempotent_and_discards_inactive_destination(tmp_path, engines):
    source, target, request_id = engines

    def unavailable_after_prepare(command, payload):
        result = target.dispatch(command, payload)
        if command == "prepare":
            raise ConnectionError("Preparation reply lost")
        return result

    journal = StateJournal(tmp_path / "journal.sqlite")
    journal.register(request_id, "source")
    coordinator = HandoffCoordinator(
        journal, {"source": source.dispatch, "target": unavailable_after_prepare}
    )
    with pytest.raises(ConnectionError):
        coordinator.start("move", request_id, "target")
    assert source.scheduler.schedule().total_num_scheduled_tokens == 0
    assert target.scheduler.schedule().total_num_scheduled_tokens == 0
    owner = coordinator.abort("move")
    assert coordinator.abort("move") == owner
    assert not target.scheduler.has_request(request_id)
    assert source.scheduler.schedule().total_num_scheduled_tokens == 1
    journal.close()


def test_destination_restart_replays_latest_durable_output(
    tmp_path, engines, scheduler_factory
):
    source, target, request_id = engines
    journal = StateJournal(tmp_path / "journal.sqlite")
    journal.register(request_id, "source")
    coordinator = HandoffCoordinator(
        journal, {"source": source.dispatch, "target": target.dispatch}
    )
    coordinator.start("move", request_id, "target")
    step = target.scheduler.schedule()
    target.scheduler.update_from_output(
        step, ModelRunnerOutput(req_ids=[request_id], sampled_token_ids=[[8]])
    )
    coordinator.publish(request_id, "target", 1, (7, 8))
    restarted = EngineStateAdapter(scheduler_factory())
    coordinator = HandoffCoordinator(
        journal, {"source": source.dispatch, "target": restarted.dispatch}
    )
    coordinator.advance("move")
    request = restarted.scheduler.requests[request_id]
    assert list(request.output_token_ids) == [7, 8]
    assert request.num_computed_tokens == 0
    assert request.max_tokens == target.scheduler.requests[request_id].max_tokens
    assert restarted.scheduler.schedule().total_num_scheduled_tokens == 3
    journal.close()
