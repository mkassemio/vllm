# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams
from vllm.satellite.replay import capture_request, restore_request
from vllm.satellite.state import RequestSnapshot
from vllm.v1.request import Request


def test_replay_preserves_original_prompt_history_and_generation_limit():
    params = SamplingParams(
        temperature=0,
        max_tokens=9,
        stop_token_ids=[15],
        extra_args={"orbital": {"deadline_s": 1000.0}},
    )
    params.update_from_generation_config({}, 14)
    original = Request(
        "request",
        [1, 2, 3],
        params,
        None,
        arrival_time=10,
        priority=-1,
        cache_salt="tenant",
    )
    original.append_output_token_ids([4, 5])
    snapshot = RequestSnapshot.decode(capture_request(original, "model").encode())
    restored = restore_request(snapshot, client_index=2)
    assert restored.prompt_token_ids == original.prompt_token_ids
    assert list(restored.output_token_ids) == [4, 5]
    assert restored.prefill_token_ids == [1, 2, 3, 4, 5]
    assert restored.num_computed_tokens == 0
    assert restored.max_tokens == 9
    assert restored.client_index == 2
    assert restored.sampling_params.all_stop_token_ids == {14, 15}
    assert restored.sampling_params.extra_args == params.extra_args
    assert restored.arrival_time == 10
    assert restored.priority == -1
    assert restored.cache_salt == "tenant"


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": 1},
        {"stop": ["stop"]},
        {"min_tokens": 2},
        {"logprobs": 1},
        {"presence_penalty": 1},
        {"thinking_token_budget": 4},
        {"extra_args": {"custom_processor": True}},
    ],
)
def test_unsupported_sampler_state_is_rejected_before_snapshot(extra):
    params = SamplingParams(**({"temperature": 0, "max_tokens": 9} | extra))
    request = Request("request", [1, 2], params, None)
    with pytest.raises(ValueError):
        capture_request(request, "model")
