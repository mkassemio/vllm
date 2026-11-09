# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture and restore the supported vLLM request contract without retokenizing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import msgspec

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.satellite.state import RequestSnapshot, canonical_json
from vllm.v1.request import Request

if TYPE_CHECKING:
    from collections.abc import Callable


def check_replay_supported(request: Request) -> None:
    params = request.sampling_params
    if params is None or params.sampling_type != SamplingType.GREEDY:
        raise ValueError("Orbital replay currently requires greedy sampling")
    if (
        request.mm_features
        or request.prompt_embeds is not None
        or request.lora_request is not None
        or request.resumable
        or request.use_structured_output
        or request.num_in_flight_tokens
        or request.spec_token_ids
        or params.n != 1
        or params.stop
        or params.min_tokens
        or params.logprobs is not None
        or params.prompt_logprobs is not None
        or params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.repetition_penalty != 1
        or params.bad_words
        or params.thinking_token_budget is not None
        or params.trace_decode_token_ids is not None
        or params.repetition_detection is not None
        or params.watermarking
        or set(params.extra_args or {}) - {"orbital"}
    ):
        raise ValueError("Request uses state not yet supported by orbital replay")


def capture_request(request: Request, fingerprint: str) -> RequestSnapshot:
    check_replay_supported(request)
    assert request.sampling_params is not None
    assert request.prompt_token_ids is not None
    metadata = {
        "sampling": msgspec.to_builtins(request.sampling_params),
        "eos_token_id": request.sampling_params.eos_token_id,
        "priority": request.priority,
        "arrival_time": request.arrival_time,
        "cache_salt": request.cache_salt,
    }
    return RequestSnapshot(
        request_id=request.request_id,
        model_fingerprint=fingerprint,
        prompt_token_ids=tuple(request.prompt_token_ids),
        output_token_ids=tuple(request.output_token_ids),
        computed_tokens=request.num_computed_tokens,
        max_output_tokens=request.max_tokens,
        sampling_json=canonical_json(metadata),
    )


def restore_request(
    snapshot: RequestSnapshot,
    client_index: int = 0,
    block_hasher: Callable | None = None,
) -> Request:
    metadata = json.loads(snapshot.sampling_json)
    if set(metadata) != {
        "sampling",
        "eos_token_id",
        "priority",
        "arrival_time",
        "cache_salt",
    }:
        raise ValueError("Invalid request replay metadata")
    params = msgspec.convert(metadata["sampling"], type=SamplingParams)
    if params.max_tokens != snapshot.max_output_tokens:
        raise ValueError("Snapshot output limit does not match sampling parameters")
    params.update_from_generation_config({}, metadata["eos_token_id"])
    request = Request(
        snapshot.request_id,
        list(snapshot.prompt_token_ids),
        params,
        pooling_params=None,
        client_index=client_index,
        arrival_time=metadata["arrival_time"],
        priority=metadata["priority"],
        cache_salt=metadata["cache_salt"],
        block_hasher=block_hasher,
    )
    request.append_output_token_ids(list(snapshot.output_token_ids))
    check_replay_supported(request)
    # Computed position remains zero until replay completes or KV import commits.
    return request
