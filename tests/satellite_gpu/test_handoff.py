# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Migrate between independent EngineCore processes on separate physical GPUs."""

import time

import pytest

from tests.satellite_gpu.test_serving import (
    MODEL,
    arguments,
    orbital_config,
    refresh_forecast,
)
from vllm import SamplingParams
from vllm.satellite.handoff import HandoffCoordinator
from vllm.satellite.state import StateJournal


@pytest.mark.parametrize("mode", ["recompute", "transfer"])
def test_cross_gpu_handoff_preserves_exact_tokens(
    vllm_runner, tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    source_path, target_path = tmp_path / "source", tmp_path / "target"
    source_path.mkdir()
    target_path.mkdir()
    options = arguments() | {
        "scheduler_cls": "vllm.satellite.scheduler.SatelliteScheduler"
    }
    with (
        vllm_runner(
            MODEL,
            **options,
            device_ids=[0],
            additional_config=orbital_config(source_path),
        ) as source,
        vllm_runner(
            MODEL,
            **options,
            device_ids=[1],
            additional_config=orbital_config(target_path),
        ) as target,
    ):
        refresh_forecast(source_path)
        refresh_forecast(target_path)
        params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        prompt = "Explain why a satellite needs reliable computer memory."
        expected = tuple(
            source.llm.generate([prompt], params.clone())[0].outputs[0].token_ids
        )
        source_engine = source.llm.llm_engine
        source_client = source_engine.engine_core
        target_client = target.llm.llm_engine.engine_core
        request_id = source_engine.add_request("cross-gpu", prompt, params)
        deadline = time.monotonic() + 60
        while True:
            assert time.monotonic() < deadline, "Source produced no tokens"
            outputs = source_engine.step()
            if any(output.outputs[0].token_ids for output in outputs):
                break
        journal = StateJournal(tmp_path / "journal.sqlite")
        try:
            journal.register(request_id, "source")
            coordinator = HandoffCoordinator(
                journal,
                {
                    "source": source_client.orbital_control,
                    "target": target_client.orbital_control,
                },
            )
            owner = coordinator.start("move", request_id, "target", mode=mode)
            tokens = list(journal.tokens(request_id))
            assert 0 < len(tokens) < params.max_tokens
            assert (owner.owner, owner.epoch) == ("target", 1)
            finished = False
            while not finished:
                assert time.monotonic() < deadline, "Destination did not complete"
                for output in target_client.get_output().outputs:
                    assert output.request_id == request_id
                    assert output.error is None
                    tokens.extend(output.new_token_ids)
                    finished = output.finish_reason is not None
                    coordinator.publish(
                        request_id, "target", 1, tuple(tokens), finished=finished
                    )
            assert journal.tokens(request_id) == expected
            assert len(tokens) == params.max_tokens
            assert coordinator.advance("move").phase == "FINISHED"
        finally:
            journal.close()
