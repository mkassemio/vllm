# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA release gates: real LLM entry point, model weights and KV continuation.

Run on a Linux CUDA host. These tests deliberately do not skip on CPU hosts.
"""

import json
import time

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.satellite.contracts import Forecast, OperatingWindow
from vllm.satellite.state import RequestSnapshot

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def orbital_config(tmp_path):
    path = tmp_path / "forecast.json"
    now = time.time()
    forecast = Forecast(
        1,
        now,
        now + 1800,
        (OperatingWindow(now, now + 1800, 1000, 1_000_000, 10, 32, 4),),
    )
    path.write_text(json.dumps(forecast.to_dict()))
    return {
        "orbital": {
            "forecast_path": str(path),
            "profile": {
                "fixed_step_s": 0.001,
                "prefill_token_s": 0.001,
                "decode_token_s": 0.001,
                "power_w": 100,
            },
        }
    }


def arguments():
    return dict(
        revision=MODEL_REVISION,
        enforce_eager=True,
        async_scheduling=False,
        enable_chunked_prefill=True,
        max_model_len=256,
        max_num_batched_tokens=32,
        max_num_seqs=4,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.2,
        seed=42,
    )


def refresh_forecast(tmp_path):
    path = tmp_path / "forecast.json"
    forecast = json.loads(path.read_text())
    forecast["version"] += 1
    forecast["issued_s"] = time.time()
    path.write_text(json.dumps(forecast))


def test_predictive_scheduler_matches_stock_llm(vllm_runner, tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    prompts = ["The capital of France is", "Explain satellite inference briefly."]
    with vllm_runner(MODEL, **arguments()) as runner:
        reference = runner.generate_greedy_logprobs(prompts, 16, 5)
    with vllm_runner(
        MODEL,
        **arguments(),
        scheduler_cls="vllm.satellite.scheduler.SatelliteScheduler",
        additional_config=orbital_config(tmp_path),
    ) as runner:
        refresh_forecast(tmp_path)
        outputs = runner.generate_greedy_logprobs(prompts, 16, 5)
        status = runner.llm.llm_engine.engine_core.orbital_control("status", {})
        assert status["executed_steps"] > 0
        assert sum(n for _, n in status["plan"]["tokens"]) <= 32
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=outputs,
        name_0="stock",
        name_1="orbital",
        always_check_logprobs=True,
    )


@pytest.mark.parametrize("use_kv", [False, True])
def test_checkpoint_restore_continues_real_model_tokens(
    vllm_runner, tmp_path, monkeypatch, use_kv
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        MODEL,
        **arguments(),
        scheduler_cls="vllm.satellite.scheduler.SatelliteScheduler",
        additional_config=orbital_config(tmp_path),
    ) as runner:
        refresh_forecast(tmp_path)
        llm = runner.llm
        prompt = "The capital of France is"
        params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
        # generate() changes its parameters to FINAL_ONLY; the stepped request
        # needs cumulative outputs so it can be paused before completion.
        expected = list(llm.generate([prompt], params.clone())[0].outputs[0].token_ids)
        engine = llm.llm_engine
        request_id = engine.add_request("recover", prompt, params)
        partial: list[int] = []
        deadline = time.monotonic() + 60
        while len(partial) < 3:
            assert time.monotonic() < deadline, "No progress before checkpoint"
            for output in engine.step():
                partial = list(output.outputs[0].token_ids)
        core = engine.engine_core
        checkpoint = core.orbital_control(
            "checkpoint", {"request_id": request_id, "include_kv": use_kv}
        )
        snapshot = RequestSnapshot.decode(checkpoint["snapshot"])
        assert snapshot.output_token_ids == tuple(partial)
        assert snapshot.computed_tokens > 0
        if use_kv:
            assert (
                checkpoint["images"][0]["computed_tokens"] == snapshot.computed_tokens
            )
        core.orbital_control("retire", {"request_id": request_id, "epoch": 0})
        core.orbital_control(
            "prepare",
            {
                "handoff_id": "restore",
                "snapshot_json": snapshot.encode(),
                "epoch": 1,
                "use_kv": use_kv,
            },
        )
        if use_kv:
            core.orbital_control(
                "load", {"handoff_id": "restore", "images": checkpoint["images"]}
            )
        core.orbital_control("activate", {"handoff_id": "restore", "epoch": 1})
        final = partial
        while engine.has_unfinished_requests():
            assert time.monotonic() < deadline, "No progress after restore"
            for output in engine.step():
                final = list(output.outputs[0].token_ids)
        assert final == expected
        assert len(final) == params.max_tokens
