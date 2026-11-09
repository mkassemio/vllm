# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU real-model numerical gate and actual communicator dispatch."""

import pytest

from tests.models.utils import check_logprobs_close
from tests.satellite_gpu.test_serving import (
    MODEL,
    arguments,
    orbital_config,
    refresh_forecast,
)


@pytest.mark.parametrize("algorithm", ["native", "direct", "allgather"])
def test_orbital_communication_matches_stock_tp(vllm_runner, algorithm, tmp_path):
    prompts = ["The capital of France is", "Satellite compute requires"]
    options = arguments() | {
        "tensor_parallel_size": 2,
        "disable_custom_all_reduce": True,
    }
    with vllm_runner(MODEL, **options) as runner:
        reference = runner.generate_greedy_logprobs(prompts, 16, 5)
    plan: dict = {
        "version": 1,
        "world_size": 2,
        "max_scratch_bytes": 64 * 1024 * 1024,
        "costs": [{"name": "native", "latency_s": 1, "seconds_per_byte": 0}],
    }
    if algorithm != "native":
        plan["costs"].append({"name": algorithm, "latency_s": 0, "seconds_per_byte": 0})
    with vllm_runner(
        MODEL,
        **options,
        scheduler_cls="vllm.satellite.scheduler.SatelliteScheduler",
        additional_config=orbital_config(tmp_path) | {"orbital_communication": plan},
    ) as runner:
        refresh_forecast(tmp_path)
        outputs = runner.generate_greedy_logprobs(prompts, 16, 5)
        stats = runner.llm.collective_rpc("orbital_communication_statistics")
        assert len(stats) == 2
        assert all(stat["calls"][algorithm] > 0 for stat in stats)
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=outputs,
        name_0="stock TP",
        name_1=algorithm,
        always_check_logprobs=True,
    )
