# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU real-model numerical gate and actual communicator dispatch."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.models.utils import check_logprobs_close
from tests.satellite.test_communication import assert_rank_order_sum, plan
from tests.satellite_gpu.test_serving import (
    MODEL,
    arguments,
    orbital_config,
    refresh_forecast,
)
from vllm.satellite.communication import CommunicationExecutor


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
        assert all(stat["transport"] == "pynccl" for stat in stats)
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=outputs,
        name_0="stock TP",
        name_1=algorithm,
        always_check_logprobs=True,
    )


def _transport_worker(rank, rendezvous):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    comm = None
    try:
        control = dist.group.WORLD
        group = dist.new_group([0, 1], backend="nccl")
        comm = PyNcclCommunicator(control, device)
        assert not comm.disabled
        for transport in (None, comm):
            executor = CommunicationExecutor(group, control, plan(), transport)
            for version, mode in enumerate(("direct", "allgather", "native"), 2):
                executor.install(plan(mode, version))
                assert_rank_order_sum(executor, rank, 2, device, mode)
        with pytest.raises(ValueError, match="disagree"):
            CommunicationExecutor(group, control, plan(), comm if rank else None)
        comm.destroy()
        with pytest.raises(RuntimeError, match="disabled"):
            executor.all_reduce(torch.ones(1, device=device))
    finally:
        if comm is not None:
            comm.destroy()
        dist.destroy_process_group()


def test_cuda_transports_preserve_sum_and_reject_rank_disagreement(tmp_path):
    mp.spawn(_transport_worker, args=((tmp_path / "transport").as_uri(),), nprocs=2)
