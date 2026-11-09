# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Gloo operations, including collective plan disagreement."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.satellite.communication import CollectivePlan, CommunicationExecutor


def plan(mode="direct", version=1, world_size=2):
    return {
        "version": version,
        "world_size": world_size,
        "max_scratch_bytes": 1000000,
        "costs": [
            {"name": name, "latency_s": 0 if name == mode else 1, "seconds_per_byte": 0}
            for name in ("native", "direct", "allgather")
        ],
    }


def _worker(rank, rendezvous, world_size):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        executor = CommunicationExecutor(
            dist.group.WORLD, dist.group.WORLD, plan(world_size=world_size)
        )
        for version, mode in enumerate(("direct", "allgather", "native"), 2):
            executor.install(plan(mode, version, world_size))
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                tensor = torch.arange(16, dtype=dtype).reshape(4, 4) + rank
                original = tensor.clone()
                reference = tensor.clone()
                dist.all_reduce(reference)
                result = executor.all_reduce(tensor)
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                torch.testing.assert_close(tensor, original, rtol=0, atol=0)
        assert executor.statistics()["calls"] == {
            "direct": 3,
            "allgather": 3,
            "native": 3,
        }
        with pytest.raises(ValueError, match="disagree"):
            executor.install(plan("direct" if rank == 0 else "native", 6, world_size))
        with pytest.raises(ValueError, match="disagree"):
            executor.install(plan(version=0 if rank == 0 else 6, world_size=world_size))
        # A rejected update preserves the last agreed plan on every rank.
        assert executor.statistics()["plan_version"] == 4
        value = executor.all_reduce(torch.tensor([rank + 1.0]))
        assert value.item() == world_size * (world_size + 1) / 2
        if world_size == 4:
            subgroup = dist.new_group([1, 3])
            if rank in (1, 3):
                executor = CommunicationExecutor(subgroup, subgroup, plan())
                result = executor.all_reduce(torch.tensor([rank + 1.0]))
                assert result.item() == 6
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_real_collectives_and_agreed_plan_updates(tmp_path, world_size):
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), world_size),
        nprocs=world_size,
        join=True,
    )


def test_calibrated_crossover_and_scratch_limit():
    values = plan()
    values["costs"] = [
        {"name": "native", "latency_s": 0.01, "seconds_per_byte": 0.000001},
        {"name": "direct", "latency_s": 0.001, "seconds_per_byte": 0.00001},
    ]
    policy = CollectivePlan.from_dict(values)
    assert policy.select(100) == "direct"
    assert policy.select(10000) == "native"
    values["max_scratch_bytes"] = 1
    assert CollectivePlan.from_dict(values).select(100) == "native"
