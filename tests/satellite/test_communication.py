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
        "max_scratch_bytes": 16 * 1024 * 1024,
        "costs": [
            {"name": name, "latency_s": 0 if name == mode else 1, "seconds_per_byte": 0}
            for name in ("native", "direct", "allgather")
        ],
    }


def assert_rank_order_sum(executor, rank, world_size, device, mode):
    """Check rounding, cancellation, odd lengths, storage offsets and ownership."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for size in (1, 257, 114688):
            inputs = [
                torch.randn(size, generator=torch.Generator().manual_seed(123 + peer))
                .mul(10)
                .to(dtype)
                for peer in range(world_size)
            ]
            for peer, value in enumerate(inputs):
                value[0] = (32768, 1, -32768, 0)[peer]
                if size > 1:
                    limits = torch.finfo(dtype)
                    value[1:9] = torch.tensor(
                        [
                            limits.tiny,
                            limits.tiny * limits.eps,
                            -limits.tiny,
                            limits.max,
                            -limits.max,
                            1,
                            limits.eps / 2,
                            -limits.eps / 2,
                        ],
                        dtype=dtype,
                    )
            accumulator_dtype = (
                torch.float64 if dtype == torch.float64 else torch.float32
            )
            expected = inputs[0].to(accumulator_dtype, copy=True)
            for value in inputs[1:]:
                expected.add_(value)
            # Exercise a contiguous view whose storage does not start at zero.
            storage = torch.cat([torch.zeros(3, dtype=dtype), inputs[rank]]).to(device)
            tensor = storage[3:].view(1, -1)
            original = tensor.clone()
            calls = executor.statistics()["calls"].get(mode, 0)
            result = executor.all_reduce(tensor)
            torch.testing.assert_close(
                result.cpu(), expected.to(dtype).view_as(result), rtol=0, atol=0
            )
            torch.testing.assert_close(tensor, original, rtol=0, atol=0)
            assert result.data_ptr() != tensor.data_ptr()
            retained = result.clone()
            executor.all_reduce(tensor)
            torch.testing.assert_close(result, retained, rtol=0, atol=0)
            assert executor.statistics()["calls"][mode] == calls + 2


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
        if world_size > 1:
            with pytest.raises(ValueError, match="disagree"):
                executor.install(
                    plan("direct" if rank == 0 else "native", 6, world_size)
                )
        with pytest.raises(ValueError, match="disagree"):
            executor.install(plan(version=0 if rank == 0 else 6, world_size=world_size))
        # A rejected update preserves the last agreed plan on every rank.
        assert executor.statistics()["plan_version"] == 4
        value = executor.all_reduce(torch.tensor([rank + 1.0]))
        assert value.item() == world_size * (world_size + 1) / 2
        for version, mode in enumerate(("direct", "allgather"), 6):
            executor.install(plan(mode, version, world_size))
            assert_rank_order_sum(executor, rank, world_size, "cpu", mode)
        if world_size == 4:
            subgroup = dist.new_group([1, 3])
            if rank in (1, 3):
                executor = CommunicationExecutor(subgroup, subgroup, plan())
                result = executor.all_reduce(torch.tensor([rank + 1.0]))
                assert result.item() == 6
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
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
