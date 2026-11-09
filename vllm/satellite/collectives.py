# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager, out-of-place SUM collectives with explicit numerical semantics."""

import torch
import torch.distributed as dist


def gathered_sum(
    tensor: torch.Tensor, group: dist.ProcessGroup, direct: bool
) -> torch.Tensor:
    """Exchange every contribution and sum in rank order; preserve input.

    All ranks must call with identical shape, dtype, device type and order.
    FP16/BF16 accumulate in FP32. Rounding can differ from a native reduction.
    Direct exchange sends `(world_size - 1) * input_bytes` on every rank;
    one dependency round does not imply one physical network hop.
    """
    if (
        tensor.dtype
        not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or not tensor.is_contiguous()
        or not tensor.numel()
        or tensor.requires_grad
    ):
        raise ValueError("Orbital SUM requires nonempty contiguous inference floats")
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    flat = tensor.view(-1)
    gathered = torch.empty(
        (size, flat.numel()), device=tensor.device, dtype=tensor.dtype
    )
    if direct:
        gathered[rank].copy_(flat)
        operations = []
        for peer, global_rank in enumerate(dist.get_process_group_ranks(group)):
            if peer != rank:
                operations.extend(
                    [
                        dist.P2POp(dist.isend, flat, global_rank, group),
                        dist.P2POp(dist.irecv, gathered[peer], global_rank, group),
                    ]
                )
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
    else:
        dist.all_gather_into_tensor(gathered.view(-1), flat, group=group)
    dtype = (
        torch.float32
        if tensor.dtype in (torch.float16, torch.bfloat16)
        else tensor.dtype
    )
    result = gathered[0].to(dtype).clone()
    for peer in range(1, size):
        result.add_(gathered[peer])
    return result.to(tensor.dtype).view_as(tensor)
