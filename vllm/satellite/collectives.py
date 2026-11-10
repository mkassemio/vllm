# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager, out-of-place SUM collectives with explicit numerical semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def _exchange(
    tensor: torch.Tensor,
    receives: list[tuple[int, torch.Tensor]],
    group: dist.ProcessGroup,
    pynccl_comm: PyNcclCommunicator | None,
) -> None:
    if pynccl_comm is not None:
        from vllm.utils.torch_utils import current_stream

        stream = current_stream()
        pynccl_comm.group_start()
        try:
            for peer, target in receives:
                pynccl_comm.send(tensor, peer, stream)
                pynccl_comm.recv(target, peer, stream)
        finally:
            pynccl_comm.group_end()
        # NCCL and the reduction use the caller's stream. No cross-stream
        # events or Work objects are needed; buffers stay live through enqueue.
    else:
        ranks = dist.get_process_group_ranks(group)
        operations = []
        for peer, target in receives:
            operations.extend(
                [
                    dist.P2POp(dist.isend, tensor, ranks[peer], group),
                    dist.P2POp(dist.irecv, target, ranks[peer], group),
                ]
            )
        for work in dist.batch_isend_irecv(operations):
            work.wait()


def gathered_sum(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    direct: bool,
    pynccl_comm: PyNcclCommunicator | None = None,
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
    if pynccl_comm is not None:
        if pynccl_comm.disabled:
            raise RuntimeError("The agreed PyNCCL transport is disabled")
        if (
            pynccl_comm.device != tensor.device
            or pynccl_comm.world_size != size
            or pynccl_comm.rank != rank
        ):
            raise ValueError("PyNCCL transport does not match the tensor/group")
    if size == 1:
        return tensor.clone()
    flat = tensor.view(-1)
    if direct and size == 2:
        received = torch.empty_like(flat)
        _exchange(flat, [(1 - rank, received)], group, pynccl_comm)
        # One binary add rounds once, just like the FP32 accumulator followed
        # by conversion. Avoid copying our own input and materializing casts.
        lhs, rhs = (flat, received) if rank == 0 else (received, flat)
        return torch.add(lhs, rhs).view_as(tensor)
    gathered = torch.empty(
        (size, flat.numel()), device=tensor.device, dtype=tensor.dtype
    )
    if direct:
        gathered[rank].copy_(flat)
        receives = [(peer, gathered[peer]) for peer in range(size) if peer != rank]
        _exchange(flat, receives, group, pynccl_comm)
    elif pynccl_comm is not None:
        pynccl_comm.all_gather(gathered.view(-1), flat)
    else:
        dist.all_gather_into_tensor(gathered.view(-1), flat, group=group)
    if size == 2:
        return torch.add(gathered[0], gathered[1]).view_as(tensor)
    dtype = (
        torch.float32
        if tensor.dtype in (torch.float16, torch.bfloat16)
        else tensor.dtype
    )
    result = gathered[0].to(dtype, copy=True)
    for peer in range(1, size):
        result.add_(gathered[peer])
    return result.to(tensor.dtype).view_as(tensor)
