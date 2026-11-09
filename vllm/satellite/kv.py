# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated portable KV images for standard dense attention cache views.

The caller must freeze a request and reserve exclusive destination blocks.
Only the standardized `[blocks, heads, tokens, channels]` cache view is accepted.
No pickle or executable checkpoint format is used.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import torch
from safetensors.torch import load, save

from vllm.satellite.contracts import identifier, integer
from vllm.satellite.state import StateConflict


def _validate_blocks(
    caches: Mapping[str, torch.Tensor],
    block_ids: Sequence[int],
    computed_tokens: int,
    block_size: int,
) -> int:
    integer(computed_tokens, "computed_tokens", minimum=1)
    integer(block_size, "block_size", minimum=1)
    count = math.ceil(computed_tokens / block_size)
    if not caches or len(block_ids) != count or len(set(block_ids)) != count:
        raise ValueError("KV image requires exactly the distinct computed blocks")
    for block in block_ids:
        integer(block, "block_id")
    for name, cache in caches.items():
        identifier(name, "layer_name")
        if cache.ndim != 4 or cache.shape[2] != block_size:
            raise ValueError("KV image requires standardized dense attention views")
        if cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Only unquantized floating-point KV images are supported")
        if max(block_ids) >= cache.shape[0]:
            raise ValueError("KV block ID exceeds the physical cache")
    return count


def export_kv(
    caches: Mapping[str, torch.Tensor],
    block_ids: Sequence[int],
    computed_tokens: int,
    block_size: int,
    fingerprint: str,
    rank: int,
    world_size: int,
    max_bytes: int = 268_435_456,
) -> dict:
    """Copy a frozen request's computed KV prefix to a checked safetensors image."""
    identifier(fingerprint, "fingerprint")
    integer(world_size, "world_size", minimum=1)
    integer(rank, "rank")
    integer(max_bytes, "max_bytes", minimum=1)
    if rank >= world_size:
        raise ValueError("Rank exceeds world size")
    count = _validate_blocks(caches, block_ids, computed_tokens, block_size)
    size = sum(
        count * math.prod(cache.shape[1:]) * cache.element_size()
        for cache in caches.values()
    )
    if size > max_bytes:
        raise ValueError("KV image exceeds configured transfer budget")
    tensors = {}
    for name, cache in caches.items():
        indices = torch.tensor(block_ids, device=cache.device, dtype=torch.int64)
        value = cache.index_select(0, indices).to("cpu").contiguous()
        # A partially computed block can contain another request's old values.
        tail = computed_tokens % block_size
        if tail:
            value[-1, :, tail:, :] = 0
        tensors[name] = value
    payload = save(tensors)
    # Allow bounded safetensors metadata in addition to tensor payload bytes.
    if len(payload) > max_bytes + 1_048_576:
        raise ValueError("Serialized KV image exceeds transfer budget")
    return {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "rank": rank,
        "world_size": world_size,
        "computed_tokens": computed_tokens,
        "block_size": block_size,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
    }


def import_kv(
    image: dict,
    caches: Mapping[str, torch.Tensor],
    block_ids: Sequence[int],
    fingerprint: str,
    rank: int,
    world_size: int,
    max_bytes: int = 268_435_456,
) -> int:
    """Validate all layers before touching exclusive, inactive destination blocks."""
    if set(image) != {
        "schema_version",
        "fingerprint",
        "rank",
        "world_size",
        "computed_tokens",
        "block_size",
        "sha256",
        "payload",
    }:
        raise ValueError("Unknown or missing KV image fields")
    if type(image["schema_version"]) is not int or image["schema_version"] != 1:
        raise ValueError("Unsupported KV image schema")
    if (image["fingerprint"], image["rank"], image["world_size"]) != (
        fingerprint,
        rank,
        world_size,
    ):
        raise StateConflict("KV image model or tensor-parallel layout is incompatible")
    payload = image["payload"]
    if not isinstance(payload, bytes) or len(payload) > max_bytes + 1_048_576:
        raise ValueError("KV payload exceeds transfer budget")
    if hashlib.sha256(payload).hexdigest() != image["sha256"]:
        raise StateConflict("KV payload checksum mismatch")
    count = _validate_blocks(
        caches, block_ids, image["computed_tokens"], image["block_size"]
    )
    tensors = load(payload)
    if tensors.keys() != caches.keys():
        raise StateConflict("KV layer set is incompatible")
    for name, value in tensors.items():
        expected = (count, *caches[name].shape[1:])
        if tuple(value.shape) != expected or value.dtype != caches[name].dtype:
            raise StateConflict(f"KV shape or dtype mismatch for {name}")
    # Stage all tensors first. If allocation or any copy fails, the caller must
    # discard the reserved destination; it must never activate a partial import.
    staged = {name: value.to(caches[name].device) for name, value in tensors.items()}
    for name, value in staged.items():
        indices = torch.tensor(block_ids, device=value.device, dtype=torch.int64)
        caches[name].index_copy_(0, indices, value)
    devices = {cache.device for cache in caches.values() if cache.device.type != "cpu"}
    for device in devices:
        torch.accelerator.synchronize(device)
    return image["computed_tokens"]
