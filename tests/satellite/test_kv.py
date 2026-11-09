# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV transfer preserves logical order and rejects corruption before writes."""

import pytest
import torch
from safetensors.torch import load

from vllm.satellite.kv import export_kv, import_kv
from vllm.satellite.state import StateConflict


def caches():
    values = torch.arange(8 * 2 * 4 * 3, dtype=torch.float32).reshape(8, 2, 4, 3)
    return {"layer.0": values, "layer.1": values + 1000}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_kv_transfer_remaps_physical_blocks_and_erases_unused_tail(dtype):
    source = {name: value.to(dtype) for name, value in caches().items()}
    target = {name: torch.full_like(value, -1) for name, value in source.items()}
    image = export_kv(source, [5, 2], 6, 4, "model", 0, 1)
    assert import_kv(image, target, [1, 7], "model", 0, 1) == 6
    for name in source:
        torch.testing.assert_close(target[name][1], source[name][5], rtol=0, atol=0)
        torch.testing.assert_close(
            target[name][7, :, :2], source[name][2, :, :2], rtol=0, atol=0
        )
        assert torch.count_nonzero(target[name][7, :, 2:]) == 0
        assert torch.all(target[name][0] == -1)
        assert torch.all(load(image["payload"])[name][-1, :, 2:] == 0)


@pytest.mark.parametrize("change", ["checksum", "rank", "model", "layer", "dtype"])
def test_invalid_image_does_not_modify_destination(change):
    source = caches()
    target = {name: torch.full_like(value, -1) for name, value in source.items()}
    image = export_kv(source, [5], 4, 4, "model", 0, 1)
    if change == "checksum":
        image["payload"] = image["payload"][:-1] + b"X"
    elif change == "rank":
        image["rank"] = 1
    elif change == "model":
        image["fingerprint"] = "other"
    elif change == "layer":
        target.pop("layer.1")
    else:
        target["layer.1"] = target["layer.1"].to(torch.float16)
    with pytest.raises(StateConflict):
        import_kv(image, target, [1], "model", 0, 1)
    assert all(torch.all(value == -1) for value in target.values())


def test_export_budget_checked_before_copying_cache():
    with pytest.raises(ValueError, match="budget"):
        export_kv(caches(), [1], 4, 4, "model", 0, 1, max_bytes=1)


def test_duplicate_physical_destination_blocks_are_rejected():
    image = export_kv(caches(), [5, 2], 8, 4, "model", 0, 1)
    with pytest.raises(ValueError, match="distinct"):
        import_kv(image, caches(), [1, 1], "model", 0, 1)
