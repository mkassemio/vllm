# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated collective plans agreed by every member of an execution group."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from vllm.satellite.contracts import finite, integer
from vllm.satellite.state import canonical_json


@dataclass(frozen=True)
class AlgorithmCost:
    name: str
    latency_s: float
    seconds_per_byte: float

    def __post_init__(self):
        if self.name not in ("native", "direct", "allgather"):
            raise ValueError("Unknown orbital SUM algorithm")
        finite(self.latency_s, "latency_s")
        finite(self.seconds_per_byte, "seconds_per_byte")


@dataclass(frozen=True)
class CollectivePlan:
    version: int
    world_size: int
    max_scratch_bytes: int
    costs: tuple[AlgorithmCost, ...]

    def __post_init__(self):
        integer(self.version, "version", minimum=1)
        integer(self.world_size, "world_size", minimum=1)
        integer(self.max_scratch_bytes, "max_scratch_bytes", minimum=1)
        names = [cost.name for cost in self.costs]
        if "native" not in names or len(set(names)) != len(names):
            raise ValueError("Plan requires native and unique algorithm costs")

    @classmethod
    def from_dict(cls, values: dict) -> CollectivePlan:
        values = dict(values)
        try:
            values["costs"] = tuple(AlgorithmCost(**cost) for cost in values["costs"])
            return cls(**values)
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid collective plan: {error}") from error

    def select(self, message_bytes: int) -> str:
        integer(message_bytes, "message_bytes", minimum=1)
        # Conservative peak includes gathered contributions and FP32 conversion
        # intermediates. Native backend scratch is managed by that backend.
        scratch = (self.world_size + 4) * message_bytes
        choices = [
            (cost.latency_s + cost.seconds_per_byte * message_bytes, order, cost.name)
            for order, cost in enumerate(self.costs)
            if cost.name == "native" or scratch <= self.max_scratch_bytes
        ]
        return min(choices)[2]


class CommunicationExecutor:
    """Choose an identical SUM schedule on every rank; never recover rank-locally.

    Plans are calibrated for this exact group and its routes. Installation and
    updates are collective operations invoked only at drained step boundaries.
    A collective failure must tear down the group and recover requests elsewhere.
    """

    def __init__(self, device_group, control_group, plan: dict):
        self.device_group = device_group
        self.control_group = control_group
        self.plan: CollectivePlan | None = None
        self.calls: Counter[str] = Counter()
        self.input_bytes: Counter[str] = Counter()
        self.install(plan)
        # Establish the process group before batched peer exchange is first used.
        import torch.distributed as dist

        dist.barrier(group=device_group)

    def install(self, values: dict) -> None:
        import torch.distributed as dist

        plan = None
        proposal: dict[str, Any]
        try:
            plan = CollectivePlan.from_dict(values)
            if plan.world_size != dist.get_world_size(self.device_group):
                raise ValueError("Collective calibration has the wrong group size")
            if self.plan is not None and (
                plan.version < self.plan.version
                or (plan.version == self.plan.version and plan != self.plan)
            ):
                raise ValueError("Collective plan versions must increase")
            proposal = {"plan": asdict(plan), "error": None}
        except (TypeError, ValueError) as error:
            proposal = {"plan": None, "error": str(error)}
        proposals: list[Any] = [None] * dist.get_world_size(self.control_group)
        dist.all_gather_object(proposals, proposal, group=self.control_group)
        if (
            any(item["error"] for item in proposals)
            or len({canonical_json(item) for item in proposals}) != 1
        ):
            raise ValueError("Ranks disagree on a valid orbital collective plan")
        assert plan is not None
        self.plan = plan

    def all_reduce(self, tensor):
        import torch.distributed as dist

        from vllm.satellite.collectives import gathered_sum

        assert self.plan is not None
        size = tensor.numel() * tensor.element_size()
        mode = self.plan.select(size)
        if mode == "native":
            result = tensor.clone()
            dist.all_reduce(result, group=self.device_group)
        else:
            result = gathered_sum(tensor, self.device_group, mode == "direct")
        self.calls[mode] += 1
        self.input_bytes[mode] += size
        return result

    def statistics(self) -> dict:
        return {
            "plan_version": self.plan.version if self.plan else None,
            "calls": dict(self.calls),
            "input_bytes": dict(self.input_bytes),
            "direct_send_bytes": self.input_bytes["direct"]
            * (self.plan.world_size - 1 if self.plan else 0),
        }
