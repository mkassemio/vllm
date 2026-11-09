# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict configuration for the explicitly selected orbital scheduler."""

from dataclasses import dataclass
from typing import Any

from vllm.satellite.contracts import CostProfile, finite, identifier


@dataclass(frozen=True)
class OrbitalConfig:
    forecast_path: str
    profile: CostProfile
    node_id: str = "satellite"
    guard_s: float = 0.01
    aging_s: float = 30.0

    def __post_init__(self) -> None:
        identifier(self.node_id, "node_id")
        if not isinstance(self.forecast_path, str) or not self.forecast_path:
            raise ValueError("forecast_path is required")
        finite(self.guard_s, "guard_s")
        finite(self.aging_s, "aging_s")
        if self.aging_s == 0:
            raise ValueError("aging_s must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrbitalConfig":
        if not isinstance(data, dict):
            raise ValueError("additional_config.orbital must be an object")
        values = dict(data)
        try:
            values["profile"] = CostProfile(**values["profile"])
            return cls(**values)
        except (TypeError, KeyError) as error:
            raise ValueError(f"Invalid orbital configuration: {error}") from error
