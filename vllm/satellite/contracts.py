# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts. Times are Unix seconds, energy joules, power watts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def finite(value: float, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{name} must be a nonempty string of at most 512 characters")


@dataclass(frozen=True)
class OperatingWindow:
    start_s: float
    end_s: float
    max_power_w: float
    energy_j: float
    max_step_s: float
    max_tokens: int
    max_requests: int

    def __post_init__(self) -> None:
        for name in ("start_s", "end_s", "max_power_w", "energy_j", "max_step_s"):
            finite(getattr(self, name), name)
        if self.end_s <= self.start_s:
            raise ValueError("Operating windows must have positive duration")
        integer(self.max_tokens, "max_tokens")
        integer(self.max_requests, "max_requests")


@dataclass(frozen=True)
class Forecast:
    version: int
    issued_s: float
    expires_s: float
    windows: tuple[OperatingWindow, ...]

    def __post_init__(self) -> None:
        integer(self.version, "version")
        finite(self.issued_s, "issued_s")
        finite(self.expires_s, "expires_s")
        if self.expires_s <= self.issued_s:
            raise ValueError("Forecast expiry must follow issue time")
        if not isinstance(self.windows, tuple):
            raise ValueError("windows must be an immutable tuple")
        for i, window in enumerate(self.windows):
            if not isinstance(window, OperatingWindow):
                raise ValueError("Invalid operating window")
            if i and self.windows[i - 1].end_s > window.start_s:
                raise ValueError("Operating windows must be ordered and nonoverlapping")

    def active(self, now_s: float, guard_s: float = 0.0) -> OperatingWindow | None:
        finite(now_s, "now_s")
        finite(guard_s, "guard_s")
        if now_s < self.issued_s or now_s + guard_s >= self.expires_s:
            return None
        return next(
            (
                w
                for w in self.windows
                if w.start_s <= now_s and now_s + guard_s < w.end_s
            ),
            None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Forecast:
        if set(data) != {"version", "issued_s", "expires_s", "windows"}:
            raise ValueError("Unknown or missing forecast fields")
        return cls(
            version=data["version"],
            issued_s=data["issued_s"],
            expires_s=data["expires_s"],
            windows=tuple(OperatingWindow(**w) for w in data["windows"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostProfile:
    """Calibrated additive time envelope; not a hard real-time guarantee."""

    fixed_step_s: float
    prefill_token_s: float
    decode_token_s: float
    power_w: float
    safety_factor: float = 1.2

    def __post_init__(self) -> None:
        finite(self.fixed_step_s, "fixed_step_s")
        for name in ("prefill_token_s", "decode_token_s", "power_w"):
            finite(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        finite(self.safety_factor, "safety_factor", minimum=1.0)

    def token_s(self, prefill: bool) -> float:
        cost = self.prefill_token_s if prefill else self.decode_token_s
        return cost * self.safety_factor


@dataclass(frozen=True)
class RequestWork:
    request_id: str
    arrival_s: float
    pending_tokens: int
    remaining_output_tokens: int
    prefill: bool
    priority: int = 0
    deadline_s: float | None = None
    suspended: bool = False

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        finite(self.arrival_s, "arrival_s")
        integer(self.pending_tokens, "pending_tokens")
        integer(self.remaining_output_tokens, "remaining_output_tokens")
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        if type(self.prefill) is not bool or type(self.suspended) is not bool:
            raise ValueError("prefill and suspended must be booleans")
        if self.deadline_s is not None:
            finite(self.deadline_s, "deadline_s")


@dataclass(frozen=True)
class StepPlan:
    forecast_version: int | None
    tokens: tuple[tuple[str, int], ...]
    deferred: tuple[tuple[str, str], ...]
    estimated_s: float
    estimated_energy_j: float

    @property
    def token_limits(self) -> dict[str, int]:
        return dict(self.tokens)
