# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic, resource-constrained scheduling without GPU dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence

from vllm.satellite.contracts import (
    CostProfile,
    Forecast,
    RequestWork,
    StepPlan,
    finite,
    integer,
)


class PredictivePolicy:
    def __init__(
        self, profile: CostProfile, guard_s: float = 0.01, aging_s: float = 30.0
    ) -> None:
        finite(guard_s, "guard_s")
        finite(aging_s, "aging_s")
        if aging_s == 0:
            raise ValueError("aging_s must be positive")
        self.profile = profile
        self.guard_s = guard_s
        self.aging_s = aging_s

    def plan(
        self,
        requests: Sequence[RequestWork],
        forecast: Forecast | None,
        now_s: float,
        max_tokens: int,
        spent_energy_j: float = 0.0,
    ) -> StepPlan:
        finite(now_s, "now_s")
        finite(spent_energy_j, "spent_energy_j")
        integer(max_tokens, "max_tokens")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Duplicate request IDs")
        window = forecast.active(now_s, self.guard_s) if forecast else None
        version = forecast.version if forecast else None
        if window is None or self.profile.power_w > window.max_power_w:
            return StepPlan(
                version,
                (),
                tuple((r.request_id, "no_feasible_operating_window") for r in requests),
                0.0,
                0.0,
            )
        assert forecast is not None
        budget = min(max_tokens, window.max_tokens)
        horizon = min(window.end_s, forecast.expires_s) - now_s - self.guard_s
        duration = min(
            window.max_step_s,
            horizon,
            max(0.0, window.energy_j - spent_energy_j) / self.profile.power_w,
        )
        fixed = self.profile.fixed_step_s * self.profile.safety_factor
        left = max(0.0, duration - fixed)
        chosen: list[tuple[str, int]] = []
        deferred: list[tuple[str, str]] = []

        def order(r: RequestWork) -> tuple[float, float, float, str]:
            age = max(0.0, now_s - r.arrival_s)
            priority = r.priority - math.floor(age / self.aging_s)
            deadline = r.deadline_s if r.deadline_s is not None else math.inf
            return priority, deadline, r.arrival_s, r.request_id

        for request in sorted(requests, key=order):
            reason = None
            if request.suspended:
                reason = "suspended"
            elif request.arrival_s > now_s:
                reason = "not_arrived"
            elif request.pending_tokens == 0 or request.remaining_output_tokens == 0:
                reason = "no_work"
            elif request.deadline_s is not None and request.deadline_s <= now_s:
                reason = "deadline_expired"
            elif len(chosen) >= window.max_requests or budget == 0:
                reason = "batch_capacity"
            if reason is not None:
                deferred.append((request.request_id, reason))
                continue
            token_s = self.profile.token_s(request.prefill)
            # Every request in a batch finishes after the whole batch, including
            # work selected later. Maintain the tightest deadline of all members.
            candidate_duration = duration
            if request.deadline_s is not None:
                candidate_duration = min(
                    candidate_duration, request.deadline_s - now_s - self.guard_s
                )
            used = duration - fixed - left
            available = min(left, candidate_duration - fixed - used)
            count = min(
                request.pending_tokens, budget, max(0, math.floor(available / token_s))
            )
            if count == 0:
                deferred.append((request.request_id, "step_time_or_energy"))
                continue
            duration = candidate_duration
            left = duration - fixed - used - count * token_s
            chosen.append((request.request_id, count))
            budget -= count

        estimated = duration - left if chosen else 0.0
        return StepPlan(
            version,
            tuple(chosen),
            tuple(deferred),
            estimated,
            estimated * self.profile.power_w,
        )
