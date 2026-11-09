# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictive policy integrated before vLLM allocation and progress accounting."""

from __future__ import annotations

from dataclasses import asdict

from vllm.satellite.config import OrbitalConfig
from vllm.satellite.contracts import RequestWork, StepPlan, finite
from vllm.satellite.forecast import EpochClock, ForecastReader
from vllm.satellite.policy import PredictivePolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus


class SatelliteScheduler(Scheduler):
    """Synchronous dense-decoder scheduler with conservative per-step budgets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        if not isinstance(config.additional_config, dict):
            raise ValueError("Orbital additional_config must be a dictionary")
        self.orbital_config = OrbitalConfig.from_dict(
            config.additional_config.get("orbital", {})
        )
        if config.scheduler_config.async_scheduling:
            raise ValueError("Orbital scheduling currently requires synchronous steps")
        if config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Orbital scheduling currently requires PP=1")
        if config.parallel_config.data_parallel_size != 1:
            raise ValueError("Use independent orbital replicas rather than engine DP")
        if config.speculative_config is not None:
            raise ValueError("Orbital speculative scheduling is not yet supported")
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError(
                "Orbital scheduler does not yet support external connectors"
            )
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        if not self.kv_cache_config.kv_cache_groups or any(
            not isinstance(group.kv_cache_spec, FullAttentionSpec)
            for group in self.kv_cache_config.kv_cache_groups
        ):
            raise ValueError("Orbital scheduling requires full-attention KV caches")
        self.orbital_clock = EpochClock()
        self.orbital_started_s = self.orbital_clock.now()
        self.orbital_reader = ForecastReader(self.orbital_config.forecast_path)
        self.orbital_policy = PredictivePolicy(
            self.orbital_config.profile,
            self.orbital_config.guard_s,
            self.orbital_config.aging_s,
        )
        self.orbital_plan = StepPlan(None, (), (), 0.0, 0.0)
        self.orbital_limits: dict[str, int] = {}
        self.orbital_suspended: set[str] = set()
        self.orbital_energy: list[tuple[float, float, float]] = []
        self.orbital_reserved_energy = 0.0
        self.orbital_steps = 0
        self.orbital_expired: list[Request] = []

    def add_request(self, request: Request) -> None:
        if request.mm_features or request.prompt_embeds is not None:
            raise ValueError("Orbital requests currently require text token inputs")
        if request.pooling_params is not None or request.resumable:
            raise ValueError("Orbital pooling and streaming-input sessions unsupported")
        if request.sampling_params is None:
            raise ValueError("Orbital requests require sampling parameters")
        metadata = (request.sampling_params.extra_args or {}).get("orbital", {})
        if not isinstance(metadata, dict) or set(metadata) - {"deadline_s"}:
            raise ValueError("Unknown orbital request fields")
        if metadata.get("deadline_s") is not None:
            finite(metadata["deadline_s"], "deadline_s")
        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False):
        now = self.orbital_clock.now()
        expired = []
        for request in self.requests.values():
            params = request.sampling_params
            metadata = (params.extra_args or {}).get("orbital", {}) if params else {}
            deadline = metadata.get("deadline_s")
            if (
                deadline is not None
                and now >= deadline
                and not request.is_finished()
                and request.request_id not in self.orbital_suspended
            ):
                expired.append(request.request_id)
        self.orbital_expired.extend(
            self.finish_requests(expired, RequestStatus.FINISHED_ABORTED)
        )
        forecast = self.orbital_reader.read()
        if forecast is not None and forecast.issued_s < self.orbital_started_s:
            self.orbital_reader.error = (
                "Forecast predates engine startup; refresh required"
            )
            forecast = None
        window = forecast.active(now, self.orbital_config.guard_s) if forecast else None
        if forecast is not None:
            self.orbital_energy = [
                (start, end, energy)
                for start, end, energy in self.orbital_energy
                if end >= forecast.issued_s
            ]
        spent = sum(
            energy
            for start, end, energy in self.orbital_energy
            if window is not None and start < window.end_s and end >= window.start_s
        )
        requests = []
        for request in self.requests.values():
            if request.is_finished():
                continue
            params = request.sampling_params
            metadata = (params.extra_args or {}).get("orbital", {}) if params else {}
            pending = max(0, request.num_tokens - request.num_computed_tokens)
            requests.append(
                RequestWork(
                    request.request_id,
                    request.arrival_time,
                    pending,
                    max(0, request.max_tokens - request.num_output_tokens),
                    pending > 1
                    or request.num_computed_tokens < request.num_prompt_tokens,
                    request.priority,
                    metadata.get("deadline_s"),
                    request.request_id in self.orbital_suspended,
                )
            )
        self.orbital_plan = self.orbital_policy.plan(
            requests,
            forecast,
            now,
            self.max_num_scheduled_tokens,
            spent,
        )
        self.orbital_limits = self.orbital_plan.token_limits
        # A selected waiting request may need a slot occupied by an unselected
        # running request. Evict only at a completed synchronous step boundary.
        selected_waiting = sum(
            r.request_id in self.orbital_limits and r.status != RequestStatus.RUNNING
            for r in self.requests.values()
        )
        slots_needed = max(
            0, len(self.running) + selected_waiting - self.max_num_active_reqs
        )
        for request in reversed(self.running.copy()):
            if slots_needed == 0:
                break
            if (
                request.request_id not in self.orbital_limits
                and request.request_id not in self.orbital_suspended
            ):
                if request.num_in_flight_tokens:
                    raise RuntimeError(
                        "Cannot preempt orbital work with tokens in flight"
                    )
                self.running.remove(request)
                self._preempt_request(request, now)
                slots_needed -= 1
        output = super().schedule(throttle_prefills)
        if output.total_num_scheduled_tokens and window is not None:
            # Reserve the full plan even if memory pressure reduced the actual
            # batch. This overestimates rather than replenishing the allowance.
            self.orbital_energy.append(
                (
                    now,
                    now + self.orbital_plan.estimated_s,
                    self.orbital_plan.estimated_energy_j,
                )
            )
            spent += self.orbital_plan.estimated_energy_j
            self.orbital_steps += 1
        self.orbital_reserved_energy = spent
        self.orbital_suspended.intersection_update(self.requests)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        from vllm.outputs import RequestError
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

        if scheduler_output.total_num_scheduled_tokens:
            start, end, energy = self.orbital_energy[-1]
            self.orbital_energy[-1] = (
                start,
                max(end, self.orbital_clock.now()),
                energy,
            )
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for request in self.orbital_expired:
            group = outputs.setdefault(request.client_index, EngineCoreOutputs())
            group.outputs.append(
                EngineCoreOutput(
                    request.request_id,
                    [],
                    finish_reason=FinishReason.ERROR,
                    error=RequestError(
                        code="orbital_deadline_exceeded",
                        message="The request deadline expired before completion.",
                        retryable=False,
                    ),
                )
            )
            if group.finished_requests is None:
                group.finished_requests = set()
            group.finished_requests.add(request.request_id)
        self.orbital_expired.clear()
        return outputs

    def get_step_token_budget(self) -> int:
        return sum(self.orbital_limits.values())

    def _request_blocks_can_be_freed(self, request: Request) -> bool:
        if request.request_id in self.orbital_suspended and not request.is_finished():
            return False
        return super()._request_blocks_can_be_freed(request)

    def get_request_token_budget(self, request: Request, proposed: int) -> int:
        return min(proposed, self.orbital_limits.get(request.request_id, 0))

    def suspend_request(self, request_id: str) -> Request:
        request = self.requests[request_id]
        if request.is_finished() or request.num_in_flight_tokens:
            raise ValueError("Only live requests at a completed step can be suspended")
        self.orbital_suspended.add(request_id)
        return request

    def resume_request(self, request_id: str) -> None:
        if request_id not in self.requests:
            raise KeyError(request_id)
        self.orbital_suspended.discard(request_id)

    def orbital_status(self) -> dict:
        return {
            "node_id": self.orbital_config.node_id,
            "engine_started_s": self.orbital_started_s,
            "plan": asdict(self.orbital_plan),
            "forecast_error": self.orbital_reader.error,
            "reserved_energy_j": self.orbital_reserved_energy,
            "executed_steps": self.orbital_steps,
            "suspended": sorted(self.orbital_suspended),
        }
