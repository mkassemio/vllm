# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private engine control-plane adapter. Call through a trusted coordinator."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import regex as re

from vllm.satellite.contracts import identifier, integer
from vllm.satellite.replay import capture_request, restore_request
from vllm.satellite.scheduler import SatelliteScheduler
from vllm.satellite.state import RequestSnapshot, StateConflict, canonical_json
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.executor.abstract import Executor


@dataclass
class PreparedRequest:
    request_id: str
    digest: str
    epoch: int
    phase: str
    computed_tokens: int
    block_ids: list[int]


def model_fingerprint(config) -> str:
    model = config.model_config
    revision = getattr(model.hf_config, "_commit_hash", None) or model.revision
    local = Path(model.model)
    if local.is_dir():
        files = sorted(
            p
            for p in local.rglob("*")
            if p.is_file()
            and (p.suffix in (".safetensors", ".bin") or p.name == "config.json")
        )
        if not any(p.suffix in (".safetensors", ".bin") for p in files):
            raise ValueError("Local model fingerprint requires weight files")
        manifest = []
        for path in files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest.append((str(path.relative_to(local)), digest))
        revision = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    if revision is None or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("State recovery requires a resolved, immutable model revision")
    model_config = model.hf_config.to_dict()
    for field in ("_name_or_path", "_commit_hash", "transformers_version"):
        model_config.pop(field, None)
    identity = {
        "revision": revision,
        "model_config": model_config,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


class EngineStateAdapter:
    def __init__(
        self,
        scheduler: SatelliteScheduler,
        executor: Executor | None = None,
        block_hasher=None,
    ) -> None:
        if not isinstance(scheduler, SatelliteScheduler):
            raise ValueError("Orbital state operations require SatelliteScheduler")
        self.scheduler = scheduler
        config = scheduler.vllm_config
        if config.model_config.logits_processors or config.watermark_config is not None:
            raise ValueError(
                "Orbital recovery does not support custom sampling processors"
            )
        self.executor = executor
        self.block_hasher = block_hasher
        self.fingerprint = model_fingerprint(scheduler.vllm_config)
        self.frozen: dict[str, int] = {}
        self.prepared: dict[str, PreparedRequest] = {}
        self.epochs: dict[str, int] = {}
        self.retired: set[tuple[str, int]] = set()

    def dispatch(self, command: str, payload: dict) -> dict:
        methods: dict[str, Callable[..., dict]] = {
            "checkpoint": self.checkpoint,
            "prepare": self.prepare,
            "load": self.load,
            "activate": self.activate,
            "discard": self.discard,
            "retire": self.retire,
            "resume": self.resume,
            "status": self.status,
            "inspect": self.inspect,
        }
        if command not in methods:
            raise ValueError("Unknown orbital state command")
        return methods[command](**payload)

    def checkpoint(
        self, request_id: str, epoch: int = 0, include_kv: bool = False
    ) -> dict:
        integer(epoch, "epoch")
        if self.epochs.get(request_id, 0) != epoch:
            raise StateConflict("Checkpoint requested from stale owner")
        was_paused = request_id in self.scheduler.orbital_suspended
        request = self.scheduler.suspend_request(request_id)
        try:
            snapshot = capture_request(request, self.fingerprint)
            images: list[dict] = []
            if include_kv and snapshot.computed_tokens:
                groups = self.scheduler.kv_cache_config.kv_cache_groups
                if len(groups) != 1:
                    raise ValueError("KV migration currently requires one cache group")
                block_size = groups[0].kv_cache_spec.block_size
                count = math.ceil(snapshot.computed_tokens / block_size)
                block_ids = self.scheduler.kv_cache_manager.get_block_ids(request_id)[
                    0
                ][:count]
                if self.executor is None:
                    raise ValueError("KV migration requires a model executor")
                images = self.executor.collective_rpc(
                    "orbital_export_kv",
                    args=(block_ids, snapshot.computed_tokens, self.fingerprint),
                )
            self.frozen[request_id] = epoch
            return {"snapshot": snapshot.encode(), "images": images}
        except Exception:
            if not was_paused:
                self.scheduler.resume_request(request_id)
            raise

    def prepare(
        self,
        handoff_id: str,
        snapshot_json: str,
        epoch: int,
        use_kv: bool = False,
        client_index: int = 0,
    ) -> dict:
        identifier(handoff_id, "handoff_id")
        integer(epoch, "epoch", minimum=1)
        integer(client_index, "client_index")
        snapshot = RequestSnapshot.decode(snapshot_json)
        digest = hashlib.sha256(snapshot_json.encode()).hexdigest()
        previous = self.prepared.get(handoff_id)
        if previous is not None:
            if previous.digest != digest or previous.epoch != epoch:
                raise StateConflict("Handoff ID reused for another snapshot")
            return {"phase": previous.phase}
        if snapshot.model_fingerprint != self.fingerprint:
            raise StateConflict("Destination model does not match snapshot")
        request_id = snapshot.request_id
        if request_id in self.scheduler.requests:
            raise StateConflict("Destination already has this request")
        if len(snapshot.output_token_ids) >= snapshot.max_output_tokens:
            raise ValueError("A completed request cannot be migrated")
        if (
            len(snapshot.prompt_token_ids) + len(snapshot.output_token_ids)
            >= self.scheduler.max_model_len
        ):
            raise ValueError("Destination context window is too small")
        request = restore_request(snapshot, client_index, self.block_hasher)
        self.scheduler.orbital_suspended.add(request_id)
        block_ids: list[int] = []
        try:
            self.scheduler.add_request(request)
            if use_kv and snapshot.computed_tokens:
                groups = self.scheduler.kv_cache_config.kv_cache_groups
                if len(groups) != 1:
                    raise ValueError("KV migration requires one cache group")
                blocks = self.scheduler.kv_cache_manager.allocate_slots(
                    request, snapshot.computed_tokens, delay_cache_blocks=True
                )
                if blocks is None:
                    raise MemoryError("Destination cannot reserve KV capacity")
                block_ids = blocks.get_block_ids()[0]
            self.prepared[handoff_id] = PreparedRequest(
                request_id,
                digest,
                epoch,
                "PREPARED" if block_ids else "READY",
                snapshot.computed_tokens if block_ids else 0,
                block_ids,
            )
        except Exception:
            self.scheduler.finish_requests([request_id], RequestStatus.FINISHED_ABORTED)
            self.scheduler.orbital_suspended.discard(request_id)
            raise
        return {"phase": self.prepared[handoff_id].phase}

    def load(self, handoff_id: str, images: list[dict]) -> dict:
        prepared = self.prepared[handoff_id]
        if prepared.phase == "READY":
            return {"phase": "READY"}
        if prepared.phase != "PREPARED":
            raise StateConflict("Destination is not awaiting a KV image")
        expected = self.scheduler.vllm_config.parallel_config.tensor_parallel_size
        if len(images) != expected or {image["rank"] for image in images} != set(
            range(expected)
        ):
            raise StateConflict("KV images do not cover every destination rank")
        if any(
            image["computed_tokens"] != prepared.computed_tokens for image in images
        ):
            raise StateConflict("KV image prefix differs from snapshot")
        # A failed import keeps the request suspended. The coordinator must
        # discard it or retry before marking the destination ready.
        if self.executor is None:
            raise ValueError("KV migration requires a model executor")
        results: list[int] = self.executor.collective_rpc(
            "orbital_import_kv", args=(images, prepared.block_ids, self.fingerprint)
        )
        if results != [prepared.computed_tokens] * expected:
            raise StateConflict("Not every worker imported the same prefix")
        request = self.scheduler.requests[prepared.request_id]
        request.num_computed_tokens = prepared.computed_tokens
        self.scheduler.kv_cache_manager.cache_blocks(request, prepared.computed_tokens)
        prepared.phase = "READY"
        return {"phase": "READY"}

    def activate(self, handoff_id: str, epoch: int) -> dict:
        prepared = self.prepared[handoff_id]
        if prepared.epoch != epoch or prepared.phase not in ("READY", "ACTIVE"):
            raise StateConflict("Destination has no compatible committed handoff")
        if prepared.phase == "READY":
            self.scheduler.resume_request(prepared.request_id)
            self.epochs[prepared.request_id] = epoch
            prepared.phase = "ACTIVE"
        return {"phase": "ACTIVE"}

    def discard(self, handoff_id: str) -> dict:
        prepared = self.prepared.get(handoff_id)
        if prepared is None:
            return {"phase": "ABSENT"}
        if prepared.phase == "DISCARDED":
            return {"phase": "DISCARDED"}
        if prepared.phase == "ACTIVE":
            raise StateConflict("Cannot discard an activated handoff")
        self.scheduler.finish_requests(
            [prepared.request_id], RequestStatus.FINISHED_ABORTED
        )
        self.scheduler.orbital_suspended.discard(prepared.request_id)
        prepared.phase = "DISCARDED"
        return {"phase": "DISCARDED"}

    def retire(self, request_id: str, epoch: int) -> dict:
        if (request_id, epoch) in self.retired:
            return {"phase": "RETIRED"}
        if request_id not in self.scheduler.requests:
            self.retired.add((request_id, epoch))
            return {"phase": "RETIRED"}
        if self.frozen.get(request_id) != epoch:
            raise StateConflict("Source is not frozen at the expected generation")
        self.scheduler.finish_requests([request_id], RequestStatus.FINISHED_ABORTED)
        self.scheduler.orbital_suspended.discard(request_id)
        self.retired.add((request_id, epoch))
        return {"phase": "RETIRED"}

    def resume(self, request_id: str, previous_epoch: int, epoch: int) -> dict:
        if (
            self.epochs.get(request_id) == epoch
            and epoch == previous_epoch + 1
            and request_id not in self.frozen
        ):
            return {"phase": "RUNNING"}
        if epoch != previous_epoch + 1 or self.frozen.get(request_id) != previous_epoch:
            raise StateConflict(
                "Cannot resume source without a new ownership generation"
            )
        self.epochs[request_id] = epoch
        self.scheduler.resume_request(request_id)
        del self.frozen[request_id]
        return {"phase": "RUNNING"}

    def status(self) -> dict:
        return self.scheduler.orbital_status()

    def inspect(self, handoff_id: str) -> dict:
        prepared = self.prepared.get(handoff_id)
        return {"phase": prepared.phase if prepared else "ABSENT"}
