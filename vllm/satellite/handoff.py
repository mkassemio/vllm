# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recoverable handoffs over trusted engine endpoints and one durable authority.

The application routes and journals output tokens here before delivering them.
Existing OpenAI frontend connections are not transferred by this control API.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import RLock

from vllm.satellite.state import Ownership, RequestSnapshot, StateConflict, StateJournal

Endpoint = Callable[[str, dict], dict]


class HandoffCoordinator:
    def __init__(self, journal: StateJournal, endpoints: Mapping[str, Endpoint]):
        self.journal = journal
        self.endpoints = dict(endpoints)
        self.lock = RLock()

    def _call(self, node: str, command: str, **payload) -> dict:
        return self.endpoints[node](command, payload)

    def publish(
        self,
        request_id: str,
        owner: str,
        epoch: int,
        cumulative_tokens: tuple[int, ...],
        finished: bool = False,
    ) -> int:
        """Commit cumulative token history; clients consume durable token offsets."""
        with self.lock:
            record = self.journal.ownership(request_id)
            if (record.owner, record.epoch) != (owner, epoch) or record.phase not in (
                ("RUNNING", "FINISHED") if finished else ("RUNNING",)
            ):
                raise StateConflict("Output came from a stale or frozen owner")
            previous = self.journal.tokens(request_id)
            if cumulative_tokens[: len(previous)] != previous:
                raise StateConflict("Output changed previously committed tokens")
            if len(cumulative_tokens) > len(previous):
                return self.journal.append_output(
                    request_id,
                    owner,
                    epoch,
                    len(previous),
                    cumulative_tokens[len(previous) :],
                    finished=finished,
                )
            if finished:
                self.journal.finish(request_id, owner, epoch)
            return len(previous)

    def start(
        self, handoff_id: str, request_id: str, target: str, mode: str = "recompute"
    ) -> Ownership:
        """Freeze and prepare, then advance; failures remain frozen for retry.

        A lost reply never implies failure of the underlying operation. Retry
        this method with the same ID, or use `advance` after PREPARED is durable.
        """
        with self.lock:
            if target not in self.endpoints or mode not in ("recompute", "transfer"):
                raise ValueError("Unknown destination or recovery mode")
            try:
                existing = self.journal.handoff(handoff_id)
            except KeyError:
                existing = None
            if existing is not None:
                if (existing["request_id"], existing["target"], existing["mode"]) != (
                    request_id,
                    target,
                    mode,
                ):
                    raise StateConflict("Handoff ID reused for another operation")
                return self.advance(handoff_id)
            record = self.journal.ownership(request_id)
            if record.phase != "RUNNING" or target == record.owner:
                raise StateConflict("Request is not eligible for handoff")
            result = self._call(
                record.owner, "checkpoint", request_id=request_id, epoch=record.epoch
            )
            snapshot = RequestSnapshot.decode(result["snapshot"])
            if snapshot.request_id != request_id:
                raise StateConflict("Engine returned another request's snapshot")
            self.publish(
                request_id, record.owner, record.epoch, snapshot.output_token_ids
            )
            digest = self.journal.put_snapshot(snapshot)
            self.journal.prepare(
                handoff_id, request_id, record.owner, record.epoch, target, digest, mode
            )
            return self.advance(handoff_id)

    def advance(self, handoff_id: str) -> Ownership:
        """Retry the durable decision without ever reviving a committed source."""
        with self.lock:
            row = self.journal.handoff(handoff_id)
            record = self.journal.ownership(row["request_id"])
            if row["phase"] in ("PREPARED", "READY"):
                snapshot = self.journal.snapshot(row["snapshot"])
                result = self._call(
                    row["target"],
                    "prepare",
                    handoff_id=handoff_id,
                    snapshot_json=snapshot.encode(),
                    epoch=row["epoch"] + 1,
                    use_kv=row["mode"] == "transfer",
                )
                if result["phase"] == "PREPARED":
                    source = self._call(
                        row["source"],
                        "checkpoint",
                        request_id=row["request_id"],
                        epoch=row["epoch"],
                        include_kv=True,
                    )
                    if source["snapshot"] != snapshot.encode():
                        raise StateConflict("Frozen source changed during transfer")
                    result = self._call(
                        row["target"],
                        "load",
                        handoff_id=handoff_id,
                        images=source["images"],
                    )
                if result["phase"] != "READY":
                    raise StateConflict("Destination is not ready for ownership")
                self.journal.ready(handoff_id, row["target"], row["snapshot"])
                record = self.journal.commit(handoff_id)
                row = self.journal.handoff(handoff_id)
            if row["phase"] == "COMMITTED":
                if (record.owner, record.epoch, record.phase) == (
                    row["target"],
                    row["epoch"] + 1,
                    "FINISHED",
                ):
                    self._call(
                        row["source"],
                        "retire",
                        request_id=row["request_id"],
                        epoch=row["epoch"],
                    )
                    return record
                if (record.owner, record.epoch, record.phase) != (
                    row["target"],
                    row["epoch"] + 1,
                    "RUNNING",
                ):
                    raise StateConflict(
                        "Committed handoff was superseded or terminated"
                    )
                state = self._call(row["target"], "inspect", handoff_id=handoff_id)
                if state["phase"] == "ABSENT":
                    # Replay committed output since the snapshot after a restart.
                    snapshot = replace(
                        self.journal.snapshot(row["snapshot"]),
                        output_token_ids=self.journal.tokens(row["request_id"]),
                        computed_tokens=0,
                    )
                    self._call(
                        row["target"],
                        "prepare",
                        handoff_id=handoff_id,
                        snapshot_json=snapshot.encode(),
                        epoch=record.epoch,
                    )
                self._call(
                    row["target"], "activate", handoff_id=handoff_id, epoch=record.epoch
                )
                self._call(
                    row["source"],
                    "retire",
                    request_id=row["request_id"],
                    epoch=row["epoch"],
                )
                return record
            if row["phase"] == "ABORTED":
                return self.abort(handoff_id)
            raise StateConflict("Handoff was cancelled")

    def abort(self, handoff_id: str) -> Ownership:
        """Record an abort before resuming the source with a fenced generation."""
        with self.lock:
            record = self.journal.abort(handoff_id)
            row = self.journal.handoff(handoff_id)
            if (record.owner, record.epoch, record.phase) != (
                row["source"],
                row["epoch"] + 1,
                "RUNNING",
            ):
                raise StateConflict("Aborted handoff was superseded or terminated")
            self._call(row["target"], "discard", handoff_id=handoff_id)
            self._call(
                row["source"],
                "resume",
                request_id=row["request_id"],
                previous_epoch=row["epoch"],
                epoch=record.epoch,
            )
            return record
