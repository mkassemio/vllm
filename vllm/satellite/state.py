# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Durable request ownership and idempotent, fenced handoff transactions.

One authority owns this SQLite database. Remote engines must access that
authority, not independent database copies. Availability is sacrificed when
ownership cannot be established. SQLite files are not distributed consensus.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vllm.satellite.contracts import identifier, integer


class StateConflict(RuntimeError):
    """A stale actor or incompatible state attempted a lifecycle transition."""


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    model_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    computed_tokens: int
    max_output_tokens: int
    sampling_json: str
    delivered_tokens: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        identifier(self.model_fingerprint, "model_fingerprint")
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Unsupported snapshot schema")
        for name in ("prompt_token_ids", "output_token_ids"):
            tokens = getattr(self, name)
            if not isinstance(tokens, tuple):
                raise ValueError("Snapshot tokens must be immutable tuples")
            for token in tokens:
                integer(token, "token_id")
        if not self.prompt_token_ids:
            raise ValueError("A replay snapshot requires prompt tokens")
        integer(self.computed_tokens, "computed_tokens")
        integer(self.max_output_tokens, "max_output_tokens", minimum=1)
        integer(self.delivered_tokens, "delivered_tokens")
        if self.computed_tokens > len(self.prompt_token_ids) + len(
            self.output_token_ids
        ):
            raise ValueError("Computed position exceeds committed token history")
        if len(self.output_token_ids) > self.max_output_tokens:
            raise ValueError("Generated history exceeds output limit")
        if self.delivered_tokens > len(self.output_token_ids):
            raise ValueError("Delivered position exceeds generated history")
        params = json.loads(self.sampling_json)
        if not isinstance(params, dict) or canonical_json(params) != self.sampling_json:
            raise ValueError("sampling_json must be a canonical JSON object")

    def encode(self) -> str:
        return canonical_json(asdict(self))

    @classmethod
    def decode(cls, payload: str) -> RequestSnapshot:
        values = json.loads(payload)
        values["prompt_token_ids"] = tuple(values["prompt_token_ids"])
        values["output_token_ids"] = tuple(values["output_token_ids"])
        return cls(**values)


@dataclass(frozen=True)
class Ownership:
    request_id: str
    owner: str
    epoch: int
    phase: str
    output_tokens: int


class StateJournal:
    """Single-authority durable ledger with transactional compare-and-set."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(version INTEGER NOT NULL);
            INSERT INTO metadata SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM metadata);
            CREATE TABLE IF NOT EXISTS requests(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, epoch INTEGER NOT NULL,
                phase TEXT NOT NULL, output_tokens INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS snapshots(
                digest TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS handoffs(
                id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                source TEXT NOT NULL, epoch INTEGER NOT NULL, target TEXT NOT NULL,
                snapshot TEXT NOT NULL, phase TEXT NOT NULL,
                mode TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outputs(
                request_id TEXT NOT NULL, offset INTEGER NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(request_id, offset));
            """
        )
        if [row[0] for row in self.db.execute("SELECT version FROM metadata")] != [1]:
            self.db.close()
            raise ValueError("Unsupported journal schema")

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            else:
                self.db.execute("COMMIT")

    def _ownership(self, request_id: str) -> Ownership:
        row = self.db.execute(
            "SELECT * FROM requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return Ownership(
            row["id"], row["owner"], row["epoch"], row["phase"], row["output_tokens"]
        )

    def ownership(self, request_id: str) -> Ownership:
        with self.lock:
            return self._ownership(request_id)

    def register(self, request_id: str, owner: str) -> Ownership:
        identifier(request_id, "request_id")
        identifier(owner, "owner")
        with self.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO requests(id, owner, epoch, phase) "
                "VALUES(?,?,0,'RUNNING')",
                (request_id, owner),
            )
            result = self._ownership(request_id)
            if result.owner != owner or result.phase != "RUNNING":
                raise StateConflict("Request is already owned or terminal")
            return result

    @staticmethod
    def _check_owner(record: Ownership, owner: str, epoch: int, phase="RUNNING"):
        integer(epoch, "epoch")
        if (record.owner, record.epoch, record.phase) != (owner, epoch, phase):
            raise StateConflict("Stale ownership generation or request phase")

    def put_snapshot(self, snapshot: RequestSnapshot) -> str:
        payload = snapshot.encode()
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES(?,?)", (digest, payload)
            )
            stored = self.db.execute(
                "SELECT payload FROM snapshots WHERE digest=?", (digest,)
            ).fetchone()[0]
            if stored != payload:
                raise StateConflict("Snapshot digest collision or storage corruption")
        return digest

    def snapshot(self, digest: str) -> RequestSnapshot:
        with self.lock:
            row = self.db.execute(
                "SELECT payload FROM snapshots WHERE digest=?", (digest,)
            ).fetchone()
            if row is None:
                raise KeyError(digest)
            payload = row[0]
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                raise StateConflict("Corrupt snapshot")
            return RequestSnapshot.decode(payload)

    def prepare(
        self,
        handoff_id: str,
        request_id: str,
        source: str,
        epoch: int,
        target: str,
        snapshot_digest: str,
        mode: str = "recompute",
    ) -> str:
        identifier(handoff_id, "handoff_id")
        identifier(target, "target")
        if target == source:
            raise ValueError("Handoff target must differ from source")
        if mode not in ("recompute", "transfer"):
            raise ValueError("Unknown handoff recovery mode")
        snapshot = self.snapshot(snapshot_digest)
        if snapshot.request_id != request_id:
            raise StateConflict("Snapshot belongs to another request")
        args = (handoff_id, request_id, source, epoch, target, snapshot_digest)
        with self.transaction():
            existing = self.db.execute(
                "SELECT * FROM handoffs WHERE id=?", (handoff_id,)
            ).fetchone()
            if existing is not None:
                if tuple(existing)[:6] != args or existing["mode"] != mode:
                    raise StateConflict("Handoff ID reused with different parameters")
                return existing["phase"]
            record = self._ownership(request_id)
            self._check_owner(record, source, epoch)
            history = self._tokens(request_id)
            if tuple(history) != snapshot.output_token_ids:
                raise StateConflict("Snapshot does not match committed output history")
            self.db.execute(
                "INSERT INTO handoffs VALUES(?,?,?,?,?,?,'PREPARED',?)", (*args, mode)
            )
            self.db.execute(
                "UPDATE requests SET phase='FROZEN' WHERE id=?", (request_id,)
            )
            return "PREPARED"

    def handoff(self, handoff_id: str) -> dict:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM handoffs WHERE id=?", (handoff_id,)
            ).fetchone()
            if row is None:
                raise KeyError(handoff_id)
            return dict(row)

    def ready(self, handoff_id: str, target: str, snapshot_digest: str) -> None:
        with self.transaction():
            row = self.handoff(handoff_id)
            if (row["target"], row["snapshot"]) != (target, snapshot_digest):
                raise StateConflict("Destination readiness does not match handoff")
            if row["phase"] in ("READY", "COMMITTED"):
                return
            if row["phase"] != "PREPARED":
                raise StateConflict("Handoff cannot become ready")
            self._check_owner(
                self._ownership(row["request_id"]),
                row["source"],
                row["epoch"],
                "FROZEN",
            )
            self.db.execute(
                "UPDATE handoffs SET phase='READY' WHERE id=?", (handoff_id,)
            )

    def commit(self, handoff_id: str) -> Ownership:
        with self.transaction():
            row = self.handoff(handoff_id)
            record = self._ownership(row["request_id"])
            if row["phase"] == "COMMITTED":
                return record
            if row["phase"] != "READY":
                raise StateConflict("Destination has not confirmed readiness")
            self._check_owner(record, row["source"], row["epoch"], "FROZEN")
            self.db.execute(
                "UPDATE requests SET owner=?, epoch=epoch+1, phase='RUNNING' "
                "WHERE id=?",
                (row["target"], row["request_id"]),
            )
            self.db.execute(
                "UPDATE handoffs SET phase='COMMITTED' WHERE id=?", (handoff_id,)
            )
            return self._ownership(row["request_id"])

    def abort(self, handoff_id: str) -> Ownership:
        with self.transaction():
            row = self.handoff(handoff_id)
            record = self._ownership(row["request_id"])
            if row["phase"] == "ABORTED":
                return record
            if row["phase"] not in ("PREPARED", "READY"):
                raise StateConflict(
                    "Committed or cancelled handoff cannot be rolled back"
                )
            self._check_owner(record, row["source"], row["epoch"], "FROZEN")
            self.db.execute(
                "UPDATE requests SET epoch=epoch+1, phase='RUNNING' WHERE id=?",
                (row["request_id"],),
            )
            self.db.execute(
                "UPDATE handoffs SET phase='ABORTED' WHERE id=?", (handoff_id,)
            )
            return self._ownership(row["request_id"])

    def append_output(
        self,
        request_id: str,
        owner: str,
        epoch: int,
        offset: int,
        tokens: tuple[int, ...],
        finished: bool = False,
    ) -> int:
        integer(offset, "offset")
        if not tokens:
            raise ValueError("Output records cannot be empty")
        for token in tokens:
            integer(token, "token_id")
        payload = canonical_json(tokens)
        with self.transaction():
            record = self._ownership(request_id)
            phase = "FINISHED" if finished and record.phase == "FINISHED" else "RUNNING"
            self._check_owner(record, owner, epoch, phase)
            existing = self.db.execute(
                "SELECT payload FROM outputs WHERE request_id=? AND offset=?",
                (request_id, offset),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise StateConflict("Duplicate output has different tokens")
                if finished:
                    if offset + len(tokens) != record.output_tokens:
                        raise StateConflict(
                            "Terminal retry does not end at the output cursor"
                        )
                    self.db.execute(
                        "UPDATE requests SET phase='FINISHED' WHERE id=?", (request_id,)
                    )
                return record.output_tokens
            if record.phase == "FINISHED":
                raise StateConflict("Cannot extend a finished request")
            if offset != record.output_tokens:
                raise StateConflict("Output offset is not the next committed position")
            self.db.execute(
                "INSERT INTO outputs VALUES(?,?,?)", (request_id, offset, payload)
            )
            position = offset + len(tokens)
            self.db.execute(
                "UPDATE requests SET output_tokens=?,phase=? WHERE id=?",
                (position, "FINISHED" if finished else "RUNNING", request_id),
            )
            return position

    def _tokens(self, request_id: str) -> list[int]:
        rows = self.db.execute(
            "SELECT offset,payload FROM outputs WHERE request_id=? ORDER BY offset",
            (request_id,),
        ).fetchall()
        result: list[int] = []
        for row in rows:
            if row["offset"] != len(result):
                raise StateConflict("Corrupt output journal")
            result.extend(json.loads(row["payload"]))
        return result

    def tokens(self, request_id: str, after: int = 0) -> tuple[int, ...]:
        integer(after, "after")
        with self.lock:
            record = self._ownership(request_id)
            if after > record.output_tokens:
                raise ValueError("Output cursor exceeds committed position")
            return tuple(self._tokens(request_id)[after:])

    def finish(self, request_id: str, owner: str, epoch: int) -> Ownership:
        with self.transaction():
            record = self._ownership(request_id)
            if record.phase == "FINISHED" and (record.owner, record.epoch) == (
                owner,
                epoch,
            ):
                return record
            self._check_owner(record, owner, epoch)
            self.db.execute(
                "UPDATE requests SET phase='FINISHED' WHERE id=?", (request_id,)
            )
            return self._ownership(request_id)

    def cancel(self, request_id: str) -> Ownership:
        with self.transaction():
            record = self._ownership(request_id)
            if record.phase in ("FINISHED", "CANCELLED"):
                return record
            self.db.execute(
                "UPDATE requests SET phase='CANCELLED',epoch=epoch+1 WHERE id=?",
                (request_id,),
            )
            self.db.execute(
                "UPDATE handoffs SET phase='CANCELLED' WHERE request_id=? "
                "AND phase IN ('PREPARED','READY')",
                (request_id,),
            )
            return self._ownership(request_id)


@dataclass(frozen=True)
class RecoveryCosts:
    keep_s: float | None
    transfer_s: float | None
    recompute_s: float | None

    def choose(
        self, remaining_s: float
    ) -> Literal["keep", "transfer", "recompute", "wait"]:
        from vllm.satellite.contracts import finite

        finite(remaining_s, "remaining_s")
        actions: tuple[Literal["keep", "transfer", "recompute"], ...] = (
            "keep",
            "transfer",
            "recompute",
        )
        candidates = []
        for order, action in enumerate(actions):
            cost = getattr(self, f"{action}_s")
            if cost is not None:
                finite(cost, f"{action}_s")
                if cost <= remaining_s:
                    candidates.append((cost, order, action))
        return min(candidates)[2] if candidates else "wait"
