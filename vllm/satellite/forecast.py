# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded forecast ingestion and a wall-clock anchor for monotonic execution."""

from __future__ import annotations

import json
import time
from pathlib import Path

from vllm.satellite.contracts import Forecast


class EpochClock:
    def __init__(self) -> None:
        self._epoch = time.time()
        self._monotonic = time.monotonic()

    def now(self) -> float:
        return self._epoch + time.monotonic() - self._monotonic


class ForecastReader:
    """Read atomically replaced JSON. Invalid updates revoke admission."""

    def __init__(self, path: str, max_bytes: int = 1_048_576) -> None:
        if not path or max_bytes <= 0:
            raise ValueError("A forecast path and positive size limit are required")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.last: Forecast | None = None
        self.error: str | None = None

    def read(self) -> Forecast | None:
        try:
            with self.path.open("rb") as stream:
                payload = stream.read(self.max_bytes + 1)
            if len(payload) > self.max_bytes:
                raise ValueError("Forecast exceeds size limit")
            candidate = Forecast.from_dict(json.loads(payload))
            if self.last is not None:
                if candidate.version < self.last.version:
                    raise ValueError("Forecast version regressed")
                if candidate.version == self.last.version and candidate != self.last:
                    raise ValueError("Forecast changed without a new version")
                if candidate.issued_s < self.last.issued_s:
                    raise ValueError("Forecast issue time regressed")
            self.last = candidate
            self.error = None
            return candidate
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.error = str(error)
            return None
