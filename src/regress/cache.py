"""Response cache: never pay twice for the same answer.

For an LLM agent every probe is a model call, and bisection re-runs whole checkpoints. If the
agent can say what state it is in (``agent.state_fingerprint()``), a response depends only
on (state, probe), so it can be cached. Re-testing an unchanged checkpoint then costs
nothing, and a learning cycle that changed nothing costs nothing either.

Caveat: this assumes deterministic answers. For a sampling agent (temperature > 0) the cache
freezes the first sample per state, which hides run-to-run noise. Either evaluate at
temperature 0, or do not cache.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Response

__all__ = ["ResponseCache", "CacheStats"]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    uncacheable: int = 0  # agent had no state fingerprint

    @property
    def calls_saved(self) -> int:
        return self.hits


class ResponseCache:
    """In-memory LRU, optionally persisted as JSON lines.

    Only JSON-serialisable outputs are written to disk; others stay in memory. A corrupt or
    truncated line in the file is skipped, never fatal.
    """

    def __init__(self, path: str | Path | None = None, max_entries: int = 200_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self.path = Path(path) if path else None
        self._data: "OrderedDict[str, Response]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()
        self.skipped_lines = 0
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def key(state: str, probe_fingerprint: str) -> str:
        return f"{state}|{probe_fingerprint}"

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                conf = rec.get("confidence")
                self._put(rec["key"], Response(rec["output"], conf, list(rec.get("retrieved", []))), persist=False)
            except Exception:
                self.skipped_lines += 1

    def get(self, key: str) -> Response | None:
        with self._lock:
            r = self._data.get(key)
            if r is None:
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            # Hand out a copy so a caller mutating it cannot poison the cache.
            return Response(r.output, r.confidence, list(r.retrieved))

    def put(self, key: str, response: Response) -> None:
        with self._lock:
            self._put(key, response, persist=True)

    def _put(self, key: str, response: Response, persist: bool) -> None:
        self._data[key] = Response(response.output, response.confidence, list(response.retrieved or ()))
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)
        if persist and self.path is not None:
            rec: dict[str, Any] = {"key": key, "output": response.output,
                                   "confidence": _finite_or_none(response.confidence),
                                   "retrieved": [str(i) for i in (response.retrieved or ())]}
            try:
                line = json.dumps(rec, allow_nan=False)
            except (TypeError, ValueError):
                return  # not JSON-serialisable: memory only
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def __len__(self) -> int:
        return len(self._data)


def _finite_or_none(x: Any) -> Any:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
