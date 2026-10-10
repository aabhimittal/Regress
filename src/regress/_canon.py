"""Canonical, collision-resistant keys for arbitrary probe inputs and agent outputs.

Two jobs:

* ``fingerprint`` — a stable SHA-256 over a value, used to freeze probe suites.
  ``repr`` is not good enough: ``repr(np.zeros(10_000))`` elides the middle of the
  array, so editing element 5_000 would not change the fingerprint.
* ``output_key`` — a short hashable string used to compare outputs across cycles.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

__all__ = ["canonical", "fingerprint", "output_key"]


def canonical(value: Any) -> Any:
    """Convert ``value`` into a JSON-serialisable structure with a deterministic layout."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return {"__float__": "nan"}
        if math.isinf(f):
            return {"__float__": "inf" if f > 0 else "-inf"}
        # -0.0 and 0.0 compare equal; keep them equal here too.
        return 0.0 if f == 0.0 else f
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__ndarray__": hashlib.sha256(arr.tobytes()).hexdigest(),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        items = [(json.dumps(canonical(k), sort_keys=True), canonical(v)) for k, v in value.items()]
        items.sort(key=lambda kv: kv[0])
        return {"__dict__": items}
    if isinstance(value, (list, tuple)):
        tag = "__tuple__" if isinstance(value, tuple) else "__list__"
        return {tag: [canonical(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        members = sorted(json.dumps(canonical(v), sort_keys=True) for v in value)
        return {"__set__": members}
    # Dataclasses and other objects: fall back to their public __dict__ if any, else repr.
    if hasattr(value, "__dataclass_fields__"):
        fields = {k: getattr(value, k) for k in value.__dataclass_fields__}
        return {"__obj__": type(value).__qualname__, "fields": canonical(fields)}
    return {"__repr__": f"{type(value).__qualname__}:{value!r}"}


def fingerprint(value: Any) -> str:
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_key(value: Any) -> str:
    """Hashable identity of an output. Short strings stay readable; everything else is hashed."""
    if isinstance(value, str) and len(value) <= 64:
        return "s:" + value
    if isinstance(value, (bool, np.bool_)):
        return "b:" + str(bool(value))
    if isinstance(value, (int, np.integer)):
        return "i:" + str(int(value))
    return "h:" + fingerprint(value)[:24]
