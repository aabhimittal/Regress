"""Frozen probe suites.

A probe suite is frozen at cycle 0 and fingerprinted, so any later edit (even a single
element of a numpy input) is detected rather than silently moving the goalposts.

Overfitting defence: a deterministic, seed-keyed fraction of probes is held out. The
monitor reports public and held-out accuracy separately; a public score that holds while
the held-out score falls is the signature of an agent that has learned the probes rather
than the task. The split is computed from a hash of the probe id, so adding probes later
never reshuffles existing ones between the two halves.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, Iterator, Sequence

from .core import Probe

__all__ = ["ProbeSuite", "ProbeSuiteTampered"]


class ProbeSuiteTampered(RuntimeError):
    """The probe suite no longer matches the fingerprint taken at cycle 0."""


class ProbeSuite:
    def __init__(self, probes: Iterable[Probe], holdout_fraction: float = 0.0, seed: int = 0):
        probes = tuple(probes)
        if not probes:
            raise ValueError("a probe suite needs at least one probe")
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        ids = [p.id for p in probes]
        dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
        if dupes:
            raise ValueError(f"duplicate probe ids: {dupes[:5]}")
        self._probes = probes
        self._by_id = {p.id: p for p in probes}
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self._holdout = frozenset(p.id for p in probes if self._is_holdout(p.id))
        # With tiny suites the hash split can put everything on one side; keep at least
        # one public probe so the suite stays usable.
        if len(self._holdout) == len(probes):
            keep = min(self._holdout)
            self._holdout = self._holdout - {keep}
        self._fingerprint = self._compute_fingerprint()

    def _is_holdout(self, probe_id: str) -> bool:
        if self.holdout_fraction == 0.0:
            return False
        h = hashlib.sha256(f"{self.seed}:{probe_id}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64 < self.holdout_fraction

    def _compute_fingerprint(self) -> str:
        parts = sorted(p.fingerprint for p in self._probes)
        meta = f"holdout={self.holdout_fraction!r};seed={self.seed}"
        return hashlib.sha256(("|".join(parts) + "|" + meta).encode()).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def verify(self, expected: str | None = None) -> None:
        """Re-hash every probe. Catches in-place mutation of mutable probe inputs."""
        current = self._compute_fingerprint()
        target = expected if expected is not None else self._fingerprint
        if current != target:
            raise ProbeSuiteTampered(
                f"probe suite fingerprint changed: expected {target[:12]}…, got {current[:12]}…"
            )

    def is_holdout(self, probe_id: str) -> bool:
        return probe_id in self._holdout

    @property
    def public(self) -> tuple[Probe, ...]:
        return tuple(p for p in self._probes if p.id not in self._holdout)

    @property
    def holdout(self) -> tuple[Probe, ...]:
        return tuple(p for p in self._probes if p.id in self._holdout)

    @property
    def ids(self) -> Sequence[str]:
        return [p.id for p in self._probes]

    def __getitem__(self, probe_id: str) -> Probe:
        return self._by_id[probe_id]

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._probes)

    def __len__(self) -> int:
        return len(self._probes)

    def __repr__(self) -> str:
        return (
            f"ProbeSuite(n={len(self)}, holdout={len(self._holdout)}, "
            f"fingerprint={self._fingerprint[:12]})"
        )
