"""Is the probe suite still able to see anything?

Probe suites go stale. A probe the agent has always passed can only ever report a
regression, never an improvement; a probe it has never passed tells you nothing until it
does; a probe that flips back and forth without a trend is noise that eats tolerance. This
module classifies every probe from the recorded history and says when to rotate.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["ProbeHealth", "SuiteHealth", "suite_health"]

PASS = 0.5


@dataclass
class ProbeHealth:
    probe_id: str
    kind: str  # saturated | unsolved | flaky | regressed | improved | mixed | unscored
    pass_rate: float
    flips: int


@dataclass
class SuiteHealth:
    cycles: int
    counts: dict[str, int]
    probes: list[ProbeHealth] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def ids(self, kind: str) -> list[str]:
        return [p.probe_id for p in self.probes if p.kind == kind]


def suite_health(history: Sequence, flaky_flips: int = 3) -> SuiteHealth:
    """Classify each probe over ``history`` (``CycleReport`` objects or dicts)."""
    if not history:
        raise ValueError("empty history")
    scores = [h.scores if hasattr(h, "scores") else h["scores"] for h in history]
    probe_ids = sorted(set().union(*scores))
    out: list[ProbeHealth] = []
    for pid in probe_ids:
        seq = [s.get(pid, math.nan) for s in scores]
        seq = [v for v in seq if v is not None and not math.isnan(v)]
        if not seq:
            out.append(ProbeHealth(pid, "unscored", math.nan, 0))
            continue
        passed = [v >= PASS for v in seq]
        flips = sum(a != b for a, b in zip(passed, passed[1:]))
        rate = sum(passed) / len(passed)
        if all(passed):
            kind = "saturated"
        elif not any(passed):
            kind = "unsolved"
        elif flips >= flaky_flips:
            kind = "flaky"
        elif passed[0] and not passed[-1]:
            kind = "regressed"
        elif not passed[0] and passed[-1]:
            kind = "improved"
        else:
            kind = "mixed"
        out.append(ProbeHealth(pid, kind, rate, flips))

    counts = dict(Counter(p.kind for p in out))
    n = len(out)
    scored = n - counts.get("unscored", 0)
    rec: list[str] = []
    if len(history) < 3:
        rec.append("fewer than 3 cycles recorded; probe classifications are provisional")
    if scored:
        sat = counts.get("saturated", 0) / scored
        uns = counts.get("unsolved", 0) / scored
        flk = counts.get("flaky", 0) / scored
        if sat >= 0.8:
            rec.append(
                f"{sat:.0%} of scored probes have passed in every cycle: the suite can detect "
                "regressions but not improvements. Add harder probes."
            )
        if uns >= 0.2:
            rec.append(f"{uns:.0%} of scored probes have never passed; they carry no signal yet.")
        if flk >= 0.05:
            rec.append(
                f"{flk:.0%} of probes flip without a trend. Check the scorer or agent sampling "
                "temperature, or raise warm-up so tolerances absorb the noise."
            )
    if not rec:
        rec.append("suite looks healthy: a mix of passing, failing and informative probes")
    return SuiteHealth(len(history), counts, out, rec)
