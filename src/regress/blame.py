"""Point at the memories behind a regression.

An alarm says *that* the agent degraded; this module ranks *which memories* are implicated,
so the fix can be "quarantine these five items" rather than "roll back a week of learning".

For a target cycle it takes the probes that regressed against cycle 0 (passed then, fail
now) and, for every memory those probes retrieved, reports:

* ``regressed_hits`` — how many regressed probes retrieved it.
* ``lift`` — how much more often regressed probes retrieve it than probes in general. A
  hub that every probe retrieves has lift ≈ 1 even with many hits; a memory that shows
  up *specifically* where things broke has lift ≫ 1.
* ``share_base`` / ``share_now`` — its share of all retrievals at cycle 0 and now. A big
  gain is the hub-formation pattern behind retrieval collapse.
* ``first_seen`` — the first recorded cycle in which any probe retrieved it. A suspect that
  first appears at the estimated onset is a strong lead.

This is correlational. A memory retrieved by failing probes is a suspect, not a proven
cause; the honest way to confirm is to remove it and re-run the suite.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .detector import ALARM

__all__ = ["Suspect", "BlameReport", "blame"]

PASS = 0.5


@dataclass
class Suspect:
    memory_id: str
    regressed_hits: int
    total_hits: int
    lift: float
    share_base: float
    share_now: float
    first_seen: int | None
    probes: list[str] = field(default_factory=list)  # up to 5 regressed probes that used it

    @property
    def share_gain(self) -> float:
        return self.share_now - self.share_base


@dataclass
class BlameReport:
    cycle: int
    regressed: list[str]
    suspects: list[Suspect]
    note: str = ""


def _shares(retrieved: dict[str, list[str]]) -> tuple[Counter, int]:
    c: Counter = Counter()
    for ids in retrieved.values():
        c.update(set(ids))  # each memory counts once per probe
    return c, sum(c.values())


def blame(history: Sequence, cycle: int | None = None, top: int = 10) -> BlameReport:
    """Rank memories implicated in the regressions at ``cycle``.

    ``history`` is a monitor history (``CycleReport`` objects, or anything with ``cycle``,
    ``scores``, ``retrieved`` and ``status``). ``cycle`` defaults to the first alarm, or the
    last cycle when nothing alarmed.
    """
    if not history:
        raise ValueError("empty history")
    base = history[0]
    if cycle is None:
        target = next((h for h in history if h.status == ALARM), history[-1])
    else:
        target = next((h for h in history if h.cycle == cycle), None)
        if target is None:
            raise ValueError(f"cycle {cycle} is not in the history")
    if not getattr(target, "retrieved", None):
        return BlameReport(target.cycle, [], [], "no per-probe retrievals recorded for this cycle")

    regressed = sorted(
        p for p, s in target.scores.items()
        if base.scores.get(p, float("nan")) >= PASS > s
    )
    if not regressed:
        return BlameReport(target.cycle, [], [], "no probe regressed against cycle 0")

    now_counts, now_total = _shares(target.retrieved)
    base_counts, base_total = _shares(getattr(base, "retrieved", {}) or {})
    reg_counts: Counter = Counter()
    reg_probes: dict[str, list[str]] = {}
    for p in regressed:
        for m in dict.fromkeys(target.retrieved.get(p, [])):
            reg_counts[m] += 1
            reg_probes.setdefault(m, []).append(p)

    first_seen: dict[str, int] = {}
    for h in history:
        if h.cycle > target.cycle:
            break
        for ids in (getattr(h, "retrieved", None) or {}).values():
            for m in ids:
                first_seen.setdefault(m, h.cycle)

    n_probes = max(1, len(target.retrieved))
    suspects = []
    for m, hits in reg_counts.items():
        total = now_counts[m]
        lift = (hits / len(regressed)) / (total / n_probes) if total else float("nan")
        suspects.append(Suspect(
            memory_id=m,
            regressed_hits=hits,
            total_hits=total,
            lift=lift,
            share_base=base_counts[m] / base_total if base_total else 0.0,
            share_now=total / now_total if now_total else 0.0,
            first_seen=first_seen.get(m),
            probes=reg_probes[m][:5],
        ))
    # Most regressed probes first; among equals, the more specific (higher lift) suspect.
    suspects.sort(key=lambda s: (-s.regressed_hits, -s.lift, -s.share_gain, s.memory_id))
    return BlameReport(target.cycle, regressed, suspects[:top])
