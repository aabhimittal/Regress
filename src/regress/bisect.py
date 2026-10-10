"""Find where degradation began.

Two different questions, often confused:

1. *Which checkpoint first fails?* — ``bisect_checkpoints``. Like ``git bisect``: you have
   saved agent states for many cycles, evaluating one is expensive, and you want the first
   bad one in O(log n) evaluations. Supports ``None`` = "skip" (checkpoint unloadable) and
   re-checks the boundary to catch flaky evaluators.

2. *When did the decline start?* — ``estimate_onset``. With slow decay, the first cycle that
   *fails a threshold* can be many cycles after the cycle where the decline *began*. Bisecting
   on the alarm finds the former; the fix usually lives at the latter. ``estimate_onset`` fits
   a "flat, then bend" (hinge) model and a "flat, then step" model to a metric series and
   returns the best change point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .detector import SIGNALS

__all__ = ["BisectResult", "bisect_checkpoints", "Onset", "estimate_onset", "locate_onset"]


class BisectError(RuntimeError):
    pass


@dataclass
class BisectResult:
    first_bad: int | None  # None: every checkpoint after the good one is good
    last_good: int | None
    evaluations: dict[int, bool | None] = field(default_factory=dict)
    skipped: list[int] = field(default_factory=list)
    # When skipped checkpoints sit on the boundary, the answer is a range, not a point.
    candidates: list[int] = field(default_factory=list)
    consistent: bool = True  # False if a re-check disagreed with the first verdict
    notes: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.first_bad is not None and len(self.candidates) <= 1 and self.consistent


def bisect_checkpoints(
    cycles: Sequence[int],
    is_bad: Callable[[int], bool | None],
    assume_first_good: bool = False,
    recheck: bool = True,
) -> BisectResult:
    """Binary-search ``cycles`` (ascending) for the first checkpoint where ``is_bad`` is True.

    ``is_bad`` may return None for a checkpoint that cannot be tested; it is skipped and a
    neighbour is tried instead. Assumes monotonicity (once bad, stays bad); with
    ``recheck=True`` the boundary pair is evaluated a second time and any disagreement is
    reported via ``consistent=False`` instead of being silently trusted.
    """
    cyc = list(cycles)
    if not cyc:
        raise ValueError("no checkpoints to bisect")
    if any(b <= a for a, b in zip(cyc, cyc[1:])):
        raise ValueError("cycles must be strictly increasing")
    evals: dict[int, bool | None] = {}
    skipped: list[int] = []
    res = BisectResult(None, None, evals, skipped)

    def test(i: int) -> bool | None:
        c = cyc[i]
        if c not in evals:
            v = is_bad(c)
            evals[c] = None if v is None else bool(v)
            if v is None:
                skipped.append(c)
        return evals[c]

    lo = 0
    if not assume_first_good:
        v = test(0)
        if v is None:
            raise BisectError(f"baseline checkpoint {cyc[0]} could not be evaluated")
        if v:
            raise BisectError(f"baseline checkpoint {cyc[0]} is already bad; nothing to bisect")
    # Find a bad upper bound from the end, skipping untestable checkpoints.
    hi = len(cyc) - 1
    while hi > lo and test(hi) is None:
        hi -= 1
    if hi == lo or not evals.get(cyc[hi]):
        res.last_good = cyc[hi]
        res.notes.append("no bad checkpoint found")
        if hi < len(cyc) - 1:
            res.notes.append(f"checkpoints {cyc[hi + 1:]} could not be evaluated")
        return res

    # Invariant: cyc[lo] good, cyc[hi] bad, everything strictly between untested or skipped.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = test(mid)
        if v is None:
            # Try the remaining untested checkpoints in (lo, hi), nearest to mid first.
            pending = sorted(
                (j for j in range(lo + 1, hi) if cyc[j] not in evals), key=lambda j: (abs(j - mid), j)
            )
            found = next((j for j in pending if test(j) is not None), None)
            if found is None:
                break  # every checkpoint strictly between lo and hi is untestable
            mid, v = found, evals[cyc[found]]
        if v:
            hi = mid
        else:
            lo = mid

    res.first_bad, res.last_good = cyc[hi], cyc[lo]
    res.candidates = [c for c in cyc[lo + 1 : hi + 1]]
    if len(res.candidates) > 1:
        res.notes.append(
            f"skipped checkpoints {res.candidates[:-1]} hide the exact boundary; "
            f"first bad is one of {res.candidates}"
        )
    if recheck:
        again_bad, again_good = is_bad(cyc[hi]), is_bad(cyc[lo])
        if again_bad is False or again_good is True:
            res.consistent = False
            res.notes.append(
                "re-evaluation disagreed with the first verdict at the boundary; the evaluator "
                "is noisy or degradation is not monotonic — raise probe count or average runs"
            )
    return res


@dataclass
class Onset:
    metric: str
    cycle: int | None
    model: str  # "hinge", "step" or "none"
    improvement: float  # fractional SSE reduction vs. a flat line, in [0, 1]


def _hinge_sse(t: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """y = a + b·max(0, t − t_k): flat until t_k, then linear. Returns (sse, slope)."""
    x = np.maximum(0.0, t - t[k])
    A = np.column_stack([np.ones_like(t), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r @ r), float(coef[1])


def _step_sse(y: np.ndarray, k: int) -> tuple[float, float]:
    """Two flat segments split before index k. Returns (sse, jump)."""
    a, b = y[:k], y[k:]
    return float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()), float(b.mean() - a.mean())


def estimate_onset(
    cycles: Sequence[int],
    values: Sequence[float],
    metric: str = "",
    min_improvement: float = 0.5,
    direction: int = 0,
) -> Onset:
    """Estimate the cycle at which ``values`` departed from its initial level.

    Returns ``cycle=None`` when neither change-point model explains at least
    ``min_improvement`` of the variance a flat line leaves — i.e. there is no clear onset.
    ``direction`` (+1: rising is bad, −1: falling is bad, 0: either) discards fits whose
    change goes the harmless way, so an early improvement is not mistaken for the onset.
    """
    pts = [(c, v) for c, v in zip(cycles, values) if v is not None and math.isfinite(v)]
    if len(pts) < 4:
        return Onset(metric, None, "none", 0.0)
    t = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    flat = float(((y - y.mean()) ** 2).sum())
    if flat <= 1e-15:
        return Onset(metric, None, "none", 0.0)
    # Each candidate is (sse, index of the first departed point, model).
    best: tuple[float, int | None, str] = (math.inf, None, "none")
    for k in range(0, len(t) - 2):  # hinge at t[k]: t[k] still flat, t[k+1] departs
        h, slope = _hinge_sse(t, y, k)
        if direction and slope * direction <= 0:
            continue
        if h < best[0] - 1e-12:
            best = (h, k + 1, "hinge")
    for k in range(1, len(t) - 1):  # step between t[k-1] and t[k]
        st, jump = _step_sse(y, k)
        if direction and jump * direction <= 0:
            continue
        # A step is the more specific claim; prefer it only when clearly better.
        if st < 0.8 * best[0]:
            best = (st, k, "step")
    sse, k, model = best
    improvement = 1.0 - sse / flat
    if k is None or improvement < min_improvement:
        return Onset(metric, None, "none", max(0.0, improvement))
    return Onset(metric, int(t[k]), model, improvement)


def locate_onset(
    history: Sequence[object],
    metrics: Sequence[str] | None = None,
    upto: int | None = None,
    horizon: int = 2,
) -> tuple[int | None, list[Onset]]:
    """Estimate onset per metric over a monitor history, and a consensus (lower median) cycle.

    ``history`` is a sequence of objects with ``.cycle`` and ``.metrics`` (``CycleReport``).
    By default the metrics are those that raised non-weak signals anywhere in the history.

    Only cycles up to ``upto`` are fitted. When ``upto`` is None and the history contains
    an alarm, it defaults to the first alarm plus ``horizon`` cycles: once a collapse is
    under way its metrics saturate and oscillate, and fitting that tail drags the
    estimated bend later than where the decline actually began.
    """
    if upto is None:
        first = next((h for h in history if getattr(h, "status", None) == "ALARM"), None)
        if first is not None:
            upto = first.cycle + horizon  # type: ignore[attr-defined]
    hist = [h for h in history if upto is None or h.cycle <= upto]  # type: ignore[attr-defined]
    directions = {s.metric: s.direction for s in SIGNALS}
    if metrics is None:
        seen: dict[str, None] = {}
        for h in hist:
            for s in h.signals:  # type: ignore[attr-defined]
                if not s.weak:
                    seen.setdefault(s.metric)
        metrics = list(seen)
    cycles = [h.cycle for h in hist]  # type: ignore[attr-defined]
    onsets = [
        estimate_onset(
            cycles,
            [h.metrics.get(m, float("nan")) for h in hist],  # type: ignore[attr-defined]
            m,
            direction=directions.get(m, 0),
        )
        for m in metrics
    ]
    found = sorted(o.cycle for o in onsets if o.cycle is not None)
    # Lower median: robust to one metric reacting late, and always an observed estimate.
    consensus = found[(len(found) - 1) // 2] if found else None
    return consensus, onsets
