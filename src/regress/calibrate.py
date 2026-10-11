"""Fit tolerances to *your* agent instead of trusting the defaults.

The default thresholds were tuned on one toy world. A stack whose healthy cycles naturally
churn 30 % of outputs, or whose memory rank wanders by 15 %, will alarm constantly on them;
a very stable stack will be under-protected.

``calibrate`` takes histories of cycles you believe were healthy and, for every signal,
measures how far that metric moved in its *harmful* direction from cycle 0. The tolerance
becomes a high quantile of those healthy moves times a safety margin. That turns the
false-alarm rate into a parameter you choose, rather than a side effect of a magic number.

Guard rails:

* At least ``min_samples`` healthy moves are needed per metric, otherwise the default is
  kept and the metric is flagged — a quantile of five numbers is not a calibration.
* Every tolerance has a floor (a fraction of the default), so a suspiciously stable healthy
  run cannot produce a hair trigger.
* Calibration is only as good as the "healthy" label. Calibrating on runs that were already
  collapsing teaches the detector to accept collapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .detector import SIGNALS, CusumConfig, Thresholds

__all__ = ["MetricCalibration", "Calibration", "calibrate", "calibrate_cusum", "family_quantile"]


def family_quantile(horizon: int, n_signals: int, alpha: float = 0.05) -> float:
    """Per-test quantile that keeps the family-wise false-alarm rate near ``alpha``.

    Running ``n_signals`` checks every cycle for ``horizon`` cycles is ``horizon × n_signals``
    tests. A 99.9 % per-test threshold gives ~1 − 0.999^(365×12) ≈ 99 % odds of at least one
    false signal in a year of nightly cycles. Bonferroni: each test gets α / (horizon × n).
    """
    if horizon < 1 or n_signals < 1 or not 0 < alpha < 1:
        raise ValueError("need horizon >= 1, n_signals >= 1, 0 < alpha < 1")
    return 1.0 - alpha / (horizon * n_signals)


@dataclass
class MetricCalibration:
    signal: str
    samples: int
    observed: float  # the chosen quantile of healthy harmful moves (nan if too few)
    default: float
    chosen: float
    reason: str


@dataclass
class Calibration:
    thresholds: Thresholds
    quantile: float
    margin: float
    metrics: list[MetricCalibration] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"calibrated at the {self.quantile:.1%} quantile × {self.margin:g} margin"]
        for m in self.metrics:
            obs = "  –  " if math.isnan(m.observed) else f"{m.observed:.4f}"
            lines.append(
                f"  {m.signal:<16} n={m.samples:<4} healthy={obs}  default={m.default:.4f}  "
                f"→ {m.chosen:.4f}  ({m.reason})"
            )
        return "\n".join(lines)


def _moves(histories: Sequence[Sequence], metric: str, direction: int, mode: str, skip: int) -> list[float]:
    out: list[float] = []
    for hist in histories:
        if len(hist) <= skip:
            continue
        base = _metrics(hist[0]).get(metric, float("nan"))
        if base is None or not math.isfinite(base) or (mode == "rel" and base <= 0):
            continue
        for h in hist[skip:]:
            v = _metrics(h).get(metric, float("nan"))
            if v is None or not math.isfinite(v):
                continue
            move = direction * (v - base)
            out.append(move / abs(base) if mode == "rel" else move)
    return out


def _metrics(h) -> Mapping[str, float]:
    return h.metrics if hasattr(h, "metrics") else h["metrics"]


def calibrate(
    histories: Sequence[Sequence],
    quantile: float = 0.999,
    margin: float = 1.5,
    floor_fraction: float = 0.25,
    min_samples: int = 10,
    base: Thresholds | None = None,
    horizon: int | None = None,
    alpha: float = 0.05,
) -> Calibration:
    """Derive :class:`Thresholds` from histories of healthy learning cycles.

    ``histories`` are monitor histories (lists of ``CycleReport`` or their dicts). Cycle 0 of
    each is its baseline and contributes no sample.

    With ``horizon`` (the number of cycles you expect to monitor), each tolerance is also
    raised to a Gaussian tail estimate at the family-wise quantile
    ``1 − alpha / (horizon × signals)``: median + z·1.4826·MAD of the healthy moves. Empirical
    quantiles cannot reach that far into the tail with a few hundred samples, so the tail is
    extrapolated. The extrapolation assumes roughly Gaussian healthy noise; heavy tails will
    still leak false alarms.
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be in [0.5, 1)")
    if margin < 1.0:
        raise ValueError("margin must be >= 1 (a margin below 1 guarantees false alarms)")
    if not histories:
        raise ValueError("need at least one healthy history")
    base = base or Thresholds()
    chosen = base.to_dict()
    rows: list[MetricCalibration] = []
    tail_z = None
    if horizon is not None:
        from statistics import NormalDist

        n_specs = sum(1 for s in SIGNALS if not s.weak)
        tail_z = NormalDist().inv_cdf(family_quantile(horizon, n_specs, alpha))
    for spec in SIGNALS:
        if spec.name not in chosen:
            continue
        default = getattr(base, spec.name)
        moves = _moves(histories, spec.metric, spec.direction, spec.mode, skip=1)
        if len(moves) < min_samples:
            rows.append(MetricCalibration(spec.name, len(moves), float("nan"), default, default,
                                          f"kept default: fewer than {min_samples} healthy samples"))
            continue
        observed = float(np.quantile(moves, quantile))
        candidate = max(observed, 0.0) * margin
        if tail_z is not None:
            arr = np.asarray(moves, dtype=float)
            med = float(np.median(arr))
            spread = 1.4826 * float(np.median(np.abs(arr - med)))
            candidate = max(candidate, med + tail_z * spread)
        floor = default * floor_fraction
        if candidate < floor:
            value, reason = floor, f"floored at {floor_fraction:.0%} of default"
        else:
            value = candidate
            reason = "loosened" if value > default else "tightened"
        chosen[spec.name] = value
        rows.append(MetricCalibration(spec.name, len(moves), observed, default, value, reason))
    return Calibration(Thresholds.from_dict(chosen), quantile, margin, rows)


def calibrate_cusum(
    histories: Sequence[Sequence],
    k: float = 0.25,
    margin: float = 1.5,
    floor: float = 2.0,
) -> CusumConfig:
    """Pick the CUSUM decision threshold ``h`` from healthy histories.

    Replays each history's recorded standardised moves (``CycleReport.moves``) through the
    CUSUM and sets ``h`` to the largest statistic any healthy run ever reached, times
    ``margin``, never below ``floor``. Healthy runs should be as long as the horizon you
    intend to monitor: a CUSUM's healthy maximum grows with run length.
    """
    if k < 0 or margin < 1 or floor <= 0:
        raise ValueError("need k >= 0, margin >= 1, floor > 0")
    peak = 0.0
    for hist in histories:
        rows = [h.moves if hasattr(h, "moves") else h.get("moves", {}) for h in hist[1:]]
        names = {n for r in rows for n in r}
        for spec in SIGNALS:
            if spec.weak or spec.name not in names:
                continue
            series = [r.get(spec.name, float("nan")) for r in rows]
            s = 0.0
            for z in series:  # running maximum, not just the final value
                if z is None or not math.isfinite(z):
                    continue
                s = max(0.0, s + min(z, 3.0) - k)
                peak = max(peak, s)
    return CusumConfig(k=k, h=max(floor, peak * margin))
