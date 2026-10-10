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

from .detector import SIGNALS, Thresholds

__all__ = ["MetricCalibration", "Calibration", "calibrate"]


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
) -> Calibration:
    """Derive :class:`Thresholds` from histories of healthy learning cycles.

    ``histories`` are monitor histories (lists of ``CycleReport`` or their dicts). Cycle 0 of
    each is its baseline and contributes no sample.
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
        floor = default * floor_fraction
        if candidate < floor:
            value, reason = floor, f"floored at {floor_fraction:.0%} of default"
        else:
            value = candidate
            reason = "loosened" if value > default else "tightened"
        chosen[spec.name] = value
        rows.append(MetricCalibration(spec.name, len(moves), observed, default, value, reason))
    return Calibration(Thresholds.from_dict(chosen), quantile, margin, rows)
