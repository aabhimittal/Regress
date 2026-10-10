"""Expressed confidence and calibration.

Rising confidence is only half of the collapse signature; it is benign when accuracy rises
with it. ``overconfidence`` (mean confidence minus accuracy) separates the two cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["ConfidenceStats", "confidence_stats", "expected_calibration_error"]


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


@dataclass
class ConfidenceStats:
    reported: int
    invalid: int  # NaN or out-of-range values the agent reported
    mean_confidence: float
    overconfidence: float  # mean confidence − accuracy, over probes with both
    ece: float


def confidence_stats(
    confidences: Sequence[float | None], scores: Sequence[float]
) -> ConfidenceStats:
    if len(confidences) != len(scores):
        raise ValueError("confidences and scores must align")
    valid_c: list[float] = []
    paired_c: list[float] = []
    paired_s: list[float] = []
    invalid = 0
    for c, s in zip(confidences, scores):
        if c is None:
            continue
        try:
            c = float(c)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if math.isnan(c) or not 0.0 <= c <= 1.0:
            invalid += 1
            continue
        valid_c.append(c)
        if not math.isnan(s):
            paired_c.append(c)
            paired_s.append(s)
    mean_c = float(np.mean(valid_c)) if valid_c else float("nan")
    if paired_c:
        pc, ps = np.array(paired_c), np.array(paired_s)
        over = float(pc.mean() - ps.mean())
        ece = expected_calibration_error(pc, ps)
    else:
        over = ece = float("nan")
    return ConfidenceStats(len(valid_c), invalid, mean_c, over, ece)
