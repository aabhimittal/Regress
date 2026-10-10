"""Behavioural drift on the frozen probes.

Two complementary views, because each is blind where the other sees:

* ``output_jsd`` compares the *marginal* distribution of outputs. It catches an agent that
  starts answering "B" to everything, but is exactly zero if the agent permutes its
  answers (every "A" becomes "B" and every "B" becomes "A").
* ``change_rate`` compares outputs *per probe*. It catches the permutation, but cannot tell
  churn that is harmless from churn that concentrates on one answer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["DriftStats", "jensen_shannon", "drift"]

PASS_THRESHOLD = 0.5


def jensen_shannon(p_counts: Mapping[str, float], q_counts: Mapping[str, float]) -> float:
    """Jensen–Shannon divergence in bits, in [0, 1]. NaN if either side is empty."""
    p_total = float(sum(p_counts.values()))
    q_total = float(sum(q_counts.values()))
    if p_total <= 0 or q_total <= 0:
        return float("nan")
    keys = set(p_counts) | set(q_counts)
    p = np.array([p_counts.get(k, 0.0) / p_total for k in keys])
    q = np.array([q_counts.get(k, 0.0) / q_total for k in keys])
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(min(1.0, max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


@dataclass
class DriftStats:
    accuracy_delta: float
    output_jsd: float
    change_rate: float
    regressions: int  # probes that passed at baseline and now fail
    fixes: int  # probes that failed at baseline and now pass
    compared: int  # probes present in both cycles


def drift(
    base_keys: Mapping[str, str | None],
    cur_keys: Mapping[str, str | None],
    base_scores: Mapping[str, float],
    cur_scores: Mapping[str, float],
) -> DriftStats:
    """Compare two cycles' per-probe output keys and scores.

    A ``None`` key means the probe errored; errors are their own output class so an agent
    that starts crashing registers as drift instead of vanishing from the comparison.
    """
    common = sorted(set(base_keys) & set(cur_keys))
    base_dist = Counter("<error>" if base_keys[k] is None else base_keys[k] for k in common)
    cur_dist = Counter("<error>" if cur_keys[k] is None else cur_keys[k] for k in common)
    jsd = jensen_shannon(base_dist, cur_dist)
    changed = sum(1 for k in common if base_keys[k] != cur_keys[k])
    change_rate = changed / len(common) if common else float("nan")

    regressions = fixes = 0
    deltas: list[float] = []
    for k in common:
        b, c = base_scores.get(k, math.nan), cur_scores.get(k, math.nan)
        if math.isnan(b) or math.isnan(c):
            continue
        deltas.append(c - b)
        if b >= PASS_THRESHOLD > c:
            regressions += 1
        elif c >= PASS_THRESHOLD > b:
            fixes += 1
    acc_delta = float(np.mean(deltas)) if deltas else float("nan")
    return DriftStats(acc_delta, jsd, change_rate, regressions, fixes, len(common))


def mean_score(scores: Sequence[float]) -> float:
    vals = [s for s in scores if not math.isnan(s)]
    return float(np.mean(vals)) if vals else float("nan")
