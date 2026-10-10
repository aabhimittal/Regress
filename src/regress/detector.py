"""Combination alarm.

No single metric is trusted to raise an alarm. Each metric is compared with the cycle-0
baseline (or the median of a warm-up window), producing *signals*; signals are grouped
into *categories* (drift, diversity, retrieval, confidence, integrity), and the status is:

* ``OK``     — no signals.
* ``WATCH``  — at least one signal, but no alarming combination.
* ``ALARM``  — the collapse signature (diversity falling *while* expressed confidence
  rises), or signals in at least ``min_categories`` independent categories, held for
  ``patience`` consecutive cycles.

Everything is compared with the *baseline*, never with the previous cycle. A starter that
sours by 3 % a day never looks different from yesterday; it only looks different from the
first day. Step-to-step comparison is how slow collapse slips through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = ["Thresholds", "SignalSpec", "Signal", "Assessment", "CollapseDetector", "CusumConfig", "SIGNALS"]

OK, WATCH, ALARM = "OK", "WATCH", "ALARM"


@dataclass(frozen=True)
class Thresholds:
    accuracy_drop: float = 0.05
    output_shift: float = 0.15  # JSD (bits) between output distributions
    churn: float = 0.35  # share of probes whose output changed
    diversity_drop: float = 0.20  # relative drop in effective rank
    spread_drop: float = 0.20  # relative drop in mean pairwise cosine distance
    near_duplicates: float = 0.15  # absolute rise in near-duplicate share
    entropy_drop: float = 0.15  # absolute drop in normalised retrieval entropy
    retrieval_hub: float = 0.15  # absolute rise in the top-1 memory's share
    confidence_rise: float = 0.05
    overconfidence: float = 0.10
    probe_overfit: float = 0.05  # lower 95% bound of (public gain − held-out gain)
    error_rate: float = 0.05
    embedding_changed: float = 0.2  # 1 − mean cosine between a memory's old and new vector

    def to_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "Thresholds":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown threshold(s): {sorted(unknown)}")
        for k, v in d.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"threshold {k} must be a finite non-negative number, got {v!r}")
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass(frozen=True)
class SignalSpec:
    name: str
    metric: str
    category: str
    direction: int  # +1: higher is worse, -1: lower is worse
    mode: str  # "abs" or "rel"
    weak: bool = False  # weak signals raise WATCH but never count toward an ALARM
    se_metric: str | None = None  # standard error of the move; gates on statistical significance


SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("accuracy_drop", "accuracy", "drift", -1, "abs", se_metric="accuracy_delta_se"),
    SignalSpec("output_shift", "output_jsd", "drift", +1, "abs"),
    SignalSpec("churn", "change_rate", "drift", +1, "abs", weak=True),
    SignalSpec("diversity_drop", "effective_rank", "diversity", -1, "rel", se_metric="effective_rank_se"),
    SignalSpec("spread_drop", "mean_cosine_distance", "diversity", -1, "rel", se_metric="mean_cosine_distance_se"),
    SignalSpec("near_duplicates", "near_duplicate_fraction", "diversity", +1, "abs"),
    SignalSpec("entropy_drop", "retrieval_entropy", "retrieval", -1, "abs", se_metric="retrieval_entropy_se"),
    SignalSpec("retrieval_hub", "retrieval_top1_share", "retrieval", +1, "abs"),
    SignalSpec("confidence_rise", "mean_confidence", "confidence", +1, "abs", se_metric="mean_confidence_se"),
    SignalSpec("overconfidence", "overconfidence", "confidence", +1, "abs"),
    SignalSpec("probe_overfit", "probe_overfit_gap", "integrity", +1, "abs"),
    SignalSpec("error_rate", "error_rate", "integrity", +1, "abs"),
    # The agent's own embedder changed under its memories. Weak: it is a warning that the
    # agent-ruler diversity numbers moved for a reason unrelated to memory health.
    SignalSpec("embedding_changed", "embedding_drift", "integrity", +1, "abs", weak=True),
)


@dataclass(frozen=True)
class CusumConfig:
    """Sequential (CUSUM) test on each signal's standardised harmful move.

    Each cycle contributes ``z = move / tolerance`` (clipped to ``clip``); the statistic is
    ``S_t = max(0, S_{t−1} + z_t − k)`` and fires when ``S_t > h``. A metric sitting at 75 % of
    its tolerance never trips the per-cycle test, but with k = 0.5 it accumulates 0.25 per
    cycle and fires after ~16 cycles. Conversely, isolated noise drains away, so over a long
    horizon CUSUM raises far fewer false alarms than repeating a per-cycle test.
    """

    k: float = 0.5
    h: float = 4.0
    clip: float = 3.0

    def __post_init__(self) -> None:
        if self.k < 0 or self.h <= 0 or self.clip <= 0:
            raise ValueError("CUSUM needs k >= 0, h > 0, clip > 0")


def cusum(moves: Sequence[float], k: float, clip: float = 3.0) -> float:
    """Final CUSUM statistic over standardised moves (NaN entries are skipped)."""
    s = 0.0
    for z in moves:
        if z is None or not math.isfinite(z):
            continue
        s = max(0.0, s + min(z, clip) - k)
    return s


@dataclass
class Signal:
    name: str
    category: str
    metric: str
    baseline: float
    value: float
    excess: float  # how far past baseline in the bad direction (abs or relative units)
    tolerance: float
    weak: bool = False

    def describe(self) -> str:
        return (
            f"{self.name}: {self.metric} {self.baseline:.3g} → {self.value:.3g} "
            f"(excess {self.excess:.3g} > tol {self.tolerance:.3g})"
        )


@dataclass
class Assessment:
    status: str
    signals: list[Signal] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    collapse_signature: bool = False
    undefined: list[str] = field(default_factory=list)  # metrics that were NaN this cycle
    reason: str = ""
    moves: dict[str, float] = field(default_factory=dict)  # standardised harmful move per signal


def _robust_baseline(values: Sequence[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), 0.0
    med = float(np.median(vals))
    if vals.size < 3:
        return med, 0.0
    return med, float(1.4826 * np.median(np.abs(vals - med)))


class CollapseDetector:
    def __init__(
        self,
        thresholds: Thresholds | None = None,
        min_categories: int = 2,
        patience: int = 1,
        noise_z: float = 3.0,
        significance_z: float = 1.96,
        specs: Sequence[SignalSpec] = SIGNALS,
        cusum: CusumConfig | None = None,
    ) -> None:
        if min_categories < 1:
            raise ValueError("min_categories must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.thresholds = thresholds or Thresholds()
        self.min_categories = min_categories
        self.patience = patience
        self.noise_z = noise_z
        self.significance_z = significance_z
        self.specs = tuple(specs)
        self.cusum = cusum

    def config(self) -> dict:
        return {
            "thresholds": self.thresholds.to_dict(),
            "min_categories": self.min_categories,
            "patience": self.patience,
            "noise_z": self.noise_z,
            "significance_z": self.significance_z,
            "cusum": None if self.cusum is None else {"k": self.cusum.k, "h": self.cusum.h, "clip": self.cusum.clip},
        }

    @classmethod
    def from_config(cls, cfg: Mapping) -> "CollapseDetector":
        return cls(
            thresholds=Thresholds.from_dict(cfg.get("thresholds", {})),
            min_categories=int(cfg.get("min_categories", 2)),
            patience=int(cfg.get("patience", 1)),
            noise_z=float(cfg.get("noise_z", 3.0)),
            significance_z=float(cfg.get("significance_z", 1.96)),
            cusum=CusumConfig(**cfg["cusum"]) if cfg.get("cusum") else None,
        )

    def signals(
        self, current: Mapping[str, float], baseline_window: Sequence[Mapping[str, float]]
    ) -> tuple[list[Signal], list[str]]:
        sig, undefined, _ = self.measure(current, baseline_window)
        return sig, undefined

    def measure(
        self,
        current: Mapping[str, float],
        baseline_window: Sequence[Mapping[str, float]],
        tol_scale: float = 1.0,
        use_se: bool = True,
    ) -> tuple[list[Signal], list[str], dict[str, float]]:
        """Signals, undefined metrics, and each signal's standardised move (move / tolerance).

        ``tol_scale`` < 1 asks "is anything *near* its tolerance?" (used to decide whether a
        cheap screening evaluation must escalate to the full suite). ``use_se=False`` drops
        the significance gate, for the same purpose: an uncertain move is a reason to look
        closer, not to relax.
        """
        out: list[Signal] = []
        undefined: list[str] = []
        moves: dict[str, float] = {}
        for spec in self.specs:
            value = current.get(spec.metric, float("nan"))
            base, noise = _robust_baseline([b.get(spec.metric, float("nan")) for b in baseline_window])
            if value is None or math.isnan(value) or math.isnan(base):
                undefined.append(spec.metric)
                continue
            tol = getattr(self.thresholds, spec.name, None)
            if tol is None:
                continue
            bad_move = spec.direction * (value - base)
            scale = 1.0
            if spec.mode == "rel":
                if base <= 0:
                    undefined.append(spec.metric)
                    continue
                scale = abs(base)
            excess, noise = bad_move / scale, noise / scale
            eff_tol = max(tol, self.noise_z * noise) * tol_scale
            if eff_tol > 0:
                moves[spec.name] = excess / max(tol, self.noise_z * noise)
            if use_se and spec.se_metric is not None:
                # Standard error of the move (paired per-probe changes, or a bootstrap over
                # memories/probes). A move that sampling noise could explain is not called.
                se = current.get(spec.se_metric, float("nan"))
                if se is not None and math.isfinite(se):
                    eff_tol = max(eff_tol, self.significance_z * se / scale)
            if excess > eff_tol + 1e-12:
                out.append(Signal(spec.name, spec.category, spec.metric, base, value, excess, eff_tol, spec.weak))
        return out, undefined, moves

    def assess(
        self,
        current: Mapping[str, float],
        baseline_window: Sequence[Mapping[str, float]],
        recent_alarm_conditions: Sequence[bool] = (),
        prior_moves: Sequence[Mapping[str, float]] = (),
    ) -> Assessment:
        """Assess one cycle.

        ``recent_alarm_conditions`` holds whether the alarm *condition* (not status) held in
        each of the immediately preceding cycles, newest last; it implements ``patience``.
        ``prior_moves`` are the standardised moves of earlier post-baseline cycles, oldest
        first; with a :class:`CusumConfig` they feed the sequential test.
        """
        signals, undefined, moves = self.measure(current, baseline_window)
        if self.cusum is not None:
            fired = {s.name for s in signals}
            for spec in self.specs:
                if spec.weak or spec.name in fired or spec.name not in moves:
                    continue
                series = [m.get(spec.name, float("nan")) for m in prior_moves] + [moves[spec.name]]
                stat = cusum(series, self.cusum.k, self.cusum.clip)
                if stat > self.cusum.h:
                    base, _ = _robust_baseline([b.get(spec.metric, float("nan")) for b in baseline_window])
                    signals.append(Signal(f"persistent_{spec.name}", spec.category, spec.metric, base,
                                          current.get(spec.metric, float("nan")), stat, self.cusum.h, False))
        strong = [s for s in signals if not s.weak]
        categories = sorted({s.category for s in strong})
        names = {s.name.removeprefix("persistent_") for s in strong}
        diversity_down = bool(names & {"diversity_drop", "spread_drop", "near_duplicates"})
        confidence_up = "confidence_rise" in names
        signature = diversity_down and confidence_up
        condition = signature or len(categories) >= self.min_categories

        streak = 1 if condition else 0
        if condition:
            for prev in reversed(list(recent_alarm_conditions)):
                if not prev:
                    break
                streak += 1

        if condition and streak >= self.patience:
            status = ALARM
            reason = (
                "collapse signature: diversity falling while expressed confidence rises"
                if signature
                else f"{len(categories)} independent categories degraded: {', '.join(categories)}"
            )
        elif signals:
            status = WATCH
            reason = (
                f"alarm condition held {streak}/{self.patience} cycles"
                if condition
                else f"{len(signals)} signal(s), no alarming combination"
            )
        else:
            status = OK
            reason = "no signals"
        a = Assessment(status, signals, categories, signature, undefined, reason)
        a.moves = moves
        return a

    @staticmethod
    def alarm_condition(a: Assessment, min_categories: int) -> bool:
        return a.collapse_signature or len(a.categories) >= min_categories
