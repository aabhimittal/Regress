"""The regression monitor: run the frozen suite after every learning cycle."""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._canon import output_key as default_output_key
from .core import Agent, MemorySnapshot, Scorer, exact_match
from .detector import ALARM, Assessment, CollapseDetector, Signal
from .metrics import confidence_stats, diversity, drift, mean_score, retrieval_stats
from .probes import ProbeSuite, ProbeSuiteTampered

__all__ = ["CycleReport", "RegressionMonitor", "ProbeSuiteMismatch"]

FORMAT_VERSION = 1


class ProbeSuiteMismatch(ValueError):
    """A saved history was recorded against a different probe suite."""


@dataclass
class CycleReport:
    cycle: int
    status: str
    metrics: dict[str, float]
    signals: list[Signal]
    categories: list[str]
    collapse_signature: bool
    reason: str
    output_keys: dict[str, str | None]
    scores: dict[str, float]
    errors: dict[str, str] = field(default_factory=dict)
    undefined: list[str] = field(default_factory=list)
    alarm_condition: bool = False

    @property
    def alarmed(self) -> bool:
        return self.status == ALARM

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = {k: _json_float(v) for k, v in self.metrics.items()}
        d["scores"] = {k: _json_float(v) for k, v in self.scores.items()}
        d["signals"] = [{k: _json_float(v) for k, v in asdict(s).items()} for s in self.signals]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CycleReport":
        return cls(
            cycle=int(d["cycle"]),
            status=d["status"],
            metrics={k: _from_json_float(v) for k, v in d["metrics"].items()},
            signals=[Signal(**{k: _from_json_float(v) for k, v in s.items()}) for s in d["signals"]],
            categories=list(d["categories"]),
            collapse_signature=bool(d["collapse_signature"]),
            reason=d["reason"],
            output_keys=dict(d["output_keys"]),
            scores={k: _from_json_float(v) for k, v in d["scores"].items()},
            errors=dict(d.get("errors", {})),
            undefined=list(d.get("undefined", [])),
            alarm_condition=bool(d.get("alarm_condition", False)),
        )


def _json_float(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, (np.floating,)):
        return _json_float(float(v))
    return v


def _from_json_float(v: Any) -> Any:
    if v is None:
        return float("nan")
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return v


class RegressionMonitor:
    """Freeze a probe suite at cycle 0, then measure every consolidation cycle against it.

    Parameters
    ----------
    suite:
        The frozen probes. Its fingerprint is pinned at construction and re-verified on
        every evaluation, so a mutated probe raises rather than shifting the baseline.
    scorer:
        ``(probe, output) -> float in [0, 1]``. Defaults to exact match.
    output_key:
        Maps an output to a hashable identity for drift. Supply one for free-text outputs
        (e.g. normalise whitespace, or bucket by an embedding cluster) so that cosmetic
        rewording does not register as drift.
    warmup:
        Number of initial cycles whose median forms the baseline. With ``warmup >= 3`` the
        spread of those cycles also widens tolerances, so a noisy agent needs a bigger move
        to trip a signal.
    """

    def __init__(
        self,
        suite: ProbeSuite,
        scorer: Scorer = exact_match,
        detector: CollapseDetector | None = None,
        output_key: Callable[[Any], str] = default_output_key,
        warmup: int = 1,
    ) -> None:
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        self.suite = suite
        self.scorer = scorer
        self.detector = detector or CollapseDetector()
        self.output_key = output_key
        self.warmup = warmup
        self.fingerprint = suite.fingerprint
        self.history: list[CycleReport] = []

    # ------------------------------------------------------------------ evaluation
    @property
    def baseline(self) -> CycleReport | None:
        return self.history[0] if self.history else None

    def _run_probes(self, agent: Agent) -> dict[str, Any]:
        keys: dict[str, str | None] = {}
        scores: dict[str, float] = {}
        confs: list[float | None] = []
        conf_scores: list[float] = []
        retrieved: list[Sequence[str]] = []
        errors: dict[str, str] = {}
        for probe in self.suite:
            try:
                resp = agent.respond(probe)
                key = self.output_key(resp.output)
                score = float(self.scorer(probe, resp.output))
                if not math.isnan(score) and not 0.0 <= score <= 1.0:
                    raise ValueError(f"scorer returned {score!r}, outside [0, 1]")
            except Exception as exc:  # an agent that crashes on a probe has regressed on it
                errors[probe.id] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                keys[probe.id] = None
                scores[probe.id] = 0.0 if probe.expected is not None else float("nan")
                continue
            keys[probe.id] = key
            scores[probe.id] = score
            confs.append(resp.confidence)
            conf_scores.append(score)
            retrieved.append(list(resp.retrieved or ()))
        return dict(keys=keys, scores=scores, confs=confs, conf_scores=conf_scores,
                    retrieved=retrieved, errors=errors)

    def _metrics(self, run: Mapping[str, Any], memory: MemorySnapshot | None, mem_error: str | None) -> dict[str, float]:
        nan = float("nan")
        scores: dict[str, float] = run["scores"]
        m: dict[str, float] = {}
        m["accuracy"] = mean_score(list(scores.values()))
        m["public_accuracy"] = mean_score([s for k, s in scores.items() if not self.suite.is_holdout(k)])
        m["holdout_accuracy"] = mean_score([s for k, s in scores.items() if self.suite.is_holdout(k)])
        m["error_rate"] = len(run["errors"]) / len(self.suite)

        base = self.baseline
        if base is None:
            m.update(output_jsd=0.0, change_rate=0.0, regressions=0.0, fixes=0.0, probe_overfit_gap=0.0)
        else:
            d = drift(base.output_keys, run["keys"], base.scores, scores)
            m.update(output_jsd=d.output_jsd, change_rate=d.change_rate,
                     regressions=float(d.regressions), fixes=float(d.fixes))
            m["probe_overfit_gap"] = self._overfit_lower_bound(base.scores, scores)

        if memory is not None:
            dv = diversity(memory.embeddings)
            m.update(memory_size=float(len(memory)), memory_dropped=float(dv.dropped),
                     mean_cosine_distance=dv.mean_cosine_distance, effective_rank=dv.effective_rank,
                     effective_rank_ratio=dv.effective_rank_ratio,
                     near_duplicate_fraction=dv.near_duplicate_fraction)
            rs = retrieval_stats(run["retrieved"], memory.ids)
        else:
            m.update(memory_size=nan, memory_dropped=nan, mean_cosine_distance=nan, effective_rank=nan,
                     effective_rank_ratio=nan, near_duplicate_fraction=nan)
            rs = retrieval_stats(run["retrieved"], None)
        m.update(retrieval_entropy=rs.normalized_entropy, retrieval_top1_share=rs.top1_share,
                 retrieval_coverage=rs.coverage, retrieval_gini=rs.gini, phantom_rate=rs.phantom_rate)

        cs = confidence_stats(run["confs"], run["conf_scores"])
        m.update(mean_confidence=cs.mean_confidence, overconfidence=cs.overconfidence, ece=cs.ece,
                 invalid_confidence=float(cs.invalid))
        m["memory_error"] = 1.0 if mem_error else 0.0
        return m

    def _overfit_lower_bound(self, base: Mapping[str, float], cur: Mapping[str, float]) -> float:
        """Lower 95% bound on (public gain − held-out gain), from paired per-probe changes.

        A raw gap is mostly noise when the held-out half is a few dozen probes; the bound
        only goes positive when the public half genuinely pulled away.
        """
        pub, hold = [], []
        for k, c in cur.items():
            b = base.get(k, math.nan)
            if math.isnan(b) or math.isnan(c):
                continue
            (hold if self.suite.is_holdout(k) else pub).append(c - b)
        if len(pub) < 2 or len(hold) < 2:
            return float("nan")
        p, h = np.array(pub), np.array(hold)
        gap = p.mean() - h.mean()
        se = math.sqrt(p.var(ddof=1) / p.size + h.var(ddof=1) / h.size)
        return float(gap - 1.96 * se)

    def evaluate(self, agent: Agent, cycle: int, record: bool = True) -> CycleReport:
        """Run the suite against ``agent`` as it stands after ``cycle``.

        With ``record=False`` the report is computed against the existing baseline but not
        appended to history; bisection uses this to re-test old checkpoints.
        """
        self.suite.verify(self.fingerprint)
        if record and self.history and cycle <= self.history[-1].cycle:
            raise ValueError(
                f"cycle {cycle} is not after the last recorded cycle {self.history[-1].cycle}"
            )
        if not record and not self.history:
            raise ValueError("record a baseline (cycle 0) before evaluating unrecorded cycles")

        run = self._run_probes(agent)
        mem_error = None
        memory: MemorySnapshot | None
        try:
            memory = agent.memory_snapshot()
        except Exception as exc:
            memory, mem_error = None, f"{type(exc).__name__}: {exc}"
        metrics = self._metrics(run, memory, mem_error)

        # The baseline window: the first `warmup` recorded cycles (or this one, at cycle 0).
        window = [r.metrics for r in self.history[: self.warmup]]
        in_warmup = record and len(self.history) < self.warmup
        if not window or in_warmup:
            window = window + [metrics]
        recent = [r.alarm_condition for r in self.history[-(self.detector.patience - 1):]] if self.detector.patience > 1 else []
        if in_warmup:
            reason = (
                "baseline recorded"
                if self.warmup == 1
                else f"warm-up cycle {len(self.history) + 1}/{self.warmup} (part of the baseline)"
            )
            assessment = Assessment(status="OK", reason=reason)
            assessment.undefined = [k for k, v in metrics.items() if isinstance(v, float) and math.isnan(v)]
        else:
            assessment = self.detector.assess(metrics, window, recent if record else [])
        errors = dict(run["errors"])
        if mem_error:
            errors["<memory_snapshot>"] = mem_error
        report = CycleReport(
            cycle=cycle,
            status=assessment.status,
            metrics=metrics,
            signals=assessment.signals,
            categories=assessment.categories,
            collapse_signature=assessment.collapse_signature,
            reason=assessment.reason,
            output_keys=run["keys"],
            scores=run["scores"],
            errors=errors,
            undefined=assessment.undefined,
            alarm_condition=CollapseDetector.alarm_condition(assessment, self.detector.min_categories),
        )
        if record:
            self.history.append(report)
        return report

    # ------------------------------------------------------------------ queries
    def first_alarm(self) -> CycleReport | None:
        return next((r for r in self.history if r.alarmed), None)

    def series(self, metric: str) -> tuple[list[int], list[float]]:
        return [r.cycle for r in self.history], [r.metrics.get(metric, float("nan")) for r in self.history]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "suite_fingerprint": self.fingerprint,
            "warmup": self.warmup,
            "history": [r.to_dict() for r in self.history],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path, suite: ProbeSuite, **kwargs: Any) -> "RegressionMonitor":
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported history format {data.get('format')!r}")
        if data["suite_fingerprint"] != suite.fingerprint:
            raise ProbeSuiteMismatch(
                "history was recorded against a different probe suite "
                f"({data['suite_fingerprint'][:12]}… vs {suite.fingerprint[:12]}…)"
            )
        try:
            suite.verify()
        except ProbeSuiteTampered as exc:  # the suite object was mutated after construction
            raise ProbeSuiteMismatch(str(exc)) from exc
        kwargs.setdefault("warmup", data.get("warmup", 1))
        mon = cls(suite, **kwargs)
        mon.history = [CycleReport.from_dict(r) for r in data["history"]]
        return mon
