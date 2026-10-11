"""Test consolidation *before* committing it: regression testing in the strict sense.

Monitoring after the fact cannot warn about damage that happens in one step. If a single
consolidation pass destroys half the facts, the first check that can see it is the one
after the damage. The only fix is to move the check in front of the commit:

1. apply a learning stage to a **copy** of the agent;
2. run the frozen probe suite on the copy;
3. keep the copy only if it passes; otherwise keep the previous agent and record the rejection.

Splitting a cycle into stages (observe → self-train → compress in chunks) means a bad stage
costs only that stage. Because every candidate is compared with the *baseline*, not the
previous cycle, many small accepted steps cannot add up past the tolerance either: the guard
bounds cumulative damage on the probe suite, not just per-step damage.

Two things the guard does *not* do:

* It does not make consolidation good. A guard that rejects every cycle is telling you the
  consolidation policy is broken (``rejection_rate`` is the alarm for that).
* It optimises against the probe suite, which over many decisions can overfit it (a form of
  adaptive data analysis). Keep an **audit** suite that the guard never sees and watch it;
  the experiments in ``regress.llm`` do exactly that.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .detector import ALARM
from .monitor import CycleReport, RegressionMonitor

__all__ = ["StageDecision", "ConsolidationGuard"]

Stage = tuple[str, Callable[[Any], Any]]


_SEVERITY = {"OK": 0, "WATCH": 1, "ALARM": 2}


def _no_worse(candidate: CycleReport, current: CycleReport) -> bool:
    a, b = candidate.metrics.get("accuracy", float("nan")), current.metrics.get("accuracy", float("nan"))
    return (_SEVERITY.get(candidate.status, 3) <= _SEVERITY.get(current.status, 3)
            and not (a < b - 1e-9))


@dataclass
class StageDecision:
    cycle: int
    stage: str
    accepted: bool
    status: str
    reason: str
    accuracy: float
    probe_calls: float


@dataclass
class ConsolidationGuard:
    """Gate each learning stage on a REGRESS evaluation of the candidate agent.

    ``block_on``: statuses that reject a candidate (default: ALARM). Add ``"WATCH"`` for a
    stricter gate that also rejects any signal. ``clone``: how to copy an agent; defaults to
    ``agent.clone()`` when it exists, else ``copy.deepcopy``.

    A frozen suite only protects what the agent knew at cycle 0, which makes a guard lean
    towards *not learning*: new observations crowd old ones and get rejected, while a
    compression that destroys recent knowledge passes because no probe asks about it. Two
    options address that:

    ``commit_always``: stage names (or name prefixes) that are committed without gating, for
    stages that bring in data rather than rewrite it (e.g. ``("observe",)``). They are still
    evaluated and logged, with status ``FORCED``.

    Once a forced stage has pushed the committed agent past ``block_on``, later stages are
    judged against the committed agent (no worse status, no lower accuracy) rather than the
    baseline; otherwise nothing could be consolidated again. The recorded cycle report
    still shows the regression the forced data caused.

    ``retention``: a function scoring an agent on knowledge acquired *since* the baseline
    (e.g. accuracy on questions about recent observations). A candidate is also rejected if
    its retention falls more than ``retention_tolerance`` below the current agent's.
    """

    monitor: RegressionMonitor
    block_on: Sequence[str] = (ALARM,)
    clone: Callable[[Any], Any] | None = None
    commit_always: Sequence[str] = ()
    retention: Callable[[Any], float] | None = None
    retention_tolerance: float = 0.05
    decisions: list[StageDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.block_on:
            raise ValueError("block_on must name at least one status")
        if not 0 <= self.retention_tolerance < 1:
            raise ValueError("retention_tolerance must be in [0, 1)")

    def _copy(self, agent: Any) -> Any:
        if self.clone is not None:
            return self.clone(agent)
        fn = getattr(agent, "clone", None)
        return fn() if callable(fn) else copy.deepcopy(agent)

    def run_cycle(self, agent: Any, cycle: int, stages: Iterable[Stage]) -> tuple[Any, CycleReport]:
        """Apply ``stages`` one by one, keeping those whose candidate passes. Returns the
        committed agent and the cycle's recorded report (of the committed agent)."""
        if not self.monitor.history:
            raise ValueError("record a baseline (cycle 0) on the monitor before guarding cycles")
        current = agent
        current_rep: CycleReport | None = None
        for name, apply in stages:
            candidate = self._copy(current)
            forced = any(name == p or name.startswith(p) for p in self.commit_always)
            try:
                apply(candidate)
                rep = self.monitor.evaluate(candidate, cycle, record=False)
                accepted = rep.status not in self.block_on
                status, reason = rep.status, rep.reason
                acc, calls = rep.metrics.get("accuracy", float("nan")), rep.metrics.get("probe_calls", 0.0)
                if not accepted and not forced and self.commit_always:
                    # A forced stage may already have pushed the committed agent over the line;
                    # then every candidate fails against the baseline and nothing could ever be
                    # consolidated again. Judge it against the committed agent instead.
                    if current_rep is None:
                        current_rep = self.monitor.evaluate(current, cycle, record=False)
                    if current_rep.status in self.block_on and _no_worse(rep, current_rep):
                        accepted = True
                        reason = f"no worse than the committed agent ({current_rep.status}); {reason}"
                if accepted and self.retention is not None and not forced:
                    before, after = self.retention(current), self.retention(candidate)
                    if after < before - self.retention_tolerance:
                        accepted, status = False, "RETENTION"
                        reason = f"recent-knowledge retention {before:.2f} -> {after:.2f}"
                if forced:
                    accepted, status = True, "FORCED"
            except Exception as exc:  # a stage that crashes is rejected, never committed (even if forced)
                accepted, status, reason = False, "ERROR", f"{type(exc).__name__}: {exc}"
                acc, calls = float("nan"), 0.0
            self.decisions.append(StageDecision(cycle, name, accepted, status, reason, acc, calls))
            if accepted:
                current, current_rep = candidate, None
        report = self.monitor.evaluate(current, cycle)  # with a cache, this re-test is free
        return current, report

    # -- summaries -------------------------------------------------------------------
    @property
    def rejection_rate(self) -> float:
        return sum(not d.accepted for d in self.decisions) / len(self.decisions) if self.decisions else 0.0

    def rejected(self) -> list[StageDecision]:
        return [d for d in self.decisions if not d.accepted]

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, list[bool]] = {}
        for d in self.decisions:
            by_stage.setdefault(d.stage.split("_")[0] if d.stage.startswith("compress") else d.stage, []).append(d.accepted)
        return {
            "decisions": len(self.decisions),
            "rejected": sum(not d.accepted for d in self.decisions),
            "rejection_rate": self.rejection_rate,
            "accept_rate_by_stage": {k: sum(v) / len(v) for k, v in by_stage.items()},
        }
