"""Run REGRESS on a real language-model agent and record everything.

For each regime the experiment:

1. builds a fact world and a frozen probe suite over the facts the agent starts with;
2. runs ``cycles`` learning cycles (each adds new people, i.e. interfering facts), recording a
   REGRESS report after every cycle and keeping a checkpoint of the agent;
3. if anything alarmed: bisects the checkpoints, estimates the onset, ranks blame suspects,
   and confirms or refutes them by ablation;
4. writes ``history-<regime>.json``, ``report-<regime>.html`` and a ``summary.json``.

The verdict that matters is whether REGRESS separates the regimes: alarms on
``recursive_summary`` and stays quiet on ``faithful``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..ablation import confirm_blame
from ..adapters import text_key
from ..bisect import bisect_checkpoints, locate_onset
from ..cache import ResponseCache
from ..detector import CollapseDetector
from ..monitor import RegressionMonitor
from .agent import NoteAgent
from .backends import Backend
from .world import FactWorld, answer_score

__all__ = ["RegimeResult", "run_regime", "run_experiment"]


@dataclass
class RegimeResult:
    regime: str
    first_alarm: int | None
    first_signal: int | None  # first cycle with any (non-warm-up) signal
    statuses: list[str]
    accuracy: list[float]
    effective_rank: list[float]
    retrieval_entropy: list[float]
    mean_confidence: list[float]
    memory_size: list[float]
    bisect: dict | None = None
    onset: int | None = None
    ablation: dict | None = None
    model_calls: int = 0
    probe_calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    consolidation: list[dict] = field(default_factory=list)


def run_regime(
    backend: Backend,
    regime: str,
    cycles: int = 8,
    people: int = 20,
    new_people_per_cycle: int = 4,
    seed: int = 0,
    detector: CollapseDetector | None = None,
    out_dir: str | Path | None = None,
    ablate: bool = True,
    agent_kwargs: dict | None = None,
) -> tuple[RegimeResult, RegressionMonitor, list[NoteAgent]]:
    t0 = time.time()
    calls0 = getattr(backend, "calls", 0)
    world = FactWorld(n_people=people, seed=seed)
    initial = list(world.facts)
    suite = world.probe_suite(initial)
    agent = NoteAgent(backend, regime=regime, seed=seed, **(agent_kwargs or {}))
    agent.add_notes([f.text for f in initial], cycle=0)
    cache = ResponseCache()
    monitor = RegressionMonitor(suite, scorer=answer_score, output_key=text_key(), detector=detector,
                                ruler="reference", cache=cache)
    checkpoints = [agent.clone()]
    monitor.evaluate(agent, 0)
    logs = []
    for c in range(1, cycles + 1):
        new = world.add_people(new_people_per_cycle)
        logs.append(agent.learn(c, observations=new, questions=world.facts))
        checkpoints.append(agent.clone())
        monitor.evaluate(agent, c)

    hist = monitor.history
    first = monitor.first_alarm()
    res = RegimeResult(
        regime=regime,
        first_alarm=first.cycle if first else None,
        first_signal=next((h.cycle for h in hist[1:] if h.signals), None),
        statuses=[h.status for h in hist],
        accuracy=[h.metrics["accuracy"] for h in hist],
        effective_rank=[h.metrics["effective_rank"] for h in hist],
        retrieval_entropy=[h.metrics["retrieval_entropy"] for h in hist],
        mean_confidence=[h.metrics["mean_confidence"] for h in hist],
        memory_size=[h.metrics["memory_size"] for h in hist],
        consolidation=logs,
    )
    if first is not None:
        b = bisect_checkpoints(
            list(range(cycles + 1)),
            lambda c: monitor.evaluate(checkpoints[c], c, record=False).alarmed,
        )
        res.bisect = {"first_bad": b.first_bad, "last_good": b.last_good,
                      "evaluations": len(b.evaluations), "consistent": b.consistent}
        res.onset, _ = locate_onset(hist)
        if ablate:
            ab = confirm_blame(monitor, checkpoints[first.cycle], cycle=first.cycle, top=4, controls=3)
            res.ablation = {k: v for k, v in asdict(ab).items() if k != "regressed"} | {
                "regressed": len(ab.regressed)}
    cost = monitor.cost()
    res.probe_calls = int(cost["probe_calls"])
    res.cache_hits = int(cache.stats.hits)
    res.model_calls = getattr(backend, "calls", 0) - calls0
    res.seconds = round(time.time() - t0, 1)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        monitor.save(out / f"history-{regime}.json")
        from ..site import render, report_payload

        title = f"Real agent ({backend.name}), regime: {regime}"
        (out / f"report-{regime}.html").write_text(render(report_payload(hist, title=title)), encoding="utf-8")
    return res, monitor, checkpoints


def run_experiment(
    backend: Backend,
    regimes: Sequence[str] = ("faithful", "recursive_summary"),
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    results = {}
    for regime in regimes:
        res, _, _ = run_regime(backend, regime, out_dir=out_dir, **kwargs)
        results[regime] = asdict(res)
    summary = {
        "backend": backend.name,
        "settings": {k: v for k, v in kwargs.items() if k != "detector"},
        "results": results,
        "separates": (
            results.get("faithful", {}).get("first_alarm") is None
            and results.get("recursive_summary", {}).get("first_alarm") is not None
        ) if {"faithful", "recursive_summary"} <= set(results) else None,
    }
    if out_dir is not None:
        Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
