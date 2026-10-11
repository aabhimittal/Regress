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
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..ablation import confirm_blame
from ..adapters import text_key
from ..bisect import bisect_checkpoints, locate_onset
from ..cache import ResponseCache
from ..detector import CollapseDetector
from ..guard import ConsolidationGuard
from ..monitor import RegressionMonitor
from .agent import NoteAgent
from .backends import Backend
from .world import answer_score, make_world

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
    task: str = "facts"
    guarded: bool = False
    guard: dict | None = None  # rejection statistics when guarded
    audit_accuracy: list[float] | None = None  # accuracy on probes the guard never saw


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
    task: str = "facts",
    guard: bool | str = False,
    audit_fraction: float = 0.0,
    label: str | None = None,
) -> tuple[RegimeResult, RegressionMonitor, list[NoteAgent]]:
    """Run one regime. ``guard=True`` gates every learning stage on a REGRESS check of the
    candidate (see :mod:`regress.guard`), rejecting ALARM; ``guard="strict"`` also rejects WATCH. ``audit_fraction`` holds back that share of the
    probe questions in a separate audit suite that only reports, never decides."""
    if not 0 <= audit_fraction < 1:
        raise ValueError("audit_fraction must be in [0, 1)")
    t0 = time.time()
    calls0 = getattr(backend, "calls", 0)
    world = make_world(task, people, seed)
    questions = list(world.initial_questions())
    audit_qs: list = []
    if audit_fraction > 0:
        rng = random.Random(seed + 4242)
        idx = sorted(rng.sample(range(len(questions)), max(1, int(round(len(questions) * audit_fraction)))))
        audit_qs = [questions[i] for i in idx]
        questions = [q for i, q in enumerate(questions) if i not in set(idx)]
    suite = world.probe_suite(questions)
    agent = NoteAgent(backend, regime=regime, seed=seed, **(agent_kwargs or {}))
    agent.add_notes(world.initial_notes(), cycle=0)
    cache = ResponseCache()
    monitor = RegressionMonitor(suite, scorer=answer_score, output_key=text_key(), detector=detector,
                                ruler="reference", cache=cache)
    audit = None
    if audit_qs:
        audit_suite = world.probe_suite(audit_qs)
        audit = RegressionMonitor(audit_suite, scorer=answer_score, output_key=text_key(), ruler="reference",
                                  cache=cache, bootstrap=0)
        audit.evaluate(agent, 0)
    if guard not in (False, True, "strict"):
        raise ValueError("guard must be False, True or 'strict'")
    gate = (ConsolidationGuard(monitor, block_on=("ALARM", "WATCH") if guard == "strict" else ("ALARM",))
            if guard else None)
    checkpoints = [agent.clone()]
    monitor.evaluate(agent, 0)
    logs = []
    for c in range(1, cycles + 1):
        new = world.grow(new_people_per_cycle)
        if gate is not None:
            stages = agent.learn_stages(c, observations=new, questions=world.all_questions())
            log: dict = {"cycle": c}
            agent, _ = gate.run_cycle(agent, c, stages)
            kept = [d.stage for d in gate.decisions if d.cycle == c and d.accepted]
            log["kept_stages"] = kept
            log["rejected_stages"] = [d.stage for d in gate.decisions if d.cycle == c and not d.accepted]
            logs.append(log)
        else:
            logs.append(agent.learn(c, observations=new, questions=world.all_questions()))
            monitor.evaluate(agent, c)
        checkpoints.append(agent.clone())
        if audit is not None:
            audit.evaluate(agent, c)

    hist = monitor.history
    first = monitor.first_alarm()
    res = RegimeResult(
        regime=label or regime,
        first_alarm=first.cycle if first else None,
        first_signal=next((h.cycle for h in hist[1:] if h.signals), None),
        statuses=[h.status for h in hist],
        accuracy=[h.metrics["accuracy"] for h in hist],
        effective_rank=[h.metrics["effective_rank"] for h in hist],
        retrieval_entropy=[h.metrics["retrieval_entropy"] for h in hist],
        mean_confidence=[h.metrics["mean_confidence"] for h in hist],
        memory_size=[h.metrics["memory_size"] for h in hist],
        consolidation=logs,
        task=task,
        guarded=bool(guard),
        guard=gate.summary() if gate is not None else None,
        audit_accuracy=[h.metrics["accuracy"] for h in audit.history] if audit is not None else None,
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
    res.probe_calls = int(cost["probe_calls"]) + (int(audit.cost()["probe_calls"]) if audit else 0) + int(
        sum(d.probe_calls for d in gate.decisions) if gate else 0)
    res.cache_hits = int(cache.stats.hits)
    res.model_calls = getattr(backend, "calls", 0) - calls0
    res.seconds = round(time.time() - t0, 1)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = label or regime
        monitor.save(out / f"history-{name}.json")
        from ..site import render, report_payload

        title = f"Real agent ({backend.name}), {task} task, regime: {name}"
        (out / f"report-{name}.html").write_text(render(report_payload(hist, title=title)), encoding="utf-8")
    return res, monitor, checkpoints


def run_experiment(
    backend: Backend,
    regimes: Sequence[str] = ("faithful", "recursive_summary"),
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    results = {}
    specs = []
    for regime in regimes:
        specs.append({"label": regime, "regime": regime, "overrides": {}} if isinstance(regime, str)
                     else {"label": regime[0], "regime": regime[1], "overrides": regime[2]})
        if isinstance(regime, str):
            res, _, _ = run_regime(backend, regime, out_dir=out_dir, **kwargs)
        else:  # (label, regime, overrides) for variants such as guarded or gradual runs
            label, base, overrides = regime
            res, _, _ = run_regime(backend, base, out_dir=out_dir, label=label, **(kwargs | overrides))
        results[res.regime] = asdict(res)
    summary = {
        "backend": backend.name,
        "settings": {k: v for k, v in kwargs.items() if k != "detector"},
        "regime_specs": specs,
        "results": results,
        "separates": _separates(results),
    }
    if out_dir is not None:
        Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _separates(results: dict) -> bool | None:
    """True when the control stays quiet and every unguarded bad regime alarms."""
    control = results.get("faithful")
    bad = [r for name, r in results.items() if name != "faithful" and not r.get("guarded")
           and name in ("recursive_summary", "full", "gradual")]
    if control is None or not bad:
        return None
    return control.get("first_alarm") is None and all(r.get("first_alarm") is not None for r in bad)
