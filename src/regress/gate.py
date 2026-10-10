"""CI gate: turn a saved history into a pass/fail verdict and a Markdown summary.

Typical use in a training pipeline::

    regress gate history.json --fail-on alarm --markdown "$GITHUB_STEP_SUMMARY"

The gate looks at the *latest* recorded cycle by default (is the agent you are about to
ship degraded?). ``--any-cycle`` fails if any cycle ever alarmed, which suits a nightly job
that wants to hear about a transient collapse even after it recovered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bisect import locate_onset
from .blame import blame
from .detector import ALARM, OK, WATCH
from .health import suite_health
from .monitor import CycleReport

__all__ = ["GateResult", "gate", "load_history", "markdown_summary"]

_RANK = {OK: 0, WATCH: 1, ALARM: 2}


@dataclass
class GateResult:
    passed: bool
    status: str  # the status the gate judged
    cycle: int
    reason: str


def load_history(path: str | Path) -> tuple[list[CycleReport], dict]:
    """Read a saved history without needing the probe suite (for reporting and gating)."""
    data = json.loads(Path(path).read_text())
    if "history" not in data:
        raise ValueError(f"{path} is not a REGRESS history file")
    return [CycleReport.from_dict(r) for r in data["history"]], data


def gate(history: Sequence[CycleReport], fail_on: str = ALARM, any_cycle: bool = False) -> GateResult:
    if fail_on not in (ALARM, WATCH):
        raise ValueError("fail_on must be 'ALARM' or 'WATCH'")
    if not history:
        return GateResult(False, "EMPTY", -1, "history has no cycles")
    if any_cycle:
        worst = max(history, key=lambda h: (_RANK.get(h.status, 0), -h.cycle))
        target = worst
    else:
        target = history[-1]
    failed = _RANK.get(target.status, 0) >= _RANK[fail_on]
    where = "worst" if any_cycle else "latest"
    reason = f"{where} cycle {target.cycle} is {target.status}: {target.reason}"
    return GateResult(not failed, target.status, target.cycle, reason)


def markdown_summary(history: Sequence[CycleReport], result: GateResult | None = None) -> str:
    """A compact Markdown report suitable for a CI job summary or a PR comment."""
    if not history:
        return "### REGRESS\n\nNo cycles recorded.\n"
    icon = {OK: "🟢", WATCH: "🟡", ALARM: "🔴"}
    last, base = history[-1], history[0]
    lines = ["### REGRESS regression check", ""]
    if result is not None:
        lines += [f"**{'PASS' if result.passed else 'FAIL'}**: {result.reason}", ""]
    lines.append("Status by cycle: " + "".join(icon.get(h.status, "⚪") for h in history))
    lines.append("")
    lines += ["| metric | baseline | latest |", "|---|---:|---:|"]
    for key, label in [("accuracy", "accuracy"), ("holdout_accuracy", "held-out accuracy"),
                       ("effective_rank", "effective rank"), ("retrieval_entropy", "retrieval entropy"),
                       ("mean_confidence", "mean confidence"), ("error_rate", "probe error rate")]:
        b, v = base.metrics.get(key), last.metrics.get(key)
        fmt = lambda x: "–" if x is None or x != x else f"{x:.3f}"  # noqa: E731
        lines.append(f"| {label} | {fmt(b)} | {fmt(v)} |")
    first = next((h for h in history if h.status == ALARM), None)
    if first is not None:
        lines += ["", f"First alarm at cycle **{first.cycle}**: {first.reason}"]
        for s in first.signals:
            if not s.weak:
                lines.append(f"- `{s.name}`: {s.metric} {s.baseline:.3f} → {s.value:.3f}")
        onset, _ = locate_onset(history)
        if onset is not None and onset < first.cycle:
            lines.append(f"\nEstimated onset: cycle **{onset}** ({first.cycle - onset} cycles before the alarm).")
        b = blame(history, first.cycle, top=5)
        if b.suspects:
            lines += ["", f"Suspect memories ({len(b.regressed)} probes regressed):", ""]
            lines += [f"- `{s.memory_id}`: {s.regressed_hits} regressed probes, lift {s.lift:.2f}" for s in b.suspects]
    health = suite_health(history)
    lines += ["", "Probe suite: " + ", ".join(f"{k} {v}" for k, v in sorted(health.counts.items()))]
    lines += [f"- {r}" for r in health.recommendations]
    return "\n".join(lines) + "\n"
