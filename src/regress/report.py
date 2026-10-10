"""Plain-text rendering of a monitor history: a per-cycle table, sparklines, and the verdict."""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence

from .bisect import BisectResult, Onset
from .monitor import CycleReport

__all__ = ["sparkline", "render_table", "render_trends", "render_alarm", "render_summary", "Style"]

_BLOCKS = "▁▂▃▄▅▆▇█"
_MARK = {"OK": "·", "WATCH": "~", "ALARM": "▲"}


def sparkline(values: Sequence[float], lo: float | None = None, hi: float | None = None) -> str:
    """Unicode sparkline. NaN renders as a space; a constant series renders flat."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return " " * len(values)
    lo = min(finite) if lo is None else lo
    hi = max(finite) if hi is None else hi
    span = hi - lo
    out = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(" ")
        elif span <= 1e-12:
            out.append(_BLOCKS[3])
        else:
            i = int(round((min(max(v, lo), hi) - lo) / span * (len(_BLOCKS) - 1)))
            out.append(_BLOCKS[i])
    return "".join(out)


class Style:
    def __init__(self, color: bool | None = None) -> None:
        if color is None:
            color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.color = color

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s: str) -> str:
        return self._w("1", s)

    def dim(self, s: str) -> str:
        return self._w("2", s)

    def status(self, st: str, width: int = 5) -> str:
        code = {"OK": "32", "WATCH": "33", "ALARM": "1;31"}.get(st, "0")
        return self._w(code, st.ljust(width))


def _f(v: float, fmt: str = ".3f") -> str:
    return "  –  " if v is None or not math.isfinite(v) else format(v, fmt)


TABLE_COLUMNS = (
    ("acc", "accuracy", ".3f"),
    ("held", "holdout_accuracy", ".3f"),
    ("eRank", "effective_rank", "5.1f"),
    ("spread", "mean_cosine_distance", ".3f"),
    ("r.ent", "retrieval_entropy", ".3f"),
    ("top1", "retrieval_top1_share", ".3f"),
    ("conf", "mean_confidence", ".3f"),
)


def render_table(history: Sequence[CycleReport], style: Style | None = None) -> str:
    st = style or Style(False)
    head = "cycle  status " + " ".join(f"{h:>6}" for h, _, _ in TABLE_COLUMNS) + "  signals"
    lines = [st.bold(head)]
    for r in history:
        cells = " ".join(f"{_f(r.metrics.get(m, math.nan), fmt):>6}" for _, m, fmt in TABLE_COLUMNS)
        sig = ", ".join(s.name for s in r.signals)
        lines.append(f"{r.cycle:>5}  {st.status(r.status)}  {cells}  {st.dim(sig)}")
    return "\n".join(lines)


TREND_METRICS = (
    ("accuracy", "accuracy"),
    ("held-out accuracy", "holdout_accuracy"),
    ("effective rank", "effective_rank"),
    ("pairwise spread", "mean_cosine_distance"),
    ("retrieval entropy", "retrieval_entropy"),
    ("mean confidence", "mean_confidence"),
)


def render_trends(history: Sequence[CycleReport], style: Style | None = None) -> str:
    st = style or Style(False)
    if not history:
        return ""
    lines = [st.bold(f"trends, cycle {history[0].cycle} → {history[-1].cycle}")]
    for label, m in TREND_METRICS:
        vals = [r.metrics.get(m, math.nan) for r in history]
        lines.append(f"  {label:<18} {sparkline(vals)}  {_f(vals[0])} → {_f(vals[-1])}")
    marks = "".join(_MARK.get(r.status, "?") for r in history)
    lines.append(f"  {'status':<18} {marks}  {st.dim('· ok  ~ watch  ▲ alarm')}")
    return "\n".join(lines)


def render_alarm(report: CycleReport, style: Style | None = None) -> str:
    st = style or Style(False)
    lines = [f"{st.status('ALARM', 0)} first raised at cycle {report.cycle}: {report.reason}"]
    for s in report.signals:
        if s.weak:
            continue
        rel = s.metric in ("effective_rank", "mean_cosine_distance")
        excess = f"{s.excess:+.1%}" if rel else f"{s.excess:+.3f}"
        tol = f"{s.tolerance:.0%}" if rel else f"{s.tolerance:.3f}"
        lines.append(
            f"  • {s.name:<16} {s.metric} {s.baseline:.3f} → {s.value:.3f}  "
            f"(worse by {excess.lstrip('+')}, tolerance {tol})"
        )
    return "\n".join(lines)


def render_summary(
    history: Sequence[CycleReport],
    bisect: BisectResult | None = None,
    onset: tuple[int | None, list[Onset]] | None = None,
    style: Style | None = None,
) -> str:
    st = style or Style(False)
    out: list[str] = []
    first = next((r for r in history if r.status == "ALARM"), None)
    if first is None:
        watches = [r.cycle for r in history if r.status == "WATCH"]
        out.append(
            st.status("OK", 0)
            + f" no alarm across {len(history)} cycles"
            + (f" (watch at cycles {watches[:8]}{'…' if len(watches) > 8 else ''})" if watches else "")
        )
        return "\n".join(out)
    out.append(render_alarm(first, st))
    if bisect is not None:
        tag = "exact" if bisect.exact else ("range" if bisect.candidates else "inconclusive")
        out.append(
            f"bisect   first bad checkpoint {bisect.first_bad}, last good {bisect.last_good} "
            f"— {len(bisect.evaluations)} evaluations [{tag}]"
        )
        for n in bisect.notes:
            out.append(f"         note: {n}")
    if onset is not None:
        cycle, per_metric = onset
        detail = ", ".join(f"{o.metric}@{o.cycle}" for o in per_metric if o.cycle is not None)
        if cycle is None:
            out.append("onset    no clear change point in the signalled metrics")
        else:
            lag = first.cycle - cycle
            out.append(f"onset    decline most likely began at cycle {cycle} ({detail})")
            if lag > 0:
                out.append(
                    f"         the alarm trailed the onset by {lag} cycle{'s' if lag != 1 else ''}: "
                    "look for the cause there, not at the alarm"
                )
    return "\n".join(out)
