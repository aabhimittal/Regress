"""Command line: ``regress demo`` and ``regress show``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .bisect import bisect_checkpoints, locate_onset
from .monitor import CycleReport, RegressionMonitor
from .report import Style, render_summary, render_table, render_trends
from .sim import SCENARIOS, Simulation

__all__ = ["main", "run_demo"]


def run_demo(scenario: str, cycles: int, seed: int, style: Style, save: str | None = None) -> RegressionMonitor:
    sc = SCENARIOS[scenario]
    print(style.bold(f"REGRESS demo · scenario '{sc.name}'") + f" — {sc.description}")
    sim = Simulation(sc, cycles=cycles, seed=seed)
    suite = sim.suite
    print(style.dim(
        f"probe suite: {len(suite)} probes ({len(suite.holdout)} held out), "
        f"fingerprint {suite.fingerprint[:12]} · toy agent: kNN over episodic memory, "
        f"{len(sim.checkpoint(0).mem.ids)} memories at cycle 0"
    ))
    print()
    mon = RegressionMonitor(suite)
    for c in range(cycles + 1):
        mon.evaluate(sim.checkpoint(c), c)
    print(render_table(mon.history, style))
    print()
    print(render_trends(mon.history, style))
    print()

    bis = None
    onset = None
    if mon.first_alarm() is not None:
        # Replay saved checkpoints, as you would with real agent snapshots: each evaluation
        # re-runs the frozen suite against that cycle's agent without touching history.
        bis = bisect_checkpoints(
            list(range(cycles + 1)),
            lambda c: mon.evaluate(sim.checkpoint(c), c, record=False).alarmed,
        )
        onset = locate_onset(mon.history)
    print(render_summary(mon.history, bis, onset, style))
    if sc.onset is not None:
        print(style.dim(f"(ground truth: this scenario's consolidation went bad at cycle {sc.onset})"))
    if save:
        mon.save(save)
        print(style.dim(f"history saved to {save}"))
    return mon


def show(path: str, style: Style) -> None:
    data = json.loads(Path(path).read_text())
    history = [CycleReport.from_dict(r) for r in data["history"]]
    print(style.dim(f"suite fingerprint {data['suite_fingerprint'][:12]} · {len(history)} cycles"))
    print(render_table(history, style))
    print()
    print(render_trends(history, style))
    print()
    print(render_summary(history, None, locate_onset(history) if history else None, style))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regress", description="Regression testing for self-evolving agents.")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run a simulated self-evolving agent through REGRESS")
    d.add_argument("--scenario", default="collapse", choices=[*SCENARIOS, "all"])
    d.add_argument("--cycles", type=int, default=20)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--save", help="write the monitor history to this JSON file")
    s = sub.add_parser("show", help="render a saved history")
    s.add_argument("path")
    args = p.parse_args(argv)
    style = Style(False if args.no_color else None)

    if args.cmd == "demo":
        if args.cycles < 1:
            p.error("--cycles must be >= 1")
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for i, name in enumerate(names):
            if i:
                print("\n" + "─" * 100 + "\n")
            save = args.save
            if save and len(names) > 1:
                stem = Path(save)
                save = str(stem.with_name(f"{stem.stem}-{name}{stem.suffix or '.json'}"))
            run_demo(name, args.cycles, args.seed, style, save)
        return 0
    if args.cmd == "show":
        show(args.path, style)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
