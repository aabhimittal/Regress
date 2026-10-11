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


def run_demo(
    scenario: str, cycles: int, seed: int, style: Style, save: str | None = None, detector=None
) -> RegressionMonitor:
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
    mon = RegressionMonitor(suite, detector=detector)
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


def _load_thresholds(path: str | None):
    if not path:
        return None
    from .detector import CollapseDetector, Thresholds

    data = json.loads(Path(path).read_text())
    return CollapseDetector(Thresholds.from_dict(data.get("thresholds", data)))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regress", description="Regression testing for self-evolving agents.")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run a simulated self-evolving agent through REGRESS")
    d.add_argument("--scenario", default="collapse", choices=[*SCENARIOS, "all"])
    d.add_argument("--cycles", type=int, default=20)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--save", help="write the monitor history to this JSON file")
    d.add_argument("--thresholds", help="JSON file of thresholds (e.g. from `regress calibrate`)")
    s = sub.add_parser("show", help="render a saved history in the terminal")
    s.add_argument("path")
    st = sub.add_parser("site", help="build the interactive demo site (index.html) into a directory")
    st.add_argument("--out", default="site")
    st.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    st.add_argument("--cycles", type=int, default=20)
    st.add_argument("--results", help="directory of recorded real-agent runs to publish under real/")
    rp = sub.add_parser("report", help="turn a saved history into an interactive HTML report")
    rp.add_argument("path")
    rp.add_argument("--html", required=True, help="output HTML file")
    rp.add_argument("--title", default="Your agent")
    g = sub.add_parser("gate", help="exit non-zero if a saved history shows a regression (for CI)")
    g.add_argument("path")
    g.add_argument("--fail-on", choices=["alarm", "watch"], default="alarm")
    g.add_argument("--any-cycle", action="store_true", help="judge the worst cycle, not the latest")
    g.add_argument("--markdown", help="also write a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    c = sub.add_parser("calibrate", help="fit thresholds to histories of healthy cycles")
    c.add_argument("paths", nargs="+")
    c.add_argument("--out", required=True, help="write thresholds JSON here")
    c.add_argument("--quantile", type=float, default=0.999)
    c.add_argument("--margin", type=float, default=1.5)
    b = sub.add_parser("blame", help="rank memories implicated in a regression")
    b.add_argument("path")
    b.add_argument("--cycle", type=int, help="default: first alarm, else the last cycle")
    b.add_argument("--top", type=int, default=10)
    h = sub.add_parser("health", help="classify probes as saturated, unsolved, flaky, ...")
    h.add_argument("path")
    e = sub.add_parser("experiment", help="run REGRESS on a real language-model agent (faithful vs recursive summary)")
    e.add_argument("--backend", choices=["rules", "llamacpp", "anthropic", "replay"], default="rules",
                   help="rules: deterministic stand-in; llamacpp: local GGUF; anthropic: Claude API; "
                        "replay: re-run a recorded experiment (--replay-from DIR) without the model")
    e.add_argument("--replay-from", help="directory with summary.json + llm-cache.jsonl from a recorded run")
    e.add_argument("--model", help="GGUF path for llamacpp, model id for anthropic (default claude-opus-5-5)")
    e.add_argument("--out", required=True)
    e.add_argument("--cycles", type=int, default=8)
    e.add_argument("--people", type=int, default=20)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--compress-fraction", type=float, default=1.0,
                   help="share of old notes the recursive regime compresses per cycle (< 1 = gradual)")
    e.add_argument("--task", choices=["facts", "multihop"], default="facts")
    e.add_argument("--guard", choices=["none", "alarm", "strict", "learn"], default="none",
                   help="test each learning stage on a copy before committing it (strict also blocks WATCH; "
                        "learn = strict, but always commit observations and also protect recently learned facts)")
    e.add_argument("--audit-fraction", type=float, default=0.0,
                   help="share of questions held back in an audit suite that only reports, never decides")
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
            run_demo(name, args.cycles, args.seed, style, save, _load_thresholds(args.thresholds))
        return 0
    if args.cmd == "show":
        show(args.path, style)
        return 0
    if args.cmd == "site":
        from .site import build_demo

        if args.cycles < 4:
            p.error("--cycles must be >= 4")
        page = build_demo(args.out, seeds=args.seeds, cycles=args.cycles, results_dir=args.results)
        print(f"wrote {page} ({page.stat().st_size / 1e6:.1f} MB)")
        return 0
    if args.cmd == "report":
        from .detector import CollapseDetector
        from .gate import load_history
        from .site import render, report_payload

        history, data = load_history(args.path)
        if not history:
            p.error(f"{args.path} has no cycles")
        det = CollapseDetector.from_config(data["detector"]) if data.get("detector") else None
        html = render(report_payload(history, title=args.title, thresholds=det.thresholds if det else None))
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"wrote {args.html}")
        return 0
    if args.cmd == "gate":
        from .gate import gate, load_history, markdown_summary

        history, _ = load_history(args.path)
        res = gate(history, fail_on=args.fail_on.upper(), any_cycle=args.any_cycle)
        print(("PASS " if res.passed else "FAIL ") + res.reason)
        if args.markdown:
            with open(args.markdown, "a", encoding="utf-8") as fh:
                fh.write(markdown_summary(history, res))
        return 0 if res.passed else 1
    if args.cmd == "calibrate":
        from .calibrate import calibrate
        from .gate import load_history

        cal = calibrate([load_history(x)[0] for x in args.paths], quantile=args.quantile, margin=args.margin)
        Path(args.out).write_text(json.dumps({"thresholds": cal.thresholds.to_dict(), "quantile": cal.quantile,
                                              "margin": cal.margin, "sources": list(args.paths)}, indent=1))
        print(cal.summary())
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "blame":
        from .blame import blame
        from .gate import load_history

        history, _ = load_history(args.path)
        rep = blame(history, args.cycle, top=args.top)
        print(f"cycle {rep.cycle}: {len(rep.regressed)} probes regressed against cycle {history[0].cycle}")
        if rep.note:
            print(rep.note)
        if rep.suspects:
            print(f"{'memory':<16} {'regressed':>9} {'total':>6} {'lift':>6} {'share then→now':>17} {'first seen':>10}")
            for x in rep.suspects:
                print(f"{x.memory_id:<16} {x.regressed_hits:>9} {x.total_hits:>6} {x.lift:>6.2f} "
                      f"{x.share_base:>7.1%} → {x.share_now:<7.1%} {str(x.first_seen):>10}")
        return 0
    if args.cmd == "experiment":
        import shutil

        from .llm import AnthropicBackend, CachedBackend, LlamaCppBackend, ReplayBackend, RuleBackend, run_experiment

        if args.cycles < 1 or args.people < 2:
            p.error("--cycles must be >= 1 and --people >= 2")
        if args.backend == "llamacpp":
            if not args.model:
                p.error("--backend llamacpp needs --model PATH.gguf")
            backend = LlamaCppBackend(args.model)
        elif args.backend == "anthropic":
            backend = AnthropicBackend(model=args.model or "claude-opus-5-5")
        elif args.backend == "replay":
            if not args.replay_from:
                p.error("--backend replay needs --replay-from DIR")
            src = Path(args.replay_from)
            recorded = json.loads((src / "summary.json").read_text())
            backend = ReplayBackend(recorded["backend"])
            st = recorded.get("settings", {})
            args.cycles, args.people, args.seed = st.get("cycles", args.cycles), st.get("people", args.people), st.get("seed", args.seed)
            args.compress_fraction = (st.get("agent_kwargs") or {}).get("compress_fraction", args.compress_fraction)
        else:
            backend = RuleBackend()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.backend == "replay":
            if (out / "llm-cache.jsonl").resolve() != (src / "llm-cache.jsonl").resolve():
                shutil.copy(src / "llm-cache.jsonl", out / "llm-cache.jsonl")
            backend = CachedBackend(backend, out / "llm-cache.jsonl")
        elif args.backend != "rules":
            backend = CachedBackend(backend, out / "llm-cache.jsonl")
        if not 0 < args.compress_fraction <= 1:
            p.error("--compress-fraction must be in (0, 1]")
        if not 0 <= args.audit_fraction < 1:
            p.error("--audit-fraction must be in [0, 1)")
        agent_kwargs = {"compress_fraction": args.compress_fraction}
        if args.task == "multihop":
            agent_kwargs["hops"] = 2
        kwargs = dict(cycles=args.cycles, people=args.people, seed=args.seed, out_dir=out, agent_kwargs=agent_kwargs)
        if args.task != "facts":
            kwargs["task"] = args.task
        if args.audit_fraction:
            kwargs["audit_fraction"] = args.audit_fraction
        regimes = ["faithful", "recursive_summary"]
        if args.guard != "none":
            guard = "strict" if args.guard in ("strict", "learn") else True
            regimes.append(("recursive_summary+guard", "recursive_summary",
                            {"guard": guard, "guard_learning": args.guard == "learn",
                             "agent_kwargs": agent_kwargs | {"compress_chunks": 2}}))
        if args.backend == "replay" and recorded.get("regime_specs"):
            regimes = [s["regime"] if not s["overrides"] and s["label"] == s["regime"]
                       else (s["label"], s["regime"], s["overrides"]) for s in recorded["regime_specs"]]
            st = recorded.get("settings", {})
            for k in ("task", "audit_fraction", "ablate", "agent_kwargs"):
                if k in st:
                    kwargs[k] = st[k]
        if args.backend == "replay" and not any("fresh_accuracy" in r for r in recorded["results"].values()):
            kwargs["measure_fresh"] = False  # recorded before new-fact accuracy existed
        summary = run_experiment(backend, regimes=regimes, **kwargs)
        for regime, r in summary["results"].items():
            marks = "".join({"OK": "·", "WATCH": "~", "ALARM": "▲"}[x] for x in r["statuses"])
            acc = " ".join(f"{a:.2f}" for a in r["accuracy"])
            print(f"{regime:<24} {marks}  first alarm: {r['first_alarm']}  accuracy: {acc}")
            if r.get("guard"):
                g = r["guard"]
                audit = r.get("audit_accuracy")
                print(f"{'':<24} guard: rejected {g['rejected']}/{g['decisions']} stages"
                      + (f"; audit accuracy {audit[0]:.2f} → {audit[-1]:.2f}" if audit else ""))
            if r.get("fresh_accuracy") is not None:
                print(f"{'':<24} facts learned during the run, answered at the end: {r['fresh_accuracy']:.2f}")
            if r.get("ablation"):
                a = r["ablation"]
                print(f"{'':<24} ablation: {a['verdict']} (recovered {a['recovered']}/{a['regressed']}, "
                      f"controls {a['control_recovered']})")
        print(f"separates regimes: {summary['separates']} · outputs in {out}/")
        return 0
    if args.cmd == "health":
        from .gate import load_history
        from .health import suite_health

        rep = suite_health(load_history(args.path)[0])
        print(f"{rep.cycles} cycles · " + ", ".join(f"{k} {v}" for k, v in sorted(rep.counts.items())))
        for r in rep.recommendations:
            print(f"- {r}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
