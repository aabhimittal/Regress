"""Build the interactive REGRESS lab page: the hosted demo, or a report for your own history.

* ``build_demo(out_dir)`` runs the simulator (all scenarios, a few seeds, default and
  calibrated thresholds) and writes a self-contained ``index.html``. This is what the
  GitHub Pages / Vercel / Hugging Face deployments serve.
* ``render(report_payload(history))`` turns any saved monitor history into the same page
  (without the memory map, which needs embeddings a history does not store).

The page is one HTML file with the data inlined: no server, no network, nothing to host but
a static file.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bisect import bisect_checkpoints, locate_onset
from .blame import blame
from .calibrate import calibrate
from .detector import CollapseDetector, Thresholds
from .health import suite_health
from .monitor import CycleReport, RegressionMonitor
from .sim import SCENARIOS, Scenario, Simulation

__all__ = ["run_payload", "report_payload", "demo_payload", "simulate_payload", "render", "build_demo"]

METRIC_KEYS = (
    "accuracy", "public_accuracy", "holdout_accuracy", "effective_rank", "mean_cosine_distance",
    "near_duplicate_fraction", "retrieval_entropy", "retrieval_top1_share", "mean_confidence",
    "overconfidence", "change_rate", "output_jsd", "probe_overfit_gap", "memory_size",
    "regressions", "fixes", "error_rate",
)


def _r(v: Any, n: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, n) if math.isfinite(f) else None


def _cycle(h: CycleReport) -> dict[str, Any]:
    return {
        "cycle": h.cycle,
        "status": h.status,
        "reason": h.reason,
        "signature": h.collapse_signature,
        "m": {k: _r(h.metrics.get(k)) for k in METRIC_KEYS},
        "signals": [
            {"name": s.name, "cat": s.category, "metric": s.metric, "base": _r(s.baseline),
             "value": _r(s.value), "excess": _r(s.excess), "tol": _r(s.tolerance), "weak": bool(s.weak)}
            for s in h.signals
        ],
    }


def _blame_by_cycle(history: Sequence[CycleReport], top: int = 6) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, h in enumerate(history[1:], start=1):
        if not h.retrieved:
            continue
        b = blame(history, h.cycle, top=top)
        if not b.regressed:
            continue
        out[str(i)] = {
            "regressed": len(b.regressed),
            "suspects": [
                {"id": s.memory_id, "hits": s.regressed_hits, "total": s.total_hits, "lift": _r(s.lift, 2),
                 "sb": _r(s.share_base), "sn": _r(s.share_now), "fs": s.first_seen}
                for s in b.suspects
            ],
        }
    return out


def run_payload(history: Sequence[CycleReport], bisect: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Everything the page needs about one recorded history."""
    if not history:
        raise ValueError("empty history")
    first = next((h for h in history if h.alarmed), None)
    onset = None
    if first is not None:
        cyc, per = locate_onset(history)
        onset = {"cycle": cyc, "per": [{"metric": o.metric, "cycle": o.cycle, "model": o.model} for o in per]}
    health = suite_health(history)
    return {
        "cycles": [_cycle(h) for h in history],
        "first_alarm": first.cycle if first else None,
        "bisect": dict(bisect) if bisect else None,
        "onset": onset,
        "blame": _blame_by_cycle(history),
        "health": {"counts": health.counts, "recommendations": health.recommendations},
    }


def _memory_map(sim: Simulation) -> list:
    """Project every cycle's memory onto the top-2 principal axes of the cycle-0 memory."""
    m0 = sim.checkpoint(0).mem.emb
    mu = m0.mean(0)
    _, _, vt = np.linalg.svd(m0 - mu, full_matrices=False)
    proj = vt[:2].T
    maps = []
    for c in range(sim.cycles + 1):
        a = sim.checkpoint(c)
        xy = (a.mem.emb - mu) @ proj
        maps.append([[round(float(x), 3), round(float(y), 3), int(lab), round(float(np.log(st)), 2)]
                     for (x, y), lab, st in zip(xy, a.mem.labels, a.mem.strength)])
    return maps


def _simulate(sim: Simulation, detector: CollapseDetector | None) -> dict[str, Any]:
    mon = RegressionMonitor(sim.suite, detector=detector)
    for c in range(sim.cycles + 1):
        mon.evaluate(sim.checkpoint(c), c)
    bis = None
    if mon.first_alarm() is not None:
        b = bisect_checkpoints(
            list(range(sim.cycles + 1)),
            lambda c: mon.evaluate(sim.checkpoint(c), c, record=False).alarmed,
        )
        bis = {"first_bad": b.first_bad, "last_good": b.last_good, "evals": sorted(b.evaluations), "exact": b.exact}
    return run_payload(mon.history, bis)


def _calibrated_detector(cycles: int, seeds: Iterable[int]) -> tuple[CollapseDetector, dict[str, Any]]:
    healthy = []
    for s in seeds:
        sim = Simulation(SCENARIOS["healthy"], cycles=cycles, seed=s)
        mon = RegressionMonitor(sim.suite)
        for c in range(cycles + 1):
            mon.evaluate(sim.checkpoint(c), c)
        healthy.append(mon.history)
    cal = calibrate(healthy)
    info = {"seeds": list(seeds), "quantile": cal.quantile, "margin": cal.margin,
            "thresholds": {k: _r(v) for k, v in cal.thresholds.to_dict().items()}}
    return CollapseDetector(cal.thresholds), info


def simulate_payload(
    scenario: Scenario,
    seed: int = 0,
    cycles: int = 20,
    calibrated: CollapseDetector | None = None,
    calibration_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A one-scenario, one-seed page (used by the Hugging Face Space for custom runs)."""
    sim = Simulation(scenario, cycles=cycles, seed=seed)
    modes = {"default": _simulate(sim, None)}
    if calibrated is not None:
        modes["calibrated"] = _simulate(sim, calibrated)
    thresholds = {"default": _thr(Thresholds())}
    if calibrated is not None:
        thresholds["calibrated"] = _thr(calibrated.thresholds)
    return {
        "meta": _demo_meta(),
        "seeds": [str(seed)],
        "modes": list(modes),
        "calibration": dict(calibration_info) if calibration_info else None,
        "thresholds": thresholds,
        "scenarios": {
            scenario.name: {
                "description": scenario.description, "truth": True, "onset": scenario.onset,
                "runs": {str(seed): {"map": _memory_map(sim), "modes": modes}},
            }
        },
    }


def demo_payload(
    seeds: Sequence[int] = (0, 1, 2),
    cycles: int = 20,
    calibration_seeds: Sequence[int] = (100, 101, 102, 103, 104),
) -> dict[str, Any]:
    """Every built-in scenario × ``seeds``, under default and calibrated thresholds.

    Calibration uses healthy runs on ``calibration_seeds``, disjoint from the seeds shown, so
    the calibrated results are out-of-sample.
    """
    det, info = _calibrated_detector(cycles, calibration_seeds)
    scenarios: dict[str, Any] = {}
    for name, sc in SCENARIOS.items():
        runs = {}
        for seed in seeds:
            sim = Simulation(sc, cycles=cycles, seed=seed)
            runs[str(seed)] = {
                "map": _memory_map(sim),
                "modes": {"default": _simulate(sim, None), "calibrated": _simulate(sim, det)},
            }
        scenarios[name] = {"description": sc.description, "truth": True, "onset": sc.onset, "runs": runs}
    return {"meta": _demo_meta(), "seeds": [str(s) for s in seeds], "modes": ["default", "calibrated"],
            "calibration": info, "thresholds": {"default": _thr(Thresholds()), "calibrated": info["thresholds"]},
            "scenarios": scenarios}


def report_payload(
    history: Sequence[CycleReport],
    title: str = "Your agent",
    description: str = "",
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """A page for a recorded history (no ground truth, no memory map)."""
    return {
        "thresholds": {"default": _thr(thresholds or Thresholds())},
        "meta": {
            "mode": "report",
            "title": "REGRESS Report",
            "eyebrow": "REGRESS · history report",
            "heading": title,
            "lede": description or (
                f"{len(history)} recorded cycles, each compared with cycle {history[0].cycle}. "
                "Scrub the cycles to see what moved, which signals fired, and which memories the "
                "regressed probes relied on."
            ),
            "footer": "Generated by REGRESS from a saved monitor history.",
        },
        "seeds": ["0"],
        "modes": ["default"],
        "calibration": None,
        "scenarios": {
            "history": {"description": description, "truth": False, "onset": None,
                        "runs": {"0": {"map": None, "modes": {"default": run_payload(history)}}}}
        },
    }


def _thr(t: Thresholds) -> dict[str, float | None]:
    return {k: _r(v) for k, v in t.to_dict().items()}


def _demo_meta() -> dict[str, str]:
    return {
        "mode": "demo",
        "title": "REGRESS Collapse Lab",
        "eyebrow": "REGRESS · regression testing for self-evolving agents",
        "heading": "Watch an agent's memory sour, cycle by cycle",
        "lede": (
            "A toy agent answers 160 frozen probe questions from an episodic memory of 240 items, then "
            "runs a learning cycle. REGRESS re-runs the same probes after every cycle and compares each "
            "metric against cycle 0. Pick a scenario, then scrub the cycles or press play. Every number "
            "on this page is the library's real output."
        ),
        "footer": (
            "Built from regress.sim by the REGRESS site builder. The simulator is deliberately built "
            "to collapse: it demonstrates the detector, not how real agents fail."
        ),
    }


def render(payload: Mapping[str, Any]) -> str:
    template = resources.files("regress").joinpath("assets/lab.html").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # Keep the inline script from being closed early by any string in the data.
    data = data.replace("</", "<\\/")
    title = str(payload.get("meta", {}).get("title") or "REGRESS Collapse Lab")
    return template.replace("/*__DATA__*/null", data).replace("__TITLE__", _escape(title))


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_demo(out_dir: str | Path, seeds: Sequence[int] = (0, 1, 2), cycles: int = 20) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(render(demo_payload(seeds=seeds, cycles=cycles)), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    return page
