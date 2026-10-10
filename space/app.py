"""REGRESS on Hugging Face Spaces.

Three tabs:
  * Collapse Lab: the prebuilt interactive demo (all scenarios, seeds 0-2).
  * Run your own scenario: pick onset, collapse strength, seed and cycles; REGRESS runs the
    simulator live, then shows the same interactive page for that run.
  * Analyze your history: upload a history.json saved by RegressionMonitor.save() and get
    the CI verdict, suspect memories, probe-suite health and an interactive report.

Runs as a plain FastAPI app with Gradio mounted on it, so the interactive pages can be
served at their own URLs and embedded with an iframe.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):  # Space layout, then repo checkout
    if (candidate / "regress").is_dir():
        sys.path.insert(0, str(candidate))
        break

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from regress import __version__  # noqa: E402
from regress.calibrate import calibrate  # noqa: E402
from regress.detector import CollapseDetector  # noqa: E402
from regress.gate import gate, markdown_summary  # noqa: E402
from regress.monitor import CycleReport, RegressionMonitor  # noqa: E402
from regress.sim import SCENARIOS, Scenario, Simulation  # noqa: E402
from regress.site import demo_payload, render, report_payload, simulate_payload  # noqa: E402

PAGES: "OrderedDict[str, str]" = OrderedDict()  # key -> html, small LRU
MAX_PAGES = 48
MAX_UPLOAD = 20 * 1024 * 1024


def _store(html: str) -> str:
    key = hashlib.sha256(html.encode()).hexdigest()[:16]
    PAGES[key] = html
    PAGES.move_to_end(key)
    while len(PAGES) > MAX_PAGES:
        PAGES.popitem(last=False)
    return key


def _iframe(key: str, height: int = 1500) -> str:
    return (f'<iframe src="/page/{key}" title="REGRESS interactive page" '
            f'style="width:100%;height:{height}px;border:0;border-radius:10px" loading="lazy"></iframe>'
            f'<p style="font-size:13px">Open it full screen: <a href="/page/{key}" target="_blank">/page/{key}</a></p>')


_CAL: tuple[CollapseDetector, dict] | None = None


def calibrated_detector() -> tuple[CollapseDetector, dict]:
    """Calibrate once per process on healthy runs with seeds the UI never shows."""
    global _CAL
    if _CAL is None:
        seeds = [100, 101, 102, 103, 104]
        hist = []
        for s in seeds:
            sim = Simulation(SCENARIOS["healthy"], cycles=20, seed=s)
            mon = RegressionMonitor(sim.suite)
            for c in range(21):
                mon.evaluate(sim.checkpoint(c), c)
            hist.append(mon.history)
        cal = calibrate(hist)
        _CAL = (CollapseDetector(cal.thresholds),
                {"seeds": seeds, "quantile": cal.quantile, "margin": cal.margin,
                 "thresholds": {k: round(v, 4) for k, v in cal.thresholds.to_dict().items()}})
    return _CAL


# ----------------------------------------------------------------------------- handlers
def run_custom(preset: str, onset: int, rate: float, leak: bool, seed: int, cycles: int):
    cycles = int(cycles)
    onset = int(onset)
    if preset == "healthy (no collapse)":
        sc = Scenario("custom", "Feedback-driven learning throughout.", None)
    else:
        if onset > cycles:
            raise gr.Error(f"onset ({onset}) must be at most the number of cycles ({cycles})")
        desc = (f"Healthy until cycle {onset}, then self-reinforcing internalisation at strength {rate:.2f}"
                + (", with the public probes leaking into an answer cache." if leak else "."))
        sc = Scenario("custom", desc, onset, float(rate), bool(leak))
    det, info = calibrated_detector()
    payload = simulate_payload(sc, seed=int(seed), cycles=cycles, calibrated=det, calibration_info=info)
    key = _store(render(payload))
    run = payload["scenarios"]["custom"]["runs"][str(int(seed))]["modes"]
    lines = []
    for mode in ("default", "calibrated"):
        r = run[mode]
        fa = r["first_alarm"]
        lag = "" if fa is None or sc.onset is None else f" ({fa - sc.onset:+d} vs true onset)"
        lines.append(f"- **{mode} thresholds**: first alarm {'none' if fa is None else f'at cycle {fa}'}{lag}"
                     + (f"; onset estimate cycle {r['onset']['cycle']}" if r.get("onset") and r["onset"]["cycle"] is not None else ""))
    truth = "no collapse" if sc.onset is None else f"collapse starts at cycle {sc.onset}"
    md = f"**Ground truth:** {truth}.\n\n" + "\n".join(lines)
    return md, _iframe(key)


def analyze(file, fail_on: str):
    if file is None:
        raise gr.Error("Upload a history.json saved with RegressionMonitor.save().")
    path = Path(file if isinstance(file, str) else file.name)
    if path.stat().st_size > MAX_UPLOAD:
        raise gr.Error("That file is over 20 MB.")
    try:
        data = json.loads(path.read_text())
        history = [CycleReport.from_dict(r) for r in data["history"]]
    except Exception as exc:
        raise gr.Error(f"Could not read a REGRESS history from that file: {exc}") from exc
    if not history:
        raise gr.Error("The history has no cycles.")
    res = gate(history, fail_on=fail_on)
    det = CollapseDetector.from_config(data["detector"]) if data.get("detector") else None
    key = _store(render(report_payload(history, title=path.stem, thresholds=det.thresholds if det else None)))
    return markdown_summary(history, res), _iframe(key)


def sample_history() -> str:
    sim = Simulation(SCENARIOS["collapse"], cycles=16, seed=0)
    mon = RegressionMonitor(sim.suite)
    for c in range(17):
        mon.evaluate(sim.checkpoint(c), c)
    out = Path(tempfile.gettempdir()) / "regress-sample-history.json"
    mon.save(out)
    return str(out)


# ----------------------------------------------------------------------------- app
api = FastAPI(title="REGRESS")
DEMO_HTML: dict[str, str] = {}


@api.get("/lab", response_class=HTMLResponse)
def lab() -> str:
    if "html" not in DEMO_HTML:
        DEMO_HTML["html"] = render(demo_payload())
    return DEMO_HTML["html"]


@api.get("/page/{key}", response_class=HTMLResponse)
def page(key: str) -> str:
    html = PAGES.get(key)
    if html is None:
        raise HTTPException(404, "This page has expired. Run it again from the app.")
    return html


@api.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": __version__}


INTRO = f"""
# REGRESS · regression testing for self-evolving agents  <small>v{__version__}</small>

Agents that learn from their own experience assume each learning cycle makes them better. REGRESS checks that
assumption after every cycle. It re-runs a frozen probe suite, measures drift, memory diversity, retrieval
entropy and confidence against cycle 0, and raises an alarm when memory **homogenises while confidence rises**.

[GitHub](https://github.com/aabhimittal/Regress) · `pip install git+https://github.com/aabhimittal/Regress`
"""

with gr.Blocks(title="REGRESS") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Collapse Lab"):
        gr.Markdown("Four scenarios (healthy, collapse, slow, leak), three seeds each, default vs calibrated thresholds. "
                    "Scrub the cycles or press play.")
        gr.HTML('<iframe src="/lab" title="REGRESS Collapse Lab" style="width:100%;height:1700px;border:0;border-radius:10px"></iframe>'
                '<p style="font-size:13px"><a href="/lab" target="_blank">Open the lab full screen</a></p>')
    with gr.Tab("Run your own scenario"):
        with gr.Row():
            preset = gr.Radio(["collapse", "healthy (no collapse)"], value="collapse", label="Learning regime")
            leak = gr.Checkbox(False, label="Leak public probes into an answer cache")
        with gr.Row():
            onset = gr.Slider(1, 30, value=6, step=1, label="Collapse onset (cycle)")
            rate = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Collapse strength per cycle")
            seed = gr.Slider(0, 999, value=7, step=1, label="Seed")
            cycles = gr.Slider(6, 30, value=18, step=1, label="Cycles")
        go = gr.Button("Run REGRESS", variant="primary")
        out_md = gr.Markdown()
        out_html = gr.HTML()
        go.click(run_custom, [preset, onset, rate, leak, seed, cycles], [out_md, out_html])
    with gr.Tab("Analyze your history"):
        gr.Markdown("Upload the JSON written by `RegressionMonitor.save()` (or `regress demo --save`). "
                    "You get the CI verdict, suspect memories, probe-suite health and an interactive report. "
                    "Files are processed in memory and not stored.")
        with gr.Row():
            up = gr.File(label="history.json", file_types=[".json"])
            fail_on = gr.Radio(["ALARM", "WATCH"], value="ALARM", label="Fail the gate on")
        with gr.Row():
            btn = gr.Button("Analyze", variant="primary")
            sample = gr.Button("Load a sample history")
        a_md = gr.Markdown()
        a_html = gr.HTML()
        btn.click(analyze, [up, fail_on], [a_md, a_html])
        sample.click(sample_history, None, up)

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
