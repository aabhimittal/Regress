"""Tests for 0.2 features: blame, calibration, suite health, adapters, gate, site, alerts."""

import json
import math

import numpy as np
import pytest
from conftest import ScriptedAgent, make_suite

from regress import CollapseDetector, FunctionAgent, Probe, ProbeSuite, RegressionMonitor, Thresholds
from regress.adapters import HashingEmbedder, memory_from_texts, text_key
from regress.blame import blame
from regress.calibrate import calibrate
from regress.cli import main
from regress.gate import gate, load_history, markdown_summary
from regress.health import suite_health
from regress.sim import SCENARIOS, Simulation
from regress.site import demo_payload, render, report_payload


def run(name, seed=0, cycles=16, detector=None):
    sim = Simulation(SCENARIOS[name], cycles=cycles, seed=seed)
    mon = RegressionMonitor(sim.suite, detector=detector)
    for c in range(cycles + 1):
        mon.evaluate(sim.checkpoint(c), c)
    return sim, mon


class H:  # minimal history row
    def __init__(self, cycle, scores, retrieved=None, status="OK", metrics=None):
        self.cycle, self.scores, self.retrieved, self.status = cycle, scores, retrieved or {}, status
        self.metrics = metrics or {}


# ----------------------------------------------------------------------------- blame
class TestBlame:
    def test_lift_and_ranking_on_a_handmade_history(self):
        base = H(0, {"a": 1, "b": 1, "c": 1, "d": 1}, {"a": ["m1"], "b": ["m2"], "c": ["m3"], "d": ["m3"]})
        # a and b regress; both now retrieve the new memory "bad"; everyone still uses m3.
        cur = H(1, {"a": 0, "b": 0, "c": 1, "d": 1},
                {"a": ["bad", "m3"], "b": ["bad", "m3"], "c": ["m3"], "d": ["m3"]}, status="ALARM")
        rep = blame([base, cur])
        assert rep.cycle == 1 and rep.regressed == ["a", "b"]
        top, hub = rep.suspects[0], next(s for s in rep.suspects if s.memory_id == "m3")
        assert top.memory_id == "bad" and top.regressed_hits == 2 and top.lift == pytest.approx(2.0)
        assert top.first_seen == 1 and top.share_base == 0.0
        assert hub.lift == pytest.approx(1.0)  # used everywhere: not specific to the failures

    def test_no_regressions_and_no_retrievals(self):
        same = H(0, {"a": 1}, {"a": ["m"]})
        assert "no probe regressed" in blame([same, H(1, {"a": 1}, {"a": ["m"]})]).note
        assert "no per-probe retrievals" in blame([same, H(1, {"a": 0}, {})]).note
        with pytest.raises(ValueError):
            blame([same], cycle=9)
        with pytest.raises(ValueError):
            blame([])

    def test_collapse_suspects_are_the_inflated_hub_memories(self):
        sim, mon = run("collapse", seed=0, cycles=14)
        rep = blame(mon.history, cycle=14, top=5)
        agent = sim.checkpoint(14)
        strength = dict(zip(agent.mem.ids, agent.mem.strength))
        top = [strength[s.memory_id] for s in rep.suspects if s.memory_id in strength]
        assert len(rep.regressed) > 20 and top
        assert np.median(top) > 10 * np.median(agent.mem.strength)


# ----------------------------------------------------------------------------- calibration
def _hist(values, metric="accuracy"):
    return [H(i, {}, metrics={metric: v}) for i, v in enumerate(values)]


class TestCalibrate:
    def test_quantile_times_margin_with_floor_and_min_samples(self):
        # accuracy wobbles down by up to 0.1 in healthy runs → tolerance ≈ 0.1 × margin
        hs = [_hist([0.9] + [0.9 - 0.1 * (i % 2) for i in range(20)]) for _ in range(2)]
        cal = calibrate(hs, quantile=0.99, margin=1.5)
        assert cal.thresholds.accuracy_drop == pytest.approx(0.15)
        row = {m.signal: m for m in cal.metrics}
        assert row["accuracy_drop"].reason == "loosened" and row["accuracy_drop"].samples == 40
        assert row["diversity_drop"].reason.startswith("kept default")  # metric absent → default
        flat = calibrate([_hist([0.9] * 30)])
        assert flat.thresholds.accuracy_drop == pytest.approx(Thresholds().accuracy_drop * 0.25)

    def test_improvements_do_not_loosen(self):
        cal = calibrate([_hist([0.5] + [0.9] * 30)])  # accuracy only ever rose
        assert cal.thresholds.accuracy_drop == pytest.approx(0.05 * 0.25)

    @pytest.mark.parametrize("kw", [dict(quantile=0.2), dict(quantile=1.0), dict(margin=0.9)])
    def test_rejects_bad_parameters(self, kw):
        with pytest.raises(ValueError):
            calibrate([_hist([0.9] * 20)], **kw)
        with pytest.raises(ValueError):
            calibrate([])

    def test_out_of_sample_calibration_alarms_earlier_without_false_alarms(self):
        healthy = [run("healthy", seed=s, cycles=16)[1].history for s in (100, 101, 102)]
        det = CollapseDetector(calibrate(healthy).thresholds)
        for s in (0, 1):
            assert run("healthy", seed=s, cycles=16, detector=det)[1].first_alarm() is None
        onset = SCENARIOS["collapse"].onset
        default = run("collapse", seed=0)[1].first_alarm().cycle
        tuned = run("collapse", seed=0, detector=det)[1].first_alarm().cycle
        assert onset <= tuned <= default


def test_thresholds_round_trip_and_validation():
    t = Thresholds(accuracy_drop=0.07)
    assert Thresholds.from_dict(t.to_dict()) == t
    for bad in ({"nope": 1.0}, {"accuracy_drop": -1}, {"accuracy_drop": float("nan")}, {"accuracy_drop": True}):
        with pytest.raises(ValueError):
            Thresholds.from_dict(bad)


# ----------------------------------------------------------------------------- health
def test_suite_health_classifies_every_kind():
    seqs = {
        "sat": [1, 1, 1, 1, 1, 1], "uns": [0, 0, 0, 0, 0, 0], "flk": [1, 0, 1, 0, 1, 0],
        "reg": [1, 1, 1, 0, 0, 0], "imp": [0, 0, 1, 1, 1, 1], "mix": [1, 0, 0, 0, 0, 1],
        "nos": [math.nan] * 6,
    }
    hist = [H(i, {k: float(v[i]) for k, v in seqs.items()}) for i in range(6)]
    rep = suite_health(hist)
    kinds = {p.probe_id: p.kind for p in rep.probes}
    assert kinds == {"sat": "saturated", "uns": "unsolved", "flk": "flaky", "reg": "regressed",
                     "imp": "improved", "mix": "mixed", "nos": "unscored"}
    assert any("flip without a trend" in r for r in rep.recommendations)
    assert rep.ids("flaky") == ["flk"]


def test_suite_health_flags_a_saturated_suite_and_short_histories():
    rep = suite_health([H(i, {f"p{j}": 1.0 for j in range(10)}) for i in range(2)])
    text = " ".join(rep.recommendations)
    assert "can detect regressions but not improvements" in text and "fewer than 3 cycles" in text
    with pytest.raises(ValueError):
        suite_health([])


# ----------------------------------------------------------------------------- adapters
class TestAdapters:
    def test_hashing_embedder(self):
        e = HashingEmbedder(dim=128)
        v = e.embed(["The cat sat on the mat.", "the cat sat on the mat", "Quarterly revenue grew 4%", ""])
        assert v.shape == (4, 128)
        assert np.allclose(np.linalg.norm(v[:3], axis=1), 1.0) and np.all(v[3] == 0)
        assert v[0] @ v[1] > 0.99 and abs(v[0] @ v[2]) < 0.5
        assert np.array_equal(e.embed(["x y"]), HashingEmbedder(dim=128).embed(["x y"]))
        with pytest.raises(ValueError):
            HashingEmbedder(dim=4)

    def test_text_key_ignores_cosmetics_but_not_content(self):
        k = text_key()
        assert k("Paris.") == k("  paris ") == k("ＰＡＲＩＳ")
        assert k("Paris") != k("Lyon")
        assert k("x" * 500).startswith("t#") and k(3) == "i:3"
        assert text_key(strip_punct=False)("a.") != text_key(strip_punct=False)("a")

    def test_memory_from_texts(self):
        snap = memory_from_texts(["a", "b"], ["alpha beta", "gamma"])
        assert snap.embeddings.shape == (2, 256)
        assert len(memory_from_texts([], [])) == 0
        with pytest.raises(ValueError):
            memory_from_texts(["a"], [])

    def test_text_agent_end_to_end(self):
        facts = {f"q{i}": f"answer {i}" for i in range(12)}
        suite = ProbeSuite([Probe(k, k, expected=v) for k, v in facts.items()])
        mem = {"m0": "answer zero"}
        mon = RegressionMonitor(suite, output_key=text_key())
        agent = FunctionAgent(lambda p: facts[p.id].upper() + "!", lambda: memory_from_texts(list(mem), list(mem.values())))
        mon.scorer = lambda p, o: float(text_key()(o) == text_key()(p.expected))
        assert mon.evaluate(agent, 0).metrics["accuracy"] == 1.0


# ----------------------------------------------------------------------------- monitor 0.2
def test_on_alarm_fires_once_per_entry_and_cannot_break_the_loop(suite, agent):
    calls = []
    m = RegressionMonitor(suite, on_alarm=calls.append)
    m.evaluate(agent, 0)
    agent.answer = lambda p: -1
    agent.retrieve = lambda p: ["m0"]
    m.evaluate(agent, 1)
    m.evaluate(agent, 2)  # still alarming: no second page
    assert [r.cycle for r in calls] == [1]
    agent.answer = lambda p: p.expected
    agent.retrieve = lambda p: [agent.ids[(p.input * 3 + k) % 40] for k in range(3)]
    m.evaluate(agent, 3)  # recovers
    agent.answer = lambda p: -1
    agent.retrieve = lambda p: ["m0"]
    m.evaluate(agent, 4)  # re-enters ALARM
    assert [r.cycle for r in calls] == [1, 4]

    def boom(report):
        raise ConnectionError("pager down")

    m2 = RegressionMonitor(suite, on_alarm=boom)
    m2.evaluate(ScriptedAgent(), 0)
    broken = ScriptedAgent()
    broken.answer = lambda p: -1
    broken.retrieve = lambda p: ["m0"]
    r = m2.evaluate(broken, 1)
    assert r.alarmed and "pager down" in r.errors["<on_alarm>"]


def test_accuracy_drop_needs_statistical_support():
    # 20 probes: one flip is a 0.05 drop, well inside sampling noise.
    small = make_suite(20)
    m = RegressionMonitor(small)
    m.evaluate(FunctionAgent(lambda p: p.expected), 0)
    one = m.evaluate(FunctionAgent(lambda p: -1 if p.input == 0 else p.expected), 1)
    two = m.evaluate(FunctionAgent(lambda p: -1 if p.input < 2 else p.expected), 2)
    assert one.metrics["accuracy_delta_se"] > 0
    assert not any(s.name == "accuracy_drop" for s in one.signals + two.signals)
    # 400 probes: a 0.05 drop is 20 probes, which is real.
    big = make_suite(400)
    m = RegressionMonitor(big)
    m.evaluate(FunctionAgent(lambda p: p.expected), 0)
    r = m.evaluate(FunctionAgent(lambda p: -1 if p.input < 24 else p.expected), 1)
    assert any(s.name == "accuracy_drop" for s in r.signals)
    # The gate can be switched off.
    m = RegressionMonitor(small, detector=CollapseDetector(significance_z=0))
    m.evaluate(FunctionAgent(lambda p: p.expected), 0)
    r = m.evaluate(FunctionAgent(lambda p: -1 if p.input < 2 else p.expected), 1)
    assert any(s.name == "accuracy_drop" for s in r.signals)


def test_history_keeps_retrievals_confidences_and_detector_config(tmp_path, suite, agent):
    det = CollapseDetector(Thresholds(accuracy_drop=0.2), min_categories=3, patience=2)
    m = RegressionMonitor(suite, detector=det)
    m.evaluate(agent, 0)
    agent.confidence = float("nan")
    m.evaluate(agent, 1)
    m.save(tmp_path / "h.json")
    m2 = RegressionMonitor.load(tmp_path / "h.json", suite)
    assert m2.detector.config() == det.config()
    assert m2.history[0].retrieved["p00"] == m.history[0].retrieved["p00"] and m2.history[0].retrieved["p00"]
    assert m2.history[0].confidences["p00"] == 0.6 and m2.history[1].confidences["p00"] is None


def test_histories_from_0_1_still_load(tmp_path, suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)
    data = m.to_dict()
    del data["detector"]
    for r in data["history"]:
        del r["retrieved"], r["confidences"]
    (tmp_path / "old.json").write_text(json.dumps(data))
    old = RegressionMonitor.load(tmp_path / "old.json", suite)
    assert old.history[0].retrieved == {} and blame(old.history).note


# ----------------------------------------------------------------------------- gate + site + CLI
def test_gate_latest_vs_any_cycle_and_watch():
    hist = [H(0, {}, status="OK"), H(1, {}, status="ALARM"), H(2, {}, status="WATCH")]
    for h in hist:
        h.reason = h.status.lower()
    assert gate(hist).passed
    assert not gate(hist, any_cycle=True).passed and gate(hist, any_cycle=True).cycle == 1
    assert not gate(hist, fail_on="WATCH").passed
    assert not gate([]).passed
    with pytest.raises(ValueError):
        gate(hist, fail_on="OK")


def test_markdown_summary_names_alarm_onset_and_suspects():
    _, mon = run("collapse", seed=0)
    md = markdown_summary(mon.history, gate(mon.history))
    assert "**FAIL**" in md and "First alarm at cycle" in md and "Suspect memories" in md
    assert "No cycles" in markdown_summary([])


def test_render_neutralises_script_breakouts_and_nan():
    hist = run("healthy", seed=0, cycles=4)[1].history
    payload = report_payload(hist, title="</script><script>alert(1)</script>")
    html = render(payload)
    assert html.count("</script>") == 1  # only the page's own closing tag
    assert "alert(1)</script>" not in html
    data = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert "NaN" not in data and json.loads(data.replace("<\\/", "</"))["meta"]["mode"] == "report"


def test_demo_payload_shape_is_small_and_out_of_sample():
    p = demo_payload(seeds=(0,), cycles=6, calibration_seeds=(100, 101))
    assert set(p["scenarios"]) == set(SCENARIOS) and p["modes"] == ["default", "calibrated"]
    run0 = p["scenarios"]["collapse"]["runs"]["0"]
    assert len(run0["map"]) == 7 and len(run0["modes"]["calibrated"]["cycles"]) == 7
    assert 0 not in p["calibration"]["seeds"]


def test_cli_site_report_gate_calibrate_blame_health(tmp_path, capsys):
    h1, col = tmp_path / "h1.json", tmp_path / "col.json"
    assert main(["--no-color", "demo", "--scenario", "healthy", "--cycles", "8", "--save", str(h1)]) == 0
    assert main(["--no-color", "demo", "--scenario", "collapse", "--cycles", "14", "--save", str(col)]) == 0
    assert main(["gate", str(h1)]) == 0
    summary = tmp_path / "summary.md"
    assert main(["gate", str(col), "--markdown", str(summary)]) == 1
    assert "REGRESS regression check" in summary.read_text()
    thr = tmp_path / "thr.json"
    assert main(["calibrate", str(h1), "--out", str(thr)]) == 0
    assert "accuracy_drop" in json.loads(thr.read_text())["thresholds"]
    assert main(["--no-color", "demo", "--scenario", "collapse", "--cycles", "10", "--thresholds", str(thr)]) == 0
    assert main(["report", str(col), "--html", str(tmp_path / "r.html")]) == 0
    assert "REGRESS Report" in (tmp_path / "r.html").read_text()
    assert main(["blame", str(col), "--top", "3"]) == 0
    assert main(["health", str(col)]) == 0
    assert main(["site", "--out", str(tmp_path / "site"), "--seeds", "0", "--cycles", "6"]) == 0
    assert (tmp_path / "site" / "index.html").stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "FAIL" in out and "regressed against cycle 0" in out
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        load_history(bad)
