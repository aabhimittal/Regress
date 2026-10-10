import json
import math

import numpy as np
import pytest
from conftest import make_suite

from regress import (
    ALARM,
    OK,
    CollapseDetector,
    FunctionAgent,
    MemorySnapshot,
    Probe,
    ProbeSuite,
    ProbeSuiteMismatch,
    ProbeSuiteTampered,
    RegressionMonitor,
    Response,
)


def test_baseline_is_ok_and_drift_is_zero(suite, agent):
    r = RegressionMonitor(suite).evaluate(agent, 0)
    assert r.status == OK
    assert r.metrics["accuracy"] == 1.0
    assert r.metrics["change_rate"] == 0.0 and r.metrics["output_jsd"] == 0.0


def test_cycles_must_strictly_increase(suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)
    m.evaluate(agent, 5)
    for bad in (5, 3):
        with pytest.raises(ValueError):
            m.evaluate(agent, bad)


def test_unrecorded_evaluation_needs_a_baseline_and_leaves_history_alone(suite, agent):
    m = RegressionMonitor(suite)
    with pytest.raises(ValueError):
        m.evaluate(agent, 3, record=False)
    m.evaluate(agent, 0)
    m.evaluate(agent, 7, record=False)
    m.evaluate(agent, 2, record=False)  # older cycles are fine when not recording
    assert [r.cycle for r in m.history] == [0]


def test_a_crashing_probe_is_a_regression_not_a_missing_value(suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)

    def flaky(p):
        if p.input % 5 == 0:
            raise RuntimeError("tool timeout")
        return p.expected

    agent.answer = flaky
    r = m.evaluate(agent, 1)
    assert r.metrics["error_rate"] == pytest.approx(0.2)
    assert r.metrics["accuracy"] == pytest.approx(0.8)
    assert r.metrics["change_rate"] == pytest.approx(0.2)
    assert r.metrics["regressions"] == 4
    assert "RuntimeError: tool timeout" in r.errors["p00"]
    assert r.output_keys["p00"] is None
    assert any(s.name == "error_rate" for s in r.signals)


def test_memory_snapshot_failure_is_recorded_and_metrics_go_undefined(suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)

    def boom():
        raise MemoryError("store offline")

    agent.memory_snapshot = boom
    r = m.evaluate(agent, 1)
    assert "<memory_snapshot>" in r.errors
    assert math.isnan(r.metrics["effective_rank"]) and "effective_rank" in r.undefined
    assert r.metrics["memory_error"] == 1.0


def test_scorer_returning_out_of_range_is_treated_as_an_error(suite, agent):
    m = RegressionMonitor(suite, scorer=lambda p, o: 7.0)
    r = m.evaluate(agent, 0)
    assert r.metrics["error_rate"] == 1.0 and "outside [0, 1]" in r.errors["p00"]


def test_out_of_range_confidence_does_not_crash(suite, agent):
    agent.confidence = 3.0
    r = RegressionMonitor(suite).evaluate(agent, 0)
    assert r.metrics["invalid_confidence"] == len(suite) and math.isnan(r.metrics["mean_confidence"])


def test_mutating_the_suite_mid_run_is_refused(agent):
    x = np.zeros(4)
    s = ProbeSuite([Probe("a", x, 0), Probe("b", 1, 1)])
    m = RegressionMonitor(s)
    m.evaluate(FunctionAgent(lambda p: p.expected), 0)
    x[0] = 1.0
    with pytest.raises(ProbeSuiteTampered):
        m.evaluate(FunctionAgent(lambda p: p.expected), 1)


def test_function_agent_and_unscored_probes():
    s = ProbeSuite([Probe("q1", "hi"), Probe("q2", "yo")])  # no expected answers: drift only
    m = RegressionMonitor(s)
    m.evaluate(FunctionAgent(lambda p: p.input.upper()), 0)
    r = m.evaluate(FunctionAgent(lambda p: "same"), 1)
    assert math.isnan(r.metrics["accuracy"])
    assert r.metrics["change_rate"] == 1.0 and r.metrics["output_jsd"] > 0
    assert math.isnan(r.metrics["effective_rank"])  # FunctionAgent without memory → empty


def test_embedding_dimension_can_change_between_cycles(suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)
    agent.emb = np.random.default_rng(3).normal(size=(40, 64))  # re-embedded with a new model
    r = m.evaluate(agent, 1)
    assert math.isfinite(r.metrics["effective_rank"])


def test_slow_decay_is_caught_against_baseline_not_step_to_step(suite, agent):
    """Boiling frog: memories drift 7 % closer to one point per cycle — no big step, but it adds up."""
    m = RegressionMonitor(suite)
    rng = np.random.default_rng(0)
    target = rng.normal(size=8)
    base = agent.emb.copy()
    statuses = []
    for c in range(16):
        w = 1 - 0.93**c
        agent.emb = (1 - w) * base + w * target  # memories drift toward one point
        agent.confidence = 0.6 + 0.02 * c
        statuses.append(m.evaluate(agent, c).status)
    steps = [abs(a.metrics["effective_rank"] - b.metrics["effective_rank"]) / b.metrics["effective_rank"]
             for a, b in zip(m.history[1:], m.history)]
    assert max(steps) < 0.2  # no single step would trip a 20 % threshold
    assert ALARM in statuses


def test_save_and_load_round_trip_including_nan(tmp_path, suite, agent):
    m = RegressionMonitor(suite)
    m.evaluate(agent, 0)
    agent.answer = lambda p: 0
    m.evaluate(agent, 1)
    path = tmp_path / "h.json"
    m.save(path)
    assert "NaN" not in path.read_text()  # strict JSON
    m2 = RegressionMonitor.load(path, suite)
    assert [r.cycle for r in m2.history] == [0, 1]
    assert m2.history[1].status == m.history[1].status
    for k, v in m.history[1].metrics.items():
        v2 = m2.history[1].metrics[k]
        assert (math.isnan(v) and math.isnan(v2)) or v == v2
    m2.evaluate(agent, 2)  # keeps working after reload, drift still measured vs cycle 0
    assert m2.history[-1].metrics["change_rate"] > 0


def test_load_rejects_a_different_suite(tmp_path, agent):
    m = RegressionMonitor(make_suite(20))
    m.evaluate(agent, 0)
    m.save(tmp_path / "h.json")
    with pytest.raises(ProbeSuiteMismatch):
        RegressionMonitor.load(tmp_path / "h.json", make_suite(21))


def test_load_rejects_unknown_format(tmp_path, suite):
    (tmp_path / "h.json").write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError, match="format"):
        RegressionMonitor.load(tmp_path / "h.json", suite)


def test_warmup_cycles_never_alarm_and_set_the_baseline(suite, agent):
    m = RegressionMonitor(suite, warmup=3)
    for c in range(3):
        agent.confidence = 0.5 + 0.2 * c  # wild warm-up
        assert m.evaluate(agent, c).status == OK
    agent.confidence = 0.7
    assert not any(s.name == "confidence_rise" for s in m.evaluate(agent, 3).signals)


def test_patience_integrates_with_history(suite, agent):
    m = RegressionMonitor(suite, detector=CollapseDetector(patience=2))
    m.evaluate(agent, 0)
    agent.answer = lambda p: -1
    agent.retrieve = lambda p: ["m0"]
    assert m.evaluate(agent, 1).status != ALARM
    assert m.evaluate(agent, 2).status == ALARM


def test_probe_overfitting_is_detected_from_the_public_holdout_split():
    s = ProbeSuite([Probe(f"p{i}", i, i % 3) for i in range(200)], holdout_fraction=0.3, seed=1)
    holdout = {p.id for p in s.holdout}
    m = RegressionMonitor(s)
    m.evaluate(FunctionAgent(lambda p: p.expected if p.input % 2 else -1), 0)
    # Later the agent aces every public probe but gets worse on the hidden ones.
    r = m.evaluate(FunctionAgent(lambda p: -1 if p.id in holdout else p.expected), 1)
    assert r.metrics["public_accuracy"] == 1.0
    assert any(s.name == "probe_overfit" for s in r.signals)


def test_probe_overfit_is_not_flagged_on_noise_alone():
    s = ProbeSuite([Probe(f"p{i}", i, 1) for i in range(120)], holdout_fraction=0.25, seed=3)
    rng = np.random.default_rng(0)
    m = RegressionMonitor(s)
    flips = [rng.random(120) < 0.15 for _ in range(10)]
    m.evaluate(FunctionAgent(lambda p: 1), 0)
    for c, f in enumerate(flips, start=1):
        r = m.evaluate(FunctionAgent(lambda p, f=f: 0 if f[int(p.input)] else 1), c)
        assert not any(sig.name == "probe_overfit" for sig in r.signals)


def test_numpy_outputs_are_compared_by_content():
    s = ProbeSuite([Probe("a", 1, expected=np.array([1, 2]))])
    m = RegressionMonitor(s)
    assert m.evaluate(FunctionAgent(lambda p: np.array([1, 2])), 0).metrics["accuracy"] == 1.0
    r = m.evaluate(FunctionAgent(lambda p: np.array([1, 3])), 1)
    assert r.metrics["accuracy"] == 0.0 and r.metrics["change_rate"] == 1.0


def test_memory_snapshot_validation():
    with pytest.raises(ValueError):
        MemorySnapshot(["a", "b"], np.zeros((3, 4)))
    with pytest.raises(ValueError):
        MemorySnapshot(["a", "a"], np.zeros((2, 4)))
    with pytest.raises(ValueError):
        MemorySnapshot(["a"], np.zeros(4))
    assert len(MemorySnapshot([], [])) == 0


def test_function_agent_accepts_raw_outputs_and_static_memory():
    snap = MemorySnapshot(["x"], np.ones((1, 3)))
    fa = FunctionAgent(lambda p: 42, snap)
    assert fa.respond(Probe("a", 1)) == Response(42)
    assert fa.memory_snapshot() is snap
