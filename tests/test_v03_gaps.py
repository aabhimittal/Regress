"""0.3: independent ruler, uncertainty + sequential tests, ablation, cost controls, real-agent harness."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ScriptedAgent, make_suite

from regress import CollapseDetector, FunctionAgent, MemorySnapshot, Probe, ProbeSuite, RegressionMonitor, Thresholds
from regress.ablation import AblationUnsupported, confirm_blame
from regress.adapters import HashingEmbedder
from regress.bisect import bisect_checkpoints
from regress.cache import ResponseCache
from regress.calibrate import calibrate, calibrate_cusum, family_quantile
from regress.core import Response, state_fingerprint, supports_without
from regress.detector import CusumConfig, cusum
from regress.llm import (
    AnthropicBackend,
    BackendRefusal,
    CachedBackend,
    Completion,
    FactWorld,
    NoteAgent,
    RuleBackend,
    ScriptedBackend,
    answer_score,
    run_experiment,
    run_regime,
)
from regress.sim import SCENARIOS, Simulation


def sim_run(name, seed=0, cycles=14, **kw):
    sim = Simulation(SCENARIOS[name], cycles=cycles, seed=seed)
    mon = RegressionMonitor(sim.suite, **kw)
    for c in range(cycles + 1):
        mon.evaluate(sim.checkpoint(c), c)
    return sim, mon


class TextAgent:
    """Text memories; the agent's own embedding space can be swapped out from under them."""

    def __init__(self, texts, dim=16, seed=0):
        self.texts = list(texts)
        self.ids = [f"t{i}" for i in range(len(self.texts))]
        self.proj = np.random.default_rng(seed).normal(size=(64, dim))
        self.answer = lambda p: p.expected

    def respond(self, probe):
        return Response(self.answer(probe), 0.7, self.ids[probe.input % len(self.ids):][:2])

    def memory_snapshot(self):
        base = HashingEmbedder(dim=64).embed(self.texts)
        return MemorySnapshot(self.ids, base @ self.proj, texts=self.texts)


TEXTS = [f"fact number {i} about topic {i % 7} and item {i * 13 % 29}" for i in range(40)]


# ============================================================================ gap 2: ruler
class TestRuler:
    def test_auto_uses_reference_when_texts_exist(self, suite):
        r = RegressionMonitor(suite).evaluate(TextAgent(TEXTS), 0)
        assert r.ruler.startswith("reference:HashingEmbedder")
        assert math.isfinite(r.metrics["effective_rank"]) and math.isfinite(r.metrics["agent_effective_rank"])

    def test_swapping_the_agents_embedder_fools_the_agent_ruler_not_the_reference(self, suite):
        results = {}
        for ruler in ("agent", "reference"):
            agent = TextAgent(TEXTS, dim=16, seed=0)
            m = RegressionMonitor(suite, ruler=ruler)
            m.evaluate(agent, 0)
            agent.proj = np.random.default_rng(1).normal(size=(64, 2))  # new, 2-D embedding model
            results[ruler] = m.evaluate(agent, 1)
        a, ref = results["agent"], results["reference"]
        assert any(s.name == "diversity_drop" for s in a.signals)  # false collapse on agent ruler
        assert not any(s.name == "diversity_drop" for s in ref.signals)
        assert ref.metrics["embedding_drift"] == 1.0  # dimension changed: flagged either way
        assert any(s.name == "embedding_changed" and s.weak for s in ref.signals)

    def test_same_dimension_re_embedding_is_measured_as_drift(self, suite):
        agent = TextAgent(TEXTS, dim=16, seed=0)
        m = RegressionMonitor(suite)
        assert m.evaluate(agent, 0).metrics["embedding_drift"] == 0.0
        assert m.evaluate(agent, 1).metrics["embedding_drift"] == pytest.approx(0.0, abs=1e-9)
        agent.proj = np.random.default_rng(5).normal(size=(64, 16))
        assert m.evaluate(agent, 2).metrics["embedding_drift"] > 0.5

    def test_reference_without_texts_degrades_with_a_note(self, suite, agent):
        r = RegressionMonitor(suite, ruler="reference").evaluate(agent, 0)
        assert math.isnan(r.metrics["effective_rank"]) and "needs MemorySnapshot.texts" in r.errors["<ruler>"]
        assert r.metrics["memory_size"] == 40

    def test_ruler_change_mid_run_is_reported(self, suite):
        agent = TextAgent(TEXTS)
        m = RegressionMonitor(suite)
        m.evaluate(agent, 0)
        snap = agent.memory_snapshot
        agent.memory_snapshot = lambda: MemorySnapshot(snap().ids, snap().embeddings)  # texts vanish
        assert "not comparable" in m.evaluate(agent, 1).errors["<ruler>"]

    def test_custom_and_invalid_rulers(self, suite):
        class Const:
            dim = 3

            def embed(self, texts):
                return np.ones((len(texts), 3))

        r = RegressionMonitor(suite, ruler=Const()).evaluate(TextAgent(TEXTS), 0)
        assert r.metrics["effective_rank"] == pytest.approx(1.0) and r.ruler == "reference:Const:3"
        with pytest.raises(ValueError):
            RegressionMonitor(suite, ruler="ruler")
        with pytest.raises(ValueError):
            RegressionMonitor(suite, ruler=object())

    def test_snapshot_text_validation(self):
        with pytest.raises(ValueError):
            MemorySnapshot(["a", "b"], np.zeros((2, 3)), texts=["only one"])


# ============================================================================ gap 3: statistics
class TestUncertaintyAndSequential:
    def test_standard_errors_are_reported(self, suite, agent):
        r = RegressionMonitor(suite).evaluate(agent, 0)
        for k in ("effective_rank_se", "mean_cosine_distance_se", "retrieval_entropy_se"):
            assert math.isfinite(r.metrics[k]) and r.metrics[k] >= 0
        assert r.metrics["mean_confidence_se"] == pytest.approx(0.0, abs=1e-12)  # every probe reported 0.6
        assert math.isnan(RegressionMonitor(suite, bootstrap=0).evaluate(agent, 0).metrics["effective_rank_se"])

    def test_tiny_memory_diversity_moves_need_significance(self, suite):
        d = CollapseDetector()
        base = {"effective_rank": 3.0}
        assert not d.signals({"effective_rank": 2.2, "effective_rank_se": 0.6}, [base])[0]
        assert d.signals({"effective_rank": 2.2, "effective_rank_se": 0.05}, [base])[0]

    def test_family_quantile(self):
        assert family_quantile(1, 1, 0.05) == pytest.approx(0.95)
        assert family_quantile(365, 12, 0.05) == pytest.approx(1 - 0.05 / 4380)
        for bad in [(0, 1, 0.05), (1, 0, 0.05), (1, 1, 0.0), (1, 1, 1.0)]:
            with pytest.raises(ValueError):
                family_quantile(*bad)

    def test_horizon_calibration_is_never_tighter(self):
        rng = np.random.default_rng(0)
        hs = [[SimpleNamespace(metrics={"accuracy": 0.9 - abs(x)}) for x in [0.0] + list(rng.normal(0, 0.02, 40))]
              for _ in range(3)]
        plain, wide = calibrate(hs), calibrate(hs, horizon=1000)
        assert wide.thresholds.accuracy_drop >= plain.thresholds.accuracy_drop

    def test_cusum_math(self):
        assert cusum([1.0, 1.0, 1.0], k=0.5) == pytest.approx(1.5)
        assert cusum([1.0, -5.0, 1.0], k=0.5) == pytest.approx(0.5)  # floored at zero, resumes
        assert cusum([10.0], k=0.0, clip=3.0) == 3.0
        assert cusum([float("nan"), None, 1.0], k=0.0) == 1.0
        with pytest.raises(ValueError):
            CusumConfig(h=0)

    def test_cusum_catches_a_persistent_sub_threshold_shift_and_ignores_noise(self):
        det = CollapseDetector(cusum=CusumConfig(k=0.25, h=3.0), significance_z=0)
        base = {"effective_rank": 20.0, "mean_confidence": 0.6}

        def run(shift, n=40, seed=0):
            rng = np.random.default_rng(seed)
            prior, first = [], None
            for t in range(n):
                er = 20.0 * (1 - shift - rng.normal(0, 0.03))  # tolerance is a 20 % drop
                a = det.assess({"effective_rank": er, "mean_confidence": 0.6}, [base], (), prior)
                prior.append(a.moves)
                if first is None and any(s.name == "persistent_diversity_drop" for s in a.signals):
                    first = t
            return first

        assert run(0.14) is not None  # 70 % of tolerance, every cycle: caught
        assert all(run(0.0, seed=s) is None for s in range(5))  # zero-mean noise: never

    def test_persistent_signals_count_toward_the_signature(self):
        det = CollapseDetector(cusum=CusumConfig(k=0.0, h=1.0), significance_z=0)
        base = {"effective_rank": 20.0, "mean_confidence": 0.6}
        cur = {"effective_rank": 18.0, "mean_confidence": 0.7}  # diversity 10 % (sub-threshold)
        a = det.assess(cur, [base], (), [{"diversity_drop": 0.6}] * 3)
        assert {"persistent_diversity_drop", "confidence_rise"} <= {s.name for s in a.signals}
        assert a.collapse_signature and a.status == "ALARM"

    def test_cusum_config_round_trip_and_calibration(self):
        det = CollapseDetector(cusum=CusumConfig(k=0.3, h=6.0))
        assert CollapseDetector.from_config(det.config()).cusum == CusumConfig(k=0.3, h=6.0)
        healthy = [[SimpleNamespace(moves={})] + [SimpleNamespace(moves={"diversity_drop": 0.5})] * 10]
        cfg = calibrate_cusum(healthy, k=0.25, margin=1.5)
        assert cfg.h == pytest.approx(10 * 0.25 * 1.5)
        assert calibrate_cusum([[SimpleNamespace(moves={})]]).h == 2.0  # floor
        with pytest.raises(ValueError):
            calibrate_cusum([], margin=0.5)

    def test_moves_are_recorded_and_persisted(self, tmp_path, suite, agent):
        m = RegressionMonitor(suite)
        m.evaluate(agent, 0)
        agent.confidence = 0.65
        r = m.evaluate(agent, 1)
        assert r.moves["confidence_rise"] == pytest.approx(0.05 / 0.05)
        m.save(tmp_path / "h.json")
        assert RegressionMonitor.load(tmp_path / "h.json", suite).history[1].moves == r.moves


# ============================================================================ gap 4: ablation
class TestAblation:
    def test_confirms_the_collapse_hubs_against_random_controls(self):
        sim, mon = sim_run("collapse", cycles=12)
        res = confirm_blame(mon, sim.checkpoint(12), cycle=12, top=5, controls=4)
        assert res.verdict == "confirmed"
        assert res.recovered > max(res.control_recovered)
        assert set(res.individual) == set(res.suspects) and res.collateral_checked == 40
        assert res.calls > 0 and 0 < res.recovery_rate <= 1

    def test_refutes_when_the_damage_is_not_in_memory(self, suite):
        class Broken(ScriptedAgent):
            def without(self, ids):
                return self  # memory removal changes nothing: the bug is in answering

        agent = Broken()
        m = RegressionMonitor(suite)
        m.evaluate(agent, 0)
        agent.answer = lambda p: -1 if p.input % 2 else p.expected
        m.evaluate(agent, 1)
        res = confirm_blame(m, agent, cycle=1, top=2, controls=3)
        assert res.verdict == "refuted" and res.recovered == 0

    def test_needs_without(self, suite, agent):
        m = RegressionMonitor(suite)
        m.evaluate(agent, 0)
        assert not supports_without(agent)
        with pytest.raises(AblationUnsupported):
            confirm_blame(m, agent)

    def test_small_and_degenerate_cases(self, suite):
        class A(ScriptedAgent):
            def without(self, ids):
                return self

        agent = A()
        m = RegressionMonitor(suite)
        m.evaluate(agent, 0)
        assert confirm_blame(m, agent).note  # nothing regressed
        agent.answer = lambda p: -1 if p.input == 0 else p.expected
        m.evaluate(agent, 1)
        res = confirm_blame(m, agent, cycle=1, top=3, controls=2)
        assert res.verdict == "inconclusive" and "fewer than 3" in res.note
        with pytest.raises(ValueError):
            confirm_blame(m, agent, top=0)


# ============================================================================ gap 5: cost
class TestCache:
    def test_bisect_replays_cost_nothing_with_a_cache(self):
        sim = Simulation(SCENARIOS["collapse"], cycles=12, seed=0)
        cache = ResponseCache()
        mon = RegressionMonitor(sim.suite, cache=cache)
        for c in range(13):
            mon.evaluate(sim.checkpoint(c), c)
        before = cache.stats.hits
        bisect_checkpoints(range(13), lambda c: mon.evaluate(sim.checkpoint(c), c, record=False).alarmed)
        rep = mon.evaluate(sim.checkpoint(12), 12, record=False)
        assert rep.metrics["probe_calls"] == 0 and rep.metrics["cache_hits"] == len(sim.suite)
        assert cache.stats.hits > before

    def test_unchanged_agent_is_free_and_changed_agent_is_not(self):
        sim = Simulation(SCENARIOS["healthy"], cycles=1, seed=0)
        mon = RegressionMonitor(sim.suite, cache=ResponseCache())
        a = sim.checkpoint(0)
        mon.evaluate(a, 0)
        assert mon.evaluate(a, 1).metrics["probe_calls"] == 0
        assert mon.evaluate(sim.checkpoint(1), 2).metrics["probe_calls"] == len(sim.suite)
        assert mon.cost()["cache_hits"] == len(sim.suite)

    def test_agents_without_fingerprints_are_never_cached(self, suite, agent):
        cache = ResponseCache()
        m = RegressionMonitor(suite, cache=cache)
        m.evaluate(agent, 0)
        assert m.evaluate(agent, 1).metrics["probe_calls"] == len(suite)
        assert cache.stats.uncacheable == 2 * len(suite) and state_fingerprint(agent) is None

    def test_persistence_corruption_lru_and_isolation(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c = ResponseCache(path, max_entries=2)
        c.put("a", Response("x", 0.5, ["m1"]))
        c.put("b", Response({"n": 1}, float("nan"), []))
        c.put("c", Response(object(), None, []))  # not JSON: memory only
        c.put("d", Response("y", None, []))
        assert len(c) == 2 and c.get("a") is None  # evicted
        got = c.get("d")
        got.retrieved.append("poison")
        assert c.get("d").retrieved == []
        with path.open("a") as fh:
            fh.write("{not json\n")
        c2 = ResponseCache(path)
        assert c2.skipped_lines == 1 and c2.get("a").output == "x" and math.isnan(c2.get("b").confidence or float("nan"))
        with pytest.raises(ValueError):
            ResponseCache(max_entries=0)

    def test_fingerprint_that_raises_is_treated_as_absent(self):
        class Weird:
            def state_fingerprint(self):
                raise RuntimeError

        assert state_fingerprint(Weird()) is None


class TestScreening:
    def test_quiet_cycles_run_only_the_screen(self):
        sim, mon = sim_run("healthy", cycles=6, screen=40)
        later = mon.history[1:]
        assert all(r.screened and r.metrics["probes_evaluated"] == 40 for r in later)
        assert all(r.status == "OK" for r in later)
        assert mon.cost()["saved_fraction"] > 0.6

    def test_trouble_escalates_to_the_full_suite(self):
        _, mon = sim_run("collapse", cycles=14, screen=40)
        alarmed = [r for r in mon.history if r.alarmed]
        assert alarmed and all(not r.screened and r.metrics["probes_evaluated"] == 160 for r in alarmed)
        assert any(r.metrics["escalated"] for r in mon.history)
        _, full = sim_run("collapse", cycles=14)
        assert abs(mon.first_alarm().cycle - full.first_alarm().cycle) <= 1

    def test_subset_is_stable_stratified_and_bounded(self):
        s = ProbeSuite([Probe(f"p{i}", i, 1, tags=("a" if i % 3 else "b",)) for i in range(90)], 0.3, seed=2)
        m = RegressionMonitor(s, screen=20)
        ids = m.screen_ids()
        assert len(ids) == 20 and ids == m.screen_ids()
        assert any(s.is_holdout(i) for i in ids) and any(not s.is_holdout(i) for i in ids)
        assert {s[i].tags[0] for i in ids} == {"a", "b"}
        assert RegressionMonitor(s, screen=500).screen_ids() == [p.id for p in s]
        for kw in (dict(screen=0), dict(screen_margin=0), dict(screen_margin=1.5), dict(bootstrap=-1)):
            with pytest.raises(ValueError):
                RegressionMonitor(s, **kw)

    def test_known_limit_regressions_outside_the_screen_are_not_seen(self):
        suite = make_suite(80)
        m = RegressionMonitor(suite, screen=20)
        screened = set(m.screen_ids())
        m.evaluate(FunctionAgent(lambda p: p.expected), 0)
        r = m.evaluate(FunctionAgent(lambda p: -1 if p.id not in screened else p.expected), 1)
        assert r.status == "OK" and r.screened  # documented trade-off
        assert m.evaluate(FunctionAgent(lambda p: -1 if p.id not in screened else p.expected), 2,
                          full=True).status != "OK"

    def test_probe_errors_always_escalate(self):
        suite = make_suite(60)
        m = RegressionMonitor(suite, screen=10)
        m.evaluate(FunctionAgent(lambda p: p.expected), 0)
        first = m.screen_ids()[0]

        def flaky(p):
            if p.id == first:
                raise TimeoutError
            return p.expected

        assert m.evaluate(FunctionAgent(flaky), 1).metrics["escalated"] == 1.0


# ============================================================================ gap 1: real agents
class TestRealAgentHarness:
    def test_world_is_deterministic_and_names_unique(self):
        a, b = FactWorld(10, seed=3), FactWorld(10, seed=3)
        assert [f.text for f in a.facts] == [f.text for f in b.facts]
        assert len(set(a.people)) == 10 and len(a.facts) == 30
        a.add_people(5)
        assert len(set(a.people)) == 15
        assert len(a.probe_suite()) == 45
        with pytest.raises(ValueError):
            FactWorld(0)

    def test_answer_score(self):
        p = Probe("x", "q", expected="glassblower")
        assert answer_score(p, "Glassblower.") == 1.0
        assert answer_score(p, "a glassblower") == 1.0
        assert answer_score(p, "unknown") == 0.0
        assert answer_score(p, "one two three four five six seven glassblower nine") == 0.0  # too long
        assert math.isnan(answer_score(Probe("y", "q"), "x"))

    def test_note_agent_capabilities(self):
        agent = NoteAgent(RuleBackend())
        ids = agent.add_notes(["Ab Cd's pet is otter.", "Ab Cd's pet is otter.", "Ef Gh's pet is lynx."])
        assert len(ids) == 2  # exact duplicate dropped
        r = agent.respond(Probe("q", "What is Ab Cd's pet?", "otter"))
        assert r.output == "otter" and r.confidence == 0.9 and r.retrieved[0] == ids[0]
        snap = agent.memory_snapshot()
        assert snap.texts and snap.embeddings.shape == (2, 512)
        fp = agent.state_fingerprint()
        smaller = agent.without([ids[0]])
        assert len(smaller.notes) == 1 and len(agent.notes) == 2 and smaller.state_fingerprint() != fp
        assert smaller.backend is agent.backend
        with pytest.raises(ValueError):
            NoteAgent(RuleBackend(), regime="chaos")

    def test_recursive_summary_compresses_and_self_trains(self):
        world = FactWorld(9, seed=1)
        agent = NoteAgent(RuleBackend(), regime="recursive_summary", self_train=2)
        agent.add_notes([f.text for f in world.facts], cycle=0)
        log = agent.learn(1, world.add_people(2), world.facts)
        assert log["summaries"] > 0 and log["removed"] >= 2 * log["summaries"] and log["self_notes"] >= 1
        assert len(agent.notes) < 27 + 6

    def test_rule_backend_experiment_separates_the_regimes(self, tmp_path):
        s = run_experiment(RuleBackend(), cycles=5, people=12, seed=0, out_dir=tmp_path)
        assert s["separates"] is True
        bad = s["results"]["recursive_summary"]
        assert bad["first_alarm"] is not None and bad["bisect"]["consistent"]
        assert bad["cache_hits"] > 0  # bisection replays were served from the cache
        for f in ("summary.json", "history-faithful.json", "history-recursive_summary.json",
                  "report-recursive_summary.html"):
            assert (tmp_path / f).exists()
        assert json.loads((tmp_path / "summary.json").read_text())["backend"] == "rules"

    def test_cached_backend(self, tmp_path):
        inner = ScriptedBackend(lambda s, u, n: Completion(u.upper(), 0.5))
        cb = CachedBackend(inner, tmp_path / "c.jsonl")
        assert cb.complete("s", "hi", 5).text == "HI" and cb.complete("s", "hi", 5).text == "HI"
        assert inner.calls == 1 and cb.hits == 1
        cb2 = CachedBackend(ScriptedBackend(lambda *a: "never"), tmp_path / "c.jsonl")
        assert cb2.complete("s", "hi", 5).confidence == 0.5 and cb2.calls == 0

    def test_anthropic_backend_request_shape_confidence_and_refusal(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            if "refuse" in kw["messages"][0]["content"]:
                return SimpleNamespace(stop_reason="refusal", stop_details="cyber", content=[])
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="thinking", text=""),
                                            SimpleNamespace(type="text", text="Ostrel || 0.85")])

        fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        b = AnthropicBackend(client=fake)
        out = b.complete("sys", "Question?", 12)
        assert out.text == "Ostrel" and out.confidence == pytest.approx(0.85)
        kw = calls[0]
        assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
        assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["max_tokens"] >= 1024
        assert "confidence" in kw["system"]
        with pytest.raises(BackendRefusal):
            b.complete("sys", "please refuse", 12)

    def test_refusals_become_probe_errors_not_crashes(self):
        def fn(system, user, n):
            raise BackendRefusal("declined")

        agent = NoteAgent(ScriptedBackend(fn))
        agent.add_notes(["Ab Cd's pet is otter."])
        suite = ProbeSuite([Probe("q", "What is Ab Cd's pet?", "otter")])
        r = RegressionMonitor(suite, scorer=answer_score).evaluate(agent, 0)
        assert r.metrics["error_rate"] == 1.0 and "BackendRefusal" in r.errors["q"]

    @pytest.mark.skipif(not os.environ.get("REGRESS_GGUF"), reason="set REGRESS_GGUF to a local model to run")
    def test_local_model_smoke(self, tmp_path):  # pragma: no cover - needs a model file
        from regress.llm import LlamaCppBackend

        res, _, _ = run_regime(LlamaCppBackend(os.environ["REGRESS_GGUF"]), "faithful", cycles=1, people=4,
                               out_dir=tmp_path, ablate=False)
        assert res.accuracy[0] > 0.5


def test_thresholds_include_embedding_changed():
    assert Thresholds().embedding_changed == 0.2


def test_cli_experiment(tmp_path, capsys):
    from regress.cli import main

    assert main(["experiment", "--backend", "rules", "--cycles", "4", "--people", "10", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "separates regimes: True" in out and "recursive_summary" in out
    with pytest.raises(SystemExit):
        main(["experiment", "--backend", "llamacpp", "--out", str(tmp_path)])  # needs --model
    with pytest.raises(SystemExit):
        main(["experiment", "--cycles", "0", "--out", str(tmp_path)])


RESULTS = __import__("pathlib").Path(__file__).resolve().parent.parent / "experiments" / "results"
RECORDED_RUNS = sorted(p for p in RESULTS.glob("*") if (p / "llm-cache.jsonl").exists()) if RESULTS.exists() else []


@pytest.mark.skipif(not RECORDED_RUNS, reason="no recorded real-model runs present")
@pytest.mark.parametrize("recorded", RECORDED_RUNS, ids=lambda p: p.name)
def test_recorded_real_model_run_replays_exactly(tmp_path, recorded):
    """Re-run a committed real-model experiment from its recorded outputs, without the model."""
    from regress.cli import main

    RECORDED = recorded
    assert main(["experiment", "--backend", "replay", "--replay-from", str(RECORDED), "--out", str(tmp_path)]) == 0
    got = json.loads((tmp_path / "summary.json").read_text())
    want = json.loads((RECORDED / "summary.json").read_text())
    assert got["separates"] is True and got["backend"] == want["backend"]
    for regime in want["results"]:
        for f in ("first_alarm", "statuses", "accuracy", "onset", "bisect"):
            assert got["results"][regime][f] == want["results"][regime][f], (regime, f)
    assert got["results"]["recursive_summary"]["ablation"]["verdict"] == "refuted"


def test_replay_diverging_from_the_recording_fails_loudly(tmp_path):
    from regress.llm import ReplayBackend

    agent = NoteAgent(CachedBackend(ReplayBackend("nothing-recorded"), tmp_path / "empty.jsonl"))
    agent.add_notes(["Ab Cd's pet is otter."])
    with pytest.raises(KeyError, match="diverged"):
        agent.backend.complete("s", "u", 5)
    with pytest.raises(SystemExit):
        from regress.cli import main

        main(["experiment", "--backend", "replay", "--out", str(tmp_path)])  # needs --replay-from


def test_gradual_compression_is_slower_damage():
    full, _, _ = run_regime(RuleBackend(), "recursive_summary", cycles=6, people=16, ablate=False)
    gentle, _, _ = run_regime(RuleBackend(), "recursive_summary", cycles=6, people=16, ablate=False,
                              agent_kwargs={"compress_fraction": 0.25})
    assert gentle.accuracy[2] > full.accuracy[2]
    assert gentle.first_alarm is not None and gentle.first_alarm >= full.first_alarm
    with pytest.raises(ValueError):
        NoteAgent(RuleBackend(), compress_fraction=0)
