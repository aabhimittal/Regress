"""Pre-commit consolidation guard: abrupt damage is rejected instead of detected afterwards."""

import copy

import numpy as np
import pytest

from regress import FunctionAgent, MemorySnapshot, Probe, ProbeSuite, RegressionMonitor, Response
from regress.cache import ResponseCache
from regress.guard import ConsolidationGuard
from regress.llm import NoteAgent, RuleBackend
from regress.llm.experiment import run_regime
from regress.llm.world import MultiHopWorld, make_world


class Breakable:
    """An agent whose stages break a given number of probes; cloned with deepcopy."""

    def __init__(self, n=100):
        self.broken: set[int] = set()
        self.n = n

    def respond(self, probe):
        return Response(-1 if probe.input in self.broken else probe.expected, 0.6, [f"m{probe.input % 10}"])

    def memory_snapshot(self):
        return MemorySnapshot([f"m{i}" for i in range(10)], np.eye(10))


def suite(n=100):
    return ProbeSuite([Probe(f"p{i:03d}", i, expected=i % 5) for i in range(n)])


def breaker(k, start):
    def apply(agent):
        agent.broken |= set(range(start, start + k))
    return apply


def test_abrupt_damage_is_rejected_and_the_old_agent_kept():
    m = RegressionMonitor(suite())
    agent = Breakable()
    m.evaluate(agent, 0)
    guard = ConsolidationGuard(m)  # blocks ALARM: 60 broken probes → accuracy + output shift
    kept, report = guard.run_cycle(agent, 1, [("catastrophe", breaker(60, 0))])
    assert kept is agent and not agent.broken
    assert report.metrics["accuracy"] == 1.0 and report.status == "OK"
    assert guard.decisions[0].accepted is False and guard.decisions[0].status == "ALARM"
    assert guard.rejection_rate == 1.0 and guard.rejected()[0].stage == "catastrophe"


def test_cumulative_small_steps_cannot_creep_past_the_tolerance():
    m = RegressionMonitor(suite())
    agent = Breakable()
    m.evaluate(agent, 0)
    guard = ConsolidationGuard(m, block_on=("ALARM", "WATCH"))
    for c in range(1, 8):  # each stage breaks 2 % of probes; every one is under tolerance alone
        agent, rep = guard.run_cycle(agent, c, [(f"step{c}", breaker(2, 2 * c))])
        assert rep.metrics["accuracy"] >= 1.0 - 0.07  # bounded against the baseline
    accepted = [d for d in guard.decisions if d.accepted]
    assert 0 < len(accepted) < 7  # early small steps pass; later ones are refused


def test_alarm_only_guard_lets_watch_level_damage_through():
    m = RegressionMonitor(suite())
    agent = Breakable()
    m.evaluate(agent, 0)
    loose = ConsolidationGuard(m)
    agent, rep = loose.run_cycle(agent, 1, [("drop", breaker(10, 0))])  # accuracy-only: WATCH
    assert loose.decisions[0].accepted and rep.status == "WATCH"


def test_stages_are_independent_and_crashes_are_rejected():
    m = RegressionMonitor(suite())
    agent = Breakable()
    m.evaluate(agent, 0)

    def crash(agent):
        raise RuntimeError("consolidation blew up")

    guard = ConsolidationGuard(m)
    kept, _ = guard.run_cycle(agent, 1, [("good", breaker(1, 0)), ("crash", crash), ("bad", breaker(70, 10)),
                                         ("good2", breaker(1, 99))])
    assert [d.accepted for d in guard.decisions] == [True, False, False, True]
    assert guard.decisions[1].status == "ERROR" and "blew up" in guard.decisions[1].reason
    assert kept.broken == {0, 99}
    assert guard.summary()["rejected"] == 2


def test_configuration_errors_and_custom_clone():
    m = RegressionMonitor(suite())
    with pytest.raises(ValueError):
        ConsolidationGuard(m).run_cycle(Breakable(), 1, [])  # no baseline yet
    with pytest.raises(ValueError):
        ConsolidationGuard(m, block_on=())
    m.evaluate(Breakable(), 0)
    calls = []

    def clone(a):
        calls.append(1)
        return copy.deepcopy(a)

    ConsolidationGuard(m, clone=clone).run_cycle(Breakable(), 1, [("s", breaker(1, 0))])
    assert calls == [1]
    assert ConsolidationGuard(m).summary()["rejection_rate"] == 0.0


def test_committed_report_is_free_with_a_cache():
    agent = NoteAgent(RuleBackend())
    world = make_world("facts", 6, 0)
    agent.add_notes(world.initial_notes())
    m = RegressionMonitor(world.probe_suite(), cache=ResponseCache())
    m.evaluate(agent, 0)
    guard = ConsolidationGuard(m)
    agent, rep = guard.run_cycle(agent, 1, agent.learn_stages(1, world.grow(2), world.all_questions()))
    assert rep.metrics["probe_calls"] == 0  # the committed state was already tested as a candidate


def test_guarded_experiment_keeps_accuracy_and_reports_the_audit_suite():
    unguarded, _, _ = run_regime(RuleBackend(), "recursive_summary", cycles=5, people=12, ablate=False,
                                 audit_fraction=0.25)
    guarded, _, _ = run_regime(RuleBackend(), "recursive_summary", cycles=5, people=12, ablate=False,
                               audit_fraction=0.25, guard="strict", agent_kwargs={"compress_chunks": 2})
    assert guarded.first_alarm is None and unguarded.first_alarm is not None
    assert guarded.accuracy[-1] >= 0.9 > unguarded.accuracy[-1]
    assert guarded.audit_accuracy and len(guarded.audit_accuracy) == 6
    assert guarded.audit_accuracy[-1] > unguarded.audit_accuracy[-1]
    assert guarded.guard["rejected"] > 0 and guarded.guarded
    with pytest.raises(ValueError):
        run_regime(RuleBackend(), "faithful", cycles=1, people=4, guard="sometimes")
    with pytest.raises(ValueError):
        run_regime(RuleBackend(), "faithful", cycles=1, people=4, audit_fraction=1.0)


def test_learn_stages_compose_to_learn():
    w1, w2 = make_world("facts", 8, 3), make_world("facts", 8, 3)
    a = NoteAgent(RuleBackend(), regime="recursive_summary", compress_chunks=3)
    b = NoteAgent(RuleBackend(), regime="recursive_summary", compress_chunks=3)
    a.add_notes(w1.initial_notes())
    b.add_notes(w2.initial_notes())
    a.learn(1, w1.grow(2), w1.all_questions())
    for _, apply in b.learn_stages(1, w2.grow(2), w2.all_questions()):
        apply(b)
    assert list(a.notes.values()) == list(b.notes.values())
    names = [n for n, _ in a.learn_stages(2, [], [])]
    assert names[:2] == ["observe", "self_train"] and names[2:] == ["compress_1/3", "compress_2/3", "compress_3/3"]
    assert [n for n, _ in NoteAgent(RuleBackend()).learn_stages(1, [], [])] == ["observe"]
    with pytest.raises(ValueError):
        NoteAgent(RuleBackend(), compress_chunks=0)


def test_multihop_world_and_two_hop_retrieval():
    w = MultiHopWorld(n_people=6, n_companies=3, seed=1)
    qs = w.initial_questions()
    assert len(qs) == 12 and {q.attribute for q in qs} == {"employer", "work city"}
    assert len(w.initial_notes()) == 3 + 6 and len(w.grow(2)) == 2 and len(w.all_questions()) == 16
    one, two = NoteAgent(RuleBackend(), k=1), NoteAgent(RuleBackend(), k=1, hops=2)
    for ag in (one, two):
        ag.add_notes(w.initial_notes())
    city_q = next(q for q in qs if q.attribute == "work city")
    probe = Probe("x", city_q.question, city_q.value)
    assert two.respond(probe).output.lower() == city_q.value.lower()
    assert len(two.retrieve(city_q.question)) == 2 and len(one.retrieve(city_q.question)) == 1
    with pytest.raises(ValueError):
        NoteAgent(RuleBackend(), hops=3)
    with pytest.raises(ValueError):
        make_world("poetry", 4, 0)
    with pytest.raises(ValueError):
        MultiHopWorld(0)


def test_cli_guard_and_task_flags(tmp_path, capsys):
    from regress.cli import main

    assert main(["experiment", "--cycles", "3", "--people", "8", "--guard", "strict", "--audit-fraction", "0.25",
                 "--out", str(tmp_path / "g")]) == 0
    out = capsys.readouterr().out
    assert "recursive_summary+guard" in out and "guard: rejected" in out and "audit accuracy" in out
    assert main(["experiment", "--cycles", "2", "--people", "6", "--task", "multihop", "--out", str(tmp_path / "m")]) == 0
    with pytest.raises(SystemExit):
        main(["experiment", "--audit-fraction", "1.5", "--out", str(tmp_path / "x")])


def test_function_agents_can_be_guarded_with_deepcopy():
    m = RegressionMonitor(suite(20))
    state = {"bad": False}
    agent = FunctionAgent(lambda p: -1 if state["bad"] else p.expected)
    m.evaluate(agent, 0)
    guard = ConsolidationGuard(m)
    kept, _ = guard.run_cycle(agent, 1, [("noop", lambda a: None)])
    assert guard.decisions[0].accepted and kept is not agent  # a deep copy was committed



def test_commit_always_forces_data_stages_and_retention_blocks_forgetting_new_facts():
    m = RegressionMonitor(suite())
    agent = Breakable()
    agent.learned = set()
    m.evaluate(agent, 0)

    def observe(a):  # brings in data that, on its own, hurts the frozen suite
        a.learned |= {1, 2, 3, 4}
        a.broken |= set(range(0, 60))

    def forget_new(a):  # harmless to the frozen suite, destroys recent knowledge
        a.learned -= {1, 2, 3}

    def retention(a):
        return len(a.learned & {1, 2, 3, 4}) / 4

    guard = ConsolidationGuard(m, commit_always=("observe",), retention=retention, retention_tolerance=0.1)
    agent, rep = guard.run_cycle(agent, 1, [("observe", observe), ("compress", forget_new)])
    assert agent.learned == {1, 2, 3, 4}  # data committed, forgetting rejected
    assert [d.status for d in guard.decisions] == ["FORCED", "RETENTION"]
    assert "retention 1.00 -> 0.25" in guard.decisions[1].reason
    assert rep.status == "ALARM"  # forcing data in does not hide what it did to old knowledge

    def crash(a):
        raise RuntimeError("boom")

    agent2, _ = guard.run_cycle(agent, 2, [("observe_bad", crash)])
    assert agent2 is agent and guard.decisions[-1].status == "ERROR"  # forced stages still cannot crash in
    with pytest.raises(ValueError):
        ConsolidationGuard(m, retention_tolerance=1.0)


def test_learning_aware_guard_keeps_new_facts_the_plain_guard_loses():
    kw = dict(cycles=4, people=10, seed=0, audit_fraction=0.25, ablate=False)
    plain, _, _ = run_regime(RuleBackend(), "recursive_summary", guard="strict",
                             agent_kwargs={"compress_chunks": 2}, **kw)
    learn, _, _ = run_regime(RuleBackend(), "recursive_summary", guard="strict", guard_learning=True,
                             agent_kwargs={"compress_chunks": 2}, **kw)
    faithful, _, _ = run_regime(RuleBackend(), "faithful", **kw)
    assert faithful.fresh_accuracy == 1.0 and faithful.fresh_probes > 0
    assert learn.fresh_accuracy > plain.fresh_accuracy + 0.2
    assert learn.guard["accept_rate_by_stage"]["observe"] == 1.0
    assert plain.guard["accept_rate_by_stage"]["observe"] < 1.0
    with pytest.raises(ValueError):
        run_regime(RuleBackend(), "faithful", guard_learning=True, **kw)

def test_matrix_results_are_indexed_and_aggregated(tmp_path):
    import importlib.util
    import json

    from regress.llm import run_experiment
    from regress.site import build_results_index

    job = tmp_path / "matrix" / "rules-facts-s0"
    regimes = [("faithful", "faithful", {}), ("full", "recursive_summary", {}),
               ("full+guard", "recursive_summary", {"guard": "strict", "agent_kwargs": {"compress_chunks": 2}}),
               ("gradual", "recursive_summary", {"agent_kwargs": {"compress_fraction": 0.25}})]
    summary = run_experiment(RuleBackend(), regimes=regimes, out_dir=job, cycles=3, people=8, seed=0,
                             audit_fraction=0.25, ablate=False)
    (job / "summary.json").write_text(json.dumps(summary, default=str))
    page = build_results_index(tmp_path / "matrix", tmp_path / "site")
    html = page.read_text()
    assert "rules-facts-s0" in html and "guard rejected" in html and "audit" in html
    assert (tmp_path / "site" / "rules-facts-s0" / "report-full+guard.html").is_file()

    spec = importlib.util.spec_from_file_location("aggregate", "experiments/aggregate.py")
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)
    assert agg.main(str(tmp_path / "matrix"), str(tmp_path / "M.md")) == 0
    checks = json.loads((tmp_path / "M.json").read_text())
    assert checks[0]["control_quiet"] and checks[0]["full_caught"]
    assert checks[0]["guard_end_acc"] > checks[0]["unguarded_end_acc"]
    assert checks[0]["control_fresh"] == 1.0 and checks[0]["guard_fresh"] is not None
    assert agg.main(str(tmp_path / "empty"), str(tmp_path / "N.md")) == 1


MATRIX = __import__("pathlib").Path(__file__).resolve().parent.parent / "experiments" / "results" / "matrix"
MATRIX_JOBS = sorted(p.parent for p in MATRIX.glob("*/summary.json")) if MATRIX.exists() else []


@pytest.mark.skipif(not MATRIX_JOBS, reason="no recorded matrix runs present")
@pytest.mark.parametrize("job", MATRIX_JOBS, ids=lambda p: p.name)
def test_recorded_matrix_job_replays_exactly(tmp_path, job):
    """Every committed model × task × seed run re-runs bit-for-bit from its recorded outputs."""
    import json

    from regress.cli import main

    assert main(["experiment", "--backend", "replay", "--replay-from", str(job), "--out", str(tmp_path)]) == 0
    got = json.loads((tmp_path / "summary.json").read_text())
    want = json.loads((job / "summary.json").read_text())
    assert got["separates"] == want["separates"]
    for regime, w in want["results"].items():
        for f in ("first_alarm", "statuses", "accuracy", "audit_accuracy", "guard", "fresh_accuracy"):
            assert got["results"][regime].get(f) == w.get(f), (regime, f)
