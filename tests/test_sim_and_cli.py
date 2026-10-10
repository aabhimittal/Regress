import numpy as np
import pytest

from regress import OK, RegressionMonitor, bisect_checkpoints, locate_onset
from regress.cli import main
from regress.sim import SCENARIOS, Simulation


def run(name, seed=0, cycles=16):
    sim = Simulation(SCENARIOS[name], cycles=cycles, seed=seed)
    mon = RegressionMonitor(sim.suite)
    for c in range(cycles + 1):
        mon.evaluate(sim.checkpoint(c), c)
    return sim, mon


def test_simulation_is_deterministic():
    a, b = run("collapse", seed=3)[1], run("collapse", seed=3)[1]
    assert [r.metrics for r in a.history] == [r.metrics for r in b.history]


def test_checkpoints_are_independent_copies():
    sim = Simulation(SCENARIOS["healthy"], cycles=2)
    ag = sim.checkpoint(1)
    ag.mem.emb[:] = 0
    assert np.any(sim.checkpoint(1).mem.emb != 0)
    with pytest.raises(IndexError):
        sim.checkpoint(3)
    with pytest.raises(ValueError):
        Simulation(SCENARIOS["healthy"], cycles=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_healthy_learning_never_alarms(seed):
    _, mon = run("healthy", seed=seed, cycles=20)
    assert all(r.status == OK for r in mon.history)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_collapse_alarms_after_onset_and_never_before(seed):
    onset = SCENARIOS["collapse"].onset
    _, mon = run("collapse", seed=seed)
    first = mon.first_alarm()
    assert first is not None and onset <= first.cycle <= onset + 5
    assert all(r.status == OK for r in mon.history if r.cycle < onset)


def test_collapse_signature_precedes_large_accuracy_loss():
    _, mon = run("collapse", seed=0)
    first = mon.first_alarm()
    assert first.collapse_signature
    assert mon.history[0].metrics["accuracy"] - first.metrics["accuracy"] < 0.05


def test_leak_is_caught_by_the_holdout_split():
    _, mon = run("leak", seed=0)
    late = mon.history[-1]
    assert late.metrics["public_accuracy"] == 1.0
    assert late.metrics["holdout_accuracy"] < mon.history[0].metrics["holdout_accuracy"] - 0.1
    assert any(s.name == "probe_overfit" for s in late.signals)


def test_slow_collapse_onset_estimate_beats_the_alarm():
    sc = SCENARIOS["slow"]
    _, mon = run("slow", seed=0, cycles=20)
    first = mon.first_alarm()
    onset, _ = locate_onset(mon.history)
    assert first.cycle - sc.onset >= 2  # the alarm trails...
    assert abs(onset - sc.onset) < first.cycle - sc.onset  # ...the onset estimate does not


def test_bisect_over_checkpoints_agrees_with_the_recorded_history():
    sim, mon = run("collapse", seed=1)
    r = bisect_checkpoints(range(17), lambda c: mon.evaluate(sim.checkpoint(c), c, record=False).alarmed)
    assert r.first_bad == mon.first_alarm().cycle and r.consistent


def test_cli_demo_and_show(tmp_path, capsys):
    path = tmp_path / "h.json"
    assert main(["--no-color", "demo", "--scenario", "collapse", "--cycles", "12", "--save", str(path)]) == 0
    out = capsys.readouterr().out
    assert "ALARM first raised at cycle" in out and "bisect" in out and "\033[" not in out
    assert main(["--no-color", "show", str(path)]) == 0
    assert "trends" in capsys.readouterr().out


def test_cli_demo_all_scenarios_saves_one_file_each(tmp_path, capsys):
    assert main(["--no-color", "demo", "--scenario", "all", "--cycles", "6", "--save", str(tmp_path / "x.json")]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"x-{n}.json" for n in SCENARIOS)
    assert "no alarm" in capsys.readouterr().out


def test_cli_rejects_bad_cycles():
    with pytest.raises(SystemExit):
        main(["demo", "--cycles", "0"])


def test_sparkline_edge_cases():
    from regress.report import sparkline

    assert sparkline([]) == ""
    assert sparkline([float("nan")] * 3) == "   "
    assert sparkline([2.0, 2.0, 2.0]) == "▄▄▄"
    assert sparkline([0.0, float("inf"), 1.0]) == "▁ █"
