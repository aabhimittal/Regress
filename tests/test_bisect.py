import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regress import BisectError, bisect_checkpoints, estimate_onset, locate_onset


class Counter:
    def __init__(self, first_bad, skip=()):
        self.first_bad, self.skip, self.calls = first_bad, set(skip), []

    def __call__(self, c):
        self.calls.append(c)
        if c in self.skip:
            return None
        return c >= self.first_bad


@given(st.integers(2, 400), st.data())
def test_finds_exact_boundary_in_log_evaluations(n, data):
    fb = data.draw(st.integers(1, n - 1))
    ev = Counter(fb)
    r = bisect_checkpoints(range(n), ev, recheck=False)
    assert r.first_bad == fb and r.last_good == fb - 1 and r.exact
    assert len(ev.calls) <= math.ceil(math.log2(n)) + 2


def test_sparse_checkpoint_numbers():
    cycles = [0, 10, 20, 50, 90, 200]
    r = bisect_checkpoints(cycles, lambda c: c >= 50)
    assert (r.last_good, r.first_bad) == (20, 50)


def test_all_good_returns_none():
    r = bisect_checkpoints(range(10), lambda c: False)
    assert r.first_bad is None and r.last_good == 9 and not r.exact


def test_baseline_must_be_good_and_testable():
    with pytest.raises(BisectError, match="already bad"):
        bisect_checkpoints(range(5), lambda c: True)
    with pytest.raises(BisectError, match="could not be evaluated"):
        bisect_checkpoints(range(5), lambda c: None if c == 0 else c > 2)


def test_assume_first_good_skips_the_baseline_call():
    ev = Counter(3)
    bisect_checkpoints(range(8), ev, assume_first_good=True, recheck=False)
    assert 0 not in ev.calls


def test_skipped_checkpoints_away_from_the_boundary_do_not_matter():
    ev = Counter(13, skip={5, 6, 7, 8, 9, 10})
    r = bisect_checkpoints(range(20), ev)
    assert r.first_bad == 13 and r.exact and set(r.skipped) <= {5, 6, 7, 8, 9, 10}


def test_skips_on_the_boundary_yield_a_candidate_range():
    ev = Counter(10, skip={8, 9, 10})
    r = bisect_checkpoints(range(16), ev)
    assert r.last_good == 7 and r.first_bad == 11
    assert r.candidates == [8, 9, 10, 11] and not r.exact
    assert any("hide the exact boundary" in n for n in r.notes)


def test_untestable_tail_is_reported():
    r = bisect_checkpoints(range(6), lambda c: None if c >= 4 else False)
    assert r.first_bad is None and r.last_good == 3
    assert any("could not be evaluated" in n for n in r.notes)


def test_flaky_evaluator_is_flagged_inconsistent():
    seen = {}

    def flaky(c):
        seen[c] = seen.get(c, 0) + 1
        if c == 6:  # bad the first time, good on re-check
            return seen[c] == 1
        return c > 6

    r = bisect_checkpoints(range(12), flaky)
    assert not r.consistent and not r.exact
    assert any("noisy" in n for n in r.notes)


def test_non_monotone_degradation_can_mislead_and_recheck_is_not_a_cure():
    # Bad at 3, recovers at 4-5, bad again from 8: bisection finds *a* boundary (8), not the
    # first bad cycle (3). This is the documented limit of the monotonicity assumption.
    bad = {3, 8, 9, 10, 11}
    r = bisect_checkpoints(range(12), lambda c: c in bad)
    assert r.first_bad == 8


def test_input_validation():
    with pytest.raises(ValueError):
        bisect_checkpoints([], lambda c: False)
    with pytest.raises(ValueError):
        bisect_checkpoints([0, 2, 2, 3], lambda c: False)


def test_single_checkpoint():
    r = bisect_checkpoints([0], lambda c: False)
    assert r.first_bad is None


# --------------------------------------------------------------------------- onset
def test_onset_of_a_step():
    y = [1.0] * 8 + [0.5] * 8
    o = estimate_onset(range(16), y)
    assert o.cycle == 8 and o.model == "step"


def test_onset_of_a_slow_decline_precedes_any_threshold_crossing():
    t = np.arange(25)
    y = np.where(t < 6, 1.0, 1.0 - 0.02 * (t - 6)) + np.random.default_rng(0).normal(0, 0.004, 25)
    crossing = int(np.argmax(y < 0.8))  # where a 20 % threshold would fire
    o = estimate_onset(t, y, direction=-1)
    assert crossing >= 15
    assert abs(o.cycle - 7) <= 1 and o.model == "hinge"


def test_flat_short_and_nan_series():
    assert estimate_onset(range(10), [3.0] * 10).cycle is None
    assert estimate_onset(range(3), [1.0, 0.0, 1.0]).cycle is None
    y = [1.0, float("nan"), 1.0, 1.0, None, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert estimate_onset(range(10), y).cycle == 6


def test_noise_alone_has_no_onset():
    y = np.random.default_rng(1).normal(0, 1, 30)
    assert estimate_onset(range(30), y).cycle is None


def test_direction_ignores_improvements():
    y = [0.5] * 5 + [0.9] * 10  # accuracy went up
    assert estimate_onset(range(15), y, direction=-1).cycle is None
    assert estimate_onset(range(15), y, direction=+1).cycle == 5


class _H:
    def __init__(self, cycle, metrics, status="OK", signals=()):
        self.cycle, self.metrics, self.status, self.signals = cycle, metrics, status, list(signals)


def test_locate_onset_truncates_after_the_alarm_and_takes_lower_median():
    hist = [
        _H(c, {"a": 1.0 if c < 5 else 0.0, "b": 1.0 if c < 7 else 0.0, "c": 1.0 if c < 9 else 3.0},
           status="ALARM" if c >= 9 else "OK")
        for c in range(30)
    ]
    cons, ons = locate_onset(hist, metrics=["a", "b", "c"])
    assert [o.cycle for o in ons] == [5, 7, 9] and cons == 7


def test_locate_onset_with_nothing_signalled():
    cons, ons = locate_onset([_H(c, {"a": 1.0}) for c in range(10)])
    assert cons is None and ons == []
