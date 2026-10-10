import pytest

from regress import ALARM, OK, WATCH, CollapseDetector, Thresholds

BASE = {
    "accuracy": 0.90,
    "output_jsd": 0.0,
    "change_rate": 0.0,
    "effective_rank": 20.0,
    "mean_cosine_distance": 0.9,
    "near_duplicate_fraction": 0.0,
    "retrieval_entropy": 0.95,
    "retrieval_top1_share": 0.02,
    "mean_confidence": 0.60,
    "overconfidence": -0.30,
    "probe_overfit_gap": 0.0,
    "error_rate": 0.0,
}


def cur(**kw):
    return {**BASE, **kw}


det = CollapseDetector()


def test_unchanged_is_ok():
    a = det.assess(cur(), [BASE])
    assert a.status == OK and not a.signals


def test_single_signal_only_watches():
    a = det.assess(cur(effective_rank=10.0), [BASE])
    assert a.status == WATCH and [s.name for s in a.signals] == ["diversity_drop"]


def test_collapse_signature_alarms_even_with_accuracy_intact():
    a = det.assess(cur(effective_rank=14.0, mean_confidence=0.70), [BASE])
    assert a.status == ALARM and a.collapse_signature
    assert "collapse signature" in a.reason


def test_healthy_learning_raises_confidence_without_alarm():
    # Confidence up *and* accuracy up, diversity intact: that's learning, not collapse.
    a = det.assess(cur(mean_confidence=0.75, accuracy=0.97, overconfidence=-0.22), [BASE])
    assert a.status == WATCH and not a.collapse_signature


def test_two_independent_categories_alarm():
    a = det.assess(cur(accuracy=0.70, retrieval_entropy=0.5), [BASE])
    assert a.status == ALARM and a.categories == ["drift", "retrieval"]


def test_two_signals_in_one_category_do_not():
    a = det.assess(cur(retrieval_entropy=0.5, retrieval_top1_share=0.5), [BASE])
    assert a.status == WATCH and a.categories == ["retrieval"]


def test_weak_churn_alone_never_alarms_even_with_min_categories_one():
    a = CollapseDetector(min_categories=1).assess(cur(change_rate=0.9), [BASE])
    assert a.status == WATCH and a.signals[0].weak


def test_improvements_never_fire():
    better = cur(accuracy=1.0, effective_rank=30.0, mean_cosine_distance=1.0,
                 retrieval_entropy=1.0, retrieval_top1_share=0.0, mean_confidence=0.5,
                 overconfidence=-0.5)
    assert det.assess(better, [BASE]).status == OK


def test_exactly_at_tolerance_does_not_fire():
    t = Thresholds()
    a = det.assess(cur(accuracy=BASE["accuracy"] - t.accuracy_drop), [BASE])
    assert not a.signals


def test_nan_metrics_are_undefined_not_fired():
    a = det.assess(cur(effective_rank=float("nan"), mean_confidence=float("nan")), [BASE])
    assert a.status == OK and {"effective_rank", "mean_confidence"} <= set(a.undefined)


def test_relative_metric_with_zero_baseline_is_undefined():
    a = det.assess(cur(effective_rank=0.0), [cur(effective_rank=0.0)])
    assert "effective_rank" in a.undefined and a.status == OK


def test_constant_warmup_window_has_zero_noise_and_no_division_error():
    a = det.assess(cur(accuracy=0.8), [BASE, BASE, BASE, BASE])
    assert [s.name for s in a.signals] == ["accuracy_drop"]
    assert a.signals[0].tolerance == pytest.approx(Thresholds().accuracy_drop)


def test_noisy_warmup_widens_tolerance():
    window = [cur(accuracy=v) for v in (0.80, 0.90, 1.00, 0.85, 0.95)]
    a = det.assess(cur(accuracy=0.80), window)  # a 0.10 drop is inside this agent's noise
    assert not any(s.name == "accuracy_drop" for s in a.signals)
    assert det.assess(cur(accuracy=0.30), window).status != OK


def test_patience_requires_consecutive_conditions():
    d = CollapseDetector(patience=3)
    bad = cur(effective_rank=10.0, mean_confidence=0.9)
    assert d.assess(bad, [BASE], []).status == WATCH
    assert d.assess(bad, [BASE], [True]).status == WATCH
    assert d.assess(bad, [BASE], [True, True]).status == ALARM
    assert d.assess(bad, [BASE], [True, False, True]).status == WATCH  # streak broken


def test_invalid_configuration():
    with pytest.raises(ValueError):
        CollapseDetector(min_categories=0)
    with pytest.raises(ValueError):
        CollapseDetector(patience=0)


def test_missing_metric_keys_are_tolerated():
    a = det.assess({"accuracy": 0.5}, [{"accuracy": 0.9}])
    assert [s.name for s in a.signals] == ["accuracy_drop"]
    assert "effective_rank" in a.undefined
