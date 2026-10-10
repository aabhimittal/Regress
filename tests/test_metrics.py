import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from regress.metrics import (
    confidence_stats,
    diversity,
    drift,
    expected_calibration_error,
    gini,
    jensen_shannon,
    mean_cosine_distance,
    normalize_rows,
    retrieval_stats,
)


# ----------------------------------------------------------------------------- diversity
class TestDiversity:
    def test_empty_and_single(self):
        d0 = diversity(np.zeros((0, 5)))
        assert d0.n == 0 and math.isnan(d0.mean_cosine_distance) and math.isnan(d0.effective_rank)
        d0b = diversity(np.array([]))  # 1-D empty is tolerated as "no memories"
        assert d0b.n == 0
        d1 = diversity(np.ones((1, 5)))
        assert d1.effective_rank == 1.0 and math.isnan(d1.mean_cosine_distance)

    def test_identical_memories_are_fully_collapsed(self):
        d = diversity(np.tile([1.0, 2.0, 3.0], (30, 1)))
        assert d.mean_cosine_distance == pytest.approx(0.0, abs=1e-9)
        assert d.effective_rank == pytest.approx(1.0)
        assert d.near_duplicate_fraction == 1.0

    def test_orthonormal_basis_is_maximally_diverse(self):
        d = diversity(np.eye(12))
        assert d.mean_cosine_distance == pytest.approx(1.0)
        assert d.effective_rank == pytest.approx(12.0)
        assert d.effective_rank_ratio == pytest.approx(1.0)
        assert d.near_duplicate_fraction == 0.0

    def test_antipodal_collapse_fools_pairwise_distance_but_not_effective_rank(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=16)
        collapsed = np.vstack([u + 1e-3 * rng.normal(size=16) for _ in range(50)]
                              + [-u + 1e-3 * rng.normal(size=16) for _ in range(50)])
        healthy = rng.normal(size=(100, 16))
        dc, dh = diversity(collapsed), diversity(healthy)
        assert dc.mean_cosine_distance > dh.mean_cosine_distance  # the blind spot
        assert dc.effective_rank < 1.1 < dh.effective_rank / 5  # effective rank sees it

    def test_zero_and_non_finite_rows_are_dropped_and_counted(self):
        emb = np.array([[1.0, 0], [0, 0], [np.nan, 1], [np.inf, 0], [0, 1.0]])
        d = diversity(emb)
        assert d.n == 2 and d.dropped == 3
        assert d.mean_cosine_distance == pytest.approx(1.0)

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError):
            normalize_rows(np.zeros((2, 2, 2)))

    def test_scale_and_permutation_invariance(self):
        rng = np.random.default_rng(1)
        e = rng.normal(size=(40, 8))
        scales = rng.uniform(0.01, 100, size=(40, 1))
        a, b = diversity(e), diversity(e[rng.permutation(40)] * scales[rng.permutation(40)])
        assert a.mean_cosine_distance == pytest.approx(b.mean_cosine_distance)
        assert a.effective_rank == pytest.approx(b.effective_rank)

    @settings(max_examples=60, deadline=None)
    @given(arrays(np.float64, st.tuples(st.integers(2, 30), st.integers(1, 6)),
                  elements=st.floats(-10, 10, allow_nan=False, width=64)))
    def test_closed_form_matches_brute_force(self, emb):
        unit, _ = normalize_rows(emb)
        if unit.shape[0] < 2:
            return
        n = unit.shape[0]
        sims = unit @ unit.T
        brute = 1.0 - (sims.sum() - np.trace(sims)) / (n * (n - 1))
        assert mean_cosine_distance(unit) == pytest.approx(float(np.clip(brute, 0, 2)), abs=1e-9)


# ----------------------------------------------------------------------------- retrieval
class TestRetrieval:
    def test_uniform_is_one_and_single_hub_is_zero(self):
        mem = [f"m{i}" for i in range(8)]
        assert retrieval_stats([[m] for m in mem], mem).normalized_entropy == pytest.approx(1.0)
        hub = retrieval_stats([["m0"]] * 50, mem)
        assert hub.normalized_entropy == pytest.approx(0.0)
        assert hub.top1_share == 1.0 and hub.gini == pytest.approx(7 / 8)

    def test_small_probe_set_against_big_memory_is_not_called_collapsed(self):
        mem = [f"m{i}" for i in range(10_000)]
        r = retrieval_stats([[f"m{i}"] for i in range(10)], mem)
        assert r.normalized_entropy == pytest.approx(1.0)
        assert r.coverage == pytest.approx(10 / 10_000)

    def test_no_events_and_single_memory_are_undefined_not_zero(self):
        assert math.isnan(retrieval_stats([], ["a"]).normalized_entropy)
        assert math.isnan(retrieval_stats([[], []], ["a"]).normalized_entropy)
        assert math.isnan(retrieval_stats([["a"]] * 5, ["a"]).normalized_entropy)

    def test_phantom_ids_are_excluded_and_reported(self):
        r = retrieval_stats([["a", "ghost"], ["b", "ghost2"]], ["a", "b"])
        assert r.phantom_rate == 0.5 and r.events == 2 and r.normalized_entropy == pytest.approx(1.0)

    def test_duplicate_ids_within_one_query_count_once(self):
        assert retrieval_stats([["a", "a", "a"], ["b"]], ["a", "b"]).normalized_entropy == pytest.approx(1.0)

    def test_without_universe(self):
        r = retrieval_stats([["x"], ["y"], ["x"]])
        assert r.distinct == 2 and 0 < r.normalized_entropy < 1

    @given(st.lists(st.integers(0, 1000), min_size=1, max_size=50))
    def test_gini_bounds(self, counts):
        g = gini(counts)
        assert math.isnan(g) if sum(counts) == 0 else -1e-9 <= g <= 1.0


# ----------------------------------------------------------------------------- drift
class TestDrift:
    def test_identical_and_disjoint(self):
        k = {"a": "s:1", "b": "s:2"}
        sc = {"a": 1.0, "b": 1.0}
        same = drift(k, k, sc, sc)
        assert same.output_jsd == 0 and same.change_rate == 0 and same.accuracy_delta == 0
        other = drift(k, {"a": "s:3", "b": "s:3"}, sc, {"a": 0.0, "b": 0.0})
        assert other.output_jsd == pytest.approx(1.0) and other.regressions == 2

    def test_permutation_is_invisible_to_jsd_but_not_to_change_rate(self):
        base = {f"p{i}": f"s:{i % 2}" for i in range(10)}
        swapped = {k: ("s:1" if v == "s:0" else "s:0") for k, v in base.items()}
        d = drift(base, swapped, {}, {})
        assert d.output_jsd == pytest.approx(0.0, abs=1e-12)
        assert d.change_rate == 1.0

    def test_errors_are_their_own_output_class(self):
        d = drift({"a": "s:1", "b": "s:1"}, {"a": None, "b": "s:1"}, {}, {})
        assert d.change_rate == 0.5 and d.output_jsd > 0

    def test_only_common_probes_are_compared(self):
        d = drift({"a": "s:1", "gone": "s:2"}, {"a": "s:1", "new": "s:9"}, {"a": 1.0}, {"a": 0.0})
        assert d.compared == 1 and d.regressions == 1 and d.accuracy_delta == -1.0

    def test_empty_comparison_is_nan(self):
        d = drift({}, {}, {}, {})
        assert math.isnan(d.change_rate) and math.isnan(d.output_jsd) and math.isnan(d.accuracy_delta)

    @given(st.dictionaries(st.text(min_size=1, max_size=3), st.integers(1, 20), min_size=1),
           st.dictionaries(st.text(min_size=1, max_size=3), st.integers(1, 20), min_size=1))
    def test_jsd_symmetric_and_bounded(self, p, q):
        a, b = jensen_shannon(p, q), jensen_shannon(q, p)
        assert a == pytest.approx(b, abs=1e-12) and 0.0 <= a <= 1.0


# ----------------------------------------------------------------------------- confidence
class TestConfidence:
    def test_invalid_values_are_counted_not_averaged(self):
        c = confidence_stats([0.5, None, float("nan"), 1.5, -0.1, "high", 0.7], [1, 1, 1, 1, 1, 1, 0])
        assert c.reported == 2 and c.invalid == 4
        assert c.mean_confidence == pytest.approx(0.6)
        assert c.overconfidence == pytest.approx(0.6 - 0.5)

    def test_unscored_probes_contribute_confidence_but_not_calibration(self):
        c = confidence_stats([0.9, 0.9], [float("nan"), 1.0])
        assert c.reported == 2 and c.overconfidence == pytest.approx(-0.1)

    def test_all_missing(self):
        c = confidence_stats([None, None], [1.0, 0.0])
        assert math.isnan(c.mean_confidence) and math.isnan(c.ece)

    def test_alignment_is_enforced(self):
        with pytest.raises(ValueError):
            confidence_stats([0.5], [1.0, 0.0])

    def test_ece_zero_when_perfectly_calibrated(self):
        conf = np.array([1.0, 1.0, 0.0, 0.0])
        assert expected_calibration_error(conf, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
        assert expected_calibration_error(np.array([0.95] * 4), np.array([0.0] * 4)) == pytest.approx(0.95)
