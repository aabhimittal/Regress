import numpy as np
import pytest

from regress import Probe, ProbeSuite, ProbeSuiteTampered
from regress._canon import fingerprint, output_key


class TestFingerprint:
    def test_large_array_edit_is_detected_even_where_repr_elides(self):
        a = np.zeros(10_000)
        b = a.copy()
        b[5_000] = 1e-9
        assert repr(a) == repr(b)  # the trap: repr hides the middle of big arrays
        assert fingerprint(a) != fingerprint(b)

    def test_dtype_and_shape_matter(self):
        assert fingerprint(np.zeros(4, dtype=np.float32)) != fingerprint(np.zeros(4, dtype=np.float64))
        assert fingerprint(np.zeros((2, 2))) != fingerprint(np.zeros(4))

    def test_non_contiguous_view_matches_its_copy(self):
        a = np.arange(20.0).reshape(4, 5)[:, ::2]
        assert fingerprint(a) == fingerprint(a.copy())

    def test_dict_order_and_set_order_do_not_matter(self):
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
        assert fingerprint({3, 1, 2}) == fingerprint({2, 3, 1})

    def test_list_tuple_bool_int_are_distinct(self):
        assert fingerprint([1, 2]) != fingerprint((1, 2))
        assert fingerprint(True) != fingerprint(1)
        assert fingerprint(1) != fingerprint(1.0)

    def test_special_floats_are_stable(self):
        assert fingerprint(float("nan")) == fingerprint(float("nan"))
        assert fingerprint(0.0) == fingerprint(-0.0)
        assert fingerprint(float("inf")) != fingerprint(float("-inf"))

    def test_output_key(self):
        assert output_key("yes") == "s:yes"
        assert output_key(True) != output_key(1)
        assert output_key(np.int64(3)) == output_key(3)
        long = "x" * 1000
        assert output_key(long).startswith("h:") and output_key(long) != output_key(long + "y")


class TestProbeSuite:
    def test_rejects_empty_duplicates_and_bad_fraction(self):
        with pytest.raises(ValueError):
            ProbeSuite([])
        with pytest.raises(ValueError, match="duplicate"):
            ProbeSuite([Probe("a", 1), Probe("a", 2)])
        with pytest.raises(ValueError):
            ProbeSuite([Probe("a", 1)], holdout_fraction=1.0)
        with pytest.raises(ValueError):
            Probe("", 1)

    def test_fingerprint_is_order_insensitive_but_content_sensitive(self):
        ps = [Probe(f"p{i}", i, i) for i in range(10)]
        assert ProbeSuite(ps).fingerprint == ProbeSuite(reversed(ps)).fingerprint
        changed = ps[:-1] + [Probe("p9", 9, expected=0)]
        assert ProbeSuite(ps).fingerprint != ProbeSuite(changed).fingerprint

    def test_holdout_settings_are_part_of_the_fingerprint(self):
        ps = [Probe(f"p{i}", i) for i in range(10)]
        assert ProbeSuite(ps, 0.2, seed=1).fingerprint != ProbeSuite(ps, 0.2, seed=2).fingerprint

    def test_in_place_mutation_of_a_probe_input_is_caught(self):
        x = np.zeros(50)
        s = ProbeSuite([Probe("a", x, 0)])
        s.verify()
        x[10] = 1.0
        with pytest.raises(ProbeSuiteTampered):
            s.verify()

    def test_holdout_split_is_stable_when_probes_are_added(self):
        base = [Probe(f"p{i}", i) for i in range(200)]
        more = base + [Probe(f"q{i}", i) for i in range(100)]
        a, b = ProbeSuite(base, 0.3, seed=7), ProbeSuite(more, 0.3, seed=7)
        assert {p.id for p in a.holdout} <= {p.id for p in b.holdout}
        assert 0.2 < len(a.holdout) / len(a) < 0.4

    def test_tiny_suite_never_ends_up_all_holdout(self):
        for seed in range(50):
            s = ProbeSuite([Probe("only", 1)], holdout_fraction=0.99, seed=seed)
            assert len(s.public) == 1
