import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from regress.budget import BudgetError, Candidate, MemoryBudget, plan_consolidation


class TestConsolidationPlan:
    def test_expected_value_rule(self):
        plan = plan_consolidation([
            Candidate("hot", p_hit=0.9, latency_saved=2.0, cost=0.5),  # +1.3
            Candidate("cold", p_hit=0.01, latency_saved=2.0, cost=0.5),  # −0.48
            Candidate("breakeven", p_hit=0.5, latency_saved=1.0, cost=0.5),  # 0 → not worth it
        ])
        assert plan.selected == ["hot"]
        assert set(plan.rejected_negative) == {"cold", "breakeven"}
        assert plan.expected_value == pytest.approx(1.3)

    def test_budget_prefers_value_density_and_takes_free_items_first(self):
        plan = plan_consolidation([
            Candidate("big", 1.0, 10.0, 5.0),  # value 5, density 1
            Candidate("small", 1.0, 3.0, 1.0),  # value 2, density 2
            Candidate("free", 0.1, 1.0, 0.0),  # value 0.1, density ∞
        ], budget=5.0)
        assert plan.selected == ["free", "small"] and plan.deferred == ["big"]
        assert plan.spent == pytest.approx(1.0)

    def test_zero_budget_still_takes_free_wins(self):
        plan = plan_consolidation([Candidate("free", 1.0, 1.0, 0.0), Candidate("x", 1, 5, 1)], budget=0.0)
        assert plan.selected == ["free"] and plan.deferred == ["x"]

    @pytest.mark.parametrize("kw", [
        dict(p_hit=1.5, latency_saved=1, cost=1),
        dict(p_hit=0.5, latency_saved=-1, cost=1),
        dict(p_hit=0.5, latency_saved=1, cost=float("nan")),
        dict(p_hit=0.5, latency_saved=float("inf"), cost=1),
    ])
    def test_rejects_invalid_candidates(self, kw):
        with pytest.raises(BudgetError):
            Candidate("x", **kw)

    def test_rejects_duplicates_and_bad_budget(self):
        c = Candidate("x", 0.5, 1, 0.1)
        with pytest.raises(BudgetError):
            plan_consolidation([c, c])
        with pytest.raises(BudgetError):
            plan_consolidation([c], budget=-1)
        with pytest.raises(BudgetError):
            plan_consolidation([c], budget=math.nan)

    def test_empty(self):
        p = plan_consolidation([])
        assert p.selected == [] and p.expected_value == 0


class TestMemoryBudget:
    def test_oversized_item_is_rejected_before_anything_is_evicted(self):
        mb = MemoryBudget(100)
        mb.put("a", 60)
        with pytest.raises(BudgetError):
            mb.put("huge", 101)
        assert mb.ids == ["a"] and mb.used == 60

    def test_replacing_an_item_never_evicts_itself(self):
        mb = MemoryBudget(100)
        mb.put("a", 50)
        mb.put("b", 50)
        evicted = mb.put("a", 70)  # grows: must evict b, not the item being written
        assert evicted == ["b"] and mb.ids == ["a"] and mb.used == 70

    def test_lru_forgets_the_stale_favourite_lfu_keeps_it(self):
        for policy, expected in (("lru", "a"), ("lfu", "b")):
            mb = MemoryBudget(30, policy)
            for k in "abc":
                mb.put(k, 10)
            for _ in range(3):
                mb.touch("a")  # a: used often, but long ago
            mb.touch("b")
            mb.touch("c")  # b and c tie on uses; b is older
            assert mb.put("d", 10) == [expected], policy

    def test_value_density(self):
        mb = MemoryBudget(100, "value_density")
        mb.put("dense", 10, value=10)
        mb.put("sparse", 80, value=10)
        assert mb.put("new", 20, value=1) == ["sparse"]

    def test_redundancy_policy_evicts_the_near_duplicate(self):
        mb = MemoryBudget(30, "redundancy")
        mb.put("x", 10, embedding=[1, 0, 0])
        mb.put("x_copy", 10, embedding=[0.99, 0.01, 0])
        mb.put("y", 10, embedding=[0, 1, 0])
        evicted = mb.put("z", 10, embedding=[0, 0, 1])
        assert evicted[0] in {"x", "x_copy"} and "y" in mb

    def test_redundancy_requires_consistent_embeddings(self):
        mb = MemoryBudget(30, "redundancy")
        with pytest.raises(BudgetError):
            mb.put("a", 10)
        mb.put("a", 10, embedding=[1, 0])
        with pytest.raises(BudgetError):
            mb.put("b", 10, embedding=[1, 0, 0])
        with pytest.raises(BudgetError):
            mb.put("c", 10, embedding=[np.nan, 0])
        mb.put("a", 10, embedding=[1, 0, 0])  # the sole item may change dimension

    @pytest.mark.parametrize("args", [(0,), (-5,), (True,), (1.5,)])
    def test_invalid_capacity(self, args):
        with pytest.raises(BudgetError):
            MemoryBudget(*args)

    def test_invalid_put(self):
        mb = MemoryBudget(10)
        for size in (0, -1, True, 2.0):
            with pytest.raises(BudgetError):
                mb.put("a", size)
        with pytest.raises(BudgetError):
            mb.put("a", 1, value=float("nan"))
        with pytest.raises(KeyError):
            mb.touch("missing")
        with pytest.raises(BudgetError):
            MemoryBudget(10, "random")

    @settings(max_examples=150, deadline=None)
    @given(
        st.sampled_from(["lru", "lfu", "value_density", "redundancy"]),
        st.integers(1, 200),
        st.lists(st.tuples(st.sampled_from("put touch remove".split()), st.integers(0, 15),
                           st.integers(1, 250), st.floats(0, 10)), max_size=80),
    )
    def test_capacity_is_an_invariant_under_any_operation_sequence(self, policy, cap, ops):
        mb = MemoryBudget(cap, policy)
        rng = np.random.default_rng(0)
        for op, key, size, value in ops:
            k = f"k{key}"
            try:
                if op == "put":
                    mb.put(k, size, value, embedding=rng.normal(size=4))
                elif op == "touch":
                    mb.touch(k)
                else:
                    mb.remove(k)
            except (BudgetError, KeyError):
                pass
            assert 0 <= mb.used <= mb.capacity
            assert mb.used == sum(mb._items[i].size for i in mb.ids)
