"""Consolidation and memory-budget policies — the *inputs* REGRESS is built to judge.

These are deliberately small. They exist here because a forgetting policy is one of the
most direct ways to cause the homogenisation REGRESS measures (evict the rare memories,
keep the popular ones, and retrieval entropy falls by construction), so the simulator and
tests need real policies to compare. They are not a full sleep-time compute scheduler.

``plan_consolidation`` — expected-value rule: consolidate item *i* iff
    p_hit_i × latency_saved_i − cost_i > 0,
then, under a compute budget, take the best value-per-cost first (greedy knapsack).

``MemoryBudget`` — a hard byte cap with a pluggable forgetting policy. The cap is an
invariant: no sequence of operations can leave ``used > capacity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

__all__ = ["Candidate", "ConsolidationPlan", "plan_consolidation", "MemoryBudget", "BudgetError"]


class BudgetError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    p_hit: float  # predicted probability the item is needed again (per horizon)
    latency_saved: float  # seconds saved per hit if consolidated
    cost: float  # compute cost to consolidate, in the same units as value

    def __post_init__(self) -> None:
        for name in ("p_hit", "latency_saved", "cost"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise BudgetError(f"{self.id}: {name} must be finite, got {v!r}")
        if not 0.0 <= self.p_hit <= 1.0:
            raise BudgetError(f"{self.id}: p_hit must be in [0, 1]")
        if self.latency_saved < 0 or self.cost < 0:
            raise BudgetError(f"{self.id}: latency_saved and cost must be non-negative")

    @property
    def value(self) -> float:
        return self.p_hit * self.latency_saved - self.cost


@dataclass
class ConsolidationPlan:
    selected: list[str]
    rejected_negative: list[str]  # EV ≤ 0: never worth consolidating
    deferred: list[str]  # EV > 0 but did not fit the budget
    spent: float
    expected_value: float


def plan_consolidation(candidates: Iterable[Candidate], budget: float = math.inf) -> ConsolidationPlan:
    if math.isnan(budget) or budget < 0:
        raise BudgetError("budget must be a non-negative number")
    cands = list(candidates)
    ids = [c.id for c in cands]
    if len(set(ids)) != len(ids):
        raise BudgetError("duplicate candidate ids")
    positive = [c for c in cands if c.value > 0]
    negative = [c.id for c in cands if c.value <= 0]

    def ratio(c: Candidate) -> float:
        return math.inf if c.cost == 0 else c.value / c.cost

    # Free items first, then by value density; ties broken by absolute value, then id.
    positive.sort(key=lambda c: (-ratio(c), -c.value, c.id))
    selected, deferred, spent, ev = [], [], 0.0, 0.0
    for c in positive:
        if spent + c.cost <= budget + 1e-12:
            selected.append(c.id)
            spent += c.cost
            ev += c.value
        else:
            deferred.append(c.id)
    return ConsolidationPlan(selected, negative, deferred, spent, ev)


Policy = Literal["lru", "lfu", "value_density", "redundancy"]


@dataclass
class _Entry:
    size: int
    value: float
    last_used: int
    uses: int
    embedding: np.ndarray | None = field(default=None, repr=False)


class MemoryBudget:
    """Hard byte cap with a forgetting policy.

    Policies:
      * ``lru`` — evict least recently used.
      * ``lfu`` — evict least frequently used. Known to homogenise memory: rarely-hit
        items are exactly the diverse tail.
      * ``value_density`` — evict lowest value per byte.
      * ``redundancy`` — evict the item most similar to another kept item (needs
        embeddings). Trades a little recall for diversity; the policy REGRESS favours.
    """

    def __init__(self, capacity_bytes: int, policy: Policy = "lru") -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise BudgetError("capacity_bytes must be a positive int")
        if policy not in ("lru", "lfu", "value_density", "redundancy"):
            raise BudgetError(f"unknown policy {policy!r}")
        self.capacity = capacity_bytes
        self.policy = policy
        self._items: dict[str, _Entry] = {}
        self._clock = 0
        self.used = 0

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ids(self) -> list[str]:
        return list(self._items)

    def touch(self, item_id: str) -> None:
        e = self._items.get(item_id)
        if e is None:
            raise KeyError(item_id)
        self._clock += 1
        e.last_used = self._clock
        e.uses += 1

    def remove(self, item_id: str) -> None:
        e = self._items.pop(item_id)
        self.used -= e.size

    def put(
        self,
        item_id: str,
        size_bytes: int,
        value: float = 1.0,
        embedding: Sequence[float] | np.ndarray | None = None,
    ) -> list[str]:
        """Insert or replace an item; returns the ids evicted to make room.

        An item larger than the whole capacity is rejected *before* anything is evicted,
        so a single oversized write cannot wipe the memory.
        """
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise BudgetError("size_bytes must be a positive int")
        if size_bytes > self.capacity:
            raise BudgetError(f"{item_id}: {size_bytes} B exceeds the {self.capacity} B cap")
        if self.policy == "redundancy" and embedding is None:
            raise BudgetError("the redundancy policy needs an embedding for every item")
        if value is None or math.isnan(value):
            raise BudgetError("value must be a number")
        emb = None if embedding is None else np.asarray(embedding, dtype=float).ravel()
        if emb is not None:
            if emb.size == 0 or not np.all(np.isfinite(emb)):
                raise BudgetError("embedding must be a non-empty finite vector")
            dims = {
                e.embedding.shape
                for k, e in self._items.items()
                if k != item_id and e.embedding is not None
            }
            if dims and dims != {emb.shape}:
                raise BudgetError(f"embedding shape {emb.shape} does not match stored {dims.pop()}")
        old = self._items.pop(item_id, None)
        if old is not None:
            self.used -= old.size
        self._clock += 1
        evicted: list[str] = []
        while self.used + size_bytes > self.capacity:
            victim = self._victim(emb)
            evicted.append(victim)
            self.remove(victim)
        self._items[item_id] = _Entry(
            size_bytes, float(value), self._clock, (old.uses if old else 0) + 1, emb
        )
        self.used += size_bytes
        assert self.used <= self.capacity
        return evicted

    def _victim(self, incoming: np.ndarray | None) -> str:
        items = self._items
        if self.policy == "lru":
            return min(items, key=lambda k: (items[k].last_used, k))
        if self.policy == "lfu":
            return min(items, key=lambda k: (items[k].uses, items[k].last_used, k))
        if self.policy == "value_density":
            return min(items, key=lambda k: (items[k].value / items[k].size, items[k].last_used, k))
        # redundancy: the kept item whose nearest neighbour (incoming included) is closest.
        keys = list(items)
        embs = [items[k].embedding for k in keys]
        if incoming is not None:
            embs.append(incoming)
        m = np.array(embs, dtype=float)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.where(norms > 0, norms, 1.0)
        sims = m @ m.T
        np.fill_diagonal(sims, -np.inf)
        nearest = sims[: len(keys)].max(axis=1)
        order = sorted(range(len(keys)), key=lambda i: (-nearest[i], items[keys[i]].last_used, keys[i]))
        return keys[order[0]]
