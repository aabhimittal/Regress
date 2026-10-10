"""Retrieval concentration.

If the same few memories answer every probe, the memory has homogenised in use even if
the embeddings still look spread out (a large memory can be mostly dead weight).

Normalisation matters. Entropy is divided by log(min(N, R)) where N is the memory size and
R the number of retrieval events: with 10 retrievals against 10 000 memories the most
even spread possible is 10 distinct items, and dividing by log(10 000) would report that
perfectly healthy pattern as "collapsed".

Retrievals of ids that are not in the current memory ("phantoms") are reported separately
and excluded from the distribution: they usually mean the agent's retrieval log and its
store disagree, which is a bug worth surfacing rather than averaging away.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["RetrievalStats", "retrieval_stats", "gini"]


def gini(counts: Sequence[float]) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


@dataclass
class RetrievalStats:
    events: int  # retrieval events that hit a live memory
    phantom_rate: float  # share of events naming ids absent from memory
    distinct: int
    entropy_bits: float
    normalized_entropy: float  # in [0, 1]; NaN when undefined (≤ 1 possible outcome)
    top1_share: float
    coverage: float  # distinct / memory size
    gini: float  # over *all* memories, zero-count ones included


def retrieval_stats(
    retrieved: Iterable[Sequence[str]], memory_ids: Iterable[str] | None = None
) -> RetrievalStats:
    # An agent that lists the same memory twice for one query consulted it once.
    lists = [list(dict.fromkeys(str(i) for i in r)) for r in retrieved]
    all_events = [str(i) for r in lists for i in r]
    universe = set(str(i) for i in memory_ids) if memory_ids is not None else None
    if universe is not None:
        live = [i for i in all_events if i in universe]
    else:
        live = all_events
    phantom = len(all_events) - len(live)
    phantom_rate = phantom / len(all_events) if all_events else float("nan")
    counts = Counter(live)
    events = len(live)
    if events == 0:
        nan = float("nan")
        return RetrievalStats(0, phantom_rate, 0, nan, nan, nan, 0.0 if universe else nan, nan)

    p = np.array(list(counts.values()), dtype=float) / events
    h = float(-np.sum(p * np.log2(p)))
    n_mem = len(universe) if universe is not None else len(counts)
    max_outcomes = min(n_mem, events)
    norm = h / math.log2(max_outcomes) if max_outcomes > 1 else float("nan")
    full = list(counts.values())
    if universe is not None:
        full += [0] * (len(universe) - len(counts))
    return RetrievalStats(
        events=events,
        phantom_rate=phantom_rate,
        distinct=len(counts),
        entropy_bits=h,
        normalized_entropy=float(min(1.0, norm)) if not math.isnan(norm) else norm,
        top1_share=float(p.max()),
        coverage=len(counts) / n_mem if n_mem else float("nan"),
        gini=gini(full),
    )
