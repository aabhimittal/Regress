"""Memory diversity.

``mean_cosine_distance`` is the metric named in the design, but it has a blind spot: a
memory that collapses into two antipodal clusters has a *higher* mean pairwise distance
than a healthy one, because every cross-cluster pair is maximally far apart. So we also
report:

* ``effective_rank`` — exp(entropy of the normalised singular-value spectrum). It counts
  how many directions the memory actually spans, and falls under any form of collapse,
  including the antipodal one.
* ``near_duplicate_fraction`` — share of memories whose nearest neighbour is almost
  identical. This is the most literal reading of "homogenised".

The mean pairwise cosine similarity is computed exactly in O(n·d) with the identity
sum_{i≠j} u_i·u_j = ‖Σu_i‖² − n, so there is no pair sampling and no n² memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DiversityStats",
    "normalize_rows",
    "mean_cosine_distance",
    "effective_rank",
    "near_duplicate_fraction",
    "diversity",
]

_EPS = 1e-12


def normalize_rows(emb: np.ndarray) -> tuple[np.ndarray, int]:
    """Unit-normalise rows; drop zero and non-finite rows. Returns (rows, n_dropped)."""
    emb = np.asarray(emb, dtype=float)
    if emb.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {emb.shape}")
    if emb.shape[0] == 0:
        return emb.reshape(0, emb.shape[1] if emb.ndim == 2 else 0), 0
    finite = np.all(np.isfinite(emb), axis=1)
    norms = np.linalg.norm(np.where(finite[:, None], emb, 0.0), axis=1)
    keep = finite & (norms > _EPS)
    out = emb[keep] / norms[keep][:, None]
    return out, int((~keep).sum())


def mean_cosine_distance(unit: np.ndarray) -> float:
    n = unit.shape[0]
    if n < 2:
        return float("nan")
    s = unit.sum(axis=0)
    mean_sim = (float(s @ s) - n) / (n * (n - 1))
    return float(np.clip(1.0 - mean_sim, 0.0, 2.0))


def effective_rank(unit: np.ndarray) -> float:
    """Roy & Vetterli effective rank of the (uncentred) unit-norm memory matrix."""
    n = unit.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0
    sv = np.linalg.svd(unit, compute_uv=False)
    p = sv**2
    total = p.sum()
    if total <= _EPS:
        return float("nan")
    p = p / total
    p = p[p > _EPS]
    return float(np.exp(-np.sum(p * np.log(p))))


def near_duplicate_fraction(
    unit: np.ndarray, threshold: float = 0.98, max_items: int = 4096, seed: int = 0
) -> float:
    n = unit.shape[0]
    if n < 2:
        return float("nan")
    if n > max_items:
        idx = np.random.default_rng(seed).choice(n, size=max_items, replace=False)
        unit = unit[idx]
        n = max_items
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    return float(np.mean(sims.max(axis=1) >= threshold))


@dataclass
class DiversityStats:
    n: int
    dropped: int  # zero / non-finite embeddings excluded from the stats
    mean_cosine_distance: float
    effective_rank: float
    effective_rank_ratio: float  # effective_rank / min(n, d), in (0, 1]
    near_duplicate_fraction: float


def diversity(embeddings: np.ndarray, near_dup_threshold: float = 0.98) -> DiversityStats:
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1 and emb.size == 0:
        emb = emb.reshape(0, 0)
    unit, dropped = normalize_rows(emb)
    n = unit.shape[0]
    erank = effective_rank(unit)
    cap = min(n, unit.shape[1]) if n else 0
    ratio = erank / cap if cap and not np.isnan(erank) else float("nan")
    return DiversityStats(
        n=n,
        dropped=dropped,
        mean_cosine_distance=mean_cosine_distance(unit),
        effective_rank=erank,
        effective_rank_ratio=float(ratio),
        near_duplicate_fraction=near_duplicate_fraction(unit, near_dup_threshold),
    )
