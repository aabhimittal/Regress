"""A toy self-evolving agent whose learning loop can go right or wrong.

The agent classifies points in an embedding space by k-nearest-neighbour vote over an
episodic memory. Each cycle it receives a batch of new experiences and *consolidates*.
Two consolidation regimes:

* ``healthy`` — store new experiences with environment feedback (true labels), evict the
  oldest when over capacity. Memory stays spread out; accuracy creeps up.
* ``collapse`` — self-reinforcing internalisation. No feedback: new experiences are
  labelled with the agent's own predictions; every memory is pulled toward the most
  retrieved memory near it ("hub") and adopts its label; hubs gain vote weight. This is a
  cartoon of internalising experience badly, and it produces the collapse signature
  without being told to: embeddings converge (diversity ↓), the same hubs answer
  everything (retrieval entropy ↓), and votes become unanimous (confidence ↑) while
  accuracy erodes.

Scenarios switch from healthy to collapse at ``onset``; ``rate`` sets how hard each bad
cycle pulls. ``leak`` additionally lets the public probes into the agent's memory, the
textbook way an agent overfits the suite that measures it.

Everything is seeded; ``Simulation.checkpoint(c)`` returns an independent copy of the agent
after cycle ``c``, which is what bisection replays.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ._canon import fingerprint
from .core import MemorySnapshot, Probe, Response
from .probes import ProbeSuite

__all__ = ["World", "ToyAgent", "Scenario", "SCENARIOS", "Simulation"]


class World:
    def __init__(self, n_classes: int = 8, dim: int = 24, noise: float = 0.24, seed: int = 0):
        rng = np.random.default_rng(seed)
        c = rng.normal(size=(n_classes, dim))
        self.centers = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.n_classes, self.dim, self.noise = n_classes, dim, noise

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, self.n_classes, size=n)
        x = self.centers[y] + rng.normal(scale=self.noise, size=(n, self.dim))
        return x, y

    def probe_suite(self, n: int = 160, holdout_fraction: float = 0.25, seed: int = 1234) -> ProbeSuite:
        rng = np.random.default_rng(seed)
        x, y = self.sample(n, rng)
        probes = [Probe(id=f"p{i:03d}", input=x[i], expected=int(y[i])) for i in range(n)]
        return ProbeSuite(probes, holdout_fraction=holdout_fraction, seed=seed)


@dataclass
class _Mem:
    emb: np.ndarray  # (n, d), unit rows
    labels: np.ndarray  # (n,)
    strength: np.ndarray  # (n,)
    ids: list[str]


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


class ToyAgent:
    def __init__(self, world: World, init_per_class: int = 30, k: int = 7, capacity: int = 240, seed: int = 0,
                 importance_weight: float = 0.08):
        rng = np.random.default_rng(seed)
        self.world, self.k, self.capacity = world, k, capacity
        self.importance_weight = importance_weight
        x, y = world.sample(init_per_class * world.n_classes, rng)
        self._next_id = 0
        self.mem = _Mem(_unit(x), y.copy(), np.ones(len(y)), [self._new_id() for _ in y])
        self.usage = np.zeros(len(y))
        self.answer_cache: dict[str, int] = {}

    def _new_id(self) -> str:
        self._next_id += 1
        return f"m{self._next_id:05d}"

    # -- inference -------------------------------------------------------------------
    def _vote(self, x: np.ndarray) -> tuple[int, float, np.ndarray]:
        sims = self.mem.emb @ _unit(np.asarray(x, dtype=float))
        # Relevance + importance, as in most agent-memory stores. With uniform strength this
        # is plain nearest-neighbour; once consolidation inflates a few memories, they start
        # winning retrieval for queries they are not actually close to.
        score = sims + self.importance_weight * np.log(self.mem.strength)
        top = np.argsort(-score)[: self.k]
        w = self.mem.strength[top] * np.clip(sims[top], 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        tally = np.bincount(self.mem.labels[top], weights=w, minlength=self.world.n_classes)
        label = int(np.argmax(tally))
        # Expressed confidence = vote agreement × familiarity. Familiarity is how strongly
        # memory "recognises" the query (relevance + importance of what was retrieved), the
        # fluency heuristic many agents' self-reports reduce to. Inflated importance makes
        # the agent feel familiar with everything.
        familiarity = 1.0 / (1.0 + np.exp(-6.0 * (float(score[top].mean()) - 0.45)))
        return label, float(tally[label] / tally.sum()) * familiarity, top

    def respond(self, probe: Probe) -> Response:
        if self.answer_cache:
            cached = self.answer_cache.get(fingerprint(probe.input))
            if cached is not None:
                return Response(output=cached, confidence=1.0, retrieved=[])
        label, conf, top = self._vote(probe.input)
        return Response(output=label, confidence=conf, retrieved=[self.mem.ids[i] for i in top])

    def without(self, memory_ids) -> "ToyAgent":
        """A copy with the given memories removed (enables ablation)."""
        drop = {str(i) for i in memory_ids}
        clone = copy.deepcopy(self)
        keep = np.array([i not in drop for i in clone.mem.ids], dtype=bool)
        m = clone.mem
        m.emb, m.labels, m.strength = m.emb[keep], m.labels[keep], m.strength[keep]
        m.ids = [i for i, k in zip(m.ids, keep) if k]
        clone.usage = clone.usage[keep]
        return clone

    def state_fingerprint(self) -> str:
        """Changes whenever anything that can change an answer changes (enables caching)."""
        import hashlib

        h = hashlib.sha256()
        for arr in (self.mem.emb, self.mem.labels, self.mem.strength):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update("|".join(self.mem.ids).encode())
        h.update(repr(sorted(self.answer_cache.items())).encode())
        h.update(f"{self.k}:{self.importance_weight}".encode())
        return h.hexdigest()

    def memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(ids=list(self.mem.ids), embeddings=self.mem.emb.copy())

    # -- learning --------------------------------------------------------------------
    def _append(self, x: np.ndarray, y: np.ndarray, strength: float = 1.0) -> None:
        m = self.mem
        m.emb = np.vstack([m.emb, _unit(x)])
        m.labels = np.concatenate([m.labels, y.astype(int)])
        m.strength = np.concatenate([m.strength, np.full(len(y), strength)])
        m.ids = m.ids + [self._new_id() for _ in y]
        self.usage = np.concatenate([self.usage, np.zeros(len(y))])
        if len(m.ids) > self.capacity:  # FIFO eviction
            drop = len(m.ids) - self.capacity
            m.emb, m.labels, m.strength = m.emb[drop:], m.labels[drop:], m.strength[drop:]
            m.ids, self.usage = m.ids[drop:], self.usage[drop:]

    def consolidate_healthy(self, x: np.ndarray, y: np.ndarray) -> None:
        for xi in x:  # track usage so the two regimes see the same statistics
            _, _, top = self._vote(xi)
            self.usage[top] += 1
        self._append(x, y)

    def consolidate_collapse(self, x: np.ndarray, rate: float, merge_threshold: float = 0.9) -> None:
        preds = []
        for xi in x:
            label, _, top = self._vote(xi)
            self.usage[top] += 1
            preds.append(label)
        m = self.mem
        sims = m.emb @ m.emb.T
        # Each memory's hub: the most-used memory among its 15 nearest neighbours.
        nbrs = np.argsort(-sims, axis=1)[:, :15]
        hubs = nbrs[np.arange(len(nbrs)), np.argmax(self.usage[nbrs], axis=1)]
        # A global pull toward the most-used memory overall homogenises across clusters.
        top_hub = int(np.argmax(self.usage))
        g = min(1.0, 1.5 * rate)
        target = _unit((1 - g) * m.emb[hubs] + g * m.emb[top_hub])
        m.emb = _unit((1 - rate) * m.emb + rate * target)
        rng = np.random.default_rng(len(m.ids) + int(self.usage.sum()))
        relabel = rng.random(len(m.ids)) < rate * 1.5
        m.labels = np.where(relabel, m.labels[hubs], m.labels)
        m.strength = m.strength * (1 + 2 * rate * (self.usage / max(self.usage.max(), 1)))
        self._merge_into_hubs(hubs, threshold=merge_threshold)
        self._append(x, np.array(preds), strength=1.0)

    def _merge_into_hubs(self, hubs: np.ndarray, threshold: float) -> None:
        """Fold memories that have become near-copies of their hub into the hub."""
        m = self.mem
        sim_to_hub = np.einsum("ij,ij->i", m.emb, m.emb[hubs])
        absorbed = (sim_to_hub >= threshold) & (hubs != np.arange(len(hubs)))
        absorbed &= ~absorbed[hubs]  # a hub that is itself being absorbed keeps its members
        if not absorbed.any():
            return
        np.add.at(m.strength, hubs[absorbed], m.strength[absorbed])
        np.add.at(self.usage, hubs[absorbed], self.usage[absorbed])
        keep = ~absorbed
        m.emb, m.labels, m.strength = m.emb[keep], m.labels[keep], m.strength[keep]
        m.ids = [i for i, k in zip(m.ids, keep) if k]
        self.usage = self.usage[keep]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    onset: int | None  # first cycle whose consolidation goes bad; None = never
    rate: float = 0.0
    leak: bool = False


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("healthy", "Feedback-driven learning; no collapse. Should stay quiet.", None),
        Scenario("collapse", "Healthy until cycle 8, then self-reinforcing internalisation.", 8, 0.25),
        Scenario("slow", "Gentle collapse from cycle 5; thresholds trip late, onset estimate should not.", 5, 0.15),
        Scenario("leak", "From cycle 6 the public probes leak into an answer cache while learning degrades.", 6, 0.2, leak=True),
        Scenario("creep", "Very gentle collapse from cycle 10: below every per-cycle tolerance for a long time.", 10, 0.08),
    )
}


@dataclass
class Simulation:
    scenario: Scenario
    cycles: int = 20
    seed: int = 0
    batch: int = 40
    world: World = field(init=False)
    suite: ProbeSuite = field(init=False)
    _checkpoints: list[ToyAgent] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.cycles < 1:
            raise ValueError("cycles must be >= 1")
        self.world = World(seed=self.seed)
        self.suite = self.world.probe_suite(seed=1234 + self.seed)
        agent = ToyAgent(self.world, seed=self.seed)
        self._checkpoints = [copy.deepcopy(agent)]
        for c in range(1, self.cycles + 1):
            self._step(agent, c)
            self._checkpoints.append(copy.deepcopy(agent))

    def _step(self, agent: ToyAgent, cycle: int) -> None:
        rng = np.random.default_rng([self.seed, cycle])
        x, y = self.world.sample(self.batch, rng)
        sc = self.scenario
        bad = sc.onset is not None and cycle >= sc.onset
        if not bad:
            agent.consolidate_healthy(x, y)
            return
        agent.consolidate_collapse(x, sc.rate)
        if sc.leak:
            # The public probes and their answers leak into an exact-match answer cache:
            # the agent now aces the questions it has seen, whatever its memory says.
            for p in self.suite.public:
                agent.answer_cache[fingerprint(p.input)] = int(p.expected)

    def checkpoint(self, cycle: int) -> ToyAgent:
        if not 0 <= cycle <= self.cycles:
            raise IndexError(f"no checkpoint for cycle {cycle}")
        return copy.deepcopy(self._checkpoints[cycle])
