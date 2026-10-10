"""Wire REGRESS into your own agent's learning loop.

The agent here is a stand-in: a tiny keyword-memory QA bot whose "consolidation" step
deduplicates memories by merging anything that shares a first word. That rule is the bug:
it looks like tidy-up, and each cycle it quietly merges distinct facts into one.

Run:  python examples/quickstart.py
"""

from __future__ import annotations

import hashlib

import numpy as np

from regress import MemorySnapshot, Probe, ProbeSuite, RegressionMonitor, Response
from regress.report import Style, render_summary, render_trends


def embed(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic bag-of-words hash embedding: enough for diversity metrics."""
    v = np.zeros(dim)
    for w in text.lower().split():
        h = int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big")
        v[h % dim] += 1.0 if h & 1 else -1.0
    return v


class KeywordAgent:
    def __init__(self, facts: dict[str, str]):
        self.memory = dict(facts)  # question -> answer
        self.merges = 0

    def respond(self, probe: Probe) -> Response:
        q = set(probe.input.lower().split())
        scored = sorted(self.memory, key=lambda k: -len(q & set(k.lower().split())))
        best = scored[:3]
        overlap = len(q & set(best[0].lower().split())) / max(len(q), 1)
        return Response(self.memory[best[0]], confidence=min(1.0, 0.4 + overlap), retrieved=best)

    def memory_snapshot(self) -> MemorySnapshot:
        keys = list(self.memory)
        return MemorySnapshot(ids=keys, embeddings=np.array([embed(k + " " + self.memory[k]) for k in keys]))

    def consolidate(self) -> None:
        """Buggy dedup: merge the first pair of memories that share an opening word."""
        keys = list(self.memory)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if a.split()[0] == b.split()[0]:
                    merged = f"{a} {b}"
                    self.memory[merged] = self.memory[a]  # keeps a's answer, loses b's
                    del self.memory[a], self.memory[b]
                    self.merges += 1
                    return


FACTS = {
    f"{w} {topic}": f"{w}-{topic}"
    for w in ("what is", "who wrote", "where is", "when was")
    for topic in ("python", "the louvre", "hamlet", "everest", "the eiffel tower", "dune")
}


def main() -> None:
    probes = [Probe(f"q{i}", k, expected=v) for i, (k, v) in enumerate(FACTS.items())]
    suite = ProbeSuite(probes, holdout_fraction=0.25, seed=0)
    agent = KeywordAgent(FACTS)
    monitor = RegressionMonitor(suite)

    for cycle in range(13):
        report = monitor.evaluate(agent, cycle)  # ← the only line your loop needs
        print(f"cycle {cycle:2d}  {report.status:<5}  acc={report.metrics['accuracy']:.2f}  "
              f"memories={int(report.metrics['memory_size'])}  {report.reason}")
        agent.consolidate()

    style = Style()
    print()
    print(render_trends(monitor.history, style))
    print()
    print(render_summary(monitor.history, style=style))


if __name__ == "__main__":
    main()
