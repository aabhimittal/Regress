import numpy as np
import pytest

from regress import MemorySnapshot, Probe, ProbeSuite, Response


def make_suite(n=20, holdout=0.0, seed=0):
    return ProbeSuite([Probe(f"p{i:02d}", i, expected=i % 4) for i in range(n)], holdout, seed)


class ScriptedAgent:
    """Answers and memory are plain attributes, so a test can mutate them between cycles."""

    def __init__(self, n_mem=40, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        self.answer = lambda p: p.expected
        self.confidence = 0.6
        self.emb = rng.normal(size=(n_mem, dim))
        self.ids = [f"m{i}" for i in range(n_mem)]
        self.retrieve = lambda p: [self.ids[(p.input * 3 + k) % len(self.ids)] for k in range(3)]

    def respond(self, probe):
        return Response(self.answer(probe), self.confidence, self.retrieve(probe))

    def memory_snapshot(self):
        return MemorySnapshot(self.ids, self.emb)


@pytest.fixture
def suite():
    return make_suite()


@pytest.fixture
def agent():
    return ScriptedAgent()
