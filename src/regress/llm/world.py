"""A fictional fact world, so the model cannot answer from pre-training.

People with invented names have three attributes each (home city, profession, pet). Values
come from small vocabularies, so many people share a value: exactly the setting where a
summary like "several people here are glassblowers" destroys who-is-what.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..core import Probe
from ..probes import ProbeSuite

__all__ = ["Fact", "FactWorld", "answer_score"]

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kl", "st", "th", "vr"]
_VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ou"]
_CODAS = ["", "n", "r", "s", "l", "th", "x", "m"]

CITIES = ["Ostrel", "Halcyon", "Varn", "Quillon", "Myrrh", "Tessaly", "Corvane", "Ebbridge"]
PROFESSIONS = ["glassblower", "cartographer", "beekeeper", "luthier", "astronomer", "tanner", "archivist", "ferryman"]
PETS = ["ferret", "heron", "tortoise", "lynx", "gecko", "raven", "otter", "mule"]
ATTRIBUTES = {"home city": CITIES, "profession": PROFESSIONS, "pet": PETS}


@dataclass(frozen=True)
class Fact:
    person: str
    attribute: str
    value: str

    @property
    def text(self) -> str:
        return f"{self.person}'s {self.attribute} is {self.value}."

    @property
    def question(self) -> str:
        return f"What is {self.person}'s {self.attribute}?"


def _name(rng: random.Random) -> str:
    def part() -> str:
        return (rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)).capitalize()

    return f"{part()} {part()}"


class FactWorld:
    def __init__(self, n_people: int = 24, seed: int = 0) -> None:
        if n_people < 1:
            raise ValueError("n_people must be >= 1")
        self.seed = seed
        self._rng = random.Random(seed)
        self.people: list[str] = []
        self.facts: list[Fact] = []
        self.add_people(n_people)

    def add_people(self, n: int) -> list[Fact]:
        new: list[Fact] = []
        used = set(self.people)
        while n > 0:
            name = _name(self._rng)
            if name in used:
                continue
            used.add(name)
            self.people.append(name)
            for attr, vocab in ATTRIBUTES.items():
                new.append(Fact(name, attr, self._rng.choice(vocab)))
            n -= 1
        self.facts.extend(new)
        return new

    def probe_suite(self, facts: list[Fact] | None = None, holdout_fraction: float = 0.25) -> ProbeSuite:
        facts = facts if facts is not None else list(self.facts)
        probes = [Probe(f"q{i:03d}", f.question, expected=f.value, tags=(f.attribute,)) for i, f in enumerate(facts)]
        return ProbeSuite(probes, holdout_fraction=holdout_fraction, seed=self.seed)


def _norm(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def answer_score(probe: Probe, output: object) -> float:
    """1 if every word of the expected value appears in a short answer, else 0.

    Long answers are not accepted as correct by containing the value somewhere: an answer
    that lists every city in the world "contains" the right one.
    """
    if probe.expected is None:
        return float("nan")
    got = _norm(output)
    want = _norm(probe.expected)
    if not want:
        return float("nan")
    if len(got) > 8:
        return 0.0
    return 1.0 if all(w in got for w in want) else 0.0
