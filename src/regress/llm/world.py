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

__all__ = ["Fact", "Claim", "FactWorld", "MultiHopWorld", "answer_score", "make_world", "TASKS"]

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


@dataclass(frozen=True)
class Claim:
    """A note the agent can read and/or a question it can be asked (multi-hop worlds)."""

    text: str
    question: str
    value: str
    attribute: str


class FactWorld:
    """Single-hop task: every question is answered by exactly one note."""

    task = "facts"

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

    # The interface experiments use (shared with MultiHopWorld).
    def initial_notes(self) -> list[str]:
        return [f.text for f in self.facts]

    def initial_questions(self) -> list[Fact]:
        return list(self.facts)

    def grow(self, n: int) -> list[Fact]:
        return self.add_people(n)

    def all_questions(self) -> list[Fact]:
        return list(self.facts)


COMPANY_PARTS = ["Brass", "Lantern", "Cobalt", "Willow", "Granite", "Ember", "Saffron", "Tidewater",
                 "Juniper", "Marrow", "Quartz", "Harbor"]
COMPANY_KINDS = ["Works", "Guild", "Foundry", "Company", "Collective", "Mills"]


class MultiHopWorld:
    """Two-hop task: "X works at C" and "C is based in Z" live in separate notes, and the
    hardest questions ("In which city does X work?") need both. A summary that keeps each
    fact but separates the two halves breaks the chain; that is a different way for
    consolidation to fail than simply dropping facts.
    """

    task = "multihop"

    def __init__(self, n_people: int = 16, n_companies: int = 8, seed: int = 0) -> None:
        if n_people < 1 or n_companies < 1:
            raise ValueError("need n_people >= 1 and n_companies >= 1")
        self.seed = seed
        self._rng = random.Random(seed + 101)
        names = [f"{a} {b}" for a in COMPANY_PARTS for b in COMPANY_KINDS]
        self._rng.shuffle(names)
        self.companies = {c: self._rng.choice(CITIES) for c in names[:n_companies]}
        self.people: list[str] = []
        self.employer: dict[str, str] = {}
        self._new_people(n_people)
        self._initial_people = list(self.people)

    def _new_people(self, n: int) -> list[str]:
        used, out = set(self.people), []
        while n > 0:
            name = _name(self._rng)
            if name in used:
                continue
            used.add(name)
            self.people.append(name)
            self.employer[name] = self._rng.choice(sorted(self.companies))
            out.append(name)
            n -= 1
        return out

    def _person_notes(self, people: list[str]) -> list[Claim]:
        return [Claim(f"{p} works at {self.employer[p]}.", f"Which company does {p} work at?",
                      self.employer[p], "employer") for p in people]

    def _company_notes(self) -> list[Claim]:
        return [Claim(f"{c} is based in {z}.", f"In which city is {c} based?", z, "company city")
                for c, z in sorted(self.companies.items())]

    def initial_notes(self) -> list[str]:
        return [c.text for c in self._company_notes() + self._person_notes(self._initial_people)]

    def _questions(self, people: list[str]) -> list[Claim]:
        qs = []
        for p in people:
            qs.append(Claim("", f"Which company does {p} work at?", self.employer[p], "employer"))
            qs.append(Claim("", f"In which city does {p} work?", self.companies[self.employer[p]], "work city"))
        return qs

    def initial_questions(self) -> list[Claim]:
        return self._questions(self._initial_people)

    def grow(self, n: int) -> list[Claim]:
        return self._person_notes(self._new_people(n))

    def all_questions(self) -> list[Claim]:
        return self._questions(self.people)

    def probe_suite(self, questions: list[Claim] | None = None, holdout_fraction: float = 0.25) -> ProbeSuite:
        qs = questions if questions is not None else self.initial_questions()
        probes = [Probe(f"m{i:03d}", q.question, expected=q.value, tags=(q.attribute,)) for i, q in enumerate(qs)]
        return ProbeSuite(probes, holdout_fraction=holdout_fraction, seed=self.seed)


TASKS = ("facts", "multihop")


def make_world(task: str, n_people: int, seed: int):
    if task == "facts":
        return FactWorld(n_people=n_people, seed=seed)
    if task == "multihop":
        return MultiHopWorld(n_people=n_people, seed=seed)
    raise ValueError(f"task must be one of {TASKS}")


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
