"""A note-memory question-answering agent driven by a real language model.

Memory is a list of short text notes. To answer, the agent retrieves the ``k`` notes most
similar to the question and asks the model to answer from them. After each cycle it
*consolidates* its memory under one of two regimes:

* ``faithful`` — new observations are stored verbatim (exact duplicates dropped). This is the
  control: memory grows, nothing is rewritten.
* ``recursive_summary`` — the internalisation pattern that compounds badly. Every cycle the
  agent (a) answers a few questions on its own and writes its answers back as notes
  (self-training, no feedback), then (b) clusters older notes and has the model compress
  each cluster into one short note. Next cycle the summaries are themselves summarised.
  Nothing tells it to collapse; whether it does is the experiment.

The agent implements the optional REGRESS capabilities ``without()`` and
``state_fingerprint()``, so ablation and the response cache both work on it.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np

from ..adapters import HashingEmbedder
from ..core import MemorySnapshot, Probe, Response
from .backends import Backend
from .world import Fact

__all__ = ["NoteAgent", "REGIMES"]

REGIMES = ("faithful", "recursive_summary")

ANSWER_SYSTEM = (
    "You answer questions using only the notes provided. Reply with just the answer, a few words, "
    "no explanation. If the notes do not contain the answer, reply: unknown"
)
COMPRESS_SYSTEM = "Compress these notes into one note of at most {words} words. Reply with the note only."


class NoteAgent:
    def __init__(
        self,
        backend: Backend,
        regime: str = "faithful",
        k: int = 4,
        group: int = 3,
        summary_words: int = 18,
        self_train: int = 6,
        embedder: HashingEmbedder | None = None,
        seed: int = 0,
        compress_fraction: float = 1.0,
    ) -> None:
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if k < 1 or group < 2 or summary_words < 3 or self_train < 0:
            raise ValueError("need k >= 1, group >= 2, summary_words >= 3, self_train >= 0")
        if not 0 < compress_fraction <= 1:
            raise ValueError("compress_fraction must be in (0, 1]")
        self.compress_fraction = compress_fraction
        self.backend = backend
        self.regime = regime
        self.k = k
        self.group = group
        self.summary_words = summary_words
        self.self_train = self_train
        # The agent's own retrieval embedder. REGRESS measures diversity with its *own*
        # reference ruler, so this can change without fooling the monitor.
        self.embedder = embedder or HashingEmbedder(dim=512)
        self.seed = seed
        self.notes: "OrderedDict[str, str]" = OrderedDict()
        self.created: dict[str, int] = {}
        self._next = 0
        self._vec_cache: dict[str, np.ndarray] = {}

    # -- memory --------------------------------------------------------------------
    def _new_id(self) -> str:
        self._next += 1
        return f"n{self._next:05d}"

    def add_notes(self, texts: Iterable[str], cycle: int = 0) -> list[str]:
        existing = set(self.notes.values())
        ids = []
        for t in texts:
            t = " ".join(str(t).split())
            if not t or t in existing:
                continue
            nid = self._new_id()
            self.notes[nid] = t
            self.created[nid] = cycle
            existing.add(t)
            ids.append(nid)
        return ids

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._vec_cache]
        if missing:
            for t, v in zip(missing, self.embedder.embed(missing)):
                self._vec_cache[t] = v
        if not texts:
            return np.zeros((0, self.embedder.dim))
        return np.vstack([self._vec_cache[t] for t in texts])

    def retrieve(self, query: str) -> list[str]:
        if not self.notes:
            return []
        ids = list(self.notes)
        mat = self._vectors([self.notes[i] for i in ids])
        q = self.embedder.embed([query])[0]
        sims = mat @ q
        order = sorted(range(len(ids)), key=lambda j: (-sims[j], ids[j]))[: self.k]
        return [ids[j] for j in order]

    # -- REGRESS protocol -------------------------------------------------------------
    def respond(self, probe: Probe) -> Response:
        question = str(probe.input)
        ids = self.retrieve(question)
        notes = "\n".join(f"- {self.notes[i]}" for i in ids) or "- (no notes)"
        out = self.backend.complete(ANSWER_SYSTEM, f"Notes:\n{notes}\n\nQuestion: {question}", 12)
        answer = out.text.strip().splitlines()[0] if out.text.strip() else ""
        return Response(output=answer, confidence=out.confidence, retrieved=ids)

    def memory_snapshot(self) -> MemorySnapshot:
        ids = list(self.notes)
        texts = [self.notes[i] for i in ids]
        return MemorySnapshot(ids=ids, embeddings=self._vectors(texts), texts=texts)

    def without(self, memory_ids: Iterable[str]) -> "NoteAgent":
        clone = self.clone()
        for i in {str(x) for x in memory_ids}:
            clone.notes.pop(i, None)
            clone.created.pop(i, None)
        return clone

    def state_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.backend.name}|{self.k}|{self.embedder.dim}".encode())
        for i, t in self.notes.items():
            h.update(f"{i}\x00{t}\x01".encode())
        return h.hexdigest()

    def clone(self) -> "NoteAgent":
        """Independent memory, shared backend (the model is not copied)."""
        backend, cache = self.backend, self._vec_cache
        self.backend, self._vec_cache = None, {}  # type: ignore[assignment]
        try:
            c = copy.deepcopy(self)
        finally:
            self.backend, self._vec_cache = backend, cache
        c.backend, c._vec_cache = backend, cache  # vectors depend only on text: safe to share
        return c

    # -- learning ---------------------------------------------------------------------
    def learn(self, cycle: int, observations: Sequence[Fact], questions: Sequence[Fact] = ()) -> dict:
        """One learning cycle. Returns a small log of what consolidation did."""
        log = {"cycle": cycle, "added": 0, "self_notes": 0, "summaries": 0, "removed": 0}
        log["added"] = len(self.add_notes([f.text for f in observations], cycle))
        if self.regime == "faithful":
            return log
        rng = random.Random(self.seed * 7919 + cycle)
        # (a) self-training: answer questions without feedback, keep the answers as memory.
        if questions and self.self_train:
            for f in rng.sample(list(questions), min(self.self_train, len(questions))):
                ans = self.respond(Probe(f"self{cycle}", f.question)).output
                log["self_notes"] += len(self.add_notes([f"{f.question} {ans}"], cycle))
        # (b) recursive compression of the oldest notes (all of them by default; a fraction
        # makes the damage gradual, which is where early warning would matter).
        old = [i for i in self.notes if self.created[i] < cycle]
        old = old[: max(2, int(round(len(old) * self.compress_fraction)))] if old else old
        while len(old) >= 2:
            seed_id = old.pop(0)
            if len(old) == 0:
                break
            vecs = self._vectors([self.notes[i] for i in old])
            sims = vecs @ self._vectors([self.notes[seed_id]])[0]
            nearest = [old[j] for j in np.argsort(-sims)[: self.group - 1]]
            members = [seed_id] + nearest
            old = [i for i in old if i not in nearest]
            text = "\n".join(f"- {self.notes[i]}" for i in members)
            out = self.backend.complete(COMPRESS_SYSTEM.format(words=self.summary_words), text,
                                        int(self.summary_words * 2))
            summary = " ".join(out.text.split()[: self.summary_words]) or self.notes[seed_id]
            for i in members:
                self.notes.pop(i, None)
                self.created.pop(i, None)
            log["removed"] += len(members)
            log["summaries"] += len(self.add_notes([summary], cycle))
        return log
