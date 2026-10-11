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
import re
from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np

from ..adapters import HashingEmbedder
from ..core import MemorySnapshot, Probe, Response
from .backends import Backend

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
        compress_chunks: int = 1,
        hops: int = 1,
    ) -> None:
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}")
        if k < 1 or group < 2 or summary_words < 3 or self_train < 0:
            raise ValueError("need k >= 1, group >= 2, summary_words >= 3, self_train >= 0")
        if not 0 < compress_fraction <= 1:
            raise ValueError("compress_fraction must be in (0, 1]")
        self.compress_fraction = compress_fraction
        if compress_chunks < 1:
            raise ValueError("compress_chunks must be >= 1")
        self.compress_chunks = compress_chunks
        if hops not in (1, 2):
            raise ValueError("hops must be 1 or 2")
        self.hops = hops
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
        first = [ids[j] for j in order]
        if self.hops == 1:
            return first
        # Second hop: query with the entity the first note introduced (words it has that the
        # question lacks, capitalised ones preferred), re-ranked MMR-style so notes that merely
        # resemble the first note ("Y also works at C") lose to notes that extend it ("C is
        # based in Z"). Up to k more notes, duplicates removed.
        asked = {w.lower() for w in re.findall(r"[A-Za-z']+", query)}
        novel = [w for w in re.findall(r"[A-Za-z']+", self.notes[first[0]]) if w.lower() not in asked]
        caps = [w for w in novel if w[:1].isupper()]
        q2 = self.embedder.embed([" ".join(caps or novel) or self.notes[first[0]]])[0]
        anchor = mat[ids.index(first[0])]
        sims2 = mat @ q2 - 0.7 * (mat @ anchor)
        order2 = sorted(range(len(ids)), key=lambda j: (-sims2[j], ids[j]))
        extra = [ids[j] for j in order2 if ids[j] not in first][: self.k]
        return first + extra

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
    def learn(self, cycle: int, observations: Sequence, questions: Sequence = ()) -> dict:
        """One learning cycle (all stages, unguarded). Returns a log of what consolidation did."""
        log = {"cycle": cycle, "added": 0, "self_notes": 0, "summaries": 0, "removed": 0}
        for _, apply in self.learn_stages(cycle, observations, questions):
            for k, v in apply(self).items():
                log[k] = log.get(k, 0) + v
        return log

    def learn_stages(self, cycle: int, observations: Sequence, questions: Sequence = ()) -> list:
        """The cycle as separable stages ``[(name, apply(agent) -> log)]``.

        A :class:`~regress.guard.ConsolidationGuard` applies each stage to a copy, tests it,
        and keeps only the stages that pass. The compression stage is split into
        ``compress_chunks`` pieces so one bad batch of summaries costs one piece, not the cycle.
        """
        texts = [f.text for f in observations]
        stages = [("observe", lambda a: {"added": len(a.add_notes(texts, cycle))})]
        if self.regime == "faithful":
            return stages
        qs = list(questions)

        def self_train(a: "NoteAgent") -> dict:
            if not qs or not a.self_train:
                return {"self_notes": 0}
            rng = random.Random(a.seed * 7919 + cycle)
            n = 0
            for f in rng.sample(qs, min(a.self_train, len(qs))):
                ans = a.respond(Probe(f"self{cycle}", f.question)).output
                n += len(a.add_notes([f"{f.question} {ans}"], cycle))
            return {"self_notes": n}

        stages.append(("self_train", self_train))
        chunks = max(1, self.compress_chunks)
        for k in range(chunks):
            stages.append((f"compress_{k + 1}/{chunks}" if chunks > 1 else "compress",
                           lambda a, k=k, chunks=chunks: a._compress(cycle, k, chunks)))
        return stages

    def _compress(self, cycle: int, chunk: int = 0, chunks: int = 1) -> dict:
        """Compress the oldest notes (``compress_fraction`` of them), slice ``chunk`` of ``chunks``."""
        log = {"summaries": 0, "removed": 0}
        old = [i for i in self.notes if self.created[i] < cycle]
        old = old[: max(2, int(round(len(old) * self.compress_fraction)))] if old else old
        if chunks > 1:
            size = -(-len(old) // chunks)
            old = old[chunk * size:(chunk + 1) * size]
        while len(old) >= 2:
            seed_id = old.pop(0)
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
