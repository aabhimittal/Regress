"""Core data types and the agent protocol REGRESS measures against."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from ._canon import fingerprint

__all__ = [
    "supports_without",
    "state_fingerprint",
    "Probe",
    "Response",
    "MemorySnapshot",
    "Agent",
    "FunctionAgent",
    "Scorer",
    "exact_match",
]


@dataclass(frozen=True)
class Probe:
    """One frozen task. ``expected`` may be None for unscored probes (drift-only)."""

    id: str
    input: Any
    expected: Any = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Probe.id must be a non-empty string")
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def fingerprint(self) -> str:
        return fingerprint({"id": self.id, "input": self.input, "expected": self.expected})


@dataclass
class Response:
    """What an agent returns for a probe.

    ``retrieved`` lists the memory ids the agent consulted; it drives retrieval entropy.
    ``confidence`` is the agent's *expressed* confidence in [0, 1], or None if it has none.
    """

    output: Any
    confidence: float | None = None
    retrieved: Sequence[str] = ()


@dataclass
class MemorySnapshot:
    """The agent's memory at one point in time.

    ``texts`` is optional but recommended: when present, REGRESS can measure diversity with
    its own fixed embedder (``ruler="reference"``) instead of trusting the agent's vectors,
    which the agent's learning loop may itself be changing.
    """

    ids: list[str]
    embeddings: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)
    texts: list[str] | None = None

    def __post_init__(self) -> None:
        self.ids = [str(i) for i in self.ids]
        if self.texts is not None:
            self.texts = [str(t) for t in self.texts]
            if len(self.texts) != len(self.ids):
                raise ValueError(f"{len(self.ids)} ids but {len(self.texts)} texts")
        emb = np.asarray(self.embeddings, dtype=float)
        if emb.size == 0 and len(self.ids) == 0 and emb.ndim < 2:
            emb = emb.reshape(0, 0)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (n, d), got shape {emb.shape}")
        if emb.shape[0] != len(self.ids):
            raise ValueError(f"{len(self.ids)} ids but {emb.shape[0]} embeddings")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("memory ids must be unique")
        self.embeddings = emb

    @classmethod
    def empty(cls, dim: int = 0) -> "MemorySnapshot":
        return cls(ids=[], embeddings=np.zeros((0, dim)))

    def __len__(self) -> int:
        return len(self.ids)


@runtime_checkable
class Agent(Protocol):
    def respond(self, probe: Probe) -> Response: ...

    def memory_snapshot(self) -> MemorySnapshot: ...


# Optional capabilities. REGRESS checks for them with hasattr and degrades gracefully:
#
#   agent.without(memory_ids) -> Agent
#       A copy of the agent with those memories removed. Enables ablation
#       (``regress.ablation.confirm_blame``), which turns blame suspects into evidence.
#
#   agent.state_fingerprint() -> str
#       Changes whenever anything that could change an answer changes. Enables the response
#       cache (``regress.cache.ResponseCache``), so re-testing an unchanged checkpoint is free.


def supports_without(agent: object) -> bool:
    return callable(getattr(agent, "without", None))


def state_fingerprint(agent: object) -> str | None:
    fn = getattr(agent, "state_fingerprint", None)
    if not callable(fn):
        return None
    try:
        fp = fn()
    except Exception:
        return None
    return str(fp) if fp else None


class FunctionAgent:
    """Adapt plain callables to the :class:`Agent` protocol."""

    def __init__(
        self,
        respond: Callable[[Probe], Response | Any],
        memory: Callable[[], MemorySnapshot] | MemorySnapshot | None = None,
    ) -> None:
        self._respond = respond
        self._memory = memory

    def respond(self, probe: Probe) -> Response:
        out = self._respond(probe)
        return out if isinstance(out, Response) else Response(output=out)

    def memory_snapshot(self) -> MemorySnapshot:
        if self._memory is None:
            return MemorySnapshot.empty()
        if isinstance(self._memory, MemorySnapshot):
            return self._memory
        return self._memory()


Scorer = Callable[[Probe, Any], float]


def exact_match(probe: Probe, output: Any) -> float:
    if probe.expected is None:
        return float("nan")
    if isinstance(probe.expected, np.ndarray) or isinstance(output, np.ndarray):
        try:
            return float(np.array_equal(np.asarray(probe.expected), np.asarray(output)))
        except Exception:
            return 0.0
    return float(output == probe.expected)
