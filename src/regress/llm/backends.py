"""Language-model backends for the real-agent experiments.

Every backend implements one method::

    complete(system: str, user: str, max_tokens: int) -> Completion

``Completion.confidence`` is the model's own probability for its answer when the backend
can provide it (llama.cpp token log-probabilities), or a self-reported number otherwise.

* :class:`LlamaCppBackend` — a local GGUF model on CPU (``pip install llama-cpp-python``).
  Deterministic at temperature 0, gives token log-probabilities, costs nothing per call.
* :class:`AnthropicBackend` — Claude via the official ``anthropic`` SDK. No log-probabilities,
  so confidence is self-reported.
* :class:`RuleBackend` — a deterministic stand-in "model" with no dependencies, for tests
  and CI. It reads the notes in the prompt like a careful but literal reader, and it
  summarises by truncation, so compressing notes really does lose facts.
* :class:`CachedBackend` — wraps any backend with a disk cache keyed on the full prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

__all__ = [
    "Completion",
    "Backend",
    "LlamaCppBackend",
    "AnthropicBackend",
    "RuleBackend",
    "ScriptedBackend",
    "CachedBackend",
    "ReplayBackend",
    "BackendRefusal",
]


@dataclass
class Completion:
    text: str
    confidence: float | None = None  # in [0, 1]


class BackendRefusal(RuntimeError):
    """The model declined to answer (e.g. a safety refusal)."""


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int) -> Completion: ...


class LlamaCppBackend:
    """A local GGUF model through llama-cpp-python, greedy decoding.

    Confidence is the geometric-mean token probability of the answer:
    ``exp(mean log p(token))``. That is the model's own certainty, not a self-report.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None, seed: int = 0) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("LlamaCppBackend needs `pip install llama-cpp-python`") from exc
        self.model_path = model_path
        self.name = f"llamacpp:{Path(model_path).name}"
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True,
                          verbose=False, seed=seed)
        self._lock = threading.Lock()
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        with self._lock:
            self.calls += 1
            r = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0, logprobs=True, top_logprobs=1,
            )
        choice = r["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        conf = None
        lp = (choice.get("logprobs") or {}).get("content") or []
        vals = [float(t["logprob"]) for t in lp if t.get("logprob") is not None]
        if vals:
            conf = float(math.exp(sum(vals) / len(vals)))
        return Completion(text, conf)


class AnthropicBackend:
    """Claude through the official SDK. Confidence is self-reported ("answer || 0.8").

    Uses server-side refusal fallbacks so a declined request is retried on another model
    inside the same call; a final refusal raises :class:`BackendRefusal`, which the monitor
    records as an error on that probe.
    """

    CONF_INSTRUCTION = (
        " After your answer write ' || ' and your confidence from 0 to 1 that the answer is correct,"
        " for example: Paris || 0.9"
    )

    def __init__(self, model: str = "claude-opus-5-5", client=None, effort: str = "low",
                 self_report_confidence: bool = True) -> None:
        if client is None:
            import anthropic  # pragma: no cover - needs credentials

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.self_report = self_report_confidence
        self.name = f"anthropic:{model}"
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        self.calls += 1
        sys_prompt = system + (self.CONF_INSTRUCTION if self.self_report else "")
        # Thinking cannot be disabled on current models, so leave room beyond the answer.
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max(1024, max_tokens),
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise BackendRefusal(str(getattr(resp, "stop_details", "") or "refused"))
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
        conf = None
        if self.self_report and "||" in text:
            text, _, tail = text.rpartition("||")
            text = text.strip()
            m = re.search(r"[01](?:\.\d+)?", tail)
            if m:
                conf = min(1.0, max(0.0, float(m.group())))
        return Completion(text, conf)


_WORD = re.compile(r"[a-z0-9']+")


class RuleBackend:
    """Deterministic stand-in model for tests and CI (no download, no network).

    * Answering: finds the note sentence that mentions the question's subject and attribute
      and returns the value after "is"/"as"/"in"; confidence 0.9 when found, else answers
      with the most frequent value-like word it sees with confidence 0.6 (a confident guess:
      the behaviour that makes collapse dangerous).
    * Compressing: keeps the first ``n`` words allowed by the prompt's word limit, so
      repeated compression genuinely discards facts.
    """

    name = "rules"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        self.calls += 1
        if "Compress" in system:
            limit = int(re.search(r"at most (\d+) words", system).group(1)) if "at most" in system else 20
            lines = [ln.strip("- ").strip() for ln in user.splitlines() if ln.strip().startswith("-")]
            words = " ".join(lines).split()
            return Completion(" ".join(words[:limit]), None)
        notes, _, question = user.partition("Question:")
        q = question.strip().rstrip("?")
        m = re.match(r"What is (.+?)'s (.+)", q)
        if not m:
            return Completion("unknown", 0.3)
        subject, attr = m.group(1).lower(), m.group(2).lower()
        for sentence in re.split(r"[.\n]", notes):
            s = sentence.lower()
            if subject in s and attr in s and " is " in s:
                value = s.split(" is ", 1)[1].strip(" .-")
                if value:
                    return Completion(value, 0.9)
        values = [w for w in _WORD.findall(notes.lower()) if w not in _STOP and len(w) > 3]
        if values:
            best = max(sorted(set(values)), key=values.count)
            return Completion(best, 0.6)
        return Completion("unknown", 0.3)


_STOP = set("""what the and with notes note works lives favourite colour profession home city pet
answer question their they this that from into were was have has about which also only""".split())


class ScriptedBackend:
    """Wraps a plain function ``(system, user, max_tokens) -> str | Completion``."""

    def __init__(self, fn: Callable[[str, str, int], object], name: str = "scripted") -> None:
        self.fn, self.name, self.calls = fn, name, 0

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        self.calls += 1
        out = self.fn(system, user, max_tokens)
        return out if isinstance(out, Completion) else Completion(str(out))


class ReplayBackend:
    """Answers only from a recording; any prompt not recorded raises ``KeyError``.

    Wrap it in :class:`CachedBackend` pointed at a recorded ``llm-cache.jsonl`` (with the
    original backend's ``name``) to re-run an experiment exactly, without the model.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        raise KeyError("prompt not in the recording; the experiment diverged from the recorded run")


class CachedBackend:
    """Disk cache in front of a backend. Safe only for deterministic (temperature 0) backends."""

    def __init__(self, backend: Backend, path: str | Path) -> None:
        self.backend = backend
        self.name = backend.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, Completion] = {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._mem[rec["k"]] = Completion(rec["text"], rec.get("confidence"))
                except Exception:
                    continue

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([self.name, system, user, max_tokens]).encode()).hexdigest()

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        k = self._key(system, user, max_tokens)
        with self._lock:
            hit = self._mem.get(k)
            if hit is not None:
                self.hits += 1
                return Completion(hit.text, hit.confidence)
        out = self.backend.complete(system, user, max_tokens)
        with self._lock:
            self.misses += 1
            self._mem[k] = out
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": k, "text": out.text, "confidence": out.confidence}) + "\n")
        return out

    @property
    def calls(self) -> int:
        return self.misses
