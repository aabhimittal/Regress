"""The regression monitor: run the frozen suite after every learning cycle."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._canon import output_key as default_output_key
from .cache import ResponseCache
from .core import Agent, MemorySnapshot, Probe, Response, Scorer, exact_match, state_fingerprint
from .detector import ALARM, Assessment, CollapseDetector, Signal
from .metrics import confidence_stats, diversity, drift, mean_score, retrieval_stats
from .metrics.diversity import effective_rank, mean_cosine_distance, normalize_rows
from .probes import ProbeSuite, ProbeSuiteTampered

__all__ = ["CycleReport", "RegressionMonitor", "ProbeSuiteMismatch"]

FORMAT_VERSION = 1


class ProbeSuiteMismatch(ValueError):
    """A saved history was recorded against a different probe suite."""


@dataclass
class CycleReport:
    cycle: int
    status: str
    metrics: dict[str, float]
    signals: list[Signal]
    categories: list[str]
    collapse_signature: bool
    reason: str
    output_keys: dict[str, str | None]
    scores: dict[str, float]
    errors: dict[str, str] = field(default_factory=dict)
    undefined: list[str] = field(default_factory=list)
    alarm_condition: bool = False
    # Per-probe retrieval and confidence, kept so a regression can be traced to the memories
    # that served it (see ``regress.blame``). Empty for histories saved before 0.2.
    retrieved: dict[str, list[str]] = field(default_factory=dict)
    confidences: dict[str, float | None] = field(default_factory=dict)
    # Standardised harmful move per signal (move / tolerance); feeds the CUSUM test.
    moves: dict[str, float] = field(default_factory=dict)
    ruler: str = "agent"  # which embedder measured memory diversity this cycle

    @property
    def alarmed(self) -> bool:
        return self.status == ALARM

    @property
    def screened(self) -> bool:
        return bool(self.metrics.get("screened", 0.0))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = {k: _json_float(v) for k, v in self.metrics.items()}
        d["scores"] = {k: _json_float(v) for k, v in self.scores.items()}
        d["signals"] = [{k: _json_float(v) for k, v in asdict(s).items()} for s in self.signals]
        d["confidences"] = {k: _json_float(v) for k, v in self.confidences.items()}
        d["moves"] = {k: _json_float(v) for k, v in self.moves.items()}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CycleReport":
        return cls(
            cycle=int(d["cycle"]),
            status=d["status"],
            metrics={k: _from_json_float(v) for k, v in d["metrics"].items()},
            signals=[Signal(**{k: _from_json_float(v) for k, v in s.items()}) for s in d["signals"]],
            categories=list(d["categories"]),
            collapse_signature=bool(d["collapse_signature"]),
            reason=d["reason"],
            output_keys=dict(d["output_keys"]),
            scores={k: _from_json_float(v) for k, v in d["scores"].items()},
            errors=dict(d.get("errors", {})),
            undefined=list(d.get("undefined", [])),
            alarm_condition=bool(d.get("alarm_condition", False)),
            retrieved={k: list(v) for k, v in d.get("retrieved", {}).items()},
            confidences={k: (None if v is None else float(v)) for k, v in d.get("confidences", {}).items()},
            moves={k: _from_json_float(v) for k, v in d.get("moves", {}).items()},
            ruler=str(d.get("ruler", "agent")),
        )


def _json_float(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, (np.floating,)):
        return _json_float(float(v))
    return v


def _from_json_float(v: Any) -> Any:
    if v is None:
        return float("nan")
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return v


def _confidence_or_none(c: Any) -> float | None:
    try:
        f = float(c)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class RegressionMonitor:
    """Freeze a probe suite at cycle 0, then measure every consolidation cycle against it.

    Parameters
    ----------
    suite:
        The frozen probes. Its fingerprint is pinned at construction and re-verified on
        every evaluation, so a mutated probe raises rather than shifting the baseline.
    scorer:
        ``(probe, output) -> float in [0, 1]``. Defaults to exact match.
    output_key:
        Maps an output to a hashable identity for drift. Supply one for free-text outputs
        (``regress.adapters.text_key``) so cosmetic rewording does not register as drift.
    warmup:
        Number of initial cycles whose median forms the baseline. With ``warmup >= 3`` the
        spread of those cycles also widens tolerances.
    on_alarm:
        Called with the :class:`CycleReport` whenever the status *enters* ALARM. Exceptions
        it raises are recorded in ``report.errors["<on_alarm>"]``.
    ruler:
        What measures memory diversity. ``"agent"`` uses the agent's own vectors;
        ``"reference"`` re-embeds memory *texts* with a fixed embedder owned by REGRESS (the
        agent cannot bend this ruler); ``"auto"`` (default) uses the reference ruler whenever
        the snapshot carries texts. An object with ``embed(list[str])`` is used as the
        reference embedder.
    cache:
        A :class:`~regress.cache.ResponseCache`. Used when the agent exposes
        ``state_fingerprint()``; re-testing an unchanged state is then free.
    screen:
        If set, cycles after the baseline first run a fixed, stratified subset of this many
        probes. The full suite runs when the screen shows any signal at ``screen_margin`` ×
        tolerance *without* the significance gate (an uncertain move is a reason to look
        closer), or any probe error. Cuts cost on quiet cycles; see the README for what
        screening can miss.
    bootstrap:
        Resamples used for the standard errors of diversity and retrieval metrics (0 = off).
    """

    def __init__(
        self,
        suite: ProbeSuite,
        scorer: Scorer = exact_match,
        detector: CollapseDetector | None = None,
        output_key: Callable[[Any], str] = default_output_key,
        warmup: int = 1,
        on_alarm: Callable[[CycleReport], Any] | None = None,
        ruler: Any = "auto",
        cache: ResponseCache | None = None,
        screen: int | None = None,
        screen_margin: float = 1.0,
        bootstrap: int = 30,
        seed: int = 0,
    ) -> None:
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if isinstance(ruler, str) and ruler not in ("auto", "agent", "reference"):
            raise ValueError("ruler must be 'auto', 'agent', 'reference' or an embedder")
        if not isinstance(ruler, str) and not callable(getattr(ruler, "embed", None)):
            raise ValueError("a custom ruler needs an embed(list[str]) method")
        if screen is not None and screen < 1:
            raise ValueError("screen must be a positive number of probes")
        if not 0 < screen_margin <= 1:
            raise ValueError("screen_margin must be in (0, 1]")
        if bootstrap < 0:
            raise ValueError("bootstrap must be >= 0")
        self.suite = suite
        self.scorer = scorer
        self.detector = detector or CollapseDetector()
        self.output_key = output_key
        self.warmup = warmup
        self.on_alarm = on_alarm
        self.ruler = ruler
        self.cache = cache
        self.screen = screen
        self.screen_margin = screen_margin
        self.bootstrap = bootstrap
        self.seed = seed
        self.fingerprint = suite.fingerprint
        self.history: list[CycleReport] = []
        self._ref_embedder = None if isinstance(ruler, str) else ruler
        self._ref_vectors: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._base_vectors: dict[str, np.ndarray] | None = None
        self._screen_ids: list[str] | None = None

    # ------------------------------------------------------------------ evaluation
    @property
    def baseline(self) -> CycleReport | None:
        return self.history[0] if self.history else None

    def _respond(self, agent: Agent, probe: Probe, state: str | None, run: dict[str, Any]) -> Response:
        key = None
        if self.cache is not None:
            if state is None:
                self.cache.stats.uncacheable += 1
            else:
                key = ResponseCache.key(state, probe.fingerprint)
                hit = self.cache.get(key)
                if hit is not None:
                    run["hits"] += 1
                    return hit
        run["calls"] += 1
        resp = agent.respond(probe)
        if not isinstance(resp, Response):
            resp = Response(output=resp)
        if key is not None:
            self.cache.put(key, resp)  # type: ignore[union-attr]
        return resp

    def _run_probes(self, agent: Agent, probes: Sequence[Probe], state: str | None, run: dict[str, Any] | None = None) -> dict[str, Any]:
        run = run or dict(keys={}, scores={}, confs={}, retrieved={}, errors={}, calls=0, hits=0)
        for probe in probes:
            try:
                resp = self._respond(agent, probe, state, run)
                key = self.output_key(resp.output)
                score = float(self.scorer(probe, resp.output))
                if not math.isnan(score) and not 0.0 <= score <= 1.0:
                    raise ValueError(f"scorer returned {score!r}, outside [0, 1]")
            except Exception as exc:  # an agent that crashes on a probe has regressed on it
                run["errors"][probe.id] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                run["keys"][probe.id] = None
                run["scores"][probe.id] = 0.0 if probe.expected is not None else float("nan")
                continue
            run["keys"][probe.id] = key
            run["scores"][probe.id] = score
            run["confs"][probe.id] = resp.confidence
            run["retrieved"][probe.id] = [str(i) for i in (resp.retrieved or ())]
        return run

    # -- memory rulers --------------------------------------------------------------
    def _ruler_name(self, memory: MemorySnapshot | None) -> str:
        if self.ruler == "agent":
            return "agent"
        has_texts = memory is not None and memory.texts is not None
        if self.ruler == "reference" or self._ref_embedder is not None:
            return f"reference:{self._embedder_id()}" if has_texts else "reference:missing-texts"
        return f"reference:{self._embedder_id()}" if has_texts else "agent"

    def _embedder_id(self) -> str:
        emb = self._get_ref_embedder()
        return f"{type(emb).__name__}:{getattr(emb, 'dim', '?')}"

    def _get_ref_embedder(self):
        if self._ref_embedder is None:
            from .adapters import HashingEmbedder

            self._ref_embedder = HashingEmbedder()
        return self._ref_embedder

    def _reference_vectors(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._get_ref_embedder()
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._ref_vectors]
        if missing:
            vecs = np.asarray(emb.embed([t for _, t in missing]), dtype=float)
            for (k, _), v in zip(missing, vecs):
                self._ref_vectors[k] = v
            while len(self._ref_vectors) > 50_000:
                self._ref_vectors.popitem(last=False)
        if not keys:
            return np.zeros((0, 0))
        return np.vstack([self._ref_vectors[k] for k in keys])

    def _bootstrap_memory_se(self, unit: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
        n = unit.shape[0]
        if self.bootstrap == 0 or n < 3:
            return float("nan"), float("nan")
        if n > 1000:
            unit = unit[rng.choice(n, 1000, replace=False)]
            n = 1000
        er, cd = [], []
        for _ in range(self.bootstrap):
            sample = unit[rng.integers(0, n, n)]
            er.append(effective_rank(sample))
            cd.append(mean_cosine_distance(sample))
        return float(np.std(er, ddof=1)), float(np.std(cd, ddof=1))

    # -- metrics --------------------------------------------------------------------
    def _metrics(
        self, run: Mapping[str, Any], ids: Sequence[str], memory: MemorySnapshot | None,
        mem_error: str | None, cycle: int,
    ) -> tuple[dict[str, float], str]:
        nan = float("nan")
        scores = {i: run["scores"][i] for i in ids if i in run["scores"]}
        m: dict[str, float] = {}
        m["accuracy"] = mean_score(list(scores.values()))
        m["public_accuracy"] = mean_score([s for k, s in scores.items() if not self.suite.is_holdout(k)])
        m["holdout_accuracy"] = mean_score([s for k, s in scores.items() if self.suite.is_holdout(k)])
        m["error_rate"] = sum(1 for i in ids if i in run["errors"]) / max(1, len(ids))

        base = self.baseline
        keys = {i: run["keys"][i] for i in ids if i in run["keys"]}
        if base is None:
            m.update(output_jsd=0.0, change_rate=0.0, regressions=0.0, fixes=0.0, probe_overfit_gap=0.0,
                     accuracy_delta_se=0.0)
        else:
            d = drift(base.output_keys, keys, base.scores, scores)
            m.update(output_jsd=d.output_jsd, change_rate=d.change_rate,
                     regressions=float(d.regressions), fixes=float(d.fixes))
            m["probe_overfit_gap"] = self._overfit_lower_bound(base.scores, scores)
            deltas = [c - base.scores[k] for k, c in scores.items()
                      if k in base.scores and not math.isnan(c) and not math.isnan(base.scores[k])]
            m["accuracy_delta_se"] = (
                float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) >= 2 else nan
            )

        rng = np.random.default_rng(self.seed * 1_000_003 + cycle)
        ruler = self._ruler_name(memory)
        memory_keys = ("memory_size", "memory_dropped", "mean_cosine_distance", "effective_rank",
                       "effective_rank_ratio", "near_duplicate_fraction", "effective_rank_se",
                       "mean_cosine_distance_se", "agent_effective_rank", "embedding_drift")
        if memory is None or ruler == "reference:missing-texts":
            m.update({k: nan for k in memory_keys})
            if memory is not None:
                m["memory_size"] = float(len(memory))
            rs = retrieval_stats([run["retrieved"][i] for i in ids if i in run["retrieved"]],
                                 memory.ids if memory is not None else None)
        else:
            vectors = memory.embeddings if ruler == "agent" else self._reference_vectors(memory.texts or [])
            if ruler != "agent" and len(memory) == 0:
                vectors = np.zeros((0, 0))
            dv = diversity(vectors)
            unit, _ = normalize_rows(vectors if vectors.ndim == 2 else vectors.reshape(0, 0))
            er_se, cd_se = self._bootstrap_memory_se(unit, rng)
            m.update(memory_size=float(len(memory)), memory_dropped=float(dv.dropped),
                     mean_cosine_distance=dv.mean_cosine_distance, effective_rank=dv.effective_rank,
                     effective_rank_ratio=dv.effective_rank_ratio,
                     near_duplicate_fraction=dv.near_duplicate_fraction,
                     effective_rank_se=er_se, mean_cosine_distance_se=cd_se)
            m["agent_effective_rank"] = (
                dv.effective_rank if ruler == "agent" else diversity(memory.embeddings).effective_rank
                if memory.embeddings.size else nan
            )
            m["embedding_drift"] = self._embedding_drift(memory)
            rs = retrieval_stats([run["retrieved"][i] for i in ids if i in run["retrieved"]], memory.ids)
        lists = [run["retrieved"][i] for i in ids if i in run["retrieved"]]
        m.update(retrieval_entropy=rs.normalized_entropy, retrieval_top1_share=rs.top1_share,
                 retrieval_coverage=rs.coverage, retrieval_gini=rs.gini, phantom_rate=rs.phantom_rate)
        m["retrieval_entropy_se"] = self._bootstrap_entropy_se(
            lists, memory.ids if memory is not None and ruler != "reference:missing-texts" else None, rng)

        confs = [run["confs"].get(i) for i in ids if i in run["scores"] and i not in run["errors"]]
        conf_scores = [run["scores"][i] for i in ids if i in run["scores"] and i not in run["errors"]]
        cs = confidence_stats(confs, conf_scores)
        valid = [c for c in (_confidence_or_none(x) for x in confs) if c is not None and 0 <= c <= 1]
        m.update(mean_confidence=cs.mean_confidence, overconfidence=cs.overconfidence, ece=cs.ece,
                 invalid_confidence=float(cs.invalid),
                 mean_confidence_se=float(np.std(valid, ddof=1) / math.sqrt(len(valid))) if len(valid) >= 2 else nan)
        m["memory_error"] = 1.0 if mem_error else 0.0
        return m, ruler

    def _bootstrap_entropy_se(self, lists: list[list[str]], universe: Sequence[str] | None,
                              rng: np.random.Generator) -> float:
        n = len(lists)
        if self.bootstrap == 0 or n < 3:
            return float("nan")
        vals = []
        for _ in range(self.bootstrap):
            idx = rng.integers(0, n, n)
            v = retrieval_stats([lists[i] for i in idx], universe).normalized_entropy
            if math.isfinite(v):
                vals.append(v)
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")

    def _embedding_drift(self, memory: MemorySnapshot) -> float:
        """1 − mean cosine between each memory's vector now and at the baseline (same id)."""
        if self.baseline is None:
            return 0.0  # the baseline is compared with itself
        if self._base_vectors is None or memory.embeddings.size == 0:
            return float("nan")
        now = {i: v for i, v in zip(memory.ids, memory.embeddings)}
        common = [i for i in self._base_vectors if i in now]
        if not common:
            return float("nan")
        sims = []
        for i in common:
            a, b = self._base_vectors[i], np.asarray(now[i], dtype=float)
            if a.shape != b.shape:
                return 1.0  # the embedding space itself changed
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims.append(float(a @ b / (na * nb)))
        return float(1.0 - np.mean(sims)) if sims else float("nan")

    def _remember_baseline_vectors(self, memory: MemorySnapshot | None) -> None:
        if memory is None or memory.embeddings.size == 0:
            return
        keep = list(zip(memory.ids, memory.embeddings))[:512]
        self._base_vectors = {i: np.array(v, dtype=float) for i, v in keep}

    def _overfit_lower_bound(self, base: Mapping[str, float], cur: Mapping[str, float]) -> float:
        """Lower 95% bound on (public gain − held-out gain), from paired per-probe changes."""
        pub, hold = [], []
        for k, c in cur.items():
            b = base.get(k, math.nan)
            if math.isnan(b) or math.isnan(c):
                continue
            (hold if self.suite.is_holdout(k) else pub).append(c - b)
        if len(pub) < 2 or len(hold) < 2:
            return float("nan")
        p, h = np.array(pub), np.array(hold)
        gap = p.mean() - h.mean()
        se = math.sqrt(p.var(ddof=1) / p.size + h.var(ddof=1) / h.size)
        return float(gap - 1.96 * se)

    # -- screening ------------------------------------------------------------------
    def screen_ids(self) -> list[str]:
        """The fixed, stratified screening subset (same probes every cycle)."""
        if self._screen_ids is None:
            n = min(self.screen or len(self.suite), len(self.suite))
            strata: dict[tuple, list[str]] = {}
            for p in self.suite:
                strata.setdefault((self.suite.is_holdout(p.id), p.tags[0] if p.tags else ""), []).append(p.id)
            for ids in strata.values():
                ids.sort(key=lambda i: hashlib.sha256(f"screen:{self.seed}:{i}".encode()).hexdigest())
            total = len(self.suite)
            quota = {k: n * len(v) / total for k, v in strata.items()}
            take = {k: min(len(strata[k]), max(1, math.floor(q))) for k, q in quota.items()}
            # Largest remainder until the budget is met (or every stratum is exhausted).
            order = sorted(strata, key=lambda k: quota[k] - math.floor(quota[k]), reverse=True)
            while sum(take.values()) < n and any(take[k] < len(strata[k]) for k in strata):
                for k in order:
                    if sum(take.values()) >= n:
                        break
                    if take[k] < len(strata[k]):
                        take[k] += 1
            while sum(take.values()) > n:  # the "at least one per stratum" rule overshot
                k = max(take, key=lambda s: take[s])
                take[k] -= 1
            chosen = {i for k, v in strata.items() for i in v[: take[k]]}
            self._screen_ids = [p.id for p in self.suite if p.id in chosen]
        return self._screen_ids

    def _baseline_metrics_for(self, ids: Sequence[str]) -> dict[str, float]:
        """Re-score the stored baseline on a probe subset, so a screen compares like with like."""
        base = self.baseline
        assert base is not None
        scores = {i: base.scores[i] for i in ids if i in base.scores}
        m = dict(base.metrics)  # memory metrics do not depend on which probes ran
        m["accuracy"] = mean_score(list(scores.values()))
        m["public_accuracy"] = mean_score([s for k, s in scores.items() if not self.suite.is_holdout(k)])
        m["holdout_accuracy"] = mean_score([s for k, s in scores.items() if self.suite.is_holdout(k)])
        m["error_rate"] = sum(1 for i in ids if base.output_keys.get(i, "") is None) / max(1, len(ids))
        m.update(output_jsd=0.0, change_rate=0.0, probe_overfit_gap=0.0)
        if base.retrieved:
            size = base.metrics.get("memory_size")
            rs = retrieval_stats([base.retrieved.get(i, []) for i in ids if i in base.retrieved], None,
                                 memory_size=int(size) if size == size and size else None)
            m.update(retrieval_entropy=rs.normalized_entropy, retrieval_top1_share=rs.top1_share)
        if base.confidences:
            ok = [i for i in ids if i in base.confidences]
            cs = confidence_stats([base.confidences[i] for i in ok], [base.scores.get(i, math.nan) for i in ok])
            m.update(mean_confidence=cs.mean_confidence, overconfidence=cs.overconfidence)
        return m

    # -- main entry point -------------------------------------------------------------
    def evaluate(self, agent: Agent, cycle: int, record: bool = True, full: bool = False) -> CycleReport:
        """Run the suite against ``agent`` as it stands after ``cycle``.

        With ``record=False`` the report is computed against the existing baseline but not
        appended to history; bisection uses this to re-test old checkpoints. ``full=True``
        bypasses screening.
        """
        self.suite.verify(self.fingerprint)
        if record and self.history and cycle <= self.history[-1].cycle:
            raise ValueError(
                f"cycle {cycle} is not after the last recorded cycle {self.history[-1].cycle}"
            )
        if not record and not self.history:
            raise ValueError("record a baseline (cycle 0) before evaluating unrecorded cycles")

        state = state_fingerprint(agent) if self.cache is not None else None
        all_ids = [p.id for p in self.suite]
        screening = (self.screen is not None and self.baseline is not None and not full
                     and self.screen < len(self.suite))
        first_ids = self.screen_ids() if screening else all_ids
        id_set = set(first_ids)
        run = self._run_probes(agent, [p for p in self.suite if p.id in id_set], state)

        mem_error = None
        memory: MemorySnapshot | None
        try:
            memory = agent.memory_snapshot()
        except Exception as exc:
            memory, mem_error = None, f"{type(exc).__name__}: {exc}"

        escalated = False
        if screening:
            metrics, ruler = self._metrics(run, first_ids, memory, mem_error, cycle)
            window = [self._baseline_metrics_for(first_ids)]
            near, _, _ = self.detector.measure(metrics, window, tol_scale=self.screen_margin, use_se=False)
            if any(not s.weak for s in near) or metrics["error_rate"] > 0:
                escalated = True
                rest = [p for p in self.suite if p.id not in id_set]
                run = self._run_probes(agent, rest, state, run)
                metrics, ruler = self._metrics(run, all_ids, memory, mem_error, cycle)
                window = [r.metrics for r in self.history[: self.warmup]]
            evaluated = all_ids if escalated else first_ids
        else:
            metrics, ruler = self._metrics(run, all_ids, memory, mem_error, cycle)
            window = [r.metrics for r in self.history[: self.warmup]]
            evaluated = all_ids
        metrics.update(probe_calls=float(run["calls"]), cache_hits=float(run["hits"]),
                       probes_evaluated=float(len(evaluated)),
                       screened=1.0 if screening and not escalated else 0.0,
                       escalated=1.0 if escalated else 0.0)

        in_warmup = record and len(self.history) < self.warmup
        if not window or in_warmup:
            window = window + [metrics]
        recent = ([r.alarm_condition for r in self.history[-(self.detector.patience - 1):]]
                  if self.detector.patience > 1 else [])
        prior_moves = [r.moves for r in self.history[self.warmup:]]
        if in_warmup:
            reason = (
                "baseline recorded"
                if self.warmup == 1
                else f"warm-up cycle {len(self.history) + 1}/{self.warmup} (part of the baseline)"
            )
            assessment = Assessment(status="OK", reason=reason)
            assessment.undefined = [k for k, v in metrics.items() if isinstance(v, float) and math.isnan(v)]
        else:
            assessment = self.detector.assess(metrics, window, recent if record else [],
                                              prior_moves if record else prior_moves)
        errors = dict(run["errors"])
        if mem_error:
            errors["<memory_snapshot>"] = mem_error
        if ruler == "reference:missing-texts":
            errors["<ruler>"] = "ruler='reference' needs MemorySnapshot.texts; diversity not measured"
        if self.baseline is not None and ruler != self.baseline.ruler:
            errors["<ruler>"] = (f"diversity measured with '{ruler}' but the baseline used "
                                 f"'{self.baseline.ruler}'; diversity changes are not comparable")
        report = CycleReport(
            cycle=cycle,
            status=assessment.status,
            metrics=metrics,
            signals=assessment.signals,
            categories=assessment.categories,
            collapse_signature=assessment.collapse_signature,
            reason=assessment.reason + (" (screened)" if screening and not escalated else ""),
            output_keys={i: run["keys"][i] for i in evaluated if i in run["keys"]},
            scores={i: run["scores"][i] for i in evaluated if i in run["scores"]},
            errors=errors,
            undefined=assessment.undefined,
            alarm_condition=CollapseDetector.alarm_condition(assessment, self.detector.min_categories),
            retrieved={i: run["retrieved"][i] for i in evaluated if i in run["retrieved"]},
            confidences={i: _confidence_or_none(run["confs"].get(i)) for i in evaluated if i in run["confs"]},
            moves=dict(getattr(assessment, "moves", {}) or {}),
            ruler=ruler,
        )
        if record:
            if not self.history:
                self._remember_baseline_vectors(memory)
            entered = report.alarmed and not (self.history and self.history[-1].alarmed)
            self.history.append(report)
            if entered and self.on_alarm is not None:
                try:
                    self.on_alarm(report)
                except Exception as exc:  # a broken pager must not break the learning loop
                    report.errors["<on_alarm>"] = f"{type(exc).__name__}: {exc}"
        return report

    # ------------------------------------------------------------------ queries
    def first_alarm(self) -> CycleReport | None:
        return next((r for r in self.history if r.alarmed), None)

    def series(self, metric: str) -> tuple[list[int], list[float]]:
        return [r.cycle for r in self.history], [r.metrics.get(metric, float("nan")) for r in self.history]

    def cost(self) -> dict[str, float]:
        """Probe calls made and saved across the recorded history."""
        calls = sum(r.metrics.get("probe_calls", 0.0) for r in self.history)
        hits = sum(r.metrics.get("cache_hits", 0.0) for r in self.history)
        full = len(self.suite) * len(self.history)
        return {"probe_calls": calls, "cache_hits": hits, "full_suite_calls": float(full),
                "saved_fraction": 1.0 - calls / full if full else 0.0}

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "suite_fingerprint": self.fingerprint,
            "warmup": self.warmup,
            "detector": self.detector.config(),
            "history": [r.to_dict() for r in self.history],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path, suite: ProbeSuite, **kwargs: Any) -> "RegressionMonitor":
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported history format {data.get('format')!r}")
        if data["suite_fingerprint"] != suite.fingerprint:
            raise ProbeSuiteMismatch(
                "history was recorded against a different probe suite "
                f"({data['suite_fingerprint'][:12]}… vs {suite.fingerprint[:12]}…)"
            )
        try:
            suite.verify()
        except ProbeSuiteTampered as exc:  # the suite object was mutated after construction
            raise ProbeSuiteMismatch(str(exc)) from exc
        kwargs.setdefault("warmup", data.get("warmup", 1))
        if "detector" not in kwargs and data.get("detector"):
            kwargs["detector"] = CollapseDetector.from_config(data["detector"])
        mon = cls(suite, **kwargs)
        mon.history = [CycleReport.from_dict(r) for r in data["history"]]
        return mon
