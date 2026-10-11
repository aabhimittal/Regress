"""Turn blame suspects into evidence by removing them and re-running the probes.

``blame`` ranks the memories that regressed probes leaned on. That is correlation: a hub
retrieved by everything will be "near" every failure. ``confirm_blame`` tests the claim:

1. take the agent as it stood at the regressed cycle;
2. remove the suspect memories (``agent.without(ids)``) and re-run the regressed probes;
3. do the same for several random *control* sets of the same size, drawn from memories the
   regressed probes also retrieved (so the control is "remove something nearby", not
   "remove something irrelevant");
4. also re-run probes that still passed, to see what removing the suspects breaks.

Verdicts:

* ``confirmed`` — removing the suspects recovers clearly more probes than any control.
* ``refuted`` — suspects recover no more than the controls do on average.
* ``inconclusive`` — in between, or too few regressed probes to tell.

Cost: (1 + controls + individual suspects) × |regressed probes| + collateral sample calls.
Pass the monitor's response cache to avoid paying for repeats.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .blame import blame
from .core import Agent, supports_without

__all__ = ["AblationResult", "confirm_blame", "AblationUnsupported"]

PASS = 0.5


class AblationUnsupported(TypeError):
    """The agent has no ``without(memory_ids)`` method."""


@dataclass
class AblationResult:
    cycle: int
    suspects: list[str]
    regressed: list[str]
    recovered: int  # regressed probes that pass again with the suspects removed
    control_recovered: list[int]  # same, for each random control set
    collateral: int  # previously passing probes that fail with the suspects removed
    collateral_checked: int
    individual: dict[str, int] = field(default_factory=dict)  # recovery per single suspect
    verdict: str = "inconclusive"
    note: str = ""
    calls: int = 0

    @property
    def recovery_rate(self) -> float:
        return self.recovered / len(self.regressed) if self.regressed else float("nan")

    @property
    def control_rate(self) -> float:
        if not self.control_recovered or not self.regressed:
            return float("nan")
        return sum(self.control_recovered) / len(self.control_recovered) / len(self.regressed)


def _passes(agent: Agent, monitor, probe_ids: Sequence[str]) -> tuple[int, int]:
    """Count probes that pass on ``agent``; returns (passes, calls)."""
    passes = calls = 0
    for pid in probe_ids:
        probe = monitor.suite[pid]
        try:
            resp = agent.respond(probe)
            calls += 1
            out = resp.output if hasattr(resp, "output") else resp
            score = float(monitor.scorer(probe, out))
        except Exception:
            continue
        if score >= PASS:
            passes += 1
    return passes, calls


def confirm_blame(
    monitor,
    agent: Agent,
    cycle: int | None = None,
    top: int = 5,
    controls: int = 3,
    individual: bool = True,
    collateral_sample: int = 40,
    seed: int = 0,
) -> AblationResult:
    """Ablate the top ``top`` blame suspects at ``cycle`` and compare with random controls.

    ``agent`` must be the agent *as it stood at that cycle* (a checkpoint) and must implement
    ``without(memory_ids)``.
    """
    if not supports_without(agent):
        raise AblationUnsupported(
            "confirm_blame needs agent.without(memory_ids) returning a copy without those memories"
        )
    if top < 1 or controls < 0 or collateral_sample < 0:
        raise ValueError("need top >= 1, controls >= 0, collateral_sample >= 0")
    rep = blame(monitor.history, cycle, top=top)
    target = next(h for h in monitor.history if h.cycle == rep.cycle)
    suspects = [s.memory_id for s in rep.suspects]
    res = AblationResult(rep.cycle, suspects, rep.regressed, 0, [], 0, 0)
    if not rep.regressed or not suspects:
        res.note = rep.note or "nothing to ablate"
        return res

    # The memories the regressed probes leaned on, minus the suspects: the control pool.
    pool = sorted({m for p in rep.regressed for m in target.retrieved.get(p, [])} - set(suspects))
    try:
        live = set(agent.memory_snapshot().ids)
        pool = [m for m in pool if m in live]
    except Exception:
        pass

    calls = 0
    rec, c = _passes(agent.without(suspects), monitor, rep.regressed)
    res.recovered, calls = rec, calls + c

    rng = random.Random(seed)
    if len(pool) >= len(suspects):
        for _ in range(controls):
            ctrl = rng.sample(pool, len(suspects))
            r, c = _passes(agent.without(ctrl), monitor, rep.regressed)
            res.control_recovered.append(r)
            calls += c
    elif controls:
        res.note = f"control pool too small ({len(pool)} memories) for {len(suspects)}-memory controls"

    if individual:
        for sid in suspects:
            r, c = _passes(agent.without([sid]), monitor, rep.regressed)
            res.individual[sid] = r
            calls += c

    still_passing = [p for p, s in target.scores.items() if s == s and s >= PASS]
    sample = still_passing if len(still_passing) <= collateral_sample else rng.sample(still_passing, collateral_sample)
    if sample:
        kept, c = _passes(agent.without(suspects), monitor, sample)
        res.collateral, res.collateral_checked = len(sample) - kept, len(sample)
        calls += c
    res.calls = calls

    n = len(rep.regressed)
    if n < 3:
        res.verdict, res.note = "inconclusive", (res.note + "; " if res.note else "") + "fewer than 3 regressed probes"
    elif not res.control_recovered:
        res.verdict = "confirmed" if res.recovered / n >= 0.3 else "inconclusive"
        res.note = (res.note + "; " if res.note else "") + "no controls: verdict uses recovery alone"
    else:
        mean_c = sum(res.control_recovered) / len(res.control_recovered)
        sd_c = math.sqrt(sum((x - mean_c) ** 2 for x in res.control_recovered) / max(1, len(res.control_recovered) - 1))
        if res.recovered > max(res.control_recovered) and res.recovered - mean_c >= max(2.0, 2 * sd_c, 0.1 * n):
            res.verdict = "confirmed"
        elif res.recovered <= mean_c:
            res.verdict = "refuted"
        else:
            res.verdict = "inconclusive"
    return res
