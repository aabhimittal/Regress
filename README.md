# REGRESS

**Regression testing for self-evolving agents.**

Agent stacks that learn from their own experience (reflection memories, skill libraries,
self-distillation, sleep-time consolidation) quietly assume that every learning cycle
leaves the agent at least as good as before. Recent work on continual experience
internalisation shows that assumption can fail: when experience is internalised badly,
repeated cycles collapse instead of compounding. Very few of these stacks run the
equivalent of a regression suite after each cycle.

REGRESS is that suite. It freezes a set of probe tasks at cycle 0 and, after every
consolidation cycle, measures:

| Family | What it measures | Why |
|---|---|---|
| **Behavioural drift** | accuracy vs. baseline, output-distribution shift (JSD), per-probe change rate, regressions/fixes | Did the agent get worse, or just different? |
| **Memory diversity** | mean pairwise cosine distance, **effective rank**, near-duplicate share | Homogenised memory is the collapse mechanism |
| **Retrieval entropy** | normalised entropy, top-1 share, coverage, Gini, phantom rate | Is the same handful of memories answering everything? |
| **Confidence** | mean expressed confidence, overconfidence (confidence − accuracy), ECE | Collapse feels like certainty from the inside |
| **Integrity** | public vs. held-out probe gap, probe error rate | Is the agent learning the task, or the test? |

It alarms on **combinations**, not single metrics, and helps you find **where the
decline began**, which is often several cycles before the alarm fires.

> **Analogy.** A sourdough starter. It needs regular feeding, but it can also go bad, and
> the failure is slow and invisible until the bread doesn't rise. You taste-test on a
> schedule rather than waiting for the loaf. REGRESS is the taste test. It also compares
> every taste with *day one*, not yesterday, because a starter that sours a little each
> day never tastes different from the day before.

---

## 60-second demo

```bash
pip install -e .
regress demo                     # self-reinforcing collapse from cycle 8
regress demo --scenario slow     # gentle collapse: the alarm trails the onset
regress demo --scenario leak     # probes leak into an answer cache
regress demo --scenario healthy  # stays quiet (false-positive check)
regress demo --scenario all --save run.json && regress show run-collapse.json
```

The demo runs a toy self-evolving agent (kNN over an episodic memory with
importance-weighted retrieval, see `regress/sim.py`) through 20 learning cycles. In the
bad scenarios the agent switches to *self-reinforcing internalisation*: it labels new
experience with its own predictions, pulls memories toward the most-retrieved "hub", and
merges near-copies into it. Nobody tells it to collapse. The signature emerges from the
loop:

```
cycle  status    acc   held  eRank spread  r.ent   top1   conf  signals
    7  OK      0.969  0.974   18.7  0.911  0.952  0.021  0.622
    8  OK      0.969  0.947   16.7  0.873  0.953  0.018  0.662
    9  WATCH   0.956  0.947   14.6  0.833  0.949  0.016  0.671  diversity_drop
   10  ALARM   0.956  0.974   13.2  0.788  0.935  0.027  0.701  diversity_drop, confidence_rise
   11  ALARM   0.919  0.868   12.6  0.757  0.907  0.044  0.631  accuracy_drop, diversity_drop
   ...
   14  ALARM   0.588  0.553   12.7  0.757  0.824  0.073  0.526  accuracy_drop, output_shift, churn, ...

trends, cycle 0 → 20
  accuracy           ███████████▇▇▇▄▄▃▂▂▂▁  0.981 → 0.212
  effective rank     ███▇▇▇▇█▆▄▃▂▃▃▂▂▂▁▁▁▁  19.246 → 11.551
  retrieval entropy  ██████████▇▆▅▄▃▃▂▂▁▁▄  0.949 → 0.835
  mean confidence    ▅▅▅▅▅▅▅▅▆▇▇▅▃▁▃▁▃▃▃▅█  0.627 → 0.724
  status             ·········~▲▲▲▲▲▲▲▲▲▲▲  · ok  ~ watch  ▲ alarm

ALARM first raised at cycle 10: collapse signature: diversity falling while expressed confidence rises
  • diversity_drop   effective_rank 19.246 → 13.188  (worse by 31.5%, tolerance 20%)
  • confidence_rise  mean_confidence 0.627 → 0.701  (worse by 0.074, tolerance 0.050)
bisect   first bad checkpoint 10, last good 9 — 7 evaluations [exact]
onset    decline most likely began at cycle 7 (effective_rank@7, accuracy@11)
```

The alarm fires at cycle 10, while accuracy is still 0.956 against a 0.981 baseline.
Accuracy only falls apart from cycle 14. That gap is the point of the project.

---

## Using it with your agent

Your agent needs two methods:

```python
from regress import Probe, ProbeSuite, RegressionMonitor, Response, MemorySnapshot

class MyAgent:
    def respond(self, probe: Probe) -> Response:
        ...
        return Response(output=answer, confidence=0.83, retrieved=["mem-17", "mem-4"])

    def memory_snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(ids=memory_ids, embeddings=memory_matrix)  # (n, d)

suite = ProbeSuite([Probe("q1", "What is ...?", expected="..."), ...],
                   holdout_fraction=0.25)          # 25 % never leaves the monitor
monitor = RegressionMonitor(suite)

for cycle in range(n_cycles):
    report = monitor.evaluate(agent, cycle)        # run the frozen suite
    if report.alarmed:
        print(report.reason, [s.describe() for s in report.signals])
    agent.consolidate()                            # your learning step

monitor.save("history.json")
```

`FunctionAgent(respond_fn, memory_fn)` wraps plain callables. `examples/quickstart.py`
shows a complete example: a keyword QA bot whose "dedupe" consolidation step merges
distinct facts. REGRESS flags it at cycle 3, while accuracy is still 0.88.

**Free-text outputs:** pass `output_key=` so that rewording does not count as drift
(for example, normalise whitespace and case, or map each answer to an embedding cluster id).
**Graded tasks:** pass `scorer=(probe, output) -> float in [0, 1]`.
**Noisy agents:** `warmup=5` uses the median of the first five cycles as the baseline and
widens each tolerance to 3× that window's robust spread (MAD).

### Finding where it started

```python
from regress import bisect_checkpoints, locate_onset

# 1. Which saved checkpoint first fails? O(log n) re-evaluations, git-bisect style.
result = bisect_checkpoints(
    cycles=range(0, 41),
    is_bad=lambda c: monitor.evaluate(load_checkpoint(c), c, record=False).alarmed,
)   # is_bad may return None for an unloadable checkpoint (skipped, like `git bisect skip`)

# 2. When did the decline actually begin? (change-point fit on the signalled metrics)
onset_cycle, per_metric = locate_onset(monitor.history)
```

These are **different questions**. With slow decay, the first checkpoint that crosses a
threshold can come many cycles after the decline began. In the `slow` demo the
consolidation goes bad at cycle 5, the alarm and bisection point at cycle 9, and the onset
estimate says cycle 3. The cause is closer to the onset than to the alarm.
`estimate_onset` fits "flat → linear decline" (hinge) and "flat → step" models. It only
accepts fits that move in the harmful direction, and it fits only up to the first alarm
plus two cycles, because a collapse in progress saturates and oscillates.

---

## The alarm rule

Each metric is compared with the **baseline** (cycle 0, or the warm-up median) and never
with the previous cycle. A move past tolerance in the harmful direction is a *signal*.
Signals are grouped into categories: drift, diversity, retrieval, confidence, integrity.

* **OK**: no signals.
* **WATCH**: some signals, but no alarming combination.
* **ALARM**: either
  * the **collapse signature**: a diversity signal *and* rising expressed confidence in
    the same cycle, or
  * signals in **≥ 2 independent categories** (`min_categories`),

  sustained for `patience` consecutive cycles (default 1).

The per-probe change rate ("churn") is a *weak* signal. It can raise WATCH but never
counts toward an alarm, because learning *should* change outputs.

Default tolerances (`Thresholds`): accuracy −0.05; output JSD 0.15 bits; effective rank and
pairwise spread −20 % (relative); near-duplicates +0.15; retrieval entropy −0.15; top-1
share +0.15; confidence +0.05; overconfidence +0.10; probe-overfit lower bound 0.05; error
rate 0.05. **These are starting points, not validated constants.** See *Limitations*.

---

## Blind spots the metrics deliberately cover

Each one has a test.

| Trap | What goes wrong naively | What REGRESS does |
|---|---|---|
| **Antipodal collapse** | Memory collapses into two opposite clusters, and mean pairwise distance *rises* | Also tracks effective rank (exp of singular-value entropy), which falls to ≈1 |
| **Answer permutation** | Every A becomes B and every B becomes A: output-distribution JSD = 0 | Per-probe change rate = 1.0 |
| **Small probe set vs. big memory** | 10 retrievals over 10 000 memories look like "low entropy" | Entropy normalised by log(min(memories, retrievals)) |
| **Phantom retrievals** | Agent reports ids no longer in memory, which inflates entropy | Excluded from the distribution and reported as `phantom_rate` |
| **Crashing probes** | Errors silently drop out of accuracy and drift | An error is a failure and its own output class, and counts in `error_rate` |
| **Boiling frog** | 7 %/cycle decline never trips a step-to-step threshold | Always compared with the baseline |
| **Holdout noise** | Raw public−held-out gap is ±0.1 on 40 probes, by chance alone | Uses the lower 95 % bound from paired per-probe changes |
| **Suite tampering** | `repr(np.zeros(10_000))` hides edits in the middle of an array | SHA-256 over raw bytes, dtype and shape, re-verified every cycle |
| **Uncalibrated self-reports** | Confidence of 1.7 or NaN poisons the mean | Counted as `invalid_confidence`, excluded |
| **Noisy agents** | Stochastic agents trip fixed thresholds | Warm-up window widens tolerances by 3 × robust spread (MAD) |
| **Flaky bisection** | A noisy evaluator sends binary search to the wrong cycle, silently | Re-checks the boundary pair and reports `consistent=False` |

---

## Budget policies (fold-ins)

`regress.budget` holds two small policies. They are here because they are among the most
direct *causes* of the collapse REGRESS measures:

* `plan_consolidation`: expected-value consolidation. Consolidate item *i* only if
  `p_hit × latency_saved − cost > 0`, then fill a compute budget by value per unit cost.
  This is ALLOCATE-style budgeting applied to sleep-time compute. It belongs here as an
  input to measure, not as a separate project.
* `MemoryBudget`: a hard byte cap with a forgetting policy (`lru`, `lfu`, `value_density`,
  `redundancy`), suited to on-device and edge memory. The cap is an invariant: a
  property-based test checks it under arbitrary operation sequences. Note that **LFU
  homogenises memory by construction**, because the rarely-hit tail is the diverse part.
  The `redundancy` policy evicts the item closest to another kept item instead.

---

## Limitations

* **Probe suites go stale, and the agent can overfit them.** The held-out split catches
  leakage of the *public* half. It does nothing if the held-out probes themselves leak, for
  example through logs the agent can read. Rotate suites, and keep held-out probes out of
  any data path the agent learns from.
* **The thresholds are arbitrary until validated across stacks.** They were tuned on the
  bundled simulator, which has zero false alarms across 8 healthy seeds × 20 cycles and
  alarms 1–5 cycles after onset in the collapse scenario. That is one toy world. Without
  cross-stack validation, treat them as hypotheses.
* **The simulator is a cartoon.** Its collapse mechanism (hub attraction plus
  importance-weighted retrieval plus self-labelling) was designed to produce the
  signature. It shows that the detector responds to the signature. It does not show that
  real agents collapse this way.
* **Expressed confidence is only as meaningful as the agent's self-report.** Agents without
  a confidence channel lose half of the collapse signature and fall back on the
  multi-category rule.
* **Embeddings must be comparable within a cycle.** Diversity is recomputed from scratch
  each cycle, so a change of embedding model does not crash anything. It will, however,
  move the diversity numbers for reasons unrelated to memory health.
* **Onset estimates are approximate.** In the simulator they land within about two cycles of
  the truth and lean early. An early estimate is the cheaper mistake, because you search a
  superset of the right range.
* **Bisection assumes monotone degradation.** If an agent goes bad, recovers, and goes bad
  again, bisection finds *a* boundary rather than the first one. A test documents this.

---

## Layout

```
src/regress/
  core.py        Probe, Response, MemorySnapshot, Agent protocol, FunctionAgent
  probes.py      ProbeSuite: fingerprinting, tamper detection, stable held-out split
  metrics/       drift, diversity, retrieval, confidence (pure functions, numpy only)
  detector.py    signals → categories → OK / WATCH / ALARM
  monitor.py     RegressionMonitor: run the suite, score, assess, persist
  bisect.py      checkpoint bisection + change-point onset estimation
  budget.py      consolidation EV rule + byte-capped memory with forgetting policies
  sim.py         toy self-evolving agent and scenarios
  report.py      tables, sparklines, verdicts
  cli.py         `regress demo`, `regress show`
```

Dependencies: `numpy` only. Python ≥ 3.10.

## Development

```bash
pip install -e ".[test]" ruff
pytest              # ~130 tests, including Hypothesis property tests
ruff check src tests examples
```

MIT licensed.
