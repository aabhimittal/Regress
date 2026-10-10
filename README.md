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

## Try it

* **Interactive lab** (static page, no install): built by `regress site` and deployed to
  GitHub Pages and Vercel from this repo (see [Deploying](#deploying)).
* **Hugging Face Space**: the lab plus *run your own scenario* and *upload your own
  history* (`space/`).

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
distinct facts. REGRESS flags it at cycle 4, with accuracy at 0.83.

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


### Fit the thresholds to your agent

The default tolerances are generic guesses. `calibrate` fits them to histories of cycles you
know were healthy: for each signal it takes a high quantile of the healthy *harmful* moves
from baseline and multiplies by a margin (defaults 99.9 % × 1.5). It never goes below 25 % of
the default, and keeps the default where it has fewer than 10 samples.

```python
from regress import CollapseDetector
from regress.calibrate import calibrate

cal = calibrate([healthy_run_1.history, healthy_run_2.history])
print(cal.summary())
monitor = RegressionMonitor(suite, detector=CollapseDetector(cal.thresholds))
```

```bash
regress calibrate healthy-*.json --out thresholds.json
```

In the simulator, calibrating on 5 healthy runs (seeds 100–104) and testing on unseen seeds:

| | default thresholds | calibrated |
|---|---|---|
| false-alarm cycles, 20 healthy seeds × 20 cycles | 0 / 400 | 0 / 400 |
| alarm lag after true onset, 6 unseen seeds each: collapse | 1 | 0–1 |
| … slow (gentle) collapse | 3–10 | 0–1 |
| … probe leak | 2–3 | 0 |

The calibrated detector alarms on the onset cycle itself, often while accuracy is still at
baseline. The usual caveat applies: calibrate on runs that were already degrading and you
teach it to accept degradation.

### Which memories did it?

`blame` traces a regression to the memories behind it. For the probes that regressed against
cycle 0, it ranks the memories they retrieved by how many regressed probes used them. It also
reports *lift* (how much more often failing probes use a memory than probes in general), the
memory's retrieval-share change and the cycle it first appeared.

```bash
regress blame history.json            # default: the first alarming cycle
```
```
cycle 10: 6 probes regressed against cycle 0
memory           regressed  total   lift    share then→now first seen
m00439                   3     30   2.67    0.0% → 2.7%             5
m00427                   2      8   6.67    0.0% → 0.7%             5
```

In the collapse simulation, the top suspects are the inflated hub memories: their median
importance weight is more than 10× the memory's median. The evidence is correlational, so the
check is to quarantine the suspects and re-run the suite.

### Is the probe suite still useful?

`regress health history.json` classifies every probe as saturated (always passed),
unsolved (never passed), flaky (flips without a trend), regressed, improved or mixed, and
says when to rotate. A suite that is 80 % saturated can catch regressions but can no longer
see improvements.

### Gate CI on it

```bash
regress gate history.json --fail-on alarm --markdown "$GITHUB_STEP_SUMMARY"   # exit 1 on ALARM
regress report history.json --html report.html                                # interactive page
```

or as a GitHub Action:

```yaml
- uses: aabhimittal/Regress@main
  with:
    history: runs/history.json
    report: regress-report.html     # optional
```

`RegressionMonitor(..., on_alarm=fn)` calls `fn(report)` once each time the status *enters*
ALARM, which makes it a paging hook. A sustained alarm pages once, and an exception inside
`fn` is recorded rather than crashing the learning loop.

### Text memories and free-text answers

```python
from regress.adapters import HashingEmbedder, memory_from_texts, text_key

monitor = RegressionMonitor(suite, output_key=text_key())     # "Paris." == " paris"
snapshot = memory_from_texts(ids, texts)                      # dependency-free hashing embedder
snapshot = memory_from_texts(ids, texts, my_sentence_model)   # anything with .embed(list[str])
```

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

An accuracy drop must also be statistically supported. It has to exceed 1.96 × the standard
error of the paired per-probe changes, so one flipped probe in a 20-probe suite is not called
a regression, while 24 flips in a 400-probe suite are (`significance_z=0` turns this off).

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

## Deploying

The lab is one self-contained HTML file. `regress site --out DIR` rebuilds it from the
library, so the numbers on the page always match the code.

| Target | What's in the repo | One-time setup |
|---|---|---|
| **GitHub Pages** | `.github/workflows/pages.yml` builds and deploys on every push to `main` | Settings → Pages → Source: **GitHub Actions** |
| **Vercel** | `vercel.json` (Python venv install → `regress site --out public`) | Import the repo in Vercel; no framework preset, no env vars |
| **Hugging Face Space** | `space/` (Gradio + FastAPI app); `.github/workflows/hf-space.yml` syncs it on every push | Add a repo secret `HF_TOKEN` (write token); optional repo variable `HF_SPACE` (default `abhimittal/regress`) |

Run the Space locally with `pip install -e ".[space]" && python space/app.py`, then open
http://localhost:7860. Without `HF_TOKEN` the sync job logs a notice and skips, so it never
turns CI red.

---

## Limitations

* **Probe suites go stale, and the agent can overfit them.** The held-out split catches
  leakage of the *public* half. It does nothing if the held-out probes themselves leak, for
  example through logs the agent can read. Rotate suites, and keep held-out probes out of
  any data path the agent learns from.
* **The default thresholds are arbitrary until validated across stacks.** They were tuned on
  the bundled simulator. `calibrate` replaces them with tolerances fitted to *your* healthy
  runs. That fixes the scale but not the label: it is only as good as your judgement of
  which runs were healthy.
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
  calibrate.py   fit thresholds to healthy runs
  blame.py       rank the memories behind a regression
  health.py      saturated / unsolved / flaky probe detection
  gate.py        CI verdict + Markdown summary
  adapters.py    text memories: hashing embedder, text output keys
  site.py        build the interactive lab / HTML reports (template in assets/lab.html)
  report.py      terminal tables, sparklines, verdicts
  cli.py         regress demo | show | site | report | gate | calibrate | blame | health
space/           Hugging Face Space (Gradio + FastAPI)
action.yml       GitHub Action wrapping `regress gate`
```

Dependencies: `numpy` only. Python ≥ 3.10.

## Development

```bash
pip install -e ".[test]" ruff
pytest              # ~160 tests, including Hypothesis property tests
ruff check src tests examples
```

MIT licensed.
