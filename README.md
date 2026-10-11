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

## Does it work on a real agent?

The simulator was built to collapse, so it cannot answer this. `regress.llm` runs REGRESS on
an agent driven by a **real language model**: a note-memory QA agent over a *fictional* fact
world (invented people, cities, professions and pets), so the model cannot answer from
pre-training. Two regimes run on identical facts:

* **faithful**: new facts are stored verbatim. This is the control.
* **recursive_summary**: the agent answers questions on its own and writes its answers back
  as notes (no feedback), then has the model compress clusters of old notes into one short
  note. Next cycle it compresses the summaries again.

Recorded run: **Qwen2.5-0.5B-Instruct (Q4 GGUF) on CPU via llama.cpp**, 20 people (60
probes), 8 cycles, 4 new people per cycle. Confidence is the model's own token probability.

| | cycle 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| faithful: status | · | · | · | · | · | · | · | · | · |
| faithful: accuracy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| recursive: status | · | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ |
| recursive: accuracy | 1.00 | 0.45 | 0.13 | 0.00 | 0.02 | 0.00 | 0.03 | 0.07 | 0.00 |
| recursive: effective rank | 18.1 | 20.5 | 17.4 | 16.4 | 15.9 | 15.3 | 13.7 | 13.9 | 14.5 |
| recursive: mean confidence | 0.98 | 0.78 | 0.81 | 0.77 | 0.75 | 0.87 | 0.84 | 0.82 | 0.84 |

What it shows:

* **REGRESS separates the regimes on a real model.** The control produced zero signals in
  9 cycles. The bad regime alarmed at cycle 1, the first bad cycle. Bisection found cycle 1
  in 5 evaluations, and the onset estimate is cycle 1.
* **The collapse signature from the design did *not* appear.** Damage was immediate, not
  gradual. Accuracy fell from 1.00 to 0.45 at the first compression. Diversity briefly *rose*
  (merged summaries mix vocabulary) before falling, and expressed confidence *fell*. The
  alarm came from the multi-category rule (accuracy, retrieval entropy, overconfidence).
  Requiring the combination was more robust than the specific "diversity ↓ + confidence ↑"
  pattern the thesis predicted.
* **Overconfidence exploded, but read it carefully.** Accuracy is 0 while confidence stays
  around 0.8. Many answers are a confident "unknown". Token probability measures certainty
  of the *output*, so a confident abstention scores as overconfident. Agents that abstain
  need a scorer or confidence channel that treats "unknown" separately.
* **Ablation correctly refused the blame.** Removing the top 4 suspects restored 1 of 33
  regressed probes, against 2 for random controls (verdict: refuted). The damage was
  information destroyed by compression, not a harmful memory that removal could fix.

**Gradual variant** (`--compress-fraction 0.25`: compress only the oldest quarter of notes
each cycle), same model and facts:

| | cycle 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| status | · | ~ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ | ▲ |
| accuracy | 1.00 | 0.90 | 0.75 | 0.65 | 0.62 | 0.55 | 0.42 | 0.37 | 0.30 |
| retrieval entropy | 0.97 | 0.89 | 0.80 | 0.77 | 0.77 | 0.76 | 0.75 | 0.74 | 0.74 |
| effective rank | 18.1 | 19.6 | 22.1 | 23.4 | 25.6 | 26.5 | 26.3 | 26.4 | 26.6 |
| mean confidence | 0.98 | 0.94 | 0.93 | 0.90 | 0.91 | 0.88 | 0.89 | 0.85 | 0.85 |

REGRESS warns at cycle 1 and alarms at cycle 2, while accuracy is still 0.75 on its way down
to 0.30. Bisection pins cycle 2 and the onset estimate is cycle 1, which is exactly where the
bad consolidation started. **Homogenisation showed up in retrieval, not in embedding
diversity.** A few summaries absorb more and more queries, so retrieval entropy falls from
0.97 to 0.74. Effective rank *rises*, because summaries combine distinct vocabulary while
new facts keep arriving. If you monitor only embedding diversity, you miss this failure.

Reproduce:

```bash
pip install -e ".[llm]"                     # llama-cpp-python
python experiments/run_local_llm.py --model qwen2.5-0.5b-instruct-q4_k_m.gguf --out runs/qwen
regress experiment --backend replay --replay-from experiments/results/qwen2.5-0.5b-q4 --out /tmp/replay
regress experiment --backend anthropic --out runs/claude      # needs Anthropic credentials
```

The recorded model outputs (`llm-cache.jsonl`) are committed. `--backend replay` re-runs the
whole experiment bit-for-bit without the model, and CI does exactly that.

### Broader evidence: 3 models, 2 tasks, 3 seeds

One model, one task and one seed could be a coincidence, so `experiments/run_matrix.py` repeats
the experiment across **Qwen2.5‑0.5B, Qwen2.5‑1.5B and Llama‑3.2‑1B** (Q4 GGUF, CPU), seeds
0–2, and a second task. In **multihop**, "X works at C" and "C is based in Z" are separate
notes, and "In which city does X work?" needs both, so consolidation can break the chain
without dropping either fact. Each job: 16 people, 6 cycles, 6 regimes on identical worlds. A
quarter of the questions are held back in an **audit** suite that only reports. *New facts*
is accuracy at the end on facts that arrived **during** the run, scored on a fixed half the
guard never uses.

| model · task · seed | control | full: alarm at (acc then → end) | gradual: alarm at (acc then → end) | guard: old / audit / new facts | guard+learn: old / audit / new facts · compressions kept |
|---|---|---|---|---|---|
| llama3.2-1b · facts · s0 | quiet | cycle 1 (0.14 → 0.03) | cycle 1 (0.78 → 0.19) | 0.94 / 1.00 / 0.37 | 0.97 / 0.92 / 0.95 · 0% |
| qwen2.5-0.5b · facts · s0 | quiet | cycle 1 (0.47 → 0.00) | cycle 2 (0.75 → 0.36) | 0.94 / 0.92 / 0.33 | 1.00 / 1.00 / 1.00 · 0% |
| qwen2.5-0.5b · facts · s1 | quiet | cycle 1 (0.53 → 0.00) | cycle 2 (0.75 → 0.42) | 0.92 / 0.92 / 0.18 | 1.00 / 1.00 / 1.00 · 0% |
| qwen2.5-0.5b · facts · s2 | quiet | cycle 1 (0.33 → 0.14) | cycle 1 (0.78 → 0.28) | 0.94 / 0.92 / 0.35 | 1.00 / 1.00 / 1.00 · 0% |
| qwen2.5-0.5b · multihop · s0 | quiet (2× watch) | cycle 1 (0.12 → 0.08) | cycle 1 (0.12 → 0.08) | 0.75 / 0.62 / 0.21 | 0.96 / 0.88 / 0.84 · 0% |
| qwen2.5-1.5b · facts · s0 | quiet | cycle 1 (0.19 → 0.06) | cycle 1 (0.86 → 0.28) | 0.94 / 0.92 / 0.72 | 1.00 / 1.00 / 1.00 · 0% |

Old / audit = accuracy at the end on the gate suite / held-out audit suite (baseline 1.00,
except Llama 0.97 / 0.92 and multihop 0.83 / 0.88). Full per-cycle table:
[`experiments/results/MATRIX.md`](experiments/results/MATRIX.md). Every job replays
bit-for-bit in the tests.

What held across all 6 jobs:

* **No false alarms, every bad regime caught.** The control never alarmed. On multihop it
  raised a single-signal WATCH for two cycles: confidence rose *together with* accuracy
  (0.83 → 1.00), a benign change the two-category rule kept from alarming. *Full* alarmed at
  cycle 1 in 6/6 jobs, and *gradual* alarmed at cycle 1–2 in 6/6.
* **Early warning is real when damage is gradual, and it is a property of the damage, not
  of the policy.** On facts, *gradual* had lost 0.14–0.25 accuracy at the first alarm and
  0.58–0.78 by the end. On multihop, the same "compress the oldest 25 %" policy was
  *abrupt* (0.83 → 0.12 in one cycle). The oldest notes are the company → city notes that
  every two-hop chain runs through, so a small edit hit a hub.
* **Monitoring cannot prevent abrupt damage. Gating can.** That is the next section.

Scope still: small open models, synthetic fictional-fact tasks, a note-memory agent, and
one seed for the non-Qwen‑0.5B jobs. This is evidence about this learning loop across
models, not a benchmark of how production agents fail.

---

## Stop the damage before it lands: the consolidation guard

Monitoring after the fact cannot warn about damage that happens in a single step. In the
*full* regime above, accuracy had already fallen to 0.12–0.53 (from 0.83–1.00) at the first check
that could see it. Early warning exists only when damage accumulates. The fix is to move
the check **in front of the commit**, which is what regression testing means in CI:

```python
from regress.guard import ConsolidationGuard

guard = ConsolidationGuard(monitor, block_on=("ALARM", "WATCH"))   # baseline already recorded
stages = [("observe", ingest_new_data), ("compress", consolidate)]  # each fn(agent) mutates it
agent, report = guard.run_cycle(agent, cycle, stages)              # the agent you keep
guard.summary()   # decisions, rejected, rejection_rate, accept_rate_by_stage
```

Each stage is applied to a **copy** of the agent and run against the frozen suite. The copy
is kept only if it passes; otherwise the previous agent is kept and the rejection is logged.
Splitting a cycle into stages means a bad stage costs only that stage. Every candidate is
compared with the **baseline**, not the previous cycle, so many small accepted steps cannot
add up past the tolerance either. With a response cache, the committed report is free,
because the copy that passed was already evaluated.

**Two failure modes of a naive guard, both found on real models, and the options that fix them:**

1. **It refuses to learn.** A frozen suite protects only what the agent knew at cycle 0.
   New observations crowd retrieval for old facts, so a strict guard rejected the
   *observe* stage in up to half the cycles. A compression that destroys recent facts
   passes, because no probe asks about them. Across the 6 jobs, the plain guard kept old
   facts at 0.75–0.94 but answered only **0.18–0.72** of the facts that arrived during the
   run (control: 0.84–1.00). On Qwen‑0.5B that is no better than not guarding at all
   (0.21–0.28).
2. **It can lock up.** If data that must be accepted pushes the agent over the line, every
   later candidate fails against the baseline, and nothing can be consolidated again.

`commit_always=("observe",)` commits data-bearing stages unconditionally (logged as
`FORCED`). `retention=fn` also rejects a stage that lowers accuracy on *recently learned*
knowledge by more than `retention_tolerance`. Once a forced stage has crossed the line,
candidates are judged against the committed agent: no worse status and no lower accuracy.
The recorded report still shows the regression. In the experiment harness this is
`guard="strict", guard_learning=True` (CLI: `--guard learn`). The retention check uses half
of the newly learned facts, and the other half is held out for scoring.

**What the guard cannot do:** make consolidation good. The learning-aware guard matched the
control in every job: old facts 0.96–1.00, new facts 0.84–1.00, audit unchanged. It got
there by rejecting **every** compression in all 6 jobs. Memory grew like the
no-compression control, and on Qwen‑0.5B past it, because the agent's self-written notes
were accepted (153 vs 120 notes). The guard turns a bad consolidation policy into *no* consolidation and tells
you so through `accept_rate_by_stage`. Read a high rejection rate as an alarm about the
policy, not as success. It also optimises against the gate suite. Keep an **audit suite**
the guard never sees (`audit_fraction=0.25` in the harness) and compare the two. If the
gate stays green while the audit falls, the guard is overfitting its own test.

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

### Measure diversity with your own ruler

Diversity used to be computed from the agent's own embeddings, which lets the agent grade
its own exam. If a learning cycle swaps the embedding model, the numbers move for reasons
unrelated to memory health. Give the snapshot the memory **texts** and REGRESS re-embeds
them with a fixed embedder it owns:

```python
MemorySnapshot(ids, agent_vectors, texts=memory_texts)   # ruler="auto" picks "reference"
RegressionMonitor(suite, ruler="reference")               # or "agent", or any obj with .embed()
```

Each report records which ruler it used, and a mid-run ruler change is reported as
not comparable. `embedding_drift` (1 − cosine between a memory's old and new vector, by id)
raises a weak `embedding_changed` signal when the agent's own space moves. In the tests,
swapping the agent's embedder for a 2-D one creates a false "collapse" on the agent ruler
and nothing on the reference ruler.

### Uncertainty, long horizons, and a negative result

* **Standard errors on every rate.** Effective rank and pairwise spread are bootstrapped over
  memories, retrieval entropy over probes, and confidence uses its sampling SE. Each signal
  must clear `significance_z × SE` as well as its tolerance, so a 4-memory store cannot
  "collapse" by losing one item.
* **Horizon-aware calibration.** Checking 12 signals every night for a year is about 4,400
  tests. `calibrate(..., horizon=365, alpha=0.05)` raises each tolerance to a Gaussian tail
  estimate at the Bonferroni quantile 1 − α / (horizon × signals).
* **CUSUM is available but off by default, because it lost the test.** `CusumConfig` adds a
  sequential test that accumulates sub-threshold harm. On 100-cycle simulator runs:

  | configuration | false-alarm cycles, 5 healthy × 100 cycles | gentle "creep" collapse caught (cycles after onset) |
  |---|---|---|
  | default thresholds | 0 | never, in 4 of 4 runs |
  | calibrated per-cycle (99.9 % × 1.5) | 0 | 2, 2, 5, 6 |
  | calibrated + horizon (H = 100) | 0 | 4, 2, 5, 6 |
  | horizon + CUSUM (k = 0.25, h calibrated) | **30** | 4, 2, 5, 6 |

  CUSUM assumes healthy moves are independent noise. Healthy memory here *drifts*
  persistently, so CUSUM accumulates healthy drift and fires on it, while catching nothing
  earlier than calibration already does. Use it only if your healthy cycles look like
  stationary noise. `calibrate_cusum` sets `h` from your own healthy runs.

### Confirm blame by ablation

`blame` names suspects. `confirm_blame` tests them: it removes the suspects from a copy of
the agent (`agent.without(ids)`), re-runs the regressed probes, and compares the result with
random control sets of the same size drawn from memories those probes also used. It also
counts collateral damage on probes that still passed.

```python
from regress.ablation import confirm_blame
res = confirm_blame(monitor, checkpoint_agent, cycle=12, top=5, controls=4)
res.verdict   # "confirmed" | "refuted" | "inconclusive"
```

In the collapse simulation at cycle 12, removing the 5 suspects recovers 9 of 17 regressed
probes, against 0–1 for each random control (confirmed). The cost is 3 of 40 previously
passing probes broken. When regressions are not caused by anything in memory, the verdict
is *refuted*. Ablation answers "is a memory that is present doing harm?". It cannot recover
information that consolidation already destroyed (see the real-agent results).

### Pay less per check

* **Response cache.** If the agent exposes `state_fingerprint()`, a response depends only on
  (state, probe), so `RegressionMonitor(..., cache=ResponseCache("cache.jsonl"))` makes
  re-testing an unchanged checkpoint free. Bisection replays are the main beneficiary: a
  7-step bisection over 21 checkpoints was served almost entirely from cache (1,200–1,440
  cache hits per run). This assumes deterministic answers (temperature 0).
* **Screening.** `RegressionMonitor(..., screen=40)` runs a fixed, stratified 40-probe subset
  (by held-out split and first tag) on cycles after the baseline. It escalates to the full
  suite when that subset shows any signal without the significance gate, or any probe error.
  Over 20-cycle simulator runs it saved 68–71 % of probe calls on healthy agents, and alarmed
  on exactly the same cycle as full evaluation in all 12 runs (healthy, collapse, slow, leak
  × 3 seeds). **Known limit:** a regression confined to probes outside the subset is
  invisible on a screened cycle. A test documents it. Stratify with tags and run `full=True`
  periodically.

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
  signature. The real-agent matrix (see *Broader evidence*) is the evidence that matters:
  3 small open models, 2 synthetic tasks, a note-memory agent. Production agents, tool use
  and long horizons are untested.
* **Monitoring cannot warn about one-step damage.** If one update destroys half the
  knowledge, the first check that can see it comes after it. Use the consolidation guard
  to test *before* committing.
* **The guard protects; it does not improve.** With small models, the learning-aware
  guard rejected 100 % of compressions in all 6 jobs, so memory grew like the
  no-compression control (or past it, with self-notes). A plain guard on a frozen suite
  keeps old knowledge by refusing new knowledge. Use `commit_always` and `retention`, and
  watch `accept_rate_by_stage`. Each guarded stage costs a suite evaluation (about 3× the
  probe calls of an unguarded run here).
* **Expressed confidence is only as meaningful as the agent's self-report.** Agents without
  a confidence channel lose half of the collapse signature and fall back on the
  multi-category rule.
* **The reference ruler is only as good as its embedder.** The default hashing embedder
  is deterministic and dependency-free. It sees lexical, not semantic, similarity, so two
  paraphrases of one fact look diverse to it. Pass a sentence-embedding model as `ruler=`
  for semantic diversity, and keep that model fixed for the life of the history.
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
  ablation.py    confirm blame suspects by removing them (agent.without)
  guard.py       ConsolidationGuard: test each learning stage on a copy before committing
  cache.py       response cache keyed on (agent state, probe)
  llm/           real-agent harness: backends (llama.cpp, Claude, rules), fact world, note agent, experiment
  cli.py         regress demo | show | site | report | gate | calibrate | blame | health | experiment
space/           Hugging Face Space (Gradio + FastAPI)
action.yml       GitHub Action wrapping `regress gate`
```

Dependencies: `numpy` only. Python ≥ 3.10.

## Development

```bash
pip install -e ".[test]" ruff
pytest              # ~230 tests, including Hypothesis property tests
ruff check src tests examples
```

MIT licensed.
