"""Broaden the real-model evidence: models × tasks × seeds × regimes, with and without the guard.

Each job (model, task, seed) runs these regimes on identical worlds:

    faithful          store facts verbatim (control)
    full              self-train + compress all old notes every cycle (abrupt damage)
    full+guard        same, but every stage is tested before it is committed (strict guard)
    full+guard+learn  guard that always commits observations and also protects recent facts
    gradual           compress only the oldest 25 % per cycle (gradual damage)
    gradual+guard     gradual, guarded (only on jobs marked "extended")

25 % of the probe questions are held back in an audit suite the guard never sees.
All model calls are cached under each job directory, so an interrupted run resumes and a
finished run can be replayed with ``regress experiment --backend replay``.

    python experiments/run_matrix.py --models-dir ~/models --queue all --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from regress.llm import CachedBackend, LlamaCppBackend, run_experiment

MODELS = {
    "qwen2.5-0.5b": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "llama3.2-1b": "llama-3.2-1b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
}

QUEUES = {
    "A": [("qwen2.5-0.5b", "facts", 0, True), ("qwen2.5-0.5b", "facts", 1, False), ("qwen2.5-0.5b", "facts", 2, False)],
    "B": [("llama3.2-1b", "facts", 0, False), ("qwen2.5-0.5b", "multihop", 0, False), ("qwen2.5-1.5b", "facts", 0, False)],
}
QUEUES["all"] = QUEUES["A"] + QUEUES["B"]  # one process at a time: parallel llama.cpp runs oversubscribe the CPU


def regimes(task: str, extended: bool) -> list:
    hops = {"hops": 2} if task == "multihop" else {}
    out = [
        ("faithful", "faithful", {"agent_kwargs": hops}),
        ("full", "recursive_summary", {"agent_kwargs": hops}),
        ("full+guard", "recursive_summary", {"guard": "strict", "agent_kwargs": hops | {"compress_chunks": 2}}),
        ("full+guard+learn", "recursive_summary",
         {"guard": "strict", "guard_learning": True, "agent_kwargs": hops | {"compress_chunks": 2}}),
        ("gradual", "recursive_summary", {"agent_kwargs": hops | {"compress_fraction": 0.25}}),
    ]
    if extended:
        out.append(("gradual+guard", "recursive_summary",
                    {"guard": "strict", "agent_kwargs": hops | {"compress_fraction": 0.25, "compress_chunks": 2}}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--queue", choices=sorted(QUEUES), required=True)
    ap.add_argument("--out", default="experiments/results/matrix")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--people", type=int, default=16)
    args = ap.parse_args()
    for model, task, seed, extended in QUEUES[args.queue]:
        out = Path(args.out) / f"{model}-{task}-s{seed}"
        if (out / "summary.json").exists():
            print(f"skip {out} (done)", file=sys.stderr)
            continue
        out.mkdir(parents=True, exist_ok=True)
        backend = CachedBackend(LlamaCppBackend(str(Path(args.models_dir) / MODELS[model]), n_threads=args.threads),
                                out / "llm-cache.jsonl")
        t0 = time.time()
        summary = run_experiment(backend, regimes=regimes(task, extended), out_dir=out, cycles=args.cycles,
                                 people=args.people, seed=seed, task=task, audit_fraction=0.25, ablate=False)
        summary["model"], summary["wall_seconds"] = model, round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"done {out} in {summary['wall_seconds']}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
