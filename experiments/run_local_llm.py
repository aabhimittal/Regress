"""Run the REGRESS real-agent experiment on a local GGUF model (CPU, no API key).

    pip install llama-cpp-python
    python experiments/run_local_llm.py --model qwen2.5-0.5b-instruct-q4_k_m.gguf --out experiments/results/qwen
"""

import argparse
import json
import sys
import time
from pathlib import Path

from regress.llm import CachedBackend, LlamaCppBackend, run_experiment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--people", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--compress-fraction", type=float, default=1.0,
                    help="share of old notes compressed per cycle in the recursive regime (gradual < 1)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = CachedBackend(LlamaCppBackend(args.model, n_threads=args.threads), out / "llm-cache.jsonl")
    t0 = time.time()
    summary = run_experiment(backend, cycles=args.cycles, people=args.people, seed=args.seed, out_dir=out,
                             agent_kwargs={"compress_fraction": args.compress_fraction})
    summary["wall_seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    for regime, r in summary["results"].items():
        print(regime, "first alarm:", r["first_alarm"], "statuses:", r["statuses"], file=sys.stderr)
    print("separates:", summary["separates"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
