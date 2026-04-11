"""
Latency benchmark: multi-agent system vs single-chain baseline.

Runs N queries through both pipelines and prints a summary table showing
per-node latencies, total end-to-end time, and the % speedup.

Usage:
    python scripts/benchmark.py --golden-set ./data/golden_set.json --n 20

    # Run a quick 5-query smoke test:
    python scripts/benchmark.py --n 5
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from app.agents.coordinator import run_query as run_multiagent
from scripts.baseline import run_baseline

logging.basicConfig(level="WARNING")  # suppress INFO noise during benchmark
logger = logging.getLogger(__name__)

# Hardcoded fallback queries if no golden set is supplied
_DEFAULT_QUERIES = [
    "What is the maximum loan-to-value ratio for residential mortgages?",
    "How is Tier 1 capital defined under the regulatory framework?",
    "What are the quarterly reporting requirements under Basel III?",
    "What does the policy state regarding insider trading prohibitions?",
    "What are the anti-money-laundering obligations for broker-dealers?",
]


def _total_ms(state: dict) -> float:
    """Sum all node latencies recorded in state."""
    return sum(state.get("node_latencies", {}).values())


def _run_suite(queries: list[str], runner, label: str) -> list[float]:
    """Run all queries through runner, return list of total latencies in ms."""
    latencies = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        try:
            result = runner(query=q)
            elapsed = (time.perf_counter() - t0) * 1000
            # Prefer summed node latencies (more precise) over wall time
            node_total = _total_ms(result)
            latencies.append(node_total if node_total > 0 else elapsed)
        except Exception as exc:
            logger.warning("[%s] query %d failed: %s", label, i, exc)
            latencies.append(float("nan"))
        print(f"  [{label}] {i}/{len(queries)} done", end="\r", flush=True)
    print()
    return latencies


def _stats(values: list[float]) -> dict:
    clean = [v for v in values if not (v != v)]  # drop NaN
    if not clean:
        return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    sorted_v = sorted(clean)
    p95_idx = int(len(sorted_v) * 0.95)
    return {
        "mean": round(statistics.mean(clean), 1),
        "p50":  round(sorted_v[len(sorted_v) // 2], 1),
        "p95":  round(sorted_v[min(p95_idx, len(sorted_v) - 1)], 1),
        "min":  round(sorted_v[0], 1),
        "max":  round(sorted_v[-1], 1),
    }


def main(golden_set_path: str | None, n: int):
    # ── Load queries ──────────────────────────────────────────────────────────
    if golden_set_path and Path(golden_set_path).exists():
        with open(golden_set_path, encoding="latin-1") as f:
            golden = json.load(f)
        queries = [item["question"] for item in golden[:n]]
    else:
        queries = _DEFAULT_QUERIES[:n]

    print(f"\nBenchmarking {len(queries)} queries ...\n")

    # ── Run baseline (sequential) ─────────────────────────────────────────────
    print("Running single-chain baseline ...")
    baseline_latencies = _run_suite(queries, run_baseline, "baseline")

    # ── Run multi-agent ───────────────────────────────────────────────────────
    print("Running multi-agent system ...")
    multiagent_latencies = _run_suite(queries, run_multiagent, "multi-agent")

    # ── Compute stats ─────────────────────────────────────────────────────────
    b = _stats(baseline_latencies)
    m = _stats(multiagent_latencies)

    speedup_mean = round((b["mean"] - m["mean"]) / b["mean"] * 100, 1) if b["mean"] else 0
    speedup_p95  = round((b["p95"]  - m["p95"])  / b["p95"]  * 100, 1) if b["p95"]  else 0

    # ── Print results ─────────────────────────────────────────────────────────
    col = 22
    print("\n" + "=" * 62)
    print(f"{'Metric':<{col}} {'Baseline (ms)':>18} {'Multi-agent (ms)':>18}")
    print("-" * 62)
    for key in ("mean", "p50", "p95", "min", "max"):
        print(f"{key:<{col}} {b[key]:>18.1f} {m[key]:>18.1f}")
    print("=" * 62)
    print(f"{'Speedup (mean)':<{col}} {speedup_mean:>17.1f}%")
    print(f"{'Speedup (p95)':<{col}} {speedup_p95:>17.1f}%")
    print("=" * 62)

    if speedup_mean >= 50:
        print(f"\n✓ Mean speedup {speedup_mean}% — consistent with resume claim (≥50%)")
    else:
        print(f"\n⚠  Mean speedup {speedup_mean}% — below 50% target; check parallelism or query mix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-set", default="./data/golden_set.json",
        help="Path to golden_set.json (default: ./data/golden_set.json)",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of queries to benchmark (default: 20)",
    )
    args = parser.parse_args()
    main(args.golden_set, args.n)
