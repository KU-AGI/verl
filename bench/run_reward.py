#!/usr/bin/env python3
"""
Unified entry point for offline reward evaluation across all benchmarks.

Dispatches to the appropriate benchmark-specific reward script based on --bench.

Supported benchmarks:
    geneval, dpgbench, t2icompbench, tiif

Usage:
    # Single benchmark
    python -m bench.run_reward --bench geneval \
        --data_dir /data/mllm/data/for_down/geneval \
        --vqa_json /data/mllm/data/geneval_dsg_final.json \
        --output_dir /data/mllm/data/for_down/geneval/reward_results \
        --step 0 1 2 --tasks 1 2 3

    # Multiple benchmarks at once
    python -m bench.run_reward --bench geneval dpgbench t2icompbench \
        --data_dir /data/mllm/data/for_down/{bench} \
        --vqa_json /data/mllm/data/{bench}_dsg_final.json \
        --output_dir /data/mllm/data/for_down/{bench}/reward_results \
        --step 0 1 2 --tasks 1 2 3

    # Benchmark-specific args (tiif)
    python -m bench.run_reward --bench tiif \
        --data_dir /data/mllm/data/for_down/tiif \
        --vqa_json /data/mllm/data/tiif_dsg_final.json \
        --output_dir /data/mllm/data/for_down/tiif/reward_results \
        --step 0 1 2 --tasks 1 2 3 \
        --desc long short

    # Benchmark-specific args (t2icompbench)
    python -m bench.run_reward --bench t2icompbench \
        --data_dir /data/mllm/data/for_down/t2icompbench \
        --vqa_json "" \
        --output_dir /data/mllm/data/for_down/t2icompbench/reward_results \
        --step 0 1 --tasks 1 2 3 \
        --categories color spatial

    # List available benchmarks
    python -m bench.run_reward --list

Adding a new benchmark:
    1. Create bench/reward_<name>.py with parse_args() and main()
    2. Add an entry to BENCH_REGISTRY in this file
"""

import argparse
import importlib
import sys
import os

# ──────────────────────────────────────────────────────────────
# Benchmark registry
# ──────────────────────────────────────────────────────────────
# Each entry maps bench name -> module path (importable via importlib).
# The module must expose:
#   - parse_args() -> argparse.Namespace
#   - main()       -> None  (uses parse_args internally)
#
# To add a new benchmark, just add one line here.

BENCH_REGISTRY = {
    "geneval":       "bench.reward_geneval",
    "dpgbench":      "bench.reward_dpgbench",
    "t2icompbench":  "bench.reward_t2icompbench",
    "tiif":          "bench.reward_tiif",
}


def list_benchmarks():
    print("Available benchmarks:")
    for name, module_path in sorted(BENCH_REGISTRY.items()):
        print(f"  {name:20s}  ({module_path})")


def run_single_bench(bench_name: str):
    """Import and run a single benchmark's main()."""
    if bench_name not in BENCH_REGISTRY:
        print(f"[ERROR] Unknown benchmark: '{bench_name}'")
        print(f"Available: {', '.join(sorted(BENCH_REGISTRY.keys()))}")
        sys.exit(1)

    module_path = BENCH_REGISTRY[bench_name]
    print(f"\n{'#'*60}")
    print(f"# Benchmark: {bench_name}")
    print(f"# Module:    {module_path}")
    print(f"{'#'*60}\n")

    mod = importlib.import_module(module_path)
    mod.main()


def main():
    # Two-phase parsing:
    #   Phase 1: extract --bench and --list from argv (known args only)
    #   Phase 2: pass remaining argv to the benchmark's own parse_args via sys.argv

    parser = argparse.ArgumentParser(
        description="Unified reward evaluation entry point",
        add_help=False,  # Don't consume -h; let it fall through to bench parsers
    )
    parser.add_argument("--bench", type=str, nargs="+", default=None,
                        help=f"Benchmark(s) to run: {', '.join(sorted(BENCH_REGISTRY.keys()))}")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")

    known, remaining = parser.parse_known_args()

    if known.list:
        list_benchmarks()
        return

    if known.bench is None:
        # No --bench provided: show help
        print("Usage: python -m bench.run_reward --bench <name> [benchmark args...]")
        print("       python -m bench.run_reward --list")
        print(f"\nAvailable benchmarks: {', '.join(sorted(BENCH_REGISTRY.keys()))}")
        print("\nUse --bench <name> -h to see benchmark-specific arguments.")
        return

    # Rewrite sys.argv so each benchmark's parse_args() sees only
    # the remaining (benchmark-specific) arguments.
    sys.argv = [sys.argv[0]] + remaining

    for bench_name in known.bench:
        run_single_bench(bench_name)


if __name__ == "__main__":
    main()
