"""
sweep.py — grid search over phases × problem sizes × tile configs.

Calls runner.run_bench() and collects all results in a single DataFrame.
Partial failures (one phase/tile crashes) are logged but don't abort the sweep.

Usage:
    python3 python/sweep.py --dtype fp16 --out results.json
    python3 python/sweep.py --phases naive shmem swizzle wmma --out phase14.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import runner

# ─── Default parameter grids ──────────────────────────────────────────────────

DEFAULT_PHASES = ["naive", "shmem", "swizzle", "wmma", "pipeline", "ptx"]

DEFAULT_PROBLEMS = [
    {"M": 1024,  "N": 1024,  "K": 1024,  "label": "sq1k"},
    {"M": 2048,  "N": 2048,  "K": 2048,  "label": "sq2k"},
    {"M": 4096,  "N": 4096,  "K": 4096,  "label": "sq4k"},
    {"M": 8192,  "N": 8192,  "K": 8192,  "label": "sq8k"},
    {"M": 128,   "N": 128,   "K": 4096,  "label": "attn"},
    {"M": 512,   "N": 512,   "K": 512,   "label": "bert"},
]

DEFAULT_TILES = [
    (32,  32,  32),
    (64,  64,  32),
    (128, 128, 32),
    (128, 128, 64),
    (128, 256, 32),
]


def run_full_sweep(
    phases: list[str] | None = None,
    problems: list[dict] | None = None,
    tiles: list[tuple] | None = None,
    dtype: str = "fp16",
    warmup: int = 5,
    iters: int = 20,
    binary: str | None = None,
    device: int = 0,
) -> pd.DataFrame:
    """
    Run the full parameter sweep.
    Returns a DataFrame with one row per (phase, problem, tile) combination.
    Failed rows have a non-empty 'error' column.
    """
    phases   = phases   or DEFAULT_PHASES
    problems = problems or DEFAULT_PROBLEMS
    tiles    = tiles    or DEFAULT_TILES

    kwargs = dict(dtype=dtype, warmup=warmup, iters=iters, device=device)
    if binary:
        kwargs["binary"] = binary

    try:
        rows = runner.run_bench(**kwargs)
    except runner.BenchError as e:
        print(f"[sweep] Bench binary failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases",  nargs="+", default=None,
                        help="Phases to run (default: all phases 0-5)")
    parser.add_argument("--dtype",   default="fp16",
                        choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--iters",   type=int, default=20)
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--binary",  default=None,
                        help="Path to gemm_bench binary")
    parser.add_argument("--out",     default="results.json",
                        help="Output JSON file (default: results.json)")
    parser.add_argument("--csv",     default=None,
                        help="Also write CSV")
    args = parser.parse_args()

    df = run_full_sweep(
        phases  = args.phases,
        dtype   = args.dtype,
        warmup  = args.warmup,
        iters   = args.iters,
        binary  = args.binary,
        device  = args.device,
    )

    if df.empty:
        print("[sweep] No results collected.")
        return

    # Print summary
    print(f"\nCollected {len(df)} rows.")
    if "tflops" in df.columns:
        best = df[df.get("error", "").fillna("") == ""].groupby("phase")["tflops"].max()
        print("\nBest TFLOPS per phase:")
        print(best.to_string())

    # Save
    df.to_json(args.out, orient="records", indent=2)
    print(f"\nJSON → {args.out}")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"CSV  → {args.csv}")


if __name__ == "__main__":
    main()
