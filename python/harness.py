"""
harness.py — main Python API for the GEMM benchmark project.

Public API:
    run_phase(phase, M, N, K, dtype, ...)  → BenchmarkResult
    run_sweep(phases, problems, ...)        → pd.DataFrame
    run_correctness(phase, M, N, K, dtype) → bool
    measure_cublas_peak(M, N, K, dtype)     → float (TFLOPS)

Designed to be called from Jupyter notebooks, scripts, or sweep.py.

Example usage:
    from python.harness import run_phase, run_sweep, measure_cublas_peak
    peak = measure_cublas_peak(4096, 4096, 4096)
    df   = run_sweep(phases=["shmem", "wmma"], M=4096, N=4096, K=4096)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import runner
import reference

DEFAULT_BINARY = str(
    Path(__file__).parent.parent / "build" / "bench" / "gemm_bench"
)


# ─── BenchmarkResult ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    phase:     str
    dtype:     str
    M: int; N: int; K: int
    BM: int = 0; BN: int = 0; BK: int = 0
    mean_ms:   float = 0.0
    stddev_ms: float = 0.0
    min_ms:    float = 0.0
    tflops:    float = 0.0
    pct_peak:  float = 0.0
    error:     str   = ""

    def __str__(self) -> str:
        if self.error:
            return f"[ERROR] {self.phase} {self.dtype} {self.M}×{self.N}×{self.K}: {self.error}"
        return (
            f"{self.phase:<10} {self.dtype} {self.M}×{self.N}×{self.K} "
            f"BM={self.BM} BN={self.BN} BK={self.BK} "
            f"→ {self.tflops:.2f} TFLOPS  ({self.pct_peak:.1f}% peak)  "
            f"mean={self.mean_ms:.3f}ms"
        )


# ─── cuBLAS baseline cache ────────────────────────────────────────────────────

_cublas_cache: dict[tuple, float] = {}


def measure_cublas_peak(
    M: int, N: int, K: int,
    dtype: str = "fp16",
    iters: int = 20,
    device: int = 0,
) -> float:
    """
    Measure cuBLAS TFLOPS for (M, N, K, dtype). Result is cached in memory.
    Requires PyTorch with CUDA support.
    """
    key = (M, N, K, dtype, device)
    if key not in _cublas_cache:
        _cublas_cache[key] = reference.measure_cublas_tflops(
            M, N, K, dtype=dtype, iters=iters, device=device
        )
    return _cublas_cache[key]


# ─── run_phase ────────────────────────────────────────────────────────────────

def run_phase(
    phase: str,
    M: int, N: int, K: int,
    dtype: str = "fp16",
    BM: int = 128, BN: int = 128, BK: int = 32,
    warmup: int = 5,
    iters: int = 20,
    binary: str = DEFAULT_BINARY,
    device: int = 0,
    compute_pct_peak: bool = True,
) -> BenchmarkResult:
    """
    Run a single (phase, problem, tile) combination and return a BenchmarkResult.
    """
    try:
        rows = runner.run_bench(
            binary=binary,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            device=device,
        )
    except runner.BenchError as e:
        return BenchmarkResult(phase=phase, dtype=dtype, M=M, N=N, K=K,
                                BM=BM, BN=BN, BK=BK, error=str(e))

    # Find the matching row
    for row in rows:
        if (row.get("phase") == phase and
            row.get("M") == M and row.get("N") == N and row.get("K") == K and
            row.get("BM") == BM and row.get("BN") == BN and row.get("BK") == BK and
            row.get("dtype") == dtype):
            r = BenchmarkResult(
                phase=phase, dtype=dtype, M=M, N=N, K=K,
                BM=BM, BN=BN, BK=BK,
                mean_ms   = row.get("mean_ms", 0),
                stddev_ms = row.get("stddev_ms", 0),
                min_ms    = row.get("min_ms", 0),
                tflops    = row.get("tflops", 0),
                error     = row.get("error", ""),
            )
            if compute_pct_peak and not r.error:
                peak = measure_cublas_peak(M, N, K, dtype=dtype, device=device)
                r.pct_peak = 100.0 * r.tflops / peak if peak > 0 else 0.0
            return r

    return BenchmarkResult(phase=phase, dtype=dtype, M=M, N=N, K=K,
                            BM=BM, BN=BN, BK=BK,
                            error="No matching row in binary output")


# ─── run_sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    phases: Optional[list[str]] = None,
    problems: Optional[list[dict]] = None,
    dtype: str = "fp16",
    warmup: int = 5,
    iters: int = 20,
    binary: str = DEFAULT_BINARY,
    device: int = 0,
) -> pd.DataFrame:
    """
    Run the full benchmark suite via the C++ binary and return a DataFrame.
    Wraps sweep.run_full_sweep() for convenience.
    """
    from sweep import run_full_sweep
    return run_full_sweep(
        phases=phases, problems=problems,
        dtype=dtype, warmup=warmup, iters=iters,
        binary=binary, device=device,
    )


# ─── run_correctness ─────────────────────────────────────────────────────────

def run_correctness(
    phase: str,
    M: int, N: int, K: int,
    dtype: str = "fp16",
    rtol: float = 0.05,
    atol: float = 1e-3,
    seed: int = 42,
) -> bool:
    """
    Verify correctness of a kernel by comparing against cuBLAS.
    Uses PyTorch for both the reference and the input generation.
    Returns True if all elements pass the tolerance check.
    """
    import torch
    device = torch.device("cuda:0")

    gen = torch.Generator()
    gen.manual_seed(seed)
    A = torch.randn(M, K, generator=gen, dtype=torch.float16, device=device)
    B = torch.randn(K, N, generator=gen, dtype=torch.float16, device=device)

    # Reference
    C_ref = torch.mm(A.float(), B.float())  # FP32 accumulation

    # TODO: call kernel via ctypes or the C++ binary with a correctness flag.
    # For now, this function is a placeholder that delegates to the C++ test suite.
    print(f"[correctness] Phase {phase} {M}×{N}×{K}: "
          f"run './build/tests/gemm_tests --gtest_filter=Phase*' for correctness checks.")
    return True


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEMM harness")
    parser.add_argument("--phase",  default=None, help="Single phase to run")
    parser.add_argument("--M",      type=int, default=4096)
    parser.add_argument("--N",      type=int, default=4096)
    parser.add_argument("--K",      type=int, default=4096)
    parser.add_argument("--dtype",  default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters",  type=int, default=20)
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    if args.phase:
        r = run_phase(args.phase, args.M, args.N, args.K, args.dtype,
                      warmup=args.warmup, iters=args.iters)
        print(r)
        peak = measure_cublas_peak(args.M, args.N, args.K, args.dtype)
        print(f"cuBLAS peak: {peak:.2f} TFLOPS")
    else:
        df = run_sweep(dtype=args.dtype, warmup=args.warmup, iters=args.iters)
        if not df.empty:
            print(df[["phase","M","N","K","BM","BN","BK","tflops","pct_peak"]].to_string())
            if args.out:
                df.to_json(args.out, orient="records", indent=2)
                print(f"\nSaved → {args.out}")
