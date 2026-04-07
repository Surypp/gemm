"""
reference.py — CPU (NumPy) and GPU (PyTorch/cuBLAS) reference implementations.

Two use cases:
  1. Correctness check: compare kernel output to a trusted reference.
  2. Baseline timing: measure cuBLAS TFLOPS for the pct_peak computation.
"""
import time
import numpy as np
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── NumPy reference (FP64, CPU, ground truth for small problems) ─────────────

def numpy_gemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """C = alpha * A @ B + beta * C  (FP64 precision)."""
    result = alpha * (A.astype(np.float64) @ B.astype(np.float64))
    if C is not None and beta != 0.0:
        result += beta * C.astype(np.float64)
    return result


# ─── PyTorch / cuBLAS reference (GPU) ────────────────────────────────────────

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch not available. Install with: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121"
        )


def cublas_gemm(
    A: "torch.Tensor",
    B: "torch.Tensor",
    alpha: float = 1.0,
    dtype: str = "fp16",
) -> "torch.Tensor":
    """
    C = alpha * A @ B  using cuBLAS (via torch.mm).
    A, B must already be on GPU.
    """
    _require_torch()
    return alpha * torch.mm(A, B)


def measure_cublas_tflops(
    M: int, N: int, K: int,
    dtype: str = "fp16",
    warmup: int = 5,
    iters: int = 20,
    device: int = 0,
) -> float:
    """
    Measure cuBLAS TFLOPS for a given problem size.
    Uses the same warmup+timing protocol as the C++ bench binary.
    """
    _require_torch()
    torch_device = torch.device(f"cuda:{device}")

    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    A = torch.randn(M, K, dtype=torch_dtype, device=torch_device)
    B = torch.randn(K, N, dtype=torch_dtype, device=torch_device)

    # Enable tensor cores
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                         enable_mem_efficient=False):
        # Warmup
        for _ in range(warmup):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()

        # Timed iters
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = torch.mm(A, B)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

    mean_ms = sum(times) / len(times)
    flops   = 2.0 * M * N * K
    return flops / (mean_ms * 1e-3) / 1e12


def random_matrices_fp16(M: int, K: int, N: int, seed: int = 42, device: int = 0):
    """Return (A, B) as GPU FP16 tensors."""
    _require_torch()
    gen = torch.Generator()
    gen.manual_seed(seed)
    torch_device = torch.device(f"cuda:{device}")
    A = torch.randn(M, K, generator=gen, dtype=torch.float16, device=torch_device)
    B = torch.randn(K, N, generator=gen, dtype=torch.float16, device=torch_device)
    return A, B
