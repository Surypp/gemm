"""
Hardware roofline constants for A100 and H100.
Import this module in roofline.py and other analysis scripts.

Sources:
  A100: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
  H100: https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
"""

from dataclasses import dataclass


@dataclass
class GpuSpec:
    name: str
    sm_arch: str           # e.g. "sm_80"

    # Compute peaks (TFLOPS)
    fp32_peak_tflops: float        # CUDA cores FP32 (no sparsity)
    fp16_tensor_tflops: float      # FP16 tensor core (no sparsity)
    bf16_tensor_tflops: float
    tf32_tensor_tflops: float      # FP32 inputs, TF32 internal (A100 default SGEMM)
    fp8_tensor_tflops: float       # FP8 E4M3/E5M2 (H100 only, 0 for A100)

    # Memory
    hbm_bandwidth_gbs: float       # GB/s (peak, uni-directional)
    l2_cache_mb: float

    # Shared memory
    smem_per_sm_kb: float          # configurable max (e.g. 192 KB on A100)
    sm_count: int


A100_SXM4_80GB = GpuSpec(
    name                  = "A100 SXM4 80GB",
    sm_arch               = "sm_80",
    fp32_peak_tflops      = 19.5,
    fp16_tensor_tflops    = 312.0,   # w/o sparsity
    bf16_tensor_tflops    = 312.0,
    tf32_tensor_tflops    = 156.0,   # w/o sparsity
    fp8_tensor_tflops     = 0.0,     # not supported
    hbm_bandwidth_gbs     = 2000.0,  # 2 TB/s
    l2_cache_mb           = 40.0,
    smem_per_sm_kb        = 192.0,
    sm_count              = 108,
)

H100_SXM5_80GB = GpuSpec(
    name                  = "H100 SXM5 80GB",
    sm_arch               = "sm_90",
    fp32_peak_tflops      = 67.0,
    fp16_tensor_tflops    = 989.4,   # w/o sparsity
    bf16_tensor_tflops    = 989.4,
    tf32_tensor_tflops    = 494.7,
    fp8_tensor_tflops     = 1978.9,  # w/o sparsity
    hbm_bandwidth_gbs     = 3350.0,  # 3.35 TB/s
    l2_cache_mb           = 50.0,
    smem_per_sm_kb        = 228.0,   # 227 KB configurable
    sm_count              = 132,
)

# Roofline helpers
def arithmetic_intensity_gemm(M: int, N: int, K: int, dtype_bytes: int = 2) -> float:
    """
    Arithmetic intensity (FLOP/byte) for a GEMM of shape M×N×K.

    FLOPs  = 2 * M * N * K  (multiply-adds)
    Bytes  = (M*K + K*N + M*N) * dtype_bytes  (read A, B; read+write C)
    """
    flops = 2.0 * M * N * K
    bytes_ = (M * K + K * N + M * N) * dtype_bytes
    return flops / bytes_


def roofline_peak(spec: GpuSpec, intensity: float, dtype: str = "fp16") -> float:
    """
    Roofline model prediction: min(compute_peak, bandwidth * intensity).
    Returns predicted TFLOPS.
    """
    if dtype == "fp16":
        compute_peak = spec.fp16_tensor_tflops
    elif dtype == "fp32":
        compute_peak = spec.fp32_peak_tflops
    elif dtype == "bf16":
        compute_peak = spec.bf16_tensor_tflops
    elif dtype == "tf32":
        compute_peak = spec.tf32_tensor_tflops
    elif dtype == "fp8":
        compute_peak = spec.fp8_tensor_tflops
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    bandwidth_tflops_per_byte = spec.hbm_bandwidth_gbs / 1e3  # TB/s = TFLOPS/FLOP_per_byte
    memory_roof = bandwidth_tflops_per_byte * intensity
    return min(compute_peak, memory_roof)


if __name__ == "__main__":
    for spec in [A100_SXM4_80GB, H100_SXM5_80GB]:
        print(f"\n{spec.name} ({spec.sm_arch})")
        print(f"  FP16 tensor peak : {spec.fp16_tensor_tflops:.1f} TFLOPS")
        print(f"  HBM bandwidth    : {spec.hbm_bandwidth_gbs:.0f} GB/s")
        for size in [512, 1024, 2048, 4096, 8192]:
            ai = arithmetic_intensity_gemm(size, size, size)
            pred = roofline_peak(spec, ai, "fp16")
            print(f"  GEMM {size}×{size}×{size} :  AI={ai:.1f} FLOP/B → roof={pred:.1f} TFLOPS")
