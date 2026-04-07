#!/usr/bin/env python3
"""
Roofline plot for GEMM benchmark results.

Usage:
    python3 tools/roofline/roofline.py results.json [--gpu a100|h100] [--dtype fp16]
    python3 tools/roofline/roofline.py results.json --gpu a100 --out roofline.png
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install with: pip install matplotlib numpy")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from hw_limits import A100_SXM4_80GB, H100_SXM5_80GB, arithmetic_intensity_gemm, roofline_peak


GPU_SPECS = {
    "a100": A100_SXM4_80GB,
    "h100": H100_SXM5_80GB,
}

PHASE_COLORS = {
    "naive":    "#e74c3c",
    "shmem":    "#e67e22",
    "swizzle":  "#f1c40f",
    "wmma":     "#2ecc71",
    "pipeline": "#3498db",
    "ptx":      "#9b59b6",
    "hopper":   "#1abc9c",
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_roofline(results: list[dict], spec, dtype: str, out_path: str | None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # ── Roofline lines ─────────────────────────────────────────────────────────
    ai_range = np.logspace(-1, 3, 500)
    roof = np.array([roofline_peak(spec, ai, dtype) for ai in ai_range])

    ax.loglog(ai_range, roof, "k-", linewidth=2, label="Roofline")

    # Mark the ridge point (transition from memory-bound to compute-bound)
    if dtype == "fp16":
        compute_peak = spec.fp16_tensor_tflops
    elif dtype == "fp32":
        compute_peak = spec.fp32_peak_tflops
    else:
        compute_peak = spec.fp16_tensor_tflops

    ridge_ai = compute_peak * 1e3 / spec.hbm_bandwidth_gbs
    ax.axvline(ridge_ai, color="gray", linestyle="--", alpha=0.5,
               label=f"Ridge point ({ridge_ai:.0f} FLOP/B)")
    ax.axhline(compute_peak, color="gray", linestyle=":", alpha=0.5,
               label=f"Compute peak ({compute_peak:.0f} TFLOPS)")

    # ── Plot measurement points ────────────────────────────────────────────────
    phase_handles = {}
    for r in results:
        if r.get("error") or r.get("dtype", dtype) != dtype:
            continue
        M, N, K = r.get("M", 0), r.get("N", 0), r.get("K", 0)
        if not all([M, N, K]):
            continue
        ai    = arithmetic_intensity_gemm(M, N, K, dtype_bytes=2 if dtype == "fp16" else 4)
        tflops = r.get("tflops", 0)
        phase  = r.get("phase", "unknown")
        color  = PHASE_COLORS.get(phase, "#95a5a6")

        ax.scatter(ai, tflops, color=color, s=80, zorder=5, alpha=0.8)

        # Annotate with problem size for the largest points
        if max(M, N, K) >= 4096:
            ax.annotate(f"{M}×{K}", (ai, tflops),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=color)

        if phase not in phase_handles:
            phase_handles[phase] = mpatches.Patch(
                color=color, label=phase)

    # ── Axes and labels ────────────────────────────────────────────────────────
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline — {spec.name}  [{dtype.upper()}]", fontsize=13)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.1, compute_peak * 1.5)
    ax.grid(True, which="both", alpha=0.3)

    legend_items = [
        mpatches.Patch(color="k", label="Roofline"),
    ] + list(phase_handles.values())
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Benchmark results JSON")
    parser.add_argument("--gpu",   default="a100", choices=["a100", "h100"])
    parser.add_argument("--dtype", default="fp16",
                        choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--out",   default=None,
                        help="Output image path (e.g. roofline.png)")
    args = parser.parse_args()

    spec    = GPU_SPECS[args.gpu]
    results = load_results(args.json_file)
    plot_roofline(results, spec, args.dtype, args.out)


if __name__ == "__main__":
    main()
