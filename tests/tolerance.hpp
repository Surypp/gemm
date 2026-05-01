#pragma once

#include <cmath>

// ─── TolerancePolicy ──────────────────────────────────────────────────────────
// Justifications:
//
//   FP32: rtol=1e-4, atol=1e-6
//     Worst-case accumulation error for K=8192 random inputs in [-1,1]:
//     K * eps_machine ≈ 8192 * 1.19e-7 ≈ 9.7e-4.
//     We use 1e-4 which is tight enough to catch logic bugs but passes
//     for well-conditioned inputs.
//
//   FP16: rtol=5e-2, atol=1e-3
//     FP16 has 10-bit mantissa (eps ≈ 9.7e-4). Inputs are quantized to FP16
//     precision before the dot product, introducing ~0.5% per-element error.
//     Kernels accumulate in FP32 (wmma/mma.sync), so accumulation itself is
//     accurate, but the quantized inputs cause ~1-2% deviation from the
//     FP64 reference for large K.
//
//   BF16: rtol=2e-2, atol=5e-4
//     BF16 has 7-bit mantissa (eps ≈ 7.8e-3). Per-element quantization error
//     is larger than FP16 but accumulators are FP32.

struct TolerancePolicy {
    double rtol;  // relative tolerance: |computed - ref| <= atol + rtol * |ref|
    double atol;  // absolute tolerance floor

    static TolerancePolicy for_fp32() { return {1e-4, 1e-6}; }
    static TolerancePolicy for_fp16() { return {5e-2, 1e-3}; }
    static TolerancePolicy for_bf16() { return {2e-2, 5e-4}; }

    // FP8 e4m3/e5m2: 3-bit and 2-bit mantissa respectively.
    // Accumulation is in FP32 (mma.sync), but the quantized inputs introduce
    // ~1-3% per-element error vs the FP64 reference.  For K=512, worst-case
    // accumulated quantization drift can reach ~5%.  atol=1e-1 covers edge cases
    // near zero where relative tolerance is meaningless.
    // NOT bit-exact vs cuBLAS FP8 — use CPU oracle (see test_phase6e.cpp).
    static TolerancePolicy for_fp8() { return {5e-2, 1e-1}; }

    bool check(double computed, double reference) const {
        return std::abs(computed - reference)
               <= atol + rtol * std::abs(reference);
    }

    bool check_f(float computed, float reference) const {
        return check(static_cast<double>(computed),
                     static_cast<double>(reference));
    }
};
