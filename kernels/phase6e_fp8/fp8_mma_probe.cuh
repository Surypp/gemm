#pragma once

// --- Phase 6e MRE probe: mma.sync FP8 on SM120 ---
//
// Minimal reproducer for mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32.
// Inputs: A = all 1.0 in e4m3 (0x38 per byte), B = all 1.0 in e5m2 (0x3C per byte).
// Expected output: D[i] = 32.0 for every accumulator (32 products of 1.0 × 1.0).
//
// Verification after build:
//   cuobjdump -sass build/tests/gemm_tests.exe | grep -A 10 "fp8_mma_probe_kernel"
//   Expected SASS opcode: HMMA.16832.F32.E4M3.E5M2 (or SM120 variant)
//
// Four-layer check:
//   1. ptxas accepts → syntax valid
//   2. SASS shows HMMA FP8 opcode → ptxas lowered correctly
//   3. result == 32.0 → numeric accumulation correct
//   4. All 32×4 outputs == 32.0 → no lane-specific error

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "gemm/error_check.cuh"

// Kernel: body only compiled for sm_89+ (FP8 mma.sync requires Ada/Hopper/Blackwell).
// __global__ declaration visible in all passes so nvcc generates the host launch stub.
__global__ void fp8_mma_probe_kernel(float* __restrict__ out)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // 1.0 in e4m3: sign=0, exp=7 (0b0111), mantissa=0b000 → bits = 0x38
    // Four bytes packed per uint32 (little-endian): 0x38383838
    constexpr uint32_t fp8e4m3_ones = 0x38383838u;

    // 1.0 in e5m2: sign=0, exp=15 (0b01111), mantissa=0b00 → bits = 0x3C
    constexpr uint32_t fp8e5m2_ones = 0x3C3C3C3Cu;

    // A: 4 × uint32, each holding 4 × e4m3-1.0 = 16 elements covering 16-row × 32-col tile
    // B: 2 × uint32, each holding 4 × e5m2-1.0 = 8 elements covering 32-row × 8-col tile
    uint32_t a[4] = {fp8e4m3_ones, fp8e4m3_ones, fp8e4m3_ones, fp8e4m3_ones};
    uint32_t b[2] = {fp8e5m2_ones, fp8e5m2_ones};
    float    d[4] = {0.f, 0.f, 0.f, 0.f};

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    // Each thread writes its 4 D-fragment floats.
    // With A=B=1.0 and K=32, every accumulator should equal 32.0.
    int lane = threadIdx.x & 31;
    out[lane * 4 + 0] = d[0];
    out[lane * 4 + 1] = d[1];
    out[lane * 4 + 2] = d[2];
    out[lane * 4 + 3] = d[3];
#else
    // Device arch < 900: write sentinel so the test can detect unsupported arch
    if (threadIdx.x == 0) out[0] = -1.f;
#endif
}

// Host-side launcher.  Returns true if all 128 output values equal 32.0.
inline bool run_fp8_mma_probe()
{
#if !defined(__CUDA_ARCH__)
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 32 * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, 32 * 4 * sizeof(float)));

    fp8_mma_probe_kernel<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_out[32 * 4];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    bool pass = true;
    int  fail_idx = -1;
    for (int i = 0; i < 32 * 4; ++i) {
        if (h_out[i] != 32.0f) { pass = false; fail_idx = i; break; }
    }
    if (pass) {
        printf("  fp8_mma_probe: PASS  (d[0]=%.1f, all 128 outputs == 32.0)\n",
               h_out[0]);
    } else {
        printf("  fp8_mma_probe: FAIL  (index %d: got %.6f, expected 32.0)\n",
               fail_idx, fail_idx >= 0 ? h_out[fail_idx] : -1.f);
    }
    return pass;
#else
    return false;
#endif
}
