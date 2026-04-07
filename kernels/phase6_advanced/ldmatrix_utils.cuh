#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// --- ldmatrix wrappers for mma.sync m16n8k16 ---
//
// ldmatrix.sync.aligned requires sm_80+.
// These helpers feed directly into mma_m16n8k16 from mma_ptx_utils.cuh —
// the register layout they produce is identical to what load_matrix_a_16x16
// and load_matrix_b_16x8 produce manually, but in a single warp-cooperative
// instruction instead of 4 scalar loads per thread.
//
// Addressing contract (must hold at the call site):
//
//   ldmatrix_a_x4: thread lane provides
//     &smem_A[a_row + lane%16][k_base + (lane>>4)*8]
//   Threads 0-15 cover mat0+mat1 (k cols 0-7), threads 16-31 cover mat2+mat3 (k cols 8-15).
//   Threads 0-7 → mat0 rows 0-7 (A rows 0-7, k 0-7)
//   Threads 8-15 → mat1 rows 0-7 (A rows 8-15, k 0-7)
//   Threads 16-23 → mat2 rows 0-7 (A rows 0-7, k 8-15)
//   Threads 24-31 → mat3 rows 0-7 (A rows 8-15, k 8-15)
//
//   ldmatrix_b_x2_trans: thread lane provides
//     &smem_B[k_base + lane%16][b_col]  (call twice: b_col and b_col+8)
//   .trans: source K-rows (8 N-values each) are transposed so each thread
//   receives K-adjacent pairs for its N-column — matching B (.col) operand layout.
//   Threads 0-7 → mat0 rows k+0..7, threads 8-15 → mat1 rows k+8..15.
//   Threads 16-31 provide duplicate addresses (ignored by x2).
//
// Alignment: source pointer must be 16-byte aligned (= 8 FP16).
// With padding (+8 FP16) in smem rows, all row starts are 16-byte aligned.

// load 4 matrices of 8×8 b16 — A fragment for mma.sync m16n8k16
__device__ __forceinline__
void ldmatrix_a_x4(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
                   const __half *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(addr)
    );
}

// load 2 matrices of 8×8 b16 with transpose — B fragment for mma.sync m16n8k16
__device__ __forceinline__
void ldmatrix_b_x2_trans(uint32_t &d0, uint32_t &d1, const __half *smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(d0), "=r"(d1)
        : "r"(addr)
    );
}
