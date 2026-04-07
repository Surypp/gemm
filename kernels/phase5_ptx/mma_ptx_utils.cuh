#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// --- mma.sync register layout ---
// Instruction: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//
// Base PTX MMA instruction on Ampere (sm_80) for FP16→FP32.
// The .m16n8k16 variant operates on a 16×8 output tile with K=16 depth.
// Two of these can be chained to get 16×16 output (matches wmma's 16×16).
//
// Register layout (per-thread):
//   Each warp (32 threads) computes a 16×8 output fragment.
//   The 128 output elements (16×8) are distributed: 4 FP32 per thread.
//
//   A fragment (16×16, FP16): 8 FP16 per thread → 4 uint32_t (2 FP16 packed per u32)
//     Registers a0..a3:
//       a[i] = {row[2i], row[2i+1]}  (exact mapping depends on lane_id)
//     See: Jia et al. 2018 "Dissecting the NVIDIA Volta GPU Architecture"
//
//   B fragment (16×8, FP16): 4 FP16 per thread → 2 uint32_t
//     Registers b0..b1
//
//   C/D fragment (16×8, FP32): 4 floats per thread
//     Registers d0..d3 (output) and c0..c3 (accumulator input)
//
// To perform 16×16 output, issue two mma.sync instructions (for two 16×8 halves).

__device__ __forceinline__
uint32_t pack_fp16(float a, float b) {
    uint32_t result;
    asm volatile(
        "{\n"
        "  .reg .f16 ra, rb;\n"
        "  cvt.rn.f16.f32 ra, %1;\n"
        "  cvt.rn.f16.f32 rb, %2;\n"
        "  mov.b32 %0, {ra, rb};\n"
        "}\n"
        : "=r"(result) : "f"(a), "f"(b)
    );
    return result;
}

// --- mma_m16n8k16 ---
// Computes D = A*B + D for a 16×8×16 warp-level fragment (in-place accumulation).
//
// a[4]: A fragment (4 × uint32_t, each holding 2 packed FP16)
// b[2]: B fragment (2 × uint32_t, each holding 2 packed FP16)
// d[4]: Accumulator (4 × float) — read as C input, overwritten with D output
//
// Uses "+f" (read-write) constraint so the compiler knows d[] is both C-input and
// D-output. Using "=f" (write-only) with aliased c/d pointers causes the compiler
// to skip loading the accumulator, producing garbage results.

__device__ __forceinline__
void mma_m16n8k16(const uint32_t a[4], const uint32_t b[2], float d[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

// Overload for callers with separate c and d arrays; copies c into d then accumulates.
__device__ __forceinline__
void mma_m16n8k16(const uint32_t a[4], const uint32_t b[2],
                   const float c[4], float d[4]) {
    if (c != d) {
        d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
    }
    mma_m16n8k16(a, b, d);
}

// --- load A fragment (16×16 → 4 registers) ---
// smem_ptr: first element of the 16×16 tile (row-major in SRAM)
// stride: leading dimension of smem in elements
// lane:   threadIdx.x % 32
//
// Row assignment per lane (Ampere m16n8k16 layout):
//   Rows 0..7:  lanes 0..7  (each holds elements at columns 0..7 and 8..15)
//   Rows 8..15: lanes 16..23
// a[0], a[1] = K strip 0..7;  a[2], a[3] = K strip 8..15

__device__ __forceinline__
void load_matrix_a_16x16(const __half* smem_ptr, int stride,
                           uint32_t a[4], int lane) {
    int row0 = (lane / 4) % 8;
    int row1 = row0 + 8;
    int col0 = (lane % 4) * 2;

    auto load2 = [&](int row, int k_off) -> uint32_t {
        uint16_t lo = *reinterpret_cast<const uint16_t*>(&smem_ptr[row * stride + k_off]);
        uint16_t hi = *reinterpret_cast<const uint16_t*>(&smem_ptr[row * stride + k_off + 1]);
        return (static_cast<uint32_t>(hi) << 16) | lo;
    };

    a[0] = load2(row0, col0);
    a[1] = load2(row1, col0);
    a[2] = load2(row0, col0 + 8);
    a[3] = load2(row1, col0 + 8);
}

// --- load B fragment (16×8 → 2 registers) ---
// smem_ptr: the 16×8 B tile (row-major in SRAM: K rows × N cols)
//
// PTX ISA m16n8k16 .col B register layout (per-thread):
//   b[0] = {B[k0, n], B[k0+1, n]}      — two consecutive K-rows, same N-col
//   b[1] = {B[k0+8, n], B[k0+8+1, n]}
// where k0 = (lane % 4) * 2,  n = lane / 4.
//
// The two packed FP16 values come from adjacent K-ROWS (stride apart in row-major
// SRAM), NOT from adjacent N-columns. Loading adjacent N-columns would transpose B
// silently — the error only shows on non-symmetric data.

__device__ __forceinline__
void load_matrix_b_16x8(const __half* smem_ptr, int stride,
                          uint32_t b[2], int lane) {
    int k0 = (lane % 4) * 2;    // K-row base: 0, 2, 4, 6
    int n  = lane / 4;           // N-column:   0 .. 7

    auto load_k_pair = [&](int k, int col) -> uint32_t {
        uint16_t lo = *reinterpret_cast<const uint16_t*>(&smem_ptr[k       * stride + col]);
        uint16_t hi = *reinterpret_cast<const uint16_t*>(&smem_ptr[(k + 1) * stride + col]);
        return (static_cast<uint32_t>(hi) << 16) | lo;
    };

    b[0] = load_k_pair(k0,     n);   // {B[k0,   n], B[k0+1, n]}
    b[1] = load_k_pair(k0 + 8, n);   // {B[k0+8, n], B[k0+9, n]}
}

// NOTE: register layout above is derived from PTX ISA 9.7.13 and validated
// experimentally. If correctness tests fail, print (lane, a_row, a_col) for a
// 16×16 identity matrix to verify the mapping.
