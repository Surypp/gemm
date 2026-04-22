// tests/test_phase5_micro.cu
//
// Micro-kernel that runs a single 16x16 MMA on identity inputs and validates
// that the PTX ISA 9.7.13 write-back formulas map each register to the correct
// (row, col) coordinate in the output matrix.
//
// Test strategy
// -------------
// A = I(16x16), B = I(16x16) → expected D = I(16x16).
// The 16x16 output needs two mma.sync.m16n8k16 calls:
//   - mma1: A * B_left  (B columns 0..7)  → d1[0..3]
//   - mma2: A * B_right (B columns 8..15) → d2[0..3]
//
// For each lane (0..31) we record (row, col, d_val, expected) using the
// PTX ISA 9.7.13 formulas.  Both value correctness and full 16x16 coverage
// (each position written exactly once) are verified.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>

#include "gemm/error_check.cuh"
#include "kernels/phase5_ptx_mma/mma_ptx_utils.cuh"

// ---------------------------------------------------------------------------
// Per-lane write record (plain struct so it can live in device memory).
// ---------------------------------------------------------------------------
struct LaneWrite {
    int   lane;
    int   elem;      // 0..3: mma1 (cols 0..7);  4..7: mma2 (cols 8..15)
    int   row;       // global row  in the 16x16 output matrix (0..15)
    int   col;       // global col  in the 16x16 output matrix (0..15)
    float d_val;     // value produced by mma.sync
    float expected;  // expected value D=I(16x16): 1 iff row==col, else 0
};

// ---------------------------------------------------------------------------
// Device kernel — one warp (32 threads), no threadIdx.y.
// ---------------------------------------------------------------------------
__global__ void mma_identity_micro(LaneWrite* out)
{
    const int lane = threadIdx.x;   // 0..31

    __shared__ __half smem_A [16][16];  // A  row-major (M=16, K=16)
    __shared__ __half smem_B1[16][ 8]; // B_left  row-major (K=16, N=8) cols 0..7
    __shared__ __half smem_B2[16][ 8]; // B_right row-major (K=16, N=8) cols 8..15

    for (int idx = lane; idx < 16 * 16; idx += 32) {
        int r = idx / 16, c = idx % 16;
        smem_A[r][c] = __float2half(r == c ? 1.0f : 0.0f);
    }
    for (int idx = lane; idx < 16 * 8; idx += 32) {
        int r = idx / 8, c = idx % 8;
        // B_left  = I(16x8):  B[k][n] = δ(k, n)
        smem_B1[r][c] = __float2half(r == c       ? 1.0f : 0.0f);
        // B_right = shifted:  B[k][n] = δ(k, n+8)
        smem_B2[r][c] = __float2half(r == (c + 8) ? 1.0f : 0.0f);
    }
    __syncthreads();

    uint32_t a_reg[4], b1[2], b2[2];
    load_matrix_a_16x16(&smem_A [0][0], 16, a_reg, lane);
    load_matrix_b_16x8 (&smem_B1[0][0],  8, b1,    lane);
    load_matrix_b_16x8 (&smem_B2[0][0],  8, b2,    lane);

    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d1[4] = {}, d2[4] = {};
    mma_m16n8k16(a_reg, b1, zero, d1);  // mma1: B left  half
    mma_m16n8k16(a_reg, b2, zero, d2);  // mma2: B right half

    // --- writeback coords (PTX ISA 9.7.13) ---
    const int r0 = lane >> 2;
    const int r1 = (lane >> 2) + 8;
    const int c0 = (lane & 3) * 2;
    const int c1 = (lane & 3) * 2 + 1;
    const int c2 = 8 + (lane & 3) * 2;
    const int c3 = 8 + (lane & 3) * 2 + 1;

    #define EXP(r,c) ((r) == (c) ? 1.0f : 0.0f)

    LaneWrite* w = out + lane * 8;

    // mma1 elements (cols 0..7)
    w[0] = {lane, 0, r0, c0, d1[0], EXP(r0, c0)};
    w[1] = {lane, 1, r0, c1, d1[1], EXP(r0, c1)};
    w[2] = {lane, 2, r1, c0, d1[2], EXP(r1, c0)};
    w[3] = {lane, 3, r1, c1, d1[3], EXP(r1, c1)};

    // mma2 elements (cols 8..15)
    w[4] = {lane, 4, r0, c2, d2[0], EXP(r0, c2)};
    w[5] = {lane, 5, r0, c3, d2[1], EXP(r0, c3)};
    w[6] = {lane, 6, r1, c2, d2[2], EXP(r1, c2)};
    w[7] = {lane, 7, r1, c3, d2[3], EXP(r1, c3)};

    #undef EXP
}

// ---------------------------------------------------------------------------
// Host test
// ---------------------------------------------------------------------------
class Phase5MicroKernelTest : public ::testing::Test {};

TEST_F(Phase5MicroKernelTest, IdentityMmaWritebackLayout)
{
    constexpr int kWrites = 32 * 8;   // 32 lanes × 8 elements = 256
    LaneWrite* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, kWrites * sizeof(LaneWrite)));
    CUDA_CHECK(cudaMemset(d_out, 0, kWrites * sizeof(LaneWrite)));

    mma_identity_micro<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<LaneWrite> h(kWrites);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out,
                          kWrites * sizeof(LaneWrite),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    printf("\n=== Phase 5 micro-kernel: mma.sync identity validation ===\n");
    printf("  A = I(16x16)  B = I(16x16)  expected D = I(16x16)\n");
    printf("  Write-back formulas: PTX ISA 9.7.13\n\n");
    printf("%-5s %-5s %-4s %-4s %-10s %-10s %s\n",
           "lane", "elem", "row", "col", "d_val", "expected", "status");
    printf("----------------------------------------------\n");

    int n_val_fail = 0;
    int cover[16][16] = {};

    for (const auto& w : h) {
        bool ok = fabsf(w.d_val - w.expected) < 1e-3f;
        if (!ok) ++n_val_fail;

        printf("%-5d %-5d %-4d %-4d %-10.4f %-10.4f %s\n",
               w.lane, w.elem, w.row, w.col,
               w.d_val, w.expected,
               ok ? "OK" : "FAIL");

        if (w.row >= 0 && w.row < 16 && w.col >= 0 && w.col < 16)
            ++cover[w.row][w.col];
    }

    printf("\n=== Coverage (16x16 = 256 positions, each must appear once) ===\n");
    int n_over = 0, n_miss = 0;
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            if (cover[r][c] != 1) {
                printf("  [%2d][%2d] = %d write(s) — expected 1\n",
                       r, c, cover[r][c]);
                if (cover[r][c] > 1) ++n_over;
                else                 ++n_miss;
            }
        }
    }
    if (n_over == 0 && n_miss == 0)
        printf("  All 256 positions written exactly once. OK\n");

    printf("\n");

    EXPECT_EQ(n_val_fail, 0)
        << n_val_fail << " element(s) have wrong value — "
        << "check load_matrix_* register layout or mma_m16n8k16 operand order";

    EXPECT_EQ(n_over, 0)
        << n_over << " position(s) written more than once — "
        << "write-back formula produces duplicate coordinates";

    EXPECT_EQ(n_miss, 0)
        << n_miss << " position(s) never written — "
        << "write-back formula misses coordinates";
}
