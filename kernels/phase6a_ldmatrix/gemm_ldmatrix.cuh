#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gemm/types.cuh"
#include "gemm/pipeline.cuh"
#include "gemm/error_check.cuh"
#include "ldmatrix_utils.cuh"
#include "phase5_ptx_mma/mma_ptx_utils.cuh"

// --- Phase 6a: ldmatrix + mma.sync GEMM ---
//
// Replaces the per-thread scalar loads of Phase 5 (load_matrix_a_16x16,
// load_matrix_b_16x8) with warp-cooperative ldmatrix instructions.
// Everything else — smem layout, double buffering, mma.sync, writeback —
// is identical to Phase 5.
//
// ldmatrix.sync.aligned requires sm_80+. This kernel targets sm_80 (A100),
// sm_90a (GH200), and sm_120 (RTX 5080). All three satisfy the requirement.
//
// Smem layout:
//   smem_A[2][BM][BK+8]  row-major, +8 FP16 padding eliminates bank conflicts
//   smem_B[2][BK][BN+8]  row-major, same padding
//   Total (BM=128, BN=128, BK=32):
//     2 * (128*40 + 32*136) * 2 = 37888 bytes < 48 KB  (safe on sm_120)

template <int BM, int BN, int BK, int WARP_M, int WARP_N>
__global__ void gemm_ldmatrix_kernel(
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float*        __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
    constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    constexpr int WM     = BM / WARP_M;
    constexpr int WN     = BN / WARP_N;
    constexpr int FRAGS_M = WM / MMA_M;
    constexpr int FRAGS_N = WN / MMA_N;

    __shared__ __half smem_A[2][BM][BK + 8];
    __shared__ __half smem_B[2][BK][BN + 8];

    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int lane     = threadIdx.x % 32;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_off = blockIdx.y * BM + warp_row * WM;
    int c_col_off = blockIdx.x * BN + warp_col * WN;

    float d[FRAGS_M][FRAGS_N][8] = {};

    int tid           = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int num_tiles     = (K + BK - 1) / BK;

    constexpr int ELEMS_PER_CP = 8;
    constexpr int A_CHUNKS_ROW = BK / ELEMS_PER_CP;
    constexpr int B_CHUNKS_ROW = BN / ELEMS_PER_CP;

    // Fast path: 16-byte cp.async (GMEM→SMEM, bypasses register file).
    // Requires 16-byte aligned source; guaranteed when BK=32/BN=128 and lda/ldb%8==0.
    // Slow path: scalar fallback for boundary tiles.
    auto async_load_tile = [&](int stage, int tile_idx) {
        const int gm_base = blockIdx.y * BM;
        const int gk_base = tile_idx   * BK;
        const int gn_base = blockIdx.x * BN;

        if (gm_base + BM <= M && gk_base + BK <= K) {
            for (int i = tid; i < BM * A_CHUNKS_ROW; i += total_threads) {
                int row = i / A_CHUNKS_ROW;
                int col = (i % A_CHUNKS_ROW) * ELEMS_PER_CP;
                cp_async_16b(&smem_A[stage][row][col],
                             A + (gm_base + row) * lda + gk_base + col);
            }
        } else {
            for (int i = tid; i < BM * BK; i += total_threads) {
                int r = i / BK, c = i % BK;
                int gr = gm_base + r, gc = gk_base + c;
                smem_A[stage][r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : __half(0);
            }
        }

        if (gk_base + BK <= K && gn_base + BN <= N) {
            for (int i = tid; i < BK * B_CHUNKS_ROW; i += total_threads) {
                int row = i / B_CHUNKS_ROW;
                int col = (i % B_CHUNKS_ROW) * ELEMS_PER_CP;
                cp_async_16b(&smem_B[stage][row][col],
                             B + (gk_base + row) * ldb + gn_base + col);
            }
        } else {
            for (int i = tid; i < BK * BN; i += total_threads) {
                int r = i / BN, c = i % BN;
                int gr = gk_base + r, gc = gn_base + c;
                smem_B[stage][r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : __half(0);
            }
        }
    };

    async_load_tile(0, 0);
    cp_async_commit();

    for (int t = 0; t < num_tiles; ++t) {
        int cs = t % 2;
        int ns = (t + 1) % 2;

        cp_async_wait<0>();
        __syncthreads();

        if (t + 1 < num_tiles) {
            async_load_tile(ns, t + 1);
            cp_async_commit();
        }

        // --- ldmatrix + mma.sync ---
        for (int k = 0; k < BK; k += MMA_K) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                uint32_t a_reg[4];
                int a_row = warp_row * WM + fm * MMA_M;

                // ldmatrix.x4: each thread provides one of 32 row addresses
                // covering the full 16×16 A tile (4 matrices of 8×8)
                ldmatrix_a_x4(
                    a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                    &smem_A[cs][a_row + lane % 16][k + (lane >> 4) * 8]);

                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    uint32_t b_reg[2], b_reg2[2];
                    int b_col = warp_col * WN + fn * MMA_N;

                    // ldmatrix.x2.trans: thread provides K-row, hardware transposes
                    // to deliver K-adjacent pairs per N-column (B .col operand)
                    ldmatrix_b_x2_trans(
                        b_reg[0], b_reg[1],
                        &smem_B[cs][k + lane % 16][b_col]);
                    ldmatrix_b_x2_trans(
                        b_reg2[0], b_reg2[1],
                        &smem_B[cs][k + lane % 16][b_col + 8]);

                    float* acc = d[fm][fn];
                    mma_m16n8k16(a_reg, b_reg,  acc);
                    mma_m16n8k16(a_reg, b_reg2, acc + 4);
                }
            }
        }
    }

    // --- writeback ---
    // identical to Phase 5: mma.sync m16n8k16 distributes 16×8 over 32 threads (4 FP32),
    // two calls cover the full 16×16 tile (8 FP32 per thread)
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            int out_row = c_row_off + fm * MMA_M;
            int out_col = c_col_off + fn * MMA_N;

            int r0 = out_row + (lane >> 2);
            int r1 = out_row + (lane >> 2) + 8;
            int c0 = out_col + (lane & 3) * 2;
            int c1 = out_col + (lane & 3) * 2 + 1;
            int c2 = out_col + 8 + (lane & 3) * 2;
            int c3 = out_col + 8 + (lane & 3) * 2 + 1;

            if (r0 < M && c0 < N) C[r0*ldc+c0] = alpha*d[fm][fn][0] + beta*C[r0*ldc+c0];
            if (r0 < M && c1 < N) C[r0*ldc+c1] = alpha*d[fm][fn][1] + beta*C[r0*ldc+c1];
            if (r1 < M && c0 < N) C[r1*ldc+c0] = alpha*d[fm][fn][2] + beta*C[r1*ldc+c0];
            if (r1 < M && c1 < N) C[r1*ldc+c1] = alpha*d[fm][fn][3] + beta*C[r1*ldc+c1];

            if (r0 < M && c2 < N) C[r0*ldc+c2] = alpha*d[fm][fn][4] + beta*C[r0*ldc+c2];
            if (r0 < M && c3 < N) C[r0*ldc+c3] = alpha*d[fm][fn][5] + beta*C[r0*ldc+c3];
            if (r1 < M && c2 < N) C[r1*ldc+c2] = alpha*d[fm][fn][6] + beta*C[r1*ldc+c2];
            if (r1 < M && c3 < N) C[r1*ldc+c3] = alpha*d[fm][fn][7] + beta*C[r1*ldc+c3];
        }
}

// --- launch wrapper ---
template <int BM, int BN, int BK, int WARP_M, int WARP_N>
void launch_gemm_ldmatrix(GemmDescRowMajor<FP16Tag>& desc, cudaStream_t stream = 0) {
    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid((desc.N + BN - 1) / BN, (desc.M + BM - 1) / BM);
    gemm_ldmatrix_kernel<BM, BN, BK, WARP_M, WARP_N>
        <<<grid, block, 0, stream>>>(
            desc.A, desc.lda, desc.B, desc.ldb,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            static_cast<float>(desc.alpha),
            static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();
}

#define DECL_LDMATRIX(BM, BN, BK, WM, WN) \
    extern template void launch_gemm_ldmatrix<BM, BN, BK, WM, WN>( \
        GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_LDMATRIX(128, 128, 32, 2, 4)
#undef DECL_LDMATRIX
