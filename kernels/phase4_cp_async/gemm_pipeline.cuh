#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include "gemm/types.cuh"
#include "gemm/pipeline.cuh"
#include "gemm/error_check.cuh"

using namespace nvcuda;

// --- pipelined wmma GEMM ---
// Extends Phase 3 with:
//   1. Double buffering (NumStages=2): while computing tile N, prefetch tile N+1
//      via cp.async (asynchronous DMA from HBM to SRAM, bypassing register file)
//   2. Register tiling (REG_M × REG_N warp fragments per warp)
//
// cp.async allows the load of tile N+1 to overlap with the compute of tile N,
// effectively hiding the HBM→SRAM latency (~100 cycles on A100).

static constexpr int PIPE_WMMA_M = 16;
static constexpr int PIPE_WMMA_N = 16;
static constexpr int PIPE_WMMA_K = 16;

template <int BM, int BN, int BK,
          int WARP_M, int WARP_N,
          int NumStages = 2>
__global__ void gemm_pipeline_kernel(
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float*        __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
    constexpr int WM = BM / WARP_M;
    constexpr int WN = BN / WARP_N;
    constexpr int FRAGS_M = WM / PIPE_WMMA_M;
    constexpr int FRAGS_N = WN / PIPE_WMMA_N;

    // Double-buffered shared memory: NumStages × BM×BK (A) and NumStages × BK×BN (B)
    __shared__ __half smem_A[NumStages][BM][BK + 8];
    __shared__ __half smem_B[NumStages][BK][BN + 8];

    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_off = blockIdx.y * BM + warp_row * WM;
    int c_col_off = blockIdx.x * BN + warp_col * WN;

    wmma::fragment<wmma::accumulator, PIPE_WMMA_M, PIPE_WMMA_N, PIPE_WMMA_K, float>
        c_frag[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    int tid           = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int num_tiles     = (K + BK - 1) / BK;

    StagedPipeline<NumStages> pipe;

    // --- prologue ---
    // Fast path: 16-byte cp.async (direct GMEM→SMEM, no register traffic).
    // Requires 16-byte aligned source; holds when lda/ldb are multiples of 8
    // and tile column offsets are multiples of 8 (guaranteed by BK=32, BN=128).
    // Slow path: scalar fallback for boundary tiles with out-of-bounds guards.
    // Both paths are correctly synchronized by consumer_wait (cp_async_wait + __syncthreads).
    constexpr int ELEMS_PER_CP = 8;                   // 16 bytes / sizeof(__half)
    constexpr int A_CHUNKS_ROW = BK / ELEMS_PER_CP;   // chunks per A-tile row
    constexpr int B_CHUNKS_ROW = BN / ELEMS_PER_CP;   // chunks per B-tile row

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
                smem_A[stage][r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : __half(0.f);
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
                smem_B[stage][r][c] = (gr < K && gc < N) ? B[gr * ldb + gc] : __half(0.f);
            }
        }
    };

    for (int s = 0; s < NumStages - 1 && s < num_tiles; ++s) {
        async_load_tile(s, s);
        pipe.producer_commit();
    }

    // --- main loop ---
    for (int t = 0; t < num_tiles; ++t) {
        int compute_stage = t % NumStages;
        int fetch_tile    = t + NumStages - 1;
        int fetch_stage   = fetch_tile % NumStages;

        pipe.consumer_wait();

        // MMA on compute_stage
        for (int k = 0; k < BK; k += PIPE_WMMA_K) {
            wmma::fragment<wmma::matrix_a, PIPE_WMMA_M, PIPE_WMMA_N, PIPE_WMMA_K,
                           __half, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, PIPE_WMMA_M, PIPE_WMMA_N, PIPE_WMMA_K,
                           __half, wmma::row_major> b_frag[FRAGS_N];
            for (int fm = 0; fm < FRAGS_M; ++fm)
                wmma::load_matrix_sync(a_frag[fm],
                    &smem_A[compute_stage][warp_row * WM + fm * PIPE_WMMA_M][k], BK + 8);
            for (int fn = 0; fn < FRAGS_N; ++fn)
                wmma::load_matrix_sync(b_frag[fn],
                    &smem_B[compute_stage][k][warp_col * WN + fn * PIPE_WMMA_N], BN + 8);
            for (int fm = 0; fm < FRAGS_M; ++fm)
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag[fm], b_frag[fn], c_frag[fm][fn]);
        }

        pipe.consumer_release();

        // Prefetch next tile
        if (fetch_tile < num_tiles) {
            async_load_tile(fetch_stage, fetch_tile);
            pipe.producer_commit();
        }
    }

    // --- writeback ---
    // Guard on beta avoids a wmma::load_matrix_sync from global when beta==0,
    // which would otherwise cause unnecessary long_scoreboard stalls.
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            int out_row = c_row_off + fm * PIPE_WMMA_M;
            int out_col = c_col_off + fn * PIPE_WMMA_N;
            if (out_row < M && out_col < N) {
                if (beta != 0.0f) {
                    wmma::fragment<wmma::accumulator, PIPE_WMMA_M, PIPE_WMMA_N, PIPE_WMMA_K, float>
                        c_ex;
                    wmma::load_matrix_sync(c_ex, C + out_row * ldc + out_col, ldc,
                                           wmma::mem_row_major);
                    for (int i = 0; i < c_frag[fm][fn].num_elements; ++i)
                        c_frag[fm][fn].x[i] = alpha * c_frag[fm][fn].x[i]
                                            + beta  * c_ex.x[i];
                } else {
                    for (int i = 0; i < c_frag[fm][fn].num_elements; ++i)
                        c_frag[fm][fn].x[i] *= alpha;
                }
                wmma::store_matrix_sync(C + out_row * ldc + out_col, c_frag[fm][fn], ldc,
                                        wmma::mem_row_major);
            }
        }
}

// --- launch wrapper ---
template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NumStages>
void launch_gemm_pipeline(GemmDescRowMajor<FP16Tag>& desc, cudaStream_t stream = 0) {
    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid((desc.N + BN - 1) / BN, (desc.M + BM - 1) / BM);
    gemm_pipeline_kernel<BM, BN, BK, WARP_M, WARP_N, NumStages>
        <<<grid, block, 0, stream>>>(
            desc.A, desc.lda, desc.B, desc.ldb,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            static_cast<float>(desc.alpha),
            static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();
}

#define DECL_PIPE(BM, BN, BK, WM, WN, NS) \
    extern template void launch_gemm_pipeline<BM, BN, BK, WM, WN, NS>( \
        GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_PIPE(128, 128, 32, 2, 4, 2)
DECL_PIPE(128, 128, 32, 2, 4, 3)
DECL_PIPE(128, 128, 64, 2, 4, 2)
#undef DECL_PIPE
