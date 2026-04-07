#pragma once

#include <cuda_runtime.h>
#include <mma.h>            // nvcuda::wmma
#include "gemm/types.cuh"
#include "gemm/swizzle.cuh"
#include "gemm/error_check.cuh"

using namespace nvcuda;

// --- wmma GEMM ---
// Fragment dimensions on Ampere (sm_80) for FP16→FP32 accumulation:
//   A fragment : 16×16  (M_FRAG × K_FRAG)
//   B fragment : 16×16  (K_FRAG × N_FRAG)
//   C fragment : 16×16  (M_FRAG × N_FRAG)
//
// Block tile (BM × BN × BK) must be multiples of the fragment dimensions.
// Typical: BM=128, BN=128, BK=32 → each block computes 8×8 = 64 warp-level MMA ops.
// Each warp handles a WARP_M × WARP_N subtile of C.

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

template <int BM, int BN, int BK,
          int WARP_M = 2, int WARP_N = 4>
__global__ void gemm_wmma_kernel(
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float*        __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
    constexpr int WM = BM / WARP_M;
    constexpr int WN = BN / WARP_N;

    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_offset = blockIdx.y * BM + warp_row * WM;
    int c_col_offset = blockIdx.x * BN + warp_col * WN;

    constexpr int FRAGS_M = WM / WMMA_M;
    constexpr int FRAGS_N = WN / WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn)
            wmma::fill_fragment(c_frag[fm][fn], 0.0f);

    // +8 padding avoids bank conflicts without swizzle
    __shared__ __half smem_A[BM][BK + 8];
    __shared__ __half smem_B[BK][BN + 8];

    int tid   = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // --- load A ---
        for (int i = tid; i < BM * BK; i += total_threads) {
            int r = i / BK, c = i % BK;
            int global_r = blockIdx.y * BM + r;
            int global_c = t * BK + c;
            smem_A[r][c] = (global_r < M && global_c < K)
                ? A[global_r * lda + global_c]
                : __half(0);
        }
        // --- load B ---
        for (int i = tid; i < BK * BN; i += total_threads) {
            int r = i / BN, c = i % BN;
            int global_r = t * BK + r;
            int global_c = blockIdx.x * BN + c;
            smem_B[r][c] = (global_r < K && global_c < N)
                ? B[global_r * ldb + global_c]
                : __half(0);
        }
        __syncthreads();

        // --- MMA ---
        for (int k = 0; k < BK; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
                a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
                b_frag[FRAGS_N];

            for (int fm = 0; fm < FRAGS_M; ++fm) {
                int a_row = warp_row * WM + fm * WMMA_M;
                wmma::load_matrix_sync(a_frag[fm],
                    &smem_A[a_row][k], BK + 8);
            }
            for (int fn = 0; fn < FRAGS_N; ++fn) {
                int b_col = warp_col * WN + fn * WMMA_N;
                wmma::load_matrix_sync(b_frag[fn],
                    &smem_B[k][b_col], BN + 8);
            }
            for (int fm = 0; fm < FRAGS_M; ++fm)
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::mma_sync(c_frag[fm][fn], a_frag[fm], b_frag[fn], c_frag[fm][fn]);
        }
        __syncthreads();
    }

    // --- writeback ---
    for (int fm = 0; fm < FRAGS_M; ++fm) {
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            int out_row = c_row_offset + fm * WMMA_M;
            int out_col = c_col_offset + fn * WMMA_N;
            if (out_row < M && out_col < N) {
                // Load existing C for beta scaling
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_existing;
                wmma::load_matrix_sync(c_existing, C + out_row * ldc + out_col, ldc,
                                       wmma::mem_row_major);
                for (int i = 0; i < c_frag[fm][fn].num_elements; ++i)
                    c_frag[fm][fn].x[i] = alpha * c_frag[fm][fn].x[i]
                                        + beta  * c_existing.x[i];
                wmma::store_matrix_sync(C + out_row * ldc + out_col, c_frag[fm][fn],
                                        ldc, wmma::mem_row_major);
            }
        }
    }
}

// --- launch wrapper ---
template <int BM, int BN, int BK,
          int WARP_M = 2, int WARP_N = 4>
void launch_gemm_wmma(GemmDescRowMajor<FP16Tag>& desc, cudaStream_t stream = 0) {
    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid(
        (desc.N + BN - 1) / BN,
        (desc.M + BM - 1) / BM
    );
    gemm_wmma_kernel<BM, BN, BK, WARP_M, WARP_N>
        <<<grid, block, 0, stream>>>(
            desc.A, desc.lda,
            desc.B, desc.ldb,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            static_cast<float>(desc.alpha),
            static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();
}

#define DECL_WMMA(BM, BN, BK) \
    extern template void launch_gemm_wmma<BM, BN, BK>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_WMMA(128, 128, 32)
DECL_WMMA(128, 256, 32)
#undef DECL_WMMA
