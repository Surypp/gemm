#pragma once

#include <cuda_runtime.h>
#include "gemm/types.cuh"
#include "gemm/swizzle.cuh"
#include "gemm/error_check.cuh"

// --- swizzle GEMM ---
// Thread-tiled version: each thread computes a TM×TN sub-tile of C.
// Block dims: (BN/TN, BM/TM) — stays at ≤1024 threads for all supported tiles.
//
// Swizzle invariant (same rule for A and B):
//   logical (row, col) → stored at smem[row][permute_col(row, col)]
//   reading back: smem[row][permute_col(row, logical_col)]
//
// smem_B coherence:
//   Write: smem_B[r][permute_col(r, c)]  (thread_row=r, thread_col=c)
//   Read:  smem_B[k][permute_col(k, col)]  — correct: same permutation, same row index

template <typename Tag, int BM, int BN, int BK,
          int TM = BM / 16, int TN = BN / 16>
__global__ void gemm_swizzle_kernel(GemmDescRowMajor<Tag> desc) {
    using scalar_t = typename Tag::scalar_t;
    using accum_t  = typename Tag::accum_t;
    using Swizzle  = SwizzlePattern<BK, dtype_size_v<Tag>>;

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(BM * BK % (BM / TM * BN / TN) == 0, "smem_A must divide evenly among threads");
    static_assert(BK * BN % (BM / TM * BN / TN) == 0, "smem_B must divide evenly among threads");

    constexpr int BLOCK_X    = BN / TN;
    constexpr int BLOCK_Y    = BM / TM;
    constexpr int BLOCK_SIZE = BLOCK_X * BLOCK_Y;

    __shared__ scalar_t smem_A[BM][BK];
    __shared__ scalar_t smem_B[BK][BN];

    int ty  = threadIdx.y;
    int tx  = threadIdx.x;
    int tid = ty * BLOCK_X + tx;

    int tile_row = blockIdx.y * BM;
    int tile_col = blockIdx.x * BN;

    accum_t acc[TM][TN] = {};

    int num_tiles = (desc.K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {

        // --- load A ---
        for (int i = tid; i < BM * BK; i += BLOCK_SIZE) {
            int row   = i / BK;
            int col   = i % BK;
            int a_col = t * BK + col;
            scalar_t val = scalar_t{0};
            if (tile_row + row < desc.M && a_col < desc.K)
                val = desc.A[(tile_row + row) * desc.lda + a_col];
            smem_A[row][Swizzle::permute_col(row, col)] = val;
        }

        // --- load B ---
        for (int i = tid; i < BK * BN; i += BLOCK_SIZE) {
            int row   = i / BN;
            int col   = i % BN;
            int b_row = t * BK + row;
            scalar_t val = scalar_t{0};
            if (b_row < desc.K && tile_col + col < desc.N)
                val = desc.B[b_row * desc.ldb + (tile_col + col)];
            smem_B[row][Swizzle::permute_col(row, col)] = val;
        }

        __syncthreads();

        // --- accumulate TM×TN thread tile ---
        for (int k = 0; k < BK; ++k) {
            // Each thread reads a distinct row of smem_A (ty*TM+m is unique per thread)
            // → broadcast within each warp, no bank conflict.
            scalar_t a_regs[TM];
            for (int m = 0; m < TM; ++m) {
                int row = ty * TM + m;
                a_regs[m] = smem_A[row][Swizzle::permute_col(row, k)];
            }

            // Threads in a warp differ in tx → stride-TN columns of smem_B.
            // permute_col(k, tx*TN+n) spreads those columns to different banks.
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    int col = tx * TN + n;
                    acc[m][n] += static_cast<accum_t>(a_regs[m])
                               * static_cast<accum_t>(
                                     smem_B[k][Swizzle::permute_col(k, col)]);
                }
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            int row = tile_row + ty * TM + m;
            int col = tile_col + tx * TN + n;
            if (row < desc.M && col < desc.N) {
                desc.C[row * desc.ldc + col] =
                    static_cast<accum_t>(desc.alpha) * acc[m][n]
                  + static_cast<accum_t>(desc.beta)  * desc.C[row * desc.ldc + col];
            }
        }
    }
}

// --- launch wrapper ---
template <typename Tag, int BM, int BN, int BK,
          int TM = BM / 16, int TN = BN / 16>
void launch_gemm_swizzle(GemmDescRowMajor<Tag>& desc, cudaStream_t stream = 0) {
    static_assert(BM * BN / (TM * TN) <= 1024,
                  "Block size (BM/TM * BN/TN) exceeds CUDA 1024-thread limit");
    dim3 block(BN / TN, BM / TM);
    dim3 grid(
        (desc.N + BN - 1) / BN,
        (desc.M + BM - 1) / BM
    );
    gemm_swizzle_kernel<Tag, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(desc);
    CUDA_CHECK_LAST();
}

#define DECL_SWIZZLE(TAG, BM, BN, BK) \
    extern template void launch_gemm_swizzle<TAG, BM, BN, BK>(GemmDescRowMajor<TAG>&, cudaStream_t);

DECL_SWIZZLE(FP16Tag,  64,  64, 32)
DECL_SWIZZLE(FP16Tag, 128, 128, 32)
DECL_SWIZZLE(FP16Tag, 128, 128, 64)
#undef DECL_SWIZZLE
