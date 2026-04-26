#pragma once

#include <hip/hip_runtime.h>
#include "gemm/types.hpp"
#include "gemm/error_check.hpp"

// --- shared memory GEMM ---
// Template parameters:
//   Tag — FP32Tag / FP16Tag / BF16Tag
//   BM  — tile rows per block
//   BN  — tile cols per block
//   BK  — tile depth (K-dimension strip)
//
// Each block computes a BM×BN submatrix of C.
// K is tiled in strips of BK: load A[BM×BK] and B[BK×BN] cooperatively
// into __shared__, __syncthreads(), accumulate, __syncthreads().

template <typename Tag, int BM, int BN, int BK>
__global__ void gemm_shmem_kernel(GemmDescRowMajor<Tag> desc) {
    using scalar_t = typename Tag::scalar_t;
    using accum_t  = typename Tag::accum_t;

    __shared__ scalar_t smem_A[BM][BK];
    __shared__ scalar_t smem_B[BK][BN];

    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    int thread_row = threadIdx.y;   // 0..BM-1
    int thread_col = threadIdx.x;   // 0..BN-1

    int global_row = tile_row * BM + thread_row;
    int global_col = tile_col * BN + thread_col;

    accum_t acc = accum_t{0};

    int num_tiles = (desc.K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        // --- load A tile ---
        // Simplified: 1 thread = 1 element of smem_A and 1 of smem_B.
        // (Only valid when BM == BN; see static_assert in launch wrapper.)
        {
            int a_col = t * BK + thread_col;
            if (global_row < desc.M && a_col < desc.K)
                smem_A[thread_row][thread_col] =
                    desc.A[global_row * desc.lda + a_col];
            else
                smem_A[thread_row][thread_col] = scalar_t{0};
        }
        {
            int b_row = t * BK + thread_row;
            if (b_row < desc.K && global_col < desc.N)
                smem_B[thread_row][thread_col] =
                    desc.B[b_row * desc.ldb + global_col];
            else
                smem_B[thread_row][thread_col] = scalar_t{0};
        }
        __syncthreads();

        // --- accumulate ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += static_cast<accum_t>(smem_A[thread_row][k])
                 * static_cast<accum_t>(smem_B[k][thread_col]);
        }
        __syncthreads();
    }

    if (global_row < desc.M && global_col < desc.N) {
        desc.C[global_row * desc.ldc + global_col] =
            static_cast<accum_t>(desc.alpha) * acc
          + static_cast<accum_t>(desc.beta)  * desc.C[global_row * desc.ldc + global_col];
    }
}

// --- launch wrapper ---
template <typename Tag, int BM, int BN, int BK>
void launch_gemm_shmem(GemmDescRowMajor<Tag>& desc, hipStream_t stream = 0) {
    static_assert(BM == BN, "Simplified kernel requires BM == BN (square thread block)");
    dim3 block(BN, BM);
    dim3 grid(
        (desc.N + BN - 1) / BN,
        (desc.M + BM - 1) / BM
    );
    gemm_shmem_kernel<Tag, BM, BN, BK><<<grid, block, 0, stream>>>(desc);
    HIP_CHECK_LAST();
}

// --- explicit instantiations ---
#define DECL_SHMEM(TAG, BM, BN, BK) \
    extern template void launch_gemm_shmem<TAG, BM, BN, BK>(GemmDescRowMajor<TAG>&, hipStream_t);

DECL_SHMEM(FP32Tag,  32,  32, 32)
DECL_SHMEM(FP32Tag,  64,  64, 32)
DECL_SHMEM(FP32Tag, 128, 128, 32)

DECL_SHMEM(FP16Tag,  32,  32, 32)
DECL_SHMEM(FP16Tag,  64,  64, 32)
DECL_SHMEM(FP16Tag, 128, 128, 32)
DECL_SHMEM(FP16Tag, 128, 128, 64)
#undef DECL_SHMEM
