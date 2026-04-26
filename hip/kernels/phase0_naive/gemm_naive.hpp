#pragma once

#include <hip/hip_runtime.h>
#include "gemm/types.hpp"
#include "gemm/error_check.hpp"

// --- naive GEMM ---
// No shared memory, no tiling, no reuse — baseline only.
// B is column-traversed (non-coalesced) → worst-case bandwidth for B.

template <typename Tag>
__global__ void gemm_naive_kernel(GemmDescRowMajor<Tag> desc) {
    using scalar_t = typename Tag::scalar_t;
    using accum_t  = typename Tag::accum_t;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= desc.M || col >= desc.N) return;

    accum_t acc = accum_t{0};
    for (int k = 0; k < desc.K; ++k) {
        acc += static_cast<accum_t>(desc.A[row * desc.lda + k])
             * static_cast<accum_t>(desc.B[k   * desc.ldb + col]);
    }

    desc.C[row * desc.ldc + col] =
        static_cast<accum_t>(desc.alpha) * acc
      + static_cast<accum_t>(desc.beta)  * desc.C[row * desc.ldc + col];
}

// --- launch wrapper ---
template <typename Tag>
void launch_gemm_naive(GemmDescRowMajor<Tag>& desc, hipStream_t stream = 0) {
    constexpr int kBlockDim = 16;
    dim3 block(kBlockDim, kBlockDim);
    dim3 grid(
        (desc.N + kBlockDim - 1) / kBlockDim,
        (desc.M + kBlockDim - 1) / kBlockDim
    );
    gemm_naive_kernel<Tag><<<grid, block, 0, stream>>>(desc);
    HIP_CHECK_LAST();
}

// --- explicit instantiations ---
extern template void launch_gemm_naive<FP32Tag>(GemmDescRowMajor<FP32Tag>&, hipStream_t);
extern template void launch_gemm_naive<FP16Tag>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
