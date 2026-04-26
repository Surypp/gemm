#pragma once
// Forward declarations for use from host-only .cpp files.
// Do NOT include gemm_swizzle.hpp here — that defines __global__ kernels
// which force device compilation for every target arch.

#include <hip/hip_runtime.h>
#include "gemm/types.hpp"

template <typename Tag, int BM, int BN, int BK,
          int TM = BM / 16, int TN = BN / 16>
void launch_gemm_swizzle(GemmDescRowMajor<Tag>& desc, hipStream_t stream = 0);

extern template void launch_gemm_swizzle<FP16Tag,  64,  64, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
extern template void launch_gemm_swizzle<FP16Tag, 128, 128, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
extern template void launch_gemm_swizzle<FP16Tag, 128, 128, 64>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
