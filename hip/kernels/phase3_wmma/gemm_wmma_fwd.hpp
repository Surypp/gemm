#pragma once
// Forward declarations for use from host-only .cpp files.
// Do NOT include gemm_wmma.hpp here — that pulls in rocwmma which requires device compilation.

#include <hip/hip_runtime.h>
#include "gemm/types.hpp"

template <int BM, int BN, int BK, int WARP_M = 2, int WARP_N = 4>
void launch_gemm_wmma(GemmDescRowMajor<FP16Tag>& desc, hipStream_t stream = 0);

extern template void launch_gemm_wmma<128, 128, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
extern template void launch_gemm_wmma<128, 128, 32, 4, 4>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
extern template void launch_gemm_wmma<128, 128, 64>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
extern template void launch_gemm_wmma<128, 256, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
