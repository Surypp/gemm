#pragma once
// Forward declarations for use from host-only .cpp files.
// Do NOT include gemm_pipeline.hpp here — that pulls in rocwmma which requires device compilation.

#include <hip/hip_runtime.h>
#include "gemm/types.hpp"

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NumStages>
void launch_gemm_pipeline(GemmDescRowMajor<FP16Tag>& desc, hipStream_t stream = 0);

extern template void launch_gemm_pipeline<128, 128, 32, 2, 4, 2>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
