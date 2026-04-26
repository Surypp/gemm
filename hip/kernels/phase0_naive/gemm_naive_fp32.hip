#include "gemm_naive.hpp"
#include "gemm/types.hpp"

template void launch_gemm_naive<FP32Tag>(GemmDescRowMajor<FP32Tag>&, hipStream_t);
