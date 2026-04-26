#include "gemm_naive.hpp"
#include "gemm/types.hpp"

template void launch_gemm_naive<FP16Tag>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
