#include "gemm_naive.cuh"
#include "gemm/types.cuh"

template void launch_gemm_naive<FP16Tag>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
