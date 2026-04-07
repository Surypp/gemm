#include "gemm_naive.cuh"
#include "gemm/types.cuh"

template void launch_gemm_naive<FP32Tag>(GemmDescRowMajor<FP32Tag>&, cudaStream_t);
