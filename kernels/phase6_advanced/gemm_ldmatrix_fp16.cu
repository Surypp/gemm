#include "gemm_ldmatrix.cuh"

template void launch_gemm_ldmatrix<128, 128, 32, 2, 4>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
