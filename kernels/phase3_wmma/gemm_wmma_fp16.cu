#include "gemm_wmma.cuh"

template void launch_gemm_wmma<128, 128, 32>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
template void launch_gemm_wmma<128, 256, 32>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
