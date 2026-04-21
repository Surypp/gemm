#include "gemm_mma_ptx.cuh"

template void launch_gemm_mma_ptx<128, 128, 32, 2, 4>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
// BK=64 exceeds 48 KB shared memory limit on sm_120 (RTX 5080)
