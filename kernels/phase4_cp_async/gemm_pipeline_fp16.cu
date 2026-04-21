#include "gemm_pipeline.cuh"

template void launch_gemm_pipeline<128, 128, 32, 2, 4, 2>(GemmDescRowMajor<FP16Tag>&, cudaStream_t);
// BK=64 and NumStages=3 exceed 48 KB shared memory limit on sm_120 (RTX 5080)
