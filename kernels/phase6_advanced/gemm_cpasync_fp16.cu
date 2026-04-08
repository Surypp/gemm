#include "gemm_cpasync.cuh"

// 2-stage: 37 888 B smem, 2 blocks/SM on sm_120 (16 warps), 1-tile lookahead
template void launch_gemm_cpasync<128, 128, 32, 2, 4, 2>(
    GemmDescRowMajor<FP16Tag>&, cudaStream_t);

// 3-stage: 56 832 B smem, 1 block/SM on sm_120 (8 warps), 2-tile lookahead
// requires cudaFuncSetAttribute(MaxDynamicSharedMemorySize) in launch wrapper
template void launch_gemm_cpasync<128, 128, 32, 2, 4, 3>(
    GemmDescRowMajor<FP16Tag>&, cudaStream_t);
