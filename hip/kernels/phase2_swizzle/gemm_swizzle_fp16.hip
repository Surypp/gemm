#include "gemm_swizzle.hpp"

// TM/TN defaults: BM/16 and BN/16 — gives 16×16 = 256 threads per block for all tiles.
// BM=BN=64 : TM=TN=4, block 16×16 = 256 threads
// BM=BN=128: TM=TN=8, block 16×16 = 256 threads

template void launch_gemm_swizzle<FP16Tag,  64,  64, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_swizzle<FP16Tag, 128, 128, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_swizzle<FP16Tag, 128, 128, 64>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
