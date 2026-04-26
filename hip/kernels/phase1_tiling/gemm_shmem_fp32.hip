#include "gemm_shmem.hpp"

template void launch_gemm_shmem<FP32Tag,  32,  32, 32>(GemmDescRowMajor<FP32Tag>&, hipStream_t);
template void launch_gemm_shmem<FP32Tag,  64,  64, 32>(GemmDescRowMajor<FP32Tag>&, hipStream_t);
template void launch_gemm_shmem<FP32Tag, 128, 128, 32>(GemmDescRowMajor<FP32Tag>&, hipStream_t);
