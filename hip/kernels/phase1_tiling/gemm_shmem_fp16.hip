#include "gemm_shmem.hpp"

template void launch_gemm_shmem<FP16Tag,  32,  32, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_shmem<FP16Tag,  64,  64, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_shmem<FP16Tag, 128, 128, 32>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_shmem<FP16Tag, 128, 128, 64>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
