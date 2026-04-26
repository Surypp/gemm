#include "gemm_pipeline.hpp"

template void launch_gemm_pipeline<128, 128, 32, 2, 4, 2>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_pipeline<128, 128, 32, 4, 4, 2>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_pipeline<128, 256, 32, 2, 4, 2>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
template void launch_gemm_pipeline<128, 256, 32, 4, 4, 2>(GemmDescRowMajor<FP16Tag>&, hipStream_t);
