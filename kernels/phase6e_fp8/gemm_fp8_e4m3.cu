#include "gemm_fp8.cuh"

// --- explicit instantiations: host launch wrappers ---
template void launch_gemm_fp8<128, 128, 32, 2, 4, 2>(GemmDescFP8&, cudaStream_t);

// --- explicit __global__ kernel instantiations ---
// The <<<>>> call in launch_gemm_fp8 is gated on #if !__CUDA_ARCH__, so the
// device pass never sees it.  Without a visible reference in the device pass,
// nvcc generates no device binary → cudaErrorInvalidDeviceFunction at runtime.
// template __global__ void f<...>() forces device compilation of the kernel body.
template __global__ void gemm_fp8_kernel<128, 128, 32, 2, 4, 2>(
    const CUtensorMap* __restrict__,
    const CUtensorMap* __restrict__,
    float* __restrict__, int,
    int, int, int,
    float, float);
