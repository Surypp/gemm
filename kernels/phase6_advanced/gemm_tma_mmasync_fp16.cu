#include "gemm_tma_mmasync.cuh"

// --- host launch wrapper instantiations ---
// Suppresses implicit instantiation in every TU that sees the extern template
// declaration from gemm_tma_mmasync.cuh.

template void launch_gemm_tma_mmasync<128, 128, 32, 2, 4, 2>(
    GemmDescRowMajor<FP16Tag>&, cudaStream_t);

// --- explicit __global__ kernel instantiations ---
// The <<<>>> call inside launch_gemm_tma_mmasync is gated on
// #if !defined(__CUDA_ARCH__), so the nvcc device compilation pass never sees
// it.  Without a visible reference in the device pass, nvcc generates no device
// binary for the kernel — cudaErrorInvalidDeviceFunction at runtime.
//
// template __global__ void f<...>() forces the device pass to compile the
// kernel body regardless of how it is called from host code.

template __global__ void gemm_tma_mmasync_kernel<128, 128, 32, 2, 4, 2>(
    const CUtensorMap* __restrict__, const CUtensorMap* __restrict__,
    float* __restrict__, int, int, int, int, float, float);
