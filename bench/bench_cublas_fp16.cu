#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gemm/types.cuh"
#include "gemm/timer.cuh"
#include "gemm/matrix.cuh"
#include "gemm/error_check.cuh"
#include "gemm/benchmark.cuh"

namespace bench_cublas {

// --- cuBLAS FP16 baseline ---
// Uses cublasGemmEx with CUDA_R_16F inputs and CUDA_R_32F accumulators.
// Matches our kernels' accumulation strategy (FP16 inputs, FP32 C).

double measure_cublas_fp16_tflops(int M, int N, int K, int iters = 20) {
    using namespace gemm;

    DeviceMatrix<__half> dA(M, K), dB(K, N);
    DeviceMatrix<float>  dC(M, N);
    {
        auto hA = HostMatrix<__half>::random(M, K);
        auto hB = HostMatrix<__half>::random(K, N);
        dA.copy_from(hA);
        dB.copy_from(hB);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // enable tensor core math
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB.ptr, CUDA_R_16F, N,
            dA.ptr, CUDA_R_16F, K,
            &beta,
            dC.ptr, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    TimingStats stats;
    GpuTimer timer;
    for (int i = 0; i < iters; ++i) {
        timer.start();
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB.ptr, CUDA_R_16F, N,
            dA.ptr, CUDA_R_16F, K,
            &beta,
            dC.ptr, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        stats.add(timer.stop_ms());
    }

    cublasDestroy(handle);
    return compute_tflops(M, N, K, stats.min());
}

} // namespace bench_cublas
