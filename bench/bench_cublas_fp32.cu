#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gemm/types.cuh"
#include "gemm/timer.cuh"
#include "gemm/matrix.cuh"
#include "gemm/error_check.cuh"
#include "gemm/benchmark.cuh"

namespace bench_cublas {

// --- cuBLAS FP32 baseline ---
// cublasSgemm expects column-major layout.
// Trick: treat row-major A(M×K) as col-major A^T(K×M), same for B.
// cublasSgemm(N, N, N, M, K, ...) with dims transposed gives C = A*B row-major.

double measure_cublas_fp32_tflops(int M, int N, int K, int iters = 20) {
    using namespace gemm;

    DeviceMatrix<float> dA(M, K), dB(K, N), dC(M, N);
    {
        auto hA = HostMatrix<float>::random(M, K);
        auto hB = HostMatrix<float>::random(K, N);
        dA.copy_from(hA);
        dB.copy_from(hB);
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB.ptr, N,
            dA.ptr, K,
            &beta,
            dC.ptr, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    TimingStats stats;
    GpuTimer timer;
    for (int i = 0; i < iters; ++i) {
        timer.start();
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, dB.ptr, N,
                    dA.ptr, K,
            &beta,  dC.ptr, N));
        stats.add(timer.stop_ms());
    }

    cublasDestroy(handle);
    return compute_tflops(M, N, K, stats.min());
}

} // namespace bench_cublas
