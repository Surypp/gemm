#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include "gemm/types.hpp"
#include "gemm/timer.hpp"
#include "gemm/matrix.hpp"
#include "gemm/error_check.hpp"
#include "gemm/benchmark.hpp"

namespace bench_rocblas {

// Row-major C[M×N] = A[M×K] * B[K×N] mapped to column-major by swapping A/B and M/N.

double measure_rocblas_fp32_tflops(int M, int N, int K, int iters = 20) {
    using namespace gemm;

    DeviceMatrix<float> dA(M, K), dB(K, N), dC(M, N);
    {
        auto hA = HostMatrix<float>::random(M, K);
        auto hB = HostMatrix<float>::random(K, N);
        dA.copy_from(hA);
        dB.copy_from(hB);
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;

    auto call = [&] {
        ROCBLAS_CHECK(rocblas_sgemm(handle,
            rocblas_operation_none, rocblas_operation_none,
            N, M, K,
            &alpha, dB.ptr, N, dA.ptr, K,
            &beta,  dC.ptr, N));
    };

    for (int i = 0; i < 5; ++i) call();
    HIP_CHECK(hipDeviceSynchronize());

    TimingStats stats;
    GpuTimer timer;
    for (int i = 0; i < iters; ++i) {
        timer.start();
        call();
        stats.add(timer.stop_ms());
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return compute_tflops(M, N, K, stats.min());
}

} // namespace bench_rocblas
