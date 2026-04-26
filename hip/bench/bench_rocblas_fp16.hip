#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include "gemm/types.hpp"
#include "gemm/timer.hpp"
#include "gemm/matrix.hpp"
#include "gemm/error_check.hpp"
#include "gemm/benchmark.hpp"

namespace bench_rocblas {

// rocBLAS is column-major. Row-major C[M×N] = A[M×K] * B[K×N] is equivalent to
// column-major C^T[N×M] = B^T[K×N] * A^T[M×K], so we call with N,M,K and swap A/B.

double measure_rocblas_fp16_tflops(int M, int N, int K, int iters = 20) {
    using namespace gemm;

    DeviceMatrix<__half> dA(M, K), dB(K, N);
    DeviceMatrix<float>  dC(M, N);
    {
        auto hA = HostMatrix<__half>::random(M, K);
        auto hB = HostMatrix<__half>::random(K, N);
        dA.copy_from(hA);
        dB.copy_from(hB);
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;

    auto call = [&] {
        ROCBLAS_CHECK(rocblas_gemm_ex(handle,
            rocblas_operation_none, rocblas_operation_none,
            N, M, K,
            &alpha,
            dB.ptr, rocblas_datatype_f16_r, N,
            dA.ptr, rocblas_datatype_f16_r, K,
            &beta,
            dC.ptr, rocblas_datatype_f32_r, N,
            dC.ptr, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard,
            0, rocblas_gemm_flags_none));
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
