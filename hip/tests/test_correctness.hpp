#pragma once

#include <cstdio>
#include <rocblas/rocblas.h>

#include "gemm/types.hpp"
#include "gemm/matrix.hpp"
#include "gemm/error_check.hpp"
#include "tolerance.hpp"

// --- rocBLAS reference ---
// rocBLAS is column-major. For row-major C = alpha*A*B + beta*C we swap
// A<->B and M<->N: computes C^T = alpha * B^T * A^T + beta * C^T.

inline gemm::HostMatrix<float> rocblas_reference_fp16(
    int M, int N, int K,
    const gemm::HostMatrix<__half>& hA,
    const gemm::HostMatrix<__half>& hB)
{
    gemm::DeviceMatrix<__half> dA(M, K), dB(K, N);
    gemm::DeviceMatrix<float>  dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M×N] = A[M×K] * B[K×N]  →  column-major: m=N, n=M, k=K,
    // first matrix = dB (N leading dim), second matrix = dA (K leading dim).
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
    HIP_CHECK(hipDeviceSynchronize());
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return gemm::HostMatrix<float>::from_device(dC);
}

inline gemm::HostMatrix<float> rocblas_reference_fp32(
    int M, int N, int K,
    const gemm::HostMatrix<float>& hA,
    const gemm::HostMatrix<float>& hB)
{
    gemm::DeviceMatrix<float> dA(M, K), dB(K, N), dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_none,
        N, M, K,
        &alpha, dB.ptr, N, dA.ptr, K,
        &beta,  dC.ptr, N));
    HIP_CHECK(hipDeviceSynchronize());
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return gemm::HostMatrix<float>::from_device(dC);
}

// --- CorrectnessResult ---

struct CorrectnessResult {
    bool   passed;
    int    violations;
    int    total_elements;
    int    first_row, first_col;
    double first_computed, first_reference;

    void print(const char* test_name) const {
        if (passed) {
            printf("  [PASS] %s  (%d elements, 0 violations)\n",
                   test_name, total_elements);
        } else {
            printf("  [FAIL] %s  (%d/%d violations)\n"
                   "         first at (%d,%d): computed=%.6f ref=%.6f diff=%.2e\n",
                   test_name, violations, total_elements,
                   first_row, first_col,
                   first_computed, first_reference,
                   std::abs(first_computed - first_reference));
        }
    }
};

template <typename KernelFn>
CorrectnessResult check_fp16_kernel(
    KernelFn kernel_fn,
    int M, int N, int K,
    const TolerancePolicy& tol,
    uint64_t seed = 42)
{
    auto hA = gemm::HostMatrix<__half>::random(M, K, -1.0f, 1.0f, seed);
    auto hB = gemm::HostMatrix<__half>::random(K, N, -1.0f, 1.0f, seed + 1);

    auto hRef = rocblas_reference_fp16(M, N, K, hA, hB);

    gemm::DeviceMatrix<__half> dA(M, K), dB(K, N);
    gemm::DeviceMatrix<float>  dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    GemmDescRowMajor<FP16Tag> desc;
    desc.M = M; desc.N = N; desc.K = K;
    desc.A = dA.ptr; desc.B = dB.ptr; desc.C = dC.ptr;
    desc.lda = K; desc.ldb = N; desc.ldc = N;
    desc.alpha = __half(1.0f); desc.beta = 0.0f;

    kernel_fn(desc);
    HIP_CHECK(hipDeviceSynchronize());

    auto hOut = gemm::HostMatrix<float>::from_device(dC);
    auto res  = gemm::HostMatrix<float>::check(hOut, hRef, tol.rtol, tol.atol);

    CorrectnessResult cr;
    cr.passed          = (res.violations == 0);
    cr.violations      = res.violations;
    cr.total_elements  = M * N;
    cr.first_row       = res.first_row;
    cr.first_col       = res.first_col;
    cr.first_computed  = res.computed;
    cr.first_reference = res.reference;
    return cr;
}
