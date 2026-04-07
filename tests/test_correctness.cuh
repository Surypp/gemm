#pragma once

#include <cstdio>
#include <cublas_v2.h>

#include "gemm/types.cuh"
#include "gemm/matrix.cuh"
#include "gemm/error_check.cuh"
#include "tolerance.hpp"

// --- cuBLAS reference ---

inline gemm::HostMatrix<float> cublas_reference_fp16(
    int M, int N, int K,
    const gemm::HostMatrix<__half>& hA,
    const gemm::HostMatrix<__half>& hB)
{
    gemm::DeviceMatrix<__half> dA(M, K), dB(K, N);
    gemm::DeviceMatrix<float>  dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f, beta = 0.0f;
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
    CUDA_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

    return gemm::HostMatrix<float>::from_device(dC);
}

inline gemm::HostMatrix<float> cublas_reference_fp32(
    int M, int N, int K,
    const gemm::HostMatrix<float>& hA,
    const gemm::HostMatrix<float>& hB)
{
    gemm::DeviceMatrix<float> dA(M, K), dB(K, N), dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha, dB.ptr, N, dA.ptr, K,
        &beta,  dC.ptr, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

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

    // Reference
    auto hRef = cublas_reference_fp16(M, N, K, hA, hB);

    // Our kernel
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
    CUDA_CHECK(cudaDeviceSynchronize());

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
