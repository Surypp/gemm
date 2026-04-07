#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// --- CUDA error check ---
// Implemented as macros so __FILE__ and __LINE__ resolve to the call site.

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr,                                                     \
                    "[CUDA error] %s:%d  %s\n",                                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Only synchronizes in debug builds to avoid perf cost.
#ifdef DEBUG
#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        CUDA_CHECK_LAST();                                                      \
    } while (0)
#else
#define CUDA_SYNC_CHECK() CUDA_CHECK_LAST()
#endif

// --- cuBLAS error check ---
inline const char* cublas_get_error_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        default:                             return "UNKNOWN_CUBLAS_ERROR";
    }
}

#define CUBLAS_CHECK(expr)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (expr);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr,                                                     \
                    "[cuBLAS error] %s:%d  %s\n",                               \
                    __FILE__, __LINE__, cublas_get_error_string(_s));           \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
