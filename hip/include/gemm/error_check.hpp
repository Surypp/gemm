#pragma once

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// --- HIP error check ---
// Implemented as macros so __FILE__ and __LINE__ resolve to the call site.

#define HIP_CHECK(expr)                                                         \
    do {                                                                        \
        hipError_t _e = (expr);                                                 \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr,                                                     \
                    "[HIP error] %s:%d  %s\n",                                  \
                    __FILE__, __LINE__, hipGetErrorString(_e));                 \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define HIP_CHECK_LAST() HIP_CHECK(hipGetLastError())

// Only synchronizes in debug builds to avoid perf cost.
#ifdef DEBUG
#define HIP_SYNC_CHECK()                                                        \
    do {                                                                        \
        HIP_CHECK(hipDeviceSynchronize());                                      \
        HIP_CHECK_LAST();                                                       \
    } while (0)
#else
#define HIP_SYNC_CHECK() HIP_CHECK_LAST()
#endif

// --- rocBLAS error check ---
#define ROCBLAS_CHECK(expr) \
    do { rocblas_status _s = (expr); \
         if (_s != rocblas_status_success) { \
             fprintf(stderr, "[rocBLAS] %s:%d  %s\n", \
                     __FILE__, __LINE__, rocblas_status_to_string(_s)); \
             std::abort(); } } while(0)
