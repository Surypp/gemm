#pragma once

// --- TMA helpers for FP8 (UINT8 elements) ---
//
// Identical to phase6c_tma/gemm_tma.cuh but uses CU_TENSOR_MAP_DATA_TYPE_UINT8
// instead of FLOAT16.  The mbar helpers (mbar_init, mbar_wait, etc.) and
// tma_load_2d are shared: include gemm_tma.cuh for those.
//
// Do NOT include both headers in the same TU — the mbar/tma helpers from
// gemm_tma.cuh are header-only inline, so include that header for them and
// only use create_tma_descriptor_fp8 from here.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#if !defined(__CUDA_ARCH__)

// Creates a 2-D TMA descriptor for a UINT8 (FP8) matrix.
// global_ptr  : device pointer to the full matrix
// rows / cols : full matrix dimensions in elements
// box_rows / box_cols : tile dimensions (BM,BK for A; BK,BN for B)
inline CUtensorMap create_tma_descriptor_fp8(
    const void* global_ptr,
    uint64_t    rows,
    uint64_t    cols,
    uint64_t    box_rows,
    uint64_t    box_cols)
{
    CUtensorMap tma_map{};

    uint64_t global_dim[2]    = {cols, rows};
    uint64_t global_stride[1] = {cols};          // row stride in bytes (elem_size=1)
    uint32_t box_dim[2]       = {static_cast<uint32_t>(box_cols),
                                  static_cast<uint32_t>(box_rows)};
    uint32_t elem_stride[2]   = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<void*>(global_ptr),
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,     // SWIZZLE_NONE: no XOR de-swizzle needed
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "create_tma_descriptor_fp8 failed (box=[%u,%u]): %s\n",
                box_dim[0], box_dim[1], err_str ? err_str : "?");
        abort();
    }
    return tma_map;
}

#endif // !defined(__CUDA_ARCH__)
