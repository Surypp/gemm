#pragma once

// --- TMA helpers ---
// Hopper (sm_90a) only.
// TMA copies tiles from HBM to shared memory asynchronously, bypassing threads.
// This frees compute resources during the copy and achieves higher bandwidth
// than cp.async because TMA operates on full cache lines in bulk.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#include <cuda_runtime.h>
#include <cuda/barrier>

// --- descriptor setup (host side) ---
// A CUtensorMap encodes the tensor shape, strides, and swizzle mode.
// Created once on the host and passed to the kernel as a const pointer;
// the kernel references it via a mbarrier mechanism.
//
// CUtensorMap is defined in cuda_runtime.h / driver_types.h when CUDA >= 12.0.

inline CUtensorMap create_tma_descriptor_2d(
    const void* global_ptr,
    uint64_t    rows,          // full matrix rows
    uint64_t    cols,          // full matrix cols
    uint64_t    box_rows,      // tile rows (BM or BK)
    uint64_t    box_cols,      // tile cols (BK or BN)
    uint32_t    elem_size,     // sizeof(scalar_t) in bytes
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B)
{
    CUtensorMap tma_map{};

    uint64_t global_dim[2]  = {cols, rows};
    uint64_t global_stride[1] = {cols * elem_size};
    uint32_t box_dim[2]     = {static_cast<uint32_t>(box_cols),
                                static_cast<uint32_t>(box_rows)};
    uint32_t elem_stride[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,   // adapt for BF16/FP32
        2,                                  // rank
        const_cast<void*>(global_ptr),
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    return tma_map;
}

// --- TMA load (device side) ---
// Issues cp.async.bulk.tensor to load a tile from global memory to shared.
// Only one thread per block should issue this instruction; completion is
// signaled via the mbarrier pointed to by mbar.

__device__ __forceinline__
void tma_load_2d(const CUtensorMap* tma_map,
                  void*              smem_dst,
                  uint64_t*          mbar,     // shared memory mbarrier
                  int                coord_x,  // tile column index (in elements)
                  int                coord_y)  // tile row index
{
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(__cvta_generic_to_shared(smem_dst)),
               "l"(tma_map),
               "r"(coord_x), "r"(coord_y),
               "r"(__cvta_generic_to_shared(mbar))
            : "memory"
        );
    }
}

// --- mbarrier ---
// mbarrier replaces __syncthreads() for TMA completion synchronization.

__device__ __forceinline__
void mbar_init(uint64_t* mbar, int expected_tx_count) {
    if (threadIdx.x == 0 && threadIdx.y == 0)
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(__cvta_generic_to_shared(mbar)), "r"(expected_tx_count));
    __syncthreads();
}

__device__ __forceinline__
void mbar_wait(uint64_t* mbar, uint64_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT: mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
        "@!done bra WAIT;\n"
        "}\n"
        :: "r"(__cvta_generic_to_shared(mbar)), "l"(phase)
    );
}

#endif // __CUDA_ARCH__ >= 900
