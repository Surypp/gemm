#pragma once

// --- TMA helpers ---
// Guard structure:
//   host pass  (__CUDA_ARCH__ undefined) : only create_tma_descriptor_2d
//   device pass (__CUDA_ARCH__ >= 900)   : only tma_load_2d, mbar_init, mbar_wait
//
// cuda.h provides CUtensorMap, cuTensorMapEncodeTiled and friends (driver API).
// cuda_runtime.h alone does NOT expose these types.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cstdio>
#include <cstdlib>

// --- host-only: TMA descriptor creation ---
// cuTensorMapEncodeTiled is a driver API call; it is only valid on the host.
// Guard: !defined(__CUDA_ARCH__) excludes this from every device compilation pass.

#if !defined(__CUDA_ARCH__)

inline CUtensorMap create_tma_descriptor_2d(
    const void* global_ptr,
    uint64_t    rows,          // full matrix rows
    uint64_t    cols,          // full matrix cols
    uint64_t    box_rows,      // tile rows (BM or BK)
    uint64_t    box_cols,      // tile cols (BK or BN)
    uint32_t    elem_size,     // sizeof(scalar_t) in bytes
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE)
{
    CUtensorMap tma_map{};

    uint64_t global_dim[2]    = {cols, rows};
    uint64_t global_stride[1] = {cols * elem_size};  // row stride in bytes
    uint32_t box_dim[2]       = {static_cast<uint32_t>(box_cols),
                                  static_cast<uint32_t>(box_rows)};
    uint32_t elem_stride[2]   = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
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
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "cuTensorMapEncodeTiled failed (swizzle=%d, box=[%u,%u]): %s\n",
                (int)swizzle, box_dim[0], box_dim[1], err_str ? err_str : "?");
        abort();
    }
    return tma_map;
}

#endif // !defined(__CUDA_ARCH__)

// --- device-only: TMA load + mbarrier helpers ---
// Guard: #ifdef __CUDA_ARCH__ prevents host-pass compilation of bodies that use
// __cvta_generic_to_shared (device-only intrinsic, 32-bit return on device,
// but 64-bit stub on host x64 — causes "r" constraint mismatch if visible).
// sm_90a and sm_120 both satisfy __CUDA_ARCH__ >= 900.

#ifdef __CUDA_ARCH__

// Issues cp.async.bulk.tensor to copy a 2D tile from global into shared memory.
// Only thread (0,0) issues the instruction; mbar is signaled when done.
__device__ __forceinline__
void tma_load_2d(const CUtensorMap* tma_map,
                  void*              smem_dst,
                  uint64_t*          mbar,
                  int                coord_x,   // tile column offset in elements
                  int                coord_y)   // tile row offset in elements
{
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        uint32_t smem_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        uint32_t mbar_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
        // shared::cta — copy into local CTA shared memory (no cluster required).
        // shared::cluster requires explicit cluster launch dims; without it the
        // driver rejects the instruction at runtime ("invalid device function").
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(smem_ptr), "l"(tma_map),
               "r"(coord_x),  "r"(coord_y),
               "r"(mbar_ptr)
            : "memory"
        );
    }
}

// __syncthreads() ensures all threads see the init before any thread issues
// tma_load or mbar_arrive_expect_tx.
__device__ __forceinline__
void mbar_init(uint64_t* mbar, int count) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(addr), "r"(count));
    }
    __syncthreads();
}

// Combined arrive + expect_tx.  Decrements arrival_count by 1 AND adds
// `bytes` to the pending transaction count.  Callers must init with count=1.
//
// Workaround for ptxas 13.2 ICE (C7907) on `mbarrier.expect_tx`:
// the standalone instruction crashes ptxas on both sm_90a and sm_120.
// The combined `mbarrier.arrive.expect_tx` uses a different ptxas codepath
// and compiles cleanly.  Semantically equivalent when paired with
// mbar_init(count=1) instead of mbar_init(count=0).
__device__ __forceinline__
void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(addr), "r"(bytes)
            : "memory"
        );
    }
}

// Busy-poll until the mbarrier completes the phase with the given parity.
// After mbar_init (parity reset to 0) and TMA complete_tx, parity flips;
// waiting with parity=0 returns once that flip has occurred.
__device__ __forceinline__
void mbar_wait(uint64_t* mbar, uint32_t parity) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "WAIT: mbarrier.try_wait.parity.shared.b64 done, [%0], %1;\n"
        "@!done bra WAIT;\n"
        "}\n"
        :: "r"(addr), "r"(parity)
    );
}

#endif // __CUDA_ARCH__
