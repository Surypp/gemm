#pragma once

// --- wgmma helpers ---
// Hopper (sm_90a) only.
//
// wgmma.mma_async operates on fragments 4× larger than mma.sync on Ampere.
// Executed by a warp group (4 warps = 128 threads) cooperatively.
// Asynchronous: issue wgmma.mma_async, then commit_group, then wait_group.
//
// Base shape: m64n8k16 for FP16 (minimum). Larger N dimensions via tiling.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- wgmma sync ---
// wgmma fence: ensure all preceding register writes are visible to wgmma
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

// Wait until at most `pending` groups remain outstanding
template <int pending>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(pending) : "memory");
}

// --- smem descriptor ---
// The wgmma instruction takes a 64-bit descriptor encoding smem address,
// stride, and swizzle mode of the A and B fragments.
// On Hopper, A must come from shared memory (not registers).
//
// Descriptor encoding (simplified — see PTX ISA 9.7.14):
//   [13:0]  base address bits [13:0] of shared mem pointer (4B aligned)
//   [29:16] stride in units of 16B
//   [61:62] swizzle mode (0=none, 1=32B, 2=64B, 3=128B)
__device__ __forceinline__
uint64_t make_smem_desc(const void* smem_ptr, int stride_bytes, int swizzle_bits = 3) {
    uint64_t smem_addr = static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t stride16  = static_cast<uint64_t>(stride_bytes / 16);
    return (smem_addr & 0x3FFF)
         | ((stride16 & 0x3FFF) << 16)
         | (static_cast<uint64_t>(swizzle_bits) << 62);
}

// --- wgmma m64n8k16 FP16→FP32 ---
// D(m64×n8) += A(m64×k16) × B(k16×n8)
// A and B are in shared memory (smem descriptors).
// d[4]: 4 FP32 accumulator registers per thread for n8 tile.
// Simplified: m64n8k16 (minimum wgmma shape) for clarity.
__device__ __forceinline__
void wgmma_m64n8k16_fp16(uint64_t desc_a, uint64_t desc_b, float d[4],
                          int scale_d = 1, int scale_a = 1, int scale_b = 1,
                          int trans_a = 0, int trans_b = 1) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0,%1,%2,%3}, %4, %5, %6, %7, %8, %9, %10;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b),
          "n"(scale_d), "n"(scale_a), "n"(scale_b),
          "n"(trans_a), "n"(trans_b)
    );
}

#endif // __CUDA_ARCH__ >= 900
