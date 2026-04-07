#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "error_check.cuh"

namespace gemm {

// --- DeviceInfo ---

struct DeviceInfo {
    int    device_id;
    char   name[256];
    int    compute_major;
    int    compute_minor;
    size_t total_global_mem_bytes;
    int    sm_count;
    int    max_shared_mem_per_block_bytes;
    int    max_shared_mem_per_sm_bytes;
    int    warp_size;
    int    max_threads_per_sm;
    int    max_threads_per_block;
    int    l2_cache_size_bytes;
    int    clock_rate_khz;
    int    memory_bus_width_bits;
    int    memory_clock_rate_khz;
};

inline DeviceInfo query_device(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceInfo info{};
    info.device_id                       = device_id;
    info.compute_major                   = prop.major;
    info.compute_minor                   = prop.minor;
    info.total_global_mem_bytes          = prop.totalGlobalMem;
    info.sm_count                        = prop.multiProcessorCount;
    info.max_shared_mem_per_block_bytes  = prop.sharedMemPerBlock;
    info.max_shared_mem_per_sm_bytes     = prop.sharedMemPerMultiprocessor;
    info.warp_size                       = prop.warpSize;
    info.max_threads_per_sm              = prop.maxThreadsPerMultiProcessor;
    info.max_threads_per_block           = prop.maxThreadsPerBlock;
    info.l2_cache_size_bytes             = prop.l2CacheSize;
    info.memory_bus_width_bits           = prop.memoryBusWidth;
#if CUDART_VERSION >= 13000
    // clockRate and memoryClockRate removed in CUDA 13
    info.clock_rate_khz                  = 0;
    info.memory_clock_rate_khz           = 0;
#else
    info.clock_rate_khz                  = prop.clockRate;
    info.memory_clock_rate_khz           = prop.memoryClockRate;
#endif
    for (int i = 0; i < 256; ++i) info.name[i] = prop.name[i];
    return info;
}

inline void print_device_info(const DeviceInfo& info) {
    printf("GPU %d : %s  (sm_%d%d)\n",
           info.device_id, info.name,
           info.compute_major, info.compute_minor);
    printf("  SMs              : %d\n",      info.sm_count);
    printf("  Global mem       : %.1f GB\n", info.total_global_mem_bytes / 1e9);
    printf("  L2 cache         : %.1f MB\n", info.l2_cache_size_bytes / 1e6);
    printf("  Shared mem/SM    : %.0f KB\n", info.max_shared_mem_per_sm_bytes / 1e3);
    printf("  Max threads/SM   : %d\n",      info.max_threads_per_sm);
    printf("  Memory bandwidth : %.1f GB/s\n",
           2.0 * info.memory_clock_rate_khz * 1e3
               * (info.memory_bus_width_bits / 8) / 1e9);
}

// --- warp utilities ---
#ifdef __CUDACC__
__device__ __forceinline__ int lane_id()  { return threadIdx.x % 32; }
__device__ __forceinline__ int warp_id()  { return threadIdx.x / 32; }
#endif

// Call before the first launch of a kernel that uses large smem.
inline void set_max_dynamic_smem(const void* func, int smem_bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(
        func,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
}

} // namespace gemm
