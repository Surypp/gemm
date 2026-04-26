#pragma once
#include <hip/hip_runtime.h>

__device__ __forceinline__
void hip_load_16b(void* smem_dst, const void* gmem_src) {
    float4 tmp;
    __builtin_memcpy(&tmp, gmem_src, 16);
    __builtin_memcpy(smem_dst, &tmp, 16);
}
__device__ __forceinline__ void hip_load_commit() {}
template<int N> __device__ __forceinline__ void hip_load_wait() {}

template<int NumStages>
struct StagedPipeline {
    __device__ __forceinline__ void producer_commit()  { hip_load_commit(); }
    __device__ __forceinline__ void consumer_wait()    { __syncthreads();   }
    __device__ __forceinline__ void consumer_release() { __syncthreads();   }
};
