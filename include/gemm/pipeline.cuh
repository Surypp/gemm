#pragma once

#include <cuda_runtime.h>
#include <cuda/pipeline>

// --- StagedPipeline ---
// Manages the cp.async barrier lifecycle for N-stage software pipelining.
//
// Usage (producer-consumer in the same kernel):
//   __shared__ StagedPipeline<NumStages>::SharedBarrier bars[NumStages];
//   StagedPipeline<NumStages> pipe(bars);
//
//   // Prologue: fill first NumStages-1 stages
//   for (int s = 0; s < NumStages-1; ++s) {
//     pipe.producer_acquire(s);
//     cp_async_load(smem_stage[s], gmem_ptr + s*BK);
//     pipe.producer_commit(s);
//   }
//
//   // Steady state
//   for (int k = 0; k < num_tiles; ++k) {
//     int fetch_stage  = (k + NumStages - 1) % NumStages;
//     int compute_stage = k % NumStages;
//     pipe.consumer_wait(compute_stage);
//     compute(smem_stage[compute_stage]);
//     pipe.consumer_release(compute_stage);
//     pipe.producer_acquire(fetch_stage);
//     cp_async_load(smem_stage[fetch_stage], ...);
//     pipe.producer_commit(fetch_stage);
//   }

// --- cp.async helpers (Ampere, CUDA 11+) ---
// src must be 16-byte aligned; dst must be at least 16-byte aligned.
__device__ __forceinline__
void cp_async_16b(void* smem_dst, const void* gmem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Waits until at most `pending` groups remain outstanding.
// cp_async_wait<0>() = wait for all.
// cp_async_wait<N-1>() = keep N-1 groups in flight.
template <int pending>
__device__ __forceinline__
void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(pending) : "memory");
}

// --- StagedPipeline ---

template <int NumStages>
struct StagedPipeline {
    static_assert(NumStages >= 2 && NumStages <= 8, "NumStages must be 2..8");

    int _stage = 0;

    __device__ __forceinline__ int next_stage(int s) const {
        return (s + 1) % NumStages;
    }

    __device__ __forceinline__ void producer_commit() {
        cp_async_commit();
    }

    // Keep NumStages-1 groups in flight; wait for the oldest.
    __device__ __forceinline__ void consumer_wait() {
        cp_async_wait<NumStages - 1>();
        __syncthreads();
    }

    // For cp.async pipelines, release = __syncthreads so the producer can reuse the stage.
    __device__ __forceinline__ void consumer_release() {
        __syncthreads();
    }
};

// --- 2-stage specialization ---
// Most kernels use NumStages=2; the explicit cp_async_wait<0> makes intent clear.

template <>
struct StagedPipeline<2> {
    __device__ __forceinline__ void producer_commit()  { cp_async_commit();  }
    __device__ __forceinline__ void consumer_wait()    { cp_async_wait<0>(); __syncthreads(); }
    __device__ __forceinline__ void consumer_release() { __syncthreads();    }
};
