#pragma once

// Umbrella include for all gemm utility headers.
// Kernel launch wrappers are NOT included here to avoid pulling in device code
// into host-only translation units. Include per-phase headers directly.

#include "gemm/types.cuh"
#include "gemm/error_check.cuh"
#include "gemm/cuda_helpers.cuh"
#include "gemm/matrix.cuh"
#include "gemm/timer.cuh"
#include "gemm/benchmark.cuh"
#include "gemm/swizzle.cuh"
#include "gemm/pipeline.cuh"
