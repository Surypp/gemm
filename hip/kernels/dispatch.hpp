#pragma once

#include "gemm/types.hpp"

#include "phase0_naive/gemm_naive.hpp"
#include "phase1_tiling/gemm_shmem.hpp"
#include "phase2_swizzle/gemm_swizzle_fwd.hpp"
#include "phase3_wmma/gemm_wmma_fwd.hpp"
#include "phase4_pipeline/gemm_pipeline_fwd.hpp"

#include <stdexcept>
#include <string>

enum class Phase { Naive=0, Shmem=1, Swizzle=2, Wmma=3, Pipeline=4 };

struct TileConfig { int BM, BN, BK; };

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Naive:    return "naive";
        case Phase::Shmem:    return "shmem";
        case Phase::Swizzle:  return "swizzle";
        case Phase::Wmma:     return "wmma";
        case Phase::Pipeline: return "pipeline";
        default:              return "unknown";
    }
}

inline void dispatch_fp16(
    Phase phase, TileConfig tile,
    GemmDescRowMajor<FP16Tag>& desc,
    hipStream_t stream = 0)
{
    auto [BM, BN, BK] = tile;
    switch (phase) {
    case Phase::Naive:
        launch_gemm_naive<FP16Tag>(desc, stream);
        break;

    case Phase::Shmem:
        if (BM * BN > 1024) throw std::runtime_error("Shmem: tile exceeds 1024 threads/block");
        if      (BM==32  && BN==32  && BK==32) launch_gemm_shmem<FP16Tag, 32,  32, 32>(desc, stream);
        else throw std::runtime_error("Shmem: unsupported tile config");
        break;

    case Phase::Swizzle:
        if      (BM==64  && BN==64  && BK==32) launch_gemm_swizzle<FP16Tag,  64,  64, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==32) launch_gemm_swizzle<FP16Tag, 128, 128, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==64) launch_gemm_swizzle<FP16Tag, 128, 128, 64>(desc, stream);
        else throw std::runtime_error("Swizzle: unsupported tile config");
        break;

    case Phase::Wmma:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_wmma<128, 128, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==64) launch_gemm_wmma<128, 128, 64>(desc, stream);
        else if (BM==128 && BN==256 && BK==32) launch_gemm_wmma<128, 256, 32>(desc, stream);
        else throw std::runtime_error("Wmma: unsupported tile config");
        break;

    case Phase::Pipeline:
        // 128×256 uses 4×4 warp grid: 512 threads, FRAGS_M=2 FRAGS_N=4 (8 frags/warp) — same density as 128×128
        if      (BM==128 && BN==128 && BK==32) launch_gemm_pipeline<128, 128, 32, 2, 4, 2>(desc, stream);
        else if (BM==128 && BN==256 && BK==32) launch_gemm_pipeline<128, 256, 32, 4, 4, 2>(desc, stream);
        else throw std::runtime_error("Pipeline: unsupported tile config");
        break;

    default:
        throw std::runtime_error("Unknown phase");
    }
}

inline void dispatch_fp32(
    Phase phase, TileConfig tile,
    GemmDescRowMajor<FP32Tag>& desc,
    hipStream_t stream = 0)
{
    auto [BM, BN, BK] = tile;
    switch (phase) {
    case Phase::Naive:
        launch_gemm_naive<FP32Tag>(desc, stream);
        break;
    case Phase::Shmem:
        if      (BM==32  && BN==32  && BK==32) launch_gemm_shmem<FP32Tag, 32,  32, 32>(desc, stream);
        else if (BM==64  && BN==64  && BK==32) launch_gemm_shmem<FP32Tag, 64,  64, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==32) launch_gemm_shmem<FP32Tag,128, 128, 32>(desc, stream);
        else throw std::runtime_error("Shmem FP32: unsupported tile config");
        break;
    default:
        throw std::runtime_error("FP32 only supported for phases 0-1");
    }
}
