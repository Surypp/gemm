#pragma once

#include "gemm/types.cuh"

#include "phase0_naive/gemm_naive.cuh"
#include "phase1_tiling/gemm_shmem.cuh"
#include "phase2_swizzle/gemm_swizzle.cuh"
#include "phase3_wmma/gemm_wmma.cuh"
#include "phase4_cp_async/gemm_pipeline.cuh"
#include "phase5_ptx_mma/gemm_mma_ptx.cuh"
#include "phase6a_ldmatrix/gemm_ldmatrix.cuh"
#include "phase6b_multistage/gemm_cpasync.cuh"
#include "phase6c_tma/gemm_tma_mmasync.cuh"

#include <stdexcept>
#include <string>

// --- Phase ---
enum class Phase {
    Naive    = 0,
    Shmem    = 1,
    Swizzle  = 2,
    Wmma     = 3,
    Pipeline = 4,
    PTX      = 5,
    LdMatrix = 6,
    Hopper   = 7,
    CpAsync  = 8,
    TMA      = 9,
};

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Naive:    return "naive";
        case Phase::Shmem:    return "shmem";
        case Phase::Swizzle:  return "swizzle";
        case Phase::Wmma:     return "wmma";
        case Phase::Pipeline: return "pipeline";
        case Phase::PTX:      return "ptx";
        case Phase::LdMatrix: return "ldmatrix";
        case Phase::Hopper:   return "hopper";
        case Phase::CpAsync:  return "cpasync";
        case Phase::TMA:      return "tma";
        default:              return "unknown";
    }
}

// --- TileConfig ---
struct TileConfig { int BM, BN, BK; };

// --- dispatch ---
// Phases 1-5 are FP16 only; FP32 goes through Naive or Shmem.
// Phase::Hopper throws on non-sm_90a builds (guarded at build time).

inline void dispatch_fp16(
    Phase phase, TileConfig tile,
    GemmDescRowMajor<FP16Tag>& desc,
    cudaStream_t stream = 0)
{
    auto [BM, BN, BK] = tile;

    switch (phase) {
    case Phase::Naive:
        launch_gemm_naive<FP16Tag>(desc, stream);
        break;

    case Phase::Shmem:
        // Phase1 uses dim3 block(BN, BM) — reject tiles with BM*BN > 1024
        if (BM * BN > 1024) throw std::runtime_error("Shmem: tile " + std::to_string(BM) + "x" + std::to_string(BN) + " needs " + std::to_string(BM*BN) + " threads/block (>1024 CUDA limit)");
        if      (BM==32  && BN==32  && BK==32) launch_gemm_shmem<FP16Tag, 32,  32, 32>(desc, stream);
        else throw std::runtime_error("Shmem: unsupported tile config");
        break;

    case Phase::Swizzle:
        if      (BM==64  && BN==64  && BK==32) launch_gemm_swizzle<FP16Tag, 64,  64, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==32) launch_gemm_swizzle<FP16Tag,128, 128, 32>(desc, stream);
        else if (BM==128 && BN==128 && BK==64) launch_gemm_swizzle<FP16Tag,128, 128, 64>(desc, stream);
        else throw std::runtime_error("Swizzle: unsupported tile config");
        break;

    case Phase::Wmma:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_wmma<128, 128, 32>(desc, stream);
        else if (BM==128 && BN==256 && BK==32) launch_gemm_wmma<128, 256, 32>(desc, stream);
        else throw std::runtime_error("Wmma: unsupported tile config");
        break;

    case Phase::Pipeline:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_pipeline<128,128,32,2,4,2>(desc, stream);
        else throw std::runtime_error("Pipeline: unsupported tile config (BK=64/NumStages=3 exceed 48KB smem on sm_120)");
        break;

    case Phase::PTX:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_mma_ptx<128,128,32,2,4>(desc, stream);
        else throw std::runtime_error("PTX: unsupported tile config (BK=64 exceeds 48KB smem on sm_120)");
        break;

    case Phase::LdMatrix:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_ldmatrix<128,128,32,2,4>(desc, stream);
        else throw std::runtime_error("LdMatrix: unsupported tile config");
        break;

    case Phase::Hopper:
        throw std::runtime_error("Phase::Hopper not available in this build (compile with GEMM_ENABLE_HOPPER=ON)");

    case Phase::CpAsync:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_cpasync<128,128,32,2,4,2>(desc, stream);
        else throw std::runtime_error("CpAsync: unsupported tile config");
        break;

    case Phase::TMA:
        if      (BM==128 && BN==128 && BK==32) launch_gemm_tma_mmasync<128,128,32,2,4,3>(desc, stream);
        else throw std::runtime_error("TMA: unsupported tile config");
        break;

    default:
        throw std::runtime_error("Unknown phase");
    }
}

inline void dispatch_fp32(
    Phase phase, TileConfig tile,
    GemmDescRowMajor<FP32Tag>& desc,
    cudaStream_t stream = 0)
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
        throw std::runtime_error("FP32 only supported for phases 0-1; use FP16 for phases 2+");
    }
}
