#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>

#include "gemm/types.cuh"
#include "gemm/matrix.cuh"
#include "gemm/benchmark.cuh"
#include "gemm/error_check.cuh"
#include "kernels/phase6e_fp8/gemm_fp8.cuh"
#include "bench_results.hpp"

namespace bench_cublas {
double measure_cublas_fp8_tflops(int M, int N, int K, int iters);
}

struct ProblemFP8 { int M, N, K; const char* label; };

static const std::vector<ProblemFP8> kProblemsFP8 = {
    { 1024, 1024, 1024, "sq1k" },
    { 2048, 2048, 2048, "sq2k" },
    { 4096, 4096, 4096, "sq4k" },
    { 8192, 8192, 8192, "sq8k" },
};

// Runs the FP8 mma.sync SM120 benchmark (Phase 6e).
// Single config: BM128/BN128/BK32/WM2/WN4/NS2.
// Baseline: cuBLAS FP8 (e4m3 x e5m2 -> f32), same dtype — like-for-like comparison.
// Returns early with empty table if sm_major < 9 (no FP8 mma.sync).
ResultTable run_suite_fp8(int warmup, int iters, const std::string& size_filter)
{
    ResultTable table;

    int sm_major;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0));
    if (sm_major < 9) {
        fprintf(stderr,
            "FP8 mma.sync requires sm_90a+ (sm_major >= 9). "
            "Current device: sm_%d0. Skipping.\n", sm_major);
        return table;
    }

    std::vector<ProblemFP8> active;
    for (auto& p : kProblemsFP8) {
        bool match_label = size_filter == p.label;
        bool match_dim   = size_filter == std::to_string(p.M)
                           && p.M == p.N && p.M == p.K;
        if (size_filter.empty() || match_label || match_dim)
            active.push_back(p);
    }
    if (active.empty()) {
        fprintf(stderr, "no size matched '%s'\n", size_filter.c_str());
        return table;
    }

    std::vector<double> fp8_base(active.size());

    printf("Measuring cuBLAS FP8 baselines (e4m3 x e5m2 -> f32)...\n");
    for (size_t pi = 0; pi < active.size(); ++pi) {
        auto& p = active[pi];
        fp8_base[pi] = bench_cublas::measure_cublas_fp8_tflops(p.M, p.N, p.K, iters);
        printf("  cuBLAS FP8   %s : %.2f TFLOPS\n", p.label, fp8_base[pi]);
    }

    for (size_t pi = 0; pi < active.size(); ++pi) {
        auto& prob = active[pi];

        BenchmarkRow row;
        row.phase = "fp8";
        row.dtype = "fp8_e4m3xe5m2";
        row.M = prob.M; row.N = prob.N; row.K = prob.K;
        row.BM = 128; row.BN = 128; row.BK = 32;

        try {
            gemm::DeviceMatrix<uint8_t> dA(prob.M, prob.K);
            gemm::DeviceMatrix<uint8_t> dB(prob.K, prob.N);
            gemm::DeviceMatrix<float>   dC(prob.M, prob.N);
            // Fill with 1.0 in each FP8 format so mma.sync is arithmetically active.
            CUDA_CHECK(cudaMemset(dA.ptr, 0x38, (size_t)prob.M * prob.K)); // 1.0 e4m3
            CUDA_CHECK(cudaMemset(dB.ptr, 0x3C, (size_t)prob.K * prob.N)); // 1.0 e5m2

            GemmDescFP8 desc;
            desc.M = prob.M; desc.N = prob.N; desc.K = prob.K;
            desc.A = dA.ptr; desc.B = dB.ptr; desc.C = dC.ptr;
            desc.lda = prob.K; desc.ldb = prob.N; desc.ldc = prob.N;
            desc.alpha = 1.0f; desc.beta = 0.0f;

            for (int w = 0; w < warmup; ++w)
                launch_gemm_fp8<128, 128, 32, 2, 4, 2>(desc);
            CUDA_CHECK(cudaDeviceSynchronize());

            gemm::TimingStats stats;
            gemm::GpuTimer    timer;
            for (int it = 0; it < iters; ++it) {
                timer.start();
                launch_gemm_fp8<128, 128, 32, 2, 4, 2>(desc);
                stats.add(timer.stop_ms());
            }

            row.min_ms         = stats.min();
            row.mean_ms        = stats.mean();
            row.stddev_ms      = stats.stddev();
            row.tflops         = gemm::compute_tflops(prob.M, prob.N, prob.K, row.min_ms);
            row.pct_cublas_peak = fp8_base[pi] > 0.0
                                 ? 100.0 * row.tflops / fp8_base[pi]
                                 : 0.0;

        } catch (const std::exception& e) {
            row.error = e.what();
        }

        table.add(row);
        if (row.error.empty())
            printf("  fp8 [%s] BM=128 BN=128 BK=32 NS=2 : %.2f TFLOPS (%.1f%% vs cuBLAS FP8)\n",
                   prob.label, row.tflops, row.pct_cublas_peak);
        else
            printf("  fp8 [%s] : SKIP (%s)\n", prob.label, row.error.c_str());
    }

    return table;
}
