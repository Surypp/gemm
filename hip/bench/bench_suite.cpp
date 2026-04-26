#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>

#include "gemm/types.hpp"
#include "gemm/matrix.hpp"
#include "gemm/benchmark.hpp"
#include "kernels/dispatch.hpp"
#include "bench_results.hpp"

namespace bench_rocblas {
double measure_rocblas_fp16_tflops(int M, int N, int K, int iters);
double measure_rocblas_fp32_tflops(int M, int N, int K, int iters);
}

// --- problem sizes ---
struct ProblemSize { int M, N, K; const char* label; };

static const std::vector<ProblemSize> kProblems = {
    { 1024,  1024,  1024, "sq1k"},
    { 2048,  2048,  2048, "sq2k"},
    { 4096,  4096,  4096, "sq4k"},
    { 8192,  8192,  8192, "sq8k"},
    {  128,   128,  4096, "attn"},
    {  512,   512,   512, "bert"},
};

static const std::vector<TileConfig> kTiles = {
    { 32,  32, 32},
    { 64,  64, 32},
    {128, 128, 32},
    {128, 128, 64},
    {128, 256, 32},
};

static const std::vector<Phase> kPhases = {
    Phase::Naive,
    Phase::Shmem,
    Phase::Swizzle,
    Phase::Wmma,
    Phase::Pipeline,
};

// --- FP16 suite ---
ResultTable run_suite_fp16(int warmup, int iters,
                           const std::string& phase_filter,
                           const std::string& size_filter) {
    ResultTable table;

    std::vector<Phase> active_phases;
    for (auto p : kPhases) {
        if (phase_filter.empty() || phase_filter == phase_name(p))
            active_phases.push_back(p);
    }
    std::vector<ProblemSize> active_problems;
    for (auto& p : kProblems) {
        bool match_label = size_filter == p.label;
        bool match_dim   = size_filter == std::to_string(p.M) && p.M == p.N && p.M == p.K;
        if (size_filter.empty() || match_label || match_dim)
            active_problems.push_back(p);
    }

    if (active_phases.empty()) {
        fprintf(stderr, "no phase matched '%s'\n", phase_filter.c_str());
        return table;
    }
    if (active_problems.empty()) {
        fprintf(stderr, "no size matched '%s'\n", size_filter.c_str());
        return table;
    }

    // rocBLAS baselines per problem size
    std::vector<double> rocblas_tflops(active_problems.size());
    printf("Measuring rocBLAS FP16 baselines...\n");
    for (size_t pi = 0; pi < active_problems.size(); ++pi) {
        auto& p = active_problems[pi];
        rocblas_tflops[pi] = bench_rocblas::measure_rocblas_fp16_tflops(
            p.M, p.N, p.K, iters);
        printf("  rocBLAS FP16  %s : %.2f TFLOPS\n", p.label, rocblas_tflops[pi]);
    }

    for (auto phase : active_phases) {
        for (size_t pi = 0; pi < active_problems.size(); ++pi) {
            auto& prob = active_problems[pi];
            for (auto& tile : kTiles) {
                BenchmarkRow row;
                row.phase = phase_name(phase);
                row.dtype = "fp16";
                row.M = prob.M; row.N = prob.N; row.K = prob.K;
                row.BM = tile.BM; row.BN = tile.BN; row.BK = tile.BK;

                try {
                    gemm::DeviceMatrix<__half> dA(prob.M, prob.K);
                    gemm::DeviceMatrix<__half> dB(prob.K, prob.N);
                    gemm::DeviceMatrix<float>  dC(prob.M, prob.N);
                    {
                        auto hA = gemm::HostMatrix<__half>::random(prob.M, prob.K);
                        auto hB = gemm::HostMatrix<__half>::random(prob.K, prob.N);
                        dA.copy_from(hA);
                        dB.copy_from(hB);
                    }

                    GemmDescRowMajor<FP16Tag> desc;
                    desc.M   = prob.M; desc.N = prob.N; desc.K = prob.K;
                    desc.A   = dA.ptr; desc.B = dB.ptr; desc.C = dC.ptr;
                    desc.lda = prob.K; desc.ldb = prob.N; desc.ldc = prob.N;
                    desc.alpha = __half(1.0f); desc.beta = 0.0f;

                    for (int w = 0; w < warmup; ++w)
                        dispatch_fp16(phase, tile, desc);
                    HIP_CHECK(hipDeviceSynchronize());

                    gemm::TimingStats stats;
                    gemm::GpuTimer timer;
                    for (int it = 0; it < iters; ++it) {
                        timer.start();
                        dispatch_fp16(phase, tile, desc);
                        stats.add(timer.stop_ms());
                    }

                    row.mean_ms         = stats.mean();
                    row.stddev_ms       = stats.stddev();
                    row.min_ms          = stats.min();
                    row.tflops          = gemm::compute_tflops(prob.M, prob.N, prob.K, row.min_ms);
                    row.pct_rocblas_peak = 100.0 * row.tflops / rocblas_tflops[pi];

                } catch (const std::exception& e) {
                    row.error = e.what();
                }

                table.add(row);
                if (row.error.empty()) {
                    printf("  %-10s [%s] BM=%d BN=%d BK=%d : %.2f TFLOPS (%.1f%%)\n",
                           row.phase.c_str(), prob.label,
                           tile.BM, tile.BN, tile.BK,
                           row.tflops, row.pct_rocblas_peak);
                } else {
                    printf("  %-10s [%s] BM=%d BN=%d BK=%d : SKIP (%s)\n",
                           row.phase.c_str(), prob.label,
                           tile.BM, tile.BN, tile.BK, row.error.c_str());
                }
            }
        }
    }

    return table;
}
