#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <functional>
#include "timer.hpp"

namespace gemm {

// --- TFLOPS formula ---
// 2*M*N*K FLOPs (counting each multiply-add as 2 ops).
// Matches rocBLAS convention so pct_rocblas_peak is comparable.

inline double compute_tflops(long long M, long long N, long long K, double ms) {
    double flops = 2.0 * static_cast<double>(M)
                       * static_cast<double>(N)
                       * static_cast<double>(K);
    double seconds = ms * 1e-3;
    return flops / seconds / 1e12;
}

// --- BenchmarkConfig ---

struct BenchmarkConfig {
    int M, N, K;
    int warmup_iters = 5;
    int timed_iters  = 20;
    // sync_each=true inserts hipDeviceSynchronize between timed iters — debug only
    bool sync_each   = false;

    std::string label;  // e.g. "phase1_fp16_BM128_BN128_BK32"
};

// --- BenchmarkResult ---

struct BenchmarkResult {
    std::string label;
    int M, N, K;

    double mean_ms   = 0.0;
    double stddev_ms = 0.0;
    double min_ms    = 0.0;
    double tflops    = 0.0;
    double pct_rocblas_peak = 0.0;   // filled by bench_suite after rocBLAS run
};

// --- RunBenchmark ---
// All timed iterations are enclosed in individual event pairs so per-iteration
// latency is captured in TimingStats (not just the total).

inline BenchmarkResult run_benchmark(
    const BenchmarkConfig& cfg,
    std::function<void(hipStream_t)> kernel_fn,
    hipStream_t stream = 0)
{
    // Warmup
    for (int i = 0; i < cfg.warmup_iters; ++i) {
        kernel_fn(stream);
    }
    if (cfg.sync_each) {
        HIP_CHECK(hipDeviceSynchronize());
    }

    // Timed iterations
    TimingStats stats;
    GpuTimer timer;

    for (int i = 0; i < cfg.timed_iters; ++i) {
        timer.start(stream);
        kernel_fn(stream);
        float ms = timer.stop_ms(stream);
        stats.add(static_cast<double>(ms));
        if (cfg.sync_each) HIP_CHECK(hipDeviceSynchronize());
    }

    BenchmarkResult r;
    r.label    = cfg.label;
    r.M        = cfg.M;
    r.N        = cfg.N;
    r.K        = cfg.K;
    r.mean_ms  = stats.mean();
    r.stddev_ms= stats.stddev();
    r.min_ms   = stats.min();
    r.tflops   = compute_tflops(cfg.M, cfg.N, cfg.K, r.min_ms);
    return r;
}

} // namespace gemm
