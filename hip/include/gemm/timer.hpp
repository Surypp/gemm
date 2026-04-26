#pragma once

#include <hip/hip_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include "error_check.hpp"

namespace gemm {

// --- GpuTimer ---
// Usage:
//   GpuTimer t;
//   t.start();
//   kernel<<<...>>>();
//   float ms = t.stop_ms();  // synchronizes

struct GpuTimer {
    hipEvent_t _start, _stop;

    GpuTimer() {
        HIP_CHECK(hipEventCreate(&_start));
        HIP_CHECK(hipEventCreate(&_stop));
    }
    ~GpuTimer() {
        hipEventDestroy(_start);
        hipEventDestroy(_stop);
    }
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(_start, stream));
    }

    // Returns elapsed milliseconds; synchronizes the stop event.
    float stop_ms(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(_stop, stream));
        HIP_CHECK(hipEventSynchronize(_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

// --- CpuTimer ---

struct CpuTimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point _start;

    void start()  { _start = Clock::now(); }

    double elapsed_ms() const {
        auto dur = Clock::now() - _start;
        return std::chrono::duration<double, std::milli>(dur).count();
    }
};

// --- TimingStats ---

struct TimingStats {
    std::vector<double> samples_ms;

    void add(double ms) { samples_ms.push_back(ms); }

    double mean() const {
        if (samples_ms.empty()) return 0.0;
        double s = 0.0;
        for (double v : samples_ms) s += v;
        return s / samples_ms.size();
    }

    double min() const {
        if (samples_ms.empty()) return 0.0;
        return *std::min_element(samples_ms.begin(), samples_ms.end());
    }

    double max() const {
        if (samples_ms.empty()) return 0.0;
        return *std::max_element(samples_ms.begin(), samples_ms.end());
    }

    double stddev() const {
        if (samples_ms.size() < 2) return 0.0;
        double m = mean();
        double var = 0.0;
        for (double v : samples_ms) var += (v - m) * (v - m);
        return std::sqrt(var / (samples_ms.size() - 1));
    }
};

} // namespace gemm
