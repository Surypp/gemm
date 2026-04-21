#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>
#include "error_check.cuh"

namespace gemm {

// --- forward declarations ---
template <typename T> struct DeviceMatrix;

// --- HostMatrix ---
// Row-major host matrix with configurable leading dimension.
// All arithmetic is done in double for correctness comparison helpers.

template <typename T>
struct HostMatrix {
    std::vector<T> data;
    int rows, cols, ld;   // ld >= cols (row-major)

    HostMatrix() : rows(0), cols(0), ld(0) {}
    HostMatrix(int rows, int cols, int ld = -1)
        : rows(rows), cols(cols), ld(ld < 0 ? cols : ld)
    {
        data.resize(static_cast<size_t>(this->rows) * this->ld, T{0});
    }

    T& at(int r, int c)       { return data[r * ld + c]; }
    T  at(int r, int c) const { return data[r * ld + c]; }

    T*       ptr()       { return data.data(); }
    const T* ptr() const { return data.data(); }

    size_t bytes() const { return data.size() * sizeof(T); }

    // --- factories ---
    static HostMatrix zeros(int rows, int cols) {
        return HostMatrix(rows, cols);
    }

    static HostMatrix random(int rows, int cols,
                              float lo = -1.0f, float hi = 1.0f,
                              uint64_t seed = 42)
    {
        HostMatrix m(rows, cols);
<<<<<<< HEAD
        std::mt19937 rng(seed);
=======
        std::mt19937_64 rng(seed);
>>>>>>> 1572328 (refactor: restructured phase directories)
        std::uniform_real_distribution<float> dist(lo, hi);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                m.at(r, c) = static_cast<T>(dist(rng));
        return m;
    }

    static HostMatrix identity(int n) {
        HostMatrix m = zeros(n, n);
        for (int i = 0; i < n; ++i) m.at(i, i) = T{1};
        return m;
    }

    static HostMatrix from_device(const DeviceMatrix<T>& d);

    // --- correctness helpers ---
    // Max absolute difference between two matrices of the same shape.
    static double max_abs_diff(const HostMatrix& a, const HostMatrix& b) {
        assert(a.rows == b.rows && a.cols == b.cols);
        double worst = 0.0;
        for (int r = 0; r < a.rows; ++r)
            for (int c = 0; c < a.cols; ++c) {
                double diff = std::abs(static_cast<double>(a.at(r,c))
                                     - static_cast<double>(b.at(r,c)));
                if (diff > worst) worst = diff;
            }
        return worst;
    }

    // Returns (num_violations, first_violation_row, first_violation_col)
    struct CheckResult {
        int    violations;
        int    first_row, first_col;
        double computed, reference;
    };

    static CheckResult check(const HostMatrix& computed, const HostMatrix& ref,
                              double rtol, double atol)
    {
        assert(computed.rows == ref.rows && computed.cols == ref.cols);
        CheckResult r{0, -1, -1, 0.0, 0.0};
        for (int row = 0; row < ref.rows; ++row)
            for (int col = 0; col < ref.cols; ++col) {
                double c = static_cast<double>(computed.at(row, col));
                double x = static_cast<double>(ref.at(row, col));
                double threshold = atol + rtol * std::abs(x);
                if (std::abs(c - x) > threshold) {
                    if (r.violations == 0) {
                        r.first_row  = row;
                        r.first_col  = col;
                        r.computed   = c;
                        r.reference  = x;
                    }
                    ++r.violations;
                }
            }
        return r;
    }
};

// --- DeviceMatrix ---
// RAII wrapper; no copy constructor — all H↔D transfers must be explicit.

template <typename T>
struct DeviceMatrix {
    T*  ptr = nullptr;
    int rows, cols, ld;

    DeviceMatrix() : rows(0), cols(0), ld(0) {}
    DeviceMatrix(int rows, int cols, int ld = -1)
        : rows(rows), cols(cols), ld(ld < 0 ? cols : ld)
    {
        CUDA_CHECK(cudaMalloc(&ptr, bytes()));
        CUDA_CHECK(cudaMemset(ptr, 0, bytes()));
    }

    ~DeviceMatrix() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }

    // Disable copy to prevent accidental double-free
    DeviceMatrix(const DeviceMatrix&)            = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    DeviceMatrix(DeviceMatrix&& o) noexcept
        : ptr(o.ptr), rows(o.rows), cols(o.cols), ld(o.ld)
    { o.ptr = nullptr; }

    DeviceMatrix& operator=(DeviceMatrix&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr = o.ptr; rows = o.rows; cols = o.cols; ld = o.ld;
            o.ptr = nullptr;
        }
        return *this;
    }

    size_t bytes() const {
        return static_cast<size_t>(rows) * ld * sizeof(T);
    }

    void copy_from(const HostMatrix<T>& h) {
        assert(h.rows == rows && h.cols == cols);
        CUDA_CHECK(cudaMemcpy(ptr, h.ptr(), bytes(), cudaMemcpyHostToDevice));
    }

    void copy_to(HostMatrix<T>& h) const {
        assert(h.rows == rows && h.cols == cols);
        CUDA_CHECK(cudaMemcpy(h.ptr(), ptr, bytes(), cudaMemcpyDeviceToHost));
    }

    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, bytes())); }
};

// --- HostMatrix::from_device ---
template <typename T>
HostMatrix<T> HostMatrix<T>::from_device(const DeviceMatrix<T>& d) {
    HostMatrix<T> h(d.rows, d.cols, d.ld);
    CUDA_CHECK(cudaMemcpy(h.ptr(), d.ptr, d.bytes(), cudaMemcpyDeviceToHost));
    return h;
}

} // namespace gemm
