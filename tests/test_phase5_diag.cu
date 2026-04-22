// tests/test_phase5_diag.cu
//
// Diagnostic tests for the Phase 5 PTX kernel.
//
// Unlike the identity micro-test, this uses:
//   1. non-identity data (sequential values) — detects K↔N swaps in B load
//   2. the kernel's exact smem layout and strides (BK+8=40, BN+8=136)
//   3. CPU reference computation for every output element
//
// Run with: gemm_tests --gtest_filter="Phase5Diag*"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>

#include "gemm/error_check.cuh"
#include "kernels/phase5_ptx_mma/mma_ptx_utils.cuh"

// --- DiagResult ---
struct DiagResult {
    int row, col;
    float computed;
    float expected;
};

// --- test 1: B load ---
// Load known data into smem_B with stride=136, read it back via load_matrix_b_16x8,
// and verify the register contents match the PTX ISA spec.
struct BLoadResult {
    int lane;
    int reg_idx;  // 0 or 1
    uint32_t value;
    uint32_t expected;
};

__global__ void diag_b_load_test(BLoadResult* out) {
    const int lane = threadIdx.x;  // 0..31

    // Kernel's exact B layout: smem_B[BK][BN+8] = [32][136]
    // Fill only the 16x8 sub-tile at [0..15][0..7]
    __shared__ __half smem_B[32][136];

    // B[k][n] = k * 8 + n (as FP16) — not symmetric, so K↔N swap is visible
    for (int idx = lane; idx < 32 * 136; idx += 32) {
        int k = idx / 136, n = idx % 136;
        if (k < 16 && n < 8)
            smem_B[k][n] = __float2half(static_cast<float>(k * 8 + n));
        else
            smem_B[k][n] = __half(0);
    }
    __syncthreads();

    uint32_t b[2];
    load_matrix_b_16x8(&smem_B[0][0], 136, b, lane);

    // Expected per PTX ISA:
    // b[0] = {B[k0, n], B[k0+1, n]}  where k0=(lane%4)*2, n=lane/4
    // b[1] = {B[k0+8, n], B[k0+9, n]}
    int k0 = (lane % 4) * 2;
    int n  = lane / 4;

    auto make_expected = [](int k_lo, int n_lo, int k_hi, int n_hi) -> uint32_t {
        __half lo = __float2half(static_cast<float>(k_lo * 8 + n_lo));
        __half hi = __float2half(static_cast<float>(k_hi * 8 + n_hi));
        uint16_t lo_bits = *reinterpret_cast<uint16_t*>(&lo);
        uint16_t hi_bits = *reinterpret_cast<uint16_t*>(&hi);
        return (static_cast<uint32_t>(hi_bits) << 16) | lo_bits;
    };

    BLoadResult* w = out + lane * 2;
    w[0] = {lane, 0, b[0], make_expected(k0,   n, k0+1, n)};
    w[1] = {lane, 1, b[1], make_expected(k0+8, n, k0+9, n)};
}

// --- test 2: A load ---
struct ALoadResult {
    int lane;
    int reg_idx;  // 0..3
    uint32_t value;
    uint32_t expected;
};

__global__ void diag_a_load_test(ALoadResult* out) {
    const int lane = threadIdx.x;

    // Kernel's exact A layout: smem_A[BM][BK+8] = [128][40]
    __shared__ __half smem_A[128][40];

    for (int idx = lane; idx < 128 * 40; idx += 32) {
        int r = idx / 40, c = idx % 40;
        if (r < 16 && c < 16)
            smem_A[r][c] = __float2half(static_cast<float>(r * 16 + c));
        else
            smem_A[r][c] = __half(0);
    }
    __syncthreads();

    uint32_t a[4];
    load_matrix_a_16x16(&smem_A[0][0], 40, a, lane);

    // Expected per PTX ISA:
    // row0 = (lane/4)%8, row1 = row0+8, col0 = (lane%4)*2
    // a[0] = {A[row0, col0], A[row0, col0+1]}
    // a[1] = {A[row1, col0], A[row1, col0+1]}
    // a[2] = {A[row0, col0+8], A[row0, col0+9]}
    // a[3] = {A[row1, col0+8], A[row1, col0+9]}
    int row0 = (lane / 4) % 8;
    int row1 = row0 + 8;
    int col0 = (lane % 4) * 2;

    auto make_expected = [](int r, int c0, int c1) -> uint32_t {
        __half lo = __float2half(static_cast<float>(r * 16 + c0));
        __half hi = __float2half(static_cast<float>(r * 16 + c1));
        uint16_t lo_bits = *reinterpret_cast<uint16_t*>(&lo);
        uint16_t hi_bits = *reinterpret_cast<uint16_t*>(&hi);
        return (static_cast<uint32_t>(hi_bits) << 16) | lo_bits;
    };

    ALoadResult* w = out + lane * 4;
    w[0] = {lane, 0, a[0], make_expected(row0, col0,   col0+1)};
    w[1] = {lane, 1, a[1], make_expected(row1, col0,   col0+1)};
    w[2] = {lane, 2, a[2], make_expected(row0, col0+8, col0+9)};
    w[3] = {lane, 3, a[3], make_expected(row1, col0+8, col0+9)};
}

// --- test 3: full MMA ---
// A = sequential 16×16, B = sequential 16×16
// Uses kernel's exact smem shapes and strides; compares to CPU reference.

__global__ void diag_full_mma_kernel_strides(
    const __half* A_global, const __half* B_global,
    DiagResult* out)
{
    const int lane = threadIdx.x;

    __shared__ __half smem_A[128][40];   // [BM][BK+8]
    __shared__ __half smem_B[32][136];   // [BK][BN+8]

    for (int idx = lane; idx < 128 * 40; idx += 32) {
        int r = idx / 40, c = idx % 40;
        if (r < 16 && c < 16)
            smem_A[r][c] = A_global[r * 16 + c];
        else
            smem_A[r][c] = __half(0);
    }

    for (int idx = lane; idx < 32 * 136; idx += 32) {
        int r = idx / 136, c = idx % 136;
        if (r < 16 && c < 16)
            smem_B[r][c] = B_global[r * 16 + c];
        else
            smem_B[r][c] = __half(0);
    }
    __syncthreads();

    uint32_t a_reg[4];
    load_matrix_a_16x16(&smem_A[0][0], 40, a_reg, lane);

    uint32_t b_reg[2], b_reg_second[2];
    load_matrix_b_16x8(&smem_B[0][0],  136, b_reg,        lane);
    load_matrix_b_16x8(&smem_B[0][8],  136, b_reg_second, lane);

    float d[8] = {};
    mma_m16n8k16(a_reg, b_reg,        d);
    mma_m16n8k16(a_reg, b_reg_second, d + 4);

    int r0 = lane >> 2;
    int r1 = (lane >> 2) + 8;
    int c0 = (lane & 3) * 2;
    int c1 = (lane & 3) * 2 + 1;
    int c2 = 8 + (lane & 3) * 2;
    int c3 = 8 + (lane & 3) * 2 + 1;

    DiagResult* w = out + lane * 8;
    w[0] = {r0, c0, d[0], 0.0f};
    w[1] = {r0, c1, d[1], 0.0f};
    w[2] = {r1, c0, d[2], 0.0f};
    w[3] = {r1, c1, d[3], 0.0f};
    w[4] = {r0, c2, d[4], 0.0f};
    w[5] = {r0, c3, d[5], 0.0f};
    w[6] = {r1, c2, d[6], 0.0f};
    w[7] = {r1, c3, d[7], 0.0f};
}

// ---

class Phase5DiagTest : public ::testing::Test {};

// --- B load registers ---
TEST_F(Phase5DiagTest, BLoadRegisters) {
    constexpr int N = 32 * 2;
    BLoadResult* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(BLoadResult)));

    diag_b_load_test<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<BLoadResult> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, N * sizeof(BLoadResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    printf("\n=== B Load Register Diagnostic (stride=136) ===\n");
    printf("Pattern: B[k][n] = k*8+n, expecting b[0]={B[k0,n],B[k0+1,n]}\n\n");
    printf("%-5s %-4s %-12s %-12s %s\n", "lane", "reg", "got", "expected", "status");

    int fails = 0;
    for (auto& r : h) {
        bool ok = (r.value == r.expected);
        if (!ok) ++fails;
        printf("%-5d %-4d 0x%08X   0x%08X   %s\n",
               r.lane, r.reg_idx, r.value, r.expected, ok ? "OK" : "FAIL");
    }
    printf("\n%d / %d registers correct\n\n", N - fails, N);
    EXPECT_EQ(fails, 0) << fails << " B-load registers have wrong values";
}

// --- A load registers ---
TEST_F(Phase5DiagTest, ALoadRegisters) {
    constexpr int N = 32 * 4;
    ALoadResult* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(ALoadResult)));

    diag_a_load_test<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<ALoadResult> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, N * sizeof(ALoadResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    printf("\n=== A Load Register Diagnostic (stride=40) ===\n");
    printf("Pattern: A[r][c] = r*16+c\n\n");
    printf("%-5s %-4s %-12s %-12s %s\n", "lane", "reg", "got", "expected", "status");

    int fails = 0;
    for (auto& r : h) {
        bool ok = (r.value == r.expected);
        if (!ok) ++fails;
        printf("%-5d %-4d 0x%08X   0x%08X   %s\n",
               r.lane, r.reg_idx, r.value, r.expected, ok ? "OK" : "FAIL");
    }
    printf("\n%d / %d registers correct\n\n", N - fails, N);
    EXPECT_EQ(fails, 0) << fails << " A-load registers have wrong values";
}

// --- full MMA ---
TEST_F(Phase5DiagTest, FullMmaKernelStrides) {
    // A[i][j] = (i*16+j) * 0.01, B same — small values to avoid FP16 overflow
    std::vector<__half> h_A(16 * 16), h_B(16 * 16);
    for (int i = 0; i < 256; ++i) {
        h_A[i] = __float2half(static_cast<float>(i) * 0.01f);
        h_B[i] = __float2half(static_cast<float>(i) * 0.01f);
    }

    // CPU reference
    float ref[16][16] = {};
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            for (int k = 0; k < 16; ++k)
                ref[r][c] += __half2float(h_A[r * 16 + k]) * __half2float(h_B[k * 16 + c]);

    __half *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), 256 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 256 * sizeof(__half), cudaMemcpyHostToDevice));

    constexpr int N = 32 * 8;
    DiagResult* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(DiagResult)));

    diag_full_mma_kernel_strides<<<1, 32>>>(d_A, d_B, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<DiagResult> h_res(N);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_out, N * sizeof(DiagResult), cudaMemcpyDeviceToHost));

    for (auto& r : h_res)
        r.expected = ref[r.row][r.col];

    printf("\n=== Full MMA Diagnostic (kernel strides: A=40, B=136) ===\n");
    printf("A[i][j] = i*16+j * 0.01, B = same\n\n");

    int fails = 0;
    float max_err = 0;
    for (auto& r : h_res) {
        float err = fabsf(r.computed - r.expected);
        float rel = (r.expected != 0) ? err / fabsf(r.expected) : err;
        bool ok = rel < 0.02f;  // 2% tolerance for FP16
        if (!ok) {
            ++fails;
            if (fails <= 20) {
                printf("  FAIL [%2d][%2d]: computed=%.6f expected=%.6f err=%.2e\n",
                       r.row, r.col, r.computed, r.expected, err);
            }
        }
        if (err > max_err) max_err = err;
    }

    printf("\n%d / %d elements correct (max abs error = %.2e)\n\n",
           N - fails, N, max_err);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_out));

    EXPECT_EQ(fails, 0) << fails << " output elements are wrong";
}
