#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase6b_multistage/gemm_cpasync.cuh"

// --- Phase 6b: real cp.async + 3-stage pipelining ---
// Same correctness suite as Phase 6a. The compute core (ldmatrix + mma.sync)
// is unchanged; only the global→smem path differs (cp.async DMA vs scalar stores).
//
// A failure here that passes in Phase 6a points to a cp.async predication bug
// (out-of-bounds chunk not zeroed) or a stage index error (wrong cs/fs modulo).
//
// Performance note: 3-stage (56 832 B smem) forces 1 block/SM on sm_120,
// halving active warps vs 2-stage. Perf regression vs 6a is expected and
// documented — the contribution is the DMA bypass + occupancy measurement.

class Phase6bKernelTest : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

// num_tiles=1 — exercises prologue-only path and cp_async_wait<0> epilogue branch
TEST_F(Phase6bKernelTest, Tile128_16x16x16) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_cpasync<128,128,32,2,4,3>(d); },
        16, 16, 16, tol);
    res.print("cpasync fp16 BM128 BN128 BK32 NS3 [16x16x16]    (1 K-tile, prologue only)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6bKernelTest, Tile128_128x128x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_cpasync<128,128,32,2,4,3>(d); },
        128, 128, 32, tol);
    res.print("cpasync fp16 BM128 BN128 BK32 NS3 [128x128x32]  (1 K-tile)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6bKernelTest, Tile128_256x256x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_cpasync<128,128,32,2,4,3>(d); },
        256, 256, 32, tol);
    res.print("cpasync fp16 BM128 BN128 BK32 NS3 [256x256x32]  (multi-block, 1 K-tile)");
    EXPECT_TRUE(res.passed);
}

// num_tiles=8 — exercises steady-state 3-stage loop (wait<1> path)
TEST_F(Phase6bKernelTest, Tile128_128x128x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_cpasync<128,128,32,2,4,3>(d); },
        128, 128, 256, tol);
    res.print("cpasync fp16 BM128 BN128 BK32 NS3 [128x128x256] (8 K-tiles, steady-state)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6bKernelTest, Tile128_256x256x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_cpasync<128,128,32,2,4,3>(d); },
        256, 256, 256, tol);
    res.print("cpasync fp16 BM128 BN128 BK32 NS3 [256x256x256]");
    EXPECT_TRUE(res.passed);
}
