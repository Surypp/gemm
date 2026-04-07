#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase6_advanced/gemm_ldmatrix.cuh"

// --- Phase 6a: ldmatrix + mma.sync ---
// Same correctness suite as Phase 5. The only difference is that A/B fragments
// are loaded via ldmatrix.sync instead of per-thread scalar reads from smem.
// A failure here that passes in Phase 5 points to an ldmatrix addressing bug.

class Phase6aKernelTest : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

TEST_F(Phase6aKernelTest, Tile128_16x16x16) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_ldmatrix<128,128,32,2,4>(d); },
        16, 16, 16, tol);
    res.print("ldmatrix fp16 BM128 BN128 BK32 [16x16x16]   (1 fragment, 1 K-tile)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6aKernelTest, Tile128_128x128x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_ldmatrix<128,128,32,2,4>(d); },
        128, 128, 32, tol);
    res.print("ldmatrix fp16 BM128 BN128 BK32 [128x128x32] (multi-frag, 1 K-tile)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6aKernelTest, Tile128_256x256x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_ldmatrix<128,128,32,2,4>(d); },
        256, 256, 32, tol);
    res.print("ldmatrix fp16 BM128 BN128 BK32 [256x256x32]  (single K-tile)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6aKernelTest, Tile128_128x128x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_ldmatrix<128,128,32,2,4>(d); },
        128, 128, 256, tol);
    res.print("ldmatrix fp16 BM128 BN128 BK32 [128x128x256] (single block, multi K-tile)");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase6aKernelTest, Tile128_256x256x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_ldmatrix<128,128,32,2,4>(d); },
        256, 256, 256, tol);
    res.print("ldmatrix fp16 BM128 BN128 BK32 [256x256x256]");
    EXPECT_TRUE(res.passed);
}
