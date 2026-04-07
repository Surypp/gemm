#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase2_swizzle/gemm_swizzle.cuh"
#include "gemm/swizzle.cuh"

// --- Phase 2 ---
// IMPORTANT: verify the swizzle pattern itself before running any kernel.
// If SwizzlePattern::verify_all() fails, kernel results are meaningless.

class Phase2SwizzleUnitTest : public ::testing::Test {};

// --- swizzle math ---
TEST_F(Phase2SwizzleUnitTest, ConflictFree_BK32_FP16) {
    bool ok = SwizzlePattern<32, 2>::verify_all(/*verbose=*/true);
    EXPECT_TRUE(ok) << "SwizzlePattern<32,2> has bank conflicts";
}

TEST_F(Phase2SwizzleUnitTest, ConflictFree_BK64_FP16) {
    bool ok = SwizzlePattern<64, 2>::verify_all(true);
    EXPECT_TRUE(ok) << "SwizzlePattern<64,2> has bank conflicts";
}

TEST_F(Phase2SwizzleUnitTest, ConflictFree_BK32_FP32) {
    bool ok = SwizzlePattern<32, 4>::verify_all(true);
    EXPECT_TRUE(ok) << "SwizzlePattern<32,4> has bank conflicts";
}

// --- bijectivity ---
TEST_F(Phase2SwizzleUnitTest, Bijective_BK32_FP16) {
    for (int row = 0; row < 128; ++row) {
        bool seen[32] = {};
        for (int col = 0; col < 32; ++col) {
            int pc = SwizzlePattern<32, 2>::permute_col(row, col);
            ASSERT_GE(pc, 0);
            ASSERT_LT(pc, 32) << "permute_col out of range at row=" << row << " col=" << col;
            EXPECT_FALSE(seen[pc]) << "collision at row=" << row << " col=" << col;
            seen[pc] = true;
        }
    }
}

// --- kernel tests ---
class Phase2KernelTest : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

TEST_F(Phase2KernelTest, Tile64_128x128x128) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_swizzle<FP16Tag,64,64,32>(d); },
        128, 128, 128, tol);
    res.print("swizzle fp16 64x64x32 [128x128x128]");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase2KernelTest, Tile128_1024x1024x1024) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_swizzle<FP16Tag,128,128,32>(d); },
        1024, 1024, 1024, tol);
    res.print("swizzle fp16 128x128x32 [1024x1024x1024]");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase2KernelTest, Tile128_BK64_1024x1024x1024) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_swizzle<FP16Tag,128,128,64>(d); },
        1024, 1024, 1024, tol);
    res.print("swizzle fp16 128x128x64 [1024x1024x1024]");
    EXPECT_TRUE(res.passed);
}

// Non-tile-divisible dimensions
TEST_F(Phase2KernelTest, EdgeCase_NonDivisible) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_swizzle<FP16Tag,128,128,32>(d); },
        300, 300, 100, tol);
    res.print("swizzle fp16 128x128x32 [300x300x100 edge]");
    EXPECT_TRUE(res.passed);
}
