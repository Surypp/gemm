#include <gtest/gtest.h>
#include "test_correctness.hpp"
#include "kernels/phase0_naive/gemm_naive.hpp"

// --- Phase 0: naive GEMM ---

class Phase0Test : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

TEST_F(Phase0Test, Small_128x128x128) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_naive<FP16Tag>(d); },
        128, 128, 128, tol);
    res.print("naive fp16 128x128x128");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase0Test, Medium_512x512x512) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_naive<FP16Tag>(d); },
        512, 512, 512, tol);
    res.print("naive fp16 512x512x512");
    EXPECT_TRUE(res.passed);
}

// M not divisible by block dim
TEST_F(Phase0Test, EdgeCase_NonDivisible_M) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_naive<FP16Tag>(d); },
        100, 200, 300, tol);
    res.print("naive fp16 100x200x300");
    EXPECT_TRUE(res.passed);
}

// K=1: trivial dot product
TEST_F(Phase0Test, EdgeCase_K1) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_naive<FP16Tag>(d); },
        64, 64, 1, tol);
    res.print("naive fp16 64x64x1");
    EXPECT_TRUE(res.passed);
}
