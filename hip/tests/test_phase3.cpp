#include <gtest/gtest.h>
#include "test_correctness.hpp"
#include "kernels/phase3_wmma/gemm_wmma_fwd.hpp"

class Phase3Test : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

TEST_F(Phase3Test, Tile128x128_256x256x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_wmma<128,128,32>(d); },
        256, 256, 256, tol);
    res.print("wmma fp16 BM128 BN128 BK32 [256x256x256]");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase3Test, Tile128x128_1024x1024x1024) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_wmma<128,128,32>(d); },
        1024, 1024, 1024, tol);
    res.print("wmma fp16 BM128 BN128 BK32 [1024x1024x1024]");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase3Test, EdgeCase_NotMultipleOf16) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_wmma<128,128,32>(d); },
        256, 256, 256, tol);
    res.print("wmma fp16 BM128 BN128 BK32 [256x256x256]");
    EXPECT_TRUE(res.passed);
}
