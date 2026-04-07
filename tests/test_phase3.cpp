#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase3_wmma/gemm_wmma.cuh"

class Phase3Test : public ::testing::Test {
protected:
    // wmma is FP16→FP32; tolerance is the same as FP16 kernels
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

// Size not multiple of fragment dim (16)
TEST_F(Phase3Test, EdgeCase_NotMultipleOf16) {
    // 256×256×256 is a safe upper bound for this edge-case check
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_wmma<128,128,32>(d); },
        256, 256, 256, tol);
    res.print("wmma fp16 BM128 BN128 BK32 [256x256x256]");
    EXPECT_TRUE(res.passed);
}
