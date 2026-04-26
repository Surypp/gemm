#include <gtest/gtest.h>
#include "test_correctness.hpp"
#include "kernels/phase4_pipeline/gemm_pipeline_fwd.hpp"

class Phase4Test : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

TEST_F(Phase4Test, Pipeline2_Tile128_512x512x512) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_pipeline<128,128,32,2,4,2>(d); },
        512, 512, 512, tol);
    res.print("pipeline fp16 BM128 BN128 BK32 S2 [512x512x512]");
    EXPECT_TRUE(res.passed);
}

TEST_F(Phase4Test, Pipeline3_Tile128_1024x1024x1024) {
    GTEST_SKIP() << "NumStages=3 (BK=32) not instantiated";
}

TEST_F(Phase4Test, Pipeline2_BK64_1024x1024x1024) {
    GTEST_SKIP() << "BK=64 not instantiated";
}
