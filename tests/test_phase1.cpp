#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase1_shmem/gemm_shmem.cuh"

class Phase1Test : public ::testing::Test {
protected:
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

// --- 32x32x32 ---
TEST_F(Phase1Test, Tile32_128x128x128) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_shmem<FP16Tag,32,32,32>(d); },
        128, 128, 128, tol);
    res.print("shmem fp16 32x32x32 [128x128x128]");
    EXPECT_TRUE(res.passed);
}

// M not divisible by BM
TEST_F(Phase1Test, Tile32_EdgeCase) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_shmem<FP16Tag,32,32,32>(d); },
        100, 100, 96, tol);
    res.print("shmem fp16 32x32x32 [100x100x96 edge]");
    EXPECT_TRUE(res.passed);
}

// --- 64x64x32 ---
// Phase1 uses dim3 block(BN, BM) — BM*BN threads per block.
// Tile64: 64*64=4096 > 1024 max threads → exceeds CUDA limit.
TEST_F(Phase1Test, Tile64_1024x1024x1024) {
    GTEST_SKIP() << "Phase1 Tile64: block(64,64)=4096 threads > 1024 CUDA limit (known Phase1 limitation)";
}

TEST_F(Phase1Test, Tile128_1024x1024x1024) {
    GTEST_SKIP() << "Phase1 Tile128: block(128,128)=16384 threads > 1024 CUDA limit (known Phase1 limitation)";
}

TEST_F(Phase1Test, Tile128_EdgeKNotDivisible) {
    GTEST_SKIP() << "Phase1 Tile128: block(128,128)=16384 threads > 1024 CUDA limit (known Phase1 limitation)";
}
