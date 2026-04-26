#include <gtest/gtest.h>
#include "test_correctness.cuh"
#include "kernels/phase6c_tma/gemm_tma_mmasync.cuh"

// --- Phase 6c: TMA multi-stage + mma.sync ---
// Same correctness suite as Phase 6a/6b.  The compute core (ldmatrix + mma.sync)
// is unchanged; only the smem fill path changes (TMA cp.async.bulk vs cp.async).
//
// Guard: sm_90a or sm_120 required.  A100 (sm_80) skipped — no TMA hardware.
//
// A failure here that passes in Phase 6b points to:
//   - TMA descriptor encoding error (wrong rows/cols/stride)
//   - Coordinate mismatch (coord_x/coord_y swapped or wrong tile offset)
//   - Mbarrier protocol error (wait before init, wrong count, wrong parity)
//   - smem slot collision (cs == fss, violates the pipeline invariant)

class Phase6cKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        int sm_major;
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0);
        if (sm_major < 9) GTEST_SKIP() << "Phase 6c requires sm_90a+ (no TMA on sm_80)";
    }
    TolerancePolicy tol = TolerancePolicy::for_fp16();
};

// 1 K-tile — exercises prologue-only path; main loop hits the epilogue branch only
TEST_F(Phase6cKernelTest, Tile128_16x16x16) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        16, 16, 16, tol);
    res.print("tma fp16 BM128 BN128 BK32 NS2 [16x16x16]    (1 K-tile, prologue only)");
    EXPECT_TRUE(res.passed);
}

// 1 K-tile, single block
TEST_F(Phase6cKernelTest, Tile128_128x128x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        128, 128, 32, tol);
    res.print("tma fp16 BM128 BN128 BK32 NS2 [128x128x32]  (1 K-tile)");
    EXPECT_TRUE(res.passed);
}

// 1 K-tile, multi-block
TEST_F(Phase6cKernelTest, Tile128_256x256x32) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        256, 256, 32, tol);
    res.print("tma fp16 BM128 BN128 BK32 NS2 [256x256x32]  (multi-block, 1 K-tile)");
    EXPECT_TRUE(res.passed);
}

// 8 K-tiles — exercises steady-state 2-stage loop (fetch t+1 while computing t)
TEST_F(Phase6cKernelTest, Tile128_128x128x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        128, 128, 256, tol);
    res.print("tma fp16 BM128 BN128 BK32 NS2 [128x128x256] (8 K-tiles, steady-state)");
    EXPECT_TRUE(res.passed);
}

// 8 K-tiles, multi-block — primary NCU profiling target
TEST_F(Phase6cKernelTest, Tile128_256x256x256) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        256, 256, 256, tol);
    res.print("tma fp16 BM128 BN128 BK32 NS2 [256x256x256]");
    EXPECT_TRUE(res.passed);
}

// --- H8/H9 diagnostic tests (2026-04-26) ---
// Isolate (num_K_tiles) vs (grid_size) as independent variables for the sq2k hang.

// 1 bloc, 64 K-tiles — isolates num_tiles without grid pressure
TEST_F(Phase6cKernelTest, Diag_1bloc_64tiles) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        128, 128, 2048, tol);
    res.print("tma diag [128x128x2048]   (1 bloc, 64 K-tiles)");
    EXPECT_TRUE(res.passed);
}

// 256 blocs, 1 K-tile — isolates grid size without K-tile pressure
TEST_F(Phase6cKernelTest, Diag_256blocs_1tile) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        2048, 2048, 32, tol);
    res.print("tma diag [2048x2048x32]   (256 blocs, 1 K-tile)");
    EXPECT_TRUE(res.passed);
}

// 64 blocs, 64 K-tiles — H9 correctness regression test (was FAIL before fence fix)
TEST_F(Phase6cKernelTest, Diag_64blocs_64tiles) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        1024, 1024, 2048, tol);
    res.print("tma diag [1024x1024x2048] (64 blocs, 64 K-tiles)");
    EXPECT_TRUE(res.passed);
}

// 128 blocs, 64 K-tiles — H9 correctness regression test (was FAIL before fence fix)
TEST_F(Phase6cKernelTest, Diag_128blocs_64tiles) {
    auto res = check_fp16_kernel(
        [](GemmDescRowMajor<FP16Tag>& d) { launch_gemm_tma_mmasync<128,128,32,2,4,2>(d); },
        1024, 2048, 2048, tol);
    res.print("tma diag [1024x2048x2048] (128 blocs, 64 K-tiles)");
    EXPECT_TRUE(res.passed);
}
