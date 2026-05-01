#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <limits>
#include "gemm/matrix.cuh"
#include "gemm/error_check.cuh"
#include "tolerance.hpp"
#include "kernels/phase6e_fp8/fp8_mma_probe.cuh"
#include "kernels/phase6e_fp8/gemm_fp8.cuh"

// --- Phase 6e: FP8 block-scaled, SM120 (RTX 5080) + SM90a (GH200) ---
//
// Oracle: CPU GEMM on FP32-decoded inputs.  No cuBLAS FP8 reference:
//   - cuBLAS FP8 on SM120 uses a different accumulation order
//   - The CPU oracle is ground truth for correctness of our mma.sync FP8 kernel
//
// Tolerance: TolerancePolicy::for_fp8() → rtol=5e-2, atol=1e-1
//   Wider than FP16 due to e4m3/e5m2 quantization errors accumulating over K.
//   NOT bit-exact.  Document in PLAN_6e.md §edge cases.
//
// Guard: sm_major < 9 → SKIP (A100 sm_80 has no FP8 mma.sync).
//   SM89 (Ada, sm_major=8, sm_minor=9) also has it but is not in our arch list;
//   the runtime check `sm_major < 9` conservatively skips it — acceptable.

// ─── CPU decoders ──────────────────────────────────────────────────────────────

// Decode FP8 e4m3 (NV format) to float.
// Encoding: 1 sign | 4 exponent | 3 mantissa, bias=7.
// Special: 0x7F / 0xFF = NaN (no infinity).  Max normal = 448.0.
static float fp8_e4m3_to_float(uint8_t bits)
{
    uint32_t sign     = (bits >> 7) & 1u;
    uint32_t exp      = (bits >> 3) & 0xFu;
    uint32_t mantissa =  bits       & 0x7u;

    if (exp == 0xFu && mantissa == 0x7u)
        return std::numeric_limits<float>::quiet_NaN();

    float result;
    if (exp == 0u) {
        // subnormal: 2^(1-bias) = 2^(-6)
        result = ldexpf((float)mantissa / 8.0f, -6);
    } else {
        result = ldexpf(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
    }
    return sign ? -result : result;
}

// Decode FP8 e5m2 (NV format) to float.
// Encoding: 1 sign | 5 exponent | 2 mantissa, bias=15.
// Infinity: exp=31, mantissa=0.  NaN: exp=31, mantissa≠0.
static float fp8_e5m2_to_float(uint8_t bits)
{
    uint32_t sign     = (bits >> 7) & 1u;
    uint32_t exp      = (bits >> 2) & 0x1Fu;
    uint32_t mantissa =  bits       & 0x3u;

    if (exp == 31u) {
        if (mantissa == 0u)
            return sign ? -std::numeric_limits<float>::infinity()
                        :  std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    float result;
    if (exp == 0u) {
        // subnormal: 2^(1-bias) = 2^(-14)
        result = ldexpf((float)mantissa / 4.0f, -14);
    } else {
        result = ldexpf(1.0f + (float)mantissa / 4.0f, (int)exp - 15);
    }
    return sign ? -result : result;
}

// ─── Oracle ────────────────────────────────────────────────────────────────────

// CPU reference GEMM: C_ref[m][n] = sum_k A_f32[m][k] * B_f32[k][n]
// A_f32 decoded from e4m3, B_f32 decoded from e5m2.
// NaN inputs treated as 0 in the product (edge-case tolerance documented above).
static gemm::HostMatrix<float> cpu_reference_fp8(
    int M, int N, int K,
    const gemm::HostMatrix<uint8_t>& hA,
    const gemm::HostMatrix<uint8_t>& hB)
{
    gemm::HostMatrix<float> ref(M, N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                float a = fp8_e4m3_to_float(hA.at(m, k));
                float b = fp8_e5m2_to_float(hB.at(k, n));
                if (!std::isnan(a) && !std::isnan(b))
                    acc += (double)a * (double)b;
            }
            ref.at(m, n) = (float)acc;
        }
    return ref;
}

// ─── Harness ───────────────────────────────────────────────────────────────────

struct FP8CorrectnessResult {
    bool   passed;
    int    violations;
    int    total_elements;
    int    first_row, first_col;
    double first_computed, first_reference;

    void print(const char* name) const {
        if (passed) {
            printf("  [PASS] %s  (%d elements, 0 violations)\n",
                   name, total_elements);
        } else {
            printf("  [FAIL] %s  (%d/%d violations)\n"
                   "         first at (%d,%d): computed=%.6f ref=%.6f diff=%.2e\n",
                   name, violations, total_elements,
                   first_row, first_col,
                   first_computed, first_reference,
                   std::abs(first_computed - first_reference));
        }
    }
};

// Generates random FP8 matrices (safe range, no NaN/Inf), runs the GPU kernel,
// compares against the CPU oracle.
template <typename KernelFn>
static FP8CorrectnessResult check_fp8_kernel(
    KernelFn   kernel_fn,
    int M, int N, int K,
    const TolerancePolicy& tol,
    uint64_t seed = 42)
{
    // Generate A as e4m3: values 0x00–0x7E (positive normal/subnormal, no NaN)
    // Generate B as e5m2: values 0x00–0x7B (positive finite, no NaN/Inf)
    gemm::HostMatrix<uint8_t> hA(M, K), hB(K, N);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> distA(0x01, 0x7E);  // e4m3: skip 0x7F=NaN
    std::uniform_int_distribution<int> distB(0x01, 0x7B);  // e5m2: skip Inf(0x7C)/NaN(0x7D-0x7F)
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            hA.at(r, c) = static_cast<uint8_t>(distA(rng));
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            hB.at(r, c) = static_cast<uint8_t>(distB(rng));

    // CPU reference
    auto hRef = cpu_reference_fp8(M, N, K, hA, hB);

    // GPU kernel
    gemm::DeviceMatrix<uint8_t> dA(M, K), dB(K, N);
    gemm::DeviceMatrix<float>   dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    GemmDescFP8 desc;
    desc.M = M; desc.N = N; desc.K = K;
    desc.A = dA.ptr; desc.B = dB.ptr; desc.C = dC.ptr;
    desc.lda = K; desc.ldb = N; desc.ldc = N;
    desc.alpha = 1.0f; desc.beta = 0.0f;

    kernel_fn(desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto hOut = gemm::HostMatrix<float>::from_device(dC);
    auto res  = gemm::HostMatrix<float>::check(hOut, hRef, tol.rtol, tol.atol);

    FP8CorrectnessResult cr;
    cr.passed          = (res.violations == 0);
    cr.violations      = res.violations;
    cr.total_elements  = M * N;
    cr.first_row       = res.first_row;
    cr.first_col       = res.first_col;
    cr.first_computed  = res.computed;
    cr.first_reference = res.reference;
    return cr;
}

// ─── Test fixture ──────────────────────────────────────────────────────────────

class Phase6eKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        int sm_major;
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0);
        // FP8 mma.sync requires SM89+ (Ada/Hopper/Blackwell).
        // sm_major check: 9 covers Hopper (sm_90a), 12 covers Blackwell (sm_120).
        // Ada (sm_89, sm_major=8) would need sm_minor check; omitted — not in arch list.
        if (sm_major < 9)
            GTEST_SKIP() << "Phase 6e requires sm_89+ (FP8 mma.sync not available on sm_80)";
    }
    TolerancePolicy tol = TolerancePolicy::for_fp8();
};

// ─── Étape 2: MRE probe ────────────────────────────────────────────────────────

// Probe: A=B=1.0, K=32 → each D accumulator must equal exactly 32.0.
// This test must pass before anything else: if it fails, the mma.sync FP8
// instruction is not working correctly at the PTX/hardware level.
TEST_F(Phase6eKernelTest, Probe_MRE_ones)
{
    bool ok = run_fp8_mma_probe();
    EXPECT_TRUE(ok) << "mma.sync FP8 MRE failed: expected all D=32.0 for A=B=1.0";
}

// ─── Étape 4: five correctness tests ──────────────────────────────────────────

// 1. Single K-tile, prologue-only path
TEST_F(Phase6eKernelTest, Tile128_16x16x32)
{
    auto res = check_fp8_kernel(
        [](GemmDescFP8& d) { launch_gemm_fp8<128,128,32,2,4,2>(d); },
        16, 16, 32, tol);
    res.print("fp8 BM128 BN128 BK32 NS2 [16x16x32]    (1 K-tile, prologue only)");
    EXPECT_TRUE(res.passed);
}

// 2. Single K-tile, single block
TEST_F(Phase6eKernelTest, Tile128_128x128x32)
{
    auto res = check_fp8_kernel(
        [](GemmDescFP8& d) { launch_gemm_fp8<128,128,32,2,4,2>(d); },
        128, 128, 32, tol);
    res.print("fp8 BM128 BN128 BK32 NS2 [128x128x32]  (1 K-tile, single block)");
    EXPECT_TRUE(res.passed);
}

// 3. Single K-tile, multi-block
TEST_F(Phase6eKernelTest, Tile128_256x256x32)
{
    auto res = check_fp8_kernel(
        [](GemmDescFP8& d) { launch_gemm_fp8<128,128,32,2,4,2>(d); },
        256, 256, 32, tol);
    res.print("fp8 BM128 BN128 BK32 NS2 [256x256x32]  (1 K-tile, multi-block)");
    EXPECT_TRUE(res.passed);
}

// 4. 8 K-tiles — steady-state 2-stage pipeline (fetch t+1 while computing t)
TEST_F(Phase6eKernelTest, Tile128_128x128x256)
{
    auto res = check_fp8_kernel(
        [](GemmDescFP8& d) { launch_gemm_fp8<128,128,32,2,4,2>(d); },
        128, 128, 256, tol);
    res.print("fp8 BM128 BN128 BK32 NS2 [128x128x256] (8 K-tiles, steady-state)");
    EXPECT_TRUE(res.passed);
}

// 5. Large — 16 K-tiles, multi-block; primary NCU profiling target
TEST_F(Phase6eKernelTest, Tile128_512x512x512)
{
    auto res = check_fp8_kernel(
        [](GemmDescFP8& d) { launch_gemm_fp8<128,128,32,2,4,2>(d); },
        512, 512, 512, tol);
    res.print("fp8 BM128 BN128 BK32 NS2 [512x512x512] (sq0.5k, perf check)");
    EXPECT_TRUE(res.passed);
}

// ─── Decoder unit tests (host-only, no GPU required) ──────────────────────────

// Verifies fp8_e4m3_to_float for zero, subnormals, 1.0, max-normal, and NaN.
// Subnormal formula: (-1)^s * 2^(1-7) * mantissa/8 = mantissa * 2^(-9).
TEST(Phase6eDecoder, SubnormalE4M3)
{
    EXPECT_EQ(fp8_e4m3_to_float(0x00),  0.0f);
    EXPECT_EQ(fp8_e4m3_to_float(0x01),  std::ldexp(1.0f / 8.0f, -6));  // 2^(-9)
    EXPECT_EQ(fp8_e4m3_to_float(0x07),  std::ldexp(7.0f / 8.0f, -6));  // 7*2^(-9)
    EXPECT_EQ(fp8_e4m3_to_float(0x81), -std::ldexp(1.0f / 8.0f, -6));  // negative subnormal
    EXPECT_EQ(fp8_e4m3_to_float(0x38),  1.0f);
    EXPECT_EQ(fp8_e4m3_to_float(0x7E),  448.0f);   // max normal: 2^8 * (1+6/8)
    EXPECT_TRUE(std::isnan(fp8_e4m3_to_float(0x7F)));  // NaN sentinel
    EXPECT_TRUE(std::isnan(fp8_e4m3_to_float(0xFF)));  // NaN sentinel (negative)
}

// Verifies fp8_e5m2_to_float for zero, subnormals, 1.0, inf, and NaN.
// Subnormal formula: 2^(1-15) * mantissa/4 = mantissa * 2^(-16).
TEST(Phase6eDecoder, SubnormalE5M2)
{
    EXPECT_EQ(fp8_e5m2_to_float(0x00),  0.0f);
    EXPECT_EQ(fp8_e5m2_to_float(0x01),  std::ldexp(1.0f / 4.0f, -14));  // 2^(-16)
    EXPECT_EQ(fp8_e5m2_to_float(0x03),  std::ldexp(3.0f / 4.0f, -14));  // 3*2^(-16)
    EXPECT_EQ(fp8_e5m2_to_float(0x81), -std::ldexp(1.0f / 4.0f, -14));  // negative subnormal
    EXPECT_EQ(fp8_e5m2_to_float(0x3C),  1.0f);
    EXPECT_TRUE(std::isinf(fp8_e5m2_to_float(0x7C)));   // +inf
    EXPECT_TRUE(std::isinf(fp8_e5m2_to_float(0xFC)));   // -inf
    EXPECT_TRUE(std::isnan(fp8_e5m2_to_float(0x7D)));   // NaN
}

// ─── B fragment layout proof ───────────────────────────────────────────────────

// Proves the B operand layout for mma.sync.m16n8k32.row.col.f32.e4m3.e5m2.f32.
//
// Construction:
//   A[m][k] = 2^(k%8) in e4m3  (distinct weight per k-group-of-8)
//   B[k][n] = delta(k,n) * 1.0_e5m2  (identity matrix, K=N=32)
//
// Expected:
//   D[m][n] = sum_k A[m][k] * B[k][n] = A[m][n] = 2^(n%8)  (exact, single non-zero term)
//
// If any B register maps to the wrong (k,n) position, D[m][n] gets the wrong
// power-of-2 weight — detectable by exact comparison.
TEST_F(Phase6eKernelTest, BLayoutIdentityDiagonal)
{
    constexpr int M = 32, N = 32, K = 32;

    // e4m3 encoding of powers of 2: 1,2,4,8,16,32,64,128
    static const uint8_t e4m3_pow2[8] = {
        0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70
    };
    static const float pow2f[8] = {1.f, 2.f, 4.f, 8.f, 16.f, 32.f, 64.f, 128.f};

    gemm::HostMatrix<uint8_t> hA(M, K), hB(K, N);  // hB zero-initialized by ctor

    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            hA.at(m, k) = e4m3_pow2[k % 8];

    for (int i = 0; i < 32; ++i)
        hB.at(i, i) = 0x3C;  // 1.0 in e5m2

    gemm::DeviceMatrix<uint8_t> dA(M, K), dB(K, N);
    gemm::DeviceMatrix<float>   dC(M, N);
    dA.copy_from(hA);
    dB.copy_from(hB);

    GemmDescFP8 desc;
    desc.M = M; desc.N = N; desc.K = K;
    desc.A = dA.ptr; desc.B = dB.ptr; desc.C = dC.ptr;
    desc.lda = K; desc.ldb = N; desc.ldc = N;
    desc.alpha = 1.0f; desc.beta = 0.0f;

    launch_gemm_fp8<128, 128, 32, 2, 4, 2>(desc);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto hOut = gemm::HostMatrix<float>::from_device(dC);

    int violations = 0;
    int first_m = -1, first_n = -1;
    float first_got = 0.f, first_exp = 0.f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float expected = pow2f[n % 8];
            float got      = hOut.at(m, n);
            if (got != expected) {
                if (violations == 0) {
                    first_m = m; first_n = n;
                    first_got = got; first_exp = expected;
                }
                ++violations;
            }
        }

    if (violations > 0)
        printf("  BLayoutIdentityDiagonal: FAIL — first at D[%d][%d]: got=%.1f expected=%.1f\n",
               first_m, first_n, first_got, first_exp);
    else
        printf("  BLayoutIdentityDiagonal: PASS — D[m][n] == 2^(n%%8) for all 32×32\n");

    EXPECT_EQ(violations, 0);
}
