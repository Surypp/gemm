#pragma once

// --- Phase 6c: TMA multi-stage pipelining + mma.sync ---
//
// Replaces cp.async (Phase 6b) with TMA (Tensor Memory Accelerator).
// TMA is a dedicated hardware unit on sm_90a+ that DMAs tiles from HBM to
// smem without thread involvement; the transfer is tracked by a mbarrier.
//
// Targets: sm_90a (GH200) and sm_120 (RTX 5080).  A100 (sm_80) has no TMA.
//
// Smem layout — SWIZZLE_NONE (production choice after perf analysis):
//   Both A and B use CU_TENSOR_MAP_SWIZZLE_NONE.
//
//   Investigation summary (see Bugs_Detailed.md §Fix B.2):
//   - TMA SWIZZLE_64B hardware pattern on SM120 was measured by tools/tma_swizzle_probe:
//       physical_col = logical_col XOR (((row / 2) & 3) * 8)
//     Period-8 paired rows; corresponds to cute::Swizzle<2,4,3>.
//     NOT (row & 3) * 8 (period-4 formula fails on SM120 — different from SM90 expectation).
//   - With correct de-swizzle formula: 5/5 correctness PASS.
//   - Performance: SWIZZLE_64B = 69.6 TFLOPS vs SWIZZLE_NONE = 84.9 TFLOPS (sq4k).
//     Root cause: ldmatrix.x4 bank conflicts are inescapable for BK=32 (8 rows/bank
//     cycle forces 16 conflicts per call regardless of XOR formula); TMA SWIZZLE_64B
//     write overhead adds ~15 TFLOPS regression with no offsetting gain.
//     Conclusion: mio_throttle at 15% is pure volume, not bank-conflict addressable.
//   - Constraint: B permanently SWIZZLE_NONE (BN*2=256 bytes > 128-byte TMA limit).
//
// Smem per block (BM=128, BN=128, BK=32) — dynamic smem, includes mbar arrays:
//   NS=2 → 32 800 B → 3 blocks/SM on sm_120
//   NS=3 → 49 200 B → 2 blocks/SM on sm_120 (exceeds 48 KB static limit by 48 B;
//           dynamic smem + cudaFuncSetAttribute required)
//
// Guard structure:
//   kernel body        : #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
//   launch wrapper     : #if !defined(__CUDA_ARCH__)  (host only)
//   extern template    : #if !defined(__CUDA_ARCH__)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gemm/types.cuh"
#include "gemm/error_check.cuh"
#include "gemm_tma.cuh"
#include "phase6a_ldmatrix/ldmatrix_utils.cuh"
#include "phase5_ptx_mma/mma_ptx_utils.cuh"

// --- kernel ---
// __global__ template must be declared in every compilation pass so nvcc can
// generate the host-side launch stub.  The body is guarded so device-only
// intrinsics (tma_load_2d, mbar_*) are only compiled when __CUDA_ARCH__ >= 900.

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
__global__ void gemm_tma_mmasync_kernel(
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_B,
    float*             __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4, "NUM_STAGES must be 2..4");

    constexpr int MMA_M   = 16, MMA_N = 16, MMA_K = 16;
    constexpr int WM      = BM / WARP_M;
    constexpr int WN      = BN / WARP_N;
    constexpr int FRAGS_M = WM / MMA_M;
    constexpr int FRAGS_N = WN / MMA_N;

    extern __shared__ char _smem_raw[];
    auto* smem_A = reinterpret_cast<__half (*)[BM][BK]>(_smem_raw);
    auto* smem_B = reinterpret_cast<__half (*)[BK][BN]>(
        _smem_raw + NUM_STAGES * BM * BK * sizeof(__half));
    auto* mbar_A = reinterpret_cast<uint64_t*>(
        _smem_raw + NUM_STAGES * (BM * BK + BK * BN) * sizeof(__half));
    auto* mbar_B = mbar_A + NUM_STAGES;

    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int lane     = threadIdx.x % 32;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_off = blockIdx.y * BM + warp_row * WM;
    int c_col_off = blockIdx.x * BN + warp_col * WN;

    float d[FRAGS_M][FRAGS_N][8] = {};

    int num_tiles = (K + BK - 1) / BK;
    int prologue  = (num_tiles < NUM_STAGES - 1) ? num_tiles : (NUM_STAGES - 1);

    // --- prologue: prefetch first (NS-1) stages ---
    for (int s = 0; s < prologue; ++s) {
        mbar_init(&mbar_A[s], 1);
        mbar_arrive_expect_tx(&mbar_A[s], BM * BK * sizeof(__half));
        tma_load_2d(tma_A, smem_A[s], &mbar_A[s],
                    s * BK,          blockIdx.y * BM);
        mbar_init(&mbar_B[s], 1);
        mbar_arrive_expect_tx(&mbar_B[s], BK * BN * sizeof(__half));
        tma_load_2d(tma_B, smem_B[s], &mbar_B[s],
                    blockIdx.x * BN, s * BK);
    }

    // --- main loop ---
    for (int t = 0; t < num_tiles; ++t) {
        int cs  = t % NUM_STAGES;
        int fs  = t + NUM_STAGES - 1;
        int fss = fs % NUM_STAGES;   // fss ≠ cs for NS≥2

        mbar_wait(&mbar_A[cs], 0);
        mbar_wait(&mbar_B[cs], 0);

        // fs and num_tiles are uniform across the block, so all threads enter
        // this branch or none — __syncthreads() inside mbar_init is safe.
        if (fs < num_tiles) {
            mbar_init(&mbar_A[fss], 1);
            mbar_arrive_expect_tx(&mbar_A[fss], BM * BK * sizeof(__half));
            tma_load_2d(tma_A, smem_A[fss], &mbar_A[fss],
                        fs * BK,         blockIdx.y * BM);
            mbar_init(&mbar_B[fss], 1);
            mbar_arrive_expect_tx(&mbar_B[fss], BK * BN * sizeof(__half));
            tma_load_2d(tma_B, smem_B[fss], &mbar_B[fss],
                        blockIdx.x * BN, fs * BK);
        }

        // --- compute: ldmatrix + mma.sync ---
        // SWIZZLE_NONE: direct logical column, no XOR needed.
        for (int k = 0; k < BK; k += MMA_K) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                uint32_t a_reg[4];
                int a_row     = warp_row * WM + fm * MMA_M;
                int a_row_idx = a_row + lane % 16;
                int a_col     = k + (lane >> 4) * 8;

                ldmatrix_a_x4(
                    a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                    &smem_A[cs][a_row_idx][a_col]);

                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    uint32_t b_reg[2], b_reg2[2];
                    int b_col     = warp_col * WN + fn * MMA_N;
                    int b_row_idx = k + lane % 16;

                    ldmatrix_b_x2_trans(b_reg[0],  b_reg[1],
                                        &smem_B[cs][b_row_idx][b_col]);
                    ldmatrix_b_x2_trans(b_reg2[0], b_reg2[1],
                                        &smem_B[cs][b_row_idx][b_col + 8]);

                    float* acc = d[fm][fn];
                    mma_m16n8k16(a_reg, b_reg,  acc);
                    mma_m16n8k16(a_reg, b_reg2, acc + 4);
                }
            }
        }
    }

    // --- writeback ---
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            int out_row = c_row_off + fm * MMA_M;
            int out_col = c_col_off + fn * MMA_N;

            int r0 = out_row + (lane >> 2);
            int r1 = out_row + (lane >> 2) + 8;
            int c0 = out_col + (lane & 3) * 2;
            int c1 = out_col + (lane & 3) * 2 + 1;
            int c2 = out_col + 8 + (lane & 3) * 2;
            int c3 = out_col + 8 + (lane & 3) * 2 + 1;

            if (r0 < M && c0 < N) C[r0*ldc+c0] = alpha*d[fm][fn][0] + beta*C[r0*ldc+c0];
            if (r0 < M && c1 < N) C[r0*ldc+c1] = alpha*d[fm][fn][1] + beta*C[r0*ldc+c1];
            if (r1 < M && c0 < N) C[r1*ldc+c0] = alpha*d[fm][fn][2] + beta*C[r1*ldc+c0];
            if (r1 < M && c1 < N) C[r1*ldc+c1] = alpha*d[fm][fn][3] + beta*C[r1*ldc+c1];

            if (r0 < M && c2 < N) C[r0*ldc+c2] = alpha*d[fm][fn][4] + beta*C[r0*ldc+c2];
            if (r0 < M && c3 < N) C[r0*ldc+c3] = alpha*d[fm][fn][5] + beta*C[r0*ldc+c3];
            if (r1 < M && c2 < N) C[r1*ldc+c2] = alpha*d[fm][fn][6] + beta*C[r1*ldc+c2];
            if (r1 < M && c3 < N) C[r1*ldc+c3] = alpha*d[fm][fn][7] + beta*C[r1*ldc+c3];
        }

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// --- launch wrapper ---
// The template must be visible in ALL compilation passes (host and device) so
// that nvcc can resolve call sites in both.  The driver API body is only valid
// on the host, so it is guarded inside.  The device pass sees an empty body,
// generates no device code for it, and does not error out.
//
// cudaFree does not wait for in-flight kernels; cudaStreamSynchronize is required.
// lda=K, ldb=N (contiguous row-major) — matches the test harness.

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
void launch_gemm_tma_mmasync(GemmDescRowMajor<FP16Tag>& desc, cudaStream_t stream = 0)
{
#if !defined(__CUDA_ARCH__)
    // SWIZZLE_NONE for A and B.  See header comment for full investigation.
    // A: SWIZZLE_64B formula verified correct but causes +15 TFLOPS regression on SM120.
    // B: SWIZZLE_NONE permanent (BN*2=256 bytes > 128-byte TMA swizzle limit).
    CUtensorMap h_tma_A = create_tma_descriptor_2d(
        desc.A, desc.M, desc.K, BM, BK, sizeof(__half),
        CU_TENSOR_MAP_SWIZZLE_NONE);
    CUtensorMap h_tma_B = create_tma_descriptor_2d(
        desc.B, desc.K, desc.N, BK, BN, sizeof(__half),
        CU_TENSOR_MAP_SWIZZLE_NONE);

    // Pre-allocate device descriptors once per template instantiation.
    // Avoids stream-ordered malloc/free overhead inside the GPU-timed hot path.
    // Re-upload only when descriptor content changes (matrix ptr or dims changed).
    static CUtensorMap *d_tma_A = nullptr, *d_tma_B = nullptr;
    static CUtensorMap cached_A{}, cached_B{};
    if (!d_tma_A) {
        CUDA_CHECK(cudaMalloc(&d_tma_A, sizeof(CUtensorMap)));
        CUDA_CHECK(cudaMalloc(&d_tma_B, sizeof(CUtensorMap)));
    }
    if (memcmp(&h_tma_A, &cached_A, sizeof(CUtensorMap)) != 0) {
        CUDA_CHECK(cudaMemcpy(d_tma_A, &h_tma_A, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        cached_A = h_tma_A;
    }
    if (memcmp(&h_tma_B, &cached_B, sizeof(CUtensorMap)) != 0) {
        CUDA_CHECK(cudaMemcpy(d_tma_B, &h_tma_B, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        cached_B = h_tma_B;
    }

    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid((desc.N + BN - 1) / BN, (desc.M + BM - 1) / BM);

    constexpr size_t smem_bytes =
        (size_t)NUM_STAGES * (BM * BK + BK * BN) * sizeof(__half)
        + 2 * NUM_STAGES * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)gemm_tma_mmasync_kernel<BM, BN, BK, WARP_M, WARP_N, NUM_STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    gemm_tma_mmasync_kernel<BM, BN, BK, WARP_M, WARP_N, NUM_STAGES>
        <<<grid, block, smem_bytes, stream>>>(
            d_tma_A, d_tma_B,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            static_cast<float>(desc.alpha),
            static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();
#endif // !defined(__CUDA_ARCH__)
}

// extern template declarations — visible in all passes; suppress implicit instantiation
// wherever this header is included.  Explicit instantiation lives in
// gemm_tma_mmasync_fp16.cu which is compiled for sm_90a and sm_120 only.

#define DECL_TMA(BM, BN, BK, WM, WN, NS) \
    extern template void launch_gemm_tma_mmasync<BM, BN, BK, WM, WN, NS>( \
        GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_TMA(128, 128, 32, 2, 4, 2)
DECL_TMA(128, 128, 32, 2, 4, 3)
#undef DECL_TMA
