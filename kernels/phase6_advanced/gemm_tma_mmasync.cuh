#pragma once

// --- Phase 6c: TMA multi-stage pipelining + mma.sync ---
//
// Replaces cp.async (Phase 6b) with TMA (Tensor Memory Accelerator).
// TMA is a dedicated hardware unit on sm_90a+ that DMAs tiles from HBM to
// smem without thread involvement; the transfer is tracked by a mbarrier.
//
// Targets: sm_90a (GH200) and sm_120 (RTX 5080).  A100 (sm_80) has no TMA.
//
// Smem layout — no padding:
//   TMA writes box_dim = {BK, BM} elements with stride BK*sizeof(__half)/row.
//   Adding +8 columns (as in Phase 6b) would misalign TMA writes: the
//   descriptor encodes BK columns but smem has BK+8.  No workaround exists
//   in the current TMA API (no smem-stride parameter).
//   Bank conflicts on ldmatrix are expected and intentional — Phase 6c isolates
//   the TMA effect.  NCU sm__pipe_smem_cycles_active will show the pressure.
//
// Smem per block (BM=128, BN=128, BK=32):
//   NS=2 → 32 768 B → 3 blocks/SM on sm_120
//   NS=3 → 49 152 B → 2 blocks/SM on sm_120
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
#include "phase6_hopper/gemm_tma.cuh"
#include "phase6_advanced/ldmatrix_utils.cuh"
#include "phase5_ptx/mma_ptx_utils.cuh"

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

    __shared__ __half   smem_A[NUM_STAGES][BM][BK];
    __shared__ __half   smem_B[NUM_STAGES][BK][BN];
    __shared__ uint64_t mbar_A[NUM_STAGES];
    __shared__ uint64_t mbar_B[NUM_STAGES];

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

        // --- compute: ldmatrix + mma.sync (identical to Phase 6a/6b) ---
        for (int k = 0; k < BK; k += MMA_K) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                uint32_t a_reg[4];
                int a_row = warp_row * WM + fm * MMA_M;

                ldmatrix_a_x4(
                    a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                    &smem_A[cs][a_row + lane % 16][k + (lane >> 4) * 8]);

                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    uint32_t b_reg[2], b_reg2[2];
                    int b_col = warp_col * WN + fn * MMA_N;

                    ldmatrix_b_x2_trans(b_reg[0],  b_reg[1],
                                        &smem_B[cs][k + lane % 16][b_col]);
                    ldmatrix_b_x2_trans(b_reg2[0], b_reg2[1],
                                        &smem_B[cs][k + lane % 16][b_col + 8]);

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
    CUtensorMap h_tma_A = create_tma_descriptor_2d(
        desc.A, desc.M, desc.K, BM, BK, sizeof(__half));
    CUtensorMap h_tma_B = create_tma_descriptor_2d(
        desc.B, desc.K, desc.N, BK, BN, sizeof(__half));

    CUtensorMap *d_tma_A, *d_tma_B;
    CUDA_CHECK(cudaMalloc(&d_tma_A, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_tma_B, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyAsync(d_tma_A, &h_tma_A,
                               sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tma_B, &h_tma_B,
                               sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream));

    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid((desc.N + BN - 1) / BN, (desc.M + BM - 1) / BM);

    gemm_tma_mmasync_kernel<BM, BN, BK, WARP_M, WARP_N, NUM_STAGES>
        <<<grid, block, 0, stream>>>(
            d_tma_A, d_tma_B,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            static_cast<float>(desc.alpha),
            static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_tma_A));
    CUDA_CHECK(cudaFree(d_tma_B));
#endif // !defined(__CUDA_ARCH__)
}

// extern template declarations — visible in all passes; suppress implicit instantiation
// wherever this header is included.  Explicit instantiation lives in
// gemm_tma_mmasync_fp16.cu which is compiled for sm_90a and sm_120 only.

#define DECL_TMA(BM, BN, BK, WM, WN, NS) \
    extern template void launch_gemm_tma_mmasync<BM, BN, BK, WM, WN, NS>( \
        GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_TMA(128, 128, 32, 2, 4, 2)
// NS=3: smem = 49200 B on BM128/BN128/BK32, exceeds sm_90a static limit (49152 B)
// by 48 B (two mbar arrays × 3 stages × 8 B each).  Omitted until BK is reduced
// or dynamic smem is used.  sm_120 would fit but no test exercises this variant.
#undef DECL_TMA
