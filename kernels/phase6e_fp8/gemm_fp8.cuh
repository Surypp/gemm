#pragma once

// --- Phase 6e: TMA multi-stage pipelining + mma.sync FP8 (e4m3 × e5m2 → f32) ---
//
// Extends Phase 6c (TMA + mma.sync FP16) to FP8 inputs.
// Instruction: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
// Targets: sm_90a (GH200) and sm_120 (RTX 5080).  A100 (sm_80) skipped.
//
// Key differences vs Phase 6c (FP16):
//   MMA_K = 32        (vs 16)  — K-tile doubles for FP8
//   smem type = uint8_t        — 1 byte/element vs 2 for __half
//   TMA descriptor  = UINT8    — via create_tma_descriptor_fp8()
//   A registers (4)  same count — but each uint32 holds 4 FP8 (vs 2 FP16)
//   B registers (2)  same count — but each uint32 holds 4 FP8 (vs 2 FP16)
//   A loading: ldmatrix.x4 with a_col offset × 16 (vs × 8 for FP16)
//   B loading: scalar packing from K-major smem (bank-conflict known, see below)
//   FRAGS_N = WN/8 = 4        (vs 2 with two-mma-per-frag in FP16)
//   D accumulator: float[FRAGS_M][FRAGS_N][4] (same total 64 floats per thread)
//
// B loading bank-conflict note (correctness unaffected, ~4× slower B fetch):
//   smem_B is stored [BK][BN] (K-major).  Thread t needs K-adjacent bytes from
//   the same N-column: smem_B[k0..k0+3][n].  In BN=128 row-major layout,
//   column n at rows k0..k0+3 lands in bank (k*128+n)/4 % 32 = 0 for n=0
//   regardless of k → 4-way conflict.  Fix: swap B to [BN][BK] smem and
//   transpose the TMA descriptor.  Not implemented here (Phase 6f optimization).
//
// A loading fragment layout hypothesis (verified by SASS/correctness):
//   ldmatrix.x4 with pointer = &smem_A[cs][lane%16][lane>>4 * 16]
//   → threads 0–15 load k=0..15 half, threads 16–31 load k=16..31 half.
//   Same addressing as FP16 scaled by ×2 (16 FP8 per 16 bytes vs 8 FP16).
//
// B loading fragment layout hypothesis (to be confirmed by SASS):
//   For m16n8k32 col B: thread lane holds
//     b[0] = FP8[k=(lane%4)*4 +  0..(lane%4)*4+ 3] at N-col (lane/4)
//     b[1] = FP8[k=(lane%4)*4 + 16..(lane%4)*4+19] at N-col (lane/4)
//   Covers K=32 total: 4 threads × 8-FP8 = 32 per N-column.
//
// Smem per block (BM=128, BN=128, BK=32, NS=2):
//   2 × (128×32 + 32×128) × 1 byte = 16 384 B = 16 KB  ← well within 48 KB
//
// Guard structure: kernel body #if __CUDA_ARCH__ >= 900, host wrapper #if !__CUDA_ARCH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "gemm/types.cuh"
#include "gemm/error_check.cuh"
#include "phase6c_tma/gemm_tma.cuh"         // mbar_init, mbar_wait, tma_load_2d
#include "phase6a_ldmatrix/ldmatrix_utils.cuh"  // ldmatrix_a_x4
#include "phase6e_fp8/fp8_tma_utils.cuh"    // create_tma_descriptor_fp8

// --- Descriptor for FP8 GEMM ---
// A is e4m3, B is e5m2, C/D is f32.
struct GemmDescFP8 {
    int M, N, K;
    const uint8_t* A;   // M×K, row-major, FP8 e4m3
    const uint8_t* B;   // K×N, row-major, FP8 e5m2
    float*         C;   // M×N, row-major, FP32 accumulator
    int lda, ldb, ldc;  // leading dimensions in elements
    float alpha, beta;
};

// --- mma.sync FP8 helper ---
__device__ __forceinline__
void mma_m16n8k32_fp8(const uint32_t a[4], const uint32_t b[2], float d[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
#endif
}

// --- kernel ---
template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
__global__ void gemm_fp8_kernel(
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_B,
    float*             __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4, "NUM_STAGES must be 2..4");

    constexpr int MMA_M   = 16;
    constexpr int MMA_N   = 8;
    constexpr int MMA_K   = 32;   // FP8 K-tile (vs 16 for FP16)
    constexpr int WM      = BM / WARP_M;
    constexpr int WN      = BN / WARP_N;
    constexpr int FRAGS_M = WM / MMA_M;
    constexpr int FRAGS_N = WN / MMA_N;

    extern __shared__ char _smem_raw[];
    // A: [NS][BM][BK] uint8 — FP8 e4m3
    auto* smem_A = reinterpret_cast<uint8_t (*)[BM][BK]>(_smem_raw);
    // B: [NS][BK][BN] uint8 — FP8 e5m2
    auto* smem_B = reinterpret_cast<uint8_t (*)[BK][BN]>(
        _smem_raw + NUM_STAGES * BM * BK);
    // mbarriers
    auto* mbar_A = reinterpret_cast<uint64_t*>(
        _smem_raw + NUM_STAGES * (BM * BK + BK * BN));
    auto* mbar_B = mbar_A + NUM_STAGES;

    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int lane     = threadIdx.x & 31;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_off = blockIdx.y * BM + warp_row * WM;
    int c_col_off = blockIdx.x * BN + warp_col * WN;

    float d[FRAGS_M][FRAGS_N][4] = {};

    int num_tiles = (K + BK - 1) / BK;
    int prologue  = (num_tiles < NUM_STAGES - 1) ? num_tiles : (NUM_STAGES - 1);

    // --- prologue: prefetch first (NS-1) stages ---
    for (int s = 0; s < prologue; ++s) {
        mbar_init(&mbar_A[s], 1);
        mbar_arrive_expect_tx(&mbar_A[s], BM * BK);           // 1 byte/element
        tma_load_2d(tma_A, smem_A[s], &mbar_A[s],
                    s * BK, blockIdx.y * BM);
        mbar_init(&mbar_B[s], 1);
        mbar_arrive_expect_tx(&mbar_B[s], BK * BN);
        tma_load_2d(tma_B, smem_B[s], &mbar_B[s],
                    blockIdx.x * BN, s * BK);
    }

    // --- main loop ---
    for (int t = 0; t < num_tiles; ++t) {
        int cs  = t % NUM_STAGES;
        int fs  = t + NUM_STAGES - 1;
        int fss = fs % NUM_STAGES;

        mbar_wait(&mbar_A[cs], 0);
        mbar_wait(&mbar_B[cs], 0);

        if (fs < num_tiles) {
            mbar_init(&mbar_A[fss], 1);
            mbar_arrive_expect_tx(&mbar_A[fss], BM * BK);
            tma_load_2d(tma_A, smem_A[fss], &mbar_A[fss],
                        fs * BK,         blockIdx.y * BM);
            mbar_init(&mbar_B[fss], 1);
            mbar_arrive_expect_tx(&mbar_B[fss], BK * BN);
            tma_load_2d(tma_B, smem_B[fss], &mbar_B[fss],
                        blockIdx.x * BN, fs * BK);
        }

        // --- compute: ldmatrix A + scalar-pack B + mma.sync FP8 ---
        // k iterates once (BK = MMA_K = 32).
        for (int k = 0; k < BK; k += MMA_K) {

            for (int fm = 0; fm < FRAGS_M; ++fm) {
                uint32_t a_reg[4];

                // A loading: ldmatrix.x4, adapted for FP8 (16-byte halves vs 8 for FP16).
                // lanes 0-15 → a_col = k+0,  covering FP8 k=0..15
                // lanes 16-31 → a_col = k+16, covering FP8 k=16..31
                int a_row = warp_row * WM + fm * MMA_M + (lane % 16);
                int a_col = k + (lane >> 4) * 16;
                ldmatrix_a_x4(a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                    &smem_A[cs][a_row][a_col]);

                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    uint32_t b_reg[2];

                    // B loading: scalar-pack 4 consecutive K-bytes from one N-column.
                    // Fragment layout for m16n8k32 col B (proven, see avancee_phase6e.md R2):
                    //   thread lane l:
                    //     n_col  = fn_n + (l / 4)   ← one of 8 N-columns
                    //     k_base = (l % 4) * 4       ← one of 4 K-groups (0,4,8,12)
                    //     b[0] = bytes [k_base+0..3]     at n_col  (K half 0..15)
                    //     b[1] = bytes [k_base+16..19]   at n_col  (K half 16..31)
                    // BLayoutIdentityDiagonal: A[m][k]=2^(k%8), B=I_32 → D[m][n]=2^(n%8) exact.
                    // SASS direct proof deferred to Phase 6f (ldmatrix.x2.trans for B).
                    // Bank conflict: K-stride access in K-major smem → ~4-way (known, see D4).
                    int fn_n  = warp_col * WN + fn * MMA_N;
                    int n_col = fn_n + (lane / 4);
                    int k_b   = (lane % 4) * 4;

                    auto pack4 = [&](int k0) -> uint32_t {
                        return  (uint32_t)smem_B[cs][k0    ][n_col]
                             | ((uint32_t)smem_B[cs][k0 + 1][n_col] << 8)
                             | ((uint32_t)smem_B[cs][k0 + 2][n_col] << 16)
                             | ((uint32_t)smem_B[cs][k0 + 3][n_col] << 24);
                    };
                    b_reg[0] = pack4(k_b);
                    b_reg[1] = pack4(k_b + 16);

                    mma_m16n8k32_fp8(a_reg, b_reg, d[fm][fn]);
                }
            }
        }
    }

    // --- writeback ---
    // m16n8k32 D layout (same as m16n8k16): 4 floats per thread cover a 16×8 sub-tile.
    //   d[0] → (out_row + lane>>2,     out_col + (lane&3)*2)
    //   d[1] → (out_row + lane>>2,     out_col + (lane&3)*2 + 1)
    //   d[2] → (out_row + lane>>2 + 8, out_col + (lane&3)*2)
    //   d[3] → (out_row + lane>>2 + 8, out_col + (lane&3)*2 + 1)
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            int out_row = c_row_off + fm * MMA_M;
            int out_col = c_col_off + fn * MMA_N;

            int r0 = out_row + (lane >> 2);
            int r1 = r0 + 8;
            int c0 = out_col + (lane & 3) * 2;
            int c1 = c0 + 1;

            float* acc = d[fm][fn];
            if (r0 < M && c0 < N) C[r0*ldc+c0] = alpha*acc[0] + beta*C[r0*ldc+c0];
            if (r0 < M && c1 < N) C[r0*ldc+c1] = alpha*acc[1] + beta*C[r0*ldc+c1];
            if (r1 < M && c0 < N) C[r1*ldc+c0] = alpha*acc[2] + beta*C[r1*ldc+c0];
            if (r1 < M && c1 < N) C[r1*ldc+c1] = alpha*acc[3] + beta*C[r1*ldc+c1];
        }

#endif // __CUDA_ARCH__ >= 900
}

// --- launch wrapper ---
template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
void launch_gemm_fp8(GemmDescFP8& desc, cudaStream_t stream = 0)
{
#if !defined(__CUDA_ARCH__)
    CUtensorMap h_tma_A = create_tma_descriptor_fp8(
        desc.A, desc.M, desc.K, BM, BK);
    CUtensorMap h_tma_B = create_tma_descriptor_fp8(
        desc.B, desc.K, desc.N, BK, BN);

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

    // smem = NS × (A stage + B stage) + mbar arrays
    constexpr size_t smem_bytes =
        (size_t)NUM_STAGES * (BM * BK + BK * BN)    // FP8 data: 1 byte/elem
        + 2 * NUM_STAGES * sizeof(uint64_t);         // mbar_A + mbar_B
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)gemm_fp8_kernel<BM,BN,BK,WARP_M,WARP_N,NUM_STAGES>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));

    gemm_fp8_kernel<BM,BN,BK,WARP_M,WARP_N,NUM_STAGES>
        <<<grid, block, smem_bytes, stream>>>(
            d_tma_A, d_tma_B,
            desc.C, desc.ldc,
            desc.M, desc.N, desc.K,
            desc.alpha, desc.beta);
    CUDA_CHECK_LAST();
#endif
}

// extern template declarations — suppress implicit instantiation
#define DECL_FP8(BM,BN,BK,WM,WN,NS) \
    extern template void launch_gemm_fp8<BM,BN,BK,WM,WN,NS>(GemmDescFP8&, cudaStream_t);

DECL_FP8(128, 128, 32, 2, 4, 2)
#undef DECL_FP8
