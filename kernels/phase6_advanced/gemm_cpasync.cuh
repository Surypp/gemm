#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gemm/types.cuh"
#include "gemm/pipeline.cuh"
#include "gemm/error_check.cuh"
#include "ldmatrix_utils.cuh"
#include "phase5_ptx/mma_ptx_utils.cuh"

// --- Phase 6b: real cp.async + multi-stage pipelining ---
//
// Replaces the scalar stores of Phase 6a's load_tile with genuine cp.async DMA:
//   cp.async.cg.shared.global [dst], [src], 16
// 16-byte DMA bypasses the register file entirely, freeing ALU/LDST for MMA.
//
// NUM_STAGES controls pipeline depth.  Two variants are instantiated:
//   NUM_STAGES=2  37 888 B smem  → 2 blocks/SM on sm_120 (16 warps)
//   NUM_STAGES=3  56 832 B smem  → 1 block/SM on sm_120 (8 warps, via dynamic smem)
//
// Performance note (sm_120, BK=32):
//   compute window per K-tile ≈ 200–320 cycles
//   cp.async HBM latency on sm_120: UNKNOWN, likely >> 320 cycles
//   Consequence: wait<NUM_STAGES-2> often stalls immediately; overlap is partial.
//   __syncthreads() idles the compute engine even while DMA progresses.
//   3-stage will likely be SLOWER than 2-stage on sm_120 due to occupancy
//   collapse (1 vs 2 blocks/SM). Both variants are instantiated for measurement.
//
// Alignment requirement: M, N, K, lda multiples of 8 (16-byte cp.async granule
// = 8 FP16; row stride (BK+8)*2 = 80 B is divisible by 16 for BK=32).
//
// Smem layout (dynamic, NUM_STAGES stages):
//   smem_raw[0 .. NUM_STAGES*BM*(BK+8) - 1]             A stages
//   smem_raw[NUM_STAGES*BM*(BK+8) .. end]                B stages
//
// Compute core is identical to Phase 6a (ldmatrix + mma.sync).

// --- predicated cp.async (16 bytes) ---
// valid=false: fills smem with zeros; gmem[src] is not accessed.
__device__ __forceinline__
void cp_async_16b_pred(void* dst, const void* src, bool valid)
{
    unsigned d = __cvta_generic_to_shared(dst);
    int sz = valid ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(d), "l"(src), "r"(sz)
        : "memory"
    );
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
__global__ void gemm_cpasync_kernel(
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float*        __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta)
{
    static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4, "NUM_STAGES must be 2..4");

    constexpr int MMA_M  = 16, MMA_N = 16, MMA_K = 16;
    constexpr int WM     = BM / WARP_M;
    constexpr int WN     = BN / WARP_N;
    constexpr int FRAGS_M = WM / MMA_M;
    constexpr int FRAGS_N = WN / MMA_N;

    // --- dynamic smem layout ---
    extern __shared__ __half smem_raw[];

    constexpr int A_STAGE_ELEMS = BM * (BK + 8);
    constexpr int B_STAGE_ELEMS = BK * (BN + 8);

    using A2D = __half (*)[BK + 8];
    using B2D = __half (*)[BN + 8];

    A2D smem_A[NUM_STAGES];
    B2D smem_B[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; ++s) {
        smem_A[s] = reinterpret_cast<A2D>(smem_raw + s * A_STAGE_ELEMS);
        smem_B[s] = reinterpret_cast<B2D>(smem_raw + NUM_STAGES * A_STAGE_ELEMS
                                                    + s * B_STAGE_ELEMS);
    }

    // --- thread/warp indexing ---
    int warp_id  = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int lane     = threadIdx.x % 32;
    int warp_row = warp_id / WARP_N;
    int warp_col = warp_id % WARP_N;

    int c_row_off = blockIdx.y * BM + warp_row * WM;
    int c_col_off = blockIdx.x * BN + warp_col * WN;

    float d[FRAGS_M][FRAGS_N][8] = {};

    int tid           = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * blockDim.y;
    int num_tiles     = (K + BK - 1) / BK;

    // --- async tile load (16-byte cp.async per 8 FP16) ---
    // Requires M, N, K, lda multiples of 8 for full 16-byte alignment.
    // Out-of-bounds chunks are predicated to zero (sz=0).
    auto load_tile_async = [&](int stage, int t) {
        // A: BM rows × BK cols, stride (BK+8)
        for (int i = tid; i < BM * (BK / 8); i += total_threads) {
            int row  = i / (BK / 8);
            int col  = (i % (BK / 8)) * 8;
            int gr   = blockIdx.y * BM + row;
            int gc   = t * BK + col;
            bool ok  = (gr < M) && (gc < K);
            const __half* src = ok ? &A[gr * lda + gc] : &A[0];
            cp_async_16b_pred(&smem_A[stage][row][col], src, ok);
        }
        // B: BK rows × BN cols, stride (BN+8)
        for (int i = tid; i < BK * (BN / 8); i += total_threads) {
            int row  = i / (BN / 8);
            int col  = (i % (BN / 8)) * 8;
            int gr   = t * BK + row;
            int gc   = blockIdx.x * BN + col;
            bool ok  = (gr < K) && (gc < N);
            const __half* src = ok ? &B[gr * ldb + gc] : &B[0];
            cp_async_16b_pred(&smem_B[stage][row][col], src, ok);
        }
    };

    // --- prologue: prefetch min(num_tiles, NUM_STAGES-1) stages ---
    int prologue = (num_tiles < NUM_STAGES - 1) ? num_tiles : (NUM_STAGES - 1);
    for (int s = 0; s < prologue; ++s) {
        load_tile_async(s, s);
        cp_async_commit();
    }

    // --- main loop ---
    for (int t = 0; t < num_tiles; ++t) {
        int cs = t % NUM_STAGES;

        // Steady-state: wait<NUM_STAGES-2> keeps 1 group in flight (2-tile lookahead
        // for 3-stage). Epilogue (last NUM_STAGES-1 tiles): drain fully.
        //
        // When compute_window < DMA latency (likely for BK=32 on sm_120), this wait
        // stalls immediately — scoreboard stalls persist despite cp.async.
        // __syncthreads() then idles the compute engine (warps not scheduled for MMA).
        // The DMA engine continues independently, but compute units are unused.
        if (t + (NUM_STAGES - 1) < num_tiles)
            cp_async_wait<NUM_STAGES - 2>();
        else
            cp_async_wait<0>();
        __syncthreads();

        // issue next prefetch (DMA begins here, overlaps with compute below
        // only if compute_window >= DMA latency — not guaranteed for BK=32)
        int fs = t + NUM_STAGES - 1;
        if (fs < num_tiles) {
            load_tile_async(fs % NUM_STAGES, fs);
            cp_async_commit();
        }

        // --- ldmatrix + mma.sync (identical to Phase 6a) ---
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

    // --- writeback (identical to Phase 6a) ---
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
}

// --- launch wrapper ---
// cudaFuncSetAttribute raises the dynamic smem limit beyond the 48 KB static default.
// 3-stage BM128/BN128/BK32: 56 832 B (fits sm_80 ≤96 KB, sm_90a ≤232 KB, sm_120 ≤~100 KB).
// 2-stage: 37 888 B — stays below 48 KB, allows 2 blocks/SM on sm_120.
template <int BM, int BN, int BK, int WARP_M, int WARP_N, int NUM_STAGES>
void launch_gemm_cpasync(GemmDescRowMajor<FP16Tag>& desc, cudaStream_t stream = 0)
{
    constexpr size_t smem =
        (size_t)NUM_STAGES * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(__half);

    auto* fn = gemm_cpasync_kernel<BM, BN, BK, WARP_M, WARP_N, NUM_STAGES>;
    CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    dim3 block(32 * WARP_N, WARP_M);
    dim3 grid((desc.N + BN - 1) / BN, (desc.M + BM - 1) / BM);
    fn<<<grid, block, smem, stream>>>(
        desc.A, desc.lda, desc.B, desc.ldb,
        desc.C, desc.ldc,
        desc.M, desc.N, desc.K,
        static_cast<float>(desc.alpha),
        static_cast<float>(desc.beta));
    CUDA_CHECK_LAST();
}

#define DECL_CPASYNC(BM, BN, BK, WM, WN, NS) \
    extern template void launch_gemm_cpasync<BM, BN, BK, WM, WN, NS>( \
        GemmDescRowMajor<FP16Tag>&, cudaStream_t);

DECL_CPASYNC(128, 128, 32, 2, 4, 2)
DECL_CPASYNC(128, 128, 32, 2, 4, 3)
#undef DECL_CPASYNC
