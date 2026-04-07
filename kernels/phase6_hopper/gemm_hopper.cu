// --- Phase 6 skeleton ---
// Compiled only when GEMM_ENABLE_HOPPER=ON (sm_90a).
// Entry point for the wgmma + TMA kernel.
// Full writeback and tile sizing are Phase 6 deliverables.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#include "gemm_wgmma.cuh"
#include "gemm_tma.cuh"
#include "gemm/types.cuh"
#include "gemm/error_check.cuh"
#include <cuda_fp16.h>

// --- kernel ---
// wgmma operates on warp groups (4 warps = 128 threads); block size must be
// a multiple of 128.

template <int BM, int BN, int BK>
__global__
__launch_bounds__(256)  // 2 warp groups per block
void gemm_hopper_kernel(
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_B,
    float*             __restrict__ C,
    int ldc, int M, int N, int K,
    float alpha, float beta)
{
    // Double-buffered A and B tiles + per-stage mbarriers
    __shared__ __half smem_A[2][BM][BK];
    __shared__ __half smem_B[2][BK][BN];
    __shared__ uint64_t mbar_A[2], mbar_B[2];

    int warp_group_id = (threadIdx.x + threadIdx.y * blockDim.x) / 128;
    int wg_lane       = (threadIdx.x + threadIdx.y * blockDim.x) % 128;

    int num_wg         = blockDim.x * blockDim.y / 128;
    int wg_n_tile_size = BN / num_wg;
    int wg_col_off     = blockIdx.x * BN + warp_group_id * wg_n_tile_size;
    int block_row_off  = blockIdx.y * BM;

    int num_tiles = (K + BK - 1) / BK;

    constexpr int ACC_SIZE = 32;  // placeholder; adjust to BN/num_wg
    float d[ACC_SIZE] = {};

    // --- prologue ---
    mbar_init(&mbar_A[0], BM * BK * sizeof(__half));
    mbar_init(&mbar_B[0], BK * BN * sizeof(__half));

    tma_load_2d(tma_A, smem_A[0], &mbar_A[0],
                0, block_row_off);
    tma_load_2d(tma_B, smem_B[0], &mbar_B[0],
                blockIdx.x * BN, 0);

    uint64_t phase = 0;
    mbar_wait(&mbar_A[0], phase);
    mbar_wait(&mbar_B[0], phase);

    // --- main loop (skeleton) ---
    for (int t = 0; t < num_tiles; ++t) {
        int cs = t % 2;

        wgmma_fence();

        uint64_t desc_a = make_smem_desc(&smem_A[cs][0][0], BK * sizeof(__half));
        uint64_t desc_b = make_smem_desc(&smem_B[cs][0][0], BN * sizeof(__half));

        for (int k = 0; k < BK; k += 16) {
            wgmma_m64n8k16_fp16(desc_a, desc_b, d);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        // Prefetch next tile
        if (t + 1 < num_tiles) {
            int ns = (t + 1) % 2;
            mbar_init(&mbar_A[ns], BM * BK * sizeof(__half));
            mbar_init(&mbar_B[ns], BK * BN * sizeof(__half));
            tma_load_2d(tma_A, smem_A[ns], &mbar_A[ns],
                        (t + 1) * BK, block_row_off);
            tma_load_2d(tma_B, smem_B[ns], &mbar_B[ns],
                        blockIdx.x * BN, (t + 1) * BK);
            phase ^= 1;
            mbar_wait(&mbar_A[ns], phase);
            mbar_wait(&mbar_B[ns], phase);
        }
    }

    // --- writeback (TODO: full wgmma register→global thread mapping) ---
    (void)d; (void)alpha; (void)beta;
}

// --- launch ---
template <int BM, int BN, int BK>
void launch_gemm_hopper(
    const CUtensorMap* tma_A,
    const CUtensorMap* tma_B,
    float* C, int ldc,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream = 0)
{
    dim3 block(128, 2);  // 2 warp groups per block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_hopper_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
        tma_A, tma_B, C, ldc, M, N, K, alpha, beta);
    CUDA_CHECK_LAST();
}

template void launch_gemm_hopper<128, 128, 64>(
    const CUtensorMap*, const CUtensorMap*,
    float*, int, int, int, int, float, float, cudaStream_t);

#endif // __CUDA_ARCH__ >= 900
