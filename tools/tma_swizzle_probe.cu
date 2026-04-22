// tma_swizzle_probe.cu — Phase 6c diagnostic tool
//
// Empirically determines the TMA SWIZZLE_64B smem layout on the current GPU.
//
// Strategy
// --------
//   1. Fill host matrix A[r][c] = c  (raw uint16 bit pattern = column index).
//      TMA copies bytes verbatim, so reading smem[r][p] back gives the logical
//      column that was stored at physical position p.
//   2. Load the tile with TMA SWIZZLE_64B into smem.
//   3. Thread 0 reads the full smem content and writes to a device output array.
//   4. Host reconstructs the XOR table: xor_val[r][p] = p ^ smem[r][p].
//      For a row-only XOR formula the value is constant across all chunks in row r.
//
// Formulas tested
// ---------------
//   (row & 3) * 8      — period-4, XOR values {0,8,16,24} per row group
//   ((row/2) & 3) * 8  — period-8 (pair-rows), derived from cute::Swizzle<2,4,3>
//   (row & 7) * 4      — period-8 (individual rows), cute::Swizzle<3,3,3> @ byte level
//
// Constraints
// -----------
//   box_dim[0] * elem_size ≤ 128 bytes for SWIZZLE ≠ NONE.
//   PROBE_K * 2 = 32 * 2 = 64 bytes ≤ 128 ✓.
//
// Build (Windows RTX 5080 sm_120)
// --------------------------------
//   .\build_gemm.bat   (or cmake --build build --target tma_swizzle_probe)
//   .\build\tools\tma_swizzle_probe.exe
//
// Targets: sm_90a (GH200), sm_120 (RTX 5080).  A100 (sm_80) has no TMA.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── error helpers ─────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s\n", f, l, cudaGetErrorString(e));
        exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// ── probe parameters ──────────────────────────────────────────────────────────

static constexpr int PROBE_M = 32;   // rows — covers ≥ 2 full XOR period cycles
static constexpr int PROBE_K = 32;   // cols — matches production BK; 32*2=64 B ≤ 128 B

// ── TMA descriptor helper ──────────────────────────────────────────────────────
// Not __device__; the device compilation pass parses but does not emit code for it.
// cuTensorMapEncodeTiled is a host driver API call — only valid at runtime on the host.

static CUtensorMap make_tma_swizzle64(
    const void* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap map{};
    uint64_t global_dim[2]    = { (uint64_t)cols, (uint64_t)rows };
    uint64_t global_stride[1] = { (uint64_t)(cols * sizeof(uint16_t)) };
    uint32_t box_dim[2]       = { (uint32_t)box_cols, (uint32_t)box_rows };
    uint32_t elem_stride[2]   = { 1, 1 };

    CUresult res = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        const_cast<void*>(ptr),
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(res, &s);
        fprintf(stderr, "cuTensorMapEncodeTiled(SWIZZLE_64B) failed: %s\n",
                s ? s : "?");
        exit(1);
    }
    return map;
}

// ── probe kernel ──────────────────────────────────────────────────────────────
// All device-specific code gated on __CUDA_ARCH__ >= 900 (TMA requires sm_90a+).
// Host compilation pass sees an empty stub so the <<<>>> launch syntax is valid.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__global__ void tma_swizzle_probe_kernel(
    const CUtensorMap* __restrict__ tma_A,
    uint16_t*          __restrict__ output,
    uint32_t                        tile_bytes)
{
    // SWIZZLE_64B requires the smem TMA destination to be 64-byte aligned.
    // No static __shared__ variables before this; dynamic smem starts at the
    // compiler-guaranteed large alignment boundary (≥ 128 bytes on all CUDA targets).
    // mbar lives at the END of the dynamic buffer to keep tile_data at offset 0.
    extern __shared__ char _smem_raw[];
    auto* smem = reinterpret_cast<uint16_t*>(_smem_raw);
    // mbar placed at byte offset = tile_bytes (always a multiple of 8 for uint16 tiles)
    auto* mbar = reinterpret_cast<uint64_t*>(_smem_raw + tile_bytes);

    if (threadIdx.x == 0) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
        uint32_t smem_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

        // mbarrier: init with arrival count 1
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n"
                     :: "r"(mbar_addr));

        // arrive + expect_tx (combined workaround for ptxas 13.2 ICE)
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(mbar_addr), "r"(tile_bytes) : "memory");

        // fire TMA: load full tile at global coords (col=0, row=0)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {0, 0}], [%2];\n"
            :: "r"(smem_ptr), "l"(tma_A), "r"(mbar_addr) : "memory");
    }
    __syncthreads();

    // all threads wait for TMA completion (parity 0 → 1)
    {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
        asm volatile(
            "{\n"
            ".reg .pred done;\n"
            "PROBE_WAIT:\n"
            "mbarrier.try_wait.parity.shared.b64 done, [%0], 0;\n"
            "@!done bra PROBE_WAIT;\n"
            "}\n"
            :: "r"(mbar_addr));
    }
    __syncthreads();

    // copy smem → global output (coalesced, all threads participate)
    int total = PROBE_M * PROBE_K;
    for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x)
        output[i] = smem[i];
}

#else // host-pass stub

__global__ void tma_swizzle_probe_kernel(
    const CUtensorMap*, uint16_t*, uint32_t) {}

#endif

// ── host analysis ─────────────────────────────────────────────────────────────

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Fill host matrix: A[r][c] = c (column index in raw uint16)
    std::vector<uint16_t> h_A(PROBE_M * PROBE_K);
    for (int r = 0; r < PROBE_M; ++r)
        for (int c = 0; c < PROBE_K; ++c)
            h_A[r * PROBE_K + c] = (uint16_t)c;

    // Device allocations
    uint16_t *d_A = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,   PROBE_M * PROBE_K * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out, PROBE_M * PROBE_K * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(),
                          PROBE_M * PROBE_K * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    // TMA descriptor
    CUtensorMap h_tma = make_tma_swizzle64(d_A, PROBE_M, PROBE_K, PROBE_M, PROBE_K);
    CUtensorMap* d_tma = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tma, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tma, &h_tma, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // tile_bytes = data only; total dynamic smem adds the mbarrier at the end.
    uint32_t tile_bytes  = PROBE_M * PROBE_K * sizeof(uint16_t);
    uint32_t smem_bytes  = tile_bytes + sizeof(uint64_t);  // tile + mbar
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)tma_swizzle_probe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes));

    tma_swizzle_probe_kernel<<<1, 128, smem_bytes>>>(d_tma, d_out, tile_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    std::vector<uint16_t> h_out(PROBE_M * PROBE_K);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          PROBE_M * PROBE_K * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_out); cudaFree(d_tma);

    // ── Full mapping table ────────────────────────────────────────────────────
    // For each (row, physical_col): show logical_col stored there by TMA.
    // XOR = physical_col ^ logical_col tells us the swizzle pattern.
    printf("TMA SWIZZLE_64B:  smem[row][physical_col] = logical_col\n");
    printf("(chunk stride = 8 FP16 = 16 bytes; XOR = physical ^ logical)\n\n");

    // header
    printf("%4s", "row");
    for (int c = 0; c < PROBE_K; c += 8)
        printf("  [p%2d] log xor", c);
    printf("\n");

    for (int r = 0; r < PROBE_M; ++r) {
        printf("%4d", r);
        for (int c = 0; c < PROBE_K; c += 8) {
            int logical = (int)h_out[r * PROBE_K + c];
            int xor_val = c ^ logical;
            printf("   p%2d->%-2d  x%2d", c, logical, xor_val);
        }
        printf("\n");
    }

    // ── XOR-per-row summary ───────────────────────────────────────────────────
    printf("\nXOR at chunk-0 (physical col 0) per row:\n");
    for (int r = 0; r < PROBE_M; ++r) {
        int xor_val = (int)h_out[r * PROBE_K];  // physical col 0, logical = h_out[r*K]
        printf("  row %2d: xor=%2d", r, xor_val);
        if ((r + 1) % 8 == 0) printf("\n");
    }
    printf("\n");

    // ── Formula matching ─────────────────────────────────────────────────────
    // Test several candidate formulas for physical_col = logical_col ^ formula(row, col).
    struct Candidate {
        const char* name;
        int (*fn)(int row, int phys_col);
    };
    Candidate candidates[] = {
        { "(row & 3) * 8",       [](int r, int) -> int { return (r & 3) * 8; } },
        { "((row/2) & 3) * 8",   [](int r, int) -> int { return ((r / 2) & 3) * 8; } },
        { "(row & 7) * 4",       [](int r, int) -> int { return (r & 7) * 4; } },
        { "((row/4) & 3) * 8",   [](int r, int) -> int { return ((r / 4) & 3) * 8; } },
        { "((row>>1) & 3) * 8",  [](int r, int) -> int { return ((r >> 1) & 3) * 8; } },
    };

    printf("Formula matching (against all %d rows × %d/8 chunks):\n",
           PROBE_M, PROBE_K);
    for (auto& cand : candidates) {
        bool match = true;
        for (int r = 0; r < PROBE_M && match; ++r) {
            for (int c = 0; c < PROBE_K; c += 8) {
                int logical  = (int)h_out[r * PROBE_K + c];
                int actual   = c ^ logical;          // XOR at this physical col
                int expected = cand.fn(r, c);
                if (actual != expected) { match = false; break; }
            }
        }
        printf("  %-22s : %s\n", cand.name, match ? "MATCH ✓" : "no");
    }

    // ── Cross-chunk consistency ───────────────────────────────────────────────
    // Check if XOR is truly column-independent within each row.
    printf("\nXOR column-consistency (should be identical within a row):\n");
    bool row_only = true;
    for (int r = 0; r < PROBE_M; ++r) {
        int xor0 = (int)h_out[r * PROBE_K] ^ 0;
        for (int c = 8; c < PROBE_K; c += 8) {
            int logical = (int)h_out[r * PROBE_K + c];
            int xor_c   = c ^ logical;
            if (xor_c != xor0) {
                printf("  row %d: xor differs — col 0 xor=%d, col %d xor=%d "
                       "(column-dependent swizzle)\n", r, xor0, c, xor_c);
                row_only = false;
            }
        }
    }
    if (row_only)
        printf("  All rows: XOR is column-independent (row-only formula) ✓\n");

    return 0;
}
