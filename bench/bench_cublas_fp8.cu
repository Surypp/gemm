// bench_cublas_fp8.cu — cublasLt FP8 (e4m3 × e5m2 → f32) baseline.
//
// Intentionally avoids all project utility headers (matrix.cuh, types.cuh, …)
// because those transitively include <cuda_fp16.h> / <cuda_bf16.h>, which in
// CUDA 13.2 contain __stg*/__ld* device intrinsics that use the "r" (32-bit)
// PTX constraint for 64-bit pointer operands — an assembler error on SM120.
// This file has no device kernels; CUDA events replace GpuTimer/TimingStats.

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define _CUDA_CHECK(x, file, line) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), file, line); \
        exit(1); \
    } \
} while(0)
#define CUDA_CHECK_L(x) _CUDA_CHECK(x, __FILE__, __LINE__)

#define CUBLASLT_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cublasLt error %d at %s:%d\n", \
                (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace bench_cublas {

// FP8 (e4m3 × e5m2 → f32) via cublasLtMatmul.
// Row-major convention via the col-major swap trick: compute C(N×M)=dB(N×K)*dA(K×M).
// Scale factors = 1.0f, workspace = 32 MB.
double measure_cublas_fp8_tflops(int M, int N, int K, int iters) {

    // --- device buffers (raw — no DeviceMatrix to avoid fp16 headers) ---
    uint8_t *dA = nullptr, *dB = nullptr;
    float   *dC = nullptr;
    CUDA_CHECK_L(cudaMalloc(&dA, (size_t)M * K));
    CUDA_CHECK_L(cudaMalloc(&dB, (size_t)K * N));
    CUDA_CHECK_L(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CUDA_CHECK_L(cudaMemset(dA, 0x38, (size_t)M * K)); // 1.0 in e4m3
    CUDA_CHECK_L(cudaMemset(dB, 0x3C, (size_t)K * N)); // 1.0 in e5m2
    CUDA_CHECK_L(cudaMemset(dC, 0,    (size_t)M * N * sizeof(float)));

    // --- per-tensor scale factors (A and B only; D is float32 so no D scale) ---
    const float h_one = 1.0f;
    float *d_scale_a = nullptr, *d_scale_b = nullptr;
    CUDA_CHECK_L(cudaMalloc(&d_scale_a, sizeof(float)));
    CUDA_CHECK_L(cudaMalloc(&d_scale_b, sizeof(float)));
    CUDA_CHECK_L(cudaMemcpy(d_scale_a, &h_one, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_L(cudaMemcpy(d_scale_b, &h_one, sizeof(float), cudaMemcpyHostToDevice));

    // --- workspace ---
    void*  workspace = nullptr;
    size_t ws_bytes  = 32ull * 1024 * 1024;
    CUDA_CHECK_L(cudaMalloc(&workspace, ws_bytes));

    // --- cublasLt setup ---
    cublasLtHandle_t lt;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // col-major swap: "A" in cublasLt = dB (N×K, e5m2), "B" = dA (K×M, e4m3)
    // D_SCALE_POINTER omitted: only valid when D type is FP8, not float32.
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a)));

    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_8F_E5M2, N, K, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_8F_E4M3, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32F,     N, M, N));

    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int n_algos = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op_desc,
        layout_a, layout_b, layout_c, layout_c,
        pref, 1, &heuristic, &n_algos));

    if (n_algos == 0) {
        fprintf(stderr,
            "cublasLt FP8: no algorithm for M=%d N=%d K=%d — skipping\n",
            M, N, K);
        cublasLtDestroy(lt);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cudaFree(d_scale_a); cudaFree(d_scale_b);
        cudaFree(workspace);
        return 0.0;
    }

    float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        CUBLASLT_CHECK(cublasLtMatmul(lt, op_desc,
            &alpha,
            dB, layout_a,
            dA, layout_b,
            &beta,
            dC, layout_c,
            dC, layout_c,
            &heuristic.algo, workspace, ws_bytes, 0));
    };

    // --- warmup ---
    for (int i = 0; i < 5; ++i) run();
    CUDA_CHECK_L(cudaDeviceSynchronize());

    // --- timed iterations via CUDA events (no GpuTimer/TimingStats headers) ---
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK_L(cudaEventCreate(&ev_start));
    CUDA_CHECK_L(cudaEventCreate(&ev_stop));

    double min_ms = 1e30;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK_L(cudaEventRecord(ev_start));
        run();
        CUDA_CHECK_L(cudaEventRecord(ev_stop));
        CUDA_CHECK_L(cudaEventSynchronize(ev_stop));
        float ms = 0.f;
        CUDA_CHECK_L(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        min_ms = std::min(min_ms, (double)ms);
    }

    CUDA_CHECK_L(cudaEventDestroy(ev_start));
    CUDA_CHECK_L(cudaEventDestroy(ev_stop));

    // --- cleanup ---
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_a));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_b));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_c));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(d_scale_a); cudaFree(d_scale_b);
    cudaFree(workspace);

    // 2*M*N*K FLOPs, same convention as compute_tflops() in benchmark.cuh
    return 2.0 * (double)M * (double)N * (double)K / (min_ms * 1e-3) / 1e12;
}

} // namespace bench_cublas
