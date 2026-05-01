# GEMM CUDA: FP16 Naive to TMA + FP8 using PTX

C = A x B in FP16/FP8 with FP32 accumulation.
9 progressive FP16 phases, from ~3.6 TFLOPS (naive baseline) to TMA + mbarrier
(Blackwell async memory), plus a standalone FP8 QMMA kernel (SM120).
Validated on sm_80 (A100), sm_90a (GH200), sm_120 (RTX 5080).

---

## Performance FP16

RTX 5080 (sm_120), FP16 input / FP32 accumulation, BM=BN=128 BK=32.
cuBLAS reference (`cublasGemmEx`, `CUBLAS_TENSOR_OP_MATH`): **115.2 TFLOPS** (sq4k) · **120.2 TFLOPS** (sq8k).

sq4k (4096×4096×4096):

| Phase | Kernel | TFLOPS | % cuBLAS FP16 |
|-------|--------|-------:|:-------------:|
| 0 | Naive (1 thread = 1 element) | 3.6 | 3.1% |
| 1 | Tiling -- shared memory | 4.5 | 3.9% |
| 2 | + XOR swizzling (zero bank conflicts) | 20.7 | 18.0% |
| 3 | + Tensor cores via wmma API | 51.9 | 45.1% |
| 4 | + cp.async double-buffer (NS=2) | **95.7** | **83.1%** |
| 5 | PTX mma.sync (manual fragment layout) | 78.5 | 68.2% |
| 6a | + ldmatrix.sync (warp-cooperative smem→reg) | 94.5 | 82.0% |
| 6b | + cp.async multi-stage (NS=2) | 91.1 | 79.2% |
| 6c | + TMA + mbarrier (async tile load) | **87.6** | **76.1%** |

sq8k (8192×8192×8192):

| Phase | Kernel | TFLOPS | % cuBLAS FP16 |
|-------|--------|-------:|:-------------:|
| 4 | cp.async double-buffer | **104.5** | **86.8%** |
| 6a | ldmatrix.sync | 102.5 | 85.2% |
| 6b | cp.async multi-stage | 99.9 | 83.0% |
| 6c | TMA + mbarrier | 99.3 | 82.5% |

Measured with `min` latency over 20 iterations (same methodology for cuBLAS and custom kernels).
`pct_cublas_peak` is computed against cuBLAS measured at the same size in the same run, not against theoretical hardware peak.
Full stall profiles and NCU counter breakdowns: `docs/` (not pushed yet).

---

## Performance FP8 (SM120)

RTX 5080 (sm_120), FP8 E4M3/E5M2 input / FP32 accumulation, BM=BN=128 BK=32 NS=2.
Uses `QMMA.16832` (SM120 native FP8 tensor core instruction).
cuBLAS FP8 reference (`cublasGemmEx`, `CUDA_R_8F_E4M3`/`CUDA_R_8F_E5M2`):

| Size | cuBLAS FP8 | Phase 6e kernel | % cuBLAS FP8 |
|------|:----------:|:---------------:|:------------:|
| 1024×1024×1024 | 173.9 TFLOPS | 53.0 TFLOPS | 30.5% |
| 2048×2048×2048 | 294.0 TFLOPS | 82.3 TFLOPS | 28.0% |
| 4096×4096×4096 | 418.2 TFLOPS | 113.7 TFLOPS | 27.2% |
| 8192×8192×8192 | 465.9 TFLOPS | 127.9 TFLOPS | 27.4% |

FP8 QMMA throughput is approximately 2× FP16 HMMA on SM120 (confirmed: sq8k 127.9 T FP8 vs 99.3 T FP16 same kernel structure). The gap to cuBLAS FP8 (~27%) reflects missing optimizations (BK=32 too small for FP8 K-tile; warp scheduling not tuned for QMMA latency). Phase 6e is a correctness + QMMA activation proof, not a tuned production kernel.

---

## Build

Requires: CUDA >= 12.3, CMake >= 3.22, GPU Ampere+ (sm_80+).
Phase 6c (TMA) requires sm_90a or sm_120.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGEMM_ARCH_LIST="80;90a;120"
cmake --build build -j$(nproc)
./build/bench/gemm_bench --phase 6c --m 4096 --n 4096 --k 4096
```

Windows (RTX 5080):
```bat
.\build_gemm.bat
.\build\tests\gemm_tests.exe
```

---

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Expected: 7/7 test suites PASS on sm_120. Tolerance (rtol/atol) is per-dtype,
documented in `tests/tolerance.hpp`. FP16 tensor cores are non-associative --
bit-exact agreement with a CPU reference is not expected or required.

---

## What this project explores

**Phase progression.** Each phase introduces one transformation and measures its
isolated effect. Phase 0 -> 1 is the dominant jump: moving the working set from
HBM (~700 cycles latency) to SRAM (~20 cycles). Everything after refines from that
base.

**PTX inline register layout.** Phase 5 uses `mma.sync.aligned.m16n8k16` with
manual fragment management. The per-thread distribution of matrix elements across
32 warp lanes is described in PTX ISA section 9.7.13 and is non-obvious. Two bugs
were caught that silent compilers do not report: B loaded in row-major instead of
column-major (wrong result, no crash) and accumulator registers declared write-only
instead of read-write (K-loop accumulation silently broken). Full layout and both
bugs documented: `docs/register_layout_mma_sync.md`.



---

## Architecture support

| Feature | sm_80 (A100) | sm_90a (GH200) | sm_120 (RTX 5080) |
|---------|:---:|:---:|:---:|
| cp.async / LDGSTS | yes | yes | yes |
| ldmatrix.sync | yes | yes | yes |
| TMA + mbarrier | no | yes | yes |
| FP8 QMMA (E4M3/E5M2) | no | partial | yes |
| wgmma | no | yes | in progress |
| FP4 mxf4nvf4 | no | no | in progress |

NOTE: Tested on sm_120 (RTX 5080). sm_80 / sm_90a columns reflect
architectural availability, not measured validation (yet).

Triple protection: CMake (`-DGEMM_ARCH_LIST`), preprocessor (`#if __CUDA_ARCH__ >= 900`),
runtime dispatch (`cudaDeviceProp::major/minor`). A binary built for sm_90a does
not silently run on sm_80.

---

## Build requirements by phase

| Phase | Minimum arch | Notes |
|-------|-------------|-------|
| 0-4 | sm_80 | Compiles on any Ampere+ |
| 5 | sm_80 | mma.sync requires real SM target, not compute_ virtual arch |
| 6a | sm_80 | ldmatrix.sync, sm_80+ |
| 6b | sm_80 | cp.async, sm_80+ |
| 6c | sm_90a | TMA hardware absent on sm_80; compiles for sm_90a and sm_120 |
| 6e | sm_120 | QMMA.16832 FP8 native to Blackwell; sm_80/sm_90a fall back to HMMA |
