# GEMM CUDA -- Naive to TMA, three architectures

C = A x B in FP16 with FP32 accumulation.
7 progressive phases, from ~2 TFLOPS (naive baseline) to TMA + mbarrier
(Blackwell async memory). Validated on sm_80 (A100), sm_90a (GH200),
sm_120 (RTX 5080).

---

## Performance

RTX 5080 (sm_120), sq4k (4096×4096×4096), FP16 input / FP32 accumulation, BM=BN=128 BK=32.
cuBLAS baseline: **115.1 TFLOPS** (`cublasGemmEx`, `CUBLAS_TENSOR_OP_MATH`, same problem).

| Phase | Kernel | TFLOPS | % cuBLAS |
|-------|--------|-------:|:--------:|
| 0 | Naive (1 thread = 1 element) | 3.6 | 3.1% |
| 1 | Tiling -- shared memory | 4.5 | 3.9% |
| 2 | + XOR swizzling (zero bank conflicts) | 20.7 | 18.0% |
| 3 | + Tensor cores via wmma API | 51.8 | 45.0% |
| 4 | + cp.async double-buffer (NS=2) | **95.9** | **83.4%** |
| 5 | PTX mma.sync (manual fragment layout) | 78.9 | 68.5% |
| 6a | + ldmatrix.sync (warp-cooperative smem→reg) | 94.6 | 82.2% |
| 6b | + cp.async multi-stage (NS=3) | 72.2 | 62.7% |
| 6c | + TMA + mbarrier (async tile load) | 84.9 | 73.8% |

Measured with `min` latency over 20 iterations (same methodology for cuBLAS and custom kernels).
Full stall profiles and NCU counter breakdowns: `docs/`.

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

**ptxas ICE C7907.** ptxas 13.2 crashes with an internal compiler error on
`mbarrier.expect_tx` targeting sm_90a or sm_120. The combined instruction
`mbarrier.arrive.expect_tx` uses a different codepath and compiles cleanly.
Minimal reproducer, analysis, and workaround: `docs/ptxas_ice_C7907.md`.

---

## SM120 (Blackwell) -- architecture notes

RTX 5080 (CC 12.0) is the primary development machine. Several NCU counter names
differ from Ampere / Hopper, and some standard counters are absent. Divergences
documented with verified replacements: `docs/sm120_divergences.md`.

Raw measurements (latencies, stall profiles, instruction counts, counter mappings):
`data/sm120_latency_database.json`. Values marked `UNKNOWN` have not been measured
and are not estimated.

---

## Architecture support

| Feature | sm_80 (A100) | sm_90a (GH200) | sm_120 (RTX 5080) |
|---------|:---:|:---:|:---:|
| cp.async / LDGSTS | yes | yes | yes |
| ldmatrix.sync | yes | yes | yes |
| TMA + mbarrier | no | yes | yes |
| wgmma | no | yes | in progress |
| FP4 mxf4nvf4 | no | no | in progress |

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
