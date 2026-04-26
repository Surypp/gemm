# HIP Port Notes — RX 9060 XT (gfx1200, RDNA 4)

## Environment

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 9060 XT |
| gfx arch | **gfx1200** (plan anticipated gfx1201) |
| ROCm | 7.2.1 at `/opt/rocm-7.2.1` |
| hipify-clang | `/home/frederic/Dev/dist/bin/hipify-clang` |
| Warp size | **64** (CUDA is 32) |

CMake always needs `-DCMAKE_PREFIX_PATH=/opt/rocm-7.2.1` to locate `hip-config.cmake`.

---

## API Mapping

| CUDA | HIP | Auto? |
|---|---|---|
| `cudaMalloc/Free/Memcpy` | `hipMalloc/hipFree/hipMemcpy` | hipify ✓ |
| `cudaStream_t` | `hipStream_t` | hipify ✓ |
| `cudaDeviceProp` | `hipDeviceProp_t` | hipify ✓ |
| `cuda_fp16.h` / `__half` | `hip/hip_fp16.h` / `__half` | hipify ✓ |
| `__nv_bfloat16` | `__hip_bfloat16` | hipify ✓ (AST) |
| `mma.h` + `nvcuda::wmma::` | `rocwmma/rocwmma.hpp` + `rocwmma::` | manual (hip_wmma.h absent on this install) |
| `CUDA_CHECK` macro name | `HIP_CHECK` | manual |
| cuBLAS | rocBLAS | manual rewrite |
| `cp.async` PTX | `float4` load + `__syncthreads` | manual rewrite |
| `__CUDACC__` guard | `__HIP_DEVICE_COMPILE__` | manual |

---

## Deviations from Plan

### hipify-clang output
hipify-clang translates internals but not user-facing macro names (`CUDA_CHECK` stays `CUDA_CHECK`). It also writes output as `<file>.hip` alongside the source — delete these from the CUDA tree after copying.

### cuda_helpers.cuh
hipify-clang fails on the `#if CUDART_VERSION >= 13000` guard around `clockRate`. Translated manually: `hipDeviceProp_t` always has `clockRate` and `memoryClockRate`, so the guard is simply dropped.

### Warp size
RDNA warp size is 64, not 32. `lane_id()` / `warp_id()` in `hip_helpers.hpp` use divisor 64.

### rocBLAS row-major convention
rocBLAS is column-major. To compute row-major `C = A·B`, swap A↔B and M↔N in the call — same trick as cuBLAS. For `rocblas_gemm_ex` the D output pointer must be provided (set it equal to C).

---

### rocwmma cmake dependencies

`roc::rocwmma` is a header-only INTERFACE target but its cmake config references `OpenMP::OpenMP_CXX` and `rocm_smi64`. Both must be found before `find_package(rocwmma)`:

```cmake
find_package(OpenMP REQUIRED)
find_package(rocm_smi REQUIRED)
find_package(rocwmma REQUIRED)
target_link_libraries(... PUBLIC roc::rocwmma OpenMP::OpenMP_CXX rocm_smi64)
```

### rocwmma namespace alias

`hip_wmma.h` is absent on ROCm 7.2.1 for gfx1200. Include `rocwmma/rocwmma.hpp` directly and add a namespace alias so kernel source is unchanged:

```cpp
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
```

All `wmma::fragment`, `wmma::fill_fragment`, `wmma::load_matrix_sync`, `wmma::mma_sync`, `wmma::store_matrix_sync`, `wmma::mem_row_major`, and `frag.x[i]` / `frag.num_elements` work unchanged through the alias. rocwmma provides `.x` and `num_elements` explicitly for nvcuda compatibility.

### rocwmma in host-compiled .cpp test files

rocwmma's `config.hpp` fires `static_assert(0, "Unsupported architecture")` when `__HIP_DEVICE_COMPILE__ == 1` but no gfx arch macro is defined. This happens when a `.cpp` test file that includes a rocwmma header is compiled as part of a HIP target (the device pass runs without a concrete arch macro).

Fix: add a `gemm_wmma_fwd.hpp` alongside the kernel header that forward-declares `launch_gemm_wmma` using only `hip/hip_runtime.h` and the types header. Test `.cpp` files include the forward-declaration header; the full rocwmma header is only ever seen by the `.hip` compilation unit.

### gfx12 uses wave32 for WMMA

RDNA 4 (gfx1200/gfx1201) runs WMMA at wave32, not wave64. The existing kernel divides thread index by 32 for `warp_id`, which is correct. rocwmma's config also asserts wave32 at compile time for gfx12.

---

## Phases Skipped

Phase 5 (PTX `mma.sync`), 6a (`ldmatrix`), 6b (multistage), 6c (TMA/mbarrier) have no RDNA equivalent and are not ported.
