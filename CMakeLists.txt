cmake_minimum_required(VERSION 3.22)
project(GemmCuda LANGUAGES CXX CUDA)

# ── Options ────────────────────────────────────────────────────────────────────
option(GEMM_ENABLE_TESTING "Build the GoogleTest correctness suite" ON)
option(GEMM_ENABLE_BENCH   "Build the benchmark binary"             ON)
option(GEMM_ENABLE_TOOLS   "Build diagnostic tools (sm_90a/sm_120 only)" ON)

# Target architectures: override with -DGEMM_ARCH_LIST="80" on Ampere-only machines.
# Phase 6c (TMA) always compiles for 90a;120 regardless of this setting.
set(GEMM_ARCH_LIST "80;90a;120" CACHE STRING
    "CUDA compute capability list (e.g. 80;90a;120)")

# ── Language standards ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

if(GEMM_ENABLE_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# ── Shared compile options ─────────────────────────────────────────────────────
include(cmake/CompileOptions.cmake)

# ── Kernel subdirectories ──────────────────────────────────────────────────────
add_subdirectory(kernels/phase0_naive)
add_subdirectory(kernels/phase1_tiling)
add_subdirectory(kernels/phase2_swizzle)
add_subdirectory(kernels/phase3_wmma)
add_subdirectory(kernels/phase4_cp_async)
add_subdirectory(kernels/phase5_ptx_mma)
add_subdirectory(kernels/phase6a_ldmatrix)
add_subdirectory(kernels/phase6b_multistage)
add_subdirectory(kernels/phase6c_tma)

if(GEMM_ENABLE_BENCH)
    add_subdirectory(bench)
endif()

if(GEMM_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GEMM_ENABLE_TOOLS)
    add_subdirectory(tools)
endif()
