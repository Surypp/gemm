cmake_minimum_required(VERSION 3.22)
<<<<<<< HEAD
project(GemmResearch LANGUAGES CXX CUDA)

# ─── Options ───────────────────────────────────────────────────────────────────
option(GEMM_ENABLE_HOPPER  "Build Phase 6 Hopper kernels (requires sm_90)"  OFF)
option(GEMM_ENABLE_TESTING "Build the GoogleTest correctness suite"          ON)
option(GEMM_ENABLE_BENCH   "Build the benchmark binary"                      ON)

# Target architectures: override with -DGEMM_ARCH_LIST="80" on Ampere-only machines
set(GEMM_ARCH_LIST "80;90" CACHE STRING "CUDA compute capability list (e.g. 80;90)")

# ─── Language standards ────────────────────────────────────────────────────────
=======
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
>>>>>>> 1572328 (refactor: restructured phase directories)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

<<<<<<< HEAD
# ─── Dependencies ─────────────────────────────────────────────────────────────
=======
# ── Dependencies ──────────────────────────────────────────────────────────────
>>>>>>> 1572328 (refactor: restructured phase directories)
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

<<<<<<< HEAD
# ─── Shared compile options ────────────────────────────────────────────────────
include(cmake/CompileOptions.cmake)

# ─── Subdirectories ───────────────────────────────────────────────────────────
add_subdirectory(kernels/phase0_naive)
add_subdirectory(kernels/phase1_shmem)
add_subdirectory(kernels/phase2_swizzle)
add_subdirectory(kernels/phase3_wmma)
add_subdirectory(kernels/phase4_pipeline)
add_subdirectory(kernels/phase5_ptx)
add_subdirectory(kernels/phase6_advanced)

if(GEMM_ENABLE_HOPPER)
    # Hopper kernels need sm_90a — verify it is in the arch list
    if(NOT "90" IN_LIST GEMM_ARCH_LIST AND NOT "90a" IN_LIST GEMM_ARCH_LIST)
        message(WARNING
            "GEMM_ENABLE_HOPPER is ON but GEMM_ARCH_LIST=${GEMM_ARCH_LIST} "
            "does not contain '90' or '90a'. Hopper kernels will not compile correctly.")
    endif()
    add_subdirectory(kernels/phase6_hopper)
endif()
=======
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
>>>>>>> 1572328 (refactor: restructured phase directories)

if(GEMM_ENABLE_BENCH)
    add_subdirectory(bench)
endif()

if(GEMM_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
<<<<<<< HEAD
=======

if(GEMM_ENABLE_TOOLS)
    add_subdirectory(tools)
endif()
>>>>>>> 1572328 (refactor: restructured phase directories)
