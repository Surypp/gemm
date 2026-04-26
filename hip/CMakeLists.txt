cmake_minimum_required(VERSION 3.21)
project(GemmHip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

include(cmake/HipCompileOptions.cmake)

add_subdirectory(kernels/phase0_naive)
add_subdirectory(kernels/phase1_tiling)
add_subdirectory(kernels/phase2_swizzle)
add_subdirectory(kernels/phase3_wmma)
add_subdirectory(kernels/phase4_pipeline)
add_subdirectory(bench)

enable_testing()
add_subdirectory(tests)
