add_executable(gemm_hip_bench
    bench_main.cpp
    bench_suite.cpp
    bench_rocblas_fp16.hip
    bench_rocblas_fp32.hip
)

target_include_directories(gemm_hip_bench PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gemm_hip_bench PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:${GEMM_HIP_FLAGS}>
)

set_target_properties(gemm_hip_bench PROPERTIES
    HIP_ARCHITECTURES "${GEMM_HIP_ARCH}"
)

target_link_libraries(gemm_hip_bench PRIVATE
    gemm_hip_phase0
    gemm_hip_phase1
    gemm_hip_phase2
    gemm_hip_phase3
    gemm_hip_phase4
    hip::host
    roc::rocblas
)
