add_executable(gemm_bench
    bench_main.cpp
    bench_suite.cpp
    bench_cublas_fp32.cu
    bench_cublas_fp16.cu
)
# bench_main.cpp and bench_suite.cpp include CUDA device headers directly;
# compile them with nvcc instead of MSVC.
set_source_files_properties(bench_main.cpp bench_suite.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(gemm_bench
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(gemm_bench
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_NVCC_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>
)
set_target_properties(gemm_bench PROPERTIES
    CUDA_ARCHITECTURES "${GEMM_ARCH_LIST}"
)
target_link_libraries(gemm_bench
    PRIVATE
        gemm_phase0
        gemm_phase1
        gemm_phase2
        gemm_phase3
        gemm_phase4
        gemm_phase5
        gemm_phase6a
        gemm_phase6b
        gemm_phase6c
        CUDA::cudart
        CUDA::cublas
)
if(GEMM_ENABLE_HOPPER)
    target_link_libraries(gemm_bench PRIVATE gemm_phase6)
    target_compile_definitions(gemm_bench PRIVATE GEMM_HOPPER_ENABLED)
endif()
