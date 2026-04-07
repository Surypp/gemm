add_executable(gemm_tests
    test_main.cpp
    test_phase0.cpp
    test_phase1.cpp
    test_phase2.cpp
    test_phase3.cpp
    test_phase4.cpp
    test_phase5.cpp
    test_phase5_micro.cu
)
# test files include CUDA device headers; compile them with nvcc.
set_source_files_properties(
    test_main.cpp test_phase0.cpp test_phase1.cpp test_phase2.cpp
    test_phase3.cpp test_phase4.cpp test_phase5.cpp
    PROPERTIES LANGUAGE CUDA
)
target_include_directories(gemm_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/kernels
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(gemm_tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${GEMM_NVCC_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>
)
set_target_properties(gemm_tests PROPERTIES
    CUDA_ARCHITECTURES "${GEMM_ARCH_LIST}"
)
target_link_libraries(gemm_tests
    PRIVATE
        gemm_phase0
        gemm_phase1
        gemm_phase2
        gemm_phase3
        gemm_phase4
        gemm_phase5
        CUDA::cudart
        CUDA::cublas
        GTest::gtest
        GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(gemm_tests)
